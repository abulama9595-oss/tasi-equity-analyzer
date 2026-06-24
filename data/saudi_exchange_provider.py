"""Saudi Exchange provider — authoritative Saudi-market source.

Default backend is SAHMK (app.sahmk.sa), a Tadawul-licensed REST API with a free tier
(~100 req/day, ~15-min delayed quotes), authenticated with an ``X-API-Key`` header.

Only the quote endpoint is *confirmed* by the build spec:
    GET https://app.sahmk.sa/api/v1/quote/2222/
      -> { symbol, name (AR), name_en, price, change, change_percent, volume,
           is_delayed, ... }
Other endpoint paths (history/financials/indicators) are NOT assumed here — those
methods return empty so the composite layer falls back to yfinance. Wire them in once
SAHMK's live docs confirm the paths.

Secrets: the API key is read by *name* from the environment by the caller and passed in.
It is never hardcoded, never logged, and never written to config. With no key the
provider is fully inert (``available`` is False) and the app runs yfinance-only.

The backend is switchable (sahmk | licensed | scraper) without touching callers.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
import requests

from .cache import DiskCache
from .provider_base import DataProvider

log = logging.getLogger(__name__)

_OHLCV = ["open", "high", "low", "close", "volume"]


class SaudiExchangeProvider(DataProvider):
    name = "saudi_exchange"

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        backend: str = "sahmk",
        quotes_delayed: bool = True,
        cache: DiskCache | None = None,
        timeout: float = 12.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key  # kept private; never logged
        self.backend = backend
        self.quotes_delayed = quotes_delayed
        self.cache = cache
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    def available(self) -> bool:
        """Active only with a key on the sahmk backend (licensed/scraper not wired)."""
        return bool(self._api_key) and self.backend == "sahmk"

    # ------------------------------------------------------------------ #
    @staticmethod
    def _code(ticker: str) -> str:
        return str(ticker).upper().replace(".SR", "").strip()

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self._api_key or "", "Accept": "application/json"}

    def _get(self, path: str) -> dict[str, Any] | None:
        """GET with backoff on 429/5xx. Returns parsed JSON dict or None. Never raises."""
        if not self.available:
            return None
        url = f"{self.base_url}/{path.lstrip('/')}"
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                resp = requests.get(url, headers=self._headers(), timeout=self.timeout)
            except requests.RequestException as exc:
                log.warning("SAHMK request error (%s): %s", path, exc)
                return None
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    return None
            if resp.status_code in (429, 500, 502, 503, 504):
                log.info("SAHMK %s on %s; backing off %.1fs", resp.status_code, path, delay)
                time.sleep(delay)
                delay *= 2
                continue
            if resp.status_code in (401, 403):
                log.warning("SAHMK auth rejected (%s) — check SAHMK_API_KEY.", resp.status_code)
                return None
            # 404 / other — endpoint not available; let composite fall back
            return None
        return None

    # ------------------------------------------------------------------ #
    def _quote(self, ticker: str) -> dict[str, Any] | None:
        """Confirmed endpoint: /quote/{code}/. Cached with the short quote TTL."""
        if not self.available:
            return None
        code = self._code(ticker)

        def _fetch():
            return self._get(f"quote/{code}/")

        if self.cache:
            return self.cache.get_or_compute("intraday_quote", f"sahmk:{code}", _fetch)
        return _fetch()

    def _company(self, ticker: str) -> dict[str, Any]:
        """/company/{code}/ - profile + nested 'fundamentals' (market cap, P/E, P/B, ...)."""
        if not self.available:
            return {}
        code = self._code(ticker)

        def _fetch():
            return self._get(f"company/{code}/")

        if self.cache:
            return self.cache.get_or_compute("company_info", f"sahmk-co:{code}", _fetch) or {}
        return _fetch() or {}

    def get_company_info(self, ticker: str) -> dict[str, Any]:
        c = self._company(ticker)
        if not c:
            q = self._quote(ticker)  # fallback to the lighter quote endpoint
            if not q:
                return {}
            return {
                "name_en": q.get("name_en"), "name_ar": q.get("name"),
                "price": q.get("price"), "previous_close": q.get("previous_close"),
                "currency": "SAR", "delayed": bool(q.get("is_delayed", self.quotes_delayed)),
                "avg_volume": q.get("volume"),
            }
        f = c.get("fundamentals") or {}
        return {
            "name_en": c.get("name_en"),
            "name_ar": c.get("name"),
            "sector": c.get("sector") or c.get("sector_name"),
            "industry": c.get("industry"),
            "description": c.get("description"),
            "currency": c.get("currency") or "SAR",
            "price": c.get("current_price"),
            "delayed": bool(c.get("is_delayed", self.quotes_delayed)),
            "market_cap": f.get("market_cap"),
            "shares_outstanding": f.get("shares_outstanding"),
            "free_float": f.get("float_shares"),
            "fifty_two_week_high": f.get("fifty_two_week_high"),
            "fifty_two_week_low": f.get("fifty_two_week_low"),
        }

    # ------- historical prices (SAHMK Starter+; free tier returns 403) ---- #
    def get_price_history(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Daily OHLCV from SAHMK's /historical endpoint. Requires a paid plan; on the free
        tier this 403s and returns empty so the composite falls back to yfinance. The exact
        response schema is parsed flexibly (see _parse_history) and verified against the live
        endpoint once a paid key is available. Empty results are not cached."""
        empty = pd.DataFrame(columns=_OHLCV)
        if not self.available:
            return empty
        code = self._code(ticker)
        ckey = f"sahmk:{code}:{period}:{interval}"
        if self.cache:
            hit = self.cache.get("price_history", ckey)
            if isinstance(hit, pd.DataFrame) and not hit.empty:
                return hit
        df = self._fetch_history(code, period, interval)
        if self.cache and isinstance(df, pd.DataFrame) and not df.empty:
            self.cache.set("price_history", ckey, df)
        return df

    def _fetch_history(self, code: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch daily history via the /historical from/to endpoint.

        SAHMK caps each response at ~1000 rows returned ascending, which truncates the
        *newest* data on long ranges. To get contiguous history up to today we request in
        backward-walking chunks (each safely under the cap, the newest ending today) and
        concatenate. Walks back using the actual earliest date returned, so it's robust to
        the exact cap value.
        """
        import datetime as dt

        years = 10
        if period and period.endswith("y") and period[:-1].isdigit():
            years = int(period[:-1])
        today = dt.date.today()
        earliest = today - dt.timedelta(days=years * 366)
        chunk = dt.timedelta(days=3 * 366)  # ~3y per request, well under the ~1000-row cap

        def fetch(start: dt.date, end: dt.date) -> pd.DataFrame:
            path = f"historical/{code}/?from={start.isoformat()}&to={end.isoformat()}&interval={interval}"
            return self._parse_history(self._get(path))

        frames: list[pd.DataFrame] = []
        # 1) Recent window: a `from` within the last few months returns data up to today
        #    (older `from` values cap at the EOD store's lag). This guarantees current data.
        recent = fetch(today - dt.timedelta(days=150), today)
        if not recent.empty:
            frames.append(recent)
            end = recent.index.min().date() - dt.timedelta(days=1)
        else:
            end = today

        # 2) Walk backward in sub-cap chunks for the rest of the history.
        for _ in range(years // 3 + 2):  # safety bound on number of requests
            if end <= earliest:
                break
            df = fetch(max(earliest, end - chunk), end)
            if df.empty:
                break
            frames.append(df)
            new_end = df.index.min().date() - dt.timedelta(days=1)
            if new_end >= end:  # no backward progress -> stop
                break
            end = new_end

        if not frames:
            return self._parse_history(self._get(f"historical/{code}/?interval={interval}"))
        out = pd.concat(frames)
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out

    @staticmethod
    def _parse_history(data: Any) -> pd.DataFrame:
        """Flexibly parse a historical-prices JSON payload into an OHLCV DataFrame."""
        if not data:
            return pd.DataFrame(columns=_OHLCV)
        records = data
        if isinstance(data, dict):
            records = None
            for k in ("data", "results", "history", "candles", "prices", "items", "quotes"):
                if isinstance(data.get(k), list):
                    records = data[k]
                    break
        if not isinstance(records, list) or not records:
            return pd.DataFrame(columns=_OHLCV)

        date_keys = ("date", "datetime", "time", "t", "timestamp", "trade_date", "tradeDate")
        field_keys = {
            "open": ("open", "o", "open_price", "openPrice"),
            "high": ("high", "h", "high_price", "highPrice"),
            "low": ("low", "l", "low_price", "lowPrice"),
            "close": ("close", "c", "close_price", "closePrice", "last", "last_price"),
            "volume": ("volume", "v", "vol", "traded_volume", "tradedVolume"),
        }

        def pick(rec: dict, keys: tuple) -> Any:
            for k in keys:
                if rec.get(k) is not None:
                    return rec[k]
            return None

        idx, rows = [], []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            d = pick(rec, date_keys)
            if d is None:
                continue
            idx.append(d)
            rows.append({c: pick(rec, field_keys[c]) for c in _OHLCV})
        if not rows:
            return pd.DataFrame(columns=_OHLCV)

        df = pd.DataFrame(rows, columns=_OHLCV)
        index = pd.to_datetime(pd.Series(idx), errors="coerce")
        if index.isna().all():  # epoch seconds/millis fallback
            num = pd.to_numeric(pd.Series(idx), errors="coerce")
            unit = "ms" if num.dropna().abs().gt(10**11).any() else "s"
            index = pd.to_datetime(num, unit=unit, errors="coerce")
        df.index = index
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        for c in _OHLCV:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close"]).sort_index()
        df.index.name = "date"
        return df

    def get_index_history(self, interval: str = "1d") -> pd.DataFrame:
        """TASI index history (symbol 'TASI' on SAHMK), for beta / relative strength on
        cloud hosts where Yahoo's ^TASI.SR is blocked."""
        if not self.available:
            return pd.DataFrame(columns=_OHLCV)
        ckey = f"sahmk:TASI:idx:{interval}"
        if self.cache:
            hit = self.cache.get("price_history", ckey)
            if isinstance(hit, pd.DataFrame) and not hit.empty:
                return hit
        df = self._fetch_history("TASI", "10y", interval)
        if self.cache and isinstance(df, pd.DataFrame) and not df.empty:
            self.cache.set("price_history", ckey, df)
        return df

    def _financials_raw(self, ticker: str) -> dict[str, Any]:
        if not self.available:
            return {}
        code = self._code(ticker)

        def _fetch():
            return self._get(f"financials/{code}/")

        if self.cache:
            return self.cache.get_or_compute("fundamentals", f"sahmk-fin:{code}", _fetch) or {}
        return _fetch() or {}

    def _dividends_raw(self, ticker: str) -> dict[str, Any]:
        if not self.available:
            return {}
        code = self._code(ticker)

        def _fetch():
            return self._get(f"dividends/{code}/")

        if self.cache:
            return self.cache.get_or_compute("fundamentals", f"sahmk-div:{code}", _fetch) or {}
        return _fetch() or {}

    def get_financials(self, ticker: str) -> dict[str, pd.DataFrame]:
        """Income / balance / cash-flow as yfinance-style DataFrames (index=line item,
        columns=report dates, newest first) so the analysis core consumes them unchanged."""
        raw = self._financials_raw(ticker)
        if not raw:
            return {}

        def _df(records: list, fmap: dict[str, str]) -> pd.DataFrame:
            if not records:
                return pd.DataFrame()
            cols = [pd.to_datetime(r.get("report_date"), errors="coerce") for r in records]
            data = {label: [r.get(src) for r in records] for label, src in fmap.items()}
            return pd.DataFrame(data, index=cols).T  # -> index=labels, columns=dates

        return {
            "income": _df(raw.get("income_statements") or [], {
                "Total Revenue": "total_revenue", "Gross Profit": "gross_profit",
                "Operating Income": "operating_income", "Net Income": "net_income"}),
            "balance": _df(raw.get("balance_sheets") or [], {
                "Total Assets": "total_assets", "Total Liabilities": "total_liabilities",
                "Stockholders Equity": "stockholders_equity", "Total Debt": "total_debt"}),
            "cashflow": _df(raw.get("cash_flows") or [], {
                "Operating Cash Flow": "operating_cash_flow", "Free Cash Flow": "free_cash_flow"}),
        }

    def get_key_stats(self, ticker: str) -> dict[str, Any]:
        """Valuation/profitability/growth ratios from the company 'fundamentals' block plus
        statements (margins/ROE/ROA/growth derived) and dividends. Returns fractions where
        the config expects them (margins, yields, growth)."""
        try:
            c = self._company(ticker)
            f = (c.get("fundamentals") or {}) if c else {}
            fin = self._financials_raw(ticker)
            inc = fin.get("income_statements") or []
            bal = fin.get("balance_sheets") or []
            cfs = fin.get("cash_flows") or []
            mc = f.get("market_cap")
            s: dict[str, Any] = {
                "pe": f.get("pe_ratio"), "forward_pe": f.get("forward_pe"),
                "pb": f.get("price_to_book"), "book_value": f.get("book_value"),
                "market_cap": mc, "shares_outstanding": f.get("shares_outstanding"),
                "trailing_eps": f.get("eps_ttm") or f.get("eps"),
            }
            if inc:
                i0 = inc[0]
                rev, ni = i0.get("total_revenue"), i0.get("net_income")
                gp, oi = i0.get("gross_profit"), i0.get("operating_income")
                if rev:
                    if mc:
                        s["ps"] = mc / rev
                    if ni is not None:
                        s["net_margin"] = ni / rev
                    if gp is not None:
                        s["gross_margin"] = gp / rev
                    if oi is not None:
                        s["operating_margin"] = oi / rev
                if len(inc) > 1:
                    r1, n1 = inc[1].get("total_revenue"), inc[1].get("net_income")
                    if rev and r1:
                        s["revenue_growth"] = (rev - r1) / abs(r1)
                    if ni is not None and n1:
                        s["eps_growth"] = (ni - n1) / abs(n1)
            if bal:
                b0 = bal[0]
                eq, ta, td = b0.get("stockholders_equity"), b0.get("total_assets"), b0.get("total_debt")
                ni0 = inc[0].get("net_income") if inc else None
                s["total_debt"] = td
                if eq:
                    if td is not None:
                        s["debt_equity"] = td / eq
                    if ni0 is not None:
                        s["roe"] = ni0 / eq
                if ta and ni0 is not None:
                    s["roa"] = ni0 / ta
            if cfs:
                c0 = cfs[0]
                fcf, ocf = c0.get("free_cash_flow"), c0.get("operating_cash_flow")
                s["free_cashflow"], s["operating_cashflow"] = fcf, ocf
                if fcf is not None and mc:
                    s["fcf_yield"] = fcf / mc
            d = self._dividends_raw(ticker)
            if d:
                dy = d.get("trailing_12m_yield")
                if dy is not None:
                    s["dividend_yield"] = dy / 100.0  # 5.09 -> 0.0509
                ttm, eps = d.get("trailing_12m_dividends"), s.get("trailing_eps")
                if ttm is not None and eps:
                    s["payout_ratio"] = ttm / eps
            return {k: v for k, v in s.items() if v is not None}
        except Exception as exc:
            log.warning("SAHMK key_stats failed for %s: %s", ticker, exc)
            return {}

    def get_dividends(self, ticker: str) -> pd.DataFrame:
        d = self._dividends_raw(ticker)
        if not d:
            return pd.DataFrame()
        recs = []
        for r in d.get("history") or []:
            when = r.get("distribution_date") or r.get("eligibility_date") or r.get("announcement_date")
            val = r.get("value")
            if when and val is not None:
                recs.append((when, val))
        if not recs:
            return pd.DataFrame()
        idx = pd.to_datetime([x[0] for x in recs], errors="coerce")
        df = pd.DataFrame({"dividend": [x[1] for x in recs]}, index=idx).dropna()
        df.index.name = "date"
        return df.sort_index()

    def get_disclosures(self, ticker: str) -> list[dict[str, Any]]:
        return []

    def supports(self, field_name: str) -> bool:
        # Until more endpoints are confirmed, SAHMK authoritatively supplies company_info.
        if not self.available:
            return False
        return field_name in {"company_info", "sector", "price"}
