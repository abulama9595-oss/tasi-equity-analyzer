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

    def get_company_info(self, ticker: str) -> dict[str, Any]:
        q = self._quote(ticker)
        if not q:
            return {}
        # SAHMK keys per the spec example; tolerate naming variants.
        return {
            "name_en": q.get("name_en") or q.get("name_english"),
            "name_ar": q.get("name") or q.get("name_ar"),
            "price": q.get("price") or q.get("last_price"),
            "previous_close": q.get("previous_close") or q.get("prev_close"),
            "currency": "SAR",
            "delayed": bool(q.get("is_delayed", self.quotes_delayed)),
            "avg_volume": q.get("volume"),
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

    def get_financials(self, ticker: str) -> dict[str, pd.DataFrame]:
        return {}

    def get_key_stats(self, ticker: str) -> dict[str, Any]:
        return {}

    def get_dividends(self, ticker: str) -> pd.DataFrame:
        return pd.DataFrame()

    def get_disclosures(self, ticker: str) -> list[dict[str, Any]]:
        return []

    def supports(self, field_name: str) -> bool:
        # Until more endpoints are confirmed, SAHMK authoritatively supplies company_info.
        if not self.available:
            return False
        return field_name in {"company_info", "sector", "price"}
