"""yfinance provider — free baseline (long price history + a fundamentals baseline).

yfinance uses Yahoo's unofficial endpoints and can return empty/None or break. Every
method is isolated here, wrapped so it never raises, and cached to disk. The composite
layer falls back to other sources when a field is missing.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

from .cache import DiskCache
from .provider_base import DataProvider

log = logging.getLogger(__name__)

try:
    import yfinance as yf
except Exception as exc:  # pragma: no cover - import guard
    yf = None
    log.error("yfinance import failed: %s", exc)


_OHLCV = ["open", "high", "low", "close", "volume"]


class YFinanceProvider(DataProvider):
    name = "yfinance"

    def __init__(self, cache: DiskCache | None = None, index_symbol: str = "^TASI.SR"):
        self.cache = cache
        self.index_symbol = index_symbol

    # ------------------------------------------------------------------ #
    def _ticker(self, symbol: str):
        return yf.Ticker(symbol)

    def get_price_history(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        if yf is None:
            return pd.DataFrame(columns=_OHLCV)
        key = f"{ticker}:{period}:{interval}"

        # Serve only a *non-empty* cached result. yfinance can return empty when Yahoo
        # rate-limits (common on shared/cloud IPs); caching that empty would persist the
        # failure for the whole TTL, so we never cache empties and retry with backoff.
        if self.cache:
            hit = self.cache.get("price_history", key)
            if isinstance(hit, pd.DataFrame) and not hit.empty:
                return hit

        out = self._fetch_history(ticker, period, interval)
        if self.cache and isinstance(out, pd.DataFrame) and not out.empty:
            self.cache.set("price_history", key, out)
        return out

    def _fetch_history(self, ticker: str, period: str, interval: str, attempts: int = 3) -> pd.DataFrame:
        delay = 1.0
        for i in range(attempts):
            try:
                df = self._ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
                out = self._normalise_ohlcv(df)
                if not out.empty:
                    return out
                log.info("yfinance returned empty for %s (attempt %d/%d)", ticker, i + 1, attempts)
            except Exception as exc:
                log.warning("yfinance price history error for %s (attempt %d/%d): %s", ticker, i + 1, attempts, exc)
            if i < attempts - 1:
                time.sleep(delay)
                delay *= 2
        return pd.DataFrame(columns=_OHLCV)

    @staticmethod
    def _normalise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=_OHLCV)
        df = df.rename(columns={c: c.lower().replace(" ", "_") for c in df.columns})
        keep = [c for c in _OHLCV if c in df.columns]
        out = df[keep].copy()
        # strip tz so resampling rules behave predictably
        if isinstance(out.index, pd.DatetimeIndex) and out.index.tz is not None:
            out.index = out.index.tz_localize(None)
        out = out.dropna(how="all")
        out.index.name = "date"
        return out

    def get_index_history(self, interval: str) -> pd.DataFrame:
        return self.get_price_history(self.index_symbol, period="10y", interval=interval)

    # ------------------------------------------------------------------ #
    def get_company_info(self, ticker: str) -> dict[str, Any]:
        if yf is None:
            return {}

        def _fetch() -> dict[str, Any]:
            try:
                info = self._ticker(ticker).info or {}
            except Exception as exc:
                log.warning("yfinance info failed for %s: %s", ticker, exc)
                return {}
            if not info:
                return {}
            return {
                "name_en": info.get("longName") or info.get("shortName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "shares_outstanding": info.get("sharesOutstanding"),
                "free_float": info.get("floatShares"),
                "description": info.get("longBusinessSummary"),
                "currency": info.get("currency"),
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "previous_close": info.get("regularMarketPreviousClose") or info.get("previousClose"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "avg_volume": info.get("averageVolume") or info.get("averageDailyVolume10Day"),
                "exchange": info.get("exchange"),
            }

        if self.cache:
            return self.cache.get_or_compute("company_info", ticker, _fetch) or {}
        return _fetch()

    def get_key_stats(self, ticker: str) -> dict[str, Any]:
        if yf is None:
            return {}

        def _fetch() -> dict[str, Any]:
            try:
                info = self._ticker(ticker).info or {}
            except Exception as exc:
                log.warning("yfinance key_stats failed for %s: %s", ticker, exc)
                return {}
            if not info:
                return {}
            return {
                "pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "pb": info.get("priceToBook"),
                "ps": info.get("priceToSalesTrailing12Months"),
                "ev_ebitda": info.get("enterpriseToEbitda"),
                "peg": info.get("pegRatio") or info.get("trailingPegRatio"),
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
                "gross_margin": info.get("grossMargins"),
                "operating_margin": info.get("operatingMargins"),
                "net_margin": info.get("profitMargins"),
                "revenue_growth": info.get("revenueGrowth"),
                "eps_growth": info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth"),
                "debt_equity": _pct_to_ratio(info.get("debtToEquity")),
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
                "dividend_yield": info.get("dividendYield"),
                "payout_ratio": info.get("payoutRatio"),
                "beta": info.get("beta"),
                "book_value": info.get("bookValue"),
                "trailing_eps": info.get("trailingEps"),
                "ebitda": info.get("ebitda"),
                "total_debt": info.get("totalDebt"),
                "total_cash": info.get("totalCash"),
                "free_cashflow": info.get("freeCashflow"),
                "operating_cashflow": info.get("operatingCashflow"),
                "enterprise_value": info.get("enterpriseValue"),
            }

        if self.cache:
            return self.cache.get_or_compute("fundamentals", f"{ticker}:keystats", _fetch) or {}
        return _fetch()

    # ------------------------------------------------------------------ #
    def get_financials(self, ticker: str) -> dict[str, pd.DataFrame]:
        if yf is None:
            return {}

        def _fetch() -> dict[str, pd.DataFrame]:
            t = self._ticker(ticker)
            out: dict[str, pd.DataFrame] = {}
            for key, attr in [
                ("income", "income_stmt"),
                ("balance", "balance_sheet"),
                ("cashflow", "cashflow"),
                ("income_q", "quarterly_income_stmt"),
                ("balance_q", "quarterly_balance_sheet"),
                ("cashflow_q", "quarterly_cashflow"),
            ]:
                try:
                    df = getattr(t, attr)
                    out[key] = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
                except Exception as exc:
                    log.warning("yfinance %s failed for %s: %s", attr, ticker, exc)
                    out[key] = pd.DataFrame()
            return out

        if self.cache:
            return self.cache.get_or_compute("fundamentals", f"{ticker}:financials", _fetch) or {}
        return _fetch()

    def get_dividends(self, ticker: str) -> pd.DataFrame:
        if yf is None:
            return pd.DataFrame()

        def _fetch() -> pd.DataFrame:
            try:
                s = self._ticker(ticker).dividends
            except Exception as exc:
                log.warning("yfinance dividends failed for %s: %s", ticker, exc)
                return pd.DataFrame()
            if s is None or len(s) == 0:
                return pd.DataFrame()
            df = s.to_frame("dividend")
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index.name = "date"
            return df

        if self.cache:
            return self.cache.get_or_compute("fundamentals", f"{ticker}:dividends", _fetch)
        return _fetch()


def _pct_to_ratio(v: Any) -> float | None:
    """yfinance reports debtToEquity as a percentage (e.g. 45.2 -> 0.452)."""
    if v is None:
        return None
    try:
        return float(v) / 100.0
    except (TypeError, ValueError):
        return None
