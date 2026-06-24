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

    # ------- not-yet-confirmed endpoints: inert, composite falls back ---- #
    def get_price_history(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

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
