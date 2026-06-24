"""Provider abstraction for market & fundamental data.

Every concrete provider (yfinance, Saudi Exchange/SAHMK, future paid providers)
implements this interface. The CompositeProvider merges them with a per-field source
preference and records provenance. Methods return plain pandas/dict structures so the
analysis core stays provider-agnostic.

Robustness contract: a provider must NEVER raise on missing data. It returns an empty
DataFrame / empty dict / None and lets the composite layer fall back and record that the
field was unavailable from that source.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Provenance:
    """Records which source supplied a field (and any conflict)."""

    source: str
    as_of: str | None = None
    delayed: bool = False
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "as_of": self.as_of,
            "delayed": self.delayed,
            "note": self.note,
        }


@dataclass
class Sourced:
    """A value plus where it came from. Mirrors the API's ValueWithSource shape."""

    value: Any
    source: str | None = None
    as_of: str | None = None
    delayed: bool = False
    conflicts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {"value": self.value, "source": self.source, "as_of": self.as_of, "delayed": self.delayed}
        if self.conflicts:
            d["conflicts"] = self.conflicts
        return d


class DataProvider(ABC):
    """Abstract market-data provider. All methods degrade gracefully."""

    name: str = "base"

    @abstractmethod
    def get_price_history(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Daily OHLCV DataFrame indexed by date with columns
        ['open','high','low','close','volume']. Empty DataFrame if unavailable."""

    @abstractmethod
    def get_company_info(self, ticker: str) -> dict[str, Any]:
        """name (EN/AR), sector, industry, market cap, shares outstanding, free float,
        description, listing info, 52-week range, currency. Empty dict if unavailable."""

    @abstractmethod
    def get_financials(self, ticker: str) -> dict[str, pd.DataFrame]:
        """{'income','balance','cashflow'} annual (+ quarterly where available)."""

    @abstractmethod
    def get_key_stats(self, ticker: str) -> dict[str, Any]:
        """Valuation & ratio metrics (pe, pb, roe, margins, ...)."""

    @abstractmethod
    def get_dividends(self, ticker: str) -> pd.DataFrame:
        """Dividend history & corporate actions. Empty DataFrame if unavailable."""

    # Optional capabilities — default to "not supported" so a provider need not
    # implement everything.
    def get_disclosures(self, ticker: str) -> list[dict[str, Any]]:
        return []

    def get_index_history(self, interval: str) -> pd.DataFrame:
        return pd.DataFrame()

    def supports(self, field_name: str) -> bool:
        """Whether this provider can in principle supply a logical field."""
        return True
