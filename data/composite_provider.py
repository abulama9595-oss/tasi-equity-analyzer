"""CompositeProvider — merge multiple sources with per-field preference + provenance.

For every logical field, providers are tried in the order given by ``field_preference``
in config.yaml. The first source with a usable value wins; others are recorded as
fallbacks/conflicts. Every returned datapoint carries provenance so the UI can show
*which source* each value came from and flag disagreements.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import pandas as pd

from .provider_base import DataProvider, Provenance, Sourced

log = logging.getLogger(__name__)


class CompositeProvider:
    def __init__(
        self,
        providers: dict[str, DataProvider],
        field_preference: dict[str, list[str]],
    ):
        self.providers = providers
        self.pref = field_preference

    # ------------------------------------------------------------------ #
    def _order(self, field: str) -> list[str]:
        order = self.pref.get(field) or list(self.providers.keys())
        return [n for n in order if n in self.providers]

    # ---------- price & index history (DataFrame-valued) --------------- #
    def get_price_history(
        self, ticker: str, period: str, interval: str
    ) -> tuple[pd.DataFrame, Provenance]:
        for name in self._order("price_history"):
            df = _safe(lambda p=self.providers[name]: p.get_price_history(ticker, period, interval))
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df, Provenance(source=name)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]), Provenance(
            source="none", note="no price history from any source"
        )

    def get_index_history(self, interval: str) -> tuple[pd.DataFrame, Provenance]:
        for name in self._order("index_history"):
            df = _safe(lambda p=self.providers[name]: p.get_index_history(interval))
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df, Provenance(source=name)
        return pd.DataFrame(), Provenance(source="none")

    # ---------- overview (field-by-field merge) ------------------------ #
    def get_overview(self, ticker: str) -> dict[str, Sourced]:
        """Merge company_info sub-fields, each per its own preference key."""
        infos = {n: _safe(lambda p=self.providers[n]: p.get_company_info(ticker)) or {} for n in self.providers}

        def pick(pref_key: str, info_key: str) -> Sourced:
            return self._pick_field(pref_key, infos, info_key)

        out = {
            "name_en": pick("company_info", "name_en"),
            "name_ar": pick("company_info", "name_ar"),
            "sector": pick("sector", "sector"),
            "industry": pick("company_info", "industry"),
            "market_cap": pick("company_info", "market_cap"),
            "shares_outstanding": pick("shares_outstanding", "shares_outstanding"),
            "free_float": pick("free_float", "free_float"),
            "description": pick("company_info", "description"),
            "currency": pick("company_info", "currency"),
            "price": pick("company_info", "price"),
            "previous_close": pick("company_info", "previous_close"),
            "fifty_two_week_low": pick("company_info", "fifty_two_week_low"),
            "fifty_two_week_high": pick("company_info", "fifty_two_week_high"),
            "avg_volume": pick("company_info", "avg_volume"),
        }
        # propagate the "delayed" flag if the chosen price came from a delayed source
        price_src = out["price"].source
        if price_src and infos.get(price_src, {}).get("delayed"):
            out["price"].delayed = True
        return out

    def _pick_field(self, pref_key: str, infos: dict[str, dict], info_key: str) -> Sourced:
        """Return the value for ``info_key`` from the first preferred source that has it,
        recording other sources' differing values as conflicts."""
        order = self._order(pref_key)
        chosen: Sourced | None = None
        for name in order:
            val = infos.get(name, {}).get(info_key)
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                if chosen is None:
                    chosen = Sourced(value=val, source=name)
                elif _differs(chosen.value, val):
                    chosen.conflicts[name] = val
        return chosen if chosen is not None else Sourced(value=None, source=None)

    # ---------- key stats --------------------------------------------- #
    def get_key_stats(self, ticker: str) -> dict[str, Sourced]:
        stats = {n: _safe(lambda p=self.providers[n]: p.get_key_stats(ticker)) or {} for n in self.providers}
        order = self._order("key_stats")
        keys: set[str] = set()
        for s in stats.values():
            keys.update(s.keys())
        out: dict[str, Sourced] = {}
        for k in keys:
            chosen: Sourced | None = None
            for name in order:
                v = stats.get(name, {}).get(k)
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    if chosen is None:
                        chosen = Sourced(value=v, source=name)
                    elif _differs(chosen.value, v):
                        chosen.conflicts[name] = v
            out[k] = chosen if chosen is not None else Sourced(value=None, source=None)
        return out

    # ---------- financials / dividends / disclosures ------------------ #
    def get_financials(self, ticker: str) -> tuple[dict[str, pd.DataFrame], Provenance]:
        for name in self._order("financials"):
            fin = _safe(lambda p=self.providers[name]: p.get_financials(ticker)) or {}
            if any(isinstance(v, pd.DataFrame) and not v.empty for v in fin.values()):
                return fin, Provenance(source=name)
        return {}, Provenance(source="none")

    def get_dividends(self, ticker: str) -> tuple[pd.DataFrame, Provenance]:
        for name in self._order("dividends"):
            df = _safe(lambda p=self.providers[name]: p.get_dividends(ticker))
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df, Provenance(source=name)
        return pd.DataFrame(), Provenance(source="none")

    def get_disclosures(self, ticker: str) -> tuple[list[dict[str, Any]], Provenance]:
        for name in self._order("disclosures"):
            items = _safe(lambda p=self.providers[name]: p.get_disclosures(ticker)) or []
            if items:
                return items, Provenance(source=name)
        return [], Provenance(source="none")

    def provenance_summary(self, overview: dict[str, Sourced], stats: dict[str, Sourced]) -> dict[str, str]:
        """field -> source map for the UI's provenance panel."""
        summary: dict[str, str] = {}
        for k, v in {**overview, **stats}.items():
            if isinstance(v, Sourced) and v.source:
                summary[k] = v.source
        return summary


def _safe(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as exc:  # provider contract is no-raise, but double-guard here
        log.warning("provider call failed: %s", exc)
        return None


def _differs(a: Any, b: Any) -> bool:
    """Loose conflict test: numbers differ >2%, strings differ case-insensitively."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == 0 and b == 0:
            return False
        denom = max(abs(a), abs(b), 1e-9)
        return abs(a - b) / denom > 0.02
    return str(a).strip().lower() != str(b).strip().lower()
