"""TASI ticker normalisation, validation, and metadata lookup.

Accepts ``1120``, ``1120.SR``, lowercase, whitespace, or a company name (EN/AR) and
resolves to the canonical ``NNNN.SR`` symbol. Backed by the bundled
``tasi_tickers.csv`` (refreshable from a maintained source). Unknown but well-formed
4-digit codes are still allowed (analysis falls back to live provider metadata) — the
registry just can't enrich them with sector / type / Shariah hints.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_CODE_RE = re.compile(r"^\d{4}$")
_REGISTRY_PATH = Path(__file__).resolve().parent / "tasi_tickers.csv"


@dataclass
class TickerRef:
    symbol: str  # canonical NNNN.SR
    code: str  # NNNN
    name_en: str | None = None
    name_ar: str | None = None
    sector: str | None = None
    company_type: str = "general"  # bank | insurance | reit | general
    shariah_flag: str | None = None  # compliant | non_compliant | unknown
    in_registry: bool = True

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "code": self.code,
            "name_en": self.name_en,
            "name_ar": self.name_ar,
            "sector": self.sector,
            "company_type": self.company_type,
            "shariah_flag": self.shariah_flag,
            "in_registry": self.in_registry,
        }


class TickerRegistry:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else _REGISTRY_PATH
        self._df = self._load()

    def _load(self) -> pd.DataFrame:
        if not self.path.exists():
            log.warning("Ticker registry not found at %s; running without bundled refs.", self.path)
            return pd.DataFrame(
                columns=["ticker", "name_en", "name_ar", "sector", "company_type", "shariah_flag"]
            )
        df = pd.read_csv(self.path, dtype={"ticker": str})
        df["ticker"] = df["ticker"].astype(str).str.strip().str.zfill(4)
        return df

    # ------------------------------------------------------------------ #
    @staticmethod
    def normalise(raw: str) -> str | None:
        """Return the bare 4-digit code from many input forms, or None if not a code."""
        if raw is None:
            return None
        s = str(raw).strip().upper()
        s = s.replace(".SR", "").replace(".SAU", "").replace("SR:", "").strip()
        s = re.sub(r"\s+", "", s)
        if _CODE_RE.match(s):
            return s
        # left-pad short numerics (e.g. "120" is not valid TASI, but "1120" is)
        if s.isdigit() and 1 <= len(s) <= 4:
            return s.zfill(4) if len(s) == 4 else None
        return None

    def resolve(self, raw: str) -> TickerRef | None:
        """Resolve any input (code, NNNN.SR, or company name) to a TickerRef.

        Returns None only when the input is neither a valid code nor a recognised name.
        """
        if raw is None or not str(raw).strip():
            return None

        code = self.normalise(raw)
        if code is not None:
            return self._ref_for_code(code)

        # Not a code → try to match a company name (EN or AR, substring, case-insensitive)
        query = str(raw).strip().lower()
        if self._df.empty:
            return None
        mask = self._df["name_en"].fillna("").str.lower().str.contains(query, regex=False) | self._df[
            "name_ar"
        ].fillna("").str.contains(query, regex=False)
        hits = self._df[mask]
        if len(hits) == 0:
            return None
        row = hits.iloc[0]
        return self._row_to_ref(row, in_registry=True)

    def _ref_for_code(self, code: str) -> TickerRef:
        row = self._df[self._df["ticker"] == code]
        if len(row) == 0:
            # Valid-looking code not in bundled file — allow, but unenriched.
            return TickerRef(
                symbol=f"{code}.SR", code=code, company_type="general",
                shariah_flag="unknown", in_registry=False,
            )
        return self._row_to_ref(row.iloc[0], in_registry=True)

    @staticmethod
    def _row_to_ref(row: pd.Series, in_registry: bool) -> TickerRef:
        code = str(row["ticker"]).zfill(4)
        return TickerRef(
            symbol=f"{code}.SR",
            code=code,
            name_en=_clean(row.get("name_en")),
            name_ar=_clean(row.get("name_ar")),
            sector=_clean(row.get("sector")),
            company_type=_clean(row.get("company_type")) or "general",
            shariah_flag=_clean(row.get("shariah_flag")) or "unknown",
            in_registry=in_registry,
        )

    # ------------------------------------------------------------------ #
    def search(self, query: str, limit: int = 10) -> list[TickerRef]:
        """Autocomplete: match code prefix or name substring (EN/AR)."""
        if self._df.empty or not query or not query.strip():
            return []
        q = query.strip().lower()
        df = self._df
        mask = (
            df["ticker"].str.startswith(re.sub(r"\D", "", q))
            if q[0].isdigit()
            else (
                df["name_en"].fillna("").str.lower().str.contains(q, regex=False)
                | df["name_ar"].fillna("").str.contains(query.strip(), regex=False)
            )
        )
        return [self._row_to_ref(r, True) for _, r in df[mask].head(limit).iterrows()]

    def all_refs(self) -> list[TickerRef]:
        return [self._row_to_ref(r, True) for _, r in self._df.iterrows()]

    def detect_company_type(self, code: str, sector: str | None = None) -> str:
        """Bank / Insurance / REIT / general — registry first, then sector heuristic."""
        ref = self._ref_for_code(code)
        if ref.in_registry and ref.company_type:
            return ref.company_type
        s = (sector or "").lower()
        if "bank" in s:
            return "bank"
        if "insur" in s:
            return "insurance"
        if "reit" in s:
            return "reit"
        return "general"


def _clean(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip()
    return s if s and s.lower() != "nan" else None
