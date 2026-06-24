"""Dividend analysis — history, yield, payout, growth, and FCF-cover sustainability."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DividendResult:
    dividend_yield: float | None
    payout_ratio: float | None
    growth_cagr: float | None
    fcf_cover: float | None
    sustainable: bool | None
    consistency: float | None  # share of recent years that paid
    history: list[dict[str, Any]] = field(default_factory=list)


def _val(stats, key):
    s = stats.get(key)
    if s is None:
        return None
    return getattr(s, "value", s)


def analyse(
    dividends: pd.DataFrame,
    key_stats: dict[str, Any],
    overview: dict[str, Any],
    cfg,
) -> DividendResult:
    dy = _val(key_stats, "dividend_yield")
    if dy is not None and dy > 1.0:  # yfinance percentage quirk
        dy = dy / 100.0
    payout = _val(key_stats, "payout_ratio")

    history: list[dict[str, Any]] = []
    growth_cagr = None
    consistency = None
    if isinstance(dividends, pd.DataFrame) and not dividends.empty:
        col = dividends.columns[0]
        annual = dividends[col].groupby(dividends.index.year).sum()
        history = [{"year": int(y), "dps": round(float(v), 4)} for y, v in annual.items()]
        recent = annual.tail(6)
        if len(recent) >= 2 and recent.iloc[0] > 0:
            years = len(recent) - 1
            growth_cagr = (recent.iloc[-1] / recent.iloc[0]) ** (1 / years) - 1
        if len(recent) > 0:
            consistency = float((recent > 0).mean())

    # FCF cover = free cash flow / total dividends paid (approx via dps_ttm * shares)
    fcf = _val(key_stats, "free_cashflow")
    shares = _val(overview, "shares_outstanding") or _val(key_stats, "shares_outstanding")
    fcf_cover = None
    if fcf and shares and history:
        last_dps = history[-1]["dps"]
        total_div = last_dps * shares
        if total_div:
            fcf_cover = fcf / total_div

    sustainable: bool | None = None
    if payout is not None or fcf_cover is not None:
        ok_payout = payout is None or payout <= 0.9
        ok_cover = fcf_cover is None or fcf_cover >= 1.0
        # need at least one positive signal to call it sustainable
        if payout is not None or fcf_cover is not None:
            sustainable = bool(ok_payout and ok_cover)

    return DividendResult(
        dividend_yield=_r(dy, 4),
        payout_ratio=_r(payout, 3),
        growth_cagr=_r(growth_cagr, 4),
        fcf_cover=_r(fcf_cover, 2),
        sustainable=sustainable,
        consistency=_r(consistency, 2),
        history=history,
    )


def _r(v, n):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(v, n)
