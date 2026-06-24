"""Indicative Shariah-compliance screen.

Applies standard quantitative screens (interest-bearing debt, cash + interest-bearing
securities, receivables — each over a configurable denominator; plus non-compliant income
where available). Thresholds vary by board (AAOIFI, S&P/MSCI, local) and are configurable.
The result is INDICATIVE only — not a fatwa. Where inputs are unavailable it says so
rather than guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ShariahCheck:
    name: str
    value: float | None
    threshold: float
    passed: bool | None  # None when the input is unavailable


@dataclass
class ShariahResult:
    compliant: bool | None
    methodology: str
    denominator: str
    checks: list[ShariahCheck] = field(default_factory=list)
    note: str = "Indicative screen — not a fatwa. Thresholds are configurable and vary by board."
    sector_flag: str | None = None  # registry hint, if any


def _bs_row(bs: pd.DataFrame, names: list[str]) -> float | None:
    if not isinstance(bs, pd.DataFrame) or bs.empty:
        return None
    for n in names:
        if n in bs.index:
            row = bs.loc[n].dropna()
            if len(row):
                return float(row.iloc[0])
    return None


def _val(stats, key):
    s = stats.get(key)
    if s is None:
        return None
    return getattr(s, "value", s)


def analyse(
    financials: dict[str, pd.DataFrame],
    overview: dict[str, Any],
    key_stats: dict[str, Any],
    cfg,
    sector_flag: str | None = None,
) -> ShariahResult:
    sh = cfg.shariah
    th = sh.thresholds
    bs = financials.get("balance") if financials else None

    market_cap = _val(overview, "market_cap")
    total_assets = _bs_row(bs, ["Total Assets"])
    denom = market_cap if sh.denominator == "market_cap" else total_assets

    debt = _bs_row(bs, ["Total Debt", "Total Debt And Capital Lease Obligation"]) or _val(key_stats, "total_debt")
    cash = _bs_row(bs, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]) or _val(
        key_stats, "total_cash"
    )
    receivables = _bs_row(bs, ["Receivables", "Accounts Receivable", "Gross Accounts Receivable"])

    def ratio(numer):
        if numer is None or not denom:
            return None
        return numer / denom

    checks = [
        ShariahCheck("Interest-bearing debt / " + sh.denominator, ratio(debt),
                     th["interest_debt_to_denom_max"], None),
        ShariahCheck("Cash + interest securities / " + sh.denominator, ratio(cash),
                     th["cash_and_interest_sec_to_denom_max"], None),
        ShariahCheck("Receivables / " + sh.denominator, ratio(receivables),
                     th["receivables_to_denom_max"], None),
        ShariahCheck("Non-compliant income %", None,
                     th["non_compliant_income_max"], None),  # input typically unavailable
    ]
    for c in checks:
        if c.value is not None:
            c.passed = c.value <= c.threshold

    evaluated = [c for c in checks if c.passed is not None]
    if not evaluated:
        compliant = None
    else:
        compliant = all(c.passed for c in evaluated)

    return ShariahResult(
        compliant=compliant,
        methodology=sh.methodology,
        denominator=sh.denominator,
        checks=checks,
        sector_flag=sector_flag,
    )
