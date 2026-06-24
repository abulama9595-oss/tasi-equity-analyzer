"""Type-aware fundamental analysis & sub-score.

Generic ratios mislead for financials, so the metric set and component weights switch by
company type (bank / reit / insurance / general) per config.yaml. Each metric is mapped
to 0..100 via the config anchors, optionally blended with a sector percentile when a peer
set is supplied. Missing metrics are dropped and weights re-normalised; coverage feeds the
verdict's data-completeness indicator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import scoring

# Which metrics make up each component, per company type. Metrics absent from the data
# are simply dropped (graceful degradation).
COMPONENT_METRICS: dict[str, dict[str, list[str]]] = {
    "general": {
        "valuation": ["pe", "pb", "ps", "ev_ebitda", "peg"],
        "profitability": ["roe", "roa", "roic", "gross_margin", "operating_margin", "net_margin"],
        "growth": ["revenue_growth", "eps_growth"],
        "financial_health": ["debt_equity", "current_ratio", "interest_coverage", "net_debt_ebitda"],
        "cash_flow": ["fcf_yield"],
    },
    "bank": {
        # net_margin & revenue_growth excluded: a bank's "total revenue" is reported
        # inconsistently year-to-year (gross vs net financing income), so anything derived
        # from it is unreliable. ROE / EPS-growth (net-income based) and P/B are trustworthy.
        "valuation": ["pb", "pe"],
        "profitability": ["roe", "nim", "cost_to_income"],
        "growth": ["eps_growth"],
        "asset_quality": ["npl_ratio", "car"],
    },
    "reit": {
        "valuation": ["p_ffo", "pb"],
        "distribution": ["dividend_yield", "ffo_payout"],
        "growth": ["revenue_growth"],
        "financial_health": ["ltv", "debt_equity"],
    },
    "insurance": {
        "valuation": ["pb", "pe"],
        "profitability": ["combined_ratio", "roe"],
        "growth": ["eps_growth"],
        "financial_health": ["current_ratio"],
    },
}

METRIC_META: dict[str, tuple[str, str]] = {
    # key: (label, unit)
    "pe": ("P/E (trailing)", "x"),
    "forward_pe": ("P/E (forward)", "x"),
    "pb": ("P/B", "x"),
    "ps": ("P/S", "x"),
    "ev_ebitda": ("EV/EBITDA", "x"),
    "peg": ("PEG", "x"),
    "roe": ("ROE", "%"),
    "roa": ("ROA", "%"),
    "roic": ("ROIC", "%"),
    "gross_margin": ("Gross margin", "%"),
    "operating_margin": ("Operating margin", "%"),
    "net_margin": ("Net margin", "%"),
    "revenue_growth": ("Revenue growth (YoY)", "%"),
    "eps_growth": ("EPS growth (YoY)", "%"),
    "debt_equity": ("Debt/Equity", "x"),
    "current_ratio": ("Current ratio", "x"),
    "interest_coverage": ("Interest coverage", "x"),
    "net_debt_ebitda": ("Net debt/EBITDA", "x"),
    "fcf_yield": ("FCF yield", "%"),
    "dividend_yield": ("Dividend yield", "%"),
    "payout_ratio": ("Payout ratio", "%"),
    "nim": ("Net interest margin", "%"),
    "cost_to_income": ("Cost-to-income", "%"),
    "npl_ratio": ("NPL ratio", "%"),
    "car": ("Capital adequacy", "%"),
    "p_ffo": ("P/FFO", "x"),
    "ffo_payout": ("FFO payout", "%"),
    "ltv": ("Loan-to-value", "%"),
    "combined_ratio": ("Combined ratio", "%"),
}

_PERCENT_METRICS = {
    "roe", "roa", "roic", "gross_margin", "operating_margin", "net_margin",
    "revenue_growth", "eps_growth", "fcf_yield", "dividend_yield", "payout_ratio",
    "nim", "cost_to_income", "npl_ratio", "car", "ffo_payout", "ltv", "combined_ratio",
}


@dataclass
class FundamentalResult:
    rubric: str
    metrics: list[dict[str, Any]]
    components: list[dict[str, Any]]
    subscore: float  # 0..100 (nan if nothing available)
    data_completeness: float  # 0..1
    history: list[dict[str, Any]] = field(default_factory=list)
    valuation_vs_peers: float = float("nan")  # 0..100, for the verdict


def _source_of(stats: dict[str, Any], key: str) -> str | None:
    s = stats.get(key)
    return getattr(s, "source", None) if s is not None else None


def _val(stats: dict[str, Any], key: str) -> Any:
    """Pull a plain value from a Sourced-or-raw stats dict."""
    s = stats.get(key)
    if s is None:
        return None
    return getattr(s, "value", s)


def _derive_metrics(stats: dict[str, Any], financials: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Compute metrics not directly provided (net_debt_ebitda, fcf_yield, interest_coverage)."""
    derived: dict[str, float] = {}
    total_debt = _val(stats, "total_debt")
    total_cash = _val(stats, "total_cash")
    ebitda = _val(stats, "ebitda")
    fcf = _val(stats, "free_cashflow")
    mcap = _val(stats, "market_cap")
    if all(v is not None for v in (total_debt, total_cash, ebitda)) and ebitda:
        derived["net_debt_ebitda"] = (total_debt - total_cash) / ebitda
    if fcf is not None and mcap:
        derived["fcf_yield"] = fcf / mcap
    # interest coverage from income statement (EBIT / interest expense)
    inc = financials.get("income")
    if isinstance(inc, pd.DataFrame) and not inc.empty:
        ebit = _stmt_row(inc, ["EBIT", "Operating Income", "Operating Income Or Loss"])
        interest = _stmt_row(inc, ["Interest Expense", "Interest Expense Non Operating"])
        if ebit is not None and interest:
            derived["interest_coverage"] = abs(ebit / interest) if interest else math.nan
    return derived


def _stmt_row(df: pd.DataFrame, names: list[str]):
    """Latest value of the first matching row label in a yfinance statement DataFrame."""
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            row = df.loc[n].dropna()
            if len(row):
                return float(row.iloc[0])
    return None


def _normalise_value(key: str, value: float) -> float:
    """yfinance dividendYield is sometimes a percentage; normalise to a fraction."""
    if value is None:
        return value
    if key == "dividend_yield" and value is not None and value > 1.0:
        return value / 100.0
    return value


def _percentile(value: float, peers: list[float], higher_is_better: bool) -> float:
    arr = np.array([p for p in peers if p is not None and not (isinstance(p, float) and math.isnan(p))])
    if arr.size == 0 or value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    pct = (arr < value).mean() * 100.0
    return pct if higher_is_better else 100.0 - pct


def analyse(
    company_type: str,
    key_stats: dict[str, Any],
    financials: dict[str, pd.DataFrame],
    overview: dict[str, Any],
    cfg,
    peer_metrics: dict[str, list[float]] | None = None,
    tasi_metrics: dict[str, float] | None = None,
) -> FundamentalResult:
    rubric = company_type if company_type in COMPONENT_METRICS else "general"
    comp_weights: dict[str, float] = dict(cfg.fundamental_score[rubric])
    metric_specs = cfg.metric_scoring.metrics
    blend = cfg.metric_scoring.blend_sector_percentile

    derived = _derive_metrics(key_stats, financials)
    peer_metrics = peer_metrics or {}
    tasi_metrics = tasi_metrics or {}

    metric_rows: list[dict[str, Any]] = []
    comp_scores: dict[str, float] = {}
    expected = 0
    present = 0

    for comp, metric_keys in COMPONENT_METRICS[rubric].items():
        scored_pairs: list[tuple[float, float]] = []
        for mk in metric_keys:
            expected += 1
            raw = derived.get(mk, _val(key_stats, mk))
            raw = _normalise_value(mk, raw)
            spec = metric_specs.get(mk)
            label, unit = METRIC_META.get(mk, (mk, ""))
            anchor = scoring.score_metric(raw, spec) if spec else float("nan")
            # blend with sector percentile if peers available
            higher = (spec or {}).get("direction") != "lower"
            pct = _percentile(raw, peer_metrics.get(mk, []), higher) if mk in peer_metrics else float("nan")
            if not scoring.is_missing(anchor) and not scoring.is_missing(pct):
                mscore = (1 - blend) * anchor + blend * pct
            else:
                mscore = anchor
            if not scoring.is_missing(raw):
                present += 1
            if not scoring.is_missing(mscore):
                scored_pairs.append((mscore, 1.0))  # equal weight within component
            metric_rows.append(
                {
                    "key": mk,
                    "label": label,
                    "value": raw,
                    "display": _display(mk, raw, unit),
                    "unit": unit,
                    "component": comp,
                    "source": _source_of(key_stats, mk),
                    "sector_median": _median(peer_metrics.get(mk)),
                    "tasi_value": tasi_metrics.get(mk),
                    "percentile": None if scoring.is_missing(pct) else round(pct, 1),
                    "metric_score": None if scoring.is_missing(mscore) else round(mscore, 1),
                }
            )
        cscore, _ = scoring.weighted_average(scored_pairs)
        comp_scores[comp] = cscore

    # Value-trap guard: a low multiple on a money-losing company is not "cheap value".
    # Cap the valuation component when the business is loss-making (negative ROE / margin / EPS).
    lossmaking = any(
        v is not None and v < 0
        for v in (_val(key_stats, "roe"), _val(key_stats, "net_margin"), _val(key_stats, "trailing_eps"))
    )
    if lossmaking and not scoring.is_missing(comp_scores.get("valuation", float("nan"))):
        comp_scores["valuation"] = min(comp_scores["valuation"], 40.0)

    # Re-normalise component weights over components that produced a score.
    present_comps = [c for c, s in comp_scores.items() if not scoring.is_missing(s)]
    norm_weights = scoring.renormalise(comp_weights, present_comps)
    subscore, _ = scoring.weighted_average(
        [(comp_scores[c], comp_weights[c]) for c in present_comps]
    )

    components = [
        {
            "name": c,
            "score": None if scoring.is_missing(comp_scores[c]) else round(comp_scores[c], 1),
            "weight": round(comp_weights[c], 3),
            "weight_used": round(norm_weights.get(c, 0.0), 3),
        }
        for c in comp_weights
    ]

    completeness = present / expected if expected else 0.0
    val_vs_peers = comp_scores.get("valuation", float("nan"))

    return FundamentalResult(
        rubric=rubric,
        metrics=metric_rows,
        components=components,
        subscore=float("nan") if scoring.is_missing(subscore) else round(subscore, 1),
        data_completeness=round(completeness, 3),
        history=_history(financials),
        valuation_vs_peers=float("nan") if scoring.is_missing(val_vs_peers) else round(val_vs_peers, 1),
    )


def _median(vals: list[float] | None):
    if not vals:
        return None
    arr = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.median(arr)) if arr else None


def _display(key: str, value: Any, unit: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    if key in _PERCENT_METRICS:
        return f"{value * 100:.1f}%"
    if unit == "x":
        return f"{value:.2f}x"
    return f"{value:,.2f}"


def _history(financials: dict[str, pd.DataFrame], years: int = 5) -> list[dict[str, Any]]:
    """Build a small multi-year history of key line items for sparklines."""
    inc = financials.get("income")
    if not isinstance(inc, pd.DataFrame) or inc.empty:
        return []
    rows: list[dict[str, Any]] = []
    revenue_names = ["Total Revenue", "Operating Revenue", "Revenue"]
    ni_names = ["Net Income", "Net Income Common Stockholders", "Net Income Continuous Operations"]
    cols = list(inc.columns)[:years]
    for col in cols:
        year = getattr(col, "year", None) or str(col)
        rev = _col_value(inc, revenue_names, col)
        ni = _col_value(inc, ni_names, col)
        rows.append(
            {
                "year": year,
                "revenue": rev,
                "net_income": ni,
                "net_margin": (ni / rev) if (rev and ni is not None) else None,
            }
        )
    return list(reversed(rows))


def _col_value(df: pd.DataFrame, names: list[str], col):
    for n in names:
        if n in df.index:
            v = df.loc[n, col]
            if pd.notna(v):
                return float(v)
    return None
