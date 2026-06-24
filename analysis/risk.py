"""Risk metrics & risk sub-score.

Beta vs ^TASI.SR, annualised volatility, max drawdown, Sharpe (config risk-free proxy),
and a simple historical 95% VaR. The risk sub-score (0..100, higher = safer) maps each
metric through the config anchors in verdict.risk_score and re-normalises over what's
available.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import scoring

TRADING_DAYS = 252


@dataclass
class RiskResult:
    beta_vs_tasi: float | None
    annualized_vol: float | None
    max_drawdown: float | None
    sharpe: float | None
    var_95: float | None
    risk_score: float  # 0..100, higher = safer
    data_completeness: float


def _daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def analyse(price_daily: pd.DataFrame, index_daily: pd.DataFrame, cfg) -> RiskResult:
    rf = cfg.risk_free.annual_rate
    out: dict[str, Any] = {
        "beta_vs_tasi": None,
        "annualized_vol": None,
        "max_drawdown": None,
        "sharpe": None,
        "var_95": None,
    }
    if price_daily is None or price_daily.empty:
        return RiskResult(**out, risk_score=float("nan"), data_completeness=0.0)

    r = _daily_returns(price_daily["close"])
    if len(r) >= 30:
        vol = float(r.std() * math.sqrt(TRADING_DAYS))
        out["annualized_vol"] = vol
        mean_annual = float(r.mean() * TRADING_DAYS)
        out["sharpe"] = (mean_annual - rf) / vol if vol else None
        out["var_95"] = float(-np.percentile(r, 5))  # positive loss magnitude
        # max drawdown
        cum = (1 + r).cumprod()
        dd = cum / cum.cummax() - 1.0
        out["max_drawdown"] = float(dd.min())

    # beta vs TASI on aligned daily returns
    if index_daily is not None and not index_daily.empty:
        ri = _daily_returns(index_daily["close"])
        joined = pd.concat([r, ri], axis=1, join="inner").dropna()
        if len(joined) >= 30:
            cov = np.cov(joined.iloc[:, 0], joined.iloc[:, 1])
            var_i = cov[1, 1]
            out["beta_vs_tasi"] = float(cov[0, 1] / var_i) if var_i else None

    # risk sub-score
    rs_cfg = cfg.verdict.risk_score
    specs = rs_cfg["inputs"]
    weights = rs_cfg["weights"]
    pairs = []
    present = 0
    for key, spec in specs.items():
        val = out.get(key)
        if key == "max_drawdown" and val is not None:
            val = abs(val)
        s = scoring.score_metric(val, spec) if val is not None else float("nan")
        if not scoring.is_missing(s):
            present += 1
        pairs.append((s, weights[key]))
    score, _ = scoring.weighted_average(pairs)
    completeness = present / len(specs) if specs else 0.0

    return RiskResult(
        beta_vs_tasi=_round(out["beta_vs_tasi"], 3),
        annualized_vol=_round(out["annualized_vol"], 4),
        max_drawdown=_round(out["max_drawdown"], 4),
        sharpe=_round(out["sharpe"], 3),
        var_95=_round(out["var_95"], 4),
        risk_score=float("nan") if scoring.is_missing(score) else round(score, 1),
        data_completeness=round(completeness, 3),
    )


def _round(v, n):
    return None if v is None else round(v, n)
