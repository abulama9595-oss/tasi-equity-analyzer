"""Probabilistic trend assessment — NOT a price forecast.

Combines trend structure (MA slope, ADX/DI, MACD, higher-highs/higher-lows) with relative
strength vs TASI into a composite trend score in [-100, +100], classified into
Strong Uptrend / Uptrend / Sideways / Downtrend / Strong Downtrend. Confidence is the
share of inputs agreeing with the composite's sign. Each call states the levels to watch
and what would invalidate it. No deterministic price target is produced.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from indicators import technical as ti
from . import scoring
from .technicals import TFData


@dataclass
class TrendCall:
    horizon: str
    classification: str
    composite_score: float  # [-100, 100]
    confidence: str  # high | medium | low
    confidence_pct: float
    reasons: list[str] = field(default_factory=list)
    levels_to_watch: list[float] = field(default_factory=list)
    invalidation: str | None = None
    inputs: dict[str, float] = field(default_factory=dict)


@dataclass
class TrendResult:
    short_term: TrendCall | None
    medium_term: TrendCall | None


def _clip(x, lo=-1.0, hi=1.0):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    return max(lo, min(hi, x))


def _ma_slope(tf: TFData) -> float:
    if "sma_50" in tf.ind and not tf.ind["sma_50"].dropna().empty:
        return _clip(ti.slope(tf.ind["sma_50"], 6) * 50.0)
    return _clip(ti.slope(tf.ind["close"], 6) * 50.0)


def _structure(tf: TFData, lookback: int = 12) -> float:
    df = tf.df.tail(lookback)
    if len(df) < 6:
        return float("nan")
    half = len(df) // 2
    hi1, hi2 = df["high"].iloc[:half].max(), df["high"].iloc[half:].max()
    lo1, lo2 = df["low"].iloc[:half].min(), df["low"].iloc[half:].min()
    score = 0.0
    score += 0.5 if hi2 > hi1 else -0.5
    score += 0.5 if lo2 > lo1 else -0.5
    return _clip(score)


def _rel_strength(price_daily: pd.DataFrame, index_daily: pd.DataFrame, bars: int) -> float:
    if price_daily.empty or index_daily is None or index_daily.empty:
        return float("nan")
    s = price_daily["close"].tail(bars)
    i = index_daily["close"].tail(bars)
    if len(s) < 2 or len(i) < 2:
        return float("nan")
    stock_ret = s.iloc[-1] / s.iloc[0] - 1
    index_ret = i.iloc[-1] / i.iloc[0] - 1
    return _clip((stock_ret - index_ret) / 0.10)


def _classify(score: float, cfg) -> str:
    c = cfg.trend.classification
    if score >= c["strong_uptrend_min"]:
        return "Strong Uptrend"
    if score >= c["uptrend_min"]:
        return "Uptrend"
    if c["sideways_band"][0] <= score <= c["sideways_band"][1]:
        return "Sideways"
    if score <= c["strong_downtrend_max"]:
        return "Strong Downtrend"
    if score <= c["downtrend_max"]:
        return "Downtrend"
    return "Sideways"


def _confidence(inputs: dict[str, float], composite: float, cfg) -> tuple[str, float]:
    present = [v for v in inputs.values() if not scoring.is_missing(v)]
    if not present or composite == 0:
        return "low", 0.0
    sign = np.sign(composite)
    agree = sum(1 for v in present if np.sign(v) == sign)
    pct = agree / len(present)
    hi = cfg.trend.confidence["high_if_agree_pct"]
    lo = cfg.trend.confidence["low_if_agree_pct"]
    level = "high" if pct >= hi else ("low" if pct < lo else "medium")
    return level, round(pct, 2)


_LABELS = {
    "ma_slope": "MA slope",
    "adx_di": "ADX/DI direction",
    "macd": "MACD",
    "structure_hhhl": "Price structure (HH/HL)",
    "rel_strength_tasi": "Relative strength vs TASI",
}


def _call(horizon: str, tf: TFData, price_daily, index_daily, rs_bars: int, cfg) -> TrendCall:
    w = cfg.trend.inputs
    macd_sig = tf.signals.get("macd_hist", float("nan"))
    if scoring.is_missing(macd_sig):
        macd_sig = tf.signals.get("macd_cross", float("nan"))
    inputs = {
        "ma_slope": _ma_slope(tf),
        "adx_di": tf.signals.get("adx_di", float("nan")),
        "macd": macd_sig,
        "structure_hhhl": _structure(tf),
        "rel_strength_tasi": _rel_strength(price_daily, index_daily, rs_bars),
    }
    composite_signal, _ = scoring.weighted_average([(inputs[k], w[k]) for k in w])
    composite = 0.0 if scoring.is_missing(composite_signal) else round(composite_signal * 100, 1)
    classification = _classify(composite, cfg)
    conf, pct = _confidence(inputs, composite, cfg)

    reasons = []
    for k, v in inputs.items():
        if scoring.is_missing(v):
            continue
        if abs(v) < 0.15:
            continue
        direction = "supports upside" if v > 0 else "weighs to the downside"
        reasons.append(f"{_LABELS[k]} {direction} ({v:+.2f})")
    if not reasons:
        reasons.append("Inputs are mixed/flat — no decisive trend.")

    support = tf.support[:1]
    resistance = tf.resistance[:1]
    levels = sorted(set(support + resistance))
    invalidation = None
    bar = "weekly" if horizon.startswith("Short") else "monthly"
    if composite > 0 and support:
        invalidation = f"A {bar} close below {support[0]:.2f} would weaken this uptrend call."
    elif composite < 0 and resistance:
        invalidation = f"A {bar} close above {resistance[0]:.2f} would weaken this downtrend call."
    elif resistance or support:
        edge = resistance[0] if resistance else support[0]
        invalidation = f"A decisive {bar} break of {edge:.2f} would resolve the range."

    return TrendCall(
        horizon=horizon,
        classification=classification,
        composite_score=composite,
        confidence=conf,
        confidence_pct=pct,
        reasons=reasons,
        levels_to_watch=levels,
        invalidation=invalidation,
        inputs={k: (None if scoring.is_missing(v) else round(v, 2)) for k, v in inputs.items()},
    )


def analyse(price_daily, index_daily, technical_result, cfg) -> TrendResult:
    by_tf = technical_result.by_timeframe
    weeks = cfg.trend.horizons["short_term_weeks"]
    months = cfg.trend.horizons["medium_term_months"]

    short = None
    if "weekly" in by_tf:
        short = _call("Short term (weeks)", by_tf["weekly"], price_daily, index_daily, weeks * 5, cfg)
    elif "daily" in by_tf:
        short = _call("Short term (weeks)", by_tf["daily"], price_daily, index_daily, weeks * 5, cfg)

    medium = None
    if "monthly" in by_tf:
        medium = _call("Medium term (months)", by_tf["monthly"], price_daily, index_daily, months * 21, cfg)
    elif "weekly" in by_tf:
        medium = _call("Medium term (months)", by_tf["weekly"], price_daily, index_daily, months * 21, cfg)

    return TrendResult(short_term=short, medium_term=medium)
