"""Scoring primitives shared by fundamentals, technicals, trend, and verdict.

Implements the scoring semantics from the build spec (Appendix A):
- linear_anchor: score = clamp(0..100) linearly interpolated between p0 (->0) and
  p100 (->100). Works in either direction; `direction` only documents which end is good.
- band: full score (100) inside [low, high], decaying linearly to 0 as the value moves
  away by the same width on either side.
- signals emit a value in [-1, +1]; helpers convert between signal-space and 0..100.
- missing inputs are DROPPED and remaining weights re-normalised (never imputed).
"""
from __future__ import annotations

import math
from typing import Any, Iterable


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def linear_anchor(value: float, p0: float, p100: float) -> float:
    """Linear interpolation from p0 (->0) to p100 (->100), clamped to 0..100."""
    if is_missing(value) or p0 == p100:
        return float("nan")
    return clamp((value - p0) / (p100 - p0) * 100.0)


def band_score(value: float, low: float, high: float) -> float:
    """100 inside [low, high]; decays linearly to 0 over one band-width outside."""
    if is_missing(value):
        return float("nan")
    if low <= value <= high:
        return 100.0
    width = max(high - low, 1e-9)
    dist = (low - value) if value < low else (value - high)
    return clamp(100.0 * (1.0 - dist / width))


def score_metric(value: float, spec: dict[str, Any]) -> float:
    """Score a single metric per its config spec ({direction, p0, p100} or band)."""
    if is_missing(value):
        return float("nan")
    direction = spec.get("direction")
    if direction == "band":
        return band_score(value, spec["low"], spec["high"])
    # 'lower'/'higher' both use linear_anchor; p0->0, p100->100 already encodes direction
    return linear_anchor(value, spec["p0"], spec["p100"])


def signal_to_score(signal: float) -> float:
    """Map a signal in [-1,+1] to 0..100 (0 -> 50)."""
    if is_missing(signal):
        return float("nan")
    return clamp((signal + 1.0) * 50.0)


def weighted_average(pairs: Iterable[tuple[float, float]]) -> tuple[float, float]:
    """Weighted average over (value, weight) pairs, dropping NaN values and
    re-normalising weights over what's present.

    Returns (score, coverage) where coverage = present_weight / total_weight in [0,1].
    score is NaN if nothing is present.
    """
    total_w = 0.0
    present_w = 0.0
    acc = 0.0
    for value, weight in pairs:
        total_w += weight
        if not is_missing(value):
            present_w += weight
            acc += value * weight
    if present_w == 0:
        return float("nan"), 0.0
    coverage = present_w / total_w if total_w else 0.0
    return acc / present_w, coverage


def renormalise(weights: dict[str, float], present_keys: Iterable[str]) -> dict[str, float]:
    """Re-normalise a weight map over the subset of keys that are present."""
    present = set(present_keys)
    sub = {k: w for k, w in weights.items() if k in present}
    tot = sum(sub.values())
    if tot == 0:
        return {}
    return {k: w / tot for k, w in sub.items()}
