"""Verdict — composite Buy/Hold/Sell with a fully auditable breakdown.

Blends fundamental, technical, trend, valuation-vs-peers and risk sub-scores (weights from
config) into a 0..100 composite, maps it to a 5-tier rating (+ the 3-tier view), and emits
a breakdown table where every input -> value -> weight -> contribution -> running composite
is shown, so the verdict is reproducible by hand. Includes plain-language bull/bear points,
a conviction indicator, and a data-completeness indicator that lowers reliability when
fundamentals are missing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import scoring

_RATING_LABEL = {
    "strong_buy": "Strong Buy",
    "buy": "Buy (Accumulate)",
    "hold": "Hold",
    "reduce": "Reduce",
    "sell": "Sell",
}


@dataclass
class VerdictResult:
    rating5: str
    rating5_label: str
    rating3: str  # buy | hold | sell
    composite: float  # 0..100
    breakdown: list[dict[str, Any]] = field(default_factory=list)
    bull: list[str] = field(default_factory=list)
    bear: list[str] = field(default_factory=list)
    conviction: str = "medium"
    data_completeness: float = 0.0
    low_reliability: bool = False
    summary: str = ""  # plain-English reasoning for the call


def _rating5(composite: float, cfg) -> str:
    b = cfg.verdict.rating_bands
    if composite >= b["strong_buy_min"]:
        return "strong_buy"
    if composite >= b["buy_min"]:
        return "buy"
    if composite >= b["hold_min"]:
        return "hold"
    if composite >= b["reduce_min"]:
        return "reduce"
    return "sell"


def _to_three(rating5: str, cfg) -> str:
    m = cfg.verdict.three_tier_map
    for tier, members in m.items():
        if rating5 in members:
            return tier
    return "hold"


def _trend_to_score(trend_result) -> tuple[float, float]:
    """Average available trend composites (-100..100) -> 0..100, plus a confidence proxy."""
    calls = [c for c in (trend_result.short_term, trend_result.medium_term) if c is not None]
    if not calls:
        return float("nan"), 0.0
    scores = [(c.composite_score + 100) / 2.0 for c in calls]
    conf = sum(c.confidence_pct for c in calls) / len(calls)
    return sum(scores) / len(scores), conf


def analyse(fundamental, technical, trend, risk, cfg) -> VerdictResult:
    weights: dict[str, float] = dict(cfg.verdict.weights)
    trend_score, trend_conf = _trend_to_score(trend)

    inputs: dict[str, float] = {
        "fundamental": fundamental.subscore,
        "technical": technical.subscore,
        "trend": trend_score,
        "valuation_vs_peers": fundamental.valuation_vs_peers,
        "risk": risk.risk_score,
    }
    # per-input data availability for the completeness indicator
    availability: dict[str, float] = {
        "fundamental": fundamental.data_completeness,
        "technical": 1.0 if not scoring.is_missing(technical.subscore) else 0.0,
        "trend": 1.0 if not scoring.is_missing(trend_score) else 0.0,
        "valuation_vs_peers": 1.0 if not scoring.is_missing(fundamental.valuation_vs_peers) else 0.0,
        "risk": risk.data_completeness,
    }

    present = [k for k, v in inputs.items() if not scoring.is_missing(v)]
    norm = scoring.renormalise(weights, present)
    composite, _ = scoring.weighted_average([(inputs[k], weights[k]) for k in present])
    composite = 0.0 if scoring.is_missing(composite) else round(composite, 1)

    # auditable breakdown with running composite
    breakdown = []
    running = 0.0
    for k in weights:
        val = inputs[k]
        used_w = norm.get(k, 0.0)
        contribution = (val * used_w) if (k in present) else None
        if contribution is not None:
            running += contribution
        breakdown.append(
            {
                "input": k,
                "value": None if scoring.is_missing(val) else round(val, 1),
                "weight": round(weights[k], 3),
                "weight_used": round(used_w, 3),
                "contribution": None if contribution is None else round(contribution, 2),
                "running_composite": round(running, 2),
            }
        )

    rating5 = _rating5(composite, cfg)
    rating3 = _to_three(rating5, cfg)

    # Overall data completeness is fundamentals-dominant: fundamentals are the field set
    # most prone to gaps for smaller Saudi names, and the spec wants reliability to drop
    # visibly when they are missing. Infra inputs (technical/trend/risk) come from price
    # data and are usually present, so they get the smaller share.
    infra_keys = ["technical", "trend", "valuation_vs_peers", "risk"]
    infra = sum(availability[k] for k in infra_keys) / len(infra_keys)
    completeness = round(0.7 * availability["fundamental"] + 0.3 * infra, 3)
    dc_cfg = cfg.verdict.data_completeness
    low_reliability = completeness < dc_cfg["low_reliability_below"]

    conviction = _conviction(inputs, present, completeness, trend_conf, dc_cfg)
    bull, bear = _rationale(fundamental, technical, trend, risk, inputs)
    summary = _summary(rating3, composite, inputs, present, trend, low_reliability)

    return VerdictResult(
        rating5=rating5,
        rating5_label=_RATING_LABEL[rating5],
        rating3=rating3,
        composite=composite,
        breakdown=breakdown,
        bull=bull,
        bear=bear,
        conviction=conviction,
        data_completeness=completeness,
        low_reliability=low_reliability,
        summary=summary,
    )


# Plain-language names for each verdict input, used in the reasoning summary.
_PILLAR_NAMES = {
    "fundamental": "fundamentals",
    "technical": "the technical setup",
    "trend": "the trend",
    "valuation_vs_peers": "valuation vs peers",
    "risk": "the risk profile",
}


def _summary(rating3, composite, inputs, present, trend, low_reliability) -> str:
    """A 2-3 sentence, plain-English rationale for the verdict."""
    call = {"buy": "a BUY", "hold": "a HOLD", "sell": "a SELL"}[rating3]
    parts = [f"This screens as {call}, with a composite score of {composite:.0f}/100."]

    if present:
        scores = {k: inputs[k] for k in present}
        best = max(scores, key=scores.get)
        worst = min(scores, key=scores.get)
        if best != worst:
            parts.append(
                f"{_PILLAR_NAMES[best].capitalize()} is the strongest pillar "
                f"({scores[best]:.0f}/100) and {_PILLAR_NAMES[worst]} the weakest "
                f"({scores[worst]:.0f}/100)."
            )
        else:
            parts.append(f"{_PILLAR_NAMES[best].capitalize()} scores {scores[best]:.0f}/100.")

    st_call, mt = trend.short_term, trend.medium_term
    if st_call and mt and st_call.classification != mt.classification:
        parts.append(
            f"Timeframes diverge — {st_call.classification.lower()} over the coming weeks "
            f"but {mt.classification.lower()} over months."
        )

    if low_reliability:
        parts.append("Fundamentals coverage is thin, so treat this with reduced confidence.")

    return " ".join(parts)


def _conviction(inputs, present, completeness, trend_conf, dc_cfg) -> str:
    """High when data is ample, sub-scores agree, and the signal is not borderline."""
    if completeness < dc_cfg["low_reliability_below"]:
        return "low"
    vals = [inputs[k] for k in present]
    if len(vals) < 2:
        return "low"
    spread = max(vals) - min(vals)  # disagreement among sub-scores
    agree = spread < 35
    ample = completeness >= dc_cfg["full_confidence_min"]
    decisive = (sum(vals) / len(vals) >= 65) or (sum(vals) / len(vals) <= 40)
    score = sum([agree, ample, decisive, trend_conf >= 0.6])
    return "high" if score >= 3 else ("medium" if score >= 1 else "low")


def _rationale(fundamental, technical, trend, risk, inputs) -> tuple[list[str], list[str]]:
    bull, bear = [], []

    def tag(label, score):
        if scoring.is_missing(score):
            return
        if score >= 62:
            bull.append(f"{label} is favourable (score {score:.0f}/100).")
        elif score <= 42:
            bear.append(f"{label} is weak (score {score:.0f}/100).")

    tag("Fundamentals", fundamental.subscore)
    tag("Technical posture", technical.subscore)
    tag("Valuation vs peers", fundamental.valuation_vs_peers)
    tag("Risk profile", risk.risk_score)

    # trend specifics
    for call in (trend.short_term, trend.medium_term):
        if call is None:
            continue
        if call.composite_score >= 20:
            bull.append(f"{call.horizon}: {call.classification} ({call.confidence} confidence).")
        elif call.composite_score <= -20:
            bear.append(f"{call.horizon}: {call.classification} ({call.confidence} confidence).")

    # standout fundamental metrics
    for m in fundamental.metrics:
        if m["metric_score"] is None:
            continue
        if m["metric_score"] >= 80:
            bull.append(f"{m['label']} screens well ({m['display']}).")
        elif m["metric_score"] <= 20:
            bear.append(f"{m['label']} is a concern ({m['display']}).")

    # risk callouts
    if risk.max_drawdown is not None and risk.max_drawdown <= -0.45:
        bear.append(f"Deep historical drawdown ({risk.max_drawdown*100:.0f}%).")
    if risk.beta_vs_tasi is not None and risk.beta_vs_tasi <= 0.8:
        bull.append(f"Defensive beta vs TASI ({risk.beta_vs_tasi:.2f}).")

    # de-duplicate, keep 3-5 each
    bull = list(dict.fromkeys(bull))[:5]
    bear = list(dict.fromkeys(bear))[:5]
    if not bull:
        bull.append("No standout positives — signals are mixed.")
    if not bear:
        bear.append("No major red flags in the available data.")
    return bull, bear
