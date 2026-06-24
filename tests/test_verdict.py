"""Validate scoring primitives, weight re-normalisation, and verdict mapping."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from analysis import scoring
from analysis import verdict as verdict_mod
from config.settings import get_config


# --------------------------- scoring primitives --------------------------- #
def test_linear_anchor_direction_and_clamp():
    # 'lower is better': p0=40 -> 0, p100=8 -> 100
    assert scoring.linear_anchor(40, 40, 8) == pytest.approx(0.0)
    assert scoring.linear_anchor(8, 40, 8) == pytest.approx(100.0)
    assert scoring.linear_anchor(24, 40, 8) == pytest.approx(50.0)
    # clamps beyond the anchors
    assert scoring.linear_anchor(60, 40, 8) == 0.0
    assert scoring.linear_anchor(4, 40, 8) == 100.0


def test_band_score():
    assert scoring.band_score(0.5, 0.3, 0.7) == 100.0
    assert scoring.band_score(0.3, 0.3, 0.7) == 100.0
    # one band-width (0.4) below low -> 0
    assert scoring.band_score(0.3 - 0.4, 0.3, 0.7) == pytest.approx(0.0)
    assert scoring.band_score(0.5 + 0.0, 0.3, 0.7) == 100.0


def test_weighted_average_drops_and_renormalises():
    # one missing value -> its weight is excluded, others renormalised
    score, coverage = scoring.weighted_average([(80.0, 0.5), (float("nan"), 0.3), (60.0, 0.2)])
    # present weights: 0.5 and 0.2 -> (80*0.5 + 60*0.2)/0.7
    assert score == pytest.approx((80 * 0.5 + 60 * 0.2) / 0.7)
    assert coverage == pytest.approx(0.7)


def test_weighted_average_all_missing():
    score, coverage = scoring.weighted_average([(float("nan"), 0.5), (float("nan"), 0.5)])
    assert math.isnan(score) and coverage == 0.0


def test_renormalise():
    w = scoring.renormalise({"a": 0.3, "b": 0.5, "c": 0.2}, ["a", "c"])
    assert w == pytest.approx({"a": 0.6, "c": 0.4})


# ------------------------------- verdict ---------------------------------- #
def _fund(subscore, val_peers=None, completeness=1.0, metrics=None):
    return SimpleNamespace(
        subscore=subscore,
        valuation_vs_peers=val_peers if val_peers is not None else subscore,
        data_completeness=completeness,
        metrics=metrics or [],
    )


def _tech(subscore):
    return SimpleNamespace(subscore=subscore, components=[], by_timeframe={})


def _trend(score):
    call = SimpleNamespace(
        horizon="Short term (weeks)", classification="Uptrend",
        composite_score=score, confidence="high", confidence_pct=0.8,
    )
    return SimpleNamespace(short_term=call, medium_term=call)


def _risk(score, completeness=1.0):
    return SimpleNamespace(
        risk_score=score, data_completeness=completeness, max_drawdown=-0.1, beta_vs_tasi=1.0
    )


def test_verdict_strong_buy_and_breakdown_reproduces_composite():
    cfg = get_config()
    v = verdict_mod.analyse(_fund(80), _tech(80), _trend(60), _risk(80), cfg)
    # trend (60 -> (60+100)/2 = 80); all inputs 80 -> composite 80 -> strong_buy
    assert v.composite == pytest.approx(80.0, abs=0.5)
    assert v.rating5 == "strong_buy"
    assert v.rating3 == "buy"
    # the running composite of the last breakdown row equals the composite
    last = v.breakdown[-1]["running_composite"]
    assert last == pytest.approx(v.composite, abs=0.2)
    # contributions of present inputs sum to the composite
    contribs = [r["contribution"] for r in v.breakdown if r["contribution"] is not None]
    assert sum(contribs) == pytest.approx(v.composite, abs=0.2)


def test_verdict_renormalises_when_input_missing():
    cfg = get_config()
    # trend missing entirely -> its weight is renormalised away; composite still 80
    no_trend = SimpleNamespace(short_term=None, medium_term=None)
    v = verdict_mod.analyse(_fund(80), _tech(80), no_trend, _risk(80), cfg)
    assert v.composite == pytest.approx(80.0, abs=0.5)
    trend_row = next(r for r in v.breakdown if r["input"] == "trend")
    assert trend_row["contribution"] is None
    assert trend_row["weight_used"] == 0.0


def test_verdict_sell_and_low_reliability():
    cfg = get_config()
    v = verdict_mod.analyse(
        _fund(25, completeness=0.2), _tech(25), _trend(-60), _risk(25, completeness=0.2), cfg
    )
    assert v.rating3 == "sell"
    assert v.rating5 in ("sell", "reduce")
    assert v.low_reliability is True
    assert v.conviction == "low"
