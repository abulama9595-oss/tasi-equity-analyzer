"""Prove daily -> weekly / monthly resampling aggregates correctly.

Incorrect resampling silently corrupts weekly/monthly RSI & MACD, so this is asserted
explicitly against a hand-constructed series with known per-period aggregates.
"""
from __future__ import annotations

import pandas as pd

from indicators.technical import resample_ohlcv


def _daily_two_tadawul_weeks() -> pd.DataFrame:
    # Tadawul trading days Sun-Thu. Two complete weeks ending on Thursdays:
    #   Week 1: 2024-01-07 (Sun) .. 2024-01-11 (Thu)
    #   Week 2: 2024-01-14 (Sun) .. 2024-01-18 (Thu)
    dates = [
        "2024-01-07", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11",
        "2024-01-14", "2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18",
    ]
    idx = pd.to_datetime(dates)
    return pd.DataFrame(
        {
            "open":   [10, 11, 12, 13, 14, 20, 21, 22, 23, 24],
            "high":   [15, 16, 12, 18, 14, 25, 26, 22, 28, 24],
            "low":    [ 9,  8, 12, 13,  7, 19, 18, 22, 23, 17],
            "close":  [11, 12, 12, 14, 13, 21, 22, 22, 24, 23],
            "volume": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        },
        index=idx,
    )


def test_weekly_aggregation_w_thu():
    df = _daily_two_tadawul_weeks()
    wk = resample_ohlcv(df, "W-THU", drop_incomplete_trailing=True)
    assert len(wk) == 2
    w1, w2 = wk.iloc[0], wk.iloc[1]

    # Week 1: open=first(10), high=max(15,16,12,18,14)=18, low=min(9,8,...)=7,
    #         close=last(13), volume=sum(1500)
    assert w1["open"] == 10
    assert w1["high"] == 18
    assert w1["low"] == 7
    assert w1["close"] == 13
    assert w1["volume"] == 1500

    # Week 2: open=20, high=28, low=17, close=23, volume=4000
    assert w2["open"] == 20
    assert w2["high"] == 28
    assert w2["low"] == 17
    assert w2["close"] == 23
    assert w2["volume"] == 4000

    # Labels are the Thursday closing days
    assert str(wk.index[0].date()) == "2024-01-11"
    assert str(wk.index[1].date()) == "2024-01-18"


def test_monthly_aggregation():
    # Three trading days in Jan + two in Feb 2024
    dates = pd.to_datetime(
        ["2024-01-10", "2024-01-20", "2024-01-31", "2024-02-05", "2024-02-29"]
    )
    df = pd.DataFrame(
        {
            "open":   [10, 11, 12, 20, 21],
            "high":   [15, 16, 17, 25, 22],
            "low":    [ 8,  9,  7, 19, 18],
            "close":  [12, 13, 16, 21, 22],
            "volume": [100, 200, 300, 400, 500],
        },
        index=dates,
    )
    mo = resample_ohlcv(df, "ME", drop_incomplete_trailing=True)
    assert len(mo) == 2
    jan, feb = mo.iloc[0], mo.iloc[1]
    assert jan["open"] == 10 and jan["high"] == 17 and jan["low"] == 7
    assert jan["close"] == 16 and jan["volume"] == 600
    assert feb["open"] == 20 and feb["high"] == 25 and feb["low"] == 18
    assert feb["close"] == 22 and feb["volume"] == 900


def test_drop_incomplete_trailing():
    # Data ends mid-week (Tuesday 2024-01-16) -> the forming week must be dropped.
    dates = pd.to_datetime(
        ["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-15", "2024-01-16"]
    )
    df = pd.DataFrame(
        {
            "open": [1, 2, 3, 4, 5, 6],
            "high": [1, 2, 3, 4, 5, 6],
            "low": [1, 2, 3, 4, 5, 6],
            "close": [1, 2, 3, 4, 5, 6],
            "volume": [1, 1, 1, 1, 1, 1],
        },
        index=dates,
    )
    kept = resample_ohlcv(df, "W-THU", drop_incomplete_trailing=True)
    dropped = resample_ohlcv(df, "W-THU", drop_incomplete_trailing=False)
    # Complete first week (ends Thu 01-11) kept; forming second week dropped
    assert len(kept) == 1
    assert str(kept.index[-1].date()) == "2024-01-11"
    # Without dropping, the forming week appears
    assert len(dropped) == 2
