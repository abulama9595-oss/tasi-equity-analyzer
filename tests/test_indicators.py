"""Validate vendored indicators against known reference values & invariants."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indicators import technical as ti


# Wilder's classic RSI dataset (widely reproduced, e.g. StockCharts). The first
# 14-period RSI value lands at index 14 and is ~70.5 across references.
WILDER_CLOSES = [
    44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08,
    45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64,
    46.21, 46.25, 45.71, 46.45, 45.78, 45.35, 44.03, 44.18, 44.22, 44.57,
    43.42, 42.66, 43.13,
]


def test_rsi_reference_value():
    close = pd.Series(WILDER_CLOSES)
    r = ti.rsi(close, period=14)
    # First defined RSI at index 14
    assert np.isnan(r.iloc[13])
    first = r.iloc[14]
    assert 69.5 <= first <= 71.5, f"first RSI={first}, expected ~70.5"
    # All defined values are within [0, 100]
    defined = r.dropna()
    assert (defined >= 0).all() and (defined <= 100).all()


def test_rsi_monotonic_extremes():
    up = pd.Series(np.arange(1, 60, dtype=float))  # strictly rising
    down = pd.Series(np.arange(60, 1, -1, dtype=float))  # strictly falling
    assert ti.rsi(up, 14).iloc[-1] == pytest.approx(100.0, abs=1e-6)
    assert ti.rsi(down, 14).iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_sma_exact():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = ti.sma(s, 3)
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[3] == pytest.approx(3.0)
    assert out.iloc[4] == pytest.approx(4.0)


def test_ema_exact():
    s = pd.Series([1, 2, 3, 4, 5], dtype=float)
    out = ti.ema(s, 3)  # alpha = 2/(3+1) = 0.5, adjust=False, seed = first value
    # e0=1, e1=0.5*2+0.5*1=1.5, e2=0.5*3+0.5*1.5=2.25, e3=3.125, e4=4.0625
    assert out.iloc[0] == pytest.approx(1.0)
    assert out.iloc[1] == pytest.approx(1.5)
    assert out.iloc[2] == pytest.approx(2.25)
    assert out.iloc[4] == pytest.approx(4.0625)


def test_macd_relationships():
    close = pd.Series(np.linspace(10, 50, 200) + np.sin(np.arange(200)) * 2)
    m = ti.macd(close, 12, 26, 9)
    # hist must equal line - signal everywhere it is defined
    valid = m.dropna()
    assert np.allclose(valid["hist"], valid["macd"] - valid["signal"])
    # On a strong uptrend the MACD line should end positive
    assert m["macd"].iloc[-1] > 0


def test_atr_positive_and_seeded():
    n = 60
    high = pd.Series(np.linspace(10, 20, n) + 0.5)
    low = pd.Series(np.linspace(10, 20, n) - 0.5)
    close = pd.Series(np.linspace(10, 20, n))
    a = ti.atr(high, low, close, 14)
    assert np.isnan(a.iloc[12])
    assert (a.dropna() > 0).all()


def test_adx_di_bounds():
    n = 120
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    trend = np.linspace(10, 40, n)
    high = pd.Series(trend + 1, index=idx)
    low = pd.Series(trend - 1, index=idx)
    close = pd.Series(trend, index=idx)
    out = ti.adx(high, low, close, 14)
    d = out.dropna()
    assert (d["adx"] >= 0).all() and (d["adx"] <= 100).all()
    # steady uptrend -> +DI dominates -DI
    assert d["plus_di"].iloc[-1] > d["minus_di"].iloc[-1]


def test_obv_direction():
    close = pd.Series([10, 11, 10, 12], dtype=float)
    vol = pd.Series([100, 200, 150, 300], dtype=float)
    o = ti.obv(close, vol)
    # +200 (up), -150 (down), +300 (up) => 0, 200, 50, 350
    assert list(o) == [0.0, 200.0, 50.0, 350.0]


def test_bollinger_structure():
    close = pd.Series(np.random.RandomState(0).normal(100, 5, 100))
    b = ti.bollinger(close, 20, 2)
    valid = b.dropna()
    assert (valid["bb_upper"] >= valid["bb_mid"]).all()
    assert (valid["bb_mid"] >= valid["bb_lower"]).all()
