"""Validate the vendored indicators against TA-Lib (the industry-standard C library).

Skipped automatically when TA-Lib isn't installed (it is not a runtime dependency — it's
painful to build on hosts like Streamlit Cloud, so we vendor the math and only cross-check it
here). Run locally with `pip install --only-binary :all: TA-Lib` to confirm parity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

talib = pytest.importorskip("talib")

from indicators import technical as ti


@pytest.fixture(scope="module")
def ohlc():
    rng = np.random.RandomState(7)
    n = 600
    close = np.abs(60 + np.cumsum(rng.normal(0, 1.0, n))) + 5.0
    spread = rng.uniform(0.2, 2.0, n)
    high = close + spread
    low = close - spread
    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    return (pd.Series(close, index=idx), pd.Series(high, index=idx), pd.Series(low, index=idx))


def test_sma_matches_talib(ohlc):
    close, _, _ = ohlc
    assert ti.sma(close, 20).iloc[-1] == pytest.approx(talib.SMA(close.to_numpy(), 20)[-1], abs=1e-6)


def test_ema_matches_talib(ohlc):
    close, _, _ = ohlc  # different warmup seed, but converges on a long series
    assert ti.ema(close, 20).iloc[-1] == pytest.approx(talib.EMA(close.to_numpy(), 20)[-1], abs=1e-3)


def test_rsi_matches_talib(ohlc):
    close, _, _ = ohlc
    assert ti.rsi(close, 14).iloc[-1] == pytest.approx(talib.RSI(close.to_numpy(), 14)[-1], abs=0.02)


def test_macd_matches_talib(ohlc):
    close, _, _ = ohlc
    m = ti.macd(close, 12, 26, 9)
    tl_macd, tl_sig, tl_hist = talib.MACD(close.to_numpy(), 12, 26, 9)
    assert m["macd"].iloc[-1] == pytest.approx(tl_macd[-1], abs=1e-2)
    assert m["signal"].iloc[-1] == pytest.approx(tl_sig[-1], abs=1e-2)
    assert m["hist"].iloc[-1] == pytest.approx(tl_hist[-1], abs=1e-2)


def test_atr_matches_talib(ohlc):
    close, high, low = ohlc
    ours = ti.atr(high, low, close, 14).iloc[-1]
    tl = talib.ATR(high.to_numpy(), low.to_numpy(), close.to_numpy(), 14)[-1]
    assert ours == pytest.approx(tl, abs=0.02)


def test_adx_matches_talib(ohlc):
    close, high, low = ohlc
    ours = ti.adx(high, low, close, 14)["adx"].iloc[-1]
    tl = talib.ADX(high.to_numpy(), low.to_numpy(), close.to_numpy(), 14)[-1]
    assert ours == pytest.approx(tl, abs=1.0)  # Wilder DX smoothing; converges within ~1 pt


def test_stochastic_matches_talib(ohlc):
    close, high, low = ohlc
    st = ti.stochastic(high, low, close, 14, 3, 3)
    slowk, slowd = talib.STOCH(high.to_numpy(), low.to_numpy(), close.to_numpy(),
                               fastk_period=14, slowk_period=3, slowk_matype=0,
                               slowd_period=3, slowd_matype=0)
    assert st["stoch_k"].iloc[-1] == pytest.approx(slowk[-1], abs=0.5)
    assert st["stoch_d"].iloc[-1] == pytest.approx(slowd[-1], abs=0.5)
