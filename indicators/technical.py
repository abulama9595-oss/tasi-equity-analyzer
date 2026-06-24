"""Vendored technical indicators + OHLCV resampling.

Implemented directly (no TA-Lib, no `ta` lib) so the dependency surface is small and
NumPy-2 safe, and every function is unit-testable against known reference values
(see tests/test_indicators.py and tests/test_resampling.py).

Conventions:
- Inputs are pandas Series/DataFrame indexed by date; outputs preserve that index.
- RSI / ATR / ADX use Wilder's smoothing (RMA) with an SMA seed, matching the values
  most charting platforms report.
- MACD / EMA use the standard exponential MA with adjust=False.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


# --------------------------------------------------------------------------- #
# Moving averages
# --------------------------------------------------------------------------- #
def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean().rename(f"sma_{period}")


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean().rename(f"ema_{period}")


def _rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (a.k.a. RMA): SMA seed at index `period`, then recursive."""
    vals = series.to_numpy(dtype=float)
    n = len(vals)
    out = np.full(n, np.nan)
    if n < period:
        return pd.Series(out, index=series.index)
    seed = np.nanmean(vals[:period])
    out[period - 1] = seed
    for i in range(period, n):
        prev = out[i - 1]
        cur = vals[i]
        if np.isnan(cur):
            out[i] = prev
        else:
            out[i] = (prev * (period - 1) + cur) / period
    return pd.Series(out, index=series.index)


# --------------------------------------------------------------------------- #
# RSI (Wilder)
# --------------------------------------------------------------------------- #
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    close = close.astype(float)
    delta = close.diff().to_numpy()
    n = len(close)
    gains = np.where(np.isnan(delta), 0.0, np.where(delta > 0, delta, 0.0))
    losses = np.where(np.isnan(delta), 0.0, np.where(delta < 0, -delta, 0.0))
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    if n > period:
        avg_gain[period] = gains[1 : period + 1].mean()
        avg_loss[period] = losses[1 : period + 1].mean()
        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        out = 100.0 - 100.0 / (1.0 + rs)
    # all-gain windows -> 100; all-flat -> NaN
    out = np.where((avg_loss == 0) & (avg_gain > 0), 100.0, out)
    out = np.where((avg_loss == 0) & (avg_gain == 0), np.nan, out)
    return pd.Series(out, index=close.index, name=f"rsi_{period}")


# --------------------------------------------------------------------------- #
# MACD
# --------------------------------------------------------------------------- #
def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return pd.DataFrame({"macd": line, "signal": sig, "hist": hist})


# --------------------------------------------------------------------------- #
# Bollinger Bands
# --------------------------------------------------------------------------- #
def bollinger(close: pd.Series, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(period, min_periods=period).mean()
    dev = close.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + std * dev
    lower = mid - std * dev
    width = (upper - lower) / mid
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_width": width})


# --------------------------------------------------------------------------- #
# ATR (Wilder)
# --------------------------------------------------------------------------- #
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return _rma(tr, period).rename(f"atr_{period}")


# --------------------------------------------------------------------------- #
# Stochastic oscillator (slow)
# --------------------------------------------------------------------------- #
def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3, smooth: int = 3
) -> pd.DataFrame:
    lowest = low.rolling(k, min_periods=k).min()
    highest = high.rolling(k, min_periods=k).max()
    rng = (highest - lowest).replace(0, np.nan)
    raw_k = 100 * (close - lowest) / rng
    k_line = raw_k.rolling(smooth, min_periods=smooth).mean()
    d_line = k_line.rolling(d, min_periods=d).mean()
    return pd.DataFrame({"stoch_k": k_line, "stoch_d": d_line})


# --------------------------------------------------------------------------- #
# ADX / DI (Wilder)
# --------------------------------------------------------------------------- #
def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )
    tr = true_range(high, low, close)
    atr_ = _rma(tr, period)
    plus_di = 100 * _rma(plus_dm, period) / atr_
    minus_di = 100 * _rma(minus_dm, period) / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = _rma(dx, period)
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx_line})


# --------------------------------------------------------------------------- #
# OBV
# --------------------------------------------------------------------------- #
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).fillna(0.0).cumsum().rename("obv")


# --------------------------------------------------------------------------- #
# Resampling — daily -> weekly / monthly (the spec calls this out as high-risk)
# --------------------------------------------------------------------------- #
def resample_ohlcv(df: pd.DataFrame, rule: str, drop_incomplete_trailing: bool = True) -> pd.DataFrame:
    """Resample daily OHLCV to a coarser timeframe with correct aggregation:
    open=first, high=max, low=min, close=last, volume=sum.

    Uses closed/label='right' so a weekly bar is labelled by its closing day (e.g. the
    Thursday that ends the Tadawul week with rule 'W-THU'). Optionally drops the trailing
    period if it hasn't closed yet (its right-edge label is beyond the last observation),
    which otherwise silently corrupts weekly/monthly RSI & MACD.
    """
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    cols = [c for c in AGG if c in df.columns]
    agg = {c: AGG[c] for c in cols}
    out = df[cols].resample(rule, label="right", closed="right").agg(agg)
    out = out.dropna(subset=["close"]) if "close" in out.columns else out.dropna(how="all")
    if drop_incomplete_trailing and not out.empty:
        last_obs = df.index.max()
        last_label = out.index.max()
        if last_label > last_obs:  # period still forming
            out = out.iloc[:-1]
    out.index.name = df.index.name or "date"
    return out


def slope(series: pd.Series, lookback: int = 5) -> float:
    """Normalised slope of the last `lookback` points (per-bar % change of a line fit)."""
    s = series.dropna()
    if len(s) < 2:
        return float("nan")
    y = s.iloc[-lookback:].to_numpy(dtype=float)
    if len(y) < 2:
        return float("nan")
    x = np.arange(len(y))
    m = np.polyfit(x, y, 1)[0]
    base = np.nanmean(y)
    return float(m / base) if base else float("nan")
