"""Cross-sectional, point-in-time price factors for the research harness.

Each factor is defined in its NATURAL direction (no hypothesis baked in) — the Information
Coefficient then reveals the sign, and the walk-forward composite orients each factor by its
*past* IC. All factors are causal (use only data up to the bar), so sampling them at a date t
is point-in-time with no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import technical as ti

# factor name -> short description (natural direction)
FACTORS: dict[str, str] = {
    "mom_12_1": "12-1 momentum: return from t-12m to t-1m (skips last month)",
    "ret_1m": "last 1-month return (tests short-term reversal/continuation)",
    "ret_3y": "trailing 3-year return to t-1m (tests long-term reversal)",
    "px_vs_sma200": "price / 200-day MA - 1 (trend)",
    "dist_52w_high": "price / 252-day high - 1 (proximity to 52-week high, <=0)",
    "vol_126": "annualised volatility over last 126 days (tests low-vol anomaly)",
    "rsi_14": "RSI(14) level",
}


def compute_factors(price_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame (same daily index) of causal factor values + 'liquidity'."""
    c = price_df["close"].astype(float)
    v = price_df["volume"].astype(float) if "volume" in price_df.columns else pd.Series(np.nan, index=c.index)
    ret = c.pct_change()

    f = pd.DataFrame(index=price_df.index)
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1.0
    f["ret_1m"] = c / c.shift(21) - 1.0
    f["ret_3y"] = c.shift(21) / c.shift(756) - 1.0
    f["px_vs_sma200"] = c / c.rolling(200, min_periods=200).mean() - 1.0
    f["dist_52w_high"] = c / c.rolling(252, min_periods=126).max() - 1.0
    f["vol_126"] = ret.rolling(126, min_periods=126).std() * np.sqrt(252)
    f["rsi_14"] = ti.rsi(c, 14)
    f["liquidity"] = (c * v).rolling(21, min_periods=21).mean()  # avg daily SAR turnover
    return f
