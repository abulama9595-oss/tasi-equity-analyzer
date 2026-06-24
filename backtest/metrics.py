"""Backtest analytics: Information Coefficient, quantile-bucket returns, rating hit-rates,
and a top-quantile long-only equity curve. No scipy dependency (Spearman via ranks)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .engine import HORIZONS


def _spearman(a: pd.Series, b: pd.Series, min_names: int = 5) -> float:
    m = a.notna() & b.notna()
    if m.sum() < min_names:
        return float("nan")
    ra, rb = a[m].rank(), b[m].rank()
    if ra.std(ddof=0) == 0 or rb.std(ddof=0) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def ic_summary(df: pd.DataFrame, score_col: str, ret_col: str, min_names: int = 5) -> dict:
    """Mean cross-sectional rank-IC across dates, with a t-stat (mean / SE)."""
    ics = []
    for _, g in df.groupby("date"):
        ic = _spearman(g[score_col], g[ret_col], min_names)
        if not math.isnan(ic):
            ics.append(ic)
    ics = np.array(ics, dtype=float)
    if ics.size == 0:
        return {"mean_ic": float("nan"), "t_stat": float("nan"), "n_dates": 0, "pct_positive": float("nan")}
    mean = float(ics.mean())
    std = float(ics.std(ddof=1)) if ics.size > 1 else float("nan")
    t = mean / (std / math.sqrt(ics.size)) if std and std > 0 else float("nan")
    return {"mean_ic": mean, "t_stat": t, "n_dates": int(ics.size),
            "pct_positive": float((ics > 0).mean()), "ic_series": ics}


def bucket_returns(df: pd.DataFrame, score_col: str, ret_col: str, n: int = 5, min_names: int = 10):
    """Average forward return per score quantile (0=lowest score), averaged across dates,
    plus the top-minus-bottom spread and its t-stat across dates."""
    per_bucket, spreads = [], []
    for _, g in df.groupby("date"):
        gg = g[[score_col, ret_col]].dropna()
        if len(gg) < min_names:
            continue
        q = pd.qcut(gg[score_col].rank(method="first"), n, labels=False)
        m = gg.groupby(q)[ret_col].mean()
        per_bucket.append(m)
        if (n - 1) in m.index and 0 in m.index:
            spreads.append(m[n - 1] - m[0])
    if not per_bucket:
        return None
    mat = pd.DataFrame(per_bucket)
    means = mat.mean()
    spreads = np.array(spreads, dtype=float)
    t = (spreads.mean() / (spreads.std(ddof=1) / math.sqrt(spreads.size))
         if spreads.size > 1 and spreads.std(ddof=1) > 0 else float("nan"))
    return {"buckets": means, "spread_mean": float(np.nanmean(spreads)) if spreads.size else float("nan"),
            "spread_t": t, "n_dates": int(spreads.size)}


def rating_hit_rates(df: pd.DataFrame, ret_col: str) -> dict:
    """Per 3-tier rating: mean forward return, % positive, % beating the index."""
    out = {}
    idx_col = f"idx_{ret_col}"
    for r in ("buy", "hold", "sell"):
        g = df[df["rating"] == r]
        rr = g[ret_col].dropna()
        entry = {"n": int(len(rr)),
                 "mean_ret": float(rr.mean()) if len(rr) else float("nan"),
                 "pct_positive": float((rr > 0).mean()) if len(rr) else float("nan")}
        if idx_col in g.columns:
            ex = (g[ret_col] - g[idx_col]).dropna()
            entry["beat_index"] = float((ex > 0).mean()) if len(ex) else float("nan")
        out[r] = entry
    return out


def equity_curve(df: pd.DataFrame, score_col: str, ret_col: str = "fwd_1m",
                 top_frac: float = 0.2, min_names: int = 10, rf_annual: float = 0.052) -> dict:
    """Long-only equal-weight top-quantile portfolio rebalanced monthly vs the TASI index.
    Uses the non-overlapping next-month (~21d) return. Returns cumulative curves + stats."""
    idx_col = f"idx_{ret_col}"
    recs = []
    for d, g in df.groupby("date"):
        gg = g[[score_col, ret_col] + ([idx_col] if idx_col in g.columns else [])].dropna(subset=[score_col, ret_col])
        if len(gg) < min_names:
            continue
        thr = gg[score_col].quantile(1 - top_frac)
        sel = gg[gg[score_col] >= thr]
        bench = float(gg[idx_col].iloc[0]) if idx_col in gg.columns and gg[idx_col].notna().any() else float(gg[ret_col].mean())
        recs.append({"date": d, "port": float(sel[ret_col].mean()), "bench": bench})
    if not recs:
        return {}
    c = pd.DataFrame(recs).set_index("date").sort_index()
    c["port_cum"] = (1 + c["port"]).cumprod()
    c["bench_cum"] = (1 + c["bench"]).cumprod()
    return {"curve": c, "port": _series_stats(c["port"], rf_annual), "bench": _series_stats(c["bench"], rf_annual)}


def _series_stats(monthly: pd.Series, rf_annual: float) -> dict:
    monthly = monthly.dropna()
    if monthly.empty:
        return {"cagr": float("nan"), "sharpe": float("nan"), "max_dd": float("nan"), "n": 0}
    cum = (1 + monthly).prod()
    yrs = len(monthly) / 12.0
    cagr = cum ** (1 / yrs) - 1 if yrs > 0 and cum > 0 else float("nan")
    ann_ret = monthly.mean() * 12
    ann_vol = monthly.std(ddof=1) * math.sqrt(12) if len(monthly) > 1 else float("nan")
    sharpe = (ann_ret - rf_annual) / ann_vol if ann_vol and ann_vol > 0 else float("nan")
    eq = (1 + monthly).cumprod()
    max_dd = float((eq / eq.cummax() - 1).min())
    return {"cagr": float(cagr), "sharpe": float(sharpe), "max_dd": max_dd, "n": int(len(monthly))}


SCORE_COLS = ["technical", "trend", "risk", "price_composite", "fundamental", "full_composite"]


def full_report_data(df: pd.DataFrame) -> dict:
    """Compute IC + bucket + (for composites) equity-curve results across all score columns
    and horizons. Returns a nested dict consumed by the report writer."""
    res = {"n_rows": len(df), "n_dates": df["date"].nunique() if not df.empty else 0,
           "n_tickers": df["ticker"].nunique() if not df.empty else 0,
           "date_min": str(df["date"].min().date()) if not df.empty else None,
           "date_max": str(df["date"].max().date()) if not df.empty else None,
           "ic": {}, "buckets": {}, "rating": {}, "equity": {}}
    for sc in SCORE_COLS:
        if sc not in df.columns or df[sc].notna().sum() == 0:
            continue
        res["ic"][sc] = {h: ic_summary(df, sc, h) for h in HORIZONS}
        res["buckets"][sc] = {h: bucket_returns(df, sc, h) for h in HORIZONS}
    res["equity"]["price_composite"] = equity_curve(df, "price_composite", "fwd_1m")
    if "full_composite" in df.columns and df["full_composite"].notna().sum() > 0:
        res["equity"]["full_composite"] = equity_curve(df, "full_composite", "fwd_1m")
    res["rating"] = {h: rating_hit_rates(df, h) for h in HORIZONS}
    return res
