"""Multi-timeframe technical analysis & sub-score.

Computes the full indicator suite on daily, weekly, and monthly series (weekly & monthly
RSI(14) and MACD(12,26,9) are required by the spec), derives signals in [-1,+1] per the
config's technical_score.signals map, and aggregates them into component scores (trend /
momentum / volume), then a per-timeframe score, then an overall 0..100 sub-score blended
by timeframe weights. Missing signals are dropped and weights re-normalised.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from indicators import technical as ti
from . import scoring


@dataclass
class TFData:
    timeframe: str
    df: pd.DataFrame  # resampled OHLCV
    ind: pd.DataFrame  # indicator overlays aligned to df.index
    latest: dict[str, Any]
    signals: dict[str, float]  # each in [-1,+1] or nan
    support: list[float] = field(default_factory=list)
    resistance: list[float] = field(default_factory=list)
    divergence: str | None = None
    score: float = float("nan")
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class TechnicalResult:
    subscore: float
    components: list[dict[str, Any]]
    by_timeframe: dict[str, TFData]
    timeframe_scores: dict[str, float]


def build_indicator_frame(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Compute all indicators for one timeframe; return a frame aligned to df.index."""
    ic = cfg.indicators
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    out = pd.DataFrame(index=df.index)
    out["close"] = close
    out["volume"] = vol
    for p in ic.sma_periods:
        out[f"sma_{p}"] = ti.sma(close, p)
    for p in ic.ema_periods:
        out[f"ema_{p}"] = ti.ema(close, p)
    out["rsi"] = ti.rsi(close, ic.rsi_period)
    macd = ti.macd(close, ic.macd.fast, ic.macd.slow, ic.macd.signal)
    out = out.join(macd)
    bb = ti.bollinger(close, ic.bollinger.period, ic.bollinger.std)
    out = out.join(bb)
    out["atr"] = ti.atr(high, low, close, ic.atr_period)
    st = ti.stochastic(high, low, close, ic.stochastic.k, ic.stochastic.d, ic.stochastic.smooth)
    out = out.join(st)
    adx = ti.adx(high, low, close, ic.adx_period)
    out = out.join(adx)
    out["obv"] = ti.obv(close, vol)
    return out


# --------------------------------------------------------------------------- #
# Signal helpers — each returns a value in [-1, +1] (or nan when undefined).
# --------------------------------------------------------------------------- #
def _clip(x: float, lo=-1.0, hi=1.0) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    return max(lo, min(hi, x))


def _longest_sma(ind: pd.DataFrame) -> tuple[str, pd.Series] | None:
    """Longest computable SMA among 200/100/50 (coarser timeframes lack enough bars for
    a 200-period MA — e.g. only ~120 monthly bars over 10y — so fall back gracefully)."""
    for p in (200, 100, 50):
        c = f"sma_{p}"
        if c in ind.columns and not ind[c].dropna().empty:
            return c, ind[c]
    return None


def _available_sma_periods(ind: pd.DataFrame) -> list[int]:
    out = []
    for p in (200, 100, 50):
        c = f"sma_{p}"
        if c in ind.columns and not math.isnan(ind[c].iloc[-1]):
            out.append(p)
    return out


def _sig_price_vs_sma200(ind: pd.DataFrame) -> float:
    ma = _longest_sma(ind)
    if ma is None:
        return float("nan")
    _, series = ma
    price = ind["close"].iloc[-1]
    mav = series.iloc[-1]
    if math.isnan(mav) or not mav:
        return float("nan")
    pos = _clip((price / mav - 1) / 0.10)
    slope = ti.slope(series, 5)
    slope_sig = _clip(np.sign(slope)) if not math.isnan(slope) else 0.0
    return _clip(0.7 * pos + 0.3 * slope_sig)


def _sig_sma50_vs_sma200(ind: pd.DataFrame) -> float:
    # Use the longest and shortest available MAs as the slow/fast pair for the cross.
    avail = _available_sma_periods(ind)
    if len(avail) < 2:
        return float("nan")
    long_p, short_p = avail[0], avail[-1]
    a, b = ind[f"sma_{short_p}"].iloc[-1], ind[f"sma_{long_p}"].iloc[-1]
    if math.isnan(a) or math.isnan(b) or not b:
        return float("nan")
    return _clip((a / b - 1) / 0.05)


def _sig_adx_di(ind: pd.DataFrame) -> float:
    for c in ("plus_di", "minus_di", "adx"):
        if c not in ind or math.isnan(ind[c].iloc[-1]):
            return float("nan")
    pdi, mdi, adx = ind["plus_di"].iloc[-1], ind["minus_di"].iloc[-1], ind["adx"].iloc[-1]
    di_diff = _clip((pdi - mdi) / 25.0)
    strength = _clip(adx / 30.0, 0.0, 1.0)
    return _clip(di_diff * strength)


def _sig_macd_hist(ind: pd.DataFrame) -> float:
    if "hist" not in ind or math.isnan(ind["hist"].iloc[-1]):
        return float("nan")
    hist = ind["hist"].iloc[-1]
    price = ind["close"].iloc[-1]
    return _clip(hist / (price * 0.02)) if price else float("nan")


def _sig_rsi(ind: pd.DataFrame) -> float:
    r = ind["rsi"].dropna()
    if r.empty:
        return float("nan")
    rsi = r.iloc[-1]
    delta = rsi - r.iloc[-2] if len(r) >= 2 else 0.0
    if rsi >= 70:
        return -0.3 if delta < 0 else 0.3
    if rsi <= 30:
        return 0.5 if delta > 0 else -0.3
    return _clip((rsi - 50) / 20.0)


def _sig_macd_cross(ind: pd.DataFrame) -> float:
    h = ind["hist"].dropna()
    if len(h) < 2:
        return float("nan")
    now, prev = h.iloc[-1], h.iloc[-2]
    if prev <= 0 < now:
        return 1.0
    if prev >= 0 > now:
        return -1.0
    return _clip(np.sign(now) * 0.3)


def _sig_stochastic(ind: pd.DataFrame) -> float:
    k = ind["stoch_k"].dropna()
    d = ind["stoch_d"].dropna()
    if k.empty or d.empty:
        return float("nan")
    kv, dv = k.iloc[-1], d.iloc[-1]
    base = _clip((kv - 50) / 50.0)
    cross = 0.3 if kv > dv else -0.3
    if kv < 20:
        return 0.5 if kv > dv else -0.2
    if kv > 80:
        return -0.5 if kv < dv else 0.2
    return _clip(0.6 * base + 0.4 * cross)


def _sig_obv_trend(ind: pd.DataFrame) -> float:
    obv = ind["obv"].dropna()
    vol = ind["volume"].dropna()
    if len(obv) < 10 or vol.empty:
        return float("nan")
    y = obv.iloc[-20:].to_numpy(dtype=float)
    x = np.arange(len(y))
    slope_per_bar = np.polyfit(x, y, 1)[0]
    avg_vol = vol.iloc[-20:].mean()
    return _clip(slope_per_bar / avg_vol) if avg_vol else float("nan")


def _sig_vol_vs_avg(ind: pd.DataFrame) -> float:
    vol = ind["volume"].dropna()
    close = ind["close"].dropna()
    if len(vol) < 20 or len(close) < 2:
        return float("nan")
    avg_vol = vol.iloc[-20:].mean()
    ratio = vol.iloc[-1] / avg_vol if avg_vol else 1.0
    excess = _clip(ratio - 1.0)
    direction = np.sign(close.iloc[-1] - close.iloc[-2])
    return _clip(direction * abs(excess))


_SIGNAL_FUNCS = {
    "price_vs_sma200": _sig_price_vs_sma200,
    "sma50_vs_sma200": _sig_sma50_vs_sma200,
    "adx_di": _sig_adx_di,
    "macd_hist": _sig_macd_hist,
    "rsi": _sig_rsi,
    "macd_cross": _sig_macd_cross,
    "stochastic": _sig_stochastic,
    "obv_trend": _sig_obv_trend,
    "vol_vs_avg": _sig_vol_vs_avg,
}

_SIGNAL_INTERP = {
    "price_vs_sma200": "Price vs long-term MA (with slope)",
    "sma50_vs_sma200": "Golden/death cross (50 vs long MA)",
    "adx_di": "Directional movement (ADX-weighted)",
    "macd_hist": "MACD histogram",
    "rsi": "RSI level & turn",
    "macd_cross": "MACD signal-line cross",
    "stochastic": "Stochastic %K/%D",
    "obv_trend": "On-balance-volume trend",
    "vol_vs_avg": "Volume vs average (directional)",
}


def _support_resistance(
    df: pd.DataFrame, ref_price: float | None, window: int = 5, n: int = 3
) -> tuple[list[float], list[float]]:
    """Swing pivots split around the *current* price: those below are support, those
    above are resistance. Both swing highs and lows are pooled (a former low above the
    current price acts as resistance, and vice versa — classic role reversal).

    ``ref_price`` must be the current/live price (latest daily close), NOT the timeframe's
    last bar close — otherwise on coarser timeframes (whose last *complete* bar can sit
    above the live price) a level could be mislabelled support when it is really above price.
    """
    if df.empty or ref_price is None or (isinstance(ref_price, float) and math.isnan(ref_price)):
        return [], []
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    pivots: set[float] = set()
    for i in range(window, len(df) - window):
        if highs[i] == highs[i - window : i + window + 1].max():
            pivots.add(round(float(highs[i]), 2))
        if lows[i] == lows[i - window : i + window + 1].min():
            pivots.add(round(float(lows[i]), 2))
    ref = round(float(ref_price), 2)  # compare on the same rounding as the pivots
    support = sorted([p for p in pivots if p < ref], reverse=True)[:n]
    resistance = sorted([p for p in pivots if p > ref])[:n]
    return support, resistance


def _divergence(ind: pd.DataFrame, lookback: int = 40) -> str | None:
    """Basic RSI/price divergence flag over the recent window."""
    sub = ind.dropna(subset=["rsi"]).tail(lookback)
    if len(sub) < 10:
        return None
    price = sub["close"]
    rsi = sub["rsi"]
    half = len(sub) // 2
    p1, p2 = price.iloc[:half].max(), price.iloc[half:].max()
    r1, r2 = rsi.iloc[:half].max(), rsi.iloc[half:].max()
    if p2 > p1 and r2 < r1:
        return "bearish (price higher high, RSI lower high)"
    pl1, pl2 = price.iloc[:half].min(), price.iloc[half:].min()
    rl1, rl2 = rsi.iloc[:half].min(), rsi.iloc[half:].min()
    if pl2 < pl1 and rl2 > rl1:
        return "bullish (price lower low, RSI higher low)"
    return None


def compute_timeframe(timeframe: str, df: pd.DataFrame, cfg, ref_price: float | None = None) -> TFData:
    ind = build_indicator_frame(df, cfg)
    signals = {k: fn(ind) for k, fn in _SIGNAL_FUNCS.items()}
    if ref_price is None and not df.empty:
        ref_price = float(df["close"].iloc[-1])
    support, resistance = _support_resistance(df, ref_price)
    latest = _latest_values(ind, cfg)
    tf = TFData(
        timeframe=timeframe,
        df=df,
        ind=ind,
        latest=latest,
        signals=signals,
        support=support,
        resistance=resistance,
        divergence=_divergence(ind),
    )
    _score_timeframe(tf, cfg)
    return tf


def _latest_values(ind: pd.DataFrame, cfg) -> dict[str, Any]:
    def last(col):
        if col not in ind:
            return None
        s = ind[col].dropna()
        return None if s.empty else float(s.iloc[-1])

    rsi = last("rsi")
    macd_line, macd_sig, macd_hist = last("macd"), last("signal"), last("hist")
    cross = None
    if macd_line is not None and macd_sig is not None:
        cross = "bullish" if macd_line > macd_sig else "bearish"
    rsi_state = None
    if rsi is not None:
        rsi_state = "overbought" if rsi >= cfg.indicators.rsi_overbought else (
            "oversold" if rsi <= cfg.indicators.rsi_oversold else "neutral"
        )
    return {
        "close": last("close"),
        "rsi": rsi,
        "rsi_state": rsi_state,
        "macd": macd_line,
        "macd_signal": macd_sig,
        "macd_hist": macd_hist,
        "macd_cross": cross,
        "sma_20": last("sma_20"),
        "sma_50": last("sma_50"),
        "sma_100": last("sma_100"),
        "sma_200": last("sma_200"),
        "ema_50": last("ema_50"),
        "atr": last("atr"),
        "adx": last("adx"),
        "plus_di": last("plus_di"),
        "minus_di": last("minus_di"),
        "stoch_k": last("stoch_k"),
        "stoch_d": last("stoch_d"),
        "bb_upper": last("bb_upper"),
        "bb_mid": last("bb_mid"),
        "bb_lower": last("bb_lower"),
        "bb_width": last("bb_width"),
    }


def _score_timeframe(tf: TFData, cfg) -> None:
    """Aggregate signals -> components (trend/momentum/volume) -> timeframe score 0..100."""
    sig_cfg = cfg.technical_score.signals
    comp_cfg = cfg.technical_score.components
    # group signal values by component
    by_comp: dict[str, list[tuple[float, float]]] = {c: [] for c in comp_cfg}
    for skey, meta in sig_cfg.items():
        comp = meta["component"]
        w = float(meta["weight"])
        val = tf.signals.get(skey, float("nan"))
        by_comp.setdefault(comp, []).append((val, w))
    comp_scores: dict[str, float] = {}
    for comp, pairs in by_comp.items():
        comp_signal, _ = scoring.weighted_average(pairs)  # in [-1,+1]
        comp_scores[comp] = scoring.signal_to_score(comp_signal)
    tf.components = comp_scores
    tf.score, _ = scoring.weighted_average(
        [(comp_scores.get(c, float("nan")), w) for c, w in comp_cfg.items()]
    )


def analyse(price_daily: pd.DataFrame, cfg) -> TechnicalResult:
    rs = cfg.timeframes.resample
    frames = {
        "daily": price_daily,
        "weekly": ti.resample_ohlcv(price_daily, rs.weekly_rule, rs.drop_incomplete_trailing),
        "monthly": ti.resample_ohlcv(price_daily, rs.monthly_rule, rs.drop_incomplete_trailing),
    }
    # Current/live price = latest daily close; used to classify support vs resistance
    # consistently across every timeframe.
    ref_price = float(price_daily["close"].iloc[-1]) if (price_daily is not None and not price_daily.empty) else None

    by_tf: dict[str, TFData] = {}
    for tf_name, fdf in frames.items():
        if fdf is None or fdf.empty:
            continue
        by_tf[tf_name] = compute_timeframe(tf_name, fdf, cfg, ref_price)

    tf_weights = cfg.technical_score.timeframe_weights
    tf_scores = {name: by_tf[name].score for name in by_tf}
    subscore, _ = scoring.weighted_average(
        [(tf_scores.get(name, float("nan")), w) for name, w in tf_weights.items()]
    )

    # overall component breakdown = timeframe-weighted average of component scores
    comp_cfg = cfg.technical_score.components
    components = []
    for comp in comp_cfg:
        cs, _ = scoring.weighted_average(
            [(by_tf[name].components.get(comp, float("nan")), tf_weights.get(name, 0))
             for name in by_tf]
        )
        components.append(
            {"name": comp, "score": None if scoring.is_missing(cs) else round(cs, 1),
             "weight": round(comp_cfg[comp], 3)}
        )

    return TechnicalResult(
        subscore=float("nan") if scoring.is_missing(subscore) else round(subscore, 1),
        components=components,
        by_timeframe=by_tf,
        timeframe_scores={k: (None if scoring.is_missing(v) else float(round(v, 1))) for k, v in tf_scores.items()},
    )


def signal_rows(tf: TFData) -> list[dict[str, Any]]:
    """Human-readable signal table for one timeframe."""
    rows = []
    for k, v in tf.signals.items():
        rows.append(
            {
                "key": k,
                "interpretation": _SIGNAL_INTERP.get(k, k),
                "value": None if scoring.is_missing(v) else round(v, 2),
            }
        )
    return rows
