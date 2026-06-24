"""Reusable UI building blocks: plotly charts, score widgets, provenance chips.

Kept free of Streamlit calls where practical (pure figure/format builders) so they are
easy to reuse from the report exporter too.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Source -> short, friendly label for provenance chips
SOURCE_LABEL = {
    "yfinance": "Yahoo",
    "saudi_exchange": "Saudi Exchange",
    "registry": "Bundled ref",
    "none": "—",
    None: "—",
}

RATING_COLOR = {
    "buy": "#1a7f37",
    "hold": "#9a6700",
    "sell": "#cf222e",
}
RATING5_COLOR = {
    "strong_buy": "#116329",
    "buy": "#1a7f37",
    "hold": "#9a6700",
    "reduce": "#bc4c00",
    "sell": "#cf222e",
}


def fmt_sourced(s: Any) -> str:
    v = getattr(s, "value", s)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return v


def source_of(s: Any) -> str | None:
    return getattr(s, "source", None)


def provenance_chip(source: str | None) -> str:
    label = SOURCE_LABEL.get(source, source or "—")
    return f"`{label}`"


def score_color(score: float | None) -> str:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "#57606a"
    if score >= 65:
        return "#1a7f37"
    if score >= 45:
        return "#9a6700"
    return "#cf222e"


def big_number(value, unit: str = "", nd: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    if abs(value) >= 1e9:
        return f"{value/1e9:,.2f}B{unit}"
    if abs(value) >= 1e6:
        return f"{value/1e6:,.2f}M{unit}"
    return f"{value:,.{nd}f}{unit}"


def candlestick_figure(tf, cfg, title: str = "") -> go.Figure:
    """3-pane chart for one timeframe: price+MA+Bollinger, RSI, MACD."""
    df = tf.df
    ind = tf.ind
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(title or "Price", "RSI(14)", "MACD(12,26,9)"),
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="Price", increasing_line_color="#1a7f37", decreasing_line_color="#cf222e",
        ),
        row=1, col=1,
    )
    # Moving-average overlays
    for p, color in [(20, "#0969da"), (50, "#8250df"), (200, "#bc4c00")]:
        col = f"sma_{p}"
        if col in ind and ind[col].notna().any():
            fig.add_trace(
                go.Scatter(x=ind.index, y=ind[col], name=f"SMA{p}",
                           line=dict(width=1.2, color=color)),
                row=1, col=1,
            )
    # Bollinger bands
    if "bb_upper" in ind and ind["bb_upper"].notna().any():
        fig.add_trace(go.Scatter(x=ind.index, y=ind["bb_upper"], name="BB up",
                                 line=dict(width=0.8, color="#8c959f", dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=ind.index, y=ind["bb_lower"], name="BB low",
                                 line=dict(width=0.8, color="#8c959f", dash="dot"),
                                 fill="tonexty", fillcolor="rgba(140,149,159,0.08)"), row=1, col=1)

    # RSI
    if "rsi" in ind:
        fig.add_trace(go.Scatter(x=ind.index, y=ind["rsi"], name="RSI",
                                 line=dict(width=1.2, color="#0969da")), row=2, col=1)
        fig.add_hline(y=cfg.indicators.rsi_overbought, line=dict(color="#cf222e", width=0.7, dash="dash"), row=2, col=1)
        fig.add_hline(y=cfg.indicators.rsi_oversold, line=dict(color="#1a7f37", width=0.7, dash="dash"), row=2, col=1)

    # MACD
    if "macd" in ind:
        colors = ["#1a7f37" if (v is not None and v >= 0) else "#cf222e" for v in ind["hist"]]
        fig.add_trace(go.Bar(x=ind.index, y=ind["hist"], name="Hist", marker_color=colors,
                             opacity=0.5), row=3, col=1)
        fig.add_trace(go.Scatter(x=ind.index, y=ind["macd"], name="MACD",
                                 line=dict(width=1.2, color="#0969da")), row=3, col=1)
        fig.add_trace(go.Scatter(x=ind.index, y=ind["signal"], name="Signal",
                                 line=dict(width=1.2, color="#bc4c00")), row=3, col=1)

    fig.update_layout(
        height=620, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.02, x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def sparkline(values: list[float], color: str = "#0969da") -> go.Figure:
    fig = go.Figure(go.Scatter(y=values, mode="lines", line=dict(color=color, width=2)))
    fig.update_layout(height=60, margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      showlegend=False)
    return fig


def subscore_bar(name: str, score: float | None) -> str:
    """A compact HTML bar for a sub-score (used in markdown via unsafe_allow_html)."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        pct, label, color = 0, "N/A", "#57606a"
    else:
        pct, label, color = score, f"{score:.0f}", score_color(score)
    return (
        f"<div style='margin:4px 0;'>"
        f"<div style='display:flex;justify-content:space-between;font-size:0.85rem;'>"
        f"<span>{name}</span><span style='color:{color};font-weight:600'>{label}</span></div>"
        f"<div style='background:#eaeef2;border-radius:6px;height:8px;'>"
        f"<div style='width:{pct}%;background:{color};height:8px;border-radius:6px;'></div></div></div>"
    )
