"""Chart builders and tooltip-aware HTML tables for the dark theme.

Pure builders (no Streamlit calls) so they can be reused by the report exporter too.
Colours come from the same palette as ui/theme.py.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui import theme as T

# palette (kept in sync with theme.py :root)
BG = "rgba(0,0,0,0)"
GRID = "rgba(255,255,255,0.06)"
TEXT = "#e6edf3"
DIM = "#8b97a7"
UP = "#22c55e"
DOWN = "#ef4444"
CYAN = "#22d3ee"
VIOLET = "#a78bfa"
AMBER = "#f59e0b"
EMERALD = "#10b981"

SOURCE_LABEL = {
    "yfinance": "Yahoo",
    "saudi_exchange": "Saudi Exchange",
    "registry": "Bundled ref",
    "none": "—",
    None: "—",
}

RATING5_COLOR = {
    "strong_buy": "#116329", "buy": "#1a7f37", "hold": "#9a6700",
    "reduce": "#bc4c00", "sell": "#cf222e",
}


def _missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def source_of(s: Any) -> str | None:
    return getattr(s, "source", None)


def fmt_sourced(s: Any) -> str:
    v = getattr(s, "value", s)
    return "N/A" if _missing(v) else v


def big_number(value, unit: str = "", nd: int = 2) -> str:
    if _missing(value):
        return "N/A"
    if abs(value) >= 1e12:
        return f"{value/1e12:,.2f}T{unit}"
    if abs(value) >= 1e9:
        return f"{value/1e9:,.2f}B{unit}"
    if abs(value) >= 1e6:
        return f"{value/1e6:,.2f}M{unit}"
    return f"{value:,.{nd}f}{unit}"


def source_chip(source: str | None) -> str:
    return T.chip(SOURCE_LABEL.get(source, source or "—"))


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def candlestick_figure(tf, cfg, title: str = "") -> go.Figure:
    df, ind = tf.df, tf.ind
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.045,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(title or "Price", "RSI(14)", "MACD(12,26,9)"),
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="Price", increasing_line_color=UP, decreasing_line_color=DOWN,
            increasing_fillcolor=UP, decreasing_fillcolor=DOWN,
        ),
        row=1, col=1,
    )
    for p, color in [(20, CYAN), (50, VIOLET), (200, AMBER)]:
        col = f"sma_{p}"
        if col in ind and ind[col].notna().any():
            fig.add_trace(go.Scatter(x=ind.index, y=ind[col], name=f"SMA{p}",
                                     line=dict(width=1.3, color=color)), row=1, col=1)
    if "bb_upper" in ind and ind["bb_upper"].notna().any():
        fig.add_trace(go.Scatter(x=ind.index, y=ind["bb_upper"], name="BB", legendgroup="bb",
                                 line=dict(width=0.7, color="rgba(139,151,167,.5)", dash="dot"),
                                 showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=ind.index, y=ind["bb_lower"], name="BB", legendgroup="bb",
                                 line=dict(width=0.7, color="rgba(139,151,167,.5)", dash="dot"),
                                 fill="tonexty", fillcolor="rgba(139,151,167,0.06)",
                                 showlegend=False), row=1, col=1)
    if "rsi" in ind:
        fig.add_trace(go.Scatter(x=ind.index, y=ind["rsi"], name="RSI",
                                 line=dict(width=1.4, color=CYAN)), row=2, col=1)
        fig.add_hline(y=cfg.indicators.rsi_overbought, line=dict(color=DOWN, width=0.7, dash="dash"), row=2, col=1)
        fig.add_hline(y=cfg.indicators.rsi_oversold, line=dict(color=UP, width=0.7, dash="dash"), row=2, col=1)
    if "macd" in ind:
        colors = [UP if (v is not None and v >= 0) else DOWN for v in ind["hist"]]
        fig.add_trace(go.Bar(x=ind.index, y=ind["hist"], name="Hist", marker_color=colors,
                             opacity=0.45), row=3, col=1)
        fig.add_trace(go.Scatter(x=ind.index, y=ind["macd"], name="MACD",
                                 line=dict(width=1.3, color=CYAN)), row=3, col=1)
        fig.add_trace(go.Scatter(x=ind.index, y=ind["signal"], name="Signal",
                                 line=dict(width=1.3, color=AMBER)), row=3, col=1)
    fig.update_layout(
        height=620, margin=dict(l=8, r=8, t=42, b=8),
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=DIM, family="Inter", size=12),
        xaxis_rangeslider_visible=False, hovermode="x unified",
        legend=dict(orientation="h", y=1.04, x=0, bgcolor="rgba(0,0,0,0)", font=dict(color=DIM)),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False)
    for ann in fig.layout.annotations:  # subplot titles
        ann.font.color = TEXT
        ann.font.size = 13
    return fig


# --------------------------------------------------------------------------- #
# Tooltip-aware HTML tables
# --------------------------------------------------------------------------- #
def _score_cls(score) -> str:
    if score is None:
        return "t-dim"
    return "t-pos" if score >= 65 else ("t-neg" if score < 45 else "")


def metrics_table_html(metrics: list[dict[str, Any]]) -> str:
    head = ("<tr><th>Metric</th><th>Value</th><th>Source</th>"
            "<th class='num'>Sector</th><th class='num'>TASI</th>"
            "<th class='num'>Pctile</th><th class='num'>Score</th></tr>")
    rows = []
    for m in metrics:
        name = T.gloss(m["label"])  # tooltip from glossary
        score = m["metric_score"]
        sec = "—" if m["sector_median"] is None else f"{m['sector_median']:.2f}"
        tasi = "—" if m["tasi_value"] is None else f"{m['tasi_value']:.2f}"
        pct = "—" if m["percentile"] is None else f"{m['percentile']:.0f}"
        scell = "—" if score is None else f"{score:.0f}"
        rows.append(
            f"<tr><td>{name}</td><td>{m['display']}</td>"
            f"<td>{source_chip(m['source'])}</td>"
            f"<td class='num t-dim'>{sec}</td><td class='num t-dim'>{tasi}</td>"
            f"<td class='num t-dim'>{pct}</td>"
            f"<td class='num {_score_cls(score)}'>{scell}</td></tr>"
        )
    return f"<table class='tasi-table'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def signals_table_html(rows: list[dict[str, Any]]) -> str:
    head = "<tr><th>Signal</th><th class='num'>Value (−1…+1)</th></tr>"
    body = []
    for r in rows:
        v = r["value"]
        if v is None:
            cell, cls = "—", "t-dim"
        else:
            cell, cls = f"{v:+.2f}", ("t-pos" if v > 0.05 else ("t-neg" if v < -0.05 else "t-dim"))
        body.append(f"<tr><td>{r['interpretation']}</td><td class='num {cls}'>{cell}</td></tr>")
    return f"<table class='tasi-table'><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"


_BREAKDOWN_LABEL = {
    "fundamental": "Fundamentals", "technical": "Technical", "trend": "Trend",
    "valuation_vs_peers": "Valuation vs peers", "risk": "Risk",
}


def breakdown_table_html(breakdown: list[dict[str, Any]]) -> str:
    head = ("<tr><th>Input</th><th class='num'>Value</th><th class='num'>Weight</th>"
            "<th class='num'>Weight used</th><th class='num'>Contribution</th>"
            "<th class='num'>Running</th></tr>")
    rows = []
    for b in breakdown:
        label = _BREAKDOWN_LABEL.get(b["input"], b["input"])
        val = "—" if b["value"] is None else f"{b['value']:.1f}"
        contrib = "—" if b["contribution"] is None else f"{b['contribution']:.2f}"
        rows.append(
            f"<tr><td>{label}</td><td class='num'>{val}</td>"
            f"<td class='num t-dim'>{b['weight']:.2f}</td>"
            f"<td class='num t-dim'>{b['weight_used']:.2f}</td>"
            f"<td class='num'>{contrib}</td>"
            f"<td class='num'>{b['running_composite']:.1f}</td></tr>"
        )
    return f"<table class='tasi-table'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def simple_table_html(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{'' if c is None else c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table class='tasi-table'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
