"""TASI Equity Analyzer — Streamlit entry point.

Single-ticker analysis for the Saudi market (Tadawul / TASI): overview, type-aware
fundamentals, multi-timeframe technicals, a probabilistic trend assessment, and a clear,
auditable Buy/Hold/Sell verdict — plus risk, dividends, an indicative Shariah screen, and a
methodology page. Dark "terminal" theme; every abbreviation has a hover tooltip.

NOT financial advice. For personal research only.
"""
from __future__ import annotations

import math

import streamlit as st

import analyzer as az
from analysis.technicals import signal_rows
from config.settings import get_config
from data.ticker_registry import TickerRegistry
from ui import components as C
from ui import theme as T

st.set_page_config(page_title="TASI Equity Analyzer", page_icon="📈",
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown(T.inject_css(), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_resources():
    cfg = get_config()
    registry = TickerRegistry()
    composite, sahmk = az.build_composite(cfg)
    return cfg, registry, composite, sahmk


@st.cache_data(show_spinner=False, ttl=3600)
def run_analysis(ticker: str):
    cfg, registry, composite, sahmk = get_resources()
    return az.analyze(ticker, cfg, registry, composite, sahmk)


def _v(s):
    return getattr(s, "value", None) if s is not None else None


def _missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def html(s: str):
    st.markdown(s, unsafe_allow_html=True)


cfg, registry, composite, sahmk = get_resources()

# ---------- header ---------- #
html(
    "<div class='brand'><div class='brand-mark'>📈</div>"
    "<div><div class='brand-title'>TASI Equity Analyzer</div>"
    "<div class='brand-sub'>Saudi market (Tadawul) · single-ticker research · all values in SAR</div>"
    "</div></div>"
)
html(
    "<div class='disclaimer'><b>Not financial advice — personal research only.</b> "
    "Every score is transparent and auditable; verify each input before acting. "
    "Quotes and fundamentals may be delayed or incomplete.</div>"
)

with st.sidebar:
    st.header("Settings")
    st.write(f"**Saudi Exchange (SAHMK):** {'🟢 connected' if sahmk.available else '⚪ off (yfinance-only)'}")
    st.write(f"**Risk-free rate (Sharpe):** {cfg.risk_free.annual_rate:.1%}")
    if st.button("Clear data cache"):
        n = az.DiskCache(cfg.cache_dir(), cfg.cache.ttl_seconds).clear()
        run_analysis.clear()
        st.success(f"Cleared {n} cached files.")
    st.caption("Add SAHMK_API_KEY to .env to enrich Saudi-Exchange fields.")

# ---------- search ---------- #
col_in, col_btn = st.columns([4, 1])
with col_in:
    query = st.text_input("Ticker", value=st.session_state.get("ticker", "1120"),
                          placeholder="e.g. 1120, 2222.SR, Aramco, الراجحي",
                          label_visibility="collapsed")
with col_btn:
    go = st.button("Analyze", type="primary", width="stretch")

if query and not query.strip().isdigit():
    sugg = registry.search(query, limit=6)
    if sugg:
        html("<div style='color:var(--text-dim);font-size:.82rem;margin:-4px 0 4px'>Suggestions: "
             + " · ".join(f"<b>{s.code}</b> {s.name_en}" for s in sugg) + "</div>")

html("<div style='color:var(--text-faint);font-size:.8rem'>Examples: 1120 (Al Rajhi · bank) · "
     "2222 (Aramco) · 7010 (stc) · 8210 (Bupa · insurance) · 4330 (Riyad REIT)</div>")

if not (go or query):
    st.stop()

ticker = query.strip()
st.session_state["ticker"] = ticker
with st.spinner(f"Analyzing {ticker}…"):
    r = run_analysis(ticker)

if r.error:
    st.error(r.error)
    for w in r.warnings:
        st.info(w)
    st.stop()
for w in r.warnings:
    st.info(w, icon="ℹ️")


# --------------------------------------------------------------------------- #
tabs = st.tabs(["Overview", "Fundamentals", "Technical", "Trend", "🟢 Verdict",
                "Risk & Income", "Shariah", "Methodology"])
ov = r.overview


# ----------------------------- Overview ------------------------------------ #
with tabs[0]:
    name_en = _v(ov.get("name_en")) or r.ref.name_en or r.ticker
    name_ar = _v(ov.get("name_ar"))
    ar = f"<span style='color:var(--text-dim);font-size:1.05rem'>{name_ar}</span>" if name_ar else ""
    html(f"<div style='display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;margin-bottom:2px'>"
         f"<span style='font-size:1.55rem;font-weight:800'>{name_en}</span>"
         f"{T.pill(r.ticker,'accent')}{ar}</div>")
    html(f"<div style='color:var(--text-dim);font-size:.86rem;margin-bottom:10px'>"
         f"Sector: <b style='color:var(--text)'>{_v(ov.get('sector')) or 'N/A'}</b> "
         f"{C.source_chip(C.source_of(ov.get('sector')))} · "
         f"Industry: {_v(ov.get('industry')) or 'N/A'} · "
         f"Type: {T.pill(r.company_type.title(),'neutral')}</div>")

    price = _v(ov.get("price"))
    delayed = getattr(ov.get("price"), "delayed", False)
    dc = r.day_change_pct
    price_sub = ("⏱ delayed · " if delayed else "") + C.SOURCE_LABEL.get(C.source_of(ov.get("price")), "—")
    cards = [
        T.stat_card(f"Price ({r.currency})", "N/A" if _missing(price) else f"{price:,.2f}",
                    delta=None if dc is None else f"{dc:+.2f}%",
                    delta_kind=("good" if (dc or 0) >= 0 else "bad"),
                    sub=f"<span class='chip'>{price_sub}</span>",
                    accent=("var(--good)" if (dc or 0) >= 0 else "var(--bad)")),
        T.stat_card("Market cap", C.big_number(_v(ov.get("market_cap"))), term="Market cap",
                    sub=C.source_chip(C.source_of(ov.get("market_cap")))),
        T.stat_card("Shares out.", C.big_number(_v(ov.get("shares_outstanding"))), term="Shares out.",
                    sub=C.source_chip(C.source_of(ov.get("shares_outstanding")))),
        T.stat_card("Free float", C.big_number(_v(ov.get("free_float"))), term="Free float",
                    sub=C.source_chip(C.source_of(ov.get("free_float")))),
    ]
    html(T.stat_grid(cards))

    if r.range_52w:
        rng = r.range_52w
        pos = max(2, min(98, rng["position_pct"]))
        html(f"<div class='tasi-card' style='margin-top:14px'>"
             f"<div class='stat-label'>{T.gloss('52-week range', label='52-week range')} ({r.currency})</div>"
             f"<div class='range-wrap' style='margin-top:10px'>"
             f"<div class='range-track'><div class='range-dot' style='left:{pos}%'></div></div>"
             f"<div class='range-labels'><span>Low {rng['low']:.2f}</span>"
             f"<span>{rng['position_pct']:.0f}% of range</span>"
             f"<span>High {rng['high']:.2f}</span></div></div></div>")

    desc = _v(ov.get("description"))
    if desc:
        with st.expander("Business description"):
            st.write(desc)
    with st.expander("Data provenance — which source supplied each field"):
        rows = [[k, C.SOURCE_LABEL.get(v, v)] for k, v in r.provenance.items()]
        html(C.simple_table_html(["Field", "Source"], rows))


# --------------------------- Fundamentals ---------------------------------- #
with tabs[1]:
    f = r.fundamentals
    html(T.section(f"Fundamentals — {f.rubric.title()} rubric",
                   "Metric set & weights switch by company type. Hover any metric name for its meaning."))
    left, right = st.columns([1, 2])
    with left:
        html(T.stat_card("Fundamental sub-score", "N/A" if _missing(f.subscore) else f"{f.subscore:.0f}",
                         term="Sub-score", sub=f"data completeness {f.data_completeness:.0%}",
                         accent=T.score_color(f.subscore)))
        html("<div style='height:8px'></div>")
        bars = "".join(T.score_bar(c["name"].replace("_", " ").title(), c["score"],
                                   weight=c["weight_used"]) for c in f.components)
        html(f"<div class='tasi-card'>{bars}</div>")
    with right:
        html(C.metrics_table_html(f.metrics))
        html("<div style='color:var(--text-faint);font-size:.76rem;margin-top:6px'>"
             "Metric score = 0.65 × anchor + 0.35 × sector percentile (anchor-only when no peer set).</div>")
    if f.history:
        html("<div style='height:10px'></div>")
        html(T.section("Multi-year history"))
        rows = [[h.get("year"),
                 C.big_number(h.get("revenue")) if h.get("revenue") else "—",
                 C.big_number(h.get("net_income")) if h.get("net_income") else "—",
                 "—" if h.get("net_margin") is None else f"{h['net_margin']*100:.1f}%"]
                for h in f.history]
        html(C.simple_table_html(["Year", "Revenue", "Net income", "Net margin"], rows))


# ---------------------------- Technical ------------------------------------ #
with tabs[2]:
    t = r.technical
    html(T.section("Technical analysis", "Daily, weekly & monthly. Hover RSI / MACD / ADX / ATR for meanings."))
    cards = [T.stat_card("Technical sub-score", "N/A" if _missing(t.subscore) else f"{t.subscore:.0f}",
                         term="Sub-score", accent=T.score_color(t.subscore))]
    for name, sc in t.timeframe_scores.items():
        cards.append(T.stat_card(f"{name.title()} score", "N/A" if sc is None else f"{sc:.0f}",
                                 accent=T.score_color(sc)))
    html(T.stat_grid(cards, min_w=150))

    avail = [tf for tf in ("monthly", "weekly", "daily") if tf in t.by_timeframe]
    tf_choice = st.radio("Timeframe", avail, horizontal=True, index=0, label_visibility="collapsed")
    tfd = t.by_timeframe[tf_choice]
    st.plotly_chart(C.candlestick_figure(tfd, cfg, title=f"{r.ticker} — {tf_choice}"),
                    width="stretch", config={"displayModeBar": False})

    lat = tfd.latest
    mcards = [
        T.stat_card("RSI(14)", "N/A" if _missing(lat["rsi"]) else f"{lat['rsi']:.1f}", term="RSI(14)",
                    sub=lat["rsi_state"]),
        T.stat_card("MACD", "N/A" if _missing(lat["macd"]) else f"{lat['macd']:.3f}", term="MACD",
                    sub=lat["macd_cross"]),
        T.stat_card("ADX", "N/A" if _missing(lat["adx"]) else f"{lat['adx']:.1f}", term="ADX"),
        T.stat_card("ATR", "N/A" if _missing(lat["atr"]) else f"{lat['atr']:.2f}", term="ATR"),
    ]
    html(T.stat_grid(mcards, min_w=150))

    cL, cR = st.columns(2)
    with cL:
        html(T.section("Signals", "Each contributes −1…+1 to the sub-score."))
        html(C.signals_table_html(signal_rows(tfd)))
    with cR:
        html(T.section("Key levels"))
        sup = ", ".join(f"{x:.2f}" for x in tfd.support) or "—"
        res = ", ".join(f"{x:.2f}" for x in tfd.resistance) or "—"
        html(f"<div class='tasi-card'><div style='margin-bottom:6px'>{T.gloss('Support')}: <b>{sup}</b></div>"
             f"<div>{T.gloss('Resistance')}: <b>{res}</b></div>"
             + (f"<div style='margin-top:10px'>{T.pill('Divergence','warn')} {tfd.divergence}</div>"
                if tfd.divergence else "") + "</div>")


# ------------------------------ Trend -------------------------------------- #
with tabs[3]:
    tr = r.trend
    html(T.section("Trend assessment", "A probabilistic classification — NOT a price forecast."))
    cols = st.columns(2)
    _tcolor = {"Strong Uptrend": "var(--good)", "Uptrend": "var(--good)", "Sideways": "var(--warn)",
               "Downtrend": "var(--bad)", "Strong Downtrend": "var(--bad)"}
    for col, call in zip(cols, (tr.short_term, tr.medium_term)):
        if call is None:
            continue
        with col:
            color = _tcolor.get(call.classification, "var(--text)")
            reasons = "".join(f"<li>{T._esc(x)}</li>" for x in call.reasons)
            levels = ", ".join(f"{x:.2f}" for x in call.levels_to_watch) or "—"
            inval = (f"<div class='hero-flag' style='background:rgba(245,158,11,.12);"
                     f"border-color:rgba(245,158,11,.35);color:#fbbf57'>{T._esc(call.invalidation)}</div>"
                     if call.invalidation else "")
            html(
                f"<div class='tasi-card'>"
                f"<div class='stat-label'>{T._esc(call.horizon)}</div>"
                f"<div style='font-size:1.5rem;font-weight:800;color:{color};margin:4px 0'>{call.classification}</div>"
                f"<div class='stat-sub'>{T.gloss('Composite', label='trend score')} {call.composite_score} "
                f"· {T.gloss('Conviction', label='confidence')} <b style='color:var(--text)'>{call.confidence}</b> "
                f"({call.confidence_pct:.0%} agree)</div>"
                f"<div style='margin:12px 0 4px;font-weight:600;font-size:.86rem'>Why</div>"
                f"<ul style='margin:0;padding-left:18px;font-size:.86rem'>{reasons}</ul>"
                f"<div style='margin-top:10px;font-size:.86rem'><b>Levels to watch:</b> {levels}</div>"
                f"{inval}</div>"
            )


# ----------------------------- Verdict ------------------------------------- #
with tabs[4]:
    v = r.verdict
    html(T.verdict_hero(v.rating5_label, v.rating3, v.composite, v.summary,
                        v.conviction, v.data_completeness, v.low_reliability))

    html("<div style='height:14px'></div>")
    html(T.section("How the score is built", "Weighted blend of five pillars (re-normalised over what's available)."))
    bars = "".join(
        T.score_bar(C._BREAKDOWN_LABEL.get(b["input"], b["input"]), b["value"], weight=b["weight_used"])
        for b in v.breakdown
    )
    html(f"<div class='tasi-card'>{bars}</div>")

    html("<div style='height:14px'></div>")
    bull = T.reason_list(v.bull, "bull")
    bear = T.reason_list(v.bear, "bear")
    html(f"<div class='reasons-wrap'>{bull}{bear}</div>")

    with st.expander("Show full calculation — auditable breakdown"):
        html(C.breakdown_table_html(v.breakdown))
        st.caption("Contribution = Value × Weight used. Missing inputs are dropped and weights re-normalised. "
                   "The running column reproduces the composite by hand.")

    html("<div style='height:10px'></div>")
    from reports.export import build_html
    st.download_button("⬇ Download full HTML report", data=build_html(r, cfg),
                       file_name=f"{r.ref.code or r.ticker}_tasi_report.html", mime="text/html")
    for d in r.disclaimers:
        st.caption(f"• {d}")


# -------------------------- Risk & Income ---------------------------------- #
with tabs[5]:
    rk, dv = r.risk, r.dividends
    html(T.section("Risk metrics", "Hover each metric for its meaning. Risk sub-score: higher = safer."))
    rcards = [
        T.stat_card("Beta vs TASI", "N/A" if rk.beta_vs_tasi is None else f"{rk.beta_vs_tasi:.2f}", term="Beta vs TASI"),
        T.stat_card("Ann. volatility", "N/A" if rk.annualized_vol is None else f"{rk.annualized_vol:.1%}", term="Ann. volatility"),
        T.stat_card("Max drawdown", "N/A" if rk.max_drawdown is None else f"{rk.max_drawdown:.1%}", term="Max drawdown"),
        T.stat_card("Sharpe", "N/A" if rk.sharpe is None else f"{rk.sharpe:.2f}", term="Sharpe"),
        T.stat_card("VaR 95% (1d)", "N/A" if rk.var_95 is None else f"{rk.var_95:.1%}", term="VaR 95% (1d)"),
        T.stat_card("Risk sub-score", "N/A" if _missing(rk.risk_score) else f"{rk.risk_score:.0f}",
                    term="Sub-score", accent=T.score_color(rk.risk_score)),
    ]
    html(T.stat_grid(rcards, min_w=150))

    html("<div style='height:14px'></div>")
    html(T.section("Dividends"))
    dcards = [
        T.stat_card("Dividend yield", "N/A" if dv.dividend_yield is None else f"{dv.dividend_yield:.2%}", term="Dividend yield"),
        T.stat_card("Payout ratio", "N/A" if dv.payout_ratio is None else f"{dv.payout_ratio:.0%}", term="Payout ratio"),
        T.stat_card("DPS CAGR", "N/A" if dv.growth_cagr is None else f"{dv.growth_cagr:.1%}", term="DPS CAGR"),
        T.stat_card("FCF cover", "N/A" if dv.fcf_cover is None else f"{dv.fcf_cover:.2f}×", term="FCF cover"),
    ]
    html(T.stat_grid(dcards, min_w=150))
    if dv.sustainable is not None:
        kind = "good" if dv.sustainable else "warn"
        txt = "Dividend looks sustainable (payout & FCF cover)" if dv.sustainable else "Dividend looks stretched"
        html(f"<div style='margin-top:8px'>{T.pill(txt, kind)}</div>")
    if dv.history:
        rows = [[h["year"], f"{h['dps']:.4f}"] for h in dv.history]
        html("<div style='height:8px'></div>")
        html(C.simple_table_html(["Year", "DPS (SAR)"], rows))


# ------------------------------ Shariah ------------------------------------ #
with tabs[6]:
    sh = r.shariah
    html(T.section("Shariah screen", "Indicative quantitative screen — not a fatwa."))
    if sh.compliant is None:
        html(f"<div style='margin-bottom:8px'>{T.pill('Insufficient data','neutral')}</div>")
    elif sh.compliant:
        html(f"<div style='margin-bottom:8px'>{T.pill('Indicatively COMPLIANT','good')}</div>")
    else:
        html(f"<div style='margin-bottom:8px'>{T.pill('Indicatively NON-COMPLIANT','bad')}</div>")
    html(f"<div style='color:var(--text-dim);font-size:.85rem;margin-bottom:10px'>"
         f"Methodology: <b style='color:var(--text)'>{sh.methodology}</b> · "
         f"Denominator: <b style='color:var(--text)'>{sh.denominator}</b>"
         + (f" · Bundled flag: {sh.sector_flag}" if sh.sector_flag else "") + "</div>")
    rows = [[c.name, "N/A" if c.value is None else f"{c.value:.1%}", f"{c.threshold:.0%}",
             "—" if c.passed is None else ("✅" if c.passed else "❌")] for c in sh.checks]
    html(C.simple_table_html(["Screen", "Value", "Threshold", "Pass"], rows))
    html(f"<div class='disclaimer' style='margin-top:12px'>{T._esc(sh.note)}</div>")


# ---------------------------- Methodology ---------------------------------- #
with tabs[7]:
    html(T.section("Methodology & assumptions",
                   "Every weight, threshold and rubric is loaded verbatim from config/config.yaml."))
    with st.expander("Verdict — weights & rating bands", expanded=True):
        st.json({"weights": cfg.verdict.weights, "rating_bands": cfg.verdict.rating_bands,
                 "three_tier_map": cfg.verdict.three_tier_map,
                 "data_completeness": cfg.verdict.data_completeness})
    with st.expander("Risk sub-score"):
        st.json(cfg.verdict.risk_score)
    with st.expander("Fundamental sub-score — component weights by type"):
        st.json(cfg.fundamental_score)
    with st.expander("Per-metric scoring anchors"):
        st.json({"blend_sector_percentile": cfg.metric_scoring.blend_sector_percentile,
                 "metrics": cfg.metric_scoring.metrics})
    with st.expander("Technical sub-score"):
        st.json({"components": cfg.technical_score.components,
                 "timeframe_weights": cfg.technical_score.timeframe_weights,
                 "signals": cfg.technical_score.signals})
    with st.expander("Trend assessment"):
        st.json({"inputs": cfg.trend.inputs, "classification": cfg.trend.classification,
                 "horizons": cfg.trend.horizons, "confidence": cfg.trend.confidence})
    with st.expander("Indicators & timeframes"):
        st.json({"indicators": {"rsi_period": cfg.indicators.rsi_period,
                                "rsi_overbought": cfg.indicators.rsi_overbought,
                                "rsi_oversold": cfg.indicators.rsi_oversold,
                                "macd": cfg.indicators.macd.model_dump(),
                                "sma_periods": cfg.indicators.sma_periods,
                                "ema_periods": cfg.indicators.ema_periods},
                 "resample": cfg.timeframes.resample.model_dump()})
    with st.expander("Shariah thresholds"):
        st.json({"methodology": cfg.shariah.methodology, "denominator": cfg.shariah.denominator,
                 "thresholds": cfg.shariah.thresholds})
    html("<div style='color:var(--text-faint);font-size:.8rem;margin-top:10px'>"
         "Scoring: linear anchors map a metric between p0→0 and p100→100; 'band' metrics score 100 inside "
         "[low,high] and decay outside; signals emit −1…+1. Missing inputs are dropped and weights re-normalised.</div>")
