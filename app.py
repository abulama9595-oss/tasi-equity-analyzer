"""TASI Equity Analyzer — Streamlit entry point.

Single-ticker analysis for the Saudi market (Tadawul / TASI): company overview, type-aware
fundamentals, multi-timeframe technicals, a probabilistic trend assessment, and a fully
auditable Buy/Hold/Sell verdict — plus risk, dividends, an indicative Shariah screen, and a
methodology page that renders config.yaml verbatim.

NOT financial advice. For personal research only.
"""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st

import analyzer as az
from config.settings import get_config
from data.ticker_registry import TickerRegistry
from ui import components as C

st.set_page_config(
    page_title="TASI Equity Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------------------------------- #
# Cached singletons & analysis
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


# --------------------------------------------------------------------------- #
# Header + search
# --------------------------------------------------------------------------- #
cfg, registry, composite, sahmk = get_resources()

st.title("📈 TASI Equity Analyzer")
st.caption("Single-ticker analysis for the Saudi market (Tadawul). Currency: SAR.")

st.warning(
    "**Not financial advice — for personal research only.** Scores are transparent and "
    "auditable; verify every input before acting. Quotes and fundamentals may be delayed "
    "or incomplete.",
    icon="⚠️",
)

with st.sidebar:
    st.header("Settings")
    st.write(f"**Saudi Exchange (SAHMK):** {'connected' if sahmk.available else 'off (yfinance-only)'}")
    st.write(f"**Risk-free (Sharpe):** {cfg.risk_free.annual_rate:.1%}")
    if st.button("Clear data cache"):
        n = az.DiskCache(cfg.cache_dir(), cfg.cache.ttl_seconds).clear()
        run_analysis.clear()
        st.success(f"Cleared {n} cached files.")
    st.caption("Add a SAHMK_API_KEY to .env to enrich Saudi-Exchange fields.")

col_in, col_btn = st.columns([4, 1])
with col_in:
    query = st.text_input(
        "Enter a TASI ticker or company name",
        value=st.session_state.get("ticker", "1120"),
        placeholder="e.g. 1120, 2222.SR, Aramco, الراجحي",
        label_visibility="collapsed",
    )
with col_btn:
    go = st.button("Analyze", type="primary", width="stretch")

# live suggestions
if query and not query.strip().isdigit():
    sugg = registry.search(query, limit=6)
    if sugg:
        st.caption("Suggestions: " + " · ".join(f"`{s.code}` {s.name_en}" for s in sugg))

examples = "Examples: 1120 (Al Rajhi · bank), 2222 (Aramco), 7010 (stc), 8210 (Bupa · insurance), 4330 (Riyad REIT)"
st.caption(examples)

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
# Tabs
# --------------------------------------------------------------------------- #
tabs = st.tabs(
    ["Overview", "Fundamentals", "Technical", "Trend", "🟢 Verdict",
     "Risk & Income", "Shariah", "Methodology"]
)


# ---------------------------- Overview ------------------------------------- #
with tabs[0]:
    ov = r.overview
    name_en = _v(ov.get("name_en")) or r.ref.name_en or r.ticker
    name_ar = _v(ov.get("name_ar"))
    head = f"### {name_en}  ·  `{r.ticker}`"
    if name_ar:
        head += f"  ·  {name_ar}"
    st.markdown(head)
    st.caption(
        f"Sector: **{_v(ov.get('sector')) or 'N/A'}** {C.provenance_chip(C.source_of(ov.get('sector')))} "
        f"· Industry: {_v(ov.get('industry')) or 'N/A'} · Detected type: **{r.company_type}**"
    )

    c1, c2, c3, c4 = st.columns(4)
    price = _v(ov.get("price"))
    delayed = getattr(ov.get("price"), "delayed", False)
    c1.metric(
        f"Price ({r.currency})",
        "N/A" if _missing(price) else f"{price:,.2f}",
        None if r.day_change_pct is None else f"{r.day_change_pct:+.2f}%",
    )
    c1.caption(("⏱ delayed · " if delayed else "") + C.provenance_chip(r.price_provenance))
    c2.metric("Market cap", C.big_number(_v(ov.get("market_cap")), " " + r.currency))
    c2.caption(C.provenance_chip(C.source_of(ov.get("market_cap"))))
    c3.metric("Shares out.", C.big_number(_v(ov.get("shares_outstanding"))))
    c3.caption(C.provenance_chip(C.source_of(ov.get("shares_outstanding"))))
    c4.metric("Free float", C.big_number(_v(ov.get("free_float"))))
    c4.caption(C.provenance_chip(C.source_of(ov.get("free_float"))))

    if r.range_52w:
        rng = r.range_52w
        st.markdown(f"**52-week range** ({r.currency})")
        st.progress(int(rng["position_pct"]))
        st.caption(f"Low {rng['low']:.2f} — Price {rng['price']:.2f} ({rng['position_pct']}% of range) — High {rng['high']:.2f}")

    desc = _v(ov.get("description"))
    if desc:
        with st.expander("Business description"):
            st.write(desc)

    with st.expander("Data provenance (per field)"):
        prov_df = pd.DataFrame(
            [{"field": k, "source": C.SOURCE_LABEL.get(v, v)} for k, v in r.provenance.items()]
        )
        st.dataframe(prov_df, width="stretch", hide_index=True)


# --------------------------- Fundamentals ---------------------------------- #
with tabs[1]:
    f = r.fundamentals
    st.subheader(f"Fundamentals — {f.rubric.title()} rubric")
    cc1, cc2 = st.columns([1, 2])
    with cc1:
        st.metric("Fundamental sub-score", "N/A" if _missing(f.subscore) else f"{f.subscore:.0f}/100")
        st.caption(f"Data completeness: {f.data_completeness:.0%}")
        st.markdown("**Components**")
        html = "".join(C.subscore_bar(c["name"].replace('_', ' ').title(), c["score"]) for c in f.components)
        st.markdown(html, unsafe_allow_html=True)
    with cc2:
        rows = []
        for m in f.metrics:
            rows.append({
                "Metric": m["label"],
                "Value": m["display"],
                "Source": C.SOURCE_LABEL.get(m["source"], m["source"] or "-"),
                # keep these columns string-typed so Streamlit's Arrow conversion
                # doesn't choke on number/placeholder mixes
                "Sector median": "-" if m["sector_median"] is None else f"{m['sector_median']:.2f}",
                "TASI": "-" if m["tasi_value"] is None else f"{m['tasi_value']:.2f}",
                "Pctile": "-" if m["percentile"] is None else f"{m['percentile']:.0f}",
                "Score": "-" if m["metric_score"] is None else f"{m['metric_score']:.0f}",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("Metric score = 0.65 × anchor + 0.35 × sector percentile (anchor-only when no peer set).")

    if f.history:
        st.markdown("**Multi-year history**")
        hdf = pd.DataFrame(f.history)
        st.dataframe(hdf, width="stretch", hide_index=True)


# ---------------------------- Technical ------------------------------------ #
with tabs[2]:
    t = r.technical
    st.subheader("Technical analysis (multi-timeframe)")
    st.metric("Technical sub-score", "N/A" if _missing(t.subscore) else f"{t.subscore:.0f}/100")
    cols = st.columns(3)
    for i, (name, sc) in enumerate(t.timeframe_scores.items()):
        cols[i % 3].metric(f"{name.title()} score", "N/A" if sc is None else f"{sc:.0f}")

    avail = [tf for tf in ("monthly", "weekly", "daily") if tf in t.by_timeframe]
    tf_choice = st.radio("Timeframe", avail, horizontal=True, index=0)
    tfd = t.by_timeframe[tf_choice]
    st.plotly_chart(C.candlestick_figure(tfd, cfg, title=f"{r.ticker} — {tf_choice}"),
                    width="stretch")

    lat = tfd.latest
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RSI(14)", "N/A" if _missing(lat["rsi"]) else f"{lat['rsi']:.1f}", lat["rsi_state"])
    m2.metric("MACD", "N/A" if _missing(lat["macd"]) else f"{lat['macd']:.3f}", lat["macd_cross"])
    m3.metric("ADX", "N/A" if _missing(lat["adx"]) else f"{lat['adx']:.1f}")
    m4.metric("ATR", "N/A" if _missing(lat["atr"]) else f"{lat['atr']:.2f}")

    cL, cR = st.columns(2)
    with cL:
        st.markdown("**Signals** (each −1…+1)")
        from analysis.technicals import signal_rows
        st.dataframe(pd.DataFrame(signal_rows(tfd)), width="stretch", hide_index=True)
    with cR:
        st.markdown("**Levels**")
        st.write(f"Support: {', '.join(f'{x:.2f}' for x in tfd.support) or '—'}")
        st.write(f"Resistance: {', '.join(f'{x:.2f}' for x in tfd.resistance) or '—'}")
        if tfd.divergence:
            st.info(f"Divergence: {tfd.divergence}")


# ------------------------------ Trend -------------------------------------- #
with tabs[3]:
    tr = r.trend
    st.subheader("Trend assessment")
    st.caption("Probabilistic classification — **not** a price forecast.")
    cols = st.columns(2)
    for col, call in zip(cols, (tr.short_term, tr.medium_term)):
        if call is None:
            continue
        with col:
            st.markdown(f"#### {call.horizon}")
            st.markdown(f"### {call.classification}")
            st.caption(f"Composite trend score: {call.composite_score} (range −100…+100)")
            st.write(f"Confidence: **{call.confidence}** ({call.confidence_pct:.0%} of inputs agree)")
            st.markdown("**Why:**")
            for reason in call.reasons:
                st.write(f"- {reason}")
            if call.levels_to_watch:
                st.write("**Levels to watch:** " + ", ".join(f"{x:.2f}" for x in call.levels_to_watch))
            if call.invalidation:
                st.warning(call.invalidation)


# ----------------------------- Verdict ------------------------------------- #
with tabs[4]:
    v = r.verdict
    st.subheader("Verdict — Buy / Hold / Sell")
    color = C.RATING5_COLOR.get(v.rating5, "#57606a")
    st.markdown(
        f"<div style='padding:14px;border-radius:10px;background:{color};color:white;"
        f"font-size:1.4rem;font-weight:700;text-align:center;'>{v.rating5_label} "
        f"&nbsp;·&nbsp; 3-tier: {v.rating3.upper()}</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Composite", f"{v.composite:.1f}/100")
    c2.metric("Conviction", v.conviction.title())
    c3.metric("Data completeness", f"{v.data_completeness:.0%}")
    if v.low_reliability:
        st.error("⚠️ Low data reliability — fundamentals are sparse; treat this verdict with extra caution.")

    st.markdown("**Auditable score breakdown** (reproducible by hand)")
    bdf = pd.DataFrame(v.breakdown)
    bdf = bdf.rename(columns={
        "input": "Input", "value": "Value (0-100)", "weight": "Weight",
        "weight_used": "Weight used", "contribution": "Contribution", "running_composite": "Running",
    })
    st.dataframe(bdf, width="stretch", hide_index=True)
    st.caption("Contribution = Value × Weight used. Missing inputs are dropped and weights re-normalised.")

    bull_col, bear_col = st.columns(2)
    with bull_col:
        st.markdown("**🟢 Bull case**")
        for b in v.bull:
            st.write(f"- {b}")
    with bear_col:
        st.markdown("**🔴 Bear case / risks**")
        for b in v.bear:
            st.write(f"- {b}")

    st.divider()
    from reports.export import build_html
    st.download_button(
        "⬇ Download full HTML report",
        data=build_html(r, cfg),
        file_name=f"{r.ref.code or r.ticker}_tasi_report.html",
        mime="text/html",
    )
    for d in r.disclaimers:
        st.caption(f"• {d}")


# -------------------------- Risk & Income ---------------------------------- #
with tabs[5]:
    rk, dv = r.risk, r.dividends
    st.subheader("Risk metrics")
    a, b, cc, d, e = st.columns(5)
    a.metric("Beta vs TASI", "N/A" if rk.beta_vs_tasi is None else f"{rk.beta_vs_tasi:.2f}")
    b.metric("Ann. volatility", "N/A" if rk.annualized_vol is None else f"{rk.annualized_vol:.1%}")
    cc.metric("Max drawdown", "N/A" if rk.max_drawdown is None else f"{rk.max_drawdown:.1%}")
    d.metric("Sharpe", "N/A" if rk.sharpe is None else f"{rk.sharpe:.2f}")
    e.metric("VaR 95% (1d)", "N/A" if rk.var_95 is None else f"{rk.var_95:.1%}")
    st.caption(f"Risk sub-score (higher = safer): **{rk.risk_score}** / 100")

    st.divider()
    st.subheader("Dividends")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Dividend yield", "N/A" if dv.dividend_yield is None else f"{dv.dividend_yield:.2%}")
    d2.metric("Payout ratio", "N/A" if dv.payout_ratio is None else f"{dv.payout_ratio:.0%}")
    d3.metric("DPS CAGR", "N/A" if dv.growth_cagr is None else f"{dv.growth_cagr:.1%}")
    d4.metric("FCF cover", "N/A" if dv.fcf_cover is None else f"{dv.fcf_cover:.2f}×")
    if dv.sustainable is not None:
        st.write(f"Sustainability (payout & FCF cover): {'✅ looks sustainable' if dv.sustainable else '⚠️ stretched'}")
    if dv.history:
        st.dataframe(pd.DataFrame(dv.history), width="stretch", hide_index=True)


# ------------------------------ Shariah ------------------------------------ #
with tabs[6]:
    sh = r.shariah
    st.subheader("Shariah screen (indicative)")
    if sh.compliant is None:
        st.info("Insufficient data to run the quantitative screen.")
    elif sh.compliant:
        st.success("Indicatively COMPLIANT on available quantitative screens.")
    else:
        st.error("Indicatively NON-COMPLIANT on one or more quantitative screens.")
    st.caption(f"Methodology: **{sh.methodology}** · Denominator: **{sh.denominator}**"
               + (f" · Bundled sector flag: {sh.sector_flag}" if sh.sector_flag else ""))
    rows = [{
        "Screen": c.name,
        "Value": "N/A" if c.value is None else f"{c.value:.1%}",
        "Threshold": f"{c.threshold:.0%}",
        "Pass": "—" if c.passed is None else ("✅" if c.passed else "❌"),
    } for c in sh.checks]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.warning(sh.note)


# ---------------------------- Methodology ---------------------------------- #
with tabs[7]:
    st.subheader("Methodology & assumptions")
    st.caption("Every weight, threshold, and rubric below is loaded verbatim from config/config.yaml — "
               "edit that file to re-tune the methodology without touching code.")

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
        st.json({"indicators": {
            "rsi_period": cfg.indicators.rsi_period,
            "rsi_overbought": cfg.indicators.rsi_overbought,
            "rsi_oversold": cfg.indicators.rsi_oversold,
            "macd": cfg.indicators.macd.model_dump(),
            "sma_periods": cfg.indicators.sma_periods,
            "ema_periods": cfg.indicators.ema_periods,
        }, "resample": cfg.timeframes.resample.model_dump()})
    with st.expander("Shariah thresholds"):
        st.json({"methodology": cfg.shariah.methodology, "denominator": cfg.shariah.denominator,
                 "thresholds": cfg.shariah.thresholds})

    st.divider()
    st.caption("Scoring semantics: linear anchors map a metric between p0→0 and p100→100; "
               "'band' metrics score 100 inside [low,high] and decay outside; signals emit −1…+1. "
               "Missing inputs are dropped and weights re-normalised over what remains.")
