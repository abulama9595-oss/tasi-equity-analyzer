"""TASI Equity Analyzer — Streamlit entry point.

Single-ticker analysis for the Saudi market (Tadawul / TASI): overview, a clear auditable
Buy/Hold/Sell verdict, a probabilistic trend assessment, multi-timeframe technicals,
type-aware fundamentals, and risk & income. Dark "terminal" theme; every abbreviation has
a hover tooltip. (Shariah and methodology are still computed and included in the exported
report; the indicative Shariah screen and config remain in analysis/ and config/.)

NOT financial advice. For personal research only.
"""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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


_CONV_RANK = {"high": 2, "medium": 1, "low": 0}


@st.cache_data(show_spinner=False, ttl=21600)  # 6h
def run_market_scan(max_n: int) -> "pd.DataFrame":
    """Score the first `max_n` names of the universe (large-caps first) and return a ranking
    table. Heavy on a cold cache (one full analysis per name) — gated behind a button + cached."""
    import pandas as pd
    cfg_, registry_, composite_, sahmk_ = get_resources()
    rows = []
    for ref in registry_.all_refs()[:max_n]:
        try:
            r = az.analyze(ref.code, cfg_, registry_, composite_, sahmk_)
        except Exception:
            continue
        if r.error or r.verdict is None:
            continue
        v, ov = r.verdict, r.overview
        rows.append({
            "code": ref.code,
            "name": _v(ov.get("name_en")) or ref.name_en or ref.code,
            "sector": _v(ov.get("sector")) or ref.sector or "",
            "price": _v(ov.get("price")),
            "composite": v.composite,
            "rating5": v.rating5_label,
            "rating3": v.rating3,
            "conviction": v.conviction,
            "conv_rank": _CONV_RANK.get(v.conviction, 0),
            "completeness": v.data_completeness,
            "low_reliability": v.low_reliability,
        })
    return pd.DataFrame(rows)


def _scan_table(df, side: str) -> str:
    kind = {"buy": "good", "hold": "warn", "sell": "bad"}
    rows = []
    for i, (_, x) in enumerate(df.iterrows(), 1):
        comp = x["composite"]
        flag = " ⚠" if x["low_reliability"] else ""
        rows.append([
            i,
            f"<b>{x['code']}</b>",
            str(x["name"])[:24],
            f"<span style='color:{T.score_color(comp)};font-weight:700'>{comp:.0f}</span>",
            T.pill(x["rating5"], kind.get(x["rating3"], "neutral")),
            f"{str(x['conviction']).title()}{flag}",
        ])
    return C.simple_table_html(["#", "Ticker", "Name", "Score", "Rating", "Conviction"], rows)


cfg, registry, composite, sahmk = get_resources()

# ---------- header ---------- #
html(
    "<div class='brand'><div class='brand-mark'>📈</div>"
    "<div><div class='brand-title'>TASI Equity Analyzer</div>"
    "<div class='brand-sub'>Saudi market (Tadawul) · single-ticker research · all values in SAR</div>"
    "</div></div>"
)
html(
    "<div class='disclaimer'><b>Not financial advice — a research screen, not a tested signal.</b> "
    "Scores are transparent and auditable but have no validated track record, and the underlying "
    "fundamentals (P/E, growth, margins) can be inaccurate — verify them before acting. "
    "Quotes and fundamentals may be delayed.</div>"
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
tab_overview, tab_verdict, tab_trend, tab_technical, tab_fund, tab_risk, tab_scan = st.tabs(
    ["Overview", "🟢 Verdict", "Trend", "Technical", "Fundamentals", "Risk & Income", "🔎 Market scan"])
ov = r.overview


# ----------------------------- Overview ------------------------------------ #
with tab_overview:
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
with tab_fund:
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
with tab_technical:
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
    components.html(C.lightweight_chart_html(tfd, cfg), height=450)
    st.caption(f"{r.ticker} — {tf_choice} · interactive chart (TradingView Lightweight Charts) · "
               "candles + SMA 20/50/200 + Bollinger + volume.")
    st.plotly_chart(C.indicator_panes_figure(tfd, cfg), width="stretch", config={"displayModeBar": False})
    html("<div style='color:var(--text-faint);font-size:.78rem;margin-top:2px'>✓ Indicator math "
         "validated against <b>TA-Lib</b> (industry standard) and Wilder's reference values "
         "(see tests/test_talib_parity.py, tests/test_indicators.py).</div>")

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
with tab_trend:
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
with tab_verdict:
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
with tab_risk:
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


# --------------------------- Market scan ----------------------------------- #
with tab_scan:
    html(T.section("Market scan",
                   "Ranks the bundled TASI universe by the verdict composite (higher = more "
                   "Buy-like). Conviction is shown; ⚠ flags low data reliability."))
    html("<div class='disclaimer'>This is a <b>screen, not a recommendation.</b> The audit/backtest "
         "found the verdict has <b>no validated predictive edge</b>, so use this only to surface "
         "candidates for your own research — not to decide trades. Verify before acting.</div>")

    universe_n = len(registry.all_refs())
    c_in, c_btn = st.columns([2, 1])
    with c_in:
        max_n = st.slider("Names to scan (universe is large-caps first)", min_value=10,
                          max_value=universe_n, value=min(60, universe_n), step=10)
    with c_btn:
        st.write("")
        if st.button("Run / refresh scan", type="primary", width="stretch"):
            run_market_scan.clear()
            st.session_state["scan_active_n"] = int(max_n)

    if "scan_active_n" in st.session_state:
        n = st.session_state["scan_active_n"]
        with st.spinner(f"Scanning {n} names… (~{max(1, n*2//60)}–{n*4//60+1} min on a cold cache; cached 6h)"):
            scan = run_market_scan(n)
        if scan is None or scan.empty:
            st.warning("Scan returned no results (data unavailable).")
        else:
            st.caption(f"Scanned {len(scan)} names · {pd.Timestamp.now():%Y-%m-%d %H:%M}")
            buys = scan.sort_values(["composite", "conv_rank"], ascending=[False, False]).head(10)
            sells = scan.sort_values(["composite", "conv_rank"], ascending=[True, False]).head(10)
            cL, cR = st.columns(2)
            with cL:
                html(T.section("Top 10 — Buy candidates", "Highest composite"))
                html(_scan_table(buys, "buy"))
            with cR:
                html(T.section("Top 10 — Sell candidates", "Lowest composite"))
                html(_scan_table(sells, "sell"))
            st.caption("Ranked by composite score (Buy: highest; Sell: lowest), conviction as tiebreak. "
                       "Not financial advice.")
    else:
        html(f"<div style='color:var(--text-dim);font-size:.9rem'>Pick how many names to scan "
             f"(the universe has {universe_n}, large-caps first) and click <b>Run / refresh scan</b>. "
             f"A cold run is ~1 min per ~30 names; results are cached for 6 hours. Scanning the full "
             f"{universe_n} can take 10–15 min on a cold cache.</div>")
