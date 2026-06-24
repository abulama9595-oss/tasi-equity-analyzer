"""Design system: global CSS (dark 'terminal' theme), glossary tooltips, and HTML
component helpers (cards, badges, score bars, verdict hero).

All colours are CSS variables in one place (:root below) so the palette can be flipped
(e.g. to a light theme) without touching component code. Components return HTML strings to
be rendered with st.markdown(..., unsafe_allow_html=True).
"""
from __future__ import annotations

import html
import math
import textwrap
from typing import Any

# --------------------------------------------------------------------------- #
# Glossary — every abbreviation/term used in the UI, in plain language.
# --------------------------------------------------------------------------- #
GLOSSARY: dict[str, str] = {
    # valuation
    "P/E": "Price-to-Earnings: share price ÷ earnings per share. Lower can mean cheaper — but compare within the sector.",
    "P/E (trailing)": "Price ÷ last 12 months' earnings per share. Lower can mean cheaper; compare within the sector.",
    "P/E (forward)": "Price ÷ expected next-year earnings per share.",
    "P/B": "Price-to-Book: price relative to net asset (book) value per share. A key metric for banks.",
    "P/S": "Price-to-Sales: price relative to revenue per share.",
    "EV/EBITDA": "Enterprise Value ÷ EBITDA: total company value vs. operating earnings — neutral to how the firm is financed.",
    "PEG": "P/E divided by earnings growth. Roughly ~1 is often seen as fair value for the growth on offer.",
    "EV": "Enterprise Value: market cap + net debt — the cost to buy the whole business.",
    "EBITDA": "Earnings Before Interest, Tax, Depreciation & Amortisation — a proxy for operating cash earnings.",
    # profitability
    "ROE": "Return on Equity: net profit as a % of shareholders' equity. Higher is better.",
    "ROA": "Return on Assets: net profit as a % of total assets.",
    "ROIC": "Return on Invested Capital: profit generated on all capital invested in the business.",
    "Gross margin": "Gross profit as a % of revenue (after cost of goods sold).",
    "Operating margin": "Operating profit as a % of revenue.",
    "Net margin": "Net profit as a % of revenue — the bottom-line cut of each riyal of sales.",
    # growth
    "Revenue growth (YoY)": "Year-over-year change in revenue.",
    "EPS growth (YoY)": "Year-over-year change in earnings per share.",
    "CAGR": "Compound Annual Growth Rate — the smoothed yearly growth rate over a period.",
    # health / cash
    "Debt/Equity": "Total debt relative to shareholders' equity — a leverage gauge.",
    "Current ratio": "Current assets ÷ current liabilities — short-term liquidity (above 1 is healthier).",
    "Quick ratio": "Like the current ratio but excludes inventory — a stricter liquidity test.",
    "Interest coverage": "Operating profit ÷ interest expense — how comfortably the firm services its debt.",
    "Net debt/EBITDA": "Net borrowings relative to operating earnings — leverage; lower is safer.",
    "FCF": "Free Cash Flow: operating cash flow minus capital spending — cash the business actually generates.",
    "FCF yield": "Free Cash Flow as a % of market value.",
    "FCF cover": "Free cash flow ÷ dividends paid — how comfortably dividends are funded by cash (>1 is healthier).",
    # income
    "Dividend yield": "Annual dividend as a % of the share price.",
    "Payout ratio": "Share of earnings paid out as dividends.",
    "DPS": "Dividend Per Share.",
    "DPS CAGR": "Compound annual growth rate of the dividend per share.",
    # banks / reits / insurance
    "NIM": "Net Interest Margin: a bank's lending spread (interest earned minus paid, vs. assets). Key bank metric.",
    "Cost-to-income": "A bank's operating costs as a % of income — lower means more efficient.",
    "NPL ratio": "Non-Performing Loans as a % of total loans — asset quality; lower is better.",
    "CAR": "Capital Adequacy Ratio: a bank's capital vs. risk-weighted assets — its safety buffer.",
    "FFO": "Funds From Operations: a REIT's core recurring cash earnings.",
    "P/FFO": "Price relative to FFO — the REIT equivalent of P/E.",
    "FFO payout": "Share of FFO paid out as distributions.",
    "LTV": "Loan-to-Value: debt relative to asset value — a REIT's leverage.",
    "Combined ratio": "Insurance claims + expenses ÷ premiums. Under 100% means an underwriting profit.",
    # technical
    "RSI": "Relative Strength Index (0–100): a momentum gauge. Above 70 = overbought, below 30 = oversold.",
    "RSI(14)": "Relative Strength Index over 14 periods (0–100). Above 70 = overbought, below 30 = oversold.",
    "MACD": "Moving Average Convergence Divergence: trend & momentum from the gap between two moving averages.",
    "ADX": "Average Directional Index: measures trend strength (not direction). Above ~25 means a real trend.",
    "ATR": "Average True Range: the typical price move per period — a volatility measure.",
    "OBV": "On-Balance Volume: a running total of volume added on up days and subtracted on down days.",
    "Stochastic": "Stochastic oscillator: where price sits within its recent range (0–100) — momentum & turns.",
    "Bollinger Bands": "Volatility bands around a moving average; a 'squeeze' can precede a big move.",
    "SMA": "Simple Moving Average: the average closing price over a window.",
    "EMA": "Exponential Moving Average: a moving average that weights recent prices more heavily.",
    "Golden/death cross": "When the 50-period average crosses above (golden, bullish) or below (death, bearish) the 200.",
    "Support": "A price level where buying has tended to halt declines.",
    "Resistance": "A price level where selling has tended to cap advances.",
    # risk
    "Beta": "Sensitivity to the TASI index: above 1 moves more than the market, below 1 moves less.",
    "Beta vs TASI": "Sensitivity to the TASI index: above 1 moves more than the market, below 1 moves less.",
    "Sharpe": "Return earned per unit of risk (volatility), above the risk-free rate. Higher is better.",
    "Max drawdown": "The largest peak-to-trough fall in the price history — the worst loss endured.",
    "VaR 95%": "Value at Risk: a one-day loss the stock exceeds only about 5% of the time — a 'bad day' estimate.",
    "VaR 95% (1d)": "Value at Risk: a one-day loss the stock exceeds only about 5% of the time.",
    "Ann. volatility": "Annualised volatility: how much the price swings over a year (standard deviation of returns).",
    "Annualized volatility": "How much the price swings over a year (standard deviation of returns).",
    # overview / meta
    "Market cap": "Total market value of all shares (price × shares outstanding).",
    "Free float": "Shares available to trade publicly, excluding locked-in / strategic holders.",
    "Shares out.": "Shares outstanding: the total number of shares issued.",
    "52-week range": "The lowest and highest price over the past year, and where the price sits within it.",
    # scoring
    "Composite": "The blended 0–100 score that drives the verdict (a weighted mix of the sub-scores).",
    "Sub-score": "A 0–100 score for one pillar (fundamentals, technicals, etc.) before blending into the composite.",
    "Conviction": "How confident the verdict is, based on data coverage and how strongly the signals agree.",
    "Data completeness": "Fraction of expected inputs that were actually available — reliability drops when it's low.",
    "Percentile": "Where this metric ranks versus peers (higher = better than more peers).",
    "Provenance": "Which data source supplied a value (e.g. Saudi Exchange vs. Yahoo).",
    "Rubric": "The metric set & weights used — it switches by company type (bank / REIT / insurance / general).",
}

# common aliases so lookups are forgiving
_ALIASES = {
    "Debt/Equity": "Debt/Equity",
    "Net debt/EBITDA": "Net debt/EBITDA",
    "Capital adequacy": "CAR",
    "Net interest margin": "NIM",
    "Loan-to-value": "LTV",
}


def lookup(term: str) -> str | None:
    if term in GLOSSARY:
        return GLOSSARY[term]
    if term in _ALIASES:
        return GLOSSARY.get(_ALIASES[term])
    # case-insensitive fallback
    low = term.strip().lower()
    for k, v in GLOSSARY.items():
        if k.lower() == low:
            return v
    return None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def _missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def gloss(term: str, text: str | None = None, label: str | None = None) -> str:
    """Return a term with a hover tooltip. `label` overrides the visible text."""
    definition = text or lookup(term)
    shown = _esc(label or term)
    if not definition:
        return shown
    return (
        f"<span class='gloss' tabindex='0'>{shown}"
        f"<span class='gloss-pop'>{_esc(definition)}</span></span>"
    )


def score_color(score: float | None) -> str:
    if _missing(score):
        return "var(--text-dim)"
    if score >= 65:
        return "var(--good)"
    if score >= 45:
        return "var(--warn)"
    return "var(--bad)"


def pill(text: str, kind: str = "neutral") -> str:
    return f"<span class='pill pill-{kind}'>{_esc(text)}</span>"


def chip(text: str) -> str:
    return f"<span class='chip'>{_esc(text)}</span>"


def stat_card(label: str, value: str, *, term: str | None = None, sub: str | None = None,
              delta: str | None = None, delta_kind: str = "neutral", accent: str | None = None) -> str:
    label_html = gloss(term, label=label) if term else _esc(label)
    bar = f"<span class='stat-accent' style='background:{accent}'></span>" if accent else ""
    delta_html = f"<span class='stat-delta delta-{delta_kind}'>{_esc(delta)}</span>" if delta else ""
    sub_html = f"<div class='stat-sub'>{sub}</div>" if sub else ""
    return (
        f"<div class='stat-card'>{bar}"
        f"<div class='stat-label'>{label_html}</div>"
        f"<div class='stat-value'>{_esc(value)} {delta_html}</div>"
        f"{sub_html}</div>"
    )


def stat_grid(cards: list[str], min_w: int = 175) -> str:
    inner = "".join(cards)
    return f"<div class='stat-grid' style='--minw:{min_w}px'>{inner}</div>"


def score_bar(name: str, score: float | None, *, term: str | None = None, weight: float | None = None) -> str:
    label = gloss(term, label=name) if term else _esc(name)
    if _missing(score):
        pct, txt, col = 0, "N/A", "var(--text-dim)"
    else:
        pct, txt, col = max(0, min(100, score)), f"{score:.0f}", score_color(score)
    wt = f"<span class='bar-weight'>{weight:.0%}</span>" if weight is not None else ""
    return (
        f"<div class='score-row'>"
        f"<div class='score-head'><span>{label} {wt}</span>"
        f"<span class='score-num' style='color:{col}'>{txt}</span></div>"
        f"<div class='bar-track'><div class='bar-fill' style='width:{pct}%;background:{col}'></div></div>"
        f"</div>"
    )


def section(title: str, subtitle: str | None = None) -> str:
    sub = f"<div class='sec-sub'>{_esc(subtitle)}</div>" if subtitle else ""
    return f"<div class='sec'><div class='sec-title'>{_esc(title)}</div>{sub}</div>"


def verdict_hero(rating_label: str, rating3: str, composite: float, summary: str,
                 conviction: str, completeness: float, low_reliability: bool) -> str:
    kind = rating3  # buy | hold | sell
    flag = (
        "<div class='hero-flag'>⚠ Low data reliability — fundamentals are sparse; treat with extra caution.</div>"
        if low_reliability else ""
    )
    return textwrap.dedent(f"""
    <div class='verdict-hero hero-{kind}'>
      <div class='hero-top'>
        <div>
          <div class='hero-eyebrow'>VERDICT</div>
          <div class='hero-rating'>{_esc(rating_label)}</div>
          <div class='hero-tier'>3-tier signal · <b>{_esc(rating3.upper())}</b></div>
        </div>
        <div class='hero-score'>
          <div class='hero-score-num'>{composite:.0f}</div>
          <div class='hero-score-cap'>composite / 100</div>
        </div>
      </div>
      <div class='hero-summary'>{_esc(summary)}</div>
      <div class='hero-meta'>
        <span>{gloss('Conviction', label='Conviction')}: <b>{_esc(conviction.title())}</b></span>
        <span>{gloss('Data completeness', label='Data completeness')}: <b>{completeness:.0%}</b></span>
      </div>
      {flag}
    </div>
    """).strip()


def reason_list(items: list[str], kind: str) -> str:
    icon = "▲" if kind == "bull" else "▼"
    head = "What supports it" if kind == "bull" else "What holds it back"
    lis = "".join(f"<li>{_esc(x)}</li>" for x in items)
    return f"<div class='reason reason-{kind}'><div class='reason-head'>{icon} {head}</div><ul>{lis}</ul></div>"


# --------------------------------------------------------------------------- #
# Global CSS
# --------------------------------------------------------------------------- #
def inject_css() -> str:
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

:root{
  --bg:#0b0f14; --surface:#131a22; --surface-2:#0f151b; --surface-3:#1a232e;
  --border:#222c38; --border-2:#2c3947;
  --text:#e6edf3; --text-dim:#8b97a7; --text-faint:#5b6675;
  --accent:#10b981; --accent-2:#22d3ee;
  --good:#22c55e; --warn:#f59e0b; --bad:#ef4444;
  --radius:14px; --radius-sm:10px;
  --shadow:0 10px 34px rgba(0,0,0,.40);
}

html, body, [class*="css"], .stApp, [data-testid="stMarkdownContainer"]{
  font-family:'Inter',-apple-system,Segoe UI,Roboto,sans-serif;
}
.stApp{ background:
  radial-gradient(1200px 600px at 85% -10%, rgba(16,185,129,.07), transparent 60%),
  radial-gradient(900px 500px at -10% 0%, rgba(34,211,238,.05), transparent 55%),
  var(--bg); }

/* hide default chrome for an app-like feel (keep sidebar toggle) */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{ visibility:hidden; height:0; }
[data-testid="stHeader"]{ background:transparent; }

.block-container{ padding-top:2.2rem; padding-bottom:4rem; max-width:1180px; }
/* let tooltips escape their containers */
[data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"],
[data-testid="column"], [data-testid="stMarkdownContainer"]{ overflow:visible !important; }

h1,h2,h3{ letter-spacing:-.02em; }

/* ---------- hero brand header ---------- */
.brand{ display:flex; align-items:center; gap:14px; margin:0 0 2px; }
.brand-mark{ width:42px;height:42px;border-radius:12px;display:grid;place-items:center;
  background:linear-gradient(135deg,var(--accent),var(--accent-2)); color:#04110c; font-size:22px;
  box-shadow:0 6px 20px rgba(16,185,129,.35); }
.brand-title{ font-size:1.7rem; font-weight:800; line-height:1; }
.brand-sub{ color:var(--text-dim); font-size:.86rem; margin-top:3px; }

/* ---------- disclaimer ---------- */
.disclaimer{ border:1px solid var(--border); background:linear-gradient(180deg,rgba(245,158,11,.07),rgba(245,158,11,.02));
  border-left:3px solid var(--warn); border-radius:var(--radius-sm); padding:10px 14px; color:var(--text-dim);
  font-size:.82rem; margin:14px 0 6px; }
.disclaimer b{ color:var(--text); }

/* ---------- cards & sections ---------- */
.tasi-card{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius);
  padding:18px 20px; box-shadow:var(--shadow); }
.sec{ margin:6px 0 12px; }
.sec-title{ font-size:1.12rem; font-weight:700; }
.sec-sub{ color:var(--text-dim); font-size:.84rem; margin-top:2px; }

/* ---------- stat cards ---------- */
.stat-grid{ display:grid; grid-template-columns:repeat(auto-fit, minmax(var(--minw,175px),1fr)); gap:12px; margin:6px 0 4px; }
.stat-card{ position:relative; background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-sm);
  padding:14px 16px; overflow:visible; transition:transform .12s ease, border-color .12s ease; }
.stat-card:hover{ transform:translateY(-2px); border-color:var(--border-2); }
.stat-accent{ position:absolute; left:0; top:0; bottom:0; width:3px; border-radius:var(--radius-sm) 0 0 var(--radius-sm); }
.stat-label{ color:var(--text-dim); font-size:.74rem; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
.stat-value{ font-size:1.5rem; font-weight:700; margin-top:4px; font-feature-settings:"tnum"; }
.stat-sub{ margin-top:6px; font-size:.74rem; color:var(--text-faint); }
.stat-delta{ font-size:.8rem; font-weight:700; margin-left:6px; }
.delta-good{ color:var(--good);} .delta-bad{ color:var(--bad);} .delta-neutral{ color:var(--text-dim);}

/* ---------- pills / chips ---------- */
.pill{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:.74rem; font-weight:600; border:1px solid transparent; }
.pill-neutral{ background:var(--surface-3); color:var(--text-dim); border-color:var(--border); }
.pill-good{ background:rgba(34,197,94,.14); color:#5ee08a; border-color:rgba(34,197,94,.3); }
.pill-warn{ background:rgba(245,158,11,.14); color:#fbbf57; border-color:rgba(245,158,11,.3); }
.pill-bad{ background:rgba(239,68,68,.14); color:#f5837e; border-color:rgba(239,68,68,.3); }
.pill-accent{ background:rgba(16,185,129,.14); color:#56e0b4; border-color:rgba(16,185,129,.3); }
.chip{ display:inline-block; padding:1px 8px; border-radius:6px; font-size:.68rem; font-weight:600;
  background:var(--surface-3); color:var(--text-dim); border:1px solid var(--border); }

/* ---------- glossary tooltip ---------- */
.gloss{ border-bottom:1px dotted var(--text-faint); cursor:help; position:relative; }
.gloss-pop{ visibility:hidden; opacity:0; position:absolute; bottom:150%; left:0; transform:translateY(4px);
  background:#0a0e13; color:var(--text); border:1px solid var(--border-2); padding:9px 11px; border-radius:10px;
  width:260px; max-width:78vw; font-size:.76rem; font-weight:400; line-height:1.4; z-index:99999;
  box-shadow:0 12px 34px rgba(0,0,0,.55); transition:opacity .14s ease, transform .14s ease; text-transform:none; letter-spacing:0; pointer-events:none; }
.gloss-pop::after{ content:''; position:absolute; top:100%; left:18px;
  border:6px solid transparent; border-top-color:var(--border-2); }
.gloss:hover .gloss-pop, .gloss:focus .gloss-pop{ visibility:visible; opacity:1; transform:translateY(0); }

/* ---------- score bars ---------- */
.score-row{ margin:9px 0; }
.score-head{ display:flex; justify-content:space-between; font-size:.84rem; margin-bottom:5px; }
.score-num{ font-weight:700; font-feature-settings:"tnum"; }
.bar-weight{ color:var(--text-faint); font-size:.72rem; margin-left:6px; }
.bar-track{ background:var(--surface-3); border-radius:999px; height:8px; overflow:hidden; }
.bar-fill{ height:8px; border-radius:999px; transition:width .5s cubic-bezier(.2,.8,.2,1); }

/* ---------- verdict hero ---------- */
.verdict-hero{ position:relative; border-radius:var(--radius); padding:22px 24px; overflow:hidden;
  border:1px solid var(--border); box-shadow:var(--shadow); }
.hero-buy{ background:linear-gradient(135deg,rgba(34,197,94,.18),rgba(16,185,129,.05)); border-color:rgba(34,197,94,.35); }
.hero-hold{ background:linear-gradient(135deg,rgba(245,158,11,.16),rgba(245,158,11,.03)); border-color:rgba(245,158,11,.32); }
.hero-sell{ background:linear-gradient(135deg,rgba(239,68,68,.16),rgba(239,68,68,.03)); border-color:rgba(239,68,68,.32); }
.hero-top{ display:flex; justify-content:space-between; align-items:flex-start; gap:16px; }
.hero-eyebrow{ font-size:.72rem; letter-spacing:.18em; color:var(--text-dim); font-weight:700; }
.hero-rating{ font-size:2.1rem; font-weight:800; line-height:1.05; margin-top:2px; }
.hero-tier{ color:var(--text-dim); font-size:.86rem; margin-top:2px; }
.hero-score{ text-align:right; }
.hero-score-num{ font-size:2.6rem; font-weight:800; font-feature-settings:"tnum"; line-height:1; }
.hero-score-cap{ font-size:.72rem; color:var(--text-dim); }
.hero-summary{ margin:16px 0 12px; font-size:1.02rem; line-height:1.55; color:var(--text); max-width:760px; }
.hero-meta{ display:flex; gap:22px; font-size:.84rem; color:var(--text-dim); flex-wrap:wrap; }
.hero-meta b{ color:var(--text); }
.hero-flag{ margin-top:14px; background:rgba(239,68,68,.12); border:1px solid rgba(239,68,68,.35);
  color:#f5a29e; padding:8px 12px; border-radius:10px; font-size:.82rem; }

/* ---------- reasons ---------- */
.reasons-wrap{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:6px; }
@media (max-width:760px){ .reasons-wrap{ grid-template-columns:1fr; } }
.reason{ background:var(--surface); border:1px solid var(--border); border-radius:var(--radius-sm); padding:14px 16px; }
.reason-head{ font-weight:700; font-size:.9rem; margin-bottom:8px; }
.reason-bull .reason-head{ color:var(--good); }
.reason-bear .reason-head{ color:var(--bad); }
.reason ul{ margin:0; padding-left:18px; }
.reason li{ margin:5px 0; font-size:.88rem; color:var(--text); line-height:1.45; }

/* ---------- HTML tables ---------- */
.tasi-table{ width:100%; border-collapse:collapse; font-size:.86rem; }
.tasi-table th{ text-align:left; color:var(--text-dim); font-weight:600; font-size:.74rem; text-transform:uppercase;
  letter-spacing:.03em; padding:8px 10px; border-bottom:1px solid var(--border); white-space:nowrap; }
.tasi-table td{ padding:9px 10px; border-bottom:1px solid var(--border); }
.tasi-table tr:hover td{ background:rgba(255,255,255,.02); }
.tasi-table .num{ text-align:right; font-feature-settings:"tnum"; }
.t-pos{ color:var(--good);} .t-neg{ color:var(--bad);} .t-dim{ color:var(--text-faint);}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"]{ gap:4px; border-bottom:1px solid var(--border); }
.stTabs [data-baseweb="tab"]{ background:transparent; border-radius:8px 8px 0 0; padding:8px 14px; color:var(--text-dim); font-weight:600; }
.stTabs [aria-selected="true"]{ color:var(--text); background:var(--surface); }

/* ---------- inputs / buttons ---------- */
.stTextInput input{ background:var(--surface); border:1px solid var(--border); border-radius:10px; color:var(--text); }
.stTextInput input:focus{ border-color:var(--accent); box-shadow:0 0 0 2px rgba(16,185,129,.18); }
.stButton button{ border-radius:10px; font-weight:700; border:1px solid transparent; }
div[data-testid="stProgress"] > div > div > div{ background:linear-gradient(90deg,var(--accent),var(--accent-2)); }

/* dataframe polish */
[data-testid="stDataFrame"]{ border:1px solid var(--border); border-radius:var(--radius-sm); }

/* range bar */
.range-wrap{ margin:4px 0 2px; }
.range-track{ position:relative; height:10px; border-radius:999px; background:linear-gradient(90deg,#1d2a36,#26424a); border:1px solid var(--border); }
.range-dot{ position:absolute; top:50%; transform:translate(-50%,-50%); width:14px; height:14px; border-radius:50%;
  background:var(--accent); border:2px solid #04110c; box-shadow:0 0 0 3px rgba(16,185,129,.25); }
.range-labels{ display:flex; justify-content:space-between; font-size:.76rem; color:var(--text-dim); margin-top:6px; }
</style>
"""
