"""Report export — self-contained HTML (and best-effort PDF).

HTML is the primary, always-available format (one file, Plotly via CDN). PDF is best-effort:
it tries WeasyPrint if installed and degrades to "HTML only" otherwise, per the build spec
(PDF tooling on Windows is painful, so it is not a hard dependency).
"""
from __future__ import annotations

import html
from typing import Any

from ui import components as C

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px;color:#1f2328;}
h1{margin:0 0 4px;} h2{margin-top:28px;border-bottom:1px solid #d0d7de;padding-bottom:4px;}
.muted{color:#57606a;} table{border-collapse:collapse;width:100%;margin:8px 0;}
th,td{border:1px solid #d0d7de;padding:6px 8px;text-align:left;font-size:0.9rem;}
th{background:#f6f8fa;} .badge{display:inline-block;padding:6px 14px;border-radius:8px;
color:#fff;font-weight:700;} .warn{background:#fff8c5;border:1px solid #d4a72c;padding:8px;border-radius:6px;}
.cols{display:flex;gap:24px;flex-wrap:wrap;} .col{flex:1;min-width:260px;}
"""


def _v(s):
    return getattr(s, "value", s) if s is not None else None


def _esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_html(r, cfg) -> str:
    ov = r.overview
    name = _v(ov.get("name_en")) or r.ticker
    v = r.verdict
    color = C.RATING5_COLOR.get(v.rating5, "#57606a")

    parts: list[str] = [f"<h1>{_esc(name)} — {_esc(r.ticker)}</h1>"]
    parts.append(f"<p class='muted'>As of {_esc(r.as_of)} · {_esc(r.currency)} · "
                 f"Sector {_esc(_v(ov.get('sector')))} · Type {_esc(r.company_type)}</p>")
    parts.append("<p class='warn'><b>Not financial advice — personal research only.</b></p>")

    # Verdict
    parts.append("<h2>Verdict</h2>")
    parts.append(f"<p><span class='badge' style='background:{color}'>{_esc(v.rating5_label)} "
                 f"· 3-tier {v.rating3.upper()}</span> &nbsp; Composite "
                 f"<b>{v.composite:.1f}/100</b> · Conviction {v.conviction} · "
                 f"Data completeness {v.data_completeness:.0%}"
                 + (" · <b>LOW RELIABILITY</b>" if v.low_reliability else "") + "</p>")
    parts.append(_table(
        ["Input", "Value", "Weight", "Weight used", "Contribution", "Running"],
        [[b["input"], b["value"], b["weight"], b["weight_used"], b["contribution"], b["running_composite"]]
         for b in v.breakdown],
    ))
    parts.append("<div class='cols'>")
    parts.append("<div class='col'><b>Bull</b><ul>" + "".join(f"<li>{_esc(x)}</li>" for x in v.bull) + "</ul></div>")
    parts.append("<div class='col'><b>Bear</b><ul>" + "".join(f"<li>{_esc(x)}</li>" for x in v.bear) + "</ul></div>")
    parts.append("</div>")

    # Overview
    parts.append("<h2>Overview</h2>")
    price = _v(ov.get("price"))
    parts.append(_table(
        ["Field", "Value", "Source"],
        [
            ["Price", "N/A" if price is None else f"{price:,.2f}", C.SOURCE_LABEL.get(C.source_of(ov.get('price')))],
            ["Day change %", r.day_change_pct, ""],
            ["Market cap", C.big_number(_v(ov.get("market_cap"))), C.SOURCE_LABEL.get(C.source_of(ov.get('market_cap')))],
            ["Shares out.", C.big_number(_v(ov.get("shares_outstanding"))), ""],
            ["52w range", None if not r.range_52w else f"{r.range_52w['low']:.2f} - {r.range_52w['high']:.2f} ({r.range_52w['position_pct']}%)", ""],
        ],
    ))

    # Fundamentals
    f = r.fundamentals
    parts.append(f"<h2>Fundamentals — {f.rubric} (sub-score {f.subscore}, completeness {f.data_completeness:.0%})</h2>")
    parts.append(_table(
        ["Metric", "Value", "Source", "Sector median", "TASI", "Pctile", "Score"],
        [[m["label"], m["display"], C.SOURCE_LABEL.get(m["source"], m["source"]),
          m["sector_median"], m["tasi_value"], m["percentile"], m["metric_score"]] for m in f.metrics],
    ))

    # Technical
    t = r.technical
    parts.append(f"<h2>Technical (sub-score {t.subscore})</h2>")
    rows = []
    for tf in ("monthly", "weekly", "daily"):
        if tf in t.by_timeframe:
            lat = t.by_timeframe[tf].latest
            rows.append([tf, _fmt(lat["rsi"]), lat["rsi_state"], _fmt(lat["macd"]),
                         lat["macd_cross"], _fmt(lat["adx"])])
    parts.append(_table(["Timeframe", "RSI", "RSI state", "MACD", "Cross", "ADX"], rows))

    # Trend
    parts.append("<h2>Trend (probabilistic — not a forecast)</h2>")
    for call in (r.trend.short_term, r.trend.medium_term):
        if call:
            parts.append(f"<p><b>{_esc(call.horizon)}:</b> {_esc(call.classification)} "
                         f"(score {call.composite_score}, {call.confidence} confidence). "
                         f"{_esc(call.invalidation or '')}</p>")

    # Risk & dividends
    rk, dv = r.risk, r.dividends
    parts.append("<h2>Risk & Income</h2>")
    parts.append(_table(
        ["Beta", "Ann. vol", "Max DD", "Sharpe", "VaR95", "Risk score", "Div yield", "Payout", "FCF cover"],
        [[rk.beta_vs_tasi, _pct(rk.annualized_vol), _pct(rk.max_drawdown), rk.sharpe,
          _pct(rk.var_95), rk.risk_score, _pct(dv.dividend_yield), _pct(dv.payout_ratio), dv.fcf_cover]],
    ))

    # Shariah
    sh = r.shariah
    compliant = "Insufficient data" if sh.compliant is None else ("Compliant" if sh.compliant else "Non-compliant")
    parts.append(f"<h2>Shariah (indicative — {_esc(sh.methodology)})</h2><p>{compliant}</p>")
    parts.append(_table(["Screen", "Value", "Threshold", "Pass"],
                        [[c.name, _pct(c.value), f"{c.threshold:.0%}",
                          "—" if c.passed is None else ("Yes" if c.passed else "No")] for c in sh.checks]))

    parts.append("<h2>Disclaimers</h2><ul>" + "".join(f"<li>{_esc(d)}</li>" for d in r.disclaimers) + "</ul>")

    return f"<!doctype html><html><head><meta charset='utf-8'><title>{_esc(name)} report</title>" \
           f"<style>{_CSS}</style></head><body>{''.join(parts)}</body></html>"


def export_html(r, cfg, path: str) -> str:
    html_str = build_html(r, cfg)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html_str)
    return path


def export_pdf(r, cfg, path: str) -> str | None:
    """Best-effort PDF via WeasyPrint; returns None if the tooling is unavailable."""
    try:
        from weasyprint import HTML  # type: ignore
    except Exception:
        return None
    try:
        HTML(string=build_html(r, cfg)).write_pdf(path)
        return path
    except Exception:
        return None


def _fmt(x):
    return "N/A" if x is None else (f"{x:.2f}" if isinstance(x, (int, float)) else str(x))


def _pct(x):
    return "N/A" if x is None else f"{x*100:.1f}%"
