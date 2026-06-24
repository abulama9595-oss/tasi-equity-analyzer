"""Run the backtest end-to-end: score point-in-time, compute analytics, write a report.

    python -m backtest.run                  # full universe, monthly
    python -m backtest.run --tickers 15     # quick subset
    python -m backtest.run --every 3        # quarterly rebalance

Outputs: backtest/results/backtest_rows.csv and backtest/results/report.html, plus a console
summary. NOTE: fundamentals are an *approximate* point-in-time (see docs/AUDIT.md §9-10);
results are indicative, not a definitive quant validation (small/short, survivorship-biased
universe, no transaction costs).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from .engine import BacktestConfig, HORIZONS, run_backtest
from . import metrics as M

RESULTS = Path(__file__).resolve().parent / "results"


def _fmt_pct(x):
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:+.2f}%"


def _fmt(x, nd=2):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def print_summary(data: dict) -> None:
    print("\n" + "=" * 78)
    print(f"BACKTEST SUMMARY  | {data['n_tickers']} tickers, {data['n_dates']} dates "
          f"({data['date_min']} -> {data['date_max']}), {data['n_rows']} observations")
    print("=" * 78)

    print("\nInformation Coefficient (rank corr of score vs forward return; t>2 ~ significant)")
    print(f"{'score':<16}" + "".join(f"{h:>14}" for h in HORIZONS))
    for sc, by_h in data["ic"].items():
        line = f"{sc:<16}"
        for h in HORIZONS:
            s = by_h[h]
            line += f"{('%.3f(t%.1f)' % (s['mean_ic'], s['t_stat'])) if s['n_dates'] else 'n/a':>14}"
        print(line)

    print("\nTop-minus-bottom quintile spread (avg forward return, top score - bottom score)")
    print(f"{'score':<16}" + "".join(f"{h:>14}" for h in HORIZONS))
    for sc, by_h in data["buckets"].items():
        line = f"{sc:<16}"
        for h in HORIZONS:
            b = by_h[h]
            line += f"{(('%+.2f%%(t%.1f)' % (b['spread_mean'] * 100, b['spread_t'])) if b else 'n/a'):>14}"
        print(line)

    print("\nRating hit-rates (3-tier)")
    for h in ("fwd_3m", "fwd_12m"):
        print(f"  [{h}]")
        for r, e in data["rating"][h].items():
            print(f"    {r:<5} n={e['n']:<5} mean={_fmt_pct(e['mean_ret'])}  "
                  f"%positive={_fmt_pct(e['pct_positive'])}  %beat_index={_fmt_pct(e.get('beat_index'))}")

    print("\nTop-quintile long-only equity curve (monthly), vs TASI")
    for sc, eq in data["equity"].items():
        if not eq:
            continue
        p, b = eq["port"], eq["bench"]
        print(f"  [{sc}] portfolio: CAGR={_fmt_pct(p['cagr'])} Sharpe={_fmt(p['sharpe'])} "
              f"maxDD={_fmt_pct(p['max_dd'])} (n={p['n']})")
        print(f"  {' ' * len(sc)}   TASI:      CAGR={_fmt_pct(b['cagr'])} Sharpe={_fmt(b['sharpe'])} "
              f"maxDD={_fmt_pct(b['max_dd'])}")
    print("=" * 78)
    print("Reminder: indicative only. Fundamentals are approximate PIT; universe is small and "
          "survivorship-biased; no costs. Not financial advice.\n")


def _ic_table_html(data: dict) -> str:
    head = "<tr><th>Score</th>" + "".join(f"<th>{h}</th>" for h in HORIZONS) + "</tr>"
    rows = ""
    for sc, by_h in data["ic"].items():
        cells = ""
        for h in HORIZONS:
            s = by_h[h]
            txt = f"{s['mean_ic']:+.3f} (t={s['t_stat']:.1f}, n={s['n_dates']})" if s["n_dates"] else "n/a"
            color = "#1a7f37" if (s["n_dates"] and s["t_stat"] and s["t_stat"] > 2) else "#57606a"
            cells += f"<td style='color:{color}'>{txt}</td>"
        rows += f"<tr><td><b>{sc}</b></td>{cells}</tr>"
    return f"<table>{head}{rows}</table>"


def _rating_table_html(data: dict) -> str:
    out = ""
    for h in ("fwd_3m", "fwd_12m"):
        out += f"<h4>{h}</h4><table><tr><th>Rating</th><th>n</th><th>Mean ret</th><th>% positive</th><th>% beat index</th></tr>"
        for r, e in data["rating"][h].items():
            out += (f"<tr><td>{r}</td><td>{e['n']}</td><td>{_fmt_pct(e['mean_ret'])}</td>"
                    f"<td>{_fmt_pct(e['pct_positive'])}</td><td>{_fmt_pct(e.get('beat_index'))}</td></tr>")
        out += "</table>"
    return out


def _equity_fig_html(data: dict) -> str:
    html = ""
    for sc, eq in data["equity"].items():
        if not eq or "curve" not in eq:
            continue
        c = eq["curve"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=c.index, y=c["port_cum"], name=f"Top-quintile ({sc})", line=dict(color="#10b981")))
        fig.add_trace(go.Scatter(x=c.index, y=c["bench_cum"], name="TASI", line=dict(color="#8b97a7")))
        fig.update_layout(title=f"Equity curve — top-quintile by {sc} vs TASI (growth of 1)",
                          height=380, margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(orientation="h"))
        html += fig.to_html(include_plotlyjs="cdn", full_html=False)
    return html


def write_html(data: dict, path: Path) -> None:
    css = ("body{font-family:-apple-system,Segoe UI,Arial,sans-serif;margin:24px;color:#1f2328}"
           "table{border-collapse:collapse;margin:8px 0 18px}th,td{border:1px solid #d0d7de;padding:6px 10px;"
           "font-size:.9rem;text-align:left}th{background:#f6f8fa}h1{margin-bottom:0}.warn{background:#fff8c5;"
           "border:1px solid #d4a72c;padding:10px;border-radius:6px;margin:12px 0}")
    body = [
        f"<h1>TASI Analyzer — Backtest report</h1>",
        f"<p>{data['n_tickers']} tickers · {data['n_dates']} dates · {data['date_min']} → {data['date_max']} · "
        f"{data['n_rows']} observations</p>",
        "<div class='warn'><b>Indicative only.</b> Fundamentals are an approximate point-in-time "
        "(latest statement with report_date ≤ t; current share count). Universe is small and "
        "survivorship-biased; no transaction costs. Not financial advice.</div>",
        "<h2>Information Coefficient</h2><p>Rank correlation between score and forward return, "
        "averaged across dates. |t| &gt; 2 (green) is roughly statistically meaningful.</p>",
        _ic_table_html(data),
        "<h2>Rating hit-rates</h2>", _rating_table_html(data),
        "<h2>Equity curves</h2>", _equity_fig_html(data),
    ]
    path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head>"
                    f"<body>{''.join(body)}</body></html>", encoding="utf-8")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=None, help="max tickers (default: full universe)")
    ap.add_argument("--every", type=int, default=1, help="rebalance every N months (default 1)")
    ap.add_argument("--min-bars", type=int, default=500, help="min daily bars before scoring")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    bt = BacktestConfig(min_bars=args.min_bars, max_tickers=args.tickers, every_n_months=args.every)
    print("Running backtest (this can take a few minutes on a full run)...")
    df = run_backtest(bt)
    if df.empty:
        print("No results (insufficient data).")
        return
    df.to_csv(RESULTS / "backtest_rows.csv", index=False)
    data = M.full_report_data(df)
    print_summary(data)
    write_html(data, RESULTS / "report.html")
    print(f"Saved: {RESULTS / 'backtest_rows.csv'}\n       {RESULTS / 'report.html'}")


if __name__ == "__main__":
    main()
