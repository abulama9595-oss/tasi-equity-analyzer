"""Phase-1 data-layer + analysis validation on the spec's test tickers.

Run:  python scripts/validate.py
Proves the composite provider (yfinance + Saudi Exchange) with provenance, correct
daily->weekly->monthly resampling, and the full analysis pipeline produce sane output
without crashing. Also exercises a deliberately small-cap / unknown ticker for graceful
degradation.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_config  # noqa: E402
from data.ticker_registry import TickerRegistry  # noqa: E402
import analyzer as az  # noqa: E402


def _v(s):
    return getattr(s, "value", None) if s is not None else None


def show(ticker: str, cfg, registry, composite, sahmk) -> None:
    print("=" * 78)
    print(f"TICKER INPUT: {ticker!r}")
    r = az.analyze(ticker, cfg, registry, composite, sahmk)
    if r.error:
        print(f"  -> ERROR (handled gracefully): {r.error}")
        for w in r.warnings:
            print(f"     warning: {w}")
        return

    ov = r.overview
    print(f"  Resolved: {r.ticker}  | type={r.company_type}  | as_of={r.as_of}")
    print(f"  Name: {_v(ov.get('name_en'))}  /  {_v(ov.get('name_ar'))}")
    print(f"  Sector: {_v(ov.get('sector'))}  | Price: {_v(ov.get('price'))} {r.currency} "
          f"(day {r.day_change_pct}%)  [price source: {r.price_provenance}]")
    mc = _v(ov.get("market_cap"))
    print(f"  Market cap: {mc:,.0f}" if mc else "  Market cap: N/A")
    if r.range_52w:
        print(f"  52w: {r.range_52w['low']:.2f} - {r.range_52w['high']:.2f} "
              f"(pos {r.range_52w['position_pct']}%)")

    print(f"  Provenance (sample): "
          + ", ".join(f"{k}:{v}" for k, v in list(r.provenance.items())[:6]))

    f = r.fundamentals
    print(f"  FUNDAMENTALS [{f.rubric}] subscore={f.subscore} "
          f"completeness={f.data_completeness:.0%}")
    present = [m for m in f.metrics if m["value"] is not None]
    print(f"     metrics present: {len(present)}/{len(f.metrics)} -> "
          + ", ".join(f"{m['label']}={m['display']}" for m in present[:5]))

    t = r.technical
    print(f"  TECHNICAL subscore={t.subscore}  timeframe_scores={t.timeframe_scores}")
    for tf in ("daily", "weekly", "monthly"):
        if tf in t.by_timeframe:
            lat = t.by_timeframe[tf].latest
            print(f"     {tf:7s}: RSI={_fmt(lat['rsi'])} state={lat['rsi_state']} "
                  f"MACD={_fmt(lat['macd'])} hist={_fmt(lat['macd_hist'])} ADX={_fmt(lat['adx'])} "
                  f"bars={len(t.by_timeframe[tf].df)}")

    tr = r.trend
    for call in (tr.short_term, tr.medium_term):
        if call:
            print(f"  TREND {call.horizon}: {call.classification} "
                  f"(score {call.composite_score}, {call.confidence} conf {call.confidence_pct})")

    rk = r.risk
    print(f"  RISK beta={rk.beta_vs_tasi} vol={_pct(rk.annualized_vol)} "
          f"maxDD={_pct(rk.max_drawdown)} sharpe={rk.sharpe} risk_score={rk.risk_score}")

    v = r.verdict
    print(f"  VERDICT: {v.rating5_label}  (3-tier: {v.rating3.upper()})  "
          f"composite={v.composite}  conviction={v.conviction}  "
          f"completeness={v.data_completeness:.0%}  low_reliability={v.low_reliability}")
    print("     breakdown:")
    for row in v.breakdown:
        print(f"        {row['input']:18s} value={str(row['value']):>6} "
              f"weight={row['weight']} used={row['weight_used']} "
              f"contrib={row['contribution']} running={row['running_composite']}")
    print(f"     BULL: {v.bull[:2]}")
    print(f"     BEAR: {v.bear[:2]}")
    if r.warnings:
        for w in r.warnings:
            print(f"     warning: {w}")


def _fmt(x):
    return "N/A" if x is None else f"{x:.2f}"


def _pct(x):
    return "N/A" if x is None else f"{x*100:.1f}%"


def main() -> None:
    cfg = get_config()
    registry = TickerRegistry()
    composite, sahmk = az.build_composite(cfg)
    print(f"SAHMK available: {sahmk.available}  (yfinance-only mode: {not sahmk.available})")
    for t in ["1120.SR", "2222.SR", "7010.SR", "9999"]:  # last is a deliberate bad/small ticker
        show(t, cfg, registry, composite, sahmk)
    print("=" * 78)
    print("DONE.")


if __name__ == "__main__":
    main()
