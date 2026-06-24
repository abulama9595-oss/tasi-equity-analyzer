"""Phase-3: sector-neutral multi-factor composite (Value + Quality + Momentum + Low-Vol).

Emulates the factor-investing playbook: combine economically-motivated, cross-sectional factors,
neutralised by group (company-type here; true GICS sector when that data is added), equal-weighted
(simple beats over-fit), and validated walk-forward OOS net of costs via the same gate as Phase 2.

Factors (each in the "higher = better" direction after orientation):
  Value   : earnings yield (EPS/price), book yield (1/PB), FCF yield, sales yield (1/PS)
  Quality : ROE, ROA, -Debt/Equity            (net-income / balance-sheet based -> reliable)
  Momentum: 12-1 month return
  Low-Vol : -126d volatility

Fixed equal weights + economic signs => no parameters fit on the data, so the composite is OOS by
construction. Fundamental factors limit those composites to the ~2y of point-in-time statements
SAHMK Starter provides (clearly underpowered); Mom/Low-Vol span the full price history.

    python -m backtest.multifactor [--tickers N] [--cost-bps 25] [--top 0.2]
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

import analyzer as az
from config.settings import get_config
from data.ticker_registry import TickerRegistry

from .engine import HORIZONS, MAX_H, _month_end_positions, pit_key_stats
from .factors import compute_factors
from . import strategy

log = logging.getLogger(__name__)

RAW = ["earnings_yield", "book_yield", "fcf_yield", "sales_yield", "roe", "roa",
       "neg_debt_equity", "mom_12_1", "neg_vol"]
GROUPS = {
    "value_z": ["z_earnings_yield", "z_book_yield", "z_fcf_yield", "z_sales_yield"],
    "quality_z": ["z_roe", "z_roa", "z_neg_debt_equity"],
    "momentum_z": ["z_mom_12_1"],
    "lowvol_z": ["z_neg_vol"],
}
COMPOSITES = {
    "MOM_LOWVOL": ["momentum_z", "lowvol_z"],          # price-only -> long sample
    "VALUE_QUALITY": ["value_z", "quality_z"],          # fundamentals -> short sample
    "VQM": ["value_z", "quality_z", "momentum_z"],
    "ALL4": ["value_z", "quality_z", "momentum_z", "lowvol_z"],
}


def build_mf_panel(min_bars: int = 300, max_tickers: int | None = None) -> pd.DataFrame:
    cfg = get_config()
    reg = TickerRegistry()
    composite, sahmk = az.build_composite(cfg)
    if not sahmk.available:
        raise SystemExit("Phase-3 needs the SAHMK feed for fundamentals.")
    index_df, _ = composite.get_index_history("1d")
    refs = reg.all_refs()[: max_tickers] if max_tickers else reg.all_refs()

    rows: list[dict] = []
    for n, ref in enumerate(refs, 1):
        sym = ref.symbol
        price_df, _ = composite.get_price_history(sym, cfg.timeframes.price_history_period, "1d")
        if price_df is None or len(price_df) < min_bars + MAX_H:
            continue
        f = compute_factors(price_df)
        close = price_df["close"].to_numpy()
        idx = index_df["close"].reindex(price_df.index).ffill().to_numpy() if not index_df.empty else None
        fin_raw, div_raw = sahmk._financials_raw(sym), sahmk._dividends_raw(sym)
        co = sahmk._company(sym)
        shares = (co.get("fundamentals") or {}).get("shares_outstanding") if co else None
        ctype = reg.detect_company_type(ref.code, ref.sector)
        if n % 20 == 0:
            log.info("[%d/%d] %s", n, len(refs), sym)

        for pos in _month_end_positions(price_df.index, min_bars, len(price_df) - 1 - MAX_H):
            t = price_df.index[pos]
            price_t = float(close[pos])
            fr = f.iloc[pos]
            pit = pit_key_stats(fin_raw, div_raw, shares, price_t, t)
            row = {"date": t, "ticker": ref.code, "type": ctype}
            # price-only factors
            row["mom_12_1"] = float(fr["mom_12_1"]) if pd.notna(fr["mom_12_1"]) else np.nan
            row["neg_vol"] = -float(fr["vol_126"]) if pd.notna(fr["vol_126"]) else np.nan
            # fundamental factors (point-in-time)
            eps = (pit or {}).get("trailing_eps")
            pb, ps = (pit or {}).get("pb"), (pit or {}).get("ps")
            row["earnings_yield"] = eps / price_t if (eps is not None and price_t) else np.nan
            row["book_yield"] = 1.0 / pb if (pb and pb > 0) else np.nan
            row["sales_yield"] = 1.0 / ps if (ps and ps > 0) else np.nan
            row["fcf_yield"] = (pit or {}).get("fcf_yield", np.nan)
            row["roe"] = (pit or {}).get("roe", np.nan)
            row["roa"] = (pit or {}).get("roa", np.nan)
            de = (pit or {}).get("debt_equity")
            row["neg_debt_equity"] = -de if de is not None else np.nan
            for name, h in HORIZONS.items():
                row[name] = float(close[pos + h] / close[pos] - 1) if pos + h < len(close) else np.nan
                if idx is not None and pos + h < len(idx) and idx[pos]:
                    row[f"idx_{name}"] = float(idx[pos + h] / idx[pos] - 1)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def neutralize_and_score(panel: pd.DataFrame, min_names: int = 12) -> pd.DataFrame:
    """Per date: demean each raw factor by company_type (sector-neutral proxy), then z-score
    (winsorised ±3) across the universe."""
    out = []
    for _, g in panel.groupby("date"):
        if len(g) < min_names:
            continue
        g = g.copy()
        for fac in RAW:
            x = g[fac].astype(float)
            dem = x - g.groupby("type")[fac].transform("mean")
            mu, sd = dem.mean(), dem.std(ddof=0)
            g[f"z_{fac}"] = ((dem - mu) / sd).clip(-3, 3) if sd and sd > 0 else np.nan
        out.append(g)
    panel = pd.concat(out).reset_index(drop=True) if out else panel
    for grp, cols in GROUPS.items():
        have = [c for c in cols if c in panel.columns]
        panel[grp] = panel[have].mean(axis=1, skipna=True)
    return panel


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=None)
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--top", type=float, default=0.2)
    args = ap.parse_args()
    cost_rate = args.cost_bps / 10000.0

    print("Building multi-factor panel (point-in-time)...")
    panel = neutralize_and_score(build_mf_panel(max_tickers=args.tickers))
    print(f"Panel: {len(panel)} obs, {panel['ticker'].nunique()} tickers, {panel['date'].nunique()} dates "
          f"({panel['date'].min().date()} -> {panel['date'].max().date()})\n")

    for name, cols in COMPOSITES.items():
        p = panel.copy()
        p["z_composite"] = p[cols].mean(axis=1, skipna=False)  # require all groups present
        sub = p.dropna(subset=["z_composite"])
        if sub["date"].nunique() < 6:
            print(f"=== {name}: insufficient sample ({sub['date'].nunique()} dates) ===\n")
            continue
        res = strategy.evaluate(sub, args.top, cost_rate)
        g = strategy.gate(res)
        icq, h1, h2 = res["ic_quarterly_3m"], res["ic_H1"], res["ic_H2"]
        eq = res["equity"] or {}
        pf, bf = eq.get("port", {}), eq.get("bench", {})
        print(f"=== {name}  ({sub['date'].nunique()} dates, {sub['date'].min().date()} -> {sub['date'].max().date()}) ===")
        print(f"  OOS 3m IC quarterly: {icq['mean']:+.3f} (t={icq['t']:.1f}, n={icq['n']})  | H1={h1['mean']:+.3f} H2={h2['mean']:+.3f}")
        if pf:
            print(f"  net equity CAGR={pf['cagr']*100:+.1f}% Sharpe={pf['sharpe']:.2f} maxDD={pf['maxdd']*100:.1f}%  "
                  f"| TASI CAGR={bf['cagr']*100:+.1f}% Sharpe={bf['sharpe']:.2f}")
        print(f"  GATE: oos_ic_sig={g['oos_ic_significant']} works_H2={g['works_in_H2']} "
              f"beats_tasi_net={g['beats_tasi_net']}  ->  {'PASS' if g['PASS'] else 'FAIL'}\n")
    print("Fixed equal weights + economic signs (no fitting => OOS by construction). Value/Quality "
          "composites are limited to ~2y of point-in-time statements (underpowered). Indicative only.")


if __name__ == "__main__":
    main()
