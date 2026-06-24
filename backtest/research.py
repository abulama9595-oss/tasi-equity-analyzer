"""Phase-1 research harness: cross-sectional factor evaluation with walk-forward OOS + costs.

Builds a point-in-time panel of natural price factors, normalises each factor cross-sectionally
within the universe per date (z-score, winsorised), then reports for every factor:
- Information Coefficient by horizon (mean + t-stat) and its decay,
- first-half / second-half IC (stability),
- top-minus-bottom quintile spread (gross),
- portfolio turnover and net-of-cost spread.
Plus a walk-forward, sign-oriented equal-weight composite (orientation uses only past,
completed forward returns -> no look-ahead) evaluated OOS, with a net-of-cost equity curve.

    python -m backtest.research [--tickers N] [--liq-drop 0.2] [--cost-bps 25]

Indicative only: small/short, survivorship-biased universe. A factor is interesting only if its
IC is significant (|t|>2) AND stable across halves AND survives costs.
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

import analyzer as az
from config.settings import get_config
from data.ticker_registry import TickerRegistry

from .engine import HORIZONS, MAX_H, _month_end_positions
from .factors import FACTORS, compute_factors
from .metrics import _spearman

log = logging.getLogger(__name__)
RESULTS = Path(__file__).resolve().parent / "results"
PRIMARY = "fwd_1m"  # horizon used for the composite equity curve (non-overlapping monthly)


def build_panel(min_bars: int = 300, max_tickers: int | None = None) -> pd.DataFrame:
    cfg = get_config()
    reg = TickerRegistry()
    composite, sahmk = az.build_composite(cfg)
    index_df, _ = composite.get_index_history("1d")

    refs = reg.all_refs()
    if max_tickers:
        refs = refs[:max_tickers]

    rows: list[dict] = []
    for n, ref in enumerate(refs, 1):
        sym = ref.symbol
        price_df, _ = composite.get_price_history(sym, cfg.timeframes.price_history_period, "1d")
        if price_df is None or len(price_df) < min_bars + MAX_H:
            continue
        f = compute_factors(price_df)
        close = price_df["close"].to_numpy()
        idx_aligned = index_df["close"].reindex(price_df.index).ffill().to_numpy() if not index_df.empty else None
        positions = _month_end_positions(price_df.index, min_bars, len(price_df) - 1 - MAX_H)
        log.info("[%d/%d] %s: %d dates", n, len(refs), sym, len(positions))
        for pos in positions:
            row = {"date": price_df.index[pos], "ticker": ref.code, "sector": ref.sector}
            frow = f.iloc[pos]
            for fac in FACTORS:
                row[fac] = float(frow[fac]) if pd.notna(frow[fac]) else np.nan
            row["liquidity"] = float(frow["liquidity"]) if pd.notna(frow["liquidity"]) else np.nan
            for name, h in HORIZONS.items():
                row[name] = float(close[pos + h] / close[pos] - 1) if pos + h < len(close) else np.nan
                if idx_aligned is not None and pos + h < len(idx_aligned) and idx_aligned[pos]:
                    row[f"idx_{name}"] = float(idx_aligned[pos + h] / idx_aligned[pos] - 1)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)


def cross_normalize(panel: pd.DataFrame, liq_drop: float = 0.2, min_names: int = 10) -> pd.DataFrame:
    """Per date: drop the least-liquid `liq_drop` fraction, then z-score (winsorised ±3)
    each factor across the surviving names. Adds z_<factor> columns."""
    out = []
    for _, g in panel.groupby("date"):
        g = g.copy()
        if liq_drop > 0 and g["liquidity"].notna().sum() >= min_names:
            thr = g["liquidity"].quantile(liq_drop)
            g = g[g["liquidity"] >= thr]
        if len(g) < min_names:
            continue
        for fac in FACTORS:
            x = g[fac]
            mu, sd = x.mean(), x.std(ddof=0)
            g[f"z_{fac}"] = ((x - mu) / sd).clip(-3, 3) if sd and sd > 0 else np.nan
        out.append(g)
    return pd.concat(out).reset_index(drop=True) if out else panel


def _ic(panel: pd.DataFrame, zcol: str, ret: str) -> dict:
    ics = []
    for _, g in panel.groupby("date"):
        v = _spearman(g[zcol], g[ret])
        if not math.isnan(v):
            ics.append(v)
    a = np.array(ics, dtype=float)
    if a.size == 0:
        return {"mean": float("nan"), "t": float("nan"), "n": 0}
    sd = a.std(ddof=1) if a.size > 1 else float("nan")
    return {"mean": float(a.mean()), "t": float(a.mean() / (sd / math.sqrt(a.size))) if sd and sd > 0 else float("nan"),
            "n": int(a.size)}


def _quintile_spread(panel: pd.DataFrame, zcol: str, ret: str, min_names: int = 15) -> float:
    spreads = []
    for _, g in panel.groupby("date"):
        gg = g[[zcol, ret]].dropna()
        if len(gg) < min_names:
            continue
        q = pd.qcut(gg[zcol].rank(method="first"), 5, labels=False)
        m = gg.groupby(q)[ret].mean()
        if 4 in m.index and 0 in m.index:
            spreads.append(m[4] - m[0])
    return float(np.mean(spreads)) if spreads else float("nan")


def _turnover_and_net(panel: pd.DataFrame, zcol: str, ret: str = PRIMARY,
                      top_frac: float = 0.2, cost_rate: float = 0.0025, min_names: int = 15):
    """Top-quintile long-only: average monthly turnover and gross/net mean return."""
    dates = sorted(panel["date"].unique())
    prev: set | None = None
    gross, net, turns = [], [], []
    for d in dates:
        g = panel[panel["date"] == d][["ticker", zcol, ret]].dropna()
        if len(g) < min_names:
            continue
        thr = g[zcol].quantile(1 - top_frac)
        sel = set(g[g[zcol] >= thr]["ticker"])
        r = g[g[zcol] >= thr][ret].mean()
        if prev is not None and sel:
            changed = len(prev.symmetric_difference(sel))
            to = changed / max(len(sel), 1)  # ~ fraction of book traded (both sides)
        else:
            to = 1.0
        cost = to * cost_rate
        gross.append(r); net.append(r - cost); turns.append(to)
        prev = sel
    if not gross:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(turns)), float(np.mean(gross)), float(np.mean(net))


def per_factor_report(panel: pd.DataFrame) -> pd.DataFrame:
    half = panel["date"].quantile(0.5)
    recs = []
    for fac in FACTORS:
        z = f"z_{fac}"
        rec = {"factor": fac}
        for h in HORIZONS:
            ic = _ic(panel, z, h)
            rec[f"IC_{h}"] = ic["mean"]
            rec[f"t_{h}"] = ic["t"]
        h1 = _ic(panel[panel["date"] <= half], z, "fwd_3m")
        h2 = _ic(panel[panel["date"] > half], z, "fwd_3m")
        rec["IC3m_H1"], rec["IC3m_H2"] = h1["mean"], h2["mean"]
        rec["spread12m"] = _quintile_spread(panel, z, "fwd_12m")
        to, gross, net = _turnover_and_net(panel, z)
        rec["turnover"], rec["net_1m"] = to, net
        recs.append(rec)
    return pd.DataFrame(recs)


def walkforward_composite(panel: pd.DataFrame, horizon: str = PRIMARY, train_min: int = 600,
                          cost_rate: float = 0.0025) -> dict:
    """Orient each factor by the sign of its IC on PAST completed data, average into a composite,
    evaluate OOS. No look-ahead: at date d only observations whose forward window ended <= d are
    used to set signs."""
    h_days = HORIZONS[horizon]
    panel = panel.sort_values("date").copy()
    dates = sorted(panel["date"].unique())
    zcols = [f"z_{f}" for f in FACTORS]
    comp_rows = []
    for d in dates:
        cutoff = pd.Timestamp(d) - pd.Timedelta(days=int(h_days * 1.6))  # forward window must have completed
        past = panel[panel["date"] <= cutoff]
        if len(past) < train_min:
            continue
        signs = {}
        for f in FACTORS:
            z = f"z_{f}"
            ic = _spearman(past[z], past[horizon])
            signs[f] = np.sign(ic) if not math.isnan(ic) and ic != 0 else 0.0
        cur = panel[panel["date"] == d].copy()
        cur["z_composite"] = np.nanmean(
            np.vstack([signs[f] * cur[f"z_{f}"].to_numpy() for f in FACTORS]), axis=0
        )
        comp_rows.append(cur)
    if not comp_rows:
        return {}
    comp = pd.concat(comp_rows).reset_index(drop=True)
    ic = {h: _ic(comp, "z_composite", h) for h in HORIZONS}
    to, gross, net = _turnover_and_net(comp, "z_composite", ret=horizon, cost_rate=cost_rate)
    # equity curve (net) vs TASI
    eq = _composite_equity(comp, horizon, cost_rate)
    return {"ic": ic, "turnover": to, "gross_1m": gross, "net_1m": net, "equity": eq, "n_obs": len(comp)}


def _composite_equity(comp: pd.DataFrame, horizon: str, cost_rate: float, top_frac: float = 0.2, min_names: int = 15):
    idx_col = f"idx_{horizon}"
    dates = sorted(comp["date"].unique())
    prev = None
    recs = []
    for d in dates:
        g = comp[comp["date"] == d][["ticker", "z_composite", horizon] + ([idx_col] if idx_col in comp.columns else [])].dropna(subset=["z_composite", horizon])
        if len(g) < min_names:
            continue
        thr = g["z_composite"].quantile(1 - top_frac)
        sel = g[g["z_composite"] >= thr]
        names = set(sel["ticker"])
        to = (len(prev.symmetric_difference(names)) / max(len(names), 1)) if prev else 1.0
        bench = float(g[idx_col].iloc[0]) if idx_col in g.columns and g[idx_col].notna().any() else float(g[horizon].mean())
        recs.append({"date": d, "port": float(sel[horizon].mean()) - to * cost_rate, "bench": bench})
        prev = names
    if not recs:
        return None
    c = pd.DataFrame(recs).set_index("date").sort_index()
    c["port_cum"] = (1 + c["port"]).cumprod()
    c["bench_cum"] = (1 + c["bench"]).cumprod()

    def stats(s):
        s = s.dropna()
        cum = (1 + s).prod(); yrs = len(s) / 12
        return {"cagr": cum ** (1 / yrs) - 1 if yrs > 0 and cum > 0 else float("nan"),
                "sharpe": (s.mean() * 12 - 0.052) / (s.std(ddof=1) * math.sqrt(12)) if s.std(ddof=1) else float("nan"),
                "maxdd": float(((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min()), "n": len(s)}
    return {"curve": c, "port": stats(c["port"]), "bench": stats(c["bench"])}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=None)
    ap.add_argument("--liq-drop", type=float, default=0.2)
    ap.add_argument("--cost-bps", type=float, default=25.0)
    args = ap.parse_args()
    cost_rate = args.cost_bps / 10000.0

    RESULTS.mkdir(parents=True, exist_ok=True)
    print("Building factor panel...")
    panel = build_panel(max_tickers=args.tickers)
    if panel.empty:
        print("No data."); return
    panel = cross_normalize(panel, liq_drop=args.liq_drop)
    panel.to_csv(RESULTS / "research_panel.csv", index=False)
    print(f"Panel: {len(panel)} obs, {panel['ticker'].nunique()} tickers, "
          f"{panel['date'].nunique()} dates ({panel['date'].min().date()} -> {panel['date'].max().date()})")

    pf = per_factor_report(panel)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\nPER-FACTOR (cross-sectional IC, t-stats; |t|>2 ~ significant)")
    show = pf[["factor", "IC_fwd_1m", "t_fwd_1m", "IC_fwd_3m", "t_fwd_3m", "IC_fwd_12m", "t_fwd_12m",
               "IC3m_H1", "IC3m_H2", "spread12m", "turnover", "net_1m"]].copy()
    for c in show.columns:
        if c != "factor":
            show[c] = show[c].astype(float).round(3)
    print(show.to_string(index=False))

    print(f"\nWALK-FORWARD COMPOSITE (sign-oriented by past {PRIMARY} IC, OOS, cost {args.cost_bps:.0f}bps)")
    wf = walkforward_composite(panel, cost_rate=cost_rate)
    if wf:
        for h in HORIZONS:
            print(f"  composite IC {h}: {wf['ic'][h]['mean']:+.3f} (t={wf['ic'][h]['t']:.1f}, n={wf['ic'][h]['n']})")
        eq = wf["equity"]
        if eq:
            p, b = eq["port"], eq["bench"]
            print(f"  net top-quintile: CAGR={p['cagr']*100:+.1f}% Sharpe={p['sharpe']:.2f} maxDD={p['maxdd']*100:.1f}% (n={p['n']})")
            print(f"  TASI            : CAGR={b['cagr']*100:+.1f}% Sharpe={b['sharpe']:.2f} maxDD={b['maxdd']*100:.1f}%")
        print(f"  turnover/mo={wf['turnover']:.2f}  gross_1m={wf['gross_1m']*100:+.2f}%  net_1m={wf['net_1m']*100:+.2f}%")
    pf.to_csv(RESULTS / "research_factors.csv", index=False)
    print(f"\nSaved: {RESULTS/'research_factors.csv'} and {RESULTS/'research_panel.csv'}")
    print("Indicative only; survivorship-biased, single market, ~7y. A factor needs significant + "
          "stable + cost-surviving IC to be worth promoting.")


if __name__ == "__main__":
    main()
