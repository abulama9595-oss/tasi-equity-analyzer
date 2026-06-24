"""Phase-2: build & validate a 3-month cross-sectional composite, walk-forward, out-of-sample.

Construction (no look-ahead): at each rebalance date the factor weights are fit ONLY on past
observations whose 3-month forward window had already completed. Two weighting schemes:
  - sign  : orient each factor by the sign of its past 3m IC, equal-weight
  - ic    : weight each factor by its past 3m IC (down-weights weak factors)

Evaluation is deliberately conservative:
  - headline IC uses QUARTERLY, non-overlapping 3m returns (monthly 3m IC overlaps -> inflated t),
  - the equity curve rebalances quarterly with a hysteresis buffer (reduce turnover) and 25bps costs,
  - robustness via first-half / second-half IC,
  - an explicit PASS/FAIL gate before anything could be promoted to the live verdict.

    python -m backtest.strategy [--cost-bps 25] [--top 0.2]
"""
from __future__ import annotations

import argparse
import logging
import math

import numpy as np
import pandas as pd

from .factors import FACTORS
from .metrics import _spearman
from .research import build_panel, cross_normalize

log = logging.getLogger(__name__)
HZ = "fwd_3m"
HZ_DAYS = 63

FACTOR_SETS = {
    "tech_3m": ["px_vs_sma200", "ret_1m", "rsi_14", "dist_52w_high"],   # the 3m-significant technicals
    "tech_lowvol": ["px_vs_sma200", "ret_1m", "rsi_14", "vol_126"],     # + low-vol
    "all": list(FACTORS),
}


def _fit_weights(past: pd.DataFrame, factors: list[str], scheme: str) -> dict[str, float]:
    w = {}
    for f in factors:
        ic = _spearman(past[f"z_{f}"], past[HZ])
        ic = 0.0 if math.isnan(ic) else ic
        w[f] = (np.sign(ic) if scheme == "sign" else ic)
    norm = sum(abs(v) for v in w.values())
    return {f: (v / norm if norm else 0.0) for f, v in w.items()}


def build_composite(panel: pd.DataFrame, factors: list[str], scheme: str = "ic",
                    train_min: int = 600) -> pd.DataFrame:
    panel = panel.sort_values("date")
    dates = sorted(panel["date"].unique())
    out = []
    for d in dates:
        cutoff = pd.Timestamp(d) - pd.Timedelta(days=int(HZ_DAYS * 1.6))
        past = panel[panel["date"] <= cutoff]
        if len(past) < train_min:
            continue
        w = _fit_weights(past, factors, scheme)
        cur = panel[panel["date"] == d].copy()
        cur["z_composite"] = sum(w[f] * cur[f"z_{f}"] for f in factors)
        out.append(cur)
    return pd.concat(out).reset_index(drop=True) if out else pd.DataFrame()


def _ic_over(dates_panel: pd.DataFrame, zcol: str, ret: str) -> dict:
    ics = []
    for _, g in dates_panel.groupby("date"):
        v = _spearman(g[zcol], g[ret])
        if not math.isnan(v):
            ics.append(v)
    a = np.array(ics, dtype=float)
    if a.size == 0:
        return {"mean": float("nan"), "t": float("nan"), "n": 0}
    sd = a.std(ddof=1) if a.size > 1 else float("nan")
    return {"mean": float(a.mean()), "t": (a.mean() / (sd / math.sqrt(a.size)) if sd and sd > 0 else float("nan")), "n": int(a.size)}


def _quarterly_dates(dates: list) -> list:
    return sorted(dates)[::3]  # every 3rd month ~ non-overlapping 3m holding


def quarterly_equity(comp: pd.DataFrame, top_frac: float, buffer_frac: float, cost_rate: float,
                     min_names: int = 15) -> dict | None:
    idx_col = f"idx_{HZ}"
    qdates = _quarterly_dates(list(comp["date"].unique()))
    held: set = set()
    recs = []
    for d in qdates:
        g = comp[comp["date"] == d][["ticker", "z_composite", HZ] + ([idx_col] if idx_col in comp.columns else [])].dropna(subset=["z_composite", HZ])
        if len(g) < min_names:
            continue
        g = g.sort_values("z_composite", ascending=False).reset_index(drop=True)
        n_target = max(1, int(round(len(g) * top_frac)))
        keep_thr_rank = int(round(len(g) * buffer_frac))
        keep_set = set(g.iloc[:keep_thr_rank]["ticker"]) & held       # hysteresis: retain if still in top buffer
        new = [t for t in g["ticker"] if t not in keep_set]
        chosen = list(keep_set) + new[: max(0, n_target - len(keep_set))]
        chosen = set(chosen[:n_target])
        ret = g[g["ticker"].isin(chosen)][HZ].mean()
        turn = (len(held.symmetric_difference(chosen)) / max(len(chosen), 1)) if held else 1.0
        bench = float(g[idx_col].iloc[0]) if idx_col in g.columns and g[idx_col].notna().any() else float(g[HZ].mean())
        recs.append({"date": d, "port": float(ret) - turn * cost_rate, "bench": bench, "turn": turn})
        held = chosen
    if not recs:
        return None
    c = pd.DataFrame(recs).set_index("date")

    def stats(s):
        s = s.dropna(); cum = (1 + s).prod(); yrs = len(s) * 0.25
        return {"cagr": cum ** (1 / yrs) - 1 if yrs > 0 and cum > 0 else float("nan"),
                "sharpe": (s.mean() * 4 - 0.052) / (s.std(ddof=1) * math.sqrt(4)) if s.std(ddof=1) else float("nan"),
                "maxdd": float(((1 + s).cumprod() / (1 + s).cumprod().cummax() - 1).min()), "n": len(s)}
    return {"port": stats(c["port"]), "bench": stats(c["bench"]), "turnover": float(c["turn"].mean())}


def evaluate(comp: pd.DataFrame, top_frac: float, cost_rate: float) -> dict:
    qd = set(_quarterly_dates(list(comp["date"].unique())))
    cq = comp[comp["date"].isin(qd)]
    half = comp["date"].quantile(0.5)
    res = {
        "ic_monthly_3m": _ic_over(comp, "z_composite", HZ),          # overlapping -> inflated t
        "ic_quarterly_3m": _ic_over(cq, "z_composite", HZ),          # non-overlapping -> honest t
        "ic_H1": _ic_over(cq[cq["date"] <= half], "z_composite", HZ),
        "ic_H2": _ic_over(cq[cq["date"] > half], "z_composite", HZ),
        "equity": quarterly_equity(comp, top_frac, buffer_frac=max(top_frac * 2, 0.4), cost_rate=cost_rate),
    }
    return res


def gate(res: dict) -> dict:
    icq = res["ic_quarterly_3m"]
    eq = res["equity"] or {}
    p, b = eq.get("port", {}), eq.get("bench", {})
    c1 = (icq["n"] > 4) and (icq["mean"] > 0) and (icq["t"] is not None and icq["t"] > 2)
    c2 = res["ic_H2"]["mean"] is not None and res["ic_H2"]["mean"] > 0
    c3 = (p.get("cagr") is not None and b.get("cagr") is not None and not math.isnan(p["cagr"])
          and p["cagr"] > b["cagr"] and p.get("sharpe", -9) > b.get("sharpe", -9))
    return {"oos_ic_significant": bool(c1), "works_in_H2": bool(c2), "beats_tasi_net": bool(c3),
            "PASS": bool(c1 and c2 and c3)}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-bps", type=float, default=25.0)
    ap.add_argument("--top", type=float, default=0.2)
    args = ap.parse_args()
    cost_rate = args.cost_bps / 10000.0

    print("Building panel...")
    panel = cross_normalize(build_panel())
    print(f"Panel: {len(panel)} obs, {panel['ticker'].nunique()} tickers, {panel['date'].nunique()} dates "
          f"({panel['date'].min().date()} -> {panel['date'].max().date()})\n")

    for set_name, facs in FACTOR_SETS.items():
        for scheme in ("ic", "sign"):
            comp = build_composite(panel, facs, scheme=scheme)
            if comp.empty:
                continue
            res = evaluate(comp, args.top, cost_rate)
            g = gate(res)
            icq, icm, h1, h2 = res["ic_quarterly_3m"], res["ic_monthly_3m"], res["ic_H1"], res["ic_H2"]
            eq = res["equity"] or {}
            p, b = eq.get("port", {}), eq.get("bench", {})
            print(f"=== {set_name} / {scheme}-weighted ===")
            print(f"  OOS 3m IC  quarterly(non-overlap): {icq['mean']:+.3f} (t={icq['t']:.1f}, n={icq['n']})   "
                  f"monthly(overlap): {icm['mean']:+.3f} (t={icm['t']:.1f})")
            print(f"  stability  H1={h1['mean']:+.3f}  H2={h2['mean']:+.3f}")
            if p:
                print(f"  net equity CAGR={p['cagr']*100:+.1f}% Sharpe={p['sharpe']:.2f} maxDD={p['maxdd']*100:.1f}%  "
                      f"| TASI CAGR={b['cagr']*100:+.1f}% Sharpe={b['sharpe']:.2f}  | turnover/q={eq['turnover']:.2f}")
            print(f"  GATE: oos_ic_sig={g['oos_ic_significant']}  works_H2={g['works_in_H2']}  "
                  f"beats_tasi_net={g['beats_tasi_net']}  ->  {'PASS' if g['PASS'] else 'FAIL'}\n")
    print("Indicative only (survivorship-biased, single market, ~7y). A PASS here is necessary, "
          "not sufficient, to trust it with real money.")


if __name__ == "__main__":
    main()
