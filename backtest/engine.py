"""Point-in-time backtest engine.

For each monthly rebalance date `t`, scores every ticker in the universe using ONLY data
available at `t` (truncated price series; financial statements whose report_date <= t), then
records realised FORWARD returns after `t`. The output is one tidy row per (ticker, date).

Two score families are produced:
- price-signals (technical / trend / risk + a re-normalised "price composite") — clean,
  ~full price history, no look-ahead. This is the rigorous part.
- fundamentals + full verdict composite — uses an *approximate* point-in-time fundamentals
  (latest statement with report_date <= t; current share count). Limited to ~the last few
  years of statements and clearly less rigorous (see docs/AUDIT.md).

No look-ahead: indicators use only bars <= t; statements are filtered by report_date <= t;
forward returns are strictly after t.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

import analyzer as az
from analysis import fundamentals as fundamentals_mod
from analysis import risk as risk_mod
from analysis import technicals as technicals_mod
from analysis import trend as trend_mod
from analysis import verdict as verdict_mod
from analysis import scoring
from config.settings import get_config
from data.ticker_registry import TickerRegistry

log = logging.getLogger(__name__)

# forward-return horizons in trading days
HORIZONS = {"fwd_1m": 21, "fwd_3m": 63, "fwd_6m": 126, "fwd_12m": 252}
MAX_H = max(HORIZONS.values())


def _report_date(s: dict):
    return pd.to_datetime(s.get("report_date"), errors="coerce")


def _div_date(d: dict):
    return pd.to_datetime(
        d.get("distribution_date") or d.get("eligibility_date") or d.get("announcement_date"),
        errors="coerce",
    )


def pit_key_stats(fin_raw, div_raw, shares, price_t, asof) -> dict | None:
    """Approximate point-in-time key stats from the latest statement with report_date <= asof.
    Mirrors SaudiExchangeProvider.get_key_stats formulas, but uses price_t and the as-of
    statement. Uses current `shares` for all dates (historical share counts unavailable)."""
    if not fin_raw or not shares or not price_t:
        return None
    asof_ts = pd.Timestamp(asof)
    inc = [s for s in fin_raw.get("income_statements", []) if _report_date(s) <= asof_ts]
    bal = [s for s in fin_raw.get("balance_sheets", []) if _report_date(s) <= asof_ts]
    cfs = [s for s in fin_raw.get("cash_flows", []) if _report_date(s) <= asof_ts]
    if not inc or not bal:
        return None
    i0, b0 = inc[0], bal[0]
    ni, rev, gp, oi = i0.get("net_income"), i0.get("total_revenue"), i0.get("gross_profit"), i0.get("operating_income")
    eq, ta, td = b0.get("stockholders_equity"), b0.get("total_assets"), b0.get("total_debt")
    mc = price_t * shares
    eps = (ni / shares) if ni is not None else None
    s: dict = {"market_cap": mc, "shares_outstanding": shares, "trailing_eps": eps}
    if eps and eps > 0:
        s["pe"] = price_t / eps
    if eq:
        s["pb"] = mc / eq
        if ni is not None:
            s["roe"] = ni / eq
        if td is not None:
            s["debt_equity"] = td / eq
            s["total_debt"] = td
    if ta and ni is not None:
        s["roa"] = ni / ta
    if rev:
        s["ps"] = mc / rev
        if ni is not None:
            s["net_margin"] = ni / rev
        if gp is not None:
            s["gross_margin"] = gp / rev
        if oi is not None:
            s["operating_margin"] = oi / rev
    if len(inc) > 1:
        r1, n1 = inc[1].get("total_revenue"), inc[1].get("net_income")
        if rev and r1 and r1 > 0:
            s["revenue_growth"] = (rev - r1) / r1
        if ni is not None and n1 and n1 > 0:
            s["eps_growth"] = (ni - n1) / n1
    if cfs:
        fcf = cfs[0].get("free_cash_flow")
        if fcf is not None:
            s["free_cashflow"] = fcf
            if mc:
                s["fcf_yield"] = fcf / mc
    if div_raw:
        lo = asof_ts - pd.Timedelta(days=365)
        ttm = sum(
            (d.get("value") or 0) for d in div_raw.get("history", [])
            if (_div_date(d) is not None and lo < _div_date(d) <= asof_ts)
        )
        if ttm > 0:
            s["dividend_yield"] = ttm / price_t
            if eps and eps > 0:
                s["payout_ratio"] = ttm / eps
    return {k: v for k, v in s.items() if v is not None}


def _price_composite(technical, trend_score, risk_score, cfg) -> float:
    """Verdict composite using only the price-driven pillars (weights re-normalised)."""
    w = cfg.verdict.weights
    pairs = [(technical, w.get("technical", 0)), (trend_score, w.get("trend", 0)),
             (risk_score, w.get("risk", 0))]
    val, _ = scoring.weighted_average(pairs)
    return val


def _month_end_positions(index: pd.DatetimeIndex, min_pos: int, max_pos: int) -> list[int]:
    ser = pd.Series(np.arange(len(index)), index=index)
    last = ser.groupby([index.year, index.month]).max()
    return [int(p) for p in last.values if min_pos <= p <= max_pos]


def _forward_returns(close: np.ndarray, pos: int) -> dict:
    out = {}
    for name, h in HORIZONS.items():
        out[name] = float(close[pos + h] / close[pos] - 1) if pos + h < len(close) else np.nan
    return out


@dataclass
class BacktestConfig:
    min_bars: int = 500          # require ~2y of history before scoring
    max_tickers: int | None = None
    every_n_months: int = 1      # 1 = monthly rebalance


def run_backtest(bt: BacktestConfig | None = None, progress=True) -> pd.DataFrame:
    bt = bt or BacktestConfig()
    cfg = get_config()
    reg = TickerRegistry()
    composite, sahmk = az.build_composite(cfg)
    if not sahmk.available:
        raise SystemExit("Backtest requires the SAHMK (paid) feed for history + fundamentals.")

    index_df, _ = composite.get_index_history("1d")

    refs = reg.all_refs()
    if bt.max_tickers:
        refs = refs[: bt.max_tickers]

    rows: list[dict] = []
    for n, ref in enumerate(refs, 1):
        sym = ref.symbol
        price_df, _ = composite.get_price_history(sym, cfg.timeframes.price_history_period, "1d")
        if price_df is None or len(price_df) < bt.min_bars + MAX_H:
            if progress:
                log.info("[%d/%d] %s skipped (insufficient history)", n, len(refs), sym)
            continue
        # align the index to THIS ticker's calendar so positional forward lookups match
        idx_aligned = index_df["close"].reindex(price_df.index).ffill() if not index_df.empty else None

        fin_raw = sahmk._financials_raw(sym)
        div_raw = sahmk._dividends_raw(sym)
        co = sahmk._company(sym)
        shares = (co.get("fundamentals") or {}).get("shares_outstanding") if co else None
        ctype = reg.detect_company_type(ref.code, ref.sector)

        close = price_df["close"].to_numpy()
        positions = _month_end_positions(price_df.index, bt.min_bars, len(price_df) - 1 - MAX_H)
        positions = positions[:: bt.every_n_months]
        if progress:
            log.info("[%d/%d] %s: %d rebalance dates", n, len(refs), sym, len(positions))

        for pos in positions:
            t = price_df.index[pos]
            sub = price_df.iloc[: pos + 1]
            idx_sub = index_df.loc[:t] if not index_df.empty else index_df
            price_t = float(close[pos])

            tech = technicals_mod.analyse(sub, cfg)
            rk = risk_mod.analyse(sub, idx_sub, cfg)
            tr = trend_mod.analyse(sub, idx_sub, tech, cfg)
            trend_score, _ = verdict_mod._trend_to_score(tr)

            pit = pit_key_stats(fin_raw, div_raw, shares, price_t, t)
            fund = fundamentals_mod.analyse(ctype, pit, {}, {}, cfg) if pit else None
            fund_score = fund.subscore if fund else np.nan

            if fund is not None:
                v = verdict_mod.analyse(fund, tech, tr, rk, cfg)
                full_comp, rating = v.composite, v.rating3
            else:
                full_comp, rating = np.nan, None

            row = {
                "date": t, "ticker": ref.code, "sector": ref.sector, "type": ctype,
                "price": price_t,
                "technical": tech.subscore, "trend": trend_score, "risk": rk.risk_score,
                "fundamental": fund_score,
                "price_composite": _price_composite(tech.subscore, trend_score, rk.risk_score, cfg),
                "full_composite": full_comp, "rating": rating,
            }
            row.update(_forward_returns(close, pos))
            if idx_aligned is not None:
                ivals = idx_aligned.to_numpy()
                for name, h in HORIZONS.items():
                    row[f"idx_{name}"] = (
                        float(ivals[pos + h] / ivals[pos] - 1) if pos + h < len(ivals) and ivals[pos] else np.nan
                    )
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    return df
