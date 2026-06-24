# TASI Equity Analyzer — Pre-Use Audit

**Date:** 2026-06-24 · **Scope:** logic, numbers, scoring, verdict methodology, code, results.
**Bottom line:** The *framework* is sound, transparent, and correctly engineered. But the
**fundamental data from the free/Starter SAHMK feed has material errors** (notably P/E and
bank "revenue"), and the **verdict methodology has a valuation double-count and no
validation/backtest**. Treat the output as a *transparent screen to generate questions*, **not**
a buy/sell trigger, and independently verify the fundamental numbers before acting.

---

## 0. How to read this

Severity:
- 🔴 **Critical** — can produce a materially wrong verdict; fix or work around before relying on it.
- 🟠 **High** — methodology weakness that biases results; should be fixed.
- 🟡 **Medium/Low** — quality/edge issues; good to fix, not dangerous.
- 🟢 **Verified correct** — checked and trustworthy.

---

## 1. 🟢 What is correct (verified)

- **Indicator math** (RSI/MACD/SMA/EMA/Bollinger/ATR/Stochastic/ADX/OBV) — unit-tested against
  reference values (`tests/test_indicators.py`). RSI matches Wilder's reference (~70.5).
- **Resampling** daily→weekly(W-THU)→monthly — unit-tested (`tests/test_resampling.py`); correct
  OHLCV aggregation and incomplete-period dropping.
- **Weight re-normalisation** — confirmed live: e.g. Al Rajhi's bank rubric drops the
  unavailable `asset_quality` component and re-normalises 0.25/0.35/0.15 → 0.333/0.467/0.20.
- **Risk metrics** — beta (cov/var vs TASI), annualised vol (σ·√252), max drawdown, Sharpe,
  historical 95% VaR — all computed by standard formulas. Beta ≈ 0.98 for Al Rajhi is sane.
- **Verdict breakdown is auditable** — contributions = value × weight-used and reproduce the
  composite by hand.
- **P/B is reliable** — matches price ÷ book value (Aramco 4.17 ✓, Al Rajhi 1.97 ✓).
- **Graceful degradation & provenance** — missing inputs become N/A, weights re-normalise, and
  every value carries its source. No crashes on loss-makers / sparse data.
- **No look-ahead bias** — indicators use only past bars; fundamentals use already-public statements.

---

## 2. 🔴 Critical — fundamental DATA accuracy (SAHMK feed)

These are data-source problems, not code bugs, but they flow straight into the scores.

### 2.1 P/E is understated ~2× because `eps_ttm` is wrong
SAHMK's `eps_ttm` (used for `pe_ratio`) disagrees with its own `basic_eps` and with the statements:

| Ticker | SAHMK P/E | SAHMK eps_ttm | SAHMK basic_eps | NI÷shares (statements) | **Real P/E** |
|---|---|---|---|---|---|
| 2222 Aramco | 7.65 | 3.45 | 1.44 | 1.44 | **≈18.3** |
| 1120 Al Rajhi | 7.93 | 9.47 | 4.13 | 4.13 | **≈16.0** |

For Al Rajhi the reported `pe_ratio` (7.93) doesn't even equal price÷eps_ttm (6.97) — internally
inconsistent. **Effect:** stocks look ~2× cheaper than reality on P/E, inflating the valuation
score (a top driver of the verdict).
**Fix:** compute P/E as `price ÷ basic_eps` (or price ÷ latest-annual-NI/shares) instead of
trusting `eps_ttm`. SAHMK's `basic_eps` and statement net income are correct, so this is a clean fix.

### 2.2 Bank/financial "total_revenue" is not year-comparable
Al Rajhi `total_revenue`: FY2023 **27.5B** → FY2024 **52.7B** → FY2025 **80.1B**. A bank's revenue
does not ~triple in two years — this is a definitional break (net financing income vs gross), not
growth. **Effect:** `revenue_growth` = +51.8% is spurious and pushed Al Rajhi's growth component to
**100/100** (wrong). Net income (16.6→19.7→24.8B) looks plausible, so margins/ROE off net income
are less affected, but anything off "revenue" (P/S, margins, revenue growth) is unreliable for
banks/financials.
**Fix:** for `company_type in {bank, insurance}`, drop or down-weight revenue-based metrics
(revenue_growth, P/S, gross/operating margin) and rely on EPS/NI growth, ROE, P/B.

### 2.3 SAHMK's own endpoints disagree
`company.fundamentals`, `/analytics/ratios`, and values computed from `/financials` give
different ROE/net-margin (e.g. Al Rajhi net margin 31% computed vs 70.7% from analytics; ROE 17%
vs a quarterly-looking 4.9%). There is no single source of truth on the Starter plan.
**Implication:** every fundamental number should be treated as approximate (±, sometimes wrong),
not precise.

---

## 3. 🟠 High — verdict METHODOLOGY

### 3.1 Valuation is double-counted (and mis-named)
`verdict.weights` has both `fundamental` (0.35) **and** `valuation_vs_peers` (0.15). But
`valuation_vs_peers` is literally the **valuation component of the fundamental sub-score** — which is
*already* inside `fundamental`. So valuation is counted twice:
- inside fundamentals: ≈0.35 × 0.30 ≈ **0.105**
- as its own input: **0.15**
- **≈0.255 of the whole composite** — the single largest driver, double-weighted, and built on the
  unreliable P/E from §2.1.

Worse, it is **not "vs peers"** at all (see §3.2). Live proof (Al Rajhi): valuation 88.5 appears
inside `fundamental`=93.1 **and** again as `valuation_vs_peers`=88.5.
**Fix:** either remove `valuation_vs_peers` as a separate input (raise other weights), or repurpose
that 0.15 for a *genuine* relative/peer valuation once a peer set exists.

### 3.2 The sector-percentile blend never runs → scores are purely absolute
The methodology claims `metric_score = 0.65·anchor + 0.35·sector_percentile`. In single-ticker
mode **no peer set is supplied**, so percentile is always N/A and `metric_score = anchor` only. The
"Sector"/"TASI"/"Pctile" columns are always "—". **Effect:** stocks are judged against fixed
absolute thresholds with **no sector context** — a generic anchor set applied to banks, REITs,
energy, etc. The Methodology page advertises a blend that doesn't happen.
**Fix:** build a peer/sector median set (batch-fetch sector constituents, cache) and actually apply
the blend — or relabel the methodology to "absolute anchors only" so it's not misleading.

### 3.3 Trend and Technical overlap → price-momentum is over-weighted
`trend` inputs (ma_slope, adx_di, macd, structure, rel_strength) largely duplicate the `technical`
signals (which also use adx_di, macd, MAs). So `technical` (0.25) + `trend` (0.15) ≈ **0.40 of the
composite is the same price-trend information**, partly correlated. The labelled weights overstate
how independent these two pillars are.
**Fix:** make `trend` add genuinely different information (e.g. only relative-strength + structure)
or fold it into the technical sub-score and re-allocate the weight.

### 3.4 No validation / no backtest — predictive value is unproven
There is **no backtest of the verdict's hit-rate**, and the many constants (rating bands 80/65/45/30,
signal scalings like `/0.10`, `close·0.02`, trend input weights) are **reasonable judgement calls,
not empirically calibrated**. The original spec asked for a backtested hit-rate for any predictive
claim; none exists. **There is therefore no evidence the verdict outperforms a coin flip.** It is a
structured opinion, not a tested signal.

### 3.5 No value-trap guard
A loss-making company scores *high* valuation on a low P/B. Live proof (SABIC 2010): valuation
component **84.7/100** while it is losing money (net margin −22%, ROE −17%). "Cheap because it's
broken" is rewarded, not flagged.
**Fix:** gate valuation on quality (e.g. suppress/penalise low-multiple scores when ROE<0 or EPS<0).

---

## 4. 🟡 Medium / Low

- **eps_growth sign-flips explode** — SABIC eps_growth computed as **−1775%** (sign flip on a swing
  to losses). It's clamped to score 0 so it doesn't break scoring, but the displayed value is
  nonsensical. *Fix:* mark growth "n/m" when the base year is ≤0.
- **dividend_yield scaling (yfinance path)** — yfinance has historically returned yield as either a
  fraction or a percent; code divides by 100 only when >1, so a genuine sub-1% yield could be read
  as a huge number. Mitigated now that SAHMK is primary (clean fractions), but the latent
  ambiguity remains in the fallback.
- **market cap vs price×shares** — small discrepancies (Al Rajhi 401.7B vs 396.0B) from price-source
  timing / share-count rounding. Cosmetic.
- **Single snapshot, no data-freshness stamp per metric** — fundamentals are last-reported annuals;
  there's no "as-of" age shown per fundamental line, so stale statements aren't obvious.

---

## 5. Concrete impact: the bugs change the verdict

For **Al Rajhi (1120)** the live verdict is **BUY (composite 69.5)**. But that is inflated by:
- understated P/E (§2.1) → valuation scored ~100 instead of ~75,
- spurious +52% bank "revenue growth" (§2.2) → growth scored 100,
- valuation double-counted (§3.1).

Correcting P/E (→~16), neutralising the bogus bank revenue growth, and removing the double-count
would pull the fundamental sub-score from ~93 toward ~70 and the composite from ~69.5 toward the
mid-50s — i.e. **HOLD, not BUY.** So today's bugs produce a *materially more bullish* call than a
corrected model would. This is exactly why this matters before investing.

---

## 6. Recommendations (priority order)

1. 🔴 **Fix P/E**: use `price ÷ basic_eps` (or NI/shares), not `eps_ttm`. (Quick, high impact.)
2. 🔴 **Type-aware revenue handling**: drop revenue_growth / P/S / revenue-margins for bank &
   insurance rubrics; rely on EPS/NI growth, ROE, P/B.
3. 🟠 **Remove the valuation double-count** (drop `valuation_vs_peers` or make it genuinely peer-based).
4. 🟠 **Either implement real sector percentiles or relabel** the methodology as absolute-only.
5. 🟠 **Add a value-trap guard** (quality gate on valuation).
6. 🟠 **De-correlate trend vs technical** (or merge + re-weight).
7. 🟡 Sanitise eps_growth ("n/m" on non-positive base); show per-metric as-of dates.
8. 🟠 **Add a backtest harness** (even a simple forward-return study on the composite) before
   treating any rating as predictive — or permanently label it "screen, not signal".

## 7. How to use it safely *now* (before fixes)

- Treat the verdict as a **starting point for research**, never an execution trigger.
- **Independently verify** P/E, EPS, net income, and revenue growth (the unreliable ones) on the
  Tadawul/Argaam/company filings before acting. P/B, price, technicals, beta/vol/drawdown are more
  trustworthy.
- Be most sceptical of **banks/insurers** (revenue definition issues) and **loss-makers** (value-trap).
- Remember: it is **not financial advice**, has **no proven track record**, and the numbers can be
  wrong. Position sizing and risk are yours.

---

## 8. Technical & Trend deep-dive

- 🟠 **T1 — RSI signal was discontinuous at 30/70.** RSI 69→70 flipped the signal from +0.95 to
  +0.3 (and 31→30 from −0.95 to +0.5). Tiny RSI moves caused large signal jumps. *Fixed* with a
  single continuous formula (trend-level, damped toward extremes, plus a recent-direction term).
- 🟠 **T2 — Monthly timeframe over-dominated** (0.45) while having the fewest bars and most N/A
  indicators; daily (most responsive) only counted 0.20. *Fixed* → 0.40 / 0.35 / 0.25.
- 🟡 **T3 — MACD-histogram scaled by a fixed `price×0.02`**, not volatility-aware → rarely
  contributed. *Fixed* → normalised by ATR.
- 🟠 **TR1 — Trend ≈75% duplicated Technical** (ma_slope/adx/macd are the same signals), so
  price-momentum was effectively ~0.40 of the verdict. *Mitigated* by re-weighting trend inputs
  toward the parts that are NOT in the technical sub-score (price structure 0.25, relative
  strength 0.20) and away from ma_slope/adx/macd.
- 🟠 **TR2 — Trend confidence was overstated.** It measured "% of inputs agreeing", but the inputs
  are correlated, so they agreed by construction → "high confidence" on weak/sideways trends.
  *Fixed* → "high" now also requires a non-trivial trend magnitude; near-flat composites are "low".
- 🟢 Verified sane: support/resistance vs current price (fixed earlier), resampling, the
  golden/death-cross & price-vs-MA fallbacks for short-history names, OBV / ADX / Stochastic math.

Note: the signal scalings (e.g. `/0.10`, `/25`, ADX `/30`) remain **heuristic, not backtested** —
see §3.4. They are reasonable and now continuous/volatility-aware, but unvalidated.

---

## 9. Fixes applied in this pass

| # | Fix | Effect (verified) |
|---|---|---|
| 1 | P/E from `basic_eps`, not `eps_ttm` | Aramco P/E 7.65→**18.3**, Al Rajhi 7.93→**16.0** |
| 2 | Bank/insurance drop revenue-based metrics (use ROE / EPS-growth / P/B) | bank growth/profitability now NI-based, not bogus revenue |
| 3 | Removed `valuation_vs_peers` double-count; weights → fund 0.45 / tech 0.25 / trend 0.15 / risk 0.15 | verdict has 4 inputs; valuation counted once |
| 4 | Value-trap guard (cap valuation ≤40 when loss-making) | SABIC valuation 84.7→**40**, verdict hold→**sell** |
| 5 | Continuous RSI signal (no 30/70 jumps) | smooth momentum signal |
| 6 | ATR-normalised MACD-histogram | volatility-aware, comparable across stocks |
| 7 | Timeframe weights 0.40/0.35/0.25 (less monthly-dominant) | more daily responsiveness |
| 8 | Trend inputs re-weighted toward structure + rel-strength | less trend/technical double-counting |
| 9 | Trend confidence requires magnitude, not just agreement | sideways trends now report "low", not "high" |
| 10 | Growth only computed on a positive base year | kills −1775%-style sign-flip blow-ups |
| 11 | In-app disclaimer: "research screen, not a tested signal; verify fundamentals" | honest expectations |

Net effect on the example: **Al Rajhi BUY (69.5) → HOLD (63.9)**, **SABIC HOLD (45.3) → SELL (30.6)**,
Aramco unchanged HOLD (~50). The model is now more conservative and the inputs are more reliable.

## 10. Still outstanding (not done this pass)

- 🔴 **No backtest / validation** — still the biggest gap. The ratings are an opinion, not a tested
  signal. Recommend a forward-return study before trusting ratings quantitatively.
- 🟠 **Real sector percentiles** — single-ticker scoring is still absolute-only (the 0.35 sector
  blend doesn't run). Needs a peer-universe fetch.
- 🟡 Bank/insurance metrics NIM/NPL/CAR/combined-ratio — Pro-plan gated on SAHMK.
- 🟡 Per-metric "as-of" dates not shown.

---

## 11. Backtest results (point-in-time, `python -m backtest.run`)

**Sample:** 59 tickers, 64 monthly dates, 2018-05 → 2025-05, 3,504 observations.
**Caveats:** survivorship-biased (current listings only), small/short single-market sample,
no transaction costs, fundamentals are approximate PIT (~2y of dates only). **Indicative, not
definitive — but the signal is clear enough to act on the conclusion.**

### Information Coefficient (rank corr of score vs forward return; |t|>2 ≈ significant)
| score | 1m | 3m | 6m | 12m |
|---|---|---|---|---|
| technical | -0.01 | +0.01 | -0.03 | **-0.088 (t-4.1)** |
| trend | +0.02 | +0.05 (t1.7) | +0.02 | **-0.066 (t-2.8)** |
| risk | -0.01 | +0.01 | 0.00 | +0.02 |
| price_composite | 0.00 | +0.02 | -0.01 | **-0.065 (t-3.1)** |
| fundamental | +0.04 | +0.01 | -0.02 | +0.05 (t1.4) |
| full_composite | +0.02 | 0.00 | +0.01 | +0.046 (t2.3) |

### Top-minus-bottom quintile spread (12m)
technical **-10.4% (t-4.4)**, trend **-8.5% (t-3.4)**, price_composite **-10.0% (t-4.3)** —
i.e. the highest-scored basket *under*performed the lowest by ~10% over a year.

### Rating hit-rates — no discrimination
3m / 12m % positive is ~40% for **buy, hold, AND sell** alike; at 12m "sell" mean return
(+0.0%) was *better* than "buy" (-1.9%). The 3-tier rating has **no demonstrated edge**.

### Equity curves (monthly top-quintile, long-only, vs TASI)
- price_composite: CAGR +10.9%, Sharpe **0.33**, maxDD -32% (vs TASI +3.1%, Sharpe -0.04).
- full_composite: CAGR +1.7%, Sharpe -0.12 — **underperformed** TASI (+3.4%).

### Honest conclusion
- **The verdict/rating has no demonstrated predictive edge.** Buy/Hold/Sell hit-rates are
  indistinguishable; the composite does not rank future returns reliably.
- **Technical & trend are contrarian at the 1-year horizon** in this sample (high score →
  *under*performance, statistically significant). Over 1-3 months they are ~neutral.
- **Fundamentals show a weak positive tilt** but it is not statistically robust (tiny PIT sample).
- The one outperformance (price_composite *monthly* top-quintile) is weak (Sharpe 0.33), high
  drawdown, contradicted by the negative longer-horizon IC, and likely an equal-weight /
  short-momentum artifact — **not a reliable edge**.

**Implication for use:** treat the tool as a **transparent research screen** to understand a
stock's profile and generate questions — **not** as a buy/sell signal, and **do not size
positions off the verdict.** (No methodology change was made in response to these results, to
avoid overfitting to one in-sample backtest.)

---

## 12. Phase-1 research harness — cross-sectional factor findings (`python -m backtest.research`)

The live tool scores **absolute** indicator levels per stock; this harness instead computes
**cross-sectional** factors (ranked within the universe each date), which is where signal
usually lives. Panel: 59 tickers, 72 monthly dates (2017-08 → 2025-05), 3,214 obs. Same
survivorship/size caveats apply; 20% least-liquid names dropped each date.

### Per-factor cross-sectional IC (t-stats; |t|>2 ≈ significant)
| factor | IC 1m | IC 3m | IC 12m | 3m IC: H1 / H2 | 12m quintile spread |
|---|---|---|---|---|---|
| mom_12_1 | +0.01 | +0.00 | **-0.125 (t-4.8)** | 0.00 / 0.00 | **-14.3%** |
| ret_1m | -0.00 | **+0.050 (t2.6)** | +0.04 | 0.08 / 0.02 | +2.3% |
| ret_3y | +0.02 | +0.057 (t1.9) | +0.04 | 0.08 / 0.04 | -1.5% |
| **px_vs_sma200** | +0.03 | **+0.079 (t3.2)** | -0.03 | 0.12 / 0.03 | -5.1% |
| dist_52w_high | +0.02 | +0.052 (t1.9) | -0.02 | 0.07 / 0.03 | -5.2% |
| vol_126 | -0.03 | -0.03 | **-0.052 (t-2.0)** | -0.05 / -0.02 | -0.3% |
| rsi_14 | +0.00 | **+0.052 (t2.4)** | +0.02 | 0.09 / 0.01 | +2.3% |

### What this says (the encouraging part)
- **There IS real cross-sectional signal at ~3 months** in the *technical* factors:
  **price-vs-200dMA (t3.2)**, **1-month return / continuation (t2.6)**, and **RSI (t2.4)** are
  statistically significant. This is the cross-sectional version of the very signals the live
  tool uses absolutely — and unlike the absolute versions, these *rank* future 3-month returns.
- **Low-volatility** has a weak, directionally-consistent edge (high vol → lower 12m returns).

### The honest caveats (the cautious part)
- **The 3-month edge has decayed**: every signal is far stronger in H1 (2017-2021) than H2
  (2021-2025) — e.g. px_vs_sma200 IC 0.12 → 0.03. Recent robustness is weak.
- **Everything reverses by 12 months** — momentum IC -0.125 (t-4.8); these are *short-horizon*
  (≈3m) signals, not buy-and-hold-a-year signals.
- **The naive 1m-oriented composite did NOT beat TASI net of 25bps costs** (CAGR 3.1% vs 3.2%,
  Sharpe 0.04) — wrong horizon to orient on, and turnover ~0.9/mo is costly.

### Direction this sets for Phase 2/3
The signal lives in a **3-month cross-sectional trend/momentum** composite (px_vs_sma200 +
ret_1m + rsi, possibly + low-vol), oriented on the **3m** horizon, with turnover control. The
gates before it can touch the live verdict: significant **out-of-sample** 3m IC, the H1→H2
decay must not kill it, and it must **survive transaction costs**. Phase 1 shows the raw
material is there — it is NOT yet a validated, cost-surviving strategy.
