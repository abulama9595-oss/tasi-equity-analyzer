# TASI Equity Analyzer

A local, runnable **single-ticker equity analysis app for the Saudi market (Tadawul / TASI)**.
Enter a ticker (`1120`, `1120.SR`, or a company name like *Aramco* / *الراجحي*) and get a
company overview, **type-aware fundamentals**, **multi-timeframe technicals** (daily / weekly /
monthly), a **probabilistic trend assessment**, and a fully **auditable Buy / Hold / Sell
verdict** — plus risk metrics, dividend analysis, an indicative Shariah screen, and a
methodology page that renders the config verbatim.

> **Not financial advice. For personal research only.** Every score is transparent and
> reproducible by hand from the breakdown tables.

---

## Quick start (Windows)

```powershell
cd tasi_analyzer

# 1. create a virtual environment (Python 3.11+; tested on 3.12)
py -m venv .venv          # or: python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. install pinned dependencies
pip install -r requirements.txt

# 3. (optional) add API keys
copy .env.example .env    # then edit .env – all keys are optional

# 4. run
streamlit run app.py
```

On macOS / Linux use `python3 -m venv .venv && source .venv/bin/activate`.

The app opens at <http://localhost:8501>.

### Run the tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

### Validate the data layer on the spec's test tickers

```powershell
.\.venv\Scripts\python.exe scripts/validate.py
```

Proves the composite provider (with provenance), correct daily→weekly→monthly resampling,
and the full pipeline on `1120.SR`, `2222.SR`, `7010.SR`, plus graceful handling of a bad
ticker.

---

## API keys & secrets

All keys are **optional** and live only in a local `.env` (never in code or `config.yaml`).
The app runs **yfinance-only** with no keys at all — Saudi-Exchange-sourced fields are simply
marked unavailable.

| Variable | Purpose | Needed? |
|---|---|---|
| `SAHMK_API_KEY` | Saudi Exchange market data via SAHMK (Tadawul-licensed; free tier ~100 req/day, ~15-min delayed) | optional |
| `MARKETAUX_API_KEY` | News / disclosures alternative (Argaam-sourced) | optional |
| `TWELVEDATA_API_KEY` | Paid fundamentals (exchange code `XSAU`) | optional |

`.env` and `.cache/` are git-ignored. No key is ever logged or printed.

---

## How it works

```
app.py                     Streamlit UI (tabs: Overview / Fundamentals / Technical /
                           Trend / Verdict / Risk & Income / Shariah / Methodology)
analyzer.py                UI-agnostic orchestrator -> AnalysisResult
config/
  config.yaml              ALL weights, thresholds, rubrics (Appendix A, verbatim)
  settings.py              pydantic v2 loader/validator; reads keys from .env
data/
  provider_base.py         DataProvider ABC + provenance types
  yfinance_provider.py     free baseline (price history + fundamentals)
  saudi_exchange_provider.py  SAHMK backend (switchable: sahmk|licensed|scraper)
  composite_provider.py    per-field source preference, fallback, provenance
  ticker_registry.py       normalisation/validation + sector/type/Shariah lookup
  cache.py                 TTL disk cache
  tasi_tickers.csv         bundled ticker reference
indicators/technical.py    vendored RSI/MACD/MA/Bollinger/ATR/Stoch/ADX/OBV + resampling
analysis/                  scoring, fundamentals, technicals, trend, risk,
                           dividends, shariah, verdict
ui/components.py           plotly charts + score widgets
reports/export.py          self-contained HTML report (PDF best-effort)
tests/                     indicator math, resampling, scoring/verdict
scripts/validate.py        end-to-end data-layer validation
```

The **analysis core is UI-agnostic** — `analyzer.analyze()` returns a structured result the
Streamlit app and the HTML report both consume, so a FastAPI/React front-end could reuse it
unchanged.

### Design notes / choices
- **Indicators are vendored** (no TA-Lib, no `ta` library). The `ta` package breaks under
  NumPy 2; vendoring keeps the dependency surface small and every indicator unit-tested
  against reference values.
- **Config-driven, no magic numbers.** All weights/thresholds/rubrics are in `config.yaml`
  and surfaced verbatim on the in-app Methodology page.
- **Graceful degradation.** Missing metrics show `N/A`, are dropped from sub-scores, and the
  remaining weights are re-normalised; the verdict's data-completeness indicator drops and
  flags low reliability when fundamentals are sparse.
- **Type-aware fundamentals.** Banks / REITs / insurance / general each use a different
  metric set and component weights (generic ratios mislead for financials).
- **Trend is probabilistic, not a forecast** — a classification with confidence and explicit
  invalidation levels, never a price target.

---

## Known data limitations

- **yfinance** uses Yahoo's unofficial endpoints and can return empty/None; it is isolated
  behind the provider interface and cached. Arabic names, official sector classification, and
  free float are limited without a Saudi-Exchange source.
- **Fundamentals coverage is thin for smaller Saudi names** and for bank/REIT/insurance-
  specific metrics (NIM, NPL, CAR, FFO, combined ratio). These show `N/A` and are excluded
  from scoring rather than guessed.
- **Sector/TASI peer comparison** (the percentile blend) is only populated when a peer set is
  supplied; single-ticker runs fall back to anchor-only metric scores.
- **SAHMK** free tier is rate-limited (~100 req/day) and quotes are ~15-min delayed — cached
  hard and labelled in the UI.

## Plugging in a paid provider

Add a new class implementing `data/provider_base.py:DataProvider`, register it in
`analyzer.build_composite`, and add its name to the relevant lists in
`config.yaml:field_preference`. The composite layer will prefer/fall back to it per field and
record provenance automatically — no other code changes needed.
