"""Top-level analysis orchestrator (UI-agnostic core).

Builds the composite provider from config, fetches all data for a ticker, runs every
analysis module, and returns a single structured AnalysisResult. The Streamlit app and the
report exporter both consume this — no analysis logic lives in the UI, so a FastAPI/React
front-end could reuse this module unchanged.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from config.settings import Config, get_config
from data.cache import DiskCache
from data.composite_provider import CompositeProvider
from data.provider_base import Sourced
from data.saudi_exchange_provider import SaudiExchangeProvider
from data.ticker_registry import TickerRegistry, TickerRef
from data.yfinance_provider import YFinanceProvider

from analysis import dividends as dividends_mod
from analysis import fundamentals as fundamentals_mod
from analysis import risk as risk_mod
from analysis import shariah as shariah_mod
from analysis import technicals as technicals_mod
from analysis import trend as trend_mod
from analysis import verdict as verdict_mod

log = logging.getLogger(__name__)

DISCLAIMER = "Not financial advice. For personal research only."


@dataclass
class AnalysisResult:
    ticker: str
    ref: TickerRef
    as_of: str
    currency: str
    company_type: str
    overview: dict[str, Sourced]
    key_stats: dict[str, Sourced]
    fundamentals: Any
    technical: Any
    trend: Any
    verdict: Any
    risk: Any
    dividends: Any
    shariah: Any
    provenance: dict[str, str]
    price_provenance: str
    warnings: list[str] = field(default_factory=list)
    disclaimers: list[str] = field(default_factory=list)
    day_change_pct: float | None = None
    range_52w: dict[str, float] | None = None
    error: str | None = None


def build_composite(cfg: Config) -> tuple[CompositeProvider, SaudiExchangeProvider]:
    cache = DiskCache(cfg.cache_dir(), cfg.cache.ttl_seconds, enabled=True)
    yf_provider = YFinanceProvider(cache=cache, index_symbol=cfg.app.index_symbol)
    se = cfg.providers.saudi_exchange
    sahmk = SaudiExchangeProvider(
        base_url=se.base_url,
        api_key=cfg.sahmk_key,
        backend=se.backend,
        quotes_delayed=se.quotes_delayed,
        cache=cache,
    )
    composite = CompositeProvider({"yfinance": yf_provider, "saudi_exchange": sahmk}, cfg.field_preference)
    return composite, sahmk


def _v(sourced: Sourced | None) -> Any:
    return getattr(sourced, "value", None) if sourced is not None else None


def analyze(
    ticker_input: str,
    cfg: Config | None = None,
    registry: TickerRegistry | None = None,
    composite: CompositeProvider | None = None,
    sahmk: SaudiExchangeProvider | None = None,
) -> AnalysisResult:
    cfg = cfg or get_config()
    registry = registry or TickerRegistry()
    if composite is None:
        composite, sahmk = build_composite(cfg)

    ref = registry.resolve(ticker_input)
    if ref is None:
        return _error_result(ticker_input, ref, cfg, f"'{ticker_input}' is not a valid TASI ticker or known name.")

    symbol = ref.symbol
    warnings: list[str] = []
    if sahmk is None or not sahmk.available:
        warnings.append("Running yfinance-only (no SAHMK key set) - Saudi-Exchange-sourced fields are limited.")
    if not ref.in_registry:
        warnings.append(f"{ref.code} is not in the bundled ticker reference; metadata comes from live providers.")

    # ---- data layer ---- #
    price_df, price_prov = composite.get_price_history(
        symbol, cfg.timeframes.price_history_period, "1d"
    )
    index_df, _ = composite.get_index_history("1d")
    overview = composite.get_overview(symbol)
    key_stats = composite.get_key_stats(symbol)
    financials, _ = composite.get_financials(symbol)
    dividends_df, _ = composite.get_dividends(symbol)

    if price_df is None or price_df.empty:
        warnings.append("No price history available from any source - technical/trend/risk skipped.")
        return _error_result(
            symbol, ref, cfg,
            "No price history available. The free price source (yfinance/Yahoo) returned "
            "nothing - this usually means Yahoo is rate-limiting this server, which is common "
            "on shared cloud hosts (e.g. Streamlit Cloud). It typically works when run locally "
            "or after a short wait. (The ticker could also be delisted.)",
            overview=overview, key_stats=key_stats, warnings=warnings,
        )

    # Backfill bilingual names / sector from the bundled registry when providers lack them
    # (e.g. yfinance has no Arabic name, and SAHMK is off without a key).
    def _backfill(key: str, val: Any) -> None:
        if val and (overview.get(key) is None or _v(overview.get(key)) is None):
            overview[key] = Sourced(value=val, source="registry")

    _backfill("name_en", ref.name_en)
    _backfill("name_ar", ref.name_ar)
    _backfill("sector", ref.sector)

    company_type = registry.detect_company_type(ref.code, _v(overview.get("sector")))

    # merge market_cap / shares from overview into stats for derived fundamentals
    merged_stats = dict(key_stats)
    for k in ("market_cap", "shares_outstanding"):
        if overview.get(k) is not None and _v(overview.get(k)) is not None:
            merged_stats.setdefault(k, overview[k])

    # ---- analysis modules ---- #
    fundamentals = fundamentals_mod.analyse(company_type, merged_stats, financials, overview, cfg)
    technical = technicals_mod.analyse(price_df, cfg)
    trend = trend_mod.analyse(price_df, index_df, technical, cfg)
    risk = risk_mod.analyse(price_df, index_df, cfg)
    dividends = dividends_mod.analyse(dividends_df, key_stats, overview, cfg)
    shariah = shariah_mod.analyse(financials, overview, key_stats, cfg, sector_flag=ref.shariah_flag)
    verdict = verdict_mod.analyse(fundamentals, technical, trend, risk, cfg)

    if fundamentals.data_completeness < 0.5:
        warnings.append(
            f"Limited fundamentals coverage ({fundamentals.data_completeness:.0%}) - "
            "scores re-normalised over available metrics and verdict reliability reduced."
        )

    # ---- derived overview bits ---- #
    day_change = _day_change(price_df)
    range_52w = _range_52w(price_df, overview)

    provenance = composite.provenance_summary(overview, key_stats)
    disclaimers = [DISCLAIMER]
    if cfg.providers.saudi_exchange.quotes_delayed and sahmk and sahmk.available:
        disclaimers.append("SAHMK quotes are ~15 minutes delayed.")
    disclaimers.append("Shariah screen and zakat helpers are indicative, not a fatwa or tax advice.")

    return AnalysisResult(
        ticker=symbol,
        ref=ref,
        as_of=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        currency=cfg.app.base_currency,
        company_type=company_type,
        overview=overview,
        key_stats=key_stats,
        fundamentals=fundamentals,
        technical=technical,
        trend=trend,
        verdict=verdict,
        risk=risk,
        dividends=dividends,
        shariah=shariah,
        provenance=provenance,
        price_provenance=price_prov.source,
        warnings=warnings,
        disclaimers=disclaimers,
        day_change_pct=day_change,
        range_52w=range_52w,
    )


def _day_change(price_df: pd.DataFrame) -> float | None:
    if len(price_df) < 2:
        return None
    last, prev = price_df["close"].iloc[-1], price_df["close"].iloc[-2]
    return round((last / prev - 1) * 100, 2) if prev else None


def _range_52w(price_df: pd.DataFrame, overview: dict[str, Sourced]) -> dict[str, float] | None:
    lo = _v(overview.get("fifty_two_week_low"))
    hi = _v(overview.get("fifty_two_week_high"))
    last = price_df["close"].iloc[-1]
    if lo is None or hi is None:
        window = price_df["close"].tail(252)
        lo, hi = float(window.min()), float(window.max())
    if hi == lo:
        return {"low": lo, "high": hi, "position_pct": 50.0, "price": float(last)}
    pos = (last - lo) / (hi - lo) * 100
    return {"low": float(lo), "high": float(hi), "position_pct": round(max(0, min(100, pos)), 1), "price": float(last)}


def _error_result(ticker, ref, cfg, msg, overview=None, key_stats=None, warnings=None) -> AnalysisResult:
    return AnalysisResult(
        ticker=ticker if isinstance(ticker, str) else getattr(ref, "symbol", str(ticker)),
        ref=ref or TickerRef(symbol=str(ticker), code="", in_registry=False),
        as_of=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        currency=cfg.app.base_currency,
        company_type="general",
        overview=overview or {},
        key_stats=key_stats or {},
        fundamentals=None, technical=None, trend=None, verdict=None,
        risk=None, dividends=None, shariah=None,
        provenance={}, price_provenance="none",
        warnings=warnings or [],
        disclaimers=[DISCLAIMER],
        error=msg,
    )
