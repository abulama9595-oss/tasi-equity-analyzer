"""Configuration loader & validator for the TASI Equity Analyzer.

Loads ``config/config.yaml`` and validates it with pydantic v2 models so a missing
or malformed key fails loudly at startup rather than surfacing as a confusing error
deep in the analysis. API keys are read from a repo-root ``.env`` (via python-dotenv)
by *name* — config.yaml only ever references the env-var name, never the value.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- #
# Paths — resolve relative to this file so launch directory does not matter.
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"

# Load .env from the repo root regardless of CWD. Silent if absent (yfinance-only).
load_dotenv(dotenv_path=ENV_PATH, override=False)


# --------------------------------------------------------------------------- #
# Pydantic models mirroring config.yaml
# --------------------------------------------------------------------------- #
class _Model(BaseModel):
    """Base that forbids unknown keys so typos in config.yaml are caught."""

    model_config = ConfigDict(extra="forbid")


class AppCfg(_Model):
    base_currency: str = "SAR"
    index_symbol: str = "^TASI.SR"
    locale: str = "en"
    watchlist_max: int = 30


class SaudiExchangeCfg(_Model):
    backend: str = "sahmk"  # sahmk | licensed | scraper
    base_url: str
    api_key_env: str
    quotes_delayed: bool = True


class NewsCfg(_Model):
    enabled: bool = False
    source: str = "saudi_exchange"
    marketaux_api_key_env: str = "MARKETAUX_API_KEY"


class TwelveDataCfg(_Model):
    enabled: bool = False
    api_key_env: str = "TWELVEDATA_API_KEY"


class ProvidersCfg(_Model):
    price_history: str = "yfinance"
    saudi_exchange: SaudiExchangeCfg
    enable_portal_scraper: bool = False
    news: NewsCfg
    twelvedata: TwelveDataCfg


class CacheCfg(_Model):
    dir: str = ".cache"
    ttl_seconds: dict[str, int]


class RiskFreeCfg(_Model):
    source: str = "config"
    annual_rate: float = 0.052


class MacdCfg(_Model):
    fast: int = 12
    slow: int = 26
    signal: int = 9


class BollingerCfg(_Model):
    period: int = 20
    std: float = 2.0


class StochasticCfg(_Model):
    k: int = 14
    d: int = 3
    smooth: int = 3


class IndicatorsCfg(_Model):
    rsi_period: int = 14
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    macd: MacdCfg
    sma_periods: list[int]
    ema_periods: list[int]
    bollinger: BollingerCfg
    atr_period: int = 14
    stochastic: StochasticCfg
    adx_period: int = 14


class ResampleCfg(_Model):
    weekly_rule: str = "W-THU"
    monthly_rule: str = "ME"
    drop_incomplete_trailing: bool = True


class TimeframesCfg(_Model):
    price_history_period: str = "10y"
    min_bars: dict[str, int]
    resample: ResampleCfg


class MetricScoringCfg(_Model):
    blend_sector_percentile: float = 0.35
    metrics: dict[str, dict[str, Any]]


class TechnicalScoreCfg(_Model):
    components: dict[str, float]
    timeframe_weights: dict[str, float]
    signals: dict[str, dict[str, Any]]


class TrendCfg(_Model):
    inputs: dict[str, float]
    composite_range: list[float]
    classification: dict[str, Any]
    horizons: dict[str, int]
    confidence: dict[str, float]
    optional_model: dict[str, Any]


class VerdictCfg(_Model):
    weights: dict[str, float]
    rating_bands: dict[str, float]
    three_tier_map: dict[str, list[str]]
    risk_score: dict[str, Any]
    data_completeness: dict[str, float]


class ShariahCfg(_Model):
    methodology: str = "aaoifi"
    denominator: str = "market_cap"
    thresholds: dict[str, float]


class Config(_Model):
    """Top-level validated configuration."""

    app: AppCfg
    providers: ProvidersCfg
    field_preference: dict[str, list[str]]
    cache: CacheCfg
    risk_free: RiskFreeCfg
    indicators: IndicatorsCfg
    timeframes: TimeframesCfg
    fundamental_score: dict[str, dict[str, float]]
    metric_scoring: MetricScoringCfg
    technical_score: TechnicalScoreCfg
    trend: TrendCfg
    verdict: VerdictCfg
    shariah: ShariahCfg

    # ----- convenience helpers -------------------------------------------- #
    def cache_dir(self) -> Path:
        """Absolute cache directory (created on demand by the cache layer)."""
        d = Path(self.cache.dir)
        return d if d.is_absolute() else PROJECT_ROOT / d

    def secret(self, env_name: str) -> str | None:
        """Read an API key from the environment by name. Never logs the value."""
        val = os.getenv(env_name)
        return val.strip() if val and val.strip() else None

    @property
    def sahmk_key(self) -> str | None:
        return self.secret(self.providers.saudi_exchange.api_key_env)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. Expected config/config.yaml at the repo root."
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"config.yaml did not parse to a mapping (got {type(data).__name__}).")
    return data


@lru_cache(maxsize=1)
def get_config(path: str | None = None) -> Config:
    """Load + validate config (cached). Raises with a clear message on bad config."""
    cfg_path = Path(path) if path else CONFIG_PATH
    raw = _read_yaml(cfg_path)
    try:
        return Config(**raw)
    except Exception as exc:  # pydantic ValidationError or similar
        raise ValueError(
            f"config.yaml failed validation — fix the keys below and retry.\n{exc}"
        ) from exc


# Field(...) import kept for forward-compatibility of stricter models.
_ = Field
