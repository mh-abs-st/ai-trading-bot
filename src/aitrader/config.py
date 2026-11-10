"""User settings, loaded from a YAML file and validated."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StrategySettings(_Model):
    name: str = "sma_cross"
    params: dict[str, Any] = Field(default_factory=dict)


class RiskSettings(_Model):
    risk_per_trade: float = Field(0.01, gt=0, le=0.1)
    stop_loss_pct: float = Field(0.03, gt=0, lt=1)
    take_profit_pct: float | None = Field(0.06, gt=0)
    max_position_fraction: float = Field(0.5, gt=0, le=1)
    max_drawdown_pct: float = Field(0.2, gt=0, le=1)


class FeeSettings(_Model):
    fee_rate: float = Field(0.001, ge=0, lt=0.1)
    slippage_bps: float = Field(5, ge=0, lt=1000)


class Settings(_Model):
    mode: Literal["paper", "live"] = "paper"
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    initial_cash: float = Field(10_000, gt=0)
    history_bars: int = Field(1000, ge=100)
    poll_seconds: int = Field(60, ge=1)
    state_file: str = "state/paper.json"
    risk_profile: Literal["conservative", "balanced", "aggressive"] = "balanced"
    strategy: StrategySettings = Field(default_factory=StrategySettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    fees: FeeSettings = Field(default_factory=FeeSettings)


def load_settings(path: str | Path | None = None) -> Settings:
    """Read settings from a YAML file. With no path, return the defaults."""
    if path is None:
        return Settings()
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return Settings.model_validate(raw)
