"""Trading strategies and the registry that creates them by name."""

from __future__ import annotations

from typing import Any

from aitrader.strategies.base import Strategy
from aitrader.strategies.ml import MLStrategy
from aitrader.strategies.rsi_reversion import RsiReversionStrategy
from aitrader.strategies.sma_cross import SmaCrossStrategy

STRATEGIES: dict[str, type[Strategy]] = {
    SmaCrossStrategy.name: SmaCrossStrategy,
    RsiReversionStrategy.name: RsiReversionStrategy,
    MLStrategy.name: MLStrategy,
}


def create_strategy(name: str, params: dict[str, Any] | None = None) -> Strategy:
    try:
        strategy_class = STRATEGIES[name]
    except KeyError:
        known = ", ".join(sorted(STRATEGIES))
        raise ValueError(f"Unknown strategy '{name}'. Available: {known}") from None
    return strategy_class(**(params or {}))


__all__ = ["STRATEGIES", "Strategy", "create_strategy"]
