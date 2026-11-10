"""Ranks strategies for a user by backtesting each one on recent history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from aitrader.backtest.engine import run_backtest
from aitrader.config import FeeSettings, RiskSettings
from aitrader.strategies import STRATEGIES, create_strategy

# How heavily each profile penalises drawdown when scoring a strategy.
DRAWDOWN_WEIGHT = {"conservative": 3.0, "balanced": 1.5, "aggressive": 0.5}

# Results from very few trades are mostly luck.
MIN_TRADES = 5

CANDIDATES: list[tuple[str, dict[str, Any]]] = [
    ("sma_cross", {"fast": 10, "slow": 30}),
    ("sma_cross", {"fast": 20, "slow": 50}),
    ("sma_cross", {"fast": 50, "slow": 200}),
    ("rsi_reversion", {"period": 14, "oversold": 30, "overbought": 70}),
    ("rsi_reversion", {"period": 7, "oversold": 25, "overbought": 75}),
    ("ml", {}),
]


@dataclass
class Recommendation:
    strategy: str
    params: dict[str, Any]
    description: str
    metrics: dict[str, float]
    score: float
    suitable: bool
    notes: list[str]


def recommend(
    candles: pd.DataFrame,
    profile: str = "balanced",
    risk: RiskSettings | None = None,
    fees: FeeSettings | None = None,
    initial_cash: float = 10_000.0,
    test_fraction: float = 0.5,
) -> list[Recommendation]:
    """Backtest every candidate on the same recent window and rank them.

    The score is return minus a drawdown penalty set by the risk profile. A
    strategy is marked suitable only if it made money, traded often enough to
    mean something, and never hit the drawdown stop.
    """
    if profile not in DRAWDOWN_WEIGHT:
        raise ValueError(f"Unknown risk profile '{profile}'.")

    results: list[Recommendation] = []
    for name, params in CANDIDATES:
        try:
            strategy = create_strategy(name, params)
            outcome = run_backtest(
                candles, strategy, risk, fees, initial_cash, train_fraction=1 - test_fraction
            )
        except ValueError as error:
            results.append(
                Recommendation(
                    name, params, STRATEGIES[name].description, {}, float("-inf"), False,
                    [f"Could not be tested: {error}"],
                )
            )
            continue

        metrics = outcome.metrics
        score = metrics["total_return"] - DRAWDOWN_WEIGHT[profile] * metrics["max_drawdown"]

        notes: list[str] = []
        if metrics["total_return"] <= 0:
            notes.append("Lost money in the test period.")
        if metrics["trades"] < MIN_TRADES:
            notes.append(f"Only {int(metrics['trades'])} trades, too few to judge.")
        if outcome.halted:
            notes.append("Hit the drawdown limit and stopped trading.")
        if metrics["total_return"] > metrics["benchmark_return"]:
            notes.append("Beat buying and holding over the test period.")
        else:
            notes.append("Did not beat buying and holding over the test period.")

        suitable = (
            metrics["total_return"] > 0 and metrics["trades"] >= MIN_TRADES and not outcome.halted
        )
        results.append(
            Recommendation(name, params, strategy.description, metrics, score, suitable, notes)
        )

    results.sort(key=lambda item: (item.suitable, item.score), reverse=True)
    return results
