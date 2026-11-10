"""Performance figures computed from an equity curve and a list of trades."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aitrader.models import Trade

SECONDS_PER_YEAR = 365 * 24 * 3600


def periods_per_year(timestamps: pd.Series) -> float:
    """How many candles fit in a year, judged from the usual gap between them."""
    if len(timestamps) < 2:
        return 0.0
    gap = timestamps.diff().dropna().median().total_seconds()
    return SECONDS_PER_YEAR / gap if gap > 0 else 0.0


def max_drawdown(equity: np.ndarray) -> float:
    """Largest fall from a peak, as a positive fraction."""
    if len(equity) == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    return float(np.max(1 - equity / peaks))


def summarize(
    equity: pd.Series,
    timestamps: pd.Series,
    trades: list[Trade],
    initial_cash: float,
    benchmark_return: float,
) -> dict[str, float]:
    values = equity.to_numpy(dtype=float)
    returns = np.diff(values) / values[:-1] if len(values) > 1 else np.array([])

    sharpe = 0.0
    if len(returns) > 1 and returns.std() > 0:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(periods_per_year(timestamps)))

    wins = [trade.pnl for trade in trades if trade.pnl > 0]
    losses = [-trade.pnl for trade in trades if trade.pnl <= 0]
    if losses and sum(losses) > 0:
        profit_factor = sum(wins) / sum(losses)
    else:
        profit_factor = float("inf") if wins else 0.0

    return {
        "total_return": float(values[-1] / initial_cash - 1),
        "benchmark_return": float(benchmark_return),
        "max_drawdown": max_drawdown(values),
        "sharpe": sharpe,
        "trades": float(len(trades)),
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "profit_factor": float(profit_factor),
        "final_equity": float(values[-1]),
    }
