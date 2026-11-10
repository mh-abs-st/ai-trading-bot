"""Replays a strategy over candle history with simulated orders."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aitrader.backtest.metrics import summarize
from aitrader.config import FeeSettings, RiskSettings
from aitrader.execution.broker import PaperBroker
from aitrader.models import Signal, Trade
from aitrader.risk.manager import RiskManager
from aitrader.strategies.base import Strategy


@dataclass
class BacktestResult:
    equity: pd.Series
    timestamps: pd.Series
    trades: list[Trade]
    metrics: dict[str, float]
    halted: bool


def run_backtest(
    candles: pd.DataFrame,
    strategy: Strategy,
    risk: RiskSettings | None = None,
    fees: FeeSettings | None = None,
    initial_cash: float = 10_000.0,
    train_fraction: float | None = None,
) -> BacktestResult:
    """Trade ``strategy`` over ``candles``.

    A signal is read when a candle closes and filled at the next candle's
    open, so no decision uses a price it could not have known. The first
    ``train_fraction`` of the history is used only for training and warm-up;
    results cover the rest. By default that is half for strategies that
    learn and nothing for those that do not.
    """
    risk = risk or RiskSettings()
    fees = fees or FeeSettings()

    if train_fraction is None:
        train_fraction = 0.5 if strategy.requires_training else 0.0
    if not 0 <= train_fraction < 1:
        raise ValueError("train_fraction must be at least 0 and below 1.")

    total = len(candles)
    start = max(int(total * train_fraction), strategy.warmup)
    if total - start < 2:
        raise ValueError("Not enough candles to backtest this strategy.")

    if strategy.requires_training:
        strategy.fit(candles.iloc[:start])
    signals = strategy.signals(candles).to_numpy()

    opens = candles["open"].to_numpy(dtype=float)
    highs = candles["high"].to_numpy(dtype=float)
    lows = candles["low"].to_numpy(dtype=float)
    closes = candles["close"].to_numpy(dtype=float)
    times = list(candles["timestamp"])

    broker = PaperBroker(initial_cash, fees.fee_rate, fees.slippage_bps)
    manager = RiskManager(risk, peak_equity=initial_cash)
    pending = Signal.HOLD
    stop: float | None = None
    take: float | None = None
    equity: list[float] = []

    for i in range(start, total):
        # 1. Fill the order decided at the previous close, at this candle's open.
        if pending == Signal.BUY and broker.quantity == 0:
            quantity = manager.position_size(
                broker.equity(opens[i]), broker.affordable(opens[i]), opens[i]
            )
            if quantity > 0:
                fill = broker.buy(quantity, opens[i], times[i])
                stop = manager.stop_price(fill.price)
                take = manager.take_price(fill.price)
        elif pending == Signal.SELL and broker.quantity > 0:
            broker.sell_all(opens[i], times[i], "signal")
        pending = Signal.HOLD

        # 2. Protective exits inside the candle. If both levels were touched the
        #    order is unknown, so assume the stop came first.
        if broker.quantity > 0 and stop is not None and lows[i] <= stop:
            broker.sell_all(min(stop, opens[i]), times[i], "stop_loss")
        elif broker.quantity > 0 and take is not None and highs[i] >= take:
            broker.sell_all(max(take, opens[i]), times[i], "take_profit")

        # 3. Mark to market at the close and decide what to do next.
        value = broker.equity(closes[i])
        manager.update(value)
        equity.append(value)

        if manager.halted:
            pending = Signal.SELL if broker.quantity > 0 else Signal.HOLD
        else:
            pending = Signal(int(signals[i]))

    if broker.quantity > 0:
        broker.sell_all(closes[-1], times[-1], "end_of_data")
        equity[-1] = broker.cash

    window = candles["timestamp"].iloc[start:].reset_index(drop=True)
    curve = pd.Series(equity)
    benchmark = closes[-1] / opens[start] - 1
    return BacktestResult(
        equity=curve,
        timestamps=window,
        trades=broker.trades,
        metrics=summarize(curve, window, broker.trades, initial_cash, benchmark),
        halted=manager.halted,
    )
