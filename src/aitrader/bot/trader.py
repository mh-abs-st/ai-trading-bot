"""The automated trading loop: read candles, decide, place simulated orders."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from aitrader.config import Settings
from aitrader.data.source import MarketDataSource
from aitrader.execution.broker import PaperBroker
from aitrader.models import Signal
from aitrader.risk.manager import RiskManager
from aitrader.strategies.base import Strategy

log = logging.getLogger(__name__)


@dataclass
class StepReport:
    timestamp: pd.Timestamp | None
    price: float
    signal: Signal
    action: str
    equity: float


class Trader:
    """Runs one strategy on one market with a paper broker.

    Each step looks at the newest closed candle. A candle that was already
    handled is skipped, so calling ``step`` more often than candles close is
    harmless.
    """

    def __init__(
        self,
        settings: Settings,
        source: MarketDataSource,
        strategy: Strategy,
        broker: PaperBroker,
        risk: RiskManager,
        state_path: str | Path | None = None,
    ):
        self.settings = settings
        self.source = source
        self.strategy = strategy
        self.broker = broker
        self.risk = risk
        self.state_path = Path(state_path) if state_path else None
        self._last_bar: pd.Timestamp | None = None
        self._stop: float | None = None
        self._take: float | None = None
        self._trained = False
        self._load_state()

    def step(self) -> StepReport:
        candles = self.source.fetch(
            self.settings.symbol, self.settings.timeframe, self.settings.history_bars
        )
        if len(candles) < self.strategy.warmup + 2:
            raise ValueError("Not enough candle history for this strategy.")

        bar = candles.iloc[-1]
        timestamp = bar["timestamp"]
        price = float(bar["close"])

        if self._last_bar is not None and timestamp <= self._last_bar:
            return StepReport(timestamp, price, Signal.HOLD, "wait", self.broker.equity(price))
        self._last_bar = timestamp

        if self.strategy.requires_training and not self._trained:
            self.strategy.fit(candles)
            self._trained = True
        signal = Signal(int(self.strategy.signals(candles).iloc[-1]))

        action = "hold"
        if self.broker.quantity > 0:
            if self._stop is not None and float(bar["low"]) <= self._stop:
                action = self._sell(price, timestamp, "stop_loss")
            elif self._take is not None and float(bar["high"]) >= self._take:
                action = self._sell(price, timestamp, "take_profit")

        self.risk.update(self.broker.equity(price))

        if self.broker.quantity > 0 and self.risk.halted:
            action = self._sell(price, timestamp, "drawdown_limit")
        elif self.broker.quantity > 0 and signal == Signal.SELL:
            action = self._sell(price, timestamp, "signal")
        elif self.broker.quantity == 0 and signal == Signal.BUY and action == "hold":
            quantity = self.risk.position_size(
                self.broker.equity(price), self.broker.affordable(price), price
            )
            if quantity > 0:
                fill = self.broker.buy(quantity, price, timestamp)
                self._stop = self.risk.stop_price(fill.price)
                self._take = self.risk.take_price(fill.price)
                action = "buy"
            elif self.risk.halted:
                action = "halted"

        self._save_state()
        report = StepReport(timestamp, price, signal, action, self.broker.equity(price))
        log.info(
            "%s price=%.2f signal=%s action=%s equity=%.2f",
            timestamp, price, signal.name, action, report.equity,
        )
        return report

    def run(self, max_steps: int | None = None) -> None:
        """Step forever, or ``max_steps`` times, sleeping between checks."""
        steps = 0
        while max_steps is None or steps < max_steps:
            try:
                self.step()
            except Exception:
                # One failed fetch or decision must not stop an unattended bot.
                log.exception("Step failed; trying again at the next poll.")
            steps += 1
            if max_steps is None or steps < max_steps:
                time.sleep(self.settings.poll_seconds)

    def _sell(self, price: float, timestamp: pd.Timestamp, reason: str) -> str:
        self.broker.sell_all(price, timestamp, reason)
        self._stop = None
        self._take = None
        return f"sell ({reason})"

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        state = {
            "symbol": self.settings.symbol,
            "broker": self.broker.to_state(),
            "stop": self._stop,
            "take": self._take,
            "last_bar": self._last_bar.isoformat() if self._last_bar is not None else None,
            "peak_equity": self.risk.peak_equity,
            "halted": self.risk.halted,
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(self.state_path)

    def _load_state(self) -> None:
        if self.state_path is None or not self.state_path.exists():
            return
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("symbol") != self.settings.symbol:
            raise ValueError(
                f"{self.state_path} belongs to {state.get('symbol')}, not {self.settings.symbol}. "
                "Use a different state_file or delete it."
            )
        self.broker.load_state(state["broker"])
        self._stop = state.get("stop")
        self._take = state.get("take")
        if state.get("last_bar"):
            self._last_bar = pd.Timestamp(state["last_bar"])
        self.risk.peak_equity = float(state.get("peak_equity", 0.0))
        self.risk.halted = bool(state.get("halted", False))
