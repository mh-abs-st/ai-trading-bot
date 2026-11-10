"""Order execution. Only the simulated broker exists so far."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from aitrader.models import Fill, Trade


class Broker(ABC):
    """Holds one long position in one market, or none."""

    @property
    @abstractmethod
    def cash(self) -> float: ...

    @property
    @abstractmethod
    def quantity(self) -> float: ...

    @abstractmethod
    def affordable(self, price: float) -> float:
        """Largest order value that the cash balance covers, fees included."""

    @abstractmethod
    def buy(self, quantity: float, price: float, timestamp: pd.Timestamp) -> Fill: ...

    @abstractmethod
    def sell_all(self, price: float, timestamp: pd.Timestamp, reason: str) -> Trade: ...

    def equity(self, price: float) -> float:
        return self.cash + self.quantity * price


class PaperBroker(Broker):
    """Simulated fills at the given price, worsened by slippage and charged a fee."""

    def __init__(self, cash: float, fee_rate: float = 0.001, slippage_bps: float = 5.0):
        self._cash = float(cash)
        self._quantity = 0.0
        self.fee_rate = fee_rate
        self.slippage = slippage_bps / 10_000
        self.entry: Fill | None = None
        self.trades: list[Trade] = []

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def quantity(self) -> float:
        return self._quantity

    def affordable(self, price: float) -> float:
        return self._cash / ((1 + self.slippage) * (1 + self.fee_rate))

    def buy(self, quantity: float, price: float, timestamp: pd.Timestamp) -> Fill:
        if quantity <= 0:
            raise ValueError("Buy quantity must be positive.")
        if self._quantity > 0:
            raise RuntimeError("A position is already open.")

        fill_price = price * (1 + self.slippage)
        cost = quantity * fill_price
        fee = cost * self.fee_rate
        if cost + fee > self._cash * (1 + 1e-9):
            raise ValueError("Not enough cash for this order.")

        self._cash -= cost + fee
        self._quantity = quantity
        self.entry = Fill(timestamp, "buy", quantity, fill_price, fee)
        return self.entry

    def sell_all(self, price: float, timestamp: pd.Timestamp, reason: str) -> Trade:
        if self._quantity <= 0 or self.entry is None:
            raise RuntimeError("There is no position to sell.")

        fill_price = price * (1 - self.slippage)
        proceeds = self._quantity * fill_price
        fee = proceeds * self.fee_rate
        self._cash += proceeds - fee

        entry = self.entry
        cost = entry.quantity * entry.price + entry.fee
        pnl = proceeds - fee - cost
        trade = Trade(
            entry_time=entry.timestamp,
            exit_time=timestamp,
            entry_price=entry.price,
            exit_price=fill_price,
            quantity=entry.quantity,
            pnl=pnl,
            return_pct=pnl / cost,
            reason=reason,
        )
        self.trades.append(trade)
        self._quantity = 0.0
        self.entry = None
        return trade

    def to_state(self) -> dict:
        """Balances and the open position, in a form that can be written as JSON."""
        entry = None
        if self.entry is not None:
            entry = {
                "timestamp": self.entry.timestamp.isoformat(),
                "quantity": self.entry.quantity,
                "price": self.entry.price,
                "fee": self.entry.fee,
            }
        return {"cash": self._cash, "quantity": self._quantity, "entry": entry}

    def load_state(self, state: dict) -> None:
        self._cash = float(state["cash"])
        self._quantity = float(state["quantity"])
        entry = state.get("entry")
        self.entry = None
        if entry is not None:
            self.entry = Fill(
                pd.Timestamp(entry["timestamp"]),
                "buy",
                float(entry["quantity"]),
                float(entry["price"]),
                float(entry["fee"]),
            )
