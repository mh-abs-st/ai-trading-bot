"""Position sizing, protective exits and the drawdown stop."""

from __future__ import annotations

from aitrader.config import RiskSettings


class RiskManager:
    """Decides how much to buy and when trading must stop."""

    def __init__(self, settings: RiskSettings, peak_equity: float = 0.0, halted: bool = False):
        self.settings = settings
        self.peak_equity = peak_equity
        self.halted = halted

    def update(self, equity: float) -> None:
        """Track the equity peak; halt for good once the drawdown limit is reached."""
        self.peak_equity = max(self.peak_equity, equity)
        if equity <= self.peak_equity * (1 - self.settings.max_drawdown_pct):
            self.halted = True

    def position_size(self, equity: float, affordable: float, price: float) -> float:
        """Quantity to buy so that hitting the stop loses ``risk_per_trade`` of equity.

        ``affordable`` is the largest order value the cash balance can pay for.
        """
        if self.halted or equity <= 0 or price <= 0:
            return 0.0

        value = equity * self.settings.risk_per_trade / self.settings.stop_loss_pct
        value = min(value, equity * self.settings.max_position_fraction, affordable)
        return max(value, 0.0) / price

    def stop_price(self, entry_price: float) -> float:
        return entry_price * (1 - self.settings.stop_loss_pct)

    def take_price(self, entry_price: float) -> float | None:
        if self.settings.take_profit_pct is None:
            return None
        return entry_price * (1 + self.settings.take_profit_pct)
