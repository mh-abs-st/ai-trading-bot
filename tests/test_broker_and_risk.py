import pandas as pd
import pytest

from aitrader.config import RiskSettings
from aitrader.execution.broker import PaperBroker
from aitrader.risk.manager import RiskManager

NOW = pd.Timestamp("2024-01-01", tz="UTC")


def test_round_trip_charges_fees_and_slippage():
    broker = PaperBroker(1000, fee_rate=0.001, slippage_bps=10)
    broker.buy(1.0, 100.0, NOW)
    trade = broker.sell_all(100.0, NOW, "signal")
    # Bought at 100.10 plus fee, sold at 99.90 minus fee.
    assert trade.pnl == pytest.approx(99.9 - 0.0999 - 100.1 - 0.1001)
    assert broker.cash == pytest.approx(1000 + trade.pnl)
    assert broker.quantity == 0


def test_cannot_overspend():
    broker = PaperBroker(100)
    with pytest.raises(ValueError):
        broker.buy(2.0, 100.0, NOW)


def test_affordable_order_always_fits():
    broker = PaperBroker(1000, fee_rate=0.001, slippage_bps=25)
    broker.buy(broker.affordable(50.0) / 50.0, 50.0, NOW)
    assert broker.cash == pytest.approx(0, abs=1e-6)


def test_state_round_trip():
    broker = PaperBroker(1000)
    broker.buy(2.0, 100.0, NOW)
    restored = PaperBroker(1)
    restored.load_state(broker.to_state())
    assert restored.cash == pytest.approx(broker.cash)
    assert restored.quantity == 2.0
    assert restored.entry == broker.entry


def test_position_size_risks_the_configured_share():
    settings = RiskSettings(risk_per_trade=0.01, stop_loss_pct=0.05, max_position_fraction=1.0)
    manager = RiskManager(settings)
    quantity = manager.position_size(equity=10_000, affordable=10_000, price=100)
    loss_at_stop = quantity * 100 * settings.stop_loss_pct
    assert loss_at_stop == pytest.approx(100)


def test_position_size_respects_caps():
    settings = RiskSettings(risk_per_trade=0.05, stop_loss_pct=0.01, max_position_fraction=0.25)
    manager = RiskManager(settings)
    assert manager.position_size(10_000, 10_000, 100) == pytest.approx(25)
    assert manager.position_size(10_000, 500, 100) == pytest.approx(5)


def test_drawdown_limit_halts_trading():
    manager = RiskManager(RiskSettings(max_drawdown_pct=0.2))
    manager.update(10_000)
    manager.update(8_100)
    assert not manager.halted
    manager.update(8_000)
    assert manager.halted
    assert manager.position_size(8_000, 8_000, 100) == 0
    manager.update(12_000)
    assert manager.halted
