import numpy as np
import pandas as pd
import pytest

from aitrader.backtest.engine import run_backtest
from aitrader.backtest.metrics import max_drawdown
from aitrader.config import FeeSettings, RiskSettings
from aitrader.models import Signal
from aitrader.recommend.advisor import CANDIDATES, recommend
from aitrader.strategies import create_strategy
from aitrader.strategies.base import Strategy


class Scripted(Strategy):
    """Emits a fixed list of signals, for exact checks of the engine."""

    name = "scripted"

    def __init__(self, script):
        self.script = script

    @property
    def warmup(self):
        return 0

    def signals(self, candles):
        values = np.zeros(len(candles), dtype=int)
        for index, signal in self.script.items():
            values[index] = int(signal)
        return pd.Series(values, index=candles.index)


def flat_candles(prices):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(prices), freq="h", tz="UTC"),
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": 1.0,
        }
    ).astype({"open": float, "high": float, "low": float, "close": float})


NO_COSTS = FeeSettings(fee_rate=0, slippage_bps=0)
FULL_SIZE = RiskSettings(
    risk_per_trade=0.1, stop_loss_pct=0.1, take_profit_pct=None,
    max_position_fraction=1.0, max_drawdown_pct=1.0,
)


def test_orders_fill_at_the_next_open():
    candles = flat_candles([100, 100, 110, 120, 120])
    strategy = Scripted({0: Signal.BUY, 2: Signal.SELL})
    result = run_backtest(candles, strategy, FULL_SIZE, NO_COSTS, 1000)
    trade = result.trades[0]
    assert trade.entry_price == 100  # decided on candle 0, filled on candle 1
    assert trade.exit_price == 120   # decided on candle 2, filled on candle 3
    assert result.metrics["total_return"] == pytest.approx(0.2)


def test_stop_loss_exits_at_the_stop_price():
    candles = flat_candles([100, 100, 100, 100])
    candles.loc[2, "low"] = 80.0
    result = run_backtest(candles, Scripted({0: Signal.BUY}), FULL_SIZE, NO_COSTS, 1000)
    assert result.trades[0].reason == "stop_loss"
    assert result.trades[0].exit_price == pytest.approx(90)


def test_take_profit_exits_at_the_target():
    risk = FULL_SIZE.model_copy(update={"take_profit_pct": 0.05})
    candles = flat_candles([100, 100, 100, 100])
    candles.loc[2, "high"] = 130.0
    result = run_backtest(candles, Scripted({0: Signal.BUY}), risk, NO_COSTS, 1000)
    assert result.trades[0].reason == "take_profit"
    assert result.trades[0].exit_price == pytest.approx(105)


def test_drawdown_limit_closes_the_position_and_stops():
    # A 20% position falling 30% costs 6% of equity, past the 5% limit.
    risk = FULL_SIZE.model_copy(update={"stop_loss_pct": 0.5, "max_drawdown_pct": 0.05})
    candles = flat_candles([100, 100, 70, 70, 70, 70])
    strategy = Scripted({0: Signal.BUY, 4: Signal.BUY})
    result = run_backtest(candles, strategy, risk, NO_COSTS, 1000)
    assert result.halted
    assert len(result.trades) == 1


def test_equity_never_goes_negative_and_costs_hurt(candles):
    strategy = create_strategy("sma_cross", {"fast": 10, "slow": 30})
    free = run_backtest(candles, strategy, fees=NO_COSTS)
    costly = run_backtest(candles, strategy, fees=FeeSettings(fee_rate=0.002, slippage_bps=20))
    assert (costly.equity > 0).all()
    assert costly.metrics["final_equity"] < free.metrics["final_equity"]
    assert len(costly.equity) == len(costly.timestamps)


def test_ml_backtest_trains_only_on_the_first_part(candles):
    result = run_backtest(candles, create_strategy("ml"))
    assert result.timestamps.iloc[0] == candles["timestamp"].iloc[len(candles) // 2]


def test_too_little_data_is_reported(candles):
    with pytest.raises(ValueError, match="Not enough candles"):
        run_backtest(candles.iloc[:30], create_strategy("sma_cross"))


def test_max_drawdown():
    assert max_drawdown(np.array([100.0, 120.0, 90.0, 130.0])) == pytest.approx(0.25)


def test_recommendations_cover_every_candidate_and_are_ranked(candles):
    ranked = recommend(candles, "conservative")
    assert len(ranked) == len(CANDIDATES)
    suitable = [item.suitable for item in ranked]
    assert suitable == sorted(suitable, reverse=True)
    scores = [item.score for item in ranked if item.suitable]
    assert scores == sorted(scores, reverse=True)
    assert all(item.notes for item in ranked)


def test_unknown_profile_is_rejected(candles):
    with pytest.raises(ValueError):
        recommend(candles, "reckless")
