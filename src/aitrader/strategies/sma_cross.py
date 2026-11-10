"""Trend following: trade when a fast moving average crosses a slow one."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aitrader.features.indicators import sma
from aitrader.models import Signal
from aitrader.strategies.base import Strategy


class SmaCrossStrategy(Strategy):
    name = "sma_cross"
    description = "Buys when the fast average rises above the slow one, sells when it falls below."

    def __init__(self, fast: int = 20, slow: int = 50):
        if not 0 < fast < slow:
            raise ValueError("sma_cross needs 0 < fast < slow.")
        self.fast = fast
        self.slow = slow

    @property
    def warmup(self) -> int:
        return self.slow + 1

    def signals(self, candles: pd.DataFrame) -> pd.Series:
        fast = sma(candles["close"], self.fast)
        slow = sma(candles["close"], self.slow)

        above = (fast > slow).to_numpy()
        ready = (fast.notna() & slow.notna()).to_numpy()
        was_above = np.concatenate(([False], above[:-1]))
        was_ready = np.concatenate(([False], ready[:-1]))

        result = np.full(len(candles), int(Signal.HOLD))
        crossing = ready & was_ready
        result[crossing & above & ~was_above] = int(Signal.BUY)
        result[crossing & ~above & was_above] = int(Signal.SELL)
        return pd.Series(result, index=candles.index)
