"""Mean reversion: buy after a sharp sell-off, sell after a sharp rally."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aitrader.features.indicators import rsi
from aitrader.models import Signal
from aitrader.strategies.base import Strategy


class RsiReversionStrategy(Strategy):
    name = "rsi_reversion"
    description = "Buys when RSI shows the market is oversold, sells when it is overbought."

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        if period < 2 or not 0 < oversold < overbought < 100:
            raise ValueError("rsi_reversion needs period >= 2 and 0 < oversold < overbought < 100.")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    @property
    def warmup(self) -> int:
        return self.period * 3

    def signals(self, candles: pd.DataFrame) -> pd.Series:
        value = rsi(candles["close"], self.period).to_numpy()
        result = np.full(len(candles), int(Signal.HOLD))
        with np.errstate(invalid="ignore"):
            result[value < self.oversold] = int(Signal.BUY)
            result[value > self.overbought] = int(Signal.SELL)
        return pd.Series(result, index=candles.index)
