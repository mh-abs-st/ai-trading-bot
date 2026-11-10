"""The interface every strategy implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Maps candle history to one signal per candle.

    The signal on a row may depend only on that row and earlier ones; it is
    acted on after that candle closes.
    """

    name: str = ""
    description: str = ""
    requires_training: bool = False

    @property
    @abstractmethod
    def warmup(self) -> int:
        """Candles needed before the first meaningful signal."""

    def fit(self, candles: pd.DataFrame) -> None:
        """Learn from history. Rule-based strategies have nothing to learn."""

    @abstractmethod
    def signals(self, candles: pd.DataFrame) -> pd.Series:
        """Signal values (see ``aitrader.models.Signal``) aligned to ``candles``."""
