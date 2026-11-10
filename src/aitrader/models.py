"""Small value types shared across the package."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import pandas as pd

CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class Signal(IntEnum):
    """What a strategy wants to do after a candle closes."""

    SELL = -1
    HOLD = 0
    BUY = 1


@dataclass(frozen=True)
class Fill:
    """One executed order."""

    timestamp: pd.Timestamp
    side: str
    quantity: float
    price: float
    fee: float


@dataclass(frozen=True)
class Trade:
    """A completed round trip: one buy and the sell that closed it."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float
    reason: str
