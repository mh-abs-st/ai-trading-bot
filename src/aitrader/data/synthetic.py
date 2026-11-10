"""Generated price history with trends and changing volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd

TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


def generate_candles(
    bars: int = 3000,
    seed: int = 7,
    timeframe: str = "1h",
    start_price: float = 30_000.0,
) -> pd.DataFrame:
    """Random-walk candles whose drift and volatility shift between regimes."""
    if bars < 2:
        raise ValueError("At least two bars are needed.")

    rng = np.random.default_rng(seed)
    step = TIMEFRAME_SECONDS.get(timeframe, 3600)

    drift = np.empty(bars)
    volatility = np.empty(bars)
    index = 0
    while index < bars:
        length = int(rng.integers(80, 400))
        drift[index : index + length] = rng.normal(0.0, 0.0006)
        volatility[index : index + length] = rng.uniform(0.004, 0.014)
        index += length

    returns = drift + volatility * rng.standard_normal(bars)
    close = start_price * np.exp(np.cumsum(returns))
    open_ = np.concatenate(([start_price], close[:-1]))

    wick = volatility * np.abs(rng.standard_normal((2, bars))) * 0.6
    high = np.maximum(open_, close) * (1 + wick[0])
    low = np.minimum(open_, close) * (1 - wick[1])
    volume = rng.lognormal(mean=4.0, sigma=0.5, size=bars) * (1 + 40 * np.abs(returns))

    timestamps = pd.date_range("2023-01-01", periods=bars, freq=pd.Timedelta(seconds=step), tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
