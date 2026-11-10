"""Technical indicators. Every value uses only data up to and including its own row."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative strength index with Wilder smoothing, from 0 to 100."""
    change = series.diff()
    gain = change.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-change.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    strength = gain / loss.replace(0, np.nan)
    result = 100 - 100 / (1 + strength)
    # No losses in the window means the index is at its maximum.
    return result.where(loss != 0, 100.0).where(gain.notna())


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(series, fast) - ema(series, slow)
    signal_line = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": signal_line, "hist": line - signal_line})


def atr(candles: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average true range: typical size of a candle including gaps."""
    previous_close = candles["close"].shift(1)
    true_range = pd.concat(
        [
            candles["high"] - candles["low"],
            (candles["high"] - previous_close).abs(),
            (candles["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def bollinger(series: pd.Series, period: int = 20, width: float = 2.0) -> pd.DataFrame:
    middle = sma(series, period)
    spread = series.rolling(period, min_periods=period).std(ddof=0) * width
    return pd.DataFrame({"middle": middle, "upper": middle + spread, "lower": middle - spread})
