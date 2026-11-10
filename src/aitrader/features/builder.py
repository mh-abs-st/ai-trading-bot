"""Turns candles into the feature table and labels used by the ML strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aitrader.features import indicators

# Longest look-back used below; rows before this have incomplete features.
FEATURE_WARMUP = 60


def build_features(candles: pd.DataFrame) -> pd.DataFrame:
    """One row of features per candle, all scale-free so they transfer across price levels."""
    close = candles["close"]
    volume = candles["volume"]
    features = pd.DataFrame(index=candles.index)

    for bars in (1, 3, 6, 12, 24):
        features[f"return_{bars}"] = close.pct_change(bars)

    features["rsi_14"] = indicators.rsi(close, 14) / 100
    features["macd_hist"] = indicators.macd(close)["hist"] / close
    features["atr_14"] = indicators.atr(candles, 14) / close

    bands = indicators.bollinger(close, 20)
    band_width = (bands["upper"] - bands["lower"]).replace(0, np.nan)
    features["band_position"] = (close - bands["lower"]) / band_width

    features["trend_20_50"] = indicators.sma(close, 20) / indicators.sma(close, 50) - 1
    features["volatility_24"] = close.pct_change().rolling(24, min_periods=24).std()

    volume_mean = volume.rolling(48, min_periods=48).mean()
    volume_std = volume.rolling(48, min_periods=48).std().replace(0, np.nan)
    features["volume_z"] = (volume - volume_mean) / volume_std

    return features


def build_labels(candles: pd.DataFrame, horizon: int, min_return: float) -> pd.Series:
    """1.0 where price rises more than ``min_return`` over the next ``horizon`` candles.

    The last ``horizon`` rows have no future yet and are left empty.
    """
    close = candles["close"]
    future_return = close.shift(-horizon) / close - 1
    return (future_return > min_return).astype(float).where(future_return.notna())
