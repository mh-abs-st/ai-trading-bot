import numpy as np
import pandas as pd

from aitrader.features import indicators


def test_sma_matches_hand_calculation():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = indicators.sma(series, 3)
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == 2.0
    assert result.iloc[4] == 4.0


def test_rsi_extremes():
    rising = pd.Series(np.arange(1.0, 40.0))
    falling = pd.Series(np.arange(40.0, 1.0, -1.0))
    assert indicators.rsi(rising, 14).iloc[-1] == 100.0
    assert indicators.rsi(falling, 14).iloc[-1] < 1.0


def test_rsi_stays_in_range(candles):
    value = indicators.rsi(candles["close"], 14).dropna()
    assert value.between(0, 100).all()


def test_bollinger_bands_bracket_the_middle(candles):
    bands = indicators.bollinger(candles["close"], 20).dropna()
    assert (bands["upper"] >= bands["middle"]).all()
    assert (bands["lower"] <= bands["middle"]).all()


def test_atr_is_positive(candles):
    assert (indicators.atr(candles, 14).dropna() > 0).all()
