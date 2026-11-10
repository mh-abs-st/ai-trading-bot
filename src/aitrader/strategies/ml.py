"""Machine-learning strategy: a classifier estimates the chance of a rise."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from aitrader.features.builder import FEATURE_WARMUP, build_features, build_labels
from aitrader.models import Signal
from aitrader.strategies.base import Strategy


class MLStrategy(Strategy):
    name = "ml"
    description = (
        "A gradient-boosting model trained on past candles estimates the probability "
        "that price rises over the next few candles."
    )
    requires_training = True

    def __init__(
        self,
        horizon: int = 6,
        min_return: float = 0.003,
        buy_threshold: float = 0.58,
        sell_threshold: float = 0.45,
        seed: int = 42,
    ):
        if horizon < 1:
            raise ValueError("ml needs horizon >= 1.")
        if not 0 < sell_threshold < buy_threshold < 1:
            raise ValueError("ml needs 0 < sell_threshold < buy_threshold < 1.")
        self.horizon = horizon
        self.min_return = min_return
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.seed = seed
        self._model: HistGradientBoostingClassifier | None = None

    @property
    def warmup(self) -> int:
        return FEATURE_WARMUP

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(self, candles: pd.DataFrame) -> None:
        features = build_features(candles)
        labels = build_labels(candles, self.horizon, self.min_return)
        usable = features.notna().all(axis=1) & labels.notna()

        if usable.sum() < 200:
            raise ValueError("The ml strategy needs at least 200 usable candles to train.")
        if labels[usable].nunique() < 2:
            raise ValueError("Training data has only one outcome; use a longer history.")

        model = HistGradientBoostingClassifier(
            max_depth=3,
            max_iter=150,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=self.seed,
        )
        model.fit(features[usable].to_numpy(), labels[usable].to_numpy())
        self._model = model

    def probabilities(self, candles: pd.DataFrame) -> pd.Series:
        """Estimated chance of a rise per candle; empty where features are incomplete."""
        if self._model is None:
            raise RuntimeError("Call fit() before asking the ml strategy for signals.")

        features = build_features(candles)
        complete = features.notna().all(axis=1).to_numpy()
        result = np.full(len(candles), np.nan)
        if complete.any():
            result[complete] = self._model.predict_proba(features[complete].to_numpy())[:, 1]
        return pd.Series(result, index=candles.index)

    def signals(self, candles: pd.DataFrame) -> pd.Series:
        chance = self.probabilities(candles).to_numpy()
        result = np.full(len(candles), int(Signal.HOLD))
        with np.errstate(invalid="ignore"):
            result[chance >= self.buy_threshold] = int(Signal.BUY)
            result[chance <= self.sell_threshold] = int(Signal.SELL)
        return pd.Series(result, index=candles.index)
