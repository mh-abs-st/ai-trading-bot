import numpy as np
import pytest

from aitrader.models import Signal
from aitrader.strategies import STRATEGIES, create_strategy


def test_registry_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown strategy"):
        create_strategy("does_not_exist")


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        create_strategy("sma_cross", {"fast": 50, "slow": 20})


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_signals_are_valid_and_aligned(name, candles):
    strategy = create_strategy(name)
    strategy.fit(candles.iloc[:1500])
    signals = strategy.signals(candles)
    assert len(signals) == len(candles)
    assert set(np.unique(signals)) <= {int(s) for s in Signal}


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_signals_do_not_look_ahead(name, candles):
    """A signal must not change when later candles are added."""
    strategy = create_strategy(name)
    strategy.fit(candles.iloc[:1500])
    full = strategy.signals(candles).to_numpy()
    for cut in (1600, 2000, 2400):
        partial = strategy.signals(candles.iloc[:cut]).to_numpy()
        assert np.array_equal(partial, full[:cut])


def test_sma_cross_produces_both_sides(candles):
    signals = create_strategy("sma_cross", {"fast": 10, "slow": 30}).signals(candles)
    assert (signals == Signal.BUY).any()
    assert (signals == Signal.SELL).any()


def test_ml_requires_fit(candles):
    with pytest.raises(RuntimeError):
        create_strategy("ml").signals(candles)
