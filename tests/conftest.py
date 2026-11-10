import pytest

from aitrader.data.synthetic import generate_candles


@pytest.fixture(scope="session")
def candles():
    return generate_candles(bars=2500, seed=11)
