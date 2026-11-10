"""Where candles come from: a CSV file, generated data, or Binance's public API."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import pandas as pd
import requests

from aitrader.data.synthetic import generate_candles
from aitrader.models import CANDLE_COLUMNS


class MarketDataSource(Protocol):
    """Returns the most recent closed candles, oldest first."""

    def fetch(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame: ...


def validate_candles(candles: pd.DataFrame) -> pd.DataFrame:
    """Check columns and ordering, and return a clean copy with a fresh index."""
    missing = [column for column in CANDLE_COLUMNS if column not in candles.columns]
    if missing:
        raise ValueError(f"Candle data is missing columns: {', '.join(missing)}")

    clean = candles[CANDLE_COLUMNS].copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True)
    for column in CANDLE_COLUMNS[1:]:
        clean[column] = pd.to_numeric(clean[column], errors="raise").astype(float)

    clean = clean.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    if clean[CANDLE_COLUMNS[1:]].isna().any().any():
        raise ValueError("Candle data contains empty prices or volumes.")
    return clean


def load_csv(path: str | Path) -> pd.DataFrame:
    return validate_candles(pd.read_csv(path))


def save_csv(candles: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    candles.to_csv(path, index=False)


class CsvSource:
    """Candles from a file. Useful for backtests and repeatable runs."""

    def __init__(self, path: str | Path):
        self._candles = load_csv(path)

    def fetch(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        return self._candles.tail(limit).reset_index(drop=True)


class SyntheticSource:
    """Generated candles, for demos and tests that must not touch the network."""

    def __init__(self, bars: int = 3000, seed: int = 7):
        self._bars = bars
        self._seed = seed

    def fetch(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        candles = generate_candles(self._bars, seed=self._seed, timeframe=timeframe)
        return candles.tail(limit).reset_index(drop=True)


def parse_klines(rows: list[list]) -> pd.DataFrame:
    """Convert Binance kline rows into the candle table used everywhere else."""
    if not rows:
        return pd.DataFrame(columns=CANDLE_COLUMNS)

    table = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([row[0] for row in rows], unit="ms", utc=True),
            "open": [float(row[1]) for row in rows],
            "high": [float(row[2]) for row in rows],
            "low": [float(row[3]) for row in rows],
            "close": [float(row[4]) for row in rows],
            "volume": [float(row[5]) for row in rows],
        }
    )
    return validate_candles(table)


class BinanceSource:
    """Public market data from Binance. Needs no account or API key."""

    PAGE_SIZE = 1000

    def __init__(
        self,
        base_url: str = "https://api.binance.com",
        session: requests.Session | None = None,
        timeout: float = 10.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._timeout = timeout

    def fetch(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        rows: list[list] = []
        end_time: int | None = None

        # Page backwards from now until enough candles are collected.
        while len(rows) < limit:
            wanted = min(self.PAGE_SIZE, limit - len(rows))
            params: dict[str, str | int] = {"symbol": symbol, "interval": timeframe, "limit": wanted}
            if end_time is not None:
                params["endTime"] = end_time

            response = self._session.get(
                f"{self._base_url}/api/v3/klines", params=params, timeout=self._timeout
            )
            response.raise_for_status()
            page = response.json()
            if not page:
                break

            rows = page + rows
            end_time = int(page[0][0]) - 1
            if len(page) < wanted:
                break

        # Field 6 is the close time; a candle that has not closed yet is still changing.
        now_ms = int(time.time() * 1000)
        closed = [row for row in rows if int(row[6]) <= now_ms]
        return parse_klines(closed)
