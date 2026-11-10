"""Placeholder for real exchange orders."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Credentials:
    api_key: str
    api_secret: str

    @classmethod
    def from_env(cls) -> "Credentials":
        """Read keys from the environment so they never sit in a config file."""
        key = os.environ.get("EXCHANGE_API_KEY", "")
        secret = os.environ.get("EXCHANGE_API_SECRET", "")
        if not key or not secret:
            raise RuntimeError("Set EXCHANGE_API_KEY and EXCHANGE_API_SECRET to trade live.")
        return cls(key, secret)


class LiveBroker:
    """Real orders on an exchange. Not implemented: the bot only paper trades for now."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Live trading is not implemented yet. Use mode: paper in the config."
        )
