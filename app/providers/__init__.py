"""Data sources. Everything here is read-only: nothing places or changes orders."""

from __future__ import annotations

from datetime import date
from typing import Protocol

from app.models import Bar, OptionQuote, StockQuote


class MarketData(Protocol):
    mode: str  # "live" or "demo"
    options_feed: str
    notes: list[str]  # things the user should know, e.g. a feed fallback

    def stock_snapshots(self, symbols: list[str]) -> dict[str, StockQuote]: ...

    def daily_bars(self, symbols: list[str], start: date) -> dict[str, list[Bar]]: ...

    def option_chain(
        self, symbol: str, *, expiry_from: date, expiry_to: date, strike_from: float, strike_to: float
    ) -> list[OptionQuote]: ...

    def open_interest(
        self, symbol: str, *, expiry_from: date, expiry_to: date, strike_from: float, strike_to: float
    ) -> dict[str, int]: ...

    def most_actives(self, top: int) -> list[str]: ...

    def movers(self, top: int) -> list[str]: ...
