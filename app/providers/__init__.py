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

    def close_on(self, symbol: str, day: date) -> float | None:
        """A stock's close on one day, as it printed that day. None when that day has no final bar.

        `daily_bars` is adjusted for every dividend and split since, which is right
        for measuring how a stock moves and wrong for setting a close beside a
        strike: the strike is not lowered with it. This one is not adjusted.
        """
        ...

    def option_chain(
        self, symbol: str, *, expiry_from: date, expiry_to: date, strike_from: float, strike_to: float
    ) -> list[OptionQuote]: ...

    def open_interest(
        self, symbol: str, *, expiry_from: date, expiry_to: date, strike_from: float, strike_to: float
    ) -> dict[str, int]: ...

    def option_quotes(self, contracts: dict[str, str]) -> dict[str, OptionQuote]:
        """Quotes for named contracts, for positions already held: no chain is pulled.

        `contracts` maps each OCC symbol to its underlying. A contract the
        provider has no quote for is left out of the answer. One whose latest
        quote is from an earlier day comes back marked `outdated`, with the
        time of that quote and no prices.
        """
        ...

    def most_actives(self, top: int) -> list[str]: ...

    def movers(self, top: int) -> list[str]: ...
