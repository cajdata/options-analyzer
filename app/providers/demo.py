"""Synthetic market data so the app runs without any API keys.

Every price, quote and earnings date here is generated, not real. The tickers
are made up. Output is repeatable for a given date.
"""

from __future__ import annotations

import math
import random
from datetime import date, timedelta

from app import occ, pricing
from app.analysis.earnings import spans
from app.models import Bar, EarningsEvent, OptionQuote, StockQuote

RATE = 0.04

# vol: ordinary annualized volatility. jump: size of a typical earnings move.
# pricing: how the options price the next report against that history
# (below 1 = cheaper than past moves, above 1 = richer).
SPECS: dict[str, dict] = {
    "ACME": dict(price=184.00, change=0.004, vol=0.28, jump=0.055, earnings_in=9, timing="post", pricing=0.70),
    "BOLT": dict(price=62.50, change=0.064, vol=0.55, jump=0.110, earnings_in=4, timing="post", pricing=1.40),
    "CRUX": dict(price=412.00, change=-0.003, vol=0.22, jump=0.040, earnings_in=38, timing="pre", pricing=1.00),
    "EMBR": dict(price=27.80, change=-0.052, vol=0.48, jump=0.090, earnings_in=71, timing="post", pricing=1.00),
    "FLUX": dict(price=96.20, change=0.011, vol=0.35, jump=0.070, earnings_in=16, timing="pre", pricing=1.10),
    "ORBT": dict(price=143.30, change=0.001, vol=0.18, jump=0.030, earnings_in=55, timing="post", pricing=0.90),
}
NO_PROVIDER_GREEKS = {"EMBR"}  # exercises the app's own volatility and Greeks math


def _weekday(day: date) -> date:
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


class DemoMarket:
    mode = "demo"
    options_feed = "synthetic"

    def __init__(self, today: date | None = None) -> None:
        self.today = today or date.today()
        self.notes: list[str] = []
        self._bar_cache: dict[str, list[Bar]] = {}

    # -- Earnings (also serves as the demo history and calendar source) -----

    name = "demo data"

    def next_event(self, symbol: str) -> EarningsEvent:
        spec = SPECS[symbol]
        day = _weekday(self.today + timedelta(days=spec["earnings_in"]))
        return EarningsEvent(day=day, timing=spec["timing"], source=self.name)

    def history(self, symbol: str) -> list[EarningsEvent]:
        if symbol not in SPECS:
            return []
        upcoming, rng = self.next_event(symbol), random.Random(f"{symbol}-surprise")
        return [
            EarningsEvent(
                day=_weekday(upcoming.day - timedelta(days=91 * n)),
                timing=upcoming.timing,
                surprise_pct=round(rng.gauss(2.0, 6.0), 1),
                source=self.name,
            )
            for n in range(1, 14)
        ]

    def calendar(self, today: date | None = None) -> dict[str, EarningsEvent]:
        # Like a real calendar, only near-term dates are listed; the rest get estimated.
        events = {symbol: self.next_event(symbol) for symbol in SPECS}
        return {s: e for s, e in events.items() if (e.day - self.today).days <= 45}

    # -- Stocks -------------------------------------------------------------

    def stock_snapshots(self, symbols: list[str]) -> dict[str, StockQuote]:
        out = {}
        for symbol in symbols:
            spec = SPECS.get(symbol)
            if spec:
                out[symbol] = StockQuote(symbol, spec["price"], round(spec["price"] / (1 + spec["change"]), 2))
        return out

    def daily_bars(self, symbols: list[str], start: date) -> dict[str, list[Bar]]:
        return {s: [b for b in self._bars(s) if b.day >= start] for s in symbols if s in SPECS}

    def _bars(self, symbol: str) -> list[Bar]:
        if symbol in self._bar_cache:
            return self._bar_cache[symbol]
        spec, rng = SPECS[symbol], random.Random(f"{symbol}-bars")
        reaction_days = set()
        for event in self.history(symbol):
            day = event.day if event.timing == "pre" else _weekday(event.day + timedelta(days=1))
            reaction_days.add(day)

        daily, level, path = spec["vol"] / math.sqrt(252.0), 0.0, []
        day = self.today - timedelta(days=1300)
        while day < self.today:
            if day.weekday() < 5:
                level += rng.gauss(0.0, daily)
                if day in reaction_days:
                    level += rng.gauss(0.0, spec["jump"])
                path.append((day, level))
            day += timedelta(days=1)
        # Shift the whole path so it ends at yesterday's close.
        shift = math.log(spec["price"] / (1 + spec["change"])) - path[-1][1]
        bars = [Bar(d, round(math.exp(lvl + shift), 2), 1_000_000.0) for d, lvl in path]
        self._bar_cache[symbol] = bars
        return bars

    # -- Options ------------------------------------------------------------

    def option_chain(
        self, symbol: str, *, expiry_from: date, expiry_to: date, strike_from: float, strike_to: float
    ) -> list[OptionQuote]:
        spec = SPECS.get(symbol)
        if spec is None:
            return []
        spot, upcoming, rng = spec["price"], self.next_event(symbol), random.Random(f"{symbol}-chain")
        step = 0.5 if spot < 30 else 1.0 if spot < 75 else 2.5 if spot < 200 else 5.0
        strikes = [n * step for n in range(math.ceil(strike_from / step), math.floor(strike_to / step) + 1)]
        base, event_std = spec["vol"] * 1.08, spec["jump"] * spec["pricing"]

        quotes: list[OptionQuote] = []
        expiry = expiry_from + timedelta(days=(4 - expiry_from.weekday()) % 7)  # first Friday
        while expiry <= expiry_to:
            t = max((expiry - self.today).days, 0.5) / 365.0
            for strike in strikes:
                moneyness = math.log(strike / spot)
                skewed = base * min(1.8, max(0.6, 1.0 - 0.6 * moneyness))
                event_var = event_std**2 if spans(expiry, upcoming) else 0.0
                iv = math.sqrt((skewed * skewed * t + event_var) / t)
                interest = int(4000 * math.exp(-((moneyness / 0.07) ** 2)) * (0.5 + rng.random()))
                for kind in ("call", "put"):
                    mid = pricing.price(kind, spot, strike, t, RATE, iv)
                    half = max(0.01, mid * (0.012 + 0.5 * abs(moneyness)))
                    bid = math.floor((mid - half) * 100) / 100
                    if bid < 0.01:
                        continue
                    quote = OptionQuote(
                        symbol=occ.build(symbol, expiry, kind, strike),
                        underlying=symbol,
                        kind=kind,
                        strike=strike,
                        expiry=expiry,
                        bid=bid,
                        ask=math.ceil((mid + half) * 100) / 100,
                        last=round(mid, 2),
                        volume=int(interest * 0.2 * rng.random()),
                        open_interest=interest,
                    )
                    if symbol not in NO_PROVIDER_GREEKS:
                        g = pricing.greeks(kind, spot, strike, t, RATE, iv)
                        quote.iv, quote.delta, quote.gamma, quote.theta, quote.vega = iv, g.delta, g.gamma, g.theta, g.vega
                    quotes.append(quote)
            expiry += timedelta(days=7)
        return quotes

    def open_interest(self, symbol: str, **_: object) -> dict[str, int]:
        return {}  # already on each quote

    # -- Discovery ----------------------------------------------------------

    def most_actives(self, top: int) -> list[str]:
        return ["BOLT", "ACME", "FLUX"][:top]

    def movers(self, top: int) -> list[str]:
        ranked = sorted(SPECS, key=lambda s: abs(SPECS[s]["change"]), reverse=True)
        return ranked[: min(top, 2)]
