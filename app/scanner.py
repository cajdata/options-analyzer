"""Pick symbols, pull their option chains and rank the contracts."""

from __future__ import annotations

import math
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from app.analysis import contracts as metrics
from app.analysis import earnings as earnings_math
from app.analysis import risk, scoring
from app.cache import DiskCache, NullCache
from app.config import ConfigError, Settings
from app.models import Bar, ContractAnalysis, EarningsEvent, ScanResult, StockQuote, UnderlyingContext
from app.providers import MarketData
from app.providers.earnings import AlphaVantageEarnings, EarningsService, EdgarEarnings

__all__ = ["ConfigError", "ScanRequest", "Scanner", "build_scanner"]

ET = ZoneInfo("America/New_York")
HISTORY_DAYS = 1200  # a little over three years: twelve earnings reports
MAX_REPORTS = 12
MIN_PRICE = 5.0  # discovery skips stocks below this
# Rule of thumb for discovery under a budget: a one-month at-the-money option costs
# about 4% of the share price, and out-of-the-money ones down to about 40% of that.
ATM_MONTH_COST = 0.04
BUDGET_REACH = 2.5
# How far from the stock price to look for strikes: at least `strike_window`, and
# wider for volatile stocks and long expiries, up to half the price either way.
WINDOW_SIGMAS = 1.5
MAX_STRIKE_WINDOW = 0.50
DEFAULT_VOL = 0.40  # assumed when a stock has too little price history to measure


@dataclass
class ScanRequest:
    symbols: list[str] = field(default_factory=list)  # empty means discover
    lens: str = "buy"  # "buy" or "sell"
    kind: str = "both"  # "call", "put" or "both"
    dte_min: int = 7
    dte_max: int = 60
    strike_window: float = 0.15  # least share of the stock price to search on each side
    max_symbols: int = 20
    top_n: int = 20
    # Most cash to put into one contract: the ask x 100 for a buyer, the collateral
    # for a seller. None means no limit.
    budget: float | None = None
    # Calls and puts kept per stock. None: one of each when discovering (a varied
    # list), five of each when specific symbols were asked for (depth).
    per_side: int | None = None


class Scanner:
    def __init__(
        self,
        market: MarketData,
        earnings: EarningsService,
        settings: Settings,
        universe: list[str] | None = None,
        today: date | None = None,
    ) -> None:
        self.market = market
        self.earnings = earnings
        self.settings = settings
        self.universe = universe or []  # symbols a discovery scan always considers
        self._today = today
        self._lock = threading.Lock()

    def scan(self, req: ScanRequest) -> ScanResult:
        if req.lens not in scoring.LENSES:
            raise ValueError(f"unknown lens {req.lens!r}")
        # One scan at a time. The data provider and its notes are shared, so two
        # scans running together would mix up each other's notes and compete for
        # the provider's request limit.
        with self._lock:
            return self._scan(req)

    def _scan(self, req: ScanRequest) -> ScanResult:
        # Tidy the symbols here so every way in (command line, API) is treated alike.
        # Share classes are written both ways (BRK-B, BRK.B); the data provider wants the dot.
        tidy = (s.strip().upper().replace("-", ".") for s in req.symbols)
        req = replace(req, symbols=list(dict.fromkeys(s for s in tidy if s)))
        today = self._today or datetime.now(ET).date()
        warnings: list[str] = []
        self.market.notes.clear()

        reasons = self._universe(req, warnings)
        quotes = self.market.stock_snapshots(list(reasons))
        for symbol in req.symbols:
            if symbol not in quotes:
                warnings.append(f"{symbol}: no stock quote found.")
        calendar = self.earnings.calendar()
        selected = self._select(req, reasons, quotes, calendar, today)
        bars = self.market.daily_bars(selected, today - timedelta(days=HISTORY_DAYS)) if selected else {}

        def analyze(symbol: str) -> tuple[UnderlyingContext, list[ContractAnalysis], float | None] | str:
            try:
                return self._analyze(symbol, quotes[symbol], bars.get(symbol, []), reasons[symbol], req, today)
            except Exception as err:  # one bad symbol must not sink the scan
                return f"{symbol}: {err}"

        underlyings: dict[str, UnderlyingContext] = {}
        scored: list[ContractAnalysis] = []
        cheapest: tuple[float, str] | None = None  # least outlay seen, before the budget was applied
        with ThreadPoolExecutor(max_workers=4) as pool:
            for outcome in pool.map(analyze, selected):
                if isinstance(outcome, str):
                    warnings.append(outcome)
                    continue
                ctx, found, least = outcome
                underlyings[ctx.symbol] = ctx
                scored.extend(found)
                if least is not None and (cheapest is None or least < cheapest[0]):
                    cheapest = (least, ctx.symbol)
                if ctx.contracts_scanned == 0:
                    warnings.append(f"{ctx.symbol}: no two-sided option quotes in range.")

        unknown = sorted(symbol for symbol, ctx in underlyings.items() if ctx.earnings_unknown)
        if unknown and not self.earnings.sources:
            warnings.append("No earnings data source is configured, so no rating accounts for an upcoming report.")
        elif unknown:
            warnings.append(
                f"No earnings date is known for {', '.join(unknown)}, so ratings there do not account for a report."
            )
        if self.earnings.calendar_note:
            warnings.append(self.earnings.calendar_note)
        warnings.extend(dict.fromkeys(self.market.notes))
        self.market.notes.clear()

        ideas = scoring.rank(scored, req.per_side or (5 if req.symbols else 1), req.top_n)
        if req.budget is not None and not ideas:
            note = f"Nothing fits a ${req.budget:,.0f} budget."
            if cheapest:
                note += f" The cheapest contract that passed the other filters needs ${cheapest[0]:,.0f} ({cheapest[1]})."
            warnings.append(note)

        return ScanResult(
            id=uuid.uuid4().hex[:12],
            generated_at=datetime.now(timezone.utc),
            mode=self.market.mode,
            options_feed=self.market.options_feed,
            lens=req.lens,
            ideas=ideas,
            underlyings=underlyings,
            warnings=warnings,
            budget=req.budget,
        )

    # -- Choosing symbols ---------------------------------------------------

    def _universe(self, req: ScanRequest, warnings: list[str]) -> dict[str, list[str]]:
        """Candidate symbols mapped to why each one is a candidate."""
        if req.symbols:
            return {symbol: ["requested"] for symbol in req.symbols}
        reasons: dict[str, list[str]] = {}
        for symbol in self.settings.watchlist:
            reasons.setdefault(symbol, []).append("watchlist")
        for label, fetch in (("most active today", self.market.most_actives), ("top mover today", self.market.movers)):
            try:
                for symbol in fetch(20):
                    reasons.setdefault(symbol, []).append(label)
            except Exception as err:  # screeners are optional extras
                warnings.append(f"Could not load the '{label}' list: {err}")
        for symbol in self.universe:
            reasons.setdefault(symbol, [])
        return reasons

    def _select(
        self,
        req: ScanRequest,
        reasons: dict[str, list[str]],
        quotes: dict[str, StockQuote],
        calendar: dict[str, EarningsEvent],
        today: date,
    ) -> list[str]:
        if req.symbols:
            return [s for s in req.symbols if s in quotes]
        interest: dict[str, float] = {}
        for symbol, why in reasons.items():
            quote = quotes.get(symbol)
            if quote is None or quote.price < MIN_PRICE:
                continue
            if req.budget is not None and req.lens == "sell":
                # Least collateral any contract in range could need: 100 shares for a
                # covered call, or the lowest strike the scan could reach for a put.
                least = quote.price if req.kind == "call" else quote.price * (1 - MAX_STRIKE_WINDOW)
                if least * scoring.SHARES > req.budget:
                    continue
            points = 100.0 if "watchlist" in why else 0.0
            points += sum(1.0 for label in why if label != "watchlist")
            if req.budget is not None and req.lens == "buy":
                # Lean toward stocks whose near-the-money contracts are within reach.
                if quote.price * ATM_MONTH_COST * scoring.SHARES <= req.budget * BUDGET_REACH:
                    points += 3.0
            listed = calendar.get(symbol)
            if listed and 0 <= (listed.day - today).days <= 21:
                points += 3.0
                days = (listed.day - today).days
                why.append("earnings today" if days == 0 else f"earnings in {days}d")
            change = quote.change_pct or 0.0
            points += min(abs(change) / 0.02, 3.0)
            if abs(change) >= 0.03:
                why.append(f"moved {change:+.1%} today")
            interest[symbol] = points
        return sorted(interest, key=lambda s: interest[s], reverse=True)[: req.max_symbols]

    # -- One symbol ---------------------------------------------------------

    @staticmethod
    def _strike_window(req: ScanRequest, bars: list[Bar]) -> float:
        """Share of the stock price to search on each side of it.

        A fixed 15% is too narrow for volatile stocks and long expiries. The
        strikes a seller wants (0.15 to 0.35 delta) sit about one standard
        deviation out, which can be 30% of the price or more. So the window grows
        with the stock's volatility and the longest expiry in the scan.
        """
        vol = metrics.realized_vol(bars, 60) or metrics.realized_vol(bars, 20) or DEFAULT_VOL
        reach = WINDOW_SIGMAS * vol * math.sqrt(req.dte_max / 365.0)
        return max(req.strike_window, min(MAX_STRIKE_WINDOW, reach))

    def _analyze(
        self, symbol: str, quote: StockQuote, bars: list[Bar], reasons: list[str], req: ScanRequest, today: date
    ) -> tuple[UnderlyingContext, list[ContractAnalysis], float | None]:
        spot = quote.price
        reach = self._strike_window(req, bars)
        window = dict(
            expiry_from=today + timedelta(days=req.dte_min),
            expiry_to=today + timedelta(days=req.dte_max),
            strike_from=spot * (1 - reach),
            strike_to=spot * (1 + reach),
        )
        chain = self.market.option_chain(symbol, **window)
        if any(q.open_interest is None for q in chain):
            try:
                interest = self.market.open_interest(symbol, **window)
            except Exception as err:
                interest = {}
                note = "Open interest was unavailable, so liquidity scores rely on spreads alone."
                if getattr(err, "status", None) in (401, 403):
                    # Market data accepts paper and live keys alike; this request does not.
                    note += " The trading API rejected the keys: check that ALPACA_PAPER matches the keys you use."
                self.market.notes.append(note)
            for q in chain:
                if q.open_interest is None:
                    q.open_interest = interest.get(q.symbol)

        rate = self.settings.risk_free_rate
        analyzed = [a for q in chain if (a := metrics.analyze_contract(q, spot, today, rate))]
        atm = metrics.atm_by_expiry(analyzed, spot)
        metrics.apply_expected_moves(analyzed, atm, spot, today, rate)

        history, note = self.earnings.history(symbol)
        upcoming = self.earnings.upcoming(symbol, history)
        moves = earnings_math.reactions(history, bars)[:MAX_REPORTS]
        # A stock that reports, with no date to go on: the lookup failed, nothing is
        # configured, the only source cannot answer for this company, or the reports
        # on file are too old to estimate from.
        unknown = note is not None or (bool(history) and upcoming is None)

        # The stock's own history: ordinary days with earnings reactions removed,
        # and the typical size of a report-day move.
        earnings_days = earnings_math.reaction_days(history, bars)
        ordinary = metrics.realized_vol(bars, 60, earnings_days) or metrics.realized_vol(bars, 20, earnings_days)
        event_var = risk.event_variance(moves, ordinary)

        implied = earnings_math.implied_move(spot, atm, upcoming, ordinary, today) if upcoming and atm else None
        for c in analyzed:
            c.spans_earnings = bool(upcoming and earnings_math.spans(c.expiry, upcoming))
            c.history_vol = risk.history_vol(c, ordinary, event_var, upcoming)

        near_month = min(atm, key=lambda e: abs((e - today).days - 30)) if atm else None
        ctx = UnderlyingContext(
            symbol=symbol,
            price=spot,
            change_pct=quote.change_pct,
            realized_vol_20d=metrics.realized_vol(bars, 20),
            realized_vol_60d=metrics.realized_vol(bars, 60),
            atm_iv=atm[near_month].iv if near_month else None,
            reasons=reasons,
            next_earnings=upcoming,
            days_to_earnings=(upcoming.day - today).days if upcoming else None,
            reactions=moves,
            earnings_stats=earnings_math.summarize(moves),
            implied_move=implied,
            earnings_note=note,
            earnings_unknown=unknown,
            contracts_scanned=len(analyzed),
        )

        wanted = [c for c in analyzed if scoring.tradable(c) and req.kind in ("both", c.kind)]
        for c in wanted:
            c.outlay = scoring.outlay(c, req.lens, spot)
        least = min((c.outlay for c in wanted), default=None)
        if req.budget is not None:
            wanted = [c for c in wanted if c.outlay <= req.budget]
        for c in wanted:
            scoring.score(c, req.lens)
            c.risk_reward = risk.assess(c, spot, rate, req.lens)
        return ctx, wanted, least


def build_scanner(settings: Settings, demo: bool = False) -> Scanner:
    """Wire up data sources from settings. `demo` uses synthetic data and no keys."""
    if demo:
        from app.providers.demo import SPECS, DemoMarket

        market = DemoMarket()
        earnings = EarningsService(NullCache(), history_sources=[market], calendar_source=market, today=market.today)
        return Scanner(market, earnings, settings, universe=list(SPECS), today=market.today)

    if not settings.has_alpaca:
        raise ConfigError(
            "Alpaca keys are not set. Copy .env.example to .env and add ALPACA_API_KEY and "
            "ALPACA_SECRET_KEY, or run with --demo to try the app on synthetic data."
        )
    from app.providers.alpaca import AlpacaData

    market = AlpacaData(
        settings.alpaca_key,
        settings.alpaca_secret,
        options_feed=settings.options_feed,
        stock_feed=settings.stock_feed,
        paper=settings.alpaca_paper,
    )
    cache = DiskCache(settings.cache_dir)
    history_sources, calendar_source = [], None
    if settings.sec_user_agent:
        history_sources.append(EdgarEarnings(settings.sec_user_agent, cache))
    if settings.alphavantage_key:
        calendar_source = AlphaVantageEarnings(settings.alphavantage_key)
        history_sources.append(calendar_source)
    earnings = EarningsService(cache, history_sources=history_sources, calendar_source=calendar_source)
    return Scanner(market, earnings, settings, universe=settings.universe())
