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
from app.analysis import events, exits, holdings, risk, scoring
from app.cache import DiskCache, MemoryCache, NullCache
from app.config import ConfigError, Settings, tidy
from app.models import (
    Bar,
    ContractAnalysis,
    EarningsEvent,
    ExitPlan,
    Holding,
    OptionQuote,
    Position,
    PositionsReview,
    Reaction,
    ScanResult,
    StockQuote,
    TermPoint,
    UnderlyingContext,
)
from app.providers import MarketData
from app.providers.earnings import AlphaVantageEarnings, EarningsService, EdgarEarnings
from app.pool import EventPool
from app.providers.news import NewsService

__all__ = ["ConfigError", "PRESETS", "ScanRequest", "Scanner", "build_scanner"]

ET = ZoneInfo("America/New_York")
HISTORY_DAYS = 1200  # a little over three years: twelve earnings reports
WIDEN_SCREEN = 50  # how many of the day's most active stocks and movers a wider pool takes in
WIDEN_BARS = 25  # stocks whose price history is asked for at once when widening the pool
# When more than this share of a stock's headlines never name it, the scan says so: either the
# feed is tagging another company's news with the ticker, or the company goes by another name.
UNNAMED_SHARE = 0.5
UNNAMED_LEAST = 10  # headlines on record before that is said
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
# With "skip earnings" on, an estimated report date is given this much room in front
# of it, since a company's reports move by days from one quarter to the next.
ESTIMATE_MARGIN = timedelta(days=7)

# Named starting points for a scan. Each is a set of ScanRequest fields.
PRESETS: dict[str, dict] = {
    "sell-premium": {
        "label": "Premium selling",
        "about": (
            "The setup most option sellers start from: 30 to 45 days to expiry, a strike "
            "at 0.16 to 0.30 delta, and nothing that is still open over an earnings report."
        ),
        "settings": {
            "lens": "sell",
            "dte_min": 30,
            "dte_max": 45,
            "delta_min": 0.16,
            "delta_max": 0.30,
            "avoid_earnings": True,
        },
    },
}


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
    # Keep only contracts whose delta, ignoring its sign, is inside this range.
    delta_min: float | None = None
    delta_max: float | None = None
    avoid_earnings: bool = False  # drop contracts still open over the next report


@dataclass
class _Found:
    """What one symbol contributed to a scan."""

    ctx: UnderlyingContext
    contracts: list[ContractAnalysis]
    least_outlay: float | None  # cheapest contract seen, before any budget was applied
    left_out: str | None = None  # why skip-earnings removed everything: "unknown" or "all open"
    earnings_days: set[date] = field(default_factory=set)  # sessions that carry an earnings reaction


@dataclass
class _Reports:
    """A stock's earnings reports set beside its prices: what a scan and a held position both start from."""

    upcoming: EarningsEvent | None
    moves: list[Reaction]  # how the stock moved across past reports, newest first
    note: str | None  # set when a source could not answer
    unknown: bool  # the stock reports, and there is no date to go on
    days: set[date]  # sessions that carry an earnings reaction
    ordinary: float | None  # realized volatility with those sessions removed
    event_var: float | None  # variance of past report-day moves


def _names(symbols: list[str], limit: int = 8) -> str:
    """A readable list of symbols, cut short when it is long."""
    symbols = sorted(symbols)
    if len(symbols) <= limit:
        return ", ".join(symbols)
    return ", ".join(symbols[:limit]) + f" and {len(symbols) - limit} more"


class Scanner:
    def __init__(
        self,
        market: MarketData,
        earnings: EarningsService,
        settings: Settings,
        universe: list[str] | None = None,
        today: date | None = None,
        news: NewsService | None = None,
        pool: EventPool | None = None,
    ) -> None:
        self.market = market
        self.earnings = earnings
        self.news = news  # None leaves the news section out of every scan
        self.pool = pool  # None leaves pooled studies out of the news section
        self.settings = settings
        self.universe = universe or []  # symbols a discovery scan always considers
        self._today = today
        self._lock = threading.Lock()

    @property
    def today(self) -> date:
        """The trading date now, unless one was fixed for a test or the demo."""
        return self._today or datetime.now(ET).date()

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
        req = replace(req, symbols=tidy(req.symbols))
        today = self._today or datetime.now(ET).date()
        warnings: list[str] = []
        self.market.notes.clear()

        reasons = self._universe(req, warnings)
        quotes = self.market.stock_snapshots(list(reasons))
        for symbol in req.symbols:
            if symbol not in quotes:
                warnings.append(f"{symbol}: no stock quote found.")
        calendar = self.earnings.calendar()
        selected = self._select(req, reasons, quotes, calendar, today, warnings)
        bars = self.market.daily_bars(selected, today - timedelta(days=HISTORY_DAYS)) if selected else {}

        def analyze(symbol: str) -> _Found | str:
            try:
                return self._analyze(symbol, quotes[symbol], bars.get(symbol, []), reasons[symbol], req, today)
            except Exception as err:  # one bad symbol must not sink the scan
                return f"{symbol}: {err}"

        underlyings: dict[str, UnderlyingContext] = {}
        scored: list[ContractAnalysis] = []
        cheapest: tuple[float, str] | None = None  # least outlay seen, before the budget was applied
        left_out: dict[str, list[str]] = {"unknown": [], "all open": []}
        no_quotes: list[str] = []
        earnings_days: dict[str, set[date]] = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for outcome in pool.map(analyze, selected):
                if isinstance(outcome, str):
                    warnings.append(outcome)
                    continue
                ctx, least = outcome.ctx, outcome.least_outlay
                underlyings[ctx.symbol] = ctx
                earnings_days[ctx.symbol] = outcome.earnings_days
                scored.extend(outcome.contracts)
                if least is not None and (cheapest is None or least < cheapest[0]):
                    cheapest = (least, ctx.symbol)
                if outcome.left_out:
                    left_out[outcome.left_out].append(ctx.symbol)
                if ctx.contracts_scanned == 0:
                    no_quotes.append(ctx.symbol)

        if no_quotes:
            # One note for all of them. The day's top movers include warrants and tiny
            # companies with no options at all, so a discovery scan can have several.
            warnings.append(f"No two-sided option quotes in range for {_names(no_quotes)}.")
        warnings.extend(self._earnings_warnings(req, underlyings, left_out))
        if self.earnings.calendar_note:
            warnings.append(self.earnings.calendar_note)

        ideas = scoring.rank(scored, req.per_side or (5 if req.symbols else 1), req.top_n)
        # News comes last and only for the stocks that made the list, so a stock
        # nobody will look at costs no request. It runs before the provider's notes
        # are collected, because loading headlines can add one.
        shown = list(dict.fromkeys(c.underlying for c in ideas))
        warnings.extend(self._add_news(shown, underlyings, bars, earnings_days, today))
        warnings.extend(dict.fromkeys(self.market.notes))
        self.market.notes.clear()
        if req.budget is not None and not ideas:
            note = f"Nothing fits a ${req.budget:,.0f} budget."
            if cheapest:
                note += f" The cheapest contract that passed the other filters needs ${cheapest[0]:,.0f} ({cheapest[1]})."
            warnings.append(note)

        rate = self.settings.risk_free_rate
        plans = {c.symbol: exits.build(c, underlyings[c.underlying], req.lens, rate, today) for c in ideas}

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
            exit_plans={symbol: plan for symbol, plan in plans.items() if plan},
        )

    def _earnings_warnings(
        self, req: ScanRequest, underlyings: dict[str, UnderlyingContext], left_out: dict[str, list[str]]
    ) -> list[str]:
        """What the person should know about earnings dates the scan could not rely on."""
        out = []
        if req.avoid_earnings:
            # The filter fails closed: a stock appears only if it can be shown clear of a report.
            if left_out["unknown"] and not self.earnings.sources:
                out.append(
                    "Nothing can be shown as clear of an earnings report, because no earnings data source is "
                    "configured. Set SEC_USER_AGENT or ALPHAVANTAGE_API_KEY in .env, or turn off the skip-earnings filter."
                )
            elif left_out["unknown"]:
                out.append(
                    f"Left out {_names(left_out['unknown'])}: no earnings date is known, "
                    "and contracts open over earnings are being skipped."
                )
            if left_out["all open"]:
                out.append(
                    f"Left out {_names(left_out['all open'])}: every contract in this range "
                    "is still open over the next earnings report."
                )
            return out
        unknown = [symbol for symbol, ctx in underlyings.items() if ctx.earnings_unknown]
        if unknown and not self.earnings.sources:
            out.append("No earnings data source is configured, so no rating accounts for an upcoming report.")
        elif unknown:
            out.append(f"No earnings date is known for {_names(unknown)}, so ratings there do not account for a report.")
        return out

    def _add_news(
        self,
        shown: list[str],
        underlyings: dict[str, UnderlyingContext],
        bars: dict[str, list[Bar]],
        earnings_days: dict[str, set[date]],
        today: date,
    ) -> list[str]:
        """Give each stock in `shown` its recent news and what followed events like it.

        For a scan these are the stocks with a ranked contract. For a review they are the stocks held.
        """
        if self.news is None:
            return []

        tallies: dict[str, dict[str, list[int]]] = {}
        listed = self.settings.aliases()  # read for each scan, so a name added to the file counts on the next one

        def load(symbol: str) -> str | None:
            ctx = underlyings[symbol]
            try:
                items, ctx.news_note, failed = self.news.items(symbol, today)
                names, also = self._called(symbol, listed, today)
                found = events.context(items, bars.get(symbol, []), earnings_days.get(symbol, set()), today, names, also)
                ctx.news, ctx.recent_events, ctx.event_studies, ctx.news_elsewhere = found[:4]
                ctx.coverage = found.coverage
                missing, of = found.unnamed
                if of >= UNNAMED_LEAST and missing > of * UNNAMED_SHARE:
                    ctx.naming_note = (
                        f"{missing} of the {of} headlines the feed tags with {symbol} never name "
                        f"{self.news.name(symbol) or 'the company'}, and are set aside. "
                        "If it goes by another name, add that to data/aliases.txt."
                    )
                # A lookup that failed has left part of the news out. A tally built from the rest
                # would take the place of a full one, and could move the stock to the other group.
                if not failed and events.unpoolable(items, bars.get(symbol, []), earnings_days.get(symbol, set())) is None:
                    tallies[symbol] = found.tally
            except Exception as err:  # news is an extra: it must not sink the scan
                ctx.news_note, failed = f"News lookup failed: {type(err).__name__}.", True
            return symbol if failed else None

        with ThreadPoolExecutor(max_workers=4) as workers:
            failed = [symbol for symbol in workers.map(load, shown) if symbol]
        self._pool_studies([underlyings[symbol] for symbol in shown], tallies)
        if not failed:
            return []
        # One note for all of them. A failure is usually the same one for every stock.
        return [f"News could not be loaded for {_names(failed)}. {underlyings[failed[0]].news_note}"]

    def _called(self, symbol: str, listed: dict[str, tuple[str, ...]], today: date) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Who the company is, and what else a headline may call it: a name it had within the price history, a listed brand."""
        since = today - timedelta(days=HISTORY_DAYS)
        also = events.also_called((*self.news.former_names(symbol, since), *listed.get(symbol, ())))
        return events.aliases(symbol, self.news.name(symbol)), also

    def _pool_studies(self, shown: list[UnderlyingContext], tallies: dict[str, dict[str, list[int]]]) -> None:
        """Add what this scan's stocks bring to the pool, then give each its pooled studies.

        A stock is pooled with the others covered about as much as it is, itself among
        them. Like the rest of the news, this is an extra and must not sink a scan.
        """
        if self.pool is None:
            return
        try:
            for ctx in shown:
                if ctx.coverage and ctx.symbol in tallies:
                    self.pool.add(ctx.symbol, ctx.coverage, tallies[ctx.symbol])
            totals = self.pool.totals()
            for ctx in shown:
                if ctx.coverage:
                    ctx.pooled_studies = events.pooled(totals, ctx.coverage, ctx.recent_events)
        except Exception:  # a pool that cannot be read leaves the stock's own study as it was
            return

    def widen_pool(self, symbols: list[str] | None = None, progress=None) -> dict:
        """Add stocks to the pool that nobody has scanned, so a pooled study rests on more of them.

        With no `symbols`, these are the universe and the day's most active stocks and
        movers. For each one its news is read (and kept, as for a scan), its report days
        are taken from what is on file or from the first earnings source, and its price
        history is fetched. A stock is passed over when a news lookup failed or when
        `events.unpoolable` gives a reason, the same test a scan applies.

        Returns how many stocks were asked for, how many were added to each group, and
        why the rest were not.
        """
        if self.news is None or self.pool is None:
            raise ConfigError("Pooling needs a news source. None is configured.")
        with self._lock:
            today = self._today or datetime.now(ET).date()
            wanted = list(symbols or self.universe)
            if not symbols:
                for fetch in (self.market.most_actives, self.market.movers):
                    try:
                        wanted += fetch(WIDEN_SCREEN)
                    except Exception:  # screeners are optional extras
                        continue
            wanted = tidy(wanted)

            added = {group: 0 for group in events.GROUPS}
            passed_over: dict[str, int] = {}
            listed = self.settings.aliases()

            def skip(why: str) -> None:
                passed_over[why] = passed_over.get(why, 0) + 1

            ready: dict[str, tuple[list, list[EarningsEvent], tuple[tuple[str, ...], tuple[str, ...]]]] = {}
            for n, symbol in enumerate(wanted, 1):
                if progress:
                    progress(n, len(wanted), symbol)
                try:
                    items, _, failed = self.news.items(symbol, today)
                except Exception:
                    items, failed = [], True
                if failed:
                    # Part of its news is missing, so whatever is already on record for it stays.
                    skip("a news lookup that failed")
                    continue
                if not any(i.kind == "headline" for i in items):
                    skip("no headlines")  # said here, before a request for report days is spent on it
                    continue
                try:
                    reports = self.earnings.reports(symbol)
                except Exception:
                    reports = []
                if not reports:
                    skip("no report days")
                    continue
                ready[symbol] = (items, reports, self._called(symbol, listed, today))

            batch = list(ready)
            for i in range(0, len(batch), WIDEN_BARS):
                group_of = batch[i : i + WIDEN_BARS]
                try:
                    bars = self.market.daily_bars(group_of, today - timedelta(days=HISTORY_DAYS))
                except Exception:
                    bars = {}
                for symbol in group_of:
                    items, reports, (names, also) = ready[symbol]
                    history = bars.get(symbol) or []
                    report_days = earnings_math.reaction_days(reports, history)
                    why = events.unpoolable(items, history, report_days)
                    if why:
                        skip(why)
                        continue
                    found = events.context(items, history, report_days, today, names, also)
                    self.pool.add(symbol, found.coverage, found.tally)
                    added[found.coverage] += 1
            self.market.notes.clear()  # nobody is shown a scan here, so its notes have no reader
            return {"asked": len(wanted), "added": added, "passed_over": passed_over, "totals": self.pool.totals()}

    # -- Positions already held ---------------------------------------------

    def review(self, positions: list[Position]) -> PositionsReview:
        """Set each position beside today's quotes and check it against the exit rules.

        The held contracts are asked for by name, in one request however many
        stocks they are on, so no chain is pulled. Each stock held then costs what
        it costs a scan: its price, its price history, its earnings dates and its
        news.

        Every position comes back, priced or not. A source that fails leaves its
        numbers out and adds a warning, and a check that needed them says it could
        not be made (`analysis/holdings.py`).
        """
        # One at a time with scans, for the same reason: the provider and its notes are shared.
        with self._lock:
            return self._review(positions)

    def _review(self, positions: list[Position]) -> PositionsReview:
        today = self.today
        rate = self.settings.risk_free_rate
        warnings: list[str] = []
        self.market.notes.clear()
        current = [p for p in positions if p.expiry >= today]  # an expired contract is no longer quoted
        symbols = list(dict.fromkeys(p.underlying for p in current))

        def fetch(what: str, call, nothing):
            try:
                return call(), True
            except Exception as err:  # the positions are still shown, with whatever could be read
                warnings.append(f"{what} could not be loaded: {err}")
                return nothing, False

        quotes: dict[str, StockQuote] = {}
        option_quotes: dict[str, OptionQuote] = {}
        bars: dict[str, list[Bar]] = {}
        options_answered = True
        if symbols:
            quotes, answered = fetch("Stock prices", lambda: self.market.stock_snapshots(symbols), {})
            if answered:
                warnings.extend(f"{symbol}: no stock quote found." for symbol in symbols if symbol not in quotes)
            wanted = {p.symbol: p.underlying for p in current}
            option_quotes, options_answered = fetch("Option quotes", lambda: self.market.option_quotes(wanted), {})
            priced = [symbol for symbol in symbols if symbol in quotes]
            if priced:
                start = today - timedelta(days=HISTORY_DAYS)
                bars, _ = fetch("Price history", lambda: self.market.daily_bars(priced, start), {})

        def context(symbol: str) -> tuple[UnderlyingContext, _Reports, str | None] | str:
            try:
                quote, history = quotes[symbol], bars.get(symbol, [])
                failed = None
                try:
                    reports = self._reports(symbol, history)
                except Exception as err:
                    # The stock's price is in hand, and its earnings dates failing takes nothing from
                    # it. What is lost is the dates: no report can be ruled out, and the context says so.
                    # The first version dropped the whole stock here and told the person it had no price.
                    failed = f"{symbol}: its earnings dates could not be looked up ({err})."
                    ordinary = metrics.realized_vol(history, 60) or metrics.realized_vol(history, 20)
                    lost = f"Earnings lookup failed: {type(err).__name__}."
                    reports = _Reports(None, [], lost, True, set(), ordinary, None)
                ctx = UnderlyingContext(
                    symbol=symbol,
                    price=quote.price,
                    change_pct=quote.change_pct,
                    realized_vol_20d=metrics.realized_vol(history, 20),
                    realized_vol_60d=metrics.realized_vol(history, 60),
                    atm_iv=None,  # that needs the chain, which a review does not pull
                    reasons=["held"],
                    next_earnings=reports.upcoming,
                    days_to_earnings=(reports.upcoming.day - today).days if reports.upcoming else None,
                    reactions=reports.moves,
                    earnings_stats=earnings_math.summarize(reports.moves),
                    earnings_note=reports.note,
                    earnings_unknown=reports.unknown,
                    ordinary_vol=reports.ordinary,
                    event_var=reports.event_var,
                )
                return ctx, reports, failed
            except Exception as err:  # one stock's lookups must not hide the other positions
                return f"{symbol}: {err}"

        underlyings: dict[str, UnderlyingContext] = {}
        reports_for: dict[str, _Reports] = {}
        dates_looked_up = False
        with ThreadPoolExecutor(max_workers=4) as pool:
            for outcome in pool.map(context, [symbol for symbol in symbols if symbol in quotes]):
                if isinstance(outcome, str):
                    warnings.append(outcome)
                    continue
                ctx, reports, failed = outcome
                underlyings[ctx.symbol], reports_for[ctx.symbol] = ctx, reports
                if failed:
                    warnings.append(failed)
                else:
                    dates_looked_up = True
        # Only when a stock's dates were looked up just now, to the end. The calendar is the last
        # thing a lookup asks, so one that failed on the way never reached it, and a note left by an
        # earlier scan says nothing about this review.
        if dates_looked_up and self.earnings.calendar_note:
            warnings.append(self.earnings.calendar_note)

        held: list[Holding] = []
        for p in positions:
            ctx, quote = underlyings.get(p.underlying), option_quotes.get(p.symbol)
            contract = plan = note = None
            if p.expiry < today:
                pass  # the one check an expired position gets says all there is to say
            elif quote is None:
                note = (
                    "No quote was found for this contract. Check its expiry and strike."
                    if options_answered
                    else "Option quotes could not be loaded, so this position is not priced."
                )
            elif quote.outdated:
                when = quote.quote_time.astimezone(ET) if quote.quote_time else None
                day = f"on {when:%b} {when.day}" if when else "on an earlier day"
                note = f"This contract was last quoted {day}. A quote from an earlier day is not used, so the position is not priced."
            elif ctx is None:
                # Say which it is: no price for the stock, or a price and a lookup that failed after it.
                note = (
                    f"{p.underlying} could not be looked up, so the Greeks and the path ahead are left out."
                    if p.underlying in quotes
                    else f"There is no stock price for {p.underlying}, so the Greeks and the path ahead are left out."
                )
            else:
                contract = metrics.analyze_contract(quote, ctx.price, today, rate)
                if contract is None:
                    note = (
                        "This quote cannot be priced: it has no bid, or it sits below what exercising the option "
                        "would pay at the stock's price. The Greeks and the path ahead are left out."
                    )
                else:
                    plan = self._held_plan(p, contract, ctx, reports_for[p.underlying], rate, today)
                    ctx.contracts_scanned += 1
            # The strike is set against the stock's price whenever there is one, whatever else failed.
            spot = quotes[p.underlying].price if p.underlying in quotes else None
            held.append(
                holdings.assess(
                    p, today, spot=spot, quote=quote, ctx=ctx, contract=contract, plan=plan, margin=ESTIMATE_MARGIN, note=note
                )
            )

        # News for every stock held, flagged events above all. As in a scan it runs
        # before the provider's notes are collected, because loading headlines can add one.
        earnings_days = {symbol: reports.days for symbol, reports in reports_for.items()}
        warnings.extend(self._add_news(list(underlyings), underlyings, bars, earnings_days, today))
        warnings.extend(dict.fromkeys(self.market.notes))
        self.market.notes.clear()
        return PositionsReview(
            generated_at=datetime.now(timezone.utc),
            mode=self.market.mode,
            options_feed=self.market.options_feed,
            holdings=held,
            underlyings=underlyings,
            warnings=warnings,
        )

    @staticmethod
    def _held_plan(
        p: Position, c: ContractAnalysis, ctx: UnderlyingContext, reports: _Reports, rate: float, today: date
    ) -> ExitPlan | None:
        """The standing-still path from here for a held contract, measured against the price it was opened at."""
        c.lens = holdings.LENS[p.side]
        c.spans_earnings = bool(reports.upcoming and earnings_math.spans(c.expiry, reports.upcoming))
        c.history_vol = risk.history_vol(c, reports.ordinary, reports.event_var, reports.upcoming)
        plan = exits.build(c, ctx, c.lens, rate, today, entry=p.entry_price)
        if plan and plan.earnings:
            # What the options price for the report takes the chain around it to read, and
            # a review reads one contract. So the step is the size past reports were.
            plan.notes.append(
                "The step across earnings is sized from this stock's past reports. What the options "
                "price in for it takes the whole chain to read, and only this contract's quote was."
            )
        return plan

    def _reports(self, symbol: str, bars: list[Bar]) -> _Reports:
        """A stock's next report, how it moved across past ones, and its ordinary volatility without them."""
        history, note = self.earnings.history(symbol)
        upcoming = self.earnings.upcoming(symbol, history)
        moves = earnings_math.reactions(history, bars)[:MAX_REPORTS]
        # A stock that reports, with no date to go on: the lookup failed, nothing is
        # configured, the only source cannot answer for this company, or the reports
        # on file are too old to estimate from. A date from the calendar is a date,
        # even when the history lookup behind it failed.
        unknown = upcoming is None and (note is not None or bool(history))
        # The stock's own history: ordinary days with earnings reactions removed,
        # and the typical size of a report-day move.
        days = earnings_math.reaction_days(history, bars)
        ordinary = metrics.realized_vol(bars, 60, days) or metrics.realized_vol(bars, 20, days)
        return _Reports(upcoming, moves, note, unknown, days, ordinary, risk.event_variance(moves, ordinary))

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
        warnings: list[str],
    ) -> list[str]:
        if req.symbols:
            return [s for s in req.symbols if s in quotes]
        interest: dict[str, float] = {}
        skipped: list[str] = []
        first_expiry = today + timedelta(days=req.dte_min)
        for symbol, why in reasons.items():
            quote = quotes.get(symbol)
            if quote is None or quote.price < MIN_PRICE:
                continue
            listed = calendar.get(symbol)
            days = (listed.day - today).days if listed else None
            if req.avoid_earnings and listed and days >= 0 and earnings_math.spans(first_expiry, listed):
                # Even the nearest expiry in range is still open over the report, so every
                # contract would be thrown away. Scanning it would waste one of the few slots.
                skipped.append(symbol)
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
            if listed and 0 <= days <= 21:
                if not req.avoid_earnings:
                    points += 3.0  # a report soon is a reason to look, unless reports are being avoided
                why.append("earnings today" if days == 0 else f"earnings in {days}d")
            change = quote.change_pct or 0.0
            points += min(abs(change) / 0.02, 3.0)
            if abs(change) >= 0.03:
                why.append(f"moved {change:+.1%} today")
            interest[symbol] = points
        if skipped:
            warnings.append(f"Skipped {_names(skipped)}: they report before any contract in this range expires.")
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
    ) -> _Found:
        spot = quote.price
        reach = self._strike_window(req, bars)
        window = dict(
            expiry_from=today + timedelta(days=req.dte_min),
            expiry_to=today + timedelta(days=req.dte_max),
            strike_from=spot * (1 - reach),
            strike_to=spot * (1 + reach),
        )
        chain = self.market.option_chain(symbol, **window)
        rate = self.settings.risk_free_rate
        analyzed = [a for q in chain if (a := metrics.analyze_contract(q, spot, today, rate))]
        if not analyzed:
            # Nothing can be priced, so nothing will be shown for this stock. Stop here,
            # before the open-interest and earnings lookups. The day's movers include
            # warrants and tiny companies with no options, and an earnings lookup for
            # one can cost one of the few requests Alpha Vantage allows in a day.
            ctx = UnderlyingContext(
                symbol=symbol,
                price=spot,
                change_pct=quote.change_pct,
                realized_vol_20d=metrics.realized_vol(bars, 20),
                realized_vol_60d=metrics.realized_vol(bars, 60),
                atm_iv=None,
                reasons=reasons,
            )
            return _Found(ctx, [], None)

        if any(c.open_interest is None for c in analyzed):
            try:
                interest = self.market.open_interest(symbol, **window)
            except Exception as err:
                interest = {}
                note = "Open interest was unavailable, so liquidity scores rely on spreads alone."
                if getattr(err, "status", None) in (401, 403):
                    # Market data accepts paper and live keys alike; this request does not.
                    note += " The trading API rejected the keys: check that ALPACA_PAPER matches the keys you use."
                self.market.notes.append(note)
            for c in analyzed:
                if c.open_interest is None:
                    c.open_interest = interest.get(c.symbol)

        atm = metrics.atm_by_expiry(analyzed, spot)
        metrics.apply_expected_moves(analyzed, atm, spot, today, rate)

        reports = self._reports(symbol, bars)
        upcoming, moves, note, unknown = reports.upcoming, reports.moves, reports.note, reports.unknown
        earnings_days, ordinary, event_var = reports.days, reports.ordinary, reports.event_var

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
            term_structure=[
                TermPoint(expiry, (expiry - today).days, atm[expiry].iv, bool(upcoming and earnings_math.spans(expiry, upcoming)))
                for expiry in sorted(atm)
            ],
            ordinary_vol=ordinary,
            event_var=event_var,
        )

        low = req.delta_min if req.delta_min is not None else 0.0
        high = req.delta_max if req.delta_max is not None else 1.0
        wanted = [
            c for c in analyzed if scoring.tradable(c) and req.kind in ("both", c.kind) and low <= abs(c.delta) <= high
        ]
        left_out = None
        if req.avoid_earnings and wanted:
            # Skipping earnings fails closed. With no date to go on, nothing can be shown
            # as clear of a report; and an estimated date gets room in front of it.
            def may_be_open(c: ContractAnalysis) -> bool:
                if c.spans_earnings:
                    return True
                return bool(upcoming and not upcoming.confirmed and c.expiry >= upcoming.day - ESTIMATE_MARGIN)

            wanted = [] if unknown else [c for c in wanted if not may_be_open(c)]
            if not wanted:
                left_out = "unknown" if unknown else "all open"
        for c in wanted:
            c.outlay = scoring.outlay(c, req.lens, spot)
        least = min((c.outlay for c in wanted), default=None)
        if req.budget is not None:
            wanted = [c for c in wanted if c.outlay <= req.budget]
        for c in wanted:
            scoring.score(c, req.lens)
            c.risk_reward = risk.assess(c, spot, rate, req.lens)
        return _Found(ctx, wanted, least, left_out, earnings_days)


def build_scanner(settings: Settings, demo: bool = False) -> Scanner:
    """Wire up data sources from settings. `demo` uses synthetic data and no keys."""
    if demo:
        from app.providers.demo import NAMES, SPECS, DemoMarket

        market = DemoMarket()
        earnings = EarningsService(NullCache(), history_sources=[market], calendar_source=market, today=market.today)
        news = NewsService(NullCache(), headline_source=market, filing_source=market, names=NAMES.get)
        # The demo's pool is kept in memory, never on disk, and starts with every demo stock in it.
        scanner = Scanner(market, earnings, settings, universe=list(SPECS), today=market.today, news=news, pool=EventPool(MemoryCache()))
        scanner.widen_pool(list(SPECS))
        return scanner

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
    edgar = EdgarEarnings(settings.sec_user_agent, cache) if settings.sec_user_agent else None
    if edgar:
        history_sources.append(edgar)
    if settings.alphavantage_key:
        calendar_source = AlphaVantageEarnings(settings.alphavantage_key)
        history_sources.append(calendar_source)
    earnings = EarningsService(cache, history_sources=history_sources, calendar_source=calendar_source)
    # Headlines come with the Alpaca keys. Filings, and the company names that tell a
    # buyer from the company being bought, need the same SEC contact line as earnings history.
    news = NewsService(
        cache,
        headline_source=market,
        filing_source=edgar,
        names=edgar.company_name if edgar else None,
        former_names=edgar.former_names if edgar else None,
    )
    return Scanner(market, earnings, settings, universe=settings.universe(), news=news, pool=EventPool(cache))
