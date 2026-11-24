"""Pick symbols, pull their option chains and rank the contracts."""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from app.analysis import contracts as metrics
from app.analysis import earnings as earnings_math
from app.analysis import scoring
from app.cache import DiskCache, NullCache
from app.config import Settings
from app.models import Bar, ContractAnalysis, EarningsEvent, ScanResult, StockQuote, UnderlyingContext
from app.providers import MarketData
from app.providers.earnings import AlphaVantageEarnings, EarningsService, EdgarEarnings

ET = ZoneInfo("America/New_York")
HISTORY_DAYS = 1200  # a little over three years: twelve earnings reports
MAX_REPORTS = 12
MIN_PRICE = 5.0  # discovery skips stocks below this


class ConfigError(RuntimeError):
    pass


@dataclass
class ScanRequest:
    symbols: list[str] = field(default_factory=list)  # empty means discover
    lens: str = "buy"  # "buy" or "sell"
    kind: str = "both"  # "call", "put" or "both"
    dte_min: int = 7
    dte_max: int = 60
    strike_window: float = 0.15  # strikes within this share of the stock price
    max_symbols: int = 20
    top_n: int = 20
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

    def scan(self, req: ScanRequest) -> ScanResult:
        if req.lens not in scoring.LENSES:
            raise ValueError(f"unknown lens {req.lens!r}")
        today = self._today or datetime.now(ET).date()
        warnings: list[str] = []

        reasons = self._universe(req, warnings)
        quotes = self.market.stock_snapshots(list(reasons))
        for symbol in req.symbols:
            if symbol not in quotes:
                warnings.append(f"{symbol}: no stock quote found.")
        calendar = self.earnings.calendar()
        selected = self._select(req, reasons, quotes, calendar, today)
        bars = self.market.daily_bars(selected, today - timedelta(days=HISTORY_DAYS)) if selected else {}

        def analyze(symbol: str) -> tuple[UnderlyingContext, list[ContractAnalysis]] | str:
            try:
                return self._analyze(symbol, quotes[symbol], bars.get(symbol, []), reasons[symbol], req, today)
            except Exception as err:  # one bad symbol must not sink the scan
                return f"{symbol}: {err}"

        underlyings: dict[str, UnderlyingContext] = {}
        scored: list[ContractAnalysis] = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for outcome in pool.map(analyze, selected):
                if isinstance(outcome, str):
                    warnings.append(outcome)
                    continue
                ctx, found = outcome
                underlyings[ctx.symbol] = ctx
                scored.extend(found)
                if ctx.contracts_scanned == 0:
                    warnings.append(f"{ctx.symbol}: no two-sided option quotes in range.")

        if self.earnings.calendar_note:
            warnings.append(self.earnings.calendar_note)
        warnings.extend(dict.fromkeys(self.market.notes))
        self.market.notes.clear()

        return ScanResult(
            id=uuid.uuid4().hex[:12],
            generated_at=datetime.now(timezone.utc),
            mode=self.market.mode,
            options_feed=self.market.options_feed,
            lens=req.lens,
            ideas=scoring.rank(scored, req.per_side or (5 if req.symbols else 1), req.top_n),
            underlyings=underlyings,
            warnings=warnings,
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
            points = 100.0 if "watchlist" in why else 0.0
            points += sum(1.0 for label in why if label != "watchlist")
            listed = calendar.get(symbol)
            if listed and 0 <= (listed.day - today).days <= 21:
                points += 3.0
                why.append(f"earnings in {(listed.day - today).days}d")
            change = quote.change_pct or 0.0
            points += min(abs(change) / 0.02, 3.0)
            if abs(change) >= 0.03:
                why.append(f"moved {change:+.1%} today")
            interest[symbol] = points
        return sorted(interest, key=lambda s: interest[s], reverse=True)[: req.max_symbols]

    # -- One symbol ---------------------------------------------------------

    def _analyze(
        self, symbol: str, quote: StockQuote, bars: list[Bar], reasons: list[str], req: ScanRequest, today: date
    ) -> tuple[UnderlyingContext, list[ContractAnalysis]]:
        spot = quote.price
        window = dict(
            expiry_from=today + timedelta(days=req.dte_min),
            expiry_to=today + timedelta(days=req.dte_max),
            strike_from=spot * (1 - req.strike_window),
            strike_to=spot * (1 + req.strike_window),
        )
        chain = self.market.option_chain(symbol, **window)
        if any(q.open_interest is None for q in chain):
            try:
                interest = self.market.open_interest(symbol, **window)
            except Exception:
                interest = {}
                self.market.notes.append("Open interest was unavailable, so liquidity scores rely on spreads alone.")
            for q in chain:
                if q.open_interest is None:
                    q.open_interest = interest.get(q.symbol)

        rate = self.settings.risk_free_rate
        analyzed = [a for q in chain if (a := metrics.analyze_contract(q, spot, today, rate))]
        atm = metrics.atm_by_expiry(analyzed, spot)
        metrics.apply_expected_moves(analyzed, atm, spot, today, rate)
        realized_20 = metrics.realized_vol(bars, 20)

        history, note = self.earnings.history(symbol)
        upcoming = self.earnings.upcoming(symbol, history)
        moves = earnings_math.reactions(history, bars)[:MAX_REPORTS]
        implied = earnings_math.implied_move(spot, atm, upcoming, realized_20, today) if upcoming and atm else None
        for c in analyzed:
            c.spans_earnings = bool(upcoming and earnings_math.spans(c.expiry, upcoming))

        near_month = min(atm, key=lambda e: abs((e - today).days - 30)) if atm else None
        ctx = UnderlyingContext(
            symbol=symbol,
            price=spot,
            change_pct=quote.change_pct,
            realized_vol_20d=realized_20,
            realized_vol_60d=metrics.realized_vol(bars, 60),
            atm_iv=atm[near_month].iv if near_month else None,
            reasons=reasons,
            next_earnings=upcoming,
            days_to_earnings=(upcoming.day - today).days if upcoming else None,
            reactions=moves,
            earnings_stats=earnings_math.summarize(moves),
            implied_move=implied,
            earnings_note=note,
            contracts_scanned=len(analyzed),
        )

        wanted = [c for c in analyzed if scoring.tradable(c) and req.kind in ("both", c.kind)]
        for c in wanted:
            scoring.score(c, ctx, req.lens)
        return ctx, wanted


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
