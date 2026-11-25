"""Plain data types shared by providers, analysis and the API."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass(frozen=True)
class Bar:
    day: date
    close: float
    volume: float = 0.0


@dataclass(frozen=True)
class StockQuote:
    symbol: str
    price: float
    prev_close: float | None = None

    @property
    def change_pct(self) -> float | None:
        if not self.prev_close:
            return None
        return self.price / self.prev_close - 1.0


@dataclass
class OptionQuote:
    """One contract as a data provider reports it. Missing values are None."""

    symbol: str
    underlying: str
    kind: str  # "call" or "put"
    strike: float
    expiry: date
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    volume: int | None = None
    open_interest: int | None = None
    iv: float | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    quote_time: datetime | None = None


@dataclass(frozen=True)
class EarningsEvent:
    day: date
    timing: str = "unknown"  # "pre", "post" or "unknown"
    surprise_pct: float | None = None
    confirmed: bool = True  # False when the date is an estimate from history
    source: str = ""


@dataclass(frozen=True)
class Reaction:
    """How the stock moved across one past earnings report."""

    day: date
    timing: str
    move_pct: float
    surprise_pct: float | None = None
    sessions: int = 1  # trading days the move spans: 2 when the report's timing was unknown


@dataclass(frozen=True)
class EarningsStats:
    count: int
    mean_abs: float
    median_abs: float
    max_abs: float
    up: int
    down: int


@dataclass(frozen=True)
class ImpliedMove:
    """What the options price in for the next earnings report."""

    expiry: date
    straddle_pct: float | None  # at-the-money straddle to `expiry`, as a share of spot
    event_move_pct: float | None  # one-day move attributed to the report itself
    baseline_vol: float
    baseline_source: str  # where ordinary volatility was taken from, as a phrase
    event_var: float | None = None  # variance the options attribute to the report itself


@dataclass(frozen=True)
class RiskReward:
    """Payoff at expiry if the stock is as volatile as its own history.

    Dollar figures are per contract of 100 shares.
    """

    history_vol: float
    win_prob: float
    avg_win: float
    avg_loss: float
    ratio: float  # average win divided by average loss
    expected_value: float
    edge_pct: float  # expected value as a share of the premium
    max_loss: float | None  # None: no cap on the option alone (a covered call is backed by shares)
    bad_case_loss: float  # loss after a two-standard-deviation move against the position
    rating: str  # "favorable", "balanced" or "unfavorable"
    # Judged on the market-implied odds. "long shot": profits less than one time in
    # five. "tail risk": profits more than four times in five, with an occasional
    # much larger loss. None otherwise.
    shape: str | None = None


@dataclass
class ContractAnalysis:
    symbol: str
    underlying: str
    kind: str
    strike: float
    expiry: date
    dte: int
    bid: float
    ask: float
    mid: float
    spread_pct: float
    volume: int | None
    open_interest: int | None
    iv: float
    iv_source: str  # "provider" or "computed"
    delta: float
    gamma: float
    theta: float
    vega: float
    greeks_source: str
    breakeven: float
    breakeven_move_pct: float
    expected_move_pct: float | None
    breakeven_vs_expected: float | None
    pop_long: float
    prob_itm: float
    theta_pct_per_day: float
    annualized_yield: float
    spans_earnings: bool = False
    history_vol: float | None = None  # volatility over the contract's life if history repeats
    risk_reward: RiskReward | None = None
    outlay: float | None = None  # cash needed to open one contract under the scan's lens
    lens: str = ""
    pop: float | None = None
    score: float | None = None
    components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TermPoint:
    """At-the-money implied volatility for one expiration."""

    expiry: date
    dte: int
    atm_iv: float
    spans_earnings: bool


@dataclass(frozen=True)
class Checkpoint:
    """A contract on a later date, with the stock where it is today."""

    day: date
    days_held: int
    label: str
    value: float  # modelled mid price per share
    close_price: float  # what closing fetches (the bid, for a holder) or costs (the ask, to buy back)
    pnl: float  # dollars per contract against the entry price
    pnl_pct: float  # as a share of the premium paid or collected


@dataclass(frozen=True)
class EarningsExit:
    """The step in a contract's price across an earnings report, stock unchanged."""

    report_day: date  # first session that reflects the report
    value_before: float
    value_after: float
    crush_pct: float  # value_after / value_before - 1
    # Stock move the session after, in the direction that raises the option's value,
    # at which closing is back to the entry price. None if no plausible move gets there.
    move_to_breakeven_pct: float | None


@dataclass
class ExitPlan:
    """What time does to one contract if the stock stands still. Built by analysis/exits.py."""

    lens: str
    entry_price: float  # the ask for a buyer, the bid for a seller, per share
    checkpoints: list[Checkpoint]
    earnings: EarningsExit | None
    strike_distance_pct: float  # strike / stock price - 1
    time_stop_day: date | None  # the time-stop rule's date, if still ahead
    notes: list[str] = field(default_factory=list)
    half_life_day: date | None = None  # buyer: when half the time value is gone
    target_price: float | None = None  # seller: buy back at or below this to hit the profit target
    target_day: date | None = None  # seller: when decay alone gets there, if it does before expiry


@dataclass(frozen=True)
class NewsItem:
    """A headline or an SEC filing about one stock."""

    when: datetime  # published, or accepted by the SEC
    kind: str  # "headline" or "filing"
    title: str
    source: str
    url: str | None = None
    ref: str = ""  # the source's own id for it, used to drop repeats
    items: tuple[str, ...] = ()  # a filing's 8-K item codes
    event: str = ""  # event type, set by analysis/events.py


@dataclass(frozen=True)
class EventMove:
    """How the stock moved across one past event."""

    day: date  # the first session whose close reflects it
    title: str
    move_pct: float  # the close before to that session's close
    after_pct: float | None  # that close to the close some sessions later; None if too recent


@dataclass(frozen=True)
class EventStudy:
    """What followed one stock's past events of one type. Built by analysis/events.py."""

    event: str
    label: str
    count: int  # sessions that carried an event of this type, earnings days left out
    since: date  # the first of them
    mean_abs: float  # average size of the move on those sessions
    up: int
    down: int
    # Average size of the move on any session over the same stretch, earnings days
    # left out. None when there is none to compare with.
    ordinary_abs: float | None
    after_count: int  # events old enough to have the sessions after them on record
    mean_abs_after: float | None  # average size of the move over those following sessions
    ordinary_abs_after: float | None  # the same measure over every session in the stretch
    moves: list[EventMove] = field(default_factory=list)  # newest first


@dataclass
class UnderlyingContext:
    symbol: str
    price: float
    change_pct: float | None
    realized_vol_20d: float | None
    realized_vol_60d: float | None
    atm_iv: float | None
    reasons: list[str] = field(default_factory=list)
    next_earnings: EarningsEvent | None = None
    days_to_earnings: int | None = None
    reactions: list[Reaction] = field(default_factory=list)
    earnings_stats: EarningsStats | None = None
    implied_move: ImpliedMove | None = None
    earnings_note: str | None = None
    # True when the stock may report inside a contract's life but no date is known
    # (no data source, a failed lookup, or a history too old to estimate from), so
    # ratings for it cannot account for a report.
    earnings_unknown: bool = False
    contracts_scanned: int = 0
    # Inputs for repricing a contract on a later date (see analysis/exits.py).
    term_structure: list[TermPoint] = field(default_factory=list)
    ordinary_vol: float | None = None  # realized volatility with earnings days removed
    event_var: float | None = None  # variance of past report-day moves
    # News, filled in only for a stock with a ranked contract (see analysis/events.py).
    news: list[NewsItem] = field(default_factory=list)  # the latest items, newest first
    recent_events: dict[str, int] = field(default_factory=dict)  # event type to how many recent items
    event_studies: dict[str, EventStudy] = field(default_factory=dict)  # by event type
    news_note: str | None = None  # set when a source could not answer


@dataclass
class ScanResult:
    id: str
    generated_at: datetime
    mode: str  # "demo" or "live"
    options_feed: str
    lens: str
    ideas: list[ContractAnalysis]
    underlyings: dict[str, UnderlyingContext]
    warnings: list[str] = field(default_factory=list)
    budget: float | None = None  # per-contract limit the scan was run with
    exit_plans: dict[str, ExitPlan] = field(default_factory=dict)  # by contract symbol
