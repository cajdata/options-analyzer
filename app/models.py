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
    straddle_pct: float  # at-the-money straddle to `expiry`, as a share of spot
    event_move_pct: float | None  # one-day move attributed to the report itself
    baseline_vol: float
    baseline_source: str  # where ordinary volatility was taken from, as a phrase


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
    lens: str = ""
    pop: float | None = None
    score: float | None = None
    components: dict[str, float] = field(default_factory=dict)


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
    contracts_scanned: int = 0


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
