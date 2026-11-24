"""Per-contract metrics: pricing, breakeven, probabilities, decay."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import date

from app import pricing
from app.models import Bar, ContractAnalysis, OptionQuote


# Volatility is annualized over 252 trading days, so one trading day carries this
# share of a year's variance. An earnings report replaces one such day.
TRADING_DAY = 1.0 / 252.0


def years_to_expiry(expiry: date, today: date) -> float:
    # Half a day on expiry day itself keeps the formulas finite.
    return max((expiry - today).days, 0.5) / 365.0


def analyze_contract(q: OptionQuote, spot: float, today: date, rate: float) -> ContractAnalysis | None:
    """Turn a raw quote into metrics. None when the quote cannot be priced."""
    if not q.bid or not q.ask or q.bid <= 0 or q.ask < q.bid:
        return None
    dte = (q.expiry - today).days
    if dte < 0 or spot <= 0 or q.strike <= 0:
        return None
    t = years_to_expiry(q.expiry, today)
    mid = (q.bid + q.ask) / 2.0
    is_call = q.kind == "call"
    # An option is always worth at least what exercising it now would pay. A quote
    # below that is stale against the stock price (common on a delayed feed), and
    # it would otherwise look like free money. This holds whether or not the feed
    # supplies its own volatility figure.
    intrinsic = max(spot - q.strike, 0.0) if is_call else max(q.strike - spot, 0.0)
    breakeven = q.strike + mid if is_call else q.strike - mid
    if mid < intrinsic or breakeven <= 0:
        return None

    if q.iv and q.iv > 0:
        iv, iv_source = q.iv, "provider"
    else:
        iv, iv_source = pricing.implied_vol(q.kind, mid, spot, q.strike, t, rate), "computed"
    if iv is None:
        return None

    if None not in (q.delta, q.gamma, q.theta, q.vega):
        delta, gamma, theta, vega, greeks_source = q.delta, q.gamma, q.theta, q.vega, "provider"
    else:
        g = pricing.greeks(q.kind, spot, q.strike, t, rate, iv)
        delta, gamma, theta, vega, greeks_source = g.delta, g.gamma, g.theta, g.vega, "computed"

    above_breakeven = pricing.prob_above(spot, breakeven, t, rate, iv)
    above_strike = pricing.prob_above(spot, q.strike, t, rate, iv)
    # Selling a put ties up the strike in cash; a covered call ties up the shares.
    collateral = spot if is_call else q.strike

    return ContractAnalysis(
        symbol=q.symbol,
        underlying=q.underlying,
        kind=q.kind,
        strike=q.strike,
        expiry=q.expiry,
        dte=dte,
        bid=q.bid,
        ask=q.ask,
        mid=mid,
        spread_pct=(q.ask - q.bid) / mid,
        volume=q.volume,
        open_interest=q.open_interest,
        iv=iv,
        iv_source=iv_source,
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        greeks_source=greeks_source,
        breakeven=breakeven,
        breakeven_move_pct=breakeven / spot - 1.0,
        expected_move_pct=None,
        breakeven_vs_expected=None,
        pop_long=above_breakeven if is_call else 1.0 - above_breakeven,
        prob_itm=above_strike if is_call else 1.0 - above_strike,
        # Decay is value lost per day. A deep in-the-money put can gain with time
        # instead (positive theta), which is not decay.
        theta_pct_per_day=max(-theta, 0.0) / mid,
        annualized_yield=mid / collateral * 365.0 / max(dte, 1),
    )


@dataclass(frozen=True)
class AtmPoint:
    strike: float
    iv: float
    straddle: float | None  # call mid + put mid; None if only one side quoted


def atm_by_expiry(contracts: list[ContractAnalysis], spot: float) -> dict[date, AtmPoint]:
    """At-the-money volatility and straddle price for each expiration."""
    by_expiry: dict[date, dict[float, dict[str, ContractAnalysis]]] = {}
    for c in contracts:
        by_expiry.setdefault(c.expiry, {}).setdefault(c.strike, {})[c.kind] = c

    points: dict[date, AtmPoint] = {}
    for expiry, strikes in by_expiry.items():
        paired = [k for k, sides in strikes.items() if len(sides) == 2]
        strike = min(paired or strikes, key=lambda k: abs(k - spot))
        sides = strikes[strike]
        iv = statistics.fmean(c.iv for c in sides.values())
        straddle = sides["call"].mid + sides["put"].mid if len(sides) == 2 else None
        points[expiry] = AtmPoint(strike=strike, iv=iv, straddle=straddle)
    return points


def apply_expected_moves(
    contracts: list[ContractAnalysis], atm: dict[date, AtmPoint], spot: float, today: date, rate: float
) -> None:
    """Fill in the one-standard-deviation move to expiry and how the breakeven compares.

    The comparison measures the breakeven from the forward price (spot grown at
    the risk-free rate), which is where option prices are centered. Measuring
    from spot would make every put look closer to its breakeven than the
    matching call. Only a move the stock still has to make counts: a breakeven
    already on the favorable side of the forward is a distance of zero.
    """
    for c in contracts:
        point = atm.get(c.expiry)
        if point is None:
            continue
        t = years_to_expiry(c.expiry, today)
        expected = point.iv * math.sqrt(t)
        if expected > 0:
            forward = spot * math.exp(rate * t)
            c.expected_move_pct = expected
            needed = c.breakeven / forward - 1.0 if c.kind == "call" else 1.0 - c.breakeven / forward
            c.breakeven_vs_expected = max(needed, 0.0) / expected


def realized_vol(bars: list[Bar], window: int, exclude: frozenset[date] | set[date] = frozenset()) -> float | None:
    """Annualized close-to-close volatility over the last `window` daily returns.

    Returns landing on a day in `exclude` (earnings reactions) are left out, so
    the result describes ordinary days only.
    """
    returns = [
        math.log(b.close / a.close)
        for a, b in zip(bars, bars[1:])
        if a.close > 0 and b.close > 0 and b.day not in exclude
    ][-window:]
    if len(returns) < window:
        return None
    return statistics.stdev(returns) * math.sqrt(252.0)
