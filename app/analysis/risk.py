"""Risk and reward: what a contract pays at expiry if the stock moves as it has before.

At the market's own prices this question has a fixed answer: an option is priced
so that expected profit is about zero, which makes the reward-to-risk ratio the
mirror image of the probability of profit (a bigger payoff always comes with
lower odds). So the payoff is measured against an independent yardstick instead:
the stock's own history, meaning its recent volatility outside earnings plus the
size of its past earnings moves. No view on direction is assumed.

Options usually cost more than history suggests. That gap is what sellers are
paid for carrying the risk of a large move, so a favorable rating for a seller
still comes with the bad-case loss shown beside it.

Two conventions keep the numbers honest:

* The premium is the price actually paid or received: a buyer lifts the ask and
  a seller hits the bid. A wide spread is a real cost and counts against the
  contract, and "most you can lose" matches the cost shown for it.
* Everything is in today's dollars. The premium changes hands now and the payoff
  arrives at expiry, so the payoff is discounted back. Without that a fairly
  priced option would show a profit equal to the interest on its premium, which
  grows with time to expiry and would tilt long-dated contracts toward buyers.
"""

from __future__ import annotations

import math
import statistics

from app import pricing
from app.analysis.contracts import TRADING_DAY
from app.analysis.earnings import spans
from app.models import ContractAnalysis, EarningsEvent, Reaction, RiskReward

MIN_REPORTS = 4  # past earnings moves needed before they are used to size the next one
RATING_EDGE = 0.10  # expected value beyond +/-10% of the premium is favorable / unfavorable
BAD_CASE_SIGMAS = 2.0
# Lopsided payoffs get a label, because the rating alone hides them. A cheap
# far-out contract can have positive expected value and still lose nearly every
# time; its mirror image for a seller wins often and occasionally loses big.
LOPSIDED = 0.20
SHARES = 100
# A contract on its last day has hours left, and how many depends on the time of day and on when the
# session ends, which nothing here knows. A history measured in whole days cannot be set beside its
# price: with the half day the pricing formulas are given, the same fairly priced option read
# "unfavorable" at the open and "favorable" by mid-afternoon. So on that day there is no volatility
# from history, and with it no rating and no comparison of volatility in the score. Said once, here.
LAST_DAY = (
    "A contract that expires today is not rated, and the score does not compare its price with the stock's "
    "history. With hours left, both would depend on how much of the day remains, which the scan does not "
    "measure. The implied volatility and the decay shown for it assume half a day is left."
)


def last_day(c: ContractAnalysis) -> bool:
    """Whether a contract expires today, when its price cannot be set beside a history in whole days."""
    return c.dte < 1


def event_variance(moves: list[Reaction], ordinary_vol: float | None) -> float | None:
    """Variance of one report-day move, from past reactions. None with too few reports.

    Log returns, to match how the rest of the volatility is measured. A reaction
    whose timing was unknown is measured across two sessions, so one ordinary
    day's variance is taken back out of it.
    """
    if len(moves) < MIN_REPORTS:
        return None
    daily = (ordinary_vol or 0.0) ** 2 * TRADING_DAY
    return statistics.fmean(max(math.log1p(m.move_pct) ** 2 - (m.sessions - 1) * daily, 0.0) for m in moves)


def history_vol(
    c: ContractAnalysis, ordinary_vol: float | None, event_var: float | None, upcoming: EarningsEvent | None
) -> float | None:
    """Annualized volatility over the contract's life if the stock moves as it has.

    Ordinary days use recent realized volatility with earnings days removed. Each
    earnings report inside the contract's life replaces one ordinary trading day
    with the variance of past report-day moves. None when a needed input is missing,
    and on the contract's last day (`LAST_DAY`).
    """
    if not ordinary_vol or last_day(c):
        return None
    t = c.dte / 365.0
    reports = 0
    if upcoming and spans(c.expiry, upcoming):
        if event_var is None:
            return None  # the contract spans a report and there is no history to size it
        reports = 1 + (c.expiry - upcoming.day).days // 91
    total = ordinary_vol**2 * max(t - reports * TRADING_DAY, 0.0) + reports * (event_var or 0.0)
    return math.sqrt(total / t)


def assess(c: ContractAnalysis, spot: float, rate: float, lens: str) -> RiskReward | None:
    """Payoff statistics for the buyer ("buy") or the seller ("sell") of `c`, in today's dollars.

    None when there is no volatility from history to go by, and on the contract's last day (`LAST_DAY`).
    """
    vol = c.history_vol
    if not vol or last_day(c):
        return None
    t = c.dte / 365.0
    growth = math.exp(rate * t)
    is_call = c.kind == "call"
    premium = c.ask if lens == "buy" else c.bid
    if premium <= 0:
        return None
    # Stock price at expiry where the option's payoff equals the premium carried to expiry.
    breakeven = c.strike + premium * growth if is_call else c.strike - premium * growth
    if breakeven <= 0:
        return None

    # The option buyer's side first; the seller's is its mirror image.
    fair = pricing.price(c.kind, spot, c.strike, t, rate, vol)  # what the payoff is worth if history repeats
    gains = pricing.price(c.kind, spot, breakeven, t, rate, vol)  # average profit, counting losses as 0
    above = pricing.prob_above(spot, breakeven, t, rate, vol)
    win_prob = above if is_call else 1.0 - above
    value = fair - premium
    losses = gains - value  # average loss, counting profits as 0
    # Subtracting nearly equal numbers can leave a meaningless sliver that turns into
    # an enormous ratio, so both sides must be a real fraction of the premium.
    floor = 1e-9 * premium
    if not 1e-6 < win_prob < 1.0 - 1e-6 or gains <= floor or losses <= floor:
        return None

    # Stock price after a two-standard-deviation move against the position.
    shock = BAD_CASE_SIGMAS * vol * math.sqrt(t)
    if lens == "buy":
        adverse = spot * math.exp(-shock if is_call else shock)
        max_loss: float | None = premium
    else:
        win_prob, gains, losses, value = 1.0 - win_prob, losses, gains, -value
        adverse = spot * math.exp(shock if is_call else -shock)
        # A short put can lose the strike less the premium. A short call has no cap
        # on its own; in a covered call the shares held make up the difference.
        max_loss = None if is_call else c.strike - premium
    intrinsic = max(adverse - c.strike, 0.0) if is_call else max(c.strike - adverse, 0.0)
    bad_case = premium - intrinsic if lens == "buy" else intrinsic - premium

    avg_win, avg_loss = gains / win_prob, losses / (1.0 - win_prob)
    ratio = avg_win / avg_loss
    if not math.isfinite(ratio):
        return None
    edge = value / premium
    rating = "favorable" if edge >= RATING_EDGE else "unfavorable" if edge <= -RATING_EDGE else "balanced"
    # Judged on the market's own odds, the probability shown beside every contract.
    odds = c.pop_long if lens == "buy" else 1.0 - c.pop_long
    shape = "long shot" if odds < LOPSIDED else "tail risk" if odds > 1.0 - LOPSIDED else None
    return RiskReward(
        history_vol=vol,
        win_prob=win_prob,
        avg_win=avg_win * SHARES,
        avg_loss=avg_loss * SHARES,
        ratio=ratio,
        expected_value=value * SHARES,
        edge_pct=edge,
        max_loss=None if max_loss is None else max_loss * SHARES,
        bad_case_loss=max(bad_case, 0.0) * SHARES,
        rating=rating,
        shape=shape,
    )
