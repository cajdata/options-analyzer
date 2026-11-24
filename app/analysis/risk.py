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
"""

from __future__ import annotations

import math

from app import pricing
from app.analysis.earnings import spans
from app.models import ContractAnalysis, EarningsEvent, RiskReward

MIN_REPORTS = 4  # past earnings moves needed before they are used to size the next one
RATING_EDGE = 0.10  # expected value beyond +/-10% of the premium is favorable / unfavorable
BAD_CASE_SIGMAS = 2.0
# Lopsided payoffs get a label, because the rating alone hides them. A cheap
# far-out contract can have positive expected value and still lose nearly every
# time; its mirror image for a seller wins often and occasionally loses big.
LOPSIDED = 0.20
SHARES = 100


def history_vol(
    c: ContractAnalysis, ordinary_vol: float | None, event_var: float | None, upcoming: EarningsEvent | None
) -> float | None:
    """Annualized volatility over the contract's life if the stock moves as it has.

    Ordinary days use recent realized volatility with earnings days removed. Each
    earnings report inside the contract's life replaces one ordinary day with the
    variance of past report-day moves. None when a needed input is missing.
    """
    if not ordinary_vol:
        return None
    t = max(c.dte, 0.5) / 365.0
    reports = 0
    if upcoming and spans(c.expiry, upcoming):
        if event_var is None:
            return None  # the contract spans a report and there is no history to size it
        reports = 1 + (c.expiry - upcoming.day).days // 91
    total = ordinary_vol**2 * max(t - reports / 365.0, 0.0) + reports * (event_var or 0.0)
    return math.sqrt(total / t)


def assess(c: ContractAnalysis, spot: float, rate: float, lens: str) -> RiskReward | None:
    """Payoff statistics at expiry for the buyer ("buy") or the seller ("sell") of `c`."""
    vol = c.history_vol
    if not vol:
        return None
    t = max(c.dte, 0.5) / 365.0
    growth = math.exp(rate * t)
    is_call = c.kind == "call"

    # Buyer's view first; the seller's is its mirror image.
    payoff = pricing.price(c.kind, spot, c.strike, t, rate, vol) * growth  # average payoff at expiry
    gains = pricing.price(c.kind, spot, c.breakeven, t, rate, vol) * growth  # average profit, counting losses as 0
    above = pricing.prob_above(spot, c.breakeven, t, rate, vol)
    win_prob = above if is_call else 1.0 - above
    value = payoff - c.mid
    losses = gains - value  # average loss, counting profits as 0
    if not 1e-6 < win_prob < 1.0 - 1e-6 or gains <= 0 or losses <= 0:
        return None

    # Stock price after a two-standard-deviation move against the position.
    shock = BAD_CASE_SIGMAS * vol * math.sqrt(t)
    if lens == "buy":
        adverse = spot * math.exp(-shock if is_call else shock)
        max_loss: float | None = c.mid
    else:
        win_prob, gains, losses, value = 1.0 - win_prob, losses, gains, -value
        adverse = spot * math.exp(shock if is_call else -shock)
        # A short put can lose the strike less the premium. A short call has no cap
        # on its own; in a covered call the shares held make up the difference.
        max_loss = None if is_call else c.strike - c.mid
    intrinsic = max(adverse - c.strike, 0.0) if is_call else max(c.strike - adverse, 0.0)
    bad_case = c.mid - intrinsic if lens == "buy" else intrinsic - c.mid

    avg_win, avg_loss = gains / win_prob, losses / (1.0 - win_prob)
    edge = value / c.mid
    rating = "favorable" if edge >= RATING_EDGE else "unfavorable" if edge <= -RATING_EDGE else "balanced"
    # Judged on the market's own odds, the probability shown beside every contract.
    odds = c.pop_long if lens == "buy" else 1.0 - c.pop_long
    shape = "long shot" if odds < LOPSIDED else "tail risk" if odds > 1.0 - LOPSIDED else None
    return RiskReward(
        history_vol=vol,
        win_prob=win_prob,
        avg_win=avg_win * SHARES,
        avg_loss=avg_loss * SHARES,
        ratio=avg_win / avg_loss,
        expected_value=value * SHARES,
        edge_pct=edge,
        max_loss=None if max_loss is None else max_loss * SHARES,
        bad_case_loss=max(bad_case, 0.0) * SHARES,
        rating=rating,
        shape=shape,
    )
