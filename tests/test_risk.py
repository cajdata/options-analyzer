"""Risk and reward: the history-based volatility and the payoff statistics built on it."""

import math
from datetime import date, timedelta

import pytest

from app import pricing
from app.analysis import contracts as metrics
from app.analysis import earnings as em
from app.analysis import risk
from app.models import Bar, EarningsEvent, OptionQuote

TODAY = date(2026, 11, 24)
RATE = 0.04
SPOT = 100.0


def contract(kind="call", strike=100.0, days=30, iv=0.30, history_vol=0.30):
    """A contract quoted exactly at its Black-Scholes value for `iv`."""
    fair = pricing.price(kind, SPOT, strike, days / 365, RATE, iv)
    quote = OptionQuote("X", "XYZ", kind, strike, TODAY + timedelta(days=days), bid=fair - 0.01, ask=fair + 0.01, iv=iv)
    c = metrics.analyze_contract(quote, SPOT, TODAY, RATE)
    c.history_vol = history_vol
    return c


def integrate(c, vol, lens, steps=40001):
    """Brute-force the same statistics by summing over the price distribution at expiry."""
    t = c.dte / 365
    drift = (RATE - 0.5 * vol * vol) * t
    width = 16.0 / (steps - 1)
    win = gain = loss = 0.0
    for i in range(steps):
        z = -8.0 + i * width
        weight = pricing.norm_pdf(z) * width
        final = SPOT * math.exp(drift + vol * math.sqrt(t) * z)
        payoff = max(final - c.strike, 0.0) if c.kind == "call" else max(c.strike - final, 0.0)
        pnl = payoff - c.mid if lens == "buy" else c.mid - payoff
        if pnl > 0:
            win += weight
            gain += weight * pnl
        else:
            loss -= weight * pnl
    return win, gain / win * 100, loss / (1 - win) * 100, (gain - loss) * 100


# -- History-based volatility ---------------------------------------------------


def test_realized_vol_can_leave_out_earnings_days():
    closes, days = [100.0], [TODAY - timedelta(days=40)]
    for i in range(30):
        jump = 1.15 if i == 20 else (1.01 if i % 2 == 0 else 0.99)
        closes.append(closes[-1] * jump)
        days.append(days[-1] + timedelta(days=1))
    bars = [Bar(d, c) for d, c in zip(days, closes)]
    jump_day = days[21]

    with_jump = metrics.realized_vol(bars, 20)
    calm = metrics.realized_vol(bars, 20, {jump_day})
    assert with_jump > 3 * calm
    assert calm == pytest.approx(0.01 * math.sqrt(252), rel=0.05)  # the alternating 1% days
    assert metrics.realized_vol(bars, 30, {jump_day}) is None  # 29 ordinary returns are not 30


def test_reaction_days_follow_the_report_timing():
    week = [Bar(date(2026, 11, 16 + i), 100.0 + i) for i in range(5)]  # Mon..Fri
    wed = date(2026, 11, 18)
    assert em.reaction_days([EarningsEvent(wed, "pre")], week) == {wed}
    assert em.reaction_days([EarningsEvent(wed, "post")], week) == {date(2026, 11, 19)}
    assert em.reaction_days([EarningsEvent(wed, "unknown")], week) == {wed, date(2026, 11, 19)}
    assert em.reaction_days([EarningsEvent(date(2026, 6, 1), "pre")], week) == set()


def test_history_vol_blends_ordinary_days_with_past_earnings_moves():
    report = EarningsEvent(TODAY + timedelta(days=9), "post")
    spanning, expiring_first = contract(days=30), contract(days=7)
    ordinary, event_var = 0.25, 0.08**2

    assert risk.history_vol(expiring_first, ordinary, event_var, report) == pytest.approx(ordinary)
    assert risk.history_vol(spanning, ordinary, event_var, None) == pytest.approx(ordinary)

    t = 30 / 365
    blended = math.sqrt((ordinary**2 * (t - 1 / 365) + event_var) / t)
    assert risk.history_vol(spanning, ordinary, event_var, report) == pytest.approx(blended)
    assert blended > 0.35  # one 8% report lifts 25% volatility well above itself over a month

    # A long-dated contract that spans two quarterly reports counts both.
    two_reports = contract(days=120)
    t = 120 / 365
    expected = math.sqrt((ordinary**2 * (t - 2 / 365) + 2 * event_var) / t)
    assert risk.history_vol(two_reports, ordinary, event_var, report) == pytest.approx(expected)


def test_history_vol_is_unknown_without_the_inputs_it_needs():
    report = EarningsEvent(TODAY + timedelta(days=9), "post")
    assert risk.history_vol(contract(), None, 0.01, report) is None
    assert risk.history_vol(contract(), 0.25, None, report) is None  # spans a report it cannot size
    assert risk.history_vol(contract(days=7), 0.25, None, report) == pytest.approx(0.25)
    unrated = contract(history_vol=None)
    assert risk.assess(unrated, SPOT, RATE, "buy") is None


# -- Payoff statistics ----------------------------------------------------------


@pytest.mark.parametrize("kind, strike", [("call", 100.0), ("call", 108.0), ("put", 100.0), ("put", 93.0)])
@pytest.mark.parametrize("lens", ["buy", "sell"])
@pytest.mark.parametrize("history_vol", [0.22, 0.30, 0.45])
def test_closed_form_matches_summing_over_the_price_distribution(kind, strike, lens, history_vol):
    c = contract(kind, strike, history_vol=history_vol)
    got = risk.assess(c, SPOT, RATE, lens)
    win, avg_win, avg_loss, value = integrate(c, history_vol, lens)
    assert got.win_prob == pytest.approx(win, abs=2e-4)
    assert got.avg_win == pytest.approx(avg_win, rel=2e-3)
    assert got.avg_loss == pytest.approx(avg_loss, rel=2e-3)
    assert got.expected_value == pytest.approx(value, abs=0.05)  # dollars per contract
    assert got.ratio == pytest.approx(avg_win / avg_loss, rel=4e-3)


@pytest.mark.parametrize("kind, strike", [("call", 100.0), ("call", 106.0), ("put", 95.0)])
def test_at_the_markets_own_price_reward_to_risk_mirrors_the_odds(kind, strike):
    c = contract(kind, strike, iv=0.30, history_vol=0.30)
    got = risk.assess(c, SPOT, RATE, "buy")
    assert got.rating == "balanced"
    assert abs(got.edge_pct) < 0.01  # only the interest on the premium
    assert got.win_prob == pytest.approx(c.pop_long, abs=1e-6)
    # Expected value near zero forces: average win / average loss = chance of loss / chance of win.
    assert got.ratio == pytest.approx((1 - got.win_prob) / got.win_prob, rel=0.02)


def test_seller_is_the_mirror_image_of_the_buyer():
    c = contract("put", 95.0, history_vol=0.24)
    buyer, seller = risk.assess(c, SPOT, RATE, "buy"), risk.assess(c, SPOT, RATE, "sell")
    assert seller.win_prob == pytest.approx(1 - buyer.win_prob)
    assert seller.avg_win == pytest.approx(buyer.avg_loss)
    assert seller.avg_loss == pytest.approx(buyer.avg_win)
    assert seller.expected_value == pytest.approx(-buyer.expected_value)
    assert seller.ratio == pytest.approx(1 / buyer.ratio)
    assert seller.avg_win <= c.mid * 100 + 1e-9  # a seller can never win more than the premium


def test_rating_follows_how_the_price_compares_with_history():
    underpriced = contract(iv=0.30, history_vol=0.40)  # the stock has moved more than the options charge for
    overpriced = contract(iv=0.30, history_vol=0.22)
    assert risk.assess(underpriced, SPOT, RATE, "buy").rating == "favorable"
    assert risk.assess(underpriced, SPOT, RATE, "sell").rating == "unfavorable"
    assert risk.assess(overpriced, SPOT, RATE, "buy").rating == "unfavorable"
    assert risk.assess(overpriced, SPOT, RATE, "sell").rating == "favorable"
    assert risk.assess(underpriced, SPOT, RATE, "buy").edge_pct == pytest.approx(0.33, abs=0.03)  # ~ 40/30 - 1 at the money


def test_worst_cases():
    call, put = contract("call", 105.0), contract("put", 95.0)
    sigma_move = 2 * 0.30 * math.sqrt(30 / 365)

    long_call = risk.assess(call, SPOT, RATE, "buy")
    assert long_call.max_loss == pytest.approx(call.mid * 100)
    assert long_call.bad_case_loss == pytest.approx(call.mid * 100)  # expires worthless after a 2 s.d. drop

    short_put = risk.assess(put, SPOT, RATE, "sell")
    assert short_put.max_loss == pytest.approx((95.0 - put.mid) * 100)
    after_drop = SPOT * math.exp(-sigma_move)
    assert short_put.bad_case_loss == pytest.approx((95.0 - after_drop - put.mid) * 100)
    assert short_put.bad_case_loss > 5 * put.mid * 100  # the tail dwarfs the premium collected

    short_call = risk.assess(call, SPOT, RATE, "sell")
    assert short_call.max_loss is None  # uncapped on its own; a covered call is backed by shares
    assert short_call.bad_case_loss == pytest.approx((SPOT * math.exp(sigma_move) - 105.0 - call.mid) * 100)


def test_lopsided_payoffs_are_labelled():
    far_call = contract("call", 118.0, days=20)  # needs a move the stock rarely makes
    buyer, seller = risk.assess(far_call, SPOT, RATE, "buy"), risk.assess(far_call, SPOT, RATE, "sell")
    assert buyer.win_prob < 0.20 and buyer.shape == "long shot"
    assert seller.win_prob > 0.80 and seller.shape == "tail risk"
    assert buyer.ratio > 5 and seller.ratio < 0.2  # big wins that rarely come; small wins with a rare big loss
    assert risk.assess(contract("call", 100.0), SPOT, RATE, "buy").shape is None

    # History above the price makes a long shot "favorable" on expected value; the label keeps it honest.
    cheap_long_shot = contract("call", 118.0, days=20, iv=0.30, history_vol=0.36)
    rated = risk.assess(cheap_long_shot, SPOT, RATE, "buy")
    assert (rated.rating, rated.shape) == ("favorable", "long shot")
