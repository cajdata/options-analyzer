"""Exit plan: repricing a contract on later dates with the stock standing still."""

import math
from datetime import date, timedelta

import pytest

from app import pricing
from app.analysis import contracts as metrics
from app.analysis import earnings as em
from app.analysis import exits
from app.analysis.contracts import TRADING_DAY
from app.models import EarningsEvent, ImpliedMove, OptionQuote, UnderlyingContext

TODAY = date(2026, 11, 24)  # a Tuesday
RATE = 0.04
SPOT = 100.0
ORDINARY, EVENT_STD = 0.30, 0.06
REPORT = EarningsEvent(TODAY + timedelta(days=9), "post")  # Thursday 3 December, after the close
REACTION = date(2026, 12, 4)  # the Friday it is felt


def context(**overrides) -> UnderlyingContext:
    base = dict(symbol="XYZ", price=SPOT, change_pct=0.0, realized_vol_20d=0.3, realized_vol_60d=0.3, atm_iv=0.3)
    return UnderlyingContext(**{**base, **overrides})


def contract(kind="call", strike=100.0, days=45, iv=0.30, half_spread=0.05, spans=False):
    """A contract quoted around its Black-Scholes value for `iv`."""
    fair = pricing.price(kind, SPOT, strike, days / 365, RATE, iv)
    quote = OptionQuote("X", "XYZ", kind, strike, TODAY + timedelta(days=days), bid=fair - half_spread, ask=fair + half_spread, iv=iv)
    c = metrics.analyze_contract(quote, SPOT, TODAY, RATE)
    c.spans_earnings = spans
    return c


def spanning(kind="call", strike=100.0, days=30, half_spread=0.05):
    """A contract whose price holds ordinary volatility plus one earnings report, and its context."""
    t = days / 365
    iv = math.sqrt((ORDINARY**2 * (t - TRADING_DAY) + EVENT_STD**2) / t)
    priced = ImpliedMove(TODAY + timedelta(days=days), None, EVENT_STD * math.sqrt(2 / math.pi), ORDINARY, "test", event_var=EVENT_STD**2)
    return contract(kind, strike, days, iv, half_spread, spans=True), context(next_earnings=REPORT, implied_move=priced)


# -- The standing-still path ----------------------------------------------------


def test_path_starts_at_todays_price_and_ends_at_exercise_value():
    for kind, strike in (("call", 95.0), ("call", 105.0), ("put", 104.0)):
        c = contract(kind, strike)
        model = exits._Model(c, context(), RATE, TODAY)
        assert model.value(TODAY) == pytest.approx(c.mid, abs=1e-4)
        exercise = max(SPOT - strike, 0.0) if kind == "call" else max(strike - SPOT, 0.0)
        assert model.value(c.expiry) == exercise
        values = [model.value(TODAY + timedelta(days=n)) for n in range(0, 46)]
        assert values == sorted(values, reverse=True)  # with no report, time only takes value away


def test_half_the_time_value_goes_in_the_last_quarter_of_an_options_life():
    # Time value shrinks with the square root of time left, so half of it remains
    # until three quarters of the life has passed. That is why decay speeds up late.
    plan = exits.build(contract(days=44), context(), "buy", RATE, TODAY)
    assert plan.half_life_day == TODAY + timedelta(days=33)
    assert plan.target_price is None and plan.target_day is None  # those belong to a seller


def test_closing_early_crosses_the_spread_a_second_time():
    c = contract(half_spread=0.25)
    model = exits._Model(c, context(), RATE, TODAY)
    for lens, entry, side in (("buy", c.ask, -1), ("sell", c.bid, +1)):
        plan = exits.build(c, context(), lens, RATE, TODAY)
        assert plan.entry_price == entry
        *early, expiry = plan.checkpoints
        for k in early:
            assert k.value == pytest.approx(model.value(k.day))
            assert k.close_price == pytest.approx(k.value * (1 + side * c.spread_pct / 2))  # the bid, or the ask
            gain = k.close_price - c.ask if lens == "buy" else c.bid - k.close_price
            assert k.pnl == pytest.approx(gain * 100) and k.pnl_pct == pytest.approx(gain / entry)
        assert expiry.label == "at expiry" and expiry.close_price == expiry.value  # nothing to cross at settlement

    # Bought and sold again a moment later, the whole spread is lost.
    same_day = model.value(TODAY) * (1 - c.spread_pct / 2) - c.ask
    assert same_day == pytest.approx(-(c.ask - c.bid), abs=1e-4)


def test_checkpoints_are_in_date_order_with_the_right_labels():
    plan = exits.build(contract(days=45), context(), "buy", RATE, TODAY)
    assert [(k.label, k.days_held) for k in plan.checkpoints] == [
        ("in 7 days", 7), ("in 14 days", 14), ("in 21 days", 21), ("time stop", 24), ("at expiry", 45),
    ]
    short = exits.build(contract(days=10), context(), "buy", RATE, TODAY)
    assert [k.label for k in short.checkpoints] == ["in 7 days", "at expiry"]
    assert short.time_stop_day is None  # the time stop is already behind it
    expiring = contract(days=10)
    expiring.dte = 0  # on its last day there is nothing left to plan
    assert exits.build(expiring, context(), "buy", RATE, expiring.expiry) is None


# -- A seller's rules ---------------------------------------------------------------


def test_seller_profit_target_and_when_decay_alone_reaches_it():
    c = contract("put", 93.0, days=45)
    plan = exits.build(c, context(), "sell", RATE, TODAY)
    assert plan.target_price == pytest.approx(c.bid * (1 - exits.PROFIT_TARGET))
    assert plan.time_stop_day == c.expiry - timedelta(days=exits.TIME_STOP_DTE)
    assert plan.strike_distance_pct == pytest.approx(-0.07)
    assert plan.half_life_day is None  # that belongs to a buyer

    # The target day is the first on which buying back, at the ask, costs no more than the target.
    model = exits._Model(c, context(), RATE, TODAY)

    def buy_back(day):
        return model.value(day) * (1 + c.spread_pct / 2)

    assert TODAY < plan.target_day < c.expiry
    assert buy_back(plan.target_day) <= plan.target_price < buy_back(plan.target_day - timedelta(days=1))


def test_decay_alone_never_halves_an_in_the_money_option():
    deep = contract("put", 112.0, days=45)  # worth at least $12 however much time passes
    plan = exits.build(deep, context(), "sell", RATE, TODAY)
    assert plan.target_day is None
    # Held to expiry with the stock unchanged, the seller keeps only the sliver that was time value.
    assert 0 < plan.checkpoints[-1].pnl_pct < 0.05


def test_seller_notes():
    inside = exits.build(contract("put", 95.0, days=14), context(), "sell", RATE, TODAY)
    assert inside.time_stop_day is None and any("already inside the 21-day window" in n for n in inside.notes)
    short_call = exits.build(contract("call", 105.0), context(), "sell", RATE, TODAY)
    assert any("ex-dividend" in n for n in short_call.notes)
    assert not any("ex-dividend" in n for n in exits.build(contract("put", 95.0), context(), "sell", RATE, TODAY).notes)
    assert exits.build(contract("call", 105.0), context(), "buy", RATE, TODAY).notes == []


# -- Across an earnings report --------------------------------------------------------


def test_reaction_day_is_the_first_session_that_reflects_the_report():
    friday = date(2026, 12, 4)
    assert em.reaction_day(EarningsEvent(friday, "pre")) == friday
    assert em.reaction_day(EarningsEvent(friday, "post")) == date(2026, 12, 7)  # the following Monday
    assert em.reaction_day(EarningsEvent(friday, "unknown")) == date(2026, 12, 7)  # never placed too early
    assert em.reaction_day(REPORT) == REACTION


def test_the_report_step_removes_exactly_the_reports_variance():
    c, ctx = spanning(half_spread=0.0)
    model = exits._Model(c, ctx, RATE, TODAY)
    assert model.report == REACTION
    assert model.value(TODAY) == pytest.approx(c.mid, abs=1e-4)

    # Once the report has landed, only ordinary volatility is left in the price.
    left = (c.expiry - REACTION).days / 365
    assert model.value(REACTION) == pytest.approx(pricing.price("call", SPOT, 100.0, left, RATE, ORDINARY), abs=1e-4)
    # The day before, the report is still ahead and still priced.
    eve = REACTION - timedelta(days=1)
    left = (c.expiry - eve).days / 365
    with_report = math.sqrt((ORDINARY**2 * (left - TRADING_DAY) + EVENT_STD**2) / left)
    assert model.value(eve) == pytest.approx(pricing.price("call", SPOT, 100.0, left, RATE, with_report), abs=1e-4)

    plan = exits.build(c, ctx, "buy", RATE, TODAY)
    step = plan.earnings
    assert (step.report_day, step.value_before, step.value_after) == (REACTION, model.value(eve), model.value(REACTION))
    assert step.crush_pct == pytest.approx(step.value_after / step.value_before - 1) and step.crush_pct < -0.10
    assert [k.label for k in plan.checkpoints] == ["in 7 days", "before earnings", "after earnings", "in 14 days", "in 21 days", "at expiry"]
    assert plan.half_life_day > REACTION  # the report takes a bite, the rest goes with time


@pytest.mark.parametrize("kind, strike", [("call", 100.0), ("call", 104.0), ("put", 100.0), ("put", 95.0)])
def test_the_move_that_undoes_the_report_step(kind, strike):
    c, ctx = spanning(kind, strike)
    model = exits._Model(c, ctx, RATE, TODAY)
    toward_the_money = 1 if kind == "call" else -1

    # A holder: the move needed the next day to sell for what was paid.
    held = exits.build(c, ctx, "buy", RATE, TODAY).earnings.move_to_breakeven_pct
    assert held * toward_the_money > 0
    assert model.value(REACTION, SPOT * (1 + held)) * (1 - c.spread_pct / 2) == pytest.approx(c.ask, abs=1e-4)

    # A seller: the move the position absorbs before buying back costs what was collected.
    sold = exits.build(c, ctx, "sell", RATE, TODAY).earnings.move_to_breakeven_pct
    assert sold * toward_the_money > 0
    assert model.value(REACTION, SPOT * (1 + sold)) * (1 + c.spread_pct / 2) == pytest.approx(c.bid, abs=1e-4)
    assert abs(sold) < abs(held)  # the spread sits between the two


def test_a_report_is_sized_from_history_when_the_options_do_not_say():
    c, ctx = spanning()
    from_history = context(next_earnings=REPORT, event_var=0.08**2)  # no implied move available
    assert exits._Model(c, from_history, RATE, TODAY).event_var == pytest.approx(0.08**2)
    assert exits.build(c, from_history, "buy", RATE, TODAY).earnings is not None

    # The report's share is capped, so the ordinary part of the price never goes negative.
    huge = context(next_earnings=REPORT, event_var=0.50**2)
    model = exits._Model(c, huge, RATE, TODAY)
    assert model.event_var == pytest.approx(exits.MAX_EVENT_SHARE * c.iv**2 * 30 / 365, rel=1e-3)
    assert model.ordinary_rate > 0


def test_a_report_that_cannot_be_modelled_is_said_so():
    c, _ = spanning()
    unsized = exits.build(c, context(next_earnings=REPORT), "buy", RATE, TODAY)
    assert unsized.earnings is None and any("could not be sized" in n for n in unsized.notes)

    estimate = EarningsEvent(REPORT.day, "post", confirmed=False)
    _, ctx = spanning()
    ctx.next_earnings = estimate
    assert any("date is an estimate" in n for n in exits.build(c, ctx, "buy", RATE, TODAY).notes)

    unknown = exits.build(contract(), context(earnings_unknown=True), "buy", RATE, TODAY)
    assert any("No earnings date is known" in n for n in unknown.notes)

    # A contract that expires before the report is priced as if there were none.
    before = contract(days=7)
    assert exits.build(before, context(next_earnings=REPORT), "buy", RATE, TODAY).earnings is None


def test_the_path_never_goes_below_exercise_value():
    # The pricing formula is for European exercise and puts a deep in-the-money put
    # under its exercise value near expiry. A real American option never trades there.
    deep = contract("put", 104.0)
    model = exits._Model(deep, context(), RATE, TODAY)
    last_day = deep.expiry - timedelta(days=1)
    assert pricing.price("put", SPOT, 104.0, 1 / 365, RATE, 0.30) < 4.0  # the raw formula does dip
    assert model.value(last_day) == 4.0
    assert all(model.value(TODAY + timedelta(days=n)) >= 4.0 for n in range(46))
