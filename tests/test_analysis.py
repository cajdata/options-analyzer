import math
import statistics
from datetime import date, timedelta

import pytest

from app import pricing
from app.analysis import contracts as metrics
from app.analysis import earnings as em
from app.analysis import scoring
from app.models import Bar, ContractAnalysis, EarningsEvent, OptionQuote

TODAY = date(2026, 11, 24)
RATE = 0.04


def quote(kind="call", strike=100.0, days=30, bid=3.0, ask=3.2, **extra) -> OptionQuote:
    return OptionQuote("X", "XYZ", kind, strike, TODAY + timedelta(days=days), bid=bid, ask=ask, **extra)


def analyzed(**overrides) -> ContractAnalysis:
    c = metrics.analyze_contract(quote(iv=0.30), 100.0, TODAY, RATE)
    for name, value in overrides.items():
        setattr(c, name, value)
    return c


# -- Contract metrics ---------------------------------------------------------


def test_contract_metrics_for_a_call_and_a_put():
    call = metrics.analyze_contract(quote("call", 100.0, bid=3.0, ask=3.2, iv=0.30), 100.0, TODAY, RATE)
    assert call.mid == pytest.approx(3.1)
    assert call.spread_pct == pytest.approx(0.2 / 3.1)
    assert call.breakeven == pytest.approx(103.1)
    assert call.breakeven_move_pct == pytest.approx(0.031)
    assert call.iv_source == "provider" and call.greeks_source == "computed"
    assert 0 < call.pop_long < call.prob_itm < 1  # breakeven sits beyond the strike

    put = metrics.analyze_contract(quote("put", 100.0, bid=2.7, ask=2.9, iv=0.30), 100.0, TODAY, RATE)
    assert put.breakeven == pytest.approx(97.2)
    assert put.delta < 0
    assert 0 < put.pop_long < put.prob_itm < 1
    assert put.annualized_yield == pytest.approx(2.8 / 100.0 * 365 / 30)


def test_missing_volatility_is_solved_from_the_quote():
    t = 30 / 365
    fair = pricing.price("call", 100.0, 105.0, t, RATE, 0.42)
    c = metrics.analyze_contract(quote("call", 105.0, bid=fair - 0.01, ask=fair + 0.01), 100.0, TODAY, RATE)
    assert c.iv_source == "computed"
    assert c.iv == pytest.approx(0.42, abs=1e-3)


def test_provider_greeks_are_used_when_complete():
    q = quote(iv=0.30, delta=0.51, gamma=0.04, theta=-0.06, vega=0.11)
    c = metrics.analyze_contract(q, 100.0, TODAY, RATE)
    assert (c.delta, c.theta, c.greeks_source) == (0.51, -0.06, "provider")


@pytest.mark.parametrize("bid, ask", [(None, 1.0), (0.0, 1.0), (1.2, 1.0), (1.0, None)])
def test_unpriceable_quotes_are_dropped(bid, ask):
    assert metrics.analyze_contract(quote(bid=bid, ask=ask, iv=0.3), 100.0, TODAY, RATE) is None


def test_quote_below_intrinsic_value_is_dropped():
    assert metrics.analyze_contract(quote("call", 80.0, bid=10.0, ask=10.2), 100.0, TODAY, RATE) is None


def test_atm_point_uses_the_paired_strike_nearest_spot():
    chain = [
        metrics.analyze_contract(quote(kind, strike, bid=bid, ask=bid + 0.2, iv=iv), 101.0, TODAY, RATE)
        for kind, strike, bid, iv in [
            ("call", 100.0, 3.4, 0.31), ("put", 100.0, 2.2, 0.33),
            ("call", 105.0, 1.4, 0.29), ("put", 105.0, 5.0, 0.30),
            ("call", 101.0, 2.9, 0.50),  # nearest strike, but only one side quoted
        ]
    ]
    point = metrics.atm_by_expiry(chain, 101.0)[TODAY + timedelta(days=30)]
    assert point.strike == 100.0
    assert point.iv == pytest.approx(0.32)
    assert point.straddle == pytest.approx(3.5 + 2.3)


def test_expected_move_is_measured_from_the_forward_price():
    call = metrics.analyze_contract(quote("call", 100.0, bid=3.4, ask=3.6, iv=0.30), 100.0, TODAY, RATE)
    put = metrics.analyze_contract(quote("put", 100.0, bid=3.1, ask=3.3, iv=0.30), 100.0, TODAY, RATE)
    atm = metrics.atm_by_expiry([call, put], 100.0)
    metrics.apply_expected_moves([call, put], atm, 100.0, TODAY, RATE)
    t = 30 / 365
    forward = 100.0 * math.exp(RATE * t)
    assert call.expected_move_pct == pytest.approx(0.30 * math.sqrt(t))
    assert call.breakeven_vs_expected == pytest.approx((103.5 / forward - 1) / call.expected_move_pct)
    assert put.breakeven_vs_expected == pytest.approx((1 - 96.8 / forward) / put.expected_move_pct)


def test_realized_vol():
    closes = [100.0]
    for i in range(20):
        closes.append(closes[-1] * (1.01 if i % 2 == 0 else 0.99))
    bars = [Bar(TODAY - timedelta(days=30 - i), c) for i, c in enumerate(closes)]
    returns = [math.log(b / a) for a, b in zip(closes, closes[1:])]
    assert metrics.realized_vol(bars, 20) == pytest.approx(statistics.stdev(returns) * math.sqrt(252))
    assert metrics.realized_vol(bars, 60) is None  # not enough history


# -- Earnings -----------------------------------------------------------------

# Mon 16 Nov .. Fri 20 Nov 2026
WEEK = [Bar(date(2026, 11, 16 + i), close) for i, close in enumerate([100.0, 102.0, 101.0, 111.1, 110.0])]
WED = date(2026, 11, 18)


def test_reaction_depends_on_report_timing():
    def move(timing):
        return em.reactions([EarningsEvent(WED, timing)], WEEK)[0].move_pct

    assert move("post") == pytest.approx(111.1 / 101.0 - 1)  # Wed close -> Thu close
    assert move("pre") == pytest.approx(101.0 / 102.0 - 1)  # Tue close -> Wed close
    assert move("unknown") == pytest.approx(111.1 / 102.0 - 1)  # Tue close -> Thu close


def test_reactions_skip_events_without_price_history_and_sort_newest_first():
    saturday = EarningsEvent(date(2026, 11, 14), "post")  # no bar before it
    future = EarningsEvent(date(2026, 11, 20), "post")  # no bar after it
    old = EarningsEvent(date(2026, 8, 1), "pre")  # far outside the bars
    tuesday = EarningsEvent(date(2026, 11, 17), "pre", surprise_pct=4.0)
    got = em.reactions([tuesday, saturday, future, old, EarningsEvent(WED, "post")], WEEK)
    assert [r.day for r in got] == [WED, date(2026, 11, 17)]
    assert got[1].surprise_pct == 4.0


def test_summary_statistics():
    from app.models import Reaction

    moves = [Reaction(WED, "post", m) for m in (0.10, -0.04, 0.02)]
    stats = em.summarize(moves)
    assert (stats.count, stats.up, stats.down) == (3, 2, 1)
    assert stats.mean_abs == pytest.approx(0.16 / 3)
    assert stats.median_abs == pytest.approx(0.04)
    assert stats.max_abs == pytest.approx(0.10)
    assert em.summarize([]) is None


def test_next_report_is_estimated_from_the_usual_gap():
    history = [EarningsEvent(date(2026, 10, 29) - timedelta(days=91 * n), "post") for n in range(6)]
    guess = em.estimate_next(history, TODAY)
    assert guess.day == date(2026, 10, 29) + timedelta(days=91)
    assert (guess.timing, guess.confirmed) == ("post", False)
    assert em.estimate_next(history, date(2027, 6, 1)) is None  # history is stale
    assert em.estimate_next([], TODAY) is None


def test_spans_counts_the_reaction_day():
    assert em.spans(WED, EarningsEvent(WED, "pre"))
    assert not em.spans(WED, EarningsEvent(WED, "post"))  # expires before the after-hours report
    assert em.spans(WED + timedelta(days=1), EarningsEvent(WED, "post"))


def test_implied_move_recovers_the_event_priced_into_the_chain():
    event = EarningsEvent(TODAY + timedelta(days=9), "post")
    before, after = TODAY + timedelta(days=5), TODAY + timedelta(days=12)
    ordinary, event_std = 0.30, 0.06
    t = 12 / 365
    iv_after = math.sqrt((ordinary**2 * (t - 1 / 365) + event_std**2) / t)
    atm = {
        before: metrics.AtmPoint(100.0, ordinary, 3.0),
        after: metrics.AtmPoint(100.0, iv_after, 8.0),
    }
    implied = em.implied_move(100.0, atm, event, realized=0.99, today=TODAY)
    assert implied.expiry == after
    assert implied.baseline_source == "the last expiry before the report"
    assert implied.straddle_pct == pytest.approx(0.08)
    assert implied.event_move_pct == pytest.approx(event_std * math.sqrt(2 / math.pi))

    # Without an expiry before the report, two expiries after it pin down ordinary volatility.
    later = TODAY + timedelta(days=19)
    t_later = 19 / 365
    iv_later = math.sqrt((ordinary**2 * (t_later - 1 / 365) + event_std**2) / t_later)
    chain = {after: atm[after], later: metrics.AtmPoint(100.0, iv_later, 9.0)}
    forward = em.implied_move(100.0, chain, event, realized=0.99, today=TODAY)
    assert forward.baseline_source == "the next expiry after it"
    assert forward.baseline_vol == pytest.approx(ordinary)
    assert forward.event_move_pct == pytest.approx(event_std * math.sqrt(2 / math.pi))

    # With a single expiry after the report, realized volatility is the last resort.
    fallback = em.implied_move(100.0, {after: atm[after]}, event, realized=ordinary, today=TODAY)
    assert fallback.baseline_source == "recent realized volatility"
    assert fallback.event_move_pct == pytest.approx(event_std * math.sqrt(2 / math.pi))
    assert em.implied_move(100.0, {after: atm[after]}, event, realized=None, today=TODAY).event_move_pct is None

    assert em.implied_move(100.0, {before: atm[before]}, event, realized=ordinary, today=TODAY) is None


# -- Scoring ------------------------------------------------------------------


def test_ramp_scales_in_either_direction_and_clamps():
    assert scoring.ramp(0.11, 0.20, 0.02) == pytest.approx(0.5)
    assert scoring.ramp(0.70, 0.50, 0.90) == pytest.approx(0.5)
    assert scoring.ramp(5.0, 0.20, 0.02) == 0.0
    assert scoring.ramp(0.0, 0.20, 0.02) == 1.0


def test_tradable_filters():
    assert scoring.tradable(analyzed())
    assert not scoring.tradable(analyzed(mid=0.03))
    assert not scoring.tradable(analyzed(spread_pct=0.8))
    assert not scoring.tradable(analyzed(delta=0.02))
    assert not scoring.tradable(analyzed(delta=-0.98))


@pytest.mark.parametrize("lens", ["buy", "sell"])
def test_score_is_the_weighted_sum_of_its_components(lens):
    c = analyzed(breakeven_vs_expected=0.6, history_vol=0.28)
    scoring.score(c, lens)
    weights = scoring.LENSES[lens]["weights"]
    assert set(c.components) == set(weights)
    assert all(0.0 <= v <= 1.0 for v in c.components.values())
    assert c.score == pytest.approx(100 * sum(weights[k] * c.components[k] for k in weights), abs=0.1)
    assert c.pop == pytest.approx(c.pop_long if lens == "buy" else 1 - c.pop_long)


def test_volatility_is_judged_against_the_stocks_own_history():
    # Implied volatility is 30%. History says 40% (cheap), 30% (in line) or 15% (rich).
    cheap, in_line, rich = (analyzed(history_vol=v) for v in (0.40, 0.30, 0.15))
    for c in (cheap, in_line, rich):
        scoring.score(c, "buy")
    assert cheap.components["vol_value"] == 1.0 > in_line.components["vol_value"] > rich.components["vol_value"] == 0.0
    for c in (cheap, in_line, rich):
        scoring.score(c, "sell")
    assert cheap.components["vol_value"] == 0.0 < in_line.components["vol_value"] < rich.components["vol_value"] == 1.0

    unknown = analyzed(history_vol=None)
    scoring.score(unknown, "buy")
    assert unknown.components["vol_value"] == 0.5  # nothing to compare against


def test_rank_keeps_the_best_call_and_put_per_stock():
    def scored(underlying, kind, value):
        c = analyzed(underlying=underlying, kind=kind)
        c.score = value
        return c

    pool = [
        scored("AAA", "call", 90), scored("AAA", "call", 85), scored("AAA", "put", 70),
        scored("BBB", "call", 88), scored("BBB", "put", 60),
    ]
    top = scoring.rank(pool, per_side=1, top_n=3)
    assert [(c.underlying, c.kind, c.score) for c in top] == [("AAA", "call", 90), ("BBB", "call", 88), ("AAA", "put", 70)]
    assert len(scoring.rank(pool, per_side=5, top_n=10)) == 5
