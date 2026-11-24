import math

import pytest

from app import occ, pricing

# Textbook case: S=100, K=100, one year, 5% rate, 20% volatility.
S, K, T, R, VOL = 100.0, 100.0, 1.0, 0.05, 0.20


def test_price_matches_textbook_values():
    assert pricing.price("call", S, K, T, R, VOL) == pytest.approx(10.4506, abs=1e-4)
    assert pricing.price("put", S, K, T, R, VOL) == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity():
    for strike in (80.0, 100.0, 125.0):
        call = pricing.price("call", S, strike, 0.4, R, 0.35)
        put = pricing.price("put", S, strike, 0.4, R, 0.35)
        assert call - put == pytest.approx(S - strike * math.exp(-R * 0.4), abs=1e-9)


def test_greeks_match_textbook_values_in_broker_units():
    call = pricing.greeks("call", S, K, T, R, VOL)
    assert call.delta == pytest.approx(0.6368, abs=1e-4)
    assert call.gamma == pytest.approx(0.018762, abs=1e-6)
    assert call.vega == pytest.approx(0.37524, abs=1e-5)  # per volatility point
    assert call.theta == pytest.approx(-6.4140 / 365, abs=1e-5)  # per calendar day
    assert call.rho == pytest.approx(0.53232, abs=1e-5)  # per rate point

    put = pricing.greeks("put", S, K, T, R, VOL)
    assert put.delta == pytest.approx(call.delta - 1.0)
    assert put.gamma == pytest.approx(call.gamma)
    assert put.theta == pytest.approx(-1.6579 / 365, abs=1e-5)


def test_greeks_agree_with_finite_differences():
    for kind in ("call", "put"):
        g = pricing.greeks(kind, 50.0, 55.0, 0.25, 0.04, 0.45)
        bump = 1e-4
        up = pricing.price(kind, 50.0 + bump, 55.0, 0.25, 0.04, 0.45)
        down = pricing.price(kind, 50.0 - bump, 55.0, 0.25, 0.04, 0.45)
        assert g.delta == pytest.approx((up - down) / (2 * bump), abs=1e-6)
        vol_up = pricing.price(kind, 50.0, 55.0, 0.25, 0.04, 0.46)
        vol_down = pricing.price(kind, 50.0, 55.0, 0.25, 0.04, 0.44)
        assert g.vega == pytest.approx((vol_up - vol_down) / 2, abs=1e-4)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("strike", [85.0, 100.0, 118.0])
@pytest.mark.parametrize("vol", [0.12, 0.45, 1.6])
def test_implied_vol_recovers_the_input(kind, strike, vol):
    target = pricing.price(kind, S, strike, 0.2, R, vol)
    assert pricing.implied_vol(kind, target, S, strike, 0.2, R) == pytest.approx(vol, abs=1e-4)


def test_implied_vol_is_none_when_no_volatility_fits():
    # A call cannot be worth less than its discounted intrinsic value.
    assert pricing.implied_vol("call", 19.0, 120.0, 100.0, 0.5, R) is None
    assert pricing.implied_vol("call", 0.0, S, K, T, R) is None
    assert pricing.implied_vol("put", 500.0, S, K, T, R) is None


def test_prob_above_is_a_probability_that_falls_with_the_level():
    probs = [pricing.prob_above(S, level, 0.5, R, 0.3) for level in (60, 90, 100, 110, 160)]
    assert all(0.0 < p < 1.0 for p in probs)
    assert probs == sorted(probs, reverse=True)


def test_occ_symbols_round_trip():
    root, expiry, kind, strike = occ.parse("AAPL261218C00200000")
    assert (root, expiry.isoformat(), kind, strike) == ("AAPL", "2026-12-18", "call", 200.0)
    assert occ.parse("F270115P00012500")[3] == 12.5
    assert occ.build(root, expiry, kind, strike) == "AAPL261218C00200000"
    for bad in ("AAPL", "AAPL261218X00200000", "261218C00200000"):
        with pytest.raises(ValueError):
            occ.parse(bad)
