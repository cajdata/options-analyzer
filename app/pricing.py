"""Black-Scholes pricing, Greeks, implied volatility and probabilities.

European-exercise formulas. US equity options are American, so values for
deep in-the-money puts and for calls just before a dividend are approximate.

Units match what brokers display:
  theta - price change per calendar day
  vega  - price change per one volatility point (1%)
  rho   - price change per one point (1%) of interest rate
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / _SQRT_2)


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _d1_d2(s: float, k: float, t: float, r: float, sigma: float, q: float) -> tuple[float, float]:
    vol_t = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / vol_t
    return d1, d1 - vol_t


def price(kind: str, s: float, k: float, t: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Theoretical price of a call or put. `t` is years to expiry."""
    d1, d2 = _d1_d2(s, k, t, r, sigma, q)
    disc_s = s * math.exp(-q * t)
    disc_k = k * math.exp(-r * t)
    if kind == "call":
        return disc_s * norm_cdf(d1) - disc_k * norm_cdf(d2)
    return disc_k * norm_cdf(-d2) - disc_s * norm_cdf(-d1)


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


def greeks(kind: str, s: float, k: float, t: float, r: float, sigma: float, q: float = 0.0) -> Greeks:
    d1, d2 = _d1_d2(s, k, t, r, sigma, q)
    sqrt_t = math.sqrt(t)
    disc_q = math.exp(-q * t)
    disc_k = k * math.exp(-r * t)
    pdf = norm_pdf(d1)

    gamma = disc_q * pdf / (s * sigma * sqrt_t)
    vega = s * disc_q * pdf * sqrt_t / 100.0
    decay = -s * disc_q * pdf * sigma / (2.0 * sqrt_t)
    if kind == "call":
        delta = disc_q * norm_cdf(d1)
        theta = (decay - r * disc_k * norm_cdf(d2) + q * s * disc_q * norm_cdf(d1)) / 365.0
        rho = disc_k * t * norm_cdf(d2) / 100.0
    else:
        delta = disc_q * (norm_cdf(d1) - 1.0)
        theta = (decay + r * disc_k * norm_cdf(-d2) - q * s * disc_q * norm_cdf(-d1)) / 365.0
        rho = -disc_k * t * norm_cdf(-d2) / 100.0
    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


def implied_vol(
    kind: str,
    target: float,
    s: float,
    k: float,
    t: float,
    r: float,
    q: float = 0.0,
    lo: float = 0.005,
    hi: float = 5.0,
) -> float | None:
    """Volatility that reproduces `target`, or None when no volatility can.

    That happens when the price is at or below the no-volatility floor (stale
    or crossed quotes) or above anything a 500% volatility would produce.
    """
    if target <= 0 or s <= 0 or k <= 0 or t <= 0:
        return None
    if target <= price(kind, s, k, t, r, lo, q) or target >= price(kind, s, k, t, r, hi, q):
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if price(kind, s, k, t, r, mid, q) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break
    return 0.5 * (lo + hi)


def prob_above(s: float, level: float, t: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Risk-neutral probability the stock finishes above `level` at expiry."""
    if level <= 0:
        return 1.0
    _, d2 = _d1_d2(s, level, t, r, sigma, q)
    return norm_cdf(d2)
