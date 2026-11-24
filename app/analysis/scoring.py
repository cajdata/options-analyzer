"""Rank contracts through a "lens": buying premium or selling it.

The score is a weighted sum of five components, each scaled 0 to 1. It orders
contracts by transparent criteria; it is not a forecast. Every threshold and
weight lives in LENSES below so it can be tuned in one place.
"""

from __future__ import annotations

from app.models import ContractAnalysis

# (zero, one) pairs: the value that scores 0 and the value that scores 1.
LENSES: dict[str, dict] = {
    "buy": {  # long calls and long puts
        "weights": {"liquidity": 0.25, "breakeven": 0.25, "vol_value": 0.20, "decay": 0.15, "delta_fit": 0.15},
        "breakeven_vs_expected": (1.5, 0.0),  # breakeven closer than the expected move is better
        "iv_vs_history": (1.6, 0.9),  # cheaper volatility is better for a buyer
        "theta_pct_per_day": (0.04, 0.005),
        "delta_band": (0.30, 0.60),
    },
    "sell": {  # cash-secured puts and covered calls
        "weights": {"liquidity": 0.25, "probability": 0.25, "vol_value": 0.20, "yield": 0.15, "delta_fit": 0.15},
        "pop": (0.50, 0.90),
        "iv_vs_history": (0.9, 1.6),  # richer volatility is better for a seller
        "annualized_yield": (0.0, 0.30),
        "delta_band": (0.15, 0.35),
    },
}
SPREAD_PCT = (0.20, 0.02)
OPEN_INTEREST_FULL = 500

# Contracts outside these limits are dropped before scoring.
MIN_MID = 0.05
MAX_SPREAD_PCT = 0.50
DELTA_LIMITS = (0.05, 0.95)


SHARES = 100


def outlay(c: ContractAnalysis, lens: str, spot: float) -> float:
    """Cash needed to open one contract.

    A buyer pays the ask. A seller sets aside collateral: the strike in cash for
    a cash-secured put, or 100 shares for a covered call.
    """
    if lens == "buy":
        return c.ask * SHARES
    return (c.strike if c.kind == "put" else spot) * SHARES


def ramp(value: float, zero: float, one: float) -> float:
    """Linear 0-to-1 scale between two anchors, clamped; works in either direction."""
    if one == zero:
        return 1.0 if value >= one else 0.0
    return min(1.0, max(0.0, (value - zero) / (one - zero)))


def tradable(c: ContractAnalysis) -> bool:
    return (
        c.mid >= MIN_MID
        and c.spread_pct <= MAX_SPREAD_PCT
        and DELTA_LIMITS[0] <= abs(c.delta) <= DELTA_LIMITS[1]
    )


def _liquidity(c: ContractAnalysis) -> float:
    spread = ramp(c.spread_pct, *SPREAD_PCT)
    if c.open_interest is None:
        return spread * 0.8
    return spread * (0.6 + 0.4 * min(c.open_interest / OPEN_INTEREST_FULL, 1.0))


def _vol_value(c: ContractAnalysis, cfg: dict) -> float:
    """Implied volatility against what the stock's history implies for the same period.

    `history_vol` blends ordinary realized volatility with past earnings moves
    for any report the contract spans, so one comparison covers both cases. The
    risk/reward rating starts from the same number and then charges for the
    bid-ask spread, so it can be a shade less favorable than this component.
    """
    if c.history_vol:
        return ramp(c.iv / c.history_vol, *cfg["iv_vs_history"])
    return 0.5  # nothing to compare against


def _delta_fit(c: ContractAnalysis, band: tuple[float, float]) -> float:
    delta = abs(c.delta)
    if band[0] <= delta <= band[1]:
        return 1.0
    distance = band[0] - delta if delta < band[0] else delta - band[1]
    return ramp(distance, 0.20, 0.0)


def score(c: ContractAnalysis, lens: str) -> None:
    cfg = LENSES[lens]
    parts = {
        "liquidity": _liquidity(c),
        "vol_value": _vol_value(c, cfg),
        "delta_fit": _delta_fit(c, cfg["delta_band"]),
    }
    if lens == "buy":
        c.pop = c.pop_long
        ratio = c.breakeven_vs_expected
        parts["breakeven"] = 0.5 if ratio is None else ramp(ratio, *cfg["breakeven_vs_expected"])
        parts["decay"] = ramp(c.theta_pct_per_day, *cfg["theta_pct_per_day"])
    else:
        c.pop = 1.0 - c.pop_long
        parts["probability"] = ramp(c.pop, *cfg["pop"])
        parts["yield"] = ramp(c.annualized_yield, *cfg["annualized_yield"])
    c.lens = lens
    c.components = {name: round(parts[name], 3) for name in cfg["weights"]}
    c.score = round(100.0 * sum(cfg["weights"][name] * parts[name] for name in cfg["weights"]), 1)


def rank(contracts: list[ContractAnalysis], per_side: int, top_n: int) -> list[ContractAnalysis]:
    """Best contracts overall, with at most `per_side` calls and `per_side` puts per stock."""
    taken: dict[tuple[str, str], int] = {}
    out: list[ContractAnalysis] = []
    if top_n <= 0:
        return out
    for c in sorted(contracts, key=lambda c: c.score or 0.0, reverse=True):
        side = (c.underlying, c.kind)
        if taken.get(side, 0) >= per_side:
            continue
        taken[side] = taken.get(side, 0) + 1
        out.append(c)
        if len(out) >= top_n:
            break
    return out
