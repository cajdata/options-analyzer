"""OCC option symbols, e.g. AAPL261218C00200000."""

from __future__ import annotations

from datetime import date


def parse(symbol: str) -> tuple[str, date, str, float]:
    """Split an OCC symbol into (root, expiry, "call"|"put", strike)."""
    if len(symbol) < 16:
        raise ValueError(f"not an OCC option symbol: {symbol!r}")
    root, ymd, flag, strike = symbol[:-15], symbol[-15:-9], symbol[-9], symbol[-8:]
    if not root or flag not in "CP" or not (ymd + strike).isdigit():
        raise ValueError(f"not an OCC option symbol: {symbol!r}")
    expiry = date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]))
    return root, expiry, "call" if flag == "C" else "put", int(strike) / 1000.0


def build(root: str, expiry: date, kind: str, strike: float) -> str:
    flag = "C" if kind == "call" else "P"
    return f"{root}{expiry:%y%m%d}{flag}{round(strike * 1000):08d}"
