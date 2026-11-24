"""Earnings context: how the stock moved on past reports and what is priced in now."""

from __future__ import annotations

import math
import statistics
from bisect import bisect_left, bisect_right
from datetime import date, timedelta

from app.analysis.contracts import AtmPoint, years_to_expiry
from app.models import Bar, EarningsEvent, EarningsStats, ImpliedMove, Reaction

_MAX_GAP = timedelta(days=6)  # skip an event if the price history has a hole around it


def reactions(events: list[EarningsEvent], bars: list[Bar]) -> list[Reaction]:
    """Close-to-close move across each past report, newest first.

    A report before the open moves the stock that same day; one after the close
    moves it the next trading day. When the timing is unknown the move is
    measured from the close before the report day to the close after it, which
    covers both cases.
    """
    days = [b.day for b in bars]
    out: list[Reaction] = []
    for event in events:
        if event.timing == "post":
            before, after = bisect_right(days, event.day) - 1, bisect_right(days, event.day)
        elif event.timing == "pre":
            before, after = bisect_left(days, event.day) - 1, bisect_left(days, event.day)
        else:
            before, after = bisect_left(days, event.day) - 1, bisect_right(days, event.day)
        if before < 0 or after >= len(bars):
            continue
        if event.day - days[before] > _MAX_GAP or days[after] - event.day > _MAX_GAP:
            continue
        move = bars[after].close / bars[before].close - 1.0
        out.append(Reaction(day=event.day, timing=event.timing, move_pct=move, surprise_pct=event.surprise_pct))
    return sorted(out, key=lambda r: r.day, reverse=True)


def summarize(moves: list[Reaction]) -> EarningsStats | None:
    if not moves:
        return None
    sizes = [abs(m.move_pct) for m in moves]
    return EarningsStats(
        count=len(moves),
        mean_abs=statistics.fmean(sizes),
        median_abs=statistics.median(sizes),
        max_abs=max(sizes),
        up=sum(1 for m in moves if m.move_pct > 0),
        down=sum(1 for m in moves if m.move_pct < 0),
    )


def estimate_next(history: list[EarningsEvent], today: date) -> EarningsEvent | None:
    """Guess the next report date from the usual gap between past reports."""
    days = sorted({e.day for e in history}, reverse=True)
    if not days:
        return None
    gaps = [(a - b).days for a, b in zip(days, days[1:]) if 60 <= (a - b).days <= 130]
    gap = round(statistics.median(gaps)) if gaps else 91
    guess = days[0] + timedelta(days=gap)
    if guess < today:
        # More than a full cycle with no report on record: the history is stale.
        return None
    timings = [e.timing for e in sorted(history, key=lambda e: e.day, reverse=True)[:4]]
    timing = max(set(timings), key=timings.count)
    return EarningsEvent(day=guess, timing=timing, confirmed=False, source="estimated from past reports")


def spans(expiry: date, event: EarningsEvent) -> bool:
    """True if a contract expiring on `expiry` is still open when the stock reacts."""
    return expiry >= event.day if event.timing == "pre" else expiry > event.day


def implied_move(
    spot: float,
    atm: dict[date, AtmPoint],
    event: EarningsEvent,
    realized: float | None,
    today: date,
) -> ImpliedMove | None:
    """Size of the earnings move the options are pricing.

    The first expiration after the report carries ordinary day-to-day
    volatility plus the report. Ordinary volatility is estimated and
    subtracted, leaving the variance attributed to the report, which is then
    converted to an average absolute move so it compares directly with the
    average of past reactions.

    Ordinary volatility comes from, in order of preference: the last
    expiration before the report; the forward volatility between the first two
    expirations after it (both contain the report, so their difference does
    not); recent realized volatility. Realized volatility usually runs below
    implied, so that last resort tends to overstate the report's share.
    """
    after = sorted(e for e in atm if spans(e, event))
    if not after:
        return None
    expiry = after[0]
    point = atm[expiry]
    if point.straddle is None:
        return None

    t = years_to_expiry(expiry, today)
    total_var = point.iv * point.iv * t
    baseline, source = None, "none"

    before = sorted(e for e in atm if not spans(e, event) and (e - today).days >= 3)
    if before:
        baseline, source = atm[before[-1]].iv, "the last expiry before the report"
    elif len(after) > 1:
        t_next = years_to_expiry(after[1], today)
        forward_var = (atm[after[1]].iv ** 2 * t_next - total_var) / (t_next - t)
        if forward_var > 0:
            baseline, source = math.sqrt(forward_var), "the next expiry after it"
    if baseline is None and realized:
        baseline, source = realized, "recent realized volatility"
    if baseline is None:
        return ImpliedMove(expiry, point.straddle / spot, None, 0.0, source)

    event_var = total_var - baseline * baseline * max(t - 1.0 / 365.0, 0.0)
    event_move = math.sqrt(event_var) * math.sqrt(2.0 / math.pi) if event_var > 0 else 0.0
    return ImpliedMove(
        expiry=expiry,
        straddle_pct=point.straddle / spot,
        event_move_pct=event_move,
        baseline_vol=baseline,
        baseline_source=source,
    )
