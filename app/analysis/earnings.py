"""Earnings context: how the stock moved on past reports and what is priced in now."""

from __future__ import annotations

import math
import statistics
from bisect import bisect_left, bisect_right
from datetime import date, timedelta

from app.analysis.contracts import TRADING_DAY, AtmPoint, years_to_expiry
from app.models import Bar, EarningsEvent, EarningsStats, ImpliedMove, Reaction

_MAX_GAP = timedelta(days=6)  # skip an event if the price history has a hole around it
# An expiry further than this past a report may contain the following report too.
NEXT_REPORT_DAYS = 75


def _window(event: EarningsEvent, days: list[date]) -> tuple[int, int] | None:
    """Indexes of the last close before the report takes effect and the first close after.

    A report before the open moves the stock that same day; one after the close
    moves it the next trading day. When the timing is unknown the window runs
    from the close before the report day to the close after it, which covers
    both cases.
    """
    if event.timing == "post":
        before, after = bisect_right(days, event.day) - 1, bisect_right(days, event.day)
    elif event.timing == "pre":
        before, after = bisect_left(days, event.day) - 1, bisect_left(days, event.day)
    else:
        before, after = bisect_left(days, event.day) - 1, bisect_right(days, event.day)
    if before < 0 or after >= len(days):
        return None
    if event.day - days[before] > _MAX_GAP or days[after] - event.day > _MAX_GAP:
        return None
    return before, after


def reactions(events: list[EarningsEvent], bars: list[Bar]) -> list[Reaction]:
    """Close-to-close move across each past report, newest first."""
    days = [b.day for b in bars]
    out: list[Reaction] = []
    for event in events:
        window = _window(event, days)
        if window is None:
            continue
        before, after = window
        out.append(
            Reaction(
                day=event.day,
                timing=event.timing,
                move_pct=bars[after].close / bars[before].close - 1.0,
                surprise_pct=event.surprise_pct,
                sessions=after - before,
            )
        )
    return sorted(out, key=lambda r: r.day, reverse=True)


def reaction_days(events: list[EarningsEvent], bars: list[Bar]) -> set[date]:
    """Trading days whose return carries an earnings reaction."""
    days = [b.day for b in bars]
    out: set[date] = set()
    for event in events:
        window = _window(event, days)
        if window is not None:
            out.update(days[window[0] + 1 : window[1] + 1])
    return out


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
    """True if a contract expiring on `expiry` may still be open when the stock reacts.

    A report after the close moves the stock the next day, so a contract expiring
    on the report day misses it. When the timing is unknown the report day
    counts, because the report may come before the open; flagging a report that
    turns out to miss the contract is the safer mistake.
    """
    return expiry > event.day if event.timing == "post" else expiry >= event.day


def reaction_day(event: EarningsEvent) -> date:
    """The first session whose close reflects the report.

    A report before the open is felt that day. One after the close, or of
    unknown timing, is taken as felt the next weekday, so the day is never
    placed too early. Market holidays are not accounted for.
    """
    if event.timing == "pre":
        return event.day
    day = event.day + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def _ambiguous(expiry: date, event: EarningsEvent) -> bool:
    """An expiry on the report day when nobody knows whether the report comes first."""
    return event.timing == "unknown" and expiry == event.day


def implied_move(
    spot: float,
    atm: dict[date, AtmPoint],
    event: EarningsEvent,
    ordinary: float | None,
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
    not), as long as the second comes before the following report; the stock's
    recent realized volatility outside earnings (`ordinary`). Realized
    volatility usually runs below implied, so that last resort tends to
    overstate the report's share.

    An expiry on the report day with the timing unknown may or may not contain
    the report, so it is used on neither side.
    """
    after = sorted(e for e in atm if spans(e, event) and not _ambiguous(e, event))
    if not after:
        return None
    expiry = after[0]
    point = atm[expiry]
    straddle_pct = None if point.straddle is None else point.straddle / spot

    t = years_to_expiry(expiry, today)
    total_var = point.iv * point.iv * t
    baseline, source = None, "none"

    before = sorted(e for e in atm if not spans(e, event) and (e - today).days >= 3)
    later = [e for e in after[1:] if (e - event.day).days < NEXT_REPORT_DAYS]
    if before:
        baseline, source = atm[before[-1]].iv, "the last expiry before the report"
    elif later:
        t_next = years_to_expiry(later[0], today)
        forward_var = (atm[later[0]].iv ** 2 * t_next - total_var) / (t_next - t)
        if forward_var > 0:
            baseline, source = math.sqrt(forward_var), "the next expiry after it"
    if baseline is None and ordinary:
        baseline, source = ordinary, "recent realized volatility"
    if baseline is None:
        return ImpliedMove(expiry, straddle_pct, None, 0.0, source)

    # The report takes the place of one ordinary trading day.
    event_var = total_var - baseline * baseline * max(t - TRADING_DAY, 0.0)
    event_move = math.sqrt(event_var) * math.sqrt(2.0 / math.pi) if event_var > 0 else 0.0
    return ImpliedMove(
        expiry=expiry,
        straddle_pct=straddle_pct,
        event_move_pct=event_move,
        baseline_vol=baseline,
        baseline_source=source,
        event_var=max(event_var, 0.0),
    )
