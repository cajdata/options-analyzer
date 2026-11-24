"""Exit plan: what happens to a contract between now and expiry if the stock stands still.

A scan scores a contract as held to expiry. This module answers the question
that comes next: when would you close it? It reprices the contract on later
dates with the stock unchanged, which isolates the two things time does to an
option:

* **Time decay.** Ordinary variance runs out at a steady rate, so the option
  loses value every day.
* **Volatility crush.** An earnings report adds a lump of variance that vanishes
  the session it lands. The option's price steps down that day even if the stock
  does nothing.

Both come from one model. The contract's implied variance is split into the part
the options price for the report and the ordinary remainder, and each is run
down on its own schedule.

Three rules keep the output honest:

* Closing early crosses the bid-ask spread a second time, so every close price
  here is the bid for a holder and the ask for someone buying back.
* "When decay alone reaches the target" is a date on a path where the stock does
  not move. It is not the chance of reaching the target; the chance of profit at
  expiry says nothing about what happens along the way.
* The profit target and time stop are rules of thumb used by option sellers. They
  manage risk; they are not signals.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

from app import pricing
from app.analysis.contracts import TRADING_DAY
from app.analysis.earnings import reaction_day
from app.models import Checkpoint, ContractAnalysis, EarningsExit, ExitPlan, UnderlyingContext

PROFIT_TARGET = 0.50  # a seller's usual target: buy back once half the premium is kept
TIME_STOP_DTE = 21  # and the usual time stop: close or roll this many days before expiry
CHECK_DAYS = (7, 14, 21)  # how far ahead the standing-still path is sampled
SHARES = 100
# The report's share of a contract's variance is capped, so the ordinary part
# never goes to zero or below when the two estimates disagree.
MAX_EVENT_SHARE = 0.90


class _Model:
    """Reprices one contract on a later date, with the stock at `spot` unless told otherwise."""

    def __init__(self, c: ContractAnalysis, ctx: UnderlyingContext, rate: float, today: date) -> None:
        self.c, self.spot, self.rate, self.today = c, ctx.price, rate, today
        t = max(c.dte, 0.5) / 365.0
        # Start from the quoted mid, so the path begins exactly at today's price.
        vol = pricing.implied_vol(c.kind, c.mid, self.spot, c.strike, t, rate) or c.iv
        total = vol * vol * t

        self.report: date | None = None  # first session that reflects the report, if modelled
        self.event_var = 0.0
        event = ctx.next_earnings
        if event and c.spans_earnings:
            day = reaction_day(event)
            # What the options price for the report, else what past reports have delivered.
            priced = ctx.implied_move.event_var if ctx.implied_move else None
            known = priced or ctx.event_var
            if known and today < day <= c.expiry:
                self.report = day
                self.event_var = min(known, MAX_EVENT_SHARE * total)
        self.report_ignored = bool(event and c.spans_earnings and self.report is None)
        ordinary_time = t - (TRADING_DAY if self.report else 0.0)
        self.ordinary_rate = (total - self.event_var) / max(ordinary_time, 1e-9)  # variance per year

    def intrinsic(self, spot: float | None = None) -> float:
        s = self.spot if spot is None else spot
        return max(s - self.c.strike, 0.0) if self.c.kind == "call" else max(self.c.strike - s, 0.0)

    def value(self, day: date, spot: float | None = None) -> float:
        """Mid price at the close of `day`."""
        s = self.spot if spot is None else spot
        left = (self.c.expiry - day).days
        if left <= 0:
            return self.intrinsic(s)
        t = left / 365.0
        ahead = self.report is not None and day < self.report
        # Until it lands, the report takes the place of one ordinary trading day.
        variance = self.ordinary_rate * max(t - (TRADING_DAY if ahead else 0.0), 0.0)
        variance += self.event_var if ahead else 0.0
        vol = math.sqrt(max(variance, 1e-12) / t)
        # The formula is for European exercise and can price a deep in-the-money put
        # below what exercising it would pay. An American option never trades there.
        return max(pricing.price(self.c.kind, s, self.c.strike, t, self.rate, vol), self.intrinsic(s))


def build(c: ContractAnalysis, ctx: UnderlyingContext, lens: str, rate: float, today: date) -> ExitPlan | None:
    """The exit plan for the buyer ("buy") or the seller ("sell") of `c`."""
    entry = c.ask if lens == "buy" else c.bid
    if entry <= 0 or c.dte < 1:
        return None
    model = _Model(c, ctx, rate, today)
    half_spread = c.spread_pct / 2.0

    def close_price(day: date, spot: float | None = None) -> float:
        """What closing fetches (a holder sells at the bid) or costs (a seller buys at the ask)."""
        value = model.value(day, spot)
        if day >= c.expiry:
            return value  # settles at exercise value; there is no spread to cross
        return value * (1.0 - half_spread) if lens == "buy" else value * (1.0 + half_spread)

    def checkpoint(day: date, label: str) -> Checkpoint:
        price = close_price(day)
        gain = price - entry if lens == "buy" else entry - price
        return Checkpoint(
            day=day,
            days_held=(day - today).days,
            label=label,
            value=model.value(day),
            close_price=price,
            pnl=gain * SHARES,
            pnl_pct=gain / entry,
        )

    # Later entries replace earlier ones on the same date, so the specific label wins.
    marks: dict[date, str] = {today + timedelta(days=n): f"in {n} days" for n in CHECK_DAYS if n < c.dte}
    time_stop = c.expiry - timedelta(days=TIME_STOP_DTE)
    time_stop_day = time_stop if time_stop > today else None
    if time_stop_day:
        marks[time_stop_day] = "time stop"
    if model.report:
        eve = model.report - timedelta(days=1)
        if eve > today:
            marks[eve] = "before earnings"
        marks[model.report] = "after earnings"
    marks[c.expiry] = "at expiry"

    days_ahead = [today + timedelta(days=n) for n in range(1, c.dte)]
    plan = ExitPlan(
        lens=lens,
        entry_price=entry,
        checkpoints=[checkpoint(day, marks[day]) for day in sorted(marks)],
        earnings=_earnings(model, close_price, entry, today),
        strike_distance_pct=c.strike / model.spot - 1.0,
        time_stop_day=time_stop_day,
        notes=_notes(c, ctx, lens, model),
    )

    if lens == "buy":
        # When half of what was paid for time has gone, with the stock unchanged.
        time_value = model.value(today) - model.intrinsic()
        if time_value > 1e-9:
            plan.half_life_day = next(
                (d for d in days_ahead + [c.expiry] if model.value(d) - model.intrinsic() <= 0.5 * time_value), None
            )
    else:
        plan.target_price = entry * (1.0 - PROFIT_TARGET)
        plan.target_day = next((d for d in days_ahead if close_price(d) <= plan.target_price), None)
    return plan


def _earnings(model: _Model, close_price, entry: float, today: date) -> EarningsExit | None:
    """The step in price across the report, and the move that would undo it."""
    if model.report is None:
        return None
    eve = max(today, model.report - timedelta(days=1))
    before, after = model.value(eve), model.value(model.report)
    if before <= 0:
        return None

    # Stock move, in the direction that makes the option worth more, at which closing
    # the session after the report is back to the entry price. For a holder that is
    # the move needed to recover; for a seller it is the move the position can absorb.
    is_call = model.c.kind == "call"
    low, high = (model.spot, model.spot * 3.0) if is_call else (model.spot * 0.05, model.spot)
    worth_more = high if is_call else low
    move: float | None
    if close_price(model.report) >= entry:
        move = 0.0
    elif close_price(model.report, worth_more) < entry:
        move = None  # out of reach within any plausible move
    else:
        for _ in range(60):
            middle = 0.5 * (low + high)
            reached = close_price(model.report, middle) >= entry
            if reached == is_call:  # a call gains as the stock rises, a put as it falls
                high = middle
            else:
                low = middle
        move = 0.5 * (low + high) / model.spot - 1.0
    return EarningsExit(
        report_day=model.report,
        value_before=before,
        value_after=after,
        crush_pct=after / before - 1.0,
        move_to_breakeven_pct=move,
    )


def _notes(c: ContractAnalysis, ctx: UnderlyingContext, lens: str, model: _Model) -> list[str]:
    notes = []
    if ctx.earnings_unknown:
        notes.append("No earnings date is known for this stock, so a report in this period is not modelled.")
    elif model.report_ignored:
        notes.append("This contract spans an earnings report that could not be sized, so these figures ignore it.")
    elif model.report and ctx.next_earnings and not ctx.next_earnings.confirmed:
        notes.append("The earnings date is an estimate, so the before and after rows may be off by days.")
    if lens == "sell" and c.dte <= TIME_STOP_DTE:
        notes.append(f"This contract is already inside the {TIME_STOP_DTE}-day window the time stop is meant to avoid.")
    if lens == "sell" and c.kind == "call":
        notes.append("Dividends are not modelled. A short call can be assigned early just before the stock goes ex-dividend.")
    return notes
