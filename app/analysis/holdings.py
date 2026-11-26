"""A position already held, set against the exit rules.

A scan asks what might be worth opening. This asks what comes after: of what
is held, what do the exit rules say today? The rules are the ones the exit plan
already draws (`exits.PROFIT_TARGET`, `exits.TIME_STOP_DTE`), with two facts
beside them: an earnings report close at hand with the contract still open over
it, and the stock on the wrong side of a short option's strike.

Four things keep the page honest:

* A result is what closing now would give. A holder sells at the bid and a
  seller buys back at the ask, so the spread is counted, as it is in the exit
  plan.
* A check that cannot be made says so. It never passes quietly: with no
  earnings date, a report before expiry cannot be ruled out.
* A position is always listed, priced or not. Whatever is missing is left
  empty and the reason is given.
* A rule that is reached is a fact about the position, not an instruction. The
  profit target and the time stop are rules of thumb, and nothing here says
  what to do about them.

Everything here is arithmetic on values handed in. `Scanner.review` does the
fetching.
"""

from __future__ import annotations

from datetime import date, timedelta

from app.analysis import exits
from app.analysis.earnings import reaction_day, spans
from app.models import ContractAnalysis, ExitPlan, Holding, OptionQuote, Position, PositionCheck, UnderlyingContext

LENS = {"long": "buy", "short": "sell"}  # the scan's word for each side
# A report this many days off or nearer, with the contract still open over it, is pointed out.
EARNINGS_SOON = 7
SHARES = 100
# What each rule is called on a page. Served by /api/status.
RULES: dict[str, str] = {
    "profit_target": "Profit target",
    "time_stop": "Time stop",
    "earnings": "Earnings",
    "strike": "Strike",
    "expired": "Expired",
}
_TIMING = {"pre": "before the open", "post": "after the close", "unknown": "time unknown"}


def _day(day: date) -> str:
    return f"{day:%b} {day.day}"


def _in_days(days: int) -> str:
    return "today" if days == 0 else "tomorrow" if days == 1 else f"in {days} days"


def result(p: Position, quote: OptionQuote | None) -> tuple[float, float, float] | None:
    """What closing now gives: (price per share, dollars across every contract, share of the premium).

    None when the side that closing needs is not quoted.
    """
    price = (quote.bid if p.side == "long" else quote.ask) if quote else None
    if not price or price <= 0:
        return None
    gain = price - p.entry_price if p.side == "long" else p.entry_price - price
    return price, gain * SHARES * p.quantity, gain / p.entry_price


def profit_target(p: Position, ask: float | None) -> PositionCheck:
    """A seller's target: buying back costs no more than what is left after keeping `exits.PROFIT_TARGET`."""
    share = f"{exits.PROFIT_TARGET:.0%}"
    target = p.entry_price * (1.0 - exits.PROFIT_TARGET)
    aim = f"The {share} target is ${target:.2f} or less."
    if not ask or ask <= 0:
        return PositionCheck("profit_target", "unknown", "profit target unknown", f"There is no ask to buy back at. {aim}")
    kept = (p.entry_price - ask) / p.entry_price
    if ask <= target:
        text = f"Buying back at ${ask:.2f} keeps {kept:.0%} of the ${p.entry_price:.2f} collected. {aim}"
        return PositionCheck("profit_target", "reached", "profit target reached", text)
    if kept >= 0:
        text = f"Buying back at ${ask:.2f} would keep {kept:.0%} of the ${p.entry_price:.2f} collected. {aim}"
    else:
        text = f"Buying back at ${ask:.2f} costs ${ask - p.entry_price:.2f} a share more than the ${p.entry_price:.2f} collected. {aim}"
    return PositionCheck("profit_target", "clear", "profit target not reached", text)


def time_stop(p: Position, today: date) -> PositionCheck:
    """Inside the last `exits.TIME_STOP_DTE` days, for either side."""
    dte, window = (p.expiry - today).days, exits.TIME_STOP_DTE
    if dte > window:
        stop = p.expiry - timedelta(days=window)
        text = f"{dte} days to expiry. The {window}-day time stop is {_day(stop)}."
        return PositionCheck("time_stop", "clear", "outside time stop", text)
    left = "Expires today." if dte == 0 else f"{dte} day{'' if dte == 1 else 's'} to expiry, inside the last {window}."
    if p.side == "short":
        why = "This close to expiry an option's price swings harder with the stock, which is why sellers usually close or roll by now."
    else:
        why = "Time decay is fastest over these last weeks."
    return PositionCheck("time_stop", "reached", "inside time stop", f"{left} {why}")


def earnings(p: Position, ctx: UnderlyingContext | None, today: date, margin: timedelta = timedelta(0)) -> PositionCheck:
    """Whether the next report is close, with the contract still open when the stock reacts.

    An estimated date is given `margin` of room on both counts, as the
    skip-earnings filter gives it: a company's reports move by days from one
    quarter to the next.
    """
    if ctx is None:
        return PositionCheck("earnings", "unknown", "earnings not checked", "The stock's earnings dates could not be looked up.")
    event = ctx.next_earnings
    if event is None:
        if ctx.earnings_unknown:
            text = "No earnings date is known for this stock, so a report before expiry cannot be ruled out."
            return PositionCheck("earnings", "unknown", "earnings date unknown", text)
        return PositionCheck("earnings", "clear", "no report on record", "No earnings report is on record for this stock, which is normal for a fund.")

    days = (event.day - today).days
    estimated = "" if event.confirmed else " The date is an estimate from past reports, and reports move by days."
    when = f"{_day(event.day)}, {_TIMING.get(event.timing, event.timing)}"
    if reaction_day(event) <= today:
        return PositionCheck("earnings", "clear", "report already out", f"Reported {when}. Today's prices already reflect it.")
    room = timedelta(0) if event.confirmed else margin
    if not spans(p.expiry, event):
        if event.confirmed or p.expiry < event.day - room:
            text = f"Expires before the next report ({when}).{estimated}"
            return PositionCheck("earnings", "clear", "expires before earnings", text)
        # An estimated date just past the expiry: it cannot be shown clear of the report.
        text = f"The next report is estimated for {_day(event.day)}, close after this contract expires, and reports move by days. It may still be open then."
        return PositionCheck("earnings", "unknown", "earnings near expiry (est.)", text)
    if days > EARNINGS_SOON + room.days:
        text = f"The next report is {when} ({_in_days(days)}), before this contract expires.{estimated}"
        return PositionCheck("earnings", "clear", "report before expiry", text)
    label = f"earnings {'today' if days <= 0 else f'in {days}d'}{'' if event.confirmed else ' (est.)'}"
    text = f"Reports {when} ({_in_days(max(days, 0))}), and this contract is still open then.{estimated}"
    return PositionCheck("earnings", "reached", label, text)


def strike(p: Position, spot: float | None) -> PositionCheck:
    """Where the stock is against the strike. Only a short option is pointed out for being through it."""
    if not spot or spot <= 0:
        return PositionCheck("strike", "unknown", "stock price unknown", "There is no stock price to set beside the strike.")
    # Measured as the exit plan measures it: the strike against the stock price.
    gap = p.strike / spot - 1.0
    if abs(gap) < 0.0005:
        where = f"The ${p.strike:g} strike is at the stock price (${spot:,.2f})"
    else:
        where = f"The ${p.strike:g} strike is {abs(gap):.1%} {'above' if gap > 0 else 'below'} the stock price (${spot:,.2f})"
    in_the_money = spot > p.strike if p.kind == "call" else spot < p.strike
    if p.side == "short":
        if in_the_money:
            text = f"{where}, so this {p.kind} is in the money and can be assigned."
            return PositionCheck("strike", "reached", "through the strike", text)
        return PositionCheck("strike", "clear", "strike not reached", f"{where}.")
    if in_the_money:
        return PositionCheck("strike", "clear", "in the money", f"{where}, so this {p.kind} is in the money.")
    way = "rise" if p.kind == "call" else "fall"
    text = f"{where}. The stock has to {way} past it for this {p.kind} to be worth anything at expiry."
    return PositionCheck("strike", "clear", "out of the money", text)


def assess(
    p: Position,
    today: date,
    *,
    spot: float | None = None,
    quote: OptionQuote | None = None,
    ctx: UnderlyingContext | None = None,
    contract: ContractAnalysis | None = None,
    plan: ExitPlan | None = None,
    margin: timedelta = timedelta(0),
    note: str | None = None,
) -> Holding:
    """One position beside whatever of today's numbers could be read. Every argument after `today` may be missing."""
    dte = (p.expiry - today).days
    if dte < 0:
        text = f"Expired on {_day(p.expiry)}. It is no longer quoted, so nothing here can be checked. Remove it once it has settled."
        return Holding(p, dte, [PositionCheck("expired", "reached", "expired", text)])
    checks = [profit_target(p, quote.ask if quote else None)] if p.side == "short" else []
    checks += [time_stop(p, today), earnings(p, ctx, today, margin), strike(p, spot)]
    closing = result(p, quote)
    return Holding(
        position=p,
        dte=dte,
        checks=checks,
        bid=quote.bid if quote else None,
        ask=quote.ask if quote else None,
        quote_time=quote.quote_time if quote else None,
        close_price=closing[0] if closing else None,
        pnl=closing[1] if closing else None,
        pnl_pct=closing[2] if closing else None,
        contract=contract,
        plan=plan,
        note=note,
    )
