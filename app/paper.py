"""Paper trading: orders the person sends to Alpaca's paper account, and the journal kept beside them.

Everything through phase 3 only read market data. This is where the app first
sends something, so what it may send is narrow and written down here:

* **Every order is the person's.** An order is written out as a ticket first,
  which sends nothing. Sending takes that ticket's id, and a ticket is used
  once. No scan, review or refresh ever sends or cancels an order.
* **Paper only.** `app/broker.py` knows one host. With keys to a live account
  (`ALPACA_PAPER` off) there is no broker at all, and the desk says why.
* **The journal is written before the order leaves**, so an order is never out
  there with no record of it. An order the account turns down outright is
  taken back out. One whose answer was lost stays, and the next review asks the
  account whether it arrived.
* **Opening goes through a scan.** A paper trade is opened from a contract a
  scan ranked, so the journal always has what the scan showed for it.

A review (`PaperDesk.review`) brings the journal up to date with the account,
then sets what the account holds against the exit rules, the way the
positions page does for real holdings. Paper and real are never mixed: the
positions file is not read here, and nothing here is written to it.
"""

from __future__ import annotations

import math
import random
import threading
import uuid
from collections import OrderedDict
from dataclasses import replace
from datetime import date, datetime, time as clock, timedelta, timezone
from typing import Callable
from zoneinfo import ZoneInfo

from app import journal as journals
from app import occ, positions
from app.analysis import holdings, signals
from app.broker import AlpacaPaper, Broker, BrokerError, DemoBroker
from app.models import (
    ClosingOrder,
    Exit,
    Order,
    PaperHolding,
    PaperReview,
    PositionCheck,
    PositionsReview,
    ScanResult,
    Ticket,
    Trade,
)

ET = ZoneInfo("America/New_York")
PREFIX = "sc"  # every order this app sends carries an id that starts with it
TICKET_MINUTES = 10  # how long an order written out waits to be confirmed
KEPT_TICKETS = 20
MAX_CONTRACTS = 100  # in one order
MAX_LIMIT = 10_000.0  # dollars a share
LIVE_KEYS = (
    "ALPACA_PAPER is off, so these are the keys of a live account. Orders are sent to a paper account only, "
    "so none can be sent. The journal can still be read."
)
CUT_SHORT = (
    "The paper account has more orders since the oldest trade still open than this app reads at once, so the list of "
    "orders still working may be short, and contracts that have left the account are left as they are in the journal. "
    "Orders sent from here are still found by name."
)
NO_PRICE = journals.NO_PRICE  # how the remark on an order that filled with no price begins
EXPIRED = (
    "That contract has expired, so no order can close it. Alpaca takes it off the account once it has settled, "
    "and the journal then records how it ended."
)
SHARES = 100


class PaperError(RuntimeError):
    """An order cannot be written or sent. The message is written for the user."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


def name(t) -> str:
    """A contract in a few words: CRUX Dec 18 395 put."""
    return f"{t.underlying} {t.expiry:%b %d} {t.strike:g} {t.kind}"


def open_id(trade: Trade) -> str:
    return f"{PREFIX}-{trade.id}-o"


unpriced = journals.unpriced  # whether a trade is waiting on an order that filled with no price


def held_id(symbol: str, side: str) -> str:
    """What a position the account holds goes by in a review. The side is part of it, so no two rows ever share one."""
    return f"{symbol}:{side}"


# -- Bringing the journal up to date with the account ---------------------------------------


def reconcile(
    trades: list[Trade],
    orders: list[Order],
    held: list[PaperHolding] | None,
    today: date,
    now: datetime,
    *,
    lookup: Callable[[str], Order | None] | None = None,
    close_on: Callable[[str, date], float | None] | None = None,
) -> list[Trade]:
    """Set the journal beside the account's orders and positions, and return the trades that changed.

    `orders` are the account's since before the oldest trade still in play, and
    `held` is what it holds, or None when that could not be read. Three passes:

    1. An opening order that has ended makes its trade open, at the average
       fill and for as many contracts as filled, or unfilled.
    2. A closing order that has ended becomes an exit, with the rules that were
       reached when it was sent.
    3. Contracts the journal has as open and the account no longer holds are
       accounted for: by an order this app did not send, else at expiry, else
       as gone with no price on record.

    An order still working changes nothing, however much of it has filled. An
    order that is missing from the list is asked after by name (`lookup`).
    Only one whose answer was lost, and that the account then says it never
    had, is taken to have never arrived. One the account confirmed is waited
    for whatever a later answer says, never written off. And an order that
    ended with contracts filled and no price for them is waited for too:
    nothing is made up for it.

    The account's positions are read before its orders (`PaperDesk.review`),
    so `held` can be older than `orders`. A trade this call has just found
    open is therefore left out of the third pass: the position its fill made
    may not have been there to read yet.
    """
    by_client = {o.client_id: o for o in orders if o.client_id}
    touched: dict[str, Trade] = {}
    just_opened: set[str] = set()

    def find(client_id: str, confirmed: bool) -> tuple[Order | None, bool]:
        """The order, and whether the answer is certain."""
        if client_id in by_client:
            return by_client[client_id], True
        if lookup is None:
            return None, False
        try:
            order = lookup(client_id)
        except BrokerError:
            return None, False
        # The account took this order once. If it now says it has none by that name, that is no answer to go by.
        return order, order is not None or not confirmed

    def waits(t: Trade, order: Order) -> None:
        """An order that ended with contracts filled and no price on it. The trade stays as it is, and says why."""
        said = f"{NO_PRICE} {order.filled} of this order's contracts filled and gives no price for them. It is asked again at the next refresh."
        if t.note != said:
            t.note = said
            touched[t.id] = t

    def priced(t: Trade) -> None:
        """The price has come, so the remark that it was missing goes."""
        if unpriced(t):
            t.note = None

    for t in trades:
        if t.state != "ordered":
            continue
        order, certain = find(open_id(t), bool(t.open_order))
        if order is None:
            if certain:
                t.state, t.note = "unfilled", "The paper account has no record of this order: it never arrived."
                touched[t.id] = t
            continue
        if t.open_order != order.id:
            t.open_order = order.id
            touched[t.id] = t
        if order.working:
            continue
        price = _fill(order)
        if order.filled > 0 and price is None:
            waits(t, order)
            continue
        priced(t)
        if order.filled > 0:
            if order.filled < t.quantity:
                t.note = f"{order.filled} of the {t.quantity} contracts ordered were filled."
            t.state, t.quantity, t.entry_price = "open", order.filled, price
            t.opened = order.filled_at or order.submitted or now
            just_opened.add(t.id)
        else:
            t.state, t.note = "unfilled", f"The order ended {order.status} with nothing filled."
        touched[t.id] = t

    for t in trades:
        if t.state != "open" or t.closing is None:
            continue
        order, certain = find(t.closing.client_id, bool(t.closing.order))
        if order is None:
            if certain:
                t.closing = None  # it never arrived, so the trade is simply still open
                touched[t.id] = t
            continue
        if t.closing.order != order.id:
            t.closing = replace(t.closing, order=order.id)
            touched[t.id] = t
        if order.working:
            continue
        price = _fill(order)
        if order.filled > 0 and price is None:
            waits(t, order)  # and the closing order stays on the trade, so nothing is concluded about its contracts
            continue
        priced(t)
        if order.filled > 0:
            count = min(order.filled, t.remaining)
            t.exits.append(Exit(count, price, _when(order, now), "order", t.closing.rules, order.id))
        t.closing = None
        touched[t.id] = t

    if held is not None:
        for t in _unheld(trades, orders, held, today, now, close_on, just_opened):
            touched[t.id] = t
    # Where each one stands is worded again here, so a caller never reads the words of the state it was in before.
    return [journals.settle(t) for t in touched.values()]


def _fill(order: Order) -> float | None:
    """The price an order's contracts filled at. None when the account gives none that can be a price.

    The bounds are a position's, at both ends, which are the journal's too (`journals._price`). A price the
    journal would not read back must never be written to it.
    """
    price = order.fill_price
    return round(price, 4) if price is not None and positions.MIN_PRICE <= price <= positions.MAX_PRICE else None


def _when(order: Order, now: datetime) -> datetime:
    """When an order's contracts filled. One that filled in part and was then cancelled has no fill time, so it goes by when it was sent."""
    return order.filled_at or order.submitted or now


def _unheld(
    trades: list[Trade],
    orders: list[Order],
    held: list[PaperHolding],
    today: date,
    now: datetime,
    close_on: Callable[[str, date], float | None] | None,
    skip: set[str] = frozenset(),
) -> list[Trade]:
    """Account for contracts the journal has as open and the account no longer holds. Returns the trades that changed.

    `skip` names the trades that were found open in this same look, whose positions may not have been read yet.
    """
    have = {(h.symbol, h.side): h.quantity for h in held}
    # An order can close more than one trade, and part of it in one look and the rest in the next. So what
    # counts is how many of its contracts the journal has already taken, not whether it was ever used.
    taken: dict[str, int] = {}
    for t in trades:
        for e in t.exits:
            if e.order:
                taken[e.order] = taken.get(e.order, 0) + e.quantity
    # Orders this app did not send, each with the contracts it filled that no trade has taken yet. One that
    # says it opens a position is not one that closes: the account's word for it is used when it gives one.
    spare = {
        o.id: o.filled - taken.get(o.id, 0)
        for o in orders
        if not o.working and o.filled > 0 and _fill(o) is not None and o.intent != "open" and not o.client_id.startswith(PREFIX + "-")
    }
    groups: dict[tuple[str, str], list[Trade]] = {}
    for t in trades:
        if t.state == "open" and t.id not in skip:
            groups.setdefault((t.symbol, t.side), []).append(t)

    changed: list[Trade] = []
    for (symbol, side), mine in groups.items():
        if any(t.closing for t in mine):
            continue  # a closing order is still working, and may have filled in part: wait for it to end
        missing = sum(t.remaining for t in mine) - have.get((symbol, side), 0)
        closes = "sell" if side == "long" else "buy"
        # The oldest trade on a contract is taken to be the one that was closed first.
        for t in sorted(mine, key=lambda t: t.opened or t.ordered):
            if missing <= 0:
                break
            before = len(t.exits)
            for o in sorted(orders, key=lambda o: _when(o, now)):
                count = min(spare.get(o.id, 0), t.remaining, missing)
                if count <= 0 or o.symbol != symbol or o.side != closes or _when(o, now) < (t.opened or t.ordered):
                    continue
                t.exits.append(Exit(count, _fill(o), _when(o, now), "outside", None, o.id))
                spare[o.id] -= count
                missing -= count
            count = min(t.remaining, missing)
            if count > 0 and t.expiry <= today:
                # Held to expiry. What it was worth then is what exercising it paid at that day's close.
                close = close_on(t.underlying, t.expiry) if close_on else None
                if close is not None:
                    worth = max(close - t.strike, 0.0) if t.kind == "call" else max(t.strike - close, 0.0)
                    t.exits.append(Exit(count, round(worth, 4), datetime.combine(t.expiry, clock(16, 0), ET), "expiry"))
                    t.note = f"Held to expiry, with {t.underlying} closing at ${close:,.2f}. " + (
                        "It expired worthless."
                        if worth <= 0
                        else f"Exercising it paid {holdings.dollars(worth)} a share. An option that expires in the money is exercised or "
                        "assigned, which leaves shares in the paper account, or a debt of them. This app does not follow those."
                    )
                    missing -= count
                # With no close for that day yet, the trade waits as it is.
            elif count > 0:
                t.exits.append(Exit(count, None, now, "unknown"))
                t.note = (
                    "These contracts left the paper account with no order on record to say what they closed at. "
                    "Resetting the paper account does that."
                )
                missing -= count
            if len(t.exits) > before:
                changed.append(journals.settle(t))
    return changed


# -- The desk -------------------------------------------------------------------------------


class PaperDesk:
    """Writes orders out, sends the ones the person confirms, and keeps the journal beside the account."""

    def __init__(
        self,
        journal: journals.Journal,
        scanner=None,
        broker: Broker | None = None,
        why: str | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        seed: Callable[["PaperDesk"], None] | None = None,
    ) -> None:
        self.journal = journal
        self.scanner = scanner  # None with no market data: the journal can still be read
        self.broker = broker  # None when no order can be sent, and `why` says why
        self.why = why
        self._clock = clock
        self._seed = seed  # fills the demo's journal the first time the desk is used
        self._tickets: OrderedDict[str, Ticket] = OrderedDict()
        self._lock = threading.RLock()  # one thing at a time: a review and an order must not cross

    @property
    def available(self) -> bool:
        return self.broker is not None

    @property
    def mode(self) -> str:
        return self.scanner.market.mode if self.scanner else "unconfigured"

    def _today(self) -> date:
        return self.scanner.today if self.scanner else self._clock().astimezone(ET).date()

    def _ready(self) -> Broker:
        if self.broker is None:
            raise PaperError(self.why or "No paper account is set up.", 503)
        if self._seed:
            seed, self._seed = self._seed, None
            seed(self)
        return self.broker

    # -- Writing an order out -----------------------------------------------

    @staticmethod
    def sized(quantity, limit) -> tuple[int, float]:
        """A number of contracts and a limit as an order takes them. Raises PaperError when either is not one."""
        if type(quantity) is not int or not 1 <= quantity <= MAX_CONTRACTS:
            raise PaperError(f"An order is for a whole number of contracts from 1 to {MAX_CONTRACTS}.")
        if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or not 0.01 <= limit <= MAX_LIMIT:
            raise PaperError(f"A limit price is from $0.01 to ${MAX_LIMIT:,.0f} a share.")
        if abs(limit * 100 - round(limit * 100)) > 1e-6:
            raise PaperError("A limit price is in whole cents.")
        return quantity, round(limit, 2)

    def _keep(self, ticket: Ticket) -> Ticket:
        self._tickets[ticket.id] = ticket
        while len(self._tickets) > KEPT_TICKETS:
            self._tickets.popitem(last=False)
        return ticket

    def _common_notes(self) -> list[str]:
        notes = ["A limit order fills only if the market reaches its price. Unfilled at today's close, it ends there."]
        if self.broker is not None and self.broker.mode == "demo":
            notes.append("This is the demo: the account is made up, lives in memory, and fills an order at once or not at all.")
        else:
            notes.append("It goes to Alpaca's paper account: pretend money, and never an exchange.")
        return notes

    def _one_side(self, symbol: str, side: str, sending: bool = False) -> None:
        """Turn away an order to open one side of a contract while the journal has a trade on the other in play.

        An account holds a contract one way or the other, never both. An order to open the other side would be
        set against the trade that is there, and the journal could not tell which had ended. This is asked when
        an order is written out and again when it is sent: another order can open the other side in between.
        """
        other = "short" if side == "long" else "long"
        if any(t.symbol == symbol and t.side == other and t.state in ("ordered", "open") for t in self.journal.load()[0]):
            raise PaperError(
                f"The journal has a trade {'sold' if other == 'short' else 'bought'} to open on this contract that has not ended. "
                "An account holds a contract one way or the other, so this order would not open a second position. "
                + ("It was not sent. " if sending else "")
                + "Close that trade instead.",
                409 if sending else 422,
            )

    def write_open(self, result: ScanResult, contract: str, quantity: int, limit: float) -> Ticket:
        """An order to open one of a scan's contracts, written out and not sent."""
        with self._lock:
            self._ready()
            c = next((c for c in result.ideas if c.symbol == contract), None)
            if c is None:
                raise PaperError("That contract is not in the scan.", 404)
            if c.expiry < self._today():
                raise PaperError("That contract has expired.")
            quantity, limit = self.sized(quantity, limit)
            buying = result.lens == "buy"
            self._one_side(c.symbol, "long" if buying else "short")
            premium = limit * SHARES * quantity
            text = f"{'Buy' if buying else 'Sell'} to open {quantity} {name(c)} at {holdings.dollars(limit)} a share or {'less' if buying else 'more'}, good for today."
            if buying:
                notes = [f"At that price it costs ${premium:,.0f}."]
            elif c.kind == "put":
                notes = [
                    f"At that price it brings in ${premium:,.0f}, and the account sets aside ${c.strike * SHARES * quantity:,.0f} "
                    "in case the shares are put to it."
                ]
            else:
                notes = [
                    f"At that price it brings in ${premium:,.0f}. Alpaca takes a call sold to open only when the account holds "
                    f"{quantity * SHARES} shares of {c.underlying}, and this app does not buy shares. Without them it turns the order down."
                ]
            if result.options_feed == "indicative":
                notes.append("The quote it was written from is delayed, and the paper account fills against live prices, so it may not fill at once.")
            return self._keep(
                Ticket(
                    id=uuid.uuid4().hex[:12],
                    action="open",
                    symbol=c.symbol,
                    underlying=c.underlying,
                    kind=c.kind,
                    strike=c.strike,
                    expiry=c.expiry,
                    order_side="buy" if buying else "sell",
                    side="long" if buying else "short",
                    quantity=quantity,
                    limit=limit,
                    text=text,
                    notes=notes + self._common_notes(),
                    made=self._clock(),
                    signals=signals.snapshot(result, c),
                )
            )

    def closable(self, trade_id: str) -> Trade:
        """The trade with that id, when an order to close it can be written out. Raises PaperError saying why not."""
        with self._lock:
            self._ready()
            t = next((t for t in self.journal.load()[0] if t.id == trade_id), None)
            if t is None:
                raise PaperError("The journal has no trade with that id.", 404)
            if t.state != "open":
                raise PaperError(f"That trade is {t.state}, so there is nothing to close.")
            if t.closing:
                raise PaperError("A closing order for it is already working. Cancel that one, or wait for it to end.")
            if t.expiry < self._today():
                raise PaperError(EXPIRED)
            return t

    def write_close(self, trade_id: str, limit: float) -> Ticket:
        """An order to close what is left of an open trade, written out and not sent."""
        with self._lock:
            t = self.closable(trade_id)
            quantity, limit = self.sized(t.remaining, limit)
            selling = t.side == "long"
            # The sign is settled after rounding to whole dollars, so nothing ends "down $0".
            gain = round((limit - t.entry_price if selling else t.entry_price - limit) * SHARES * quantity)
            ends = "even" if gain == 0 else f"{'up' if gain > 0 else 'down'} ${abs(gain):,}"
            text = f"{'Sell' if selling else 'Buy'} to close {quantity} {name(t)} at {holdings.dollars(limit)} a share or {'more' if selling else 'less'}, good for today."
            notes = [f"It was {'bought' if selling else 'sold'} at {holdings.dollars(t.entry_price)}. Filled at the limit, the trade ends {ends}."]
            return self._keep(
                Ticket(
                    id=uuid.uuid4().hex[:12],
                    action="close",
                    symbol=t.symbol,
                    underlying=t.underlying,
                    kind=t.kind,
                    strike=t.strike,
                    expiry=t.expiry,
                    order_side="sell" if selling else "buy",
                    side=t.side,
                    quantity=quantity,
                    limit=limit,
                    text=text,
                    notes=notes + self._common_notes(),
                    made=self._clock(),
                    trade=t.id,
                )
            )

    # -- Sending one ----------------------------------------------------------

    def send(self, ticket_id: str) -> Trade:
        """Send the order a ticket describes. A ticket is used once, and an old one is not sent."""
        with self._lock:
            broker = self._ready()
            ticket = self._tickets.pop(ticket_id, None)
            now = self._clock()
            if ticket is None or now - ticket.made > timedelta(minutes=TICKET_MINUTES):
                raise PaperError("That order is no longer waiting to be confirmed. Write it out again.", 409)
            # A contract can expire while its order waits: one written out late on its last day and confirmed after midnight.
            if ticket.expiry < max(self._today(), now.astimezone(ET).date()):
                raise PaperError("That contract has expired since the order was written out, so it was not sent.", 409)
            return self._open(broker, ticket, now) if ticket.action == "open" else self._close(broker, ticket, now)

    def _open(self, broker: Broker, ticket: Ticket, now: datetime) -> Trade:
        # Two orders for the two sides of one contract can both be written out before either is sent. So the
        # journal is asked again here, as a closing order asks whether its trade has changed.
        self._one_side(ticket.symbol, ticket.side, sending=True)
        try:
            t = journals.new(
                ticket.underlying, ticket.kind, ticket.strike, ticket.expiry, ticket.side, ticket.quantity, ticket.limit, now, ticket.signals
            )
            self.journal.add(t)  # before the order leaves: an order is never out there with no record of it
        except ValueError as err:
            # No entry, no order. What the journal will not take is said, and never answered with a bare failure.
            raise PaperError(f"The journal could not take this order, so it was not sent: {err}.") from err
        try:
            order = broker.submit(t.symbol, ticket.order_side, "open", t.quantity, t.limit, open_id(t))
        except BrokerError as err:
            if err.refused:
                self.journal.remove(t.id)  # nothing happened, so there is nothing to keep
                raise PaperError(str(err)) from err
            raise PaperError(
                f"{err} The journal keeps the entry as ordered, and the next refresh asks the account whether it arrived.", 502
            ) from err
        reconcile([t], [order], None, self._today(), now)
        t.open_order = order.id
        self.journal.save([t])
        return t

    def _close(self, broker: Broker, ticket: Ticket, now: datetime) -> Trade:
        t = next((t for t in self.journal.load()[0] if t.id == ticket.trade), None)
        if t is None or t.state != "open" or t.closing or t.remaining != ticket.quantity:
            raise PaperError("That trade has changed since the order was written out. Refresh, and write it out again.", 409)
        # The rules that are reached now go on the order, so the journal can say at which rule a trade was closed.
        # Each closing order gets an id of its own: the account takes an id once, and a trade can need a second try.
        sent = ClosingOrder(f"{PREFIX}-{t.id}-c{uuid.uuid4().hex[:6]}", ticket.quantity, ticket.limit, now, self._rules_now(t))
        t.closing = sent
        self.journal.save([t])
        try:
            order = broker.submit(t.symbol, ticket.order_side, "close", ticket.quantity, ticket.limit, sent.client_id)
        except BrokerError as err:
            if err.refused:
                t.closing = None
                self.journal.save([t])
                raise PaperError(str(err)) from err
            raise PaperError(f"{err} The journal keeps the closing order as sent, and the next refresh asks the account whether it arrived.", 502) from err
        t.closing = replace(sent, order=order.id)
        reconcile([t], [order], None, self._today(), now)
        self.journal.save([t])
        return t

    def _rules_now(self, t: Trade) -> tuple[str, ...] | None:
        """The exit rules this trade has reached, on the quotes of this moment. None when they cannot be told.

        That is so when the review fails, and also when it answers "unknown" for a rule the journal sorts a
        closed trade by (`journals.ENDED_BY`): no ask to set beside the profit target, or a quote that is no
        price to close at. Left out of the list, such a rule would read as not reached, and a trade closed at
        its target would be filed as closed before it. An unknown check never passes, here as on the page.

        The news is left out of this review. A headline cannot change a rule, and the order is waiting to leave.
        """
        if self.scanner is None:
            return None
        try:
            held = self._position(t.symbol, t.side, t.remaining, t.entry_price, [t])
            (holding,) = self.scanner.review([held], news=False).holdings
        except Exception:  # the order is the person's to send whether or not the rules can be read
            return None
        if any(check.state == "unknown" and check.rule in journals.ENDED_BY for check in holding.checks):
            return None
        return tuple(check.rule for check in holding.checks if check.state == "reached")

    def cancel(self, order_id: str) -> None:
        with self._lock:
            try:
                self._ready().cancel(order_id)
            except BrokerError as err:
                raise PaperError(str(err), err.status if err.refused and err.status in (404, 422) else 502) from err

    # -- Looking -----------------------------------------------------------------

    def _position(self, symbol: str, side: str, quantity: int, price: float, mine: list[Trade]):
        """What the account holds of one contract, as a position the exit rules can be set against."""
        root, expiry, kind, strike = occ.parse(symbol)
        today = self._today()
        # An option's symbol has no dot in it (BRKB), so the stock's own spelling comes from the journal.
        underlying = mine[0].underlying if mine else root
        opened = min([(t.opened or t.ordered).astimezone(ET).date() for t in mine] + [today, expiry])
        return positions.make(underlying, kind, strike, expiry, side, quantity, round(price, 4), opened, id=held_id(symbol, side))

    def _close_on(self, underlying: str, day: date) -> float | None:
        """The stock's close on one day, for what an option held to expiry was worth. None until that day's bar is final."""
        if self.scanner is None:
            return None
        try:
            return self.scanner.close_on(underlying, day)  # which waits its turn with scans: the provider is shared
        except Exception:
            return None

    def review(self) -> PaperReview:
        """Bring the journal up to date with the paper account, and set what the account holds against the exit rules."""
        with self._lock:
            if self.broker is not None:
                self._ready()
            now, today = self._clock(), self._today()
            warnings: list[str] = []
            trades, unread = self.journal.load()
            if unread:
                warnings.append(journals.unread_note(unread, self.journal.path))

            def fetch(what: str, call):
                try:
                    return call()
                except BrokerError as err:
                    warnings.append(f"{what} could not be read: {err}")
                    return None

            account = held = orders = None
            if self.broker is None:
                warnings.extend([self.why] if self.why else [])
            else:
                account = fetch("The paper account", self.broker.account)
                # Positions first, then orders, and it matters. Whatever the positions show has then already reached
                # the list of orders: contracts that have gone are never looked for among orders read before they went.
                # The other way round is covered in `reconcile`: a trade it finds open just now is not yet looked for
                # among positions that were read before its order filled.
                held = fetch("The paper account's positions", self.broker.positions)
                in_play = [t.ordered for t in trades if t.state in ("ordered", "open")]
                # From a little before the oldest trade still in play: this computer's clock and Alpaca's are not one clock.
                since = min(in_play + [now - timedelta(days=1)]) - timedelta(hours=1)
                listed = fetch("The paper account's orders", lambda: self.broker.orders(since))
                if listed is not None:
                    orders, whole = listed
                    if not whole:
                        warnings.append(CUT_SHORT)
                    # With part of the list missing, an order that closed a trade may be in the part not read.
                    # So nothing is concluded about contracts that have gone, as when the positions cannot be read.
                    changed = reconcile(
                        trades, orders, held if whole else None, today, now, lookup=self.broker.order, close_on=self._close_on
                    )
                    self.journal.save(changed)

            checked, opened_by = self._held_review(held, trades, warnings)
            return PaperReview(
                generated_at=now,
                mode=self.mode,
                can_order=self.available,
                account=account,
                positions=checked,
                opened_by=opened_by,
                orders=sorted((o for o in orders or [] if o.working), key=lambda o: o.submitted or now, reverse=True),
                trades=sorted(trades, key=lambda t: t.ordered, reverse=True),
                summary=signals.summary(trades),
                warnings=list(dict.fromkeys(warnings)),
            )

    def _held_review(
        self, held: list[PaperHolding] | None, trades: list[Trade], warnings: list[str]
    ) -> tuple[PositionsReview | None, dict[str, list[str]]]:
        if held is None or self.scanner is None:
            return None, {}
        listed, opened_by = [], {}
        for h in held:
            mine = [t for t in trades if t.symbol == h.symbol and t.side == h.side and t.state in ("ordered", "open")]
            try:
                p = self._position(h.symbol, h.side, h.quantity, h.entry_price, mine)
            except ValueError:
                warnings.append(f"The paper account holds {h.symbol}, which does not read as an option position here.")
                continue
            listed.append(p)
            opened_by[p.id] = [t.id for t in mine]
        try:
            checked = self.scanner.review(listed)
        except Exception as err:  # as for the positions page: say so rather than fail
            warnings.append(f"What the paper account holds could not be checked: {err}")
            return None, opened_by
        for h in checked.holdings:
            if h.dte < 0:
                # The positions page says to remove an expired entry. Here the account does that itself.
                text = (
                    f"Expired on {h.position.expiry:%b} {h.position.expiry.day}. Alpaca takes it off the account once it has "
                    "settled, and the journal then records how it ended."
                )
                h.checks = [PositionCheck("expired", "reached", "expired", text)]
        warnings.extend(checked.warnings)
        checked.warnings = []
        return checked, opened_by


# -- Setting one up ---------------------------------------------------------------------------


def _demo_start(desk: PaperDesk) -> None:
    """Made-up trades for the demo: one held for each lens and the rest closed, so every part of the page has something in it."""
    from app.scanner import ScanRequest

    today = desk.scanner.today
    at = lambda day: datetime.combine(day, clock(11, 0), ET)  # noqa: E731
    for lens, request in (
        ("sell", ScanRequest(lens="sell", kind="put", dte_min=20, dte_max=60, per_side=3, top_n=14)),
        ("buy", ScanRequest(lens="buy", dte_min=20, dte_max=60, per_side=2, top_n=13)),
    ):
        result = desk.scanner.scan(request)
        side = "long" if lens == "buy" else "short"
        for n, c in enumerate(result.ideas):
            rng = random.Random(f"{c.symbol}-journal")
            quoted = c.ask if lens == "buy" else c.bid
            held_for = rng.randrange(4, 26)
            opened = today - timedelta(days=rng.randrange(3, 9) if n == 0 else held_for + rng.randrange(2, 30))
            # The two that are held: a put sold for well over what it costs now, and a call bought for less.
            factor = rng.uniform(0.85, 1.25) if n else 2.3 if lens == "sell" else 0.8
            entry = max(round(quoted * factor, 2), 0.05)
            t = journals.new(c.underlying, c.kind, c.strike, c.expiry, side, 1, entry, at(opened), signals.snapshot(result, c), id=f"demo{lens[0]}{n + 1}")
            t.state, t.opened, t.entry_price, t.open_order = "open", at(opened), entry, f"demo-{t.id}"
            if n == 0:
                desk.broker.hold(t.symbol, side, 1, entry)
            else:
                if lens == "sell":
                    won = rng.random() < 0.72
                    left = entry * (rng.uniform(0.25, 0.5) if won else rng.uniform(1.4, 3.0))
                    rules = ("profit_target",) if won else rng.choice([("time_stop",), ("strike",), ()])
                else:
                    won = rng.random() < 0.38
                    left = entry * (rng.uniform(1.3, 2.4) if won else rng.uniform(0.15, 0.8))
                    rules = rng.choice([("time_stop",), ()])
                t.exits.append(Exit(1, max(round(left, 2), 0.01), at(opened + timedelta(days=held_for)), "order", rules, f"demo-{t.id}-c"))
            desk.journal.add(t)


def _demo_clock(day: date) -> Callable[[], datetime]:
    """A clock for the demo. It starts on the demo's day, at the time it is now in New York, and runs on from there.

    The demo's day is its market's day, which a test holds still, so an order is never dated after the demo's
    own contracts have expired. The clock is set once and then only moves forward, by the time that has really
    passed. Read afresh as "the demo's day at the time on the wall", it went back a day at midnight: an order
    written out at 11:55 p.m. was still taken twenty minutes later, and a trade closed after midnight was
    closed before it opened. It also went back an hour when the clocks did.
    """
    started = datetime.now(timezone.utc)
    start = datetime.combine(day, started.astimezone(ET).time(), ET).astimezone(timezone.utc)
    return lambda: start + (datetime.now(timezone.utc) - started)


def build_desk(settings, scanner=None, demo: bool = False, setup_error: str | None = None) -> PaperDesk:
    """The paper desk for these settings. There is always one, so the journal can be read whatever else is missing."""
    if scanner is not None and demo:
        now = _demo_clock(scanner.today)
        return PaperDesk(journals.open_journal(settings, demo=True), scanner, DemoBroker(scanner.market, now), clock=now, seed=_demo_start)
    journal = journals.open_journal(settings)
    if scanner is None:
        return PaperDesk(journal, None, None, setup_error)
    if not settings.alpaca_paper:
        return PaperDesk(journal, scanner, None, LIVE_KEYS)
    return PaperDesk(journal, scanner, AlpacaPaper(settings.alpaca_key, settings.alpaca_secret))
