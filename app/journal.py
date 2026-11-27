"""The journal: every paper trade, with what the scan showed when it was opened and how it ended.

A scan ranks contracts by visible rules. The journal is where those rules meet
what happened next: each entry keeps the signals of the moment the order was
written, and then the fills, the exit rules that were reached when it was
closed, and the result.

Like the positions file, the journal is the person's own and cannot be fetched
again, so the same three rules hold (`app/store.py`): a file that does not read
is an error and never an empty journal, an entry that does not read stays in the
file as it is and is counted, and a write that fails is an error.

Nothing here talks to the broker. `app/paper.py` sets the journal beside the
paper account and writes down what changed.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from app import positions
from app.models import ClosingOrder, Exit, Trade
from app.store import ListFile, StoreError

__all__ = ["ENDED_BY", "HOW", "NO_PRICE", "Journal", "StoreError", "ended", "new", "open_journal", "settle", "unpriced", "unread_note"]

ET = ZoneInfo("America/New_York")
SHARES = 100
STATES =("ordered", "unfilled", "open", "closed")
# How contracts left a trade, as a page says it.
HOW: dict[str, str] = {
    "order": "closed by an order sent here",
    "outside": "closed by an order this app did not send",
    "expiry": "held to expiry",
    "unknown": "left the account with no order on record",
}
# The exit rules a closed trade is sorted by, first match first, and how each is said. A closing order
# whose reading of one of these could not be made has no rules on record at all (`PaperDesk._rules_now`).
ENDED_BY: dict[str, str] = {"profit_target": "at the profit target", "time_stop": "inside the time stop"}
# How the remark on an order that filled with no price begins. `paper.reconcile` writes the remark and takes
# it away again, and it is the only record that a trade is waiting on such an order.
NO_PRICE = "The account says"


def unpriced(trade: Trade) -> bool:
    """Whether a trade is waiting on an order that ended with contracts filled and no price for them.

    Such an order is neither waiting to fill nor working, and whatever is said about it has to say so.
    """
    return bool(trade.note and trade.note.startswith(NO_PRICE))


def new(
    underlying: str,
    kind: str,
    strike: float,
    expiry: date,
    side: str,
    quantity: int,
    limit: float,
    ordered: datetime,
    signals: dict,
    id: str | None = None,
) -> Trade:
    """An entry for an order about to be sent. The contract is checked as a position's is."""
    if ordered.tzinfo is None:
        raise ValueError("the time an order was sent needs a zone")
    # The day in New York, where the contract trades and expires.
    held = positions.make(underlying, kind, strike, expiry, side, quantity, limit, ordered.astimezone(ET).date(), id=id)
    return Trade(
        id=held.id,
        underlying=held.underlying,
        kind=held.kind,
        strike=held.strike,
        expiry=held.expiry,
        symbol=held.symbol,
        side=held.side,
        quantity=held.quantity,
        limit=held.entry_price,
        ordered=ordered,
        signals=signals,
    )


def ended(t: Trade) -> str | None:
    """At which rule a closed trade's last contracts left, by the rules that were reached when the closing order was sent."""
    last = max(t.exits, key=lambda e: e.when, default=None)
    if last is None or t.state != "closed":
        return None
    if last.how == "expiry":
        return "held to expiry"
    if last.rules is None:
        return "rules not on record"
    for rule, said in ENDED_BY.items():
        if rule in last.rules:
            return said
    # Only a seller has a profit target (`holdings.assess`), so a bought option has the one rule to be ahead of.
    return "before either rule was reached" if t.side == "short" else "before the time stop"


def _stands(t: Trade) -> str:
    """Where a trade is, or how it ended, in a few words. Worded here once, for the page and the command line."""
    # An order that ended with contracts filled and no price for them is neither waiting to fill nor working,
    # so the words beside the remark that says so must not say it is.
    if t.state == "ordered":
        return "ordered, waiting for the price it filled at" if unpriced(t) else "ordered, not filled yet"
    if t.state == "unfilled":
        return "not filled"
    if t.state == "open":
        if not t.closing:
            return "open"
        return "open, waiting for the price its closing order filled at" if unpriced(t) else "open, closing order working"
    last = max(t.exits, key=lambda e: e.when, default=None)
    if last is not None and last.how in ("outside", "unknown"):
        return HOW[last.how]
    return ended(t) or "closed"


def settle(trade: Trade) -> Trade:
    """Work out where a trade stands from its exits: closed or not, and the result once every exit has a price.

    A result is in dollars across every contract, and as a share of the premium
    paid or collected, as a held position's is. An exit with no price on record
    leaves the trade closed with no result, and it is counted in no summary.
    """
    left = sum(e.quantity for e in trade.exits)
    if trade.state == "open" and trade.exits and left >= trade.quantity:
        trade.state, trade.closing = "closed", None
    trade.closed = trade.exit_price = trade.pnl = trade.pnl_pct = None
    trade.stands = _stands(trade)
    if trade.state != "closed" or not trade.exits:
        return trade
    trade.closed = max(e.when for e in trade.exits)
    if trade.entry_price and all(e.price is not None for e in trade.exits):
        sign = 1.0 if trade.side == "long" else -1.0
        gain = sum((e.price - trade.entry_price) * sign * e.quantity for e in trade.exits)
        trade.exit_price = round(sum(e.price * e.quantity for e in trade.exits) / left, 4)
        trade.pnl = round(gain * SHARES, 2)  # prices are amounts in cents, and a float is not one
        trade.pnl_pct = round(gain / (trade.entry_price * left), 6)
    return trade


# -- To the file and back ----------------------------------------------------------------


def _plain(trade: Trade) -> dict:
    """A trade as it is written: dates and times as text, the rest as it is."""

    def text(value: Any) -> Any:
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        if isinstance(value, dict):
            return {key: text(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [text(item) for item in value]
        return value

    return text(asdict(trade))


def _written(trade: Trade) -> dict:
    """A trade as it goes to the file, with where it stands worked out afresh.

    Nothing this app writes holds a figure that is not a number: `json.dumps` raises ValueError first. (An
    entry that does not read goes back to the file as it came, whatever is in it: that is `store.ListFile`.)
    """
    plain = _plain(settle(trade))
    json.dumps(plain, allow_nan=False)
    return plain


def _time(value: Any) -> datetime:
    when = datetime.fromisoformat(value)
    if when.tzinfo is None:
        # With no zone it cannot be set beside the broker's times.
        raise ValueError("a time needs a zone")
    return when


def _rules(value: Any) -> tuple[str, ...] | None:
    return None if value is None else tuple(str(rule) for rule in value)


def _price(value: Any, needed: bool = True, least: float = positions.MIN_PRICE) -> float | None:
    """A price per share within a position's bounds. `least` is 0 only for an exit: an option can expire worth nothing.

    A price to measure a result against has the same floor as a position's entry price. One of 1e-310 was
    read as a price, and every result against it was infinite.
    """
    if value is None and not needed:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not least <= value <= positions.MAX_PRICE:
        raise ValueError("not a price")
    return float(value)


def _count(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= positions.MAX_QUANTITY:
        raise ValueError("not a number of contracts")
    return value


def _trade(row: Any) -> Trade | None:
    """One entry of the file as a trade. None when it does not read as one."""
    try:
        if row["state"] not in STATES or not isinstance(row["signals"], dict):
            return None
        # What the scan showed is handed out as it is, so a figure in it that is not a number (NaN, Infinity,
        # which Python reads from a file) would leave with it. Such an entry does not read.
        json.dumps(row["signals"], allow_nan=False)
        made = new(
            row["underlying"], row["kind"], row["strike"], row["expiry"], row["side"], row["quantity"],
            row["limit"], _time(row["ordered"]), row["signals"], id=row["id"],
        )  # fmt: skip
        made.state = row["state"]
        made.open_order = str(row.get("open_order") or "")
        made.opened = _time(row["opened"]) if row.get("opened") else None
        made.entry_price = _price(row.get("entry_price"), needed=False)
        made.note = str(row["note"]) if row.get("note") else None
        if closing := row.get("closing"):
            made.closing = ClosingOrder(
                str(closing["client_id"]), _count(closing["quantity"]), _price(closing["limit"]), _time(closing["sent"]),
                _rules(closing.get("rules")), str(closing.get("order") or ""),
            )  # fmt: skip
        for e in row.get("exits") or []:
            if e["how"] not in HOW:
                return None
            made.exits.append(
                Exit(_count(e["quantity"]), _price(e.get("price"), needed=False, least=0.0), _time(e["when"]), e["how"], _rules(e.get("rules")), str(e.get("order") or ""))
            )
        if made.state in ("open", "closed") and not made.entry_price:
            return None  # a trade that was filled has the price it was filled at
        # The result is worked out again from the exits, whatever the file says it was.
        return settle(made)
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


class Journal:
    """Paper trades in one JSON file, or in memory only when no path is given, as the demo's are."""

    def __init__(self, path: Path | str | None, seed: Iterable[Trade] = ()) -> None:
        self._file = ListFile(path, "trades", "journal file", (_plain(t) for t in seed))
        self.path = self._file.path
        self._lock = threading.Lock()

    @staticmethod
    def _read(rows: list[Any]) -> list[Trade | None]:
        """Each entry as a trade. None where it does not read as one, or repeats an earlier entry's id."""
        seen: set[str] = set()
        found: list[Trade | None] = []
        for row in rows:
            trade = _trade(row)
            if trade is not None and trade.id in seen:
                trade = None
            if trade is not None:
                seen.add(trade.id)
            found.append(trade)
        return found

    def load(self) -> tuple[list[Trade], int]:
        """The trades on file, oldest first, and how many entries could not be used."""
        with self._lock:
            found = self._read(self._file.rows())
        return [t for t in found if t is not None], sum(1 for t in found if t is None)

    def add(self, trade: Trade) -> Trade:
        with self._lock:
            rows = self._file.rows()
            if any((held := _trade(row)) and held.id == trade.id for row in rows):
                raise ValueError(f"there is already a trade with the id {trade.id}")
            self._file.write(rows + [_written(trade)])
        return trade

    def save(self, trades: Iterable[Trade]) -> None:
        """Write these trades over their entries. Anything else in an entry, such as a remark added by hand, is kept."""
        changed = {t.id: _written(t) for t in trades}
        if not changed:
            return
        with self._lock:
            rows = self._file.rows()
            for n, held in enumerate(self._read(rows)):
                if held is not None and held.id in changed:
                    rows[n] = {**rows[n], **changed.pop(held.id)}
            if changed:
                raise ValueError(f"the journal has no trade with the id {next(iter(changed))}")
            self._file.write(rows)

    def remove(self, id: str) -> bool:
        """Take an entry out. Only for an order the broker turned down outright: nothing happened, so there is nothing to keep."""
        with self._lock:
            rows = self._file.rows()
            kept = [row for row, held in zip(rows, self._read(rows)) if not (held and held.id == id)]
            if len(kept) == len(rows):
                return False
            self._file.write(kept)
        return True


def open_journal(settings, demo: bool = False) -> Journal:
    """The person's own journal file. For the demo, a journal in memory."""
    return Journal(None if demo else settings.journal_file)


def unread_note(count: int, path: Path | None) -> str:
    """What to say when entries in the journal cannot be used as trades."""
    where = path or "the journal"
    if count == 1:
        return f"1 entry in {where} does not read as a trade, or repeats another entry's id. It is left there untouched."
    return f"{count} entries in {where} do not read as trades, or repeat another entry's id. They are left there untouched."
