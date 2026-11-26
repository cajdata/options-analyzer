"""The positions a person holds, as they entered them.

This is the one thing the app keeps that is the person's own and cannot be
fetched again. So it is handled the opposite way to the cache:

* A file that does not read is an error, never "no positions". Read as empty,
  it would be written over by the next entry.
* An entry that does not read is left in the file as it is and counted, never
  dropped.
* A write that fails is an error, never skipped.

The file is read on every use, so an edit made by hand counts without a
restart. Nothing here fetches or prices anything: `Scanner.review` sets each
position beside today's quotes.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import tempfile
import threading
import uuid
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from app import occ
from app.config import tidy
from app.models import Position

KINDS = ("call", "put")
SIDES = ("long", "short")
MAX_QUANTITY = 10_000
# One to six letters or digits, optionally a class suffix: AAPL, F, BRK.B.
_SYMBOL = re.compile(r"[A-Z0-9]{1,6}(\.[A-Z0-9]{1,3})?")
_FIELDS = ("underlying", "kind", "strike", "expiry", "side", "quantity", "entry_price", "opened")


class StoreError(RuntimeError):
    """The positions file cannot be read or written. The message is written for the user."""


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{what} must be a number above zero")
    return float(value)


def _date(value: Any, what: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"{what} must be a date such as 2026-12-18") from None


def make(
    underlying: str,
    kind: str,
    strike: float,
    expiry: date | str,
    side: str,
    quantity: int,
    entry_price: float,
    opened: date | str,
    id: str | None = None,
) -> Position:
    """A position from what a person typed. Raises ValueError, worded for them, when it is not one."""
    symbols = tidy([underlying]) if isinstance(underlying, str) else []
    if len(symbols) != 1 or not _SYMBOL.fullmatch(symbols[0]):
        raise ValueError(f"not a ticker symbol: {underlying!r}")
    if kind not in KINDS:
        raise ValueError("kind must be call or put")
    if side not in SIDES:
        raise ValueError("side must be long (bought) or short (sold)")
    strike = round(_number(strike, "strike"), 3)  # an option's symbol carries a strike to a tenth of a cent
    if not 0 < strike < 100_000:
        raise ValueError("strike must be at least 0.001 and below 100,000")
    if type(quantity) is not int or not 1 <= quantity <= MAX_QUANTITY:
        raise ValueError(f"quantity must be a whole number of contracts from 1 to {MAX_QUANTITY:,}")
    entry_price = _number(entry_price, "entry price")
    expiry, opened = _date(expiry, "expiry"), _date(opened, "opened")
    if not 2000 <= expiry.year <= 2099:
        raise ValueError("expiry must be a date in this century")
    if opened > expiry:
        raise ValueError("a position cannot be opened after its contract expires")
    if id is not None and (not isinstance(id, str) or not id.strip()):
        raise ValueError("id must be a short piece of text")
    # An option's symbol has no dot in it: BRK.B options are listed under BRKB.
    contract = occ.build(symbols[0].replace(".", ""), expiry, kind, strike)
    return Position(
        id=id.strip() if id else uuid.uuid4().hex[:8],
        underlying=symbols[0],
        kind=kind,
        strike=strike,
        expiry=expiry,
        side=side,
        quantity=quantity,
        entry_price=entry_price,
        opened=opened,
        symbol=contract,
    )


def _plain(p: Position) -> dict:
    return {
        "id": p.id,
        "underlying": p.underlying,
        "kind": p.kind,
        "strike": p.strike,
        "expiry": p.expiry.isoformat(),
        "side": p.side,
        "quantity": p.quantity,
        "entry_price": p.entry_price,
        "opened": p.opened.isoformat(),
    }


def _position(row: Any) -> Position | None:
    """One entry of the file as a position. None when it does not read as one."""
    try:
        return make(*(row[name] for name in _FIELDS), id=row["id"])
    except (KeyError, TypeError, ValueError):
        return None


class PositionStore:
    """Positions in one JSON file, or in memory only when no path is given.

    The demo's positions live in memory: demo data must never reach the disk,
    and the person's own file must never be shown as demo data.
    """

    def __init__(self, path: Path | str | None, seed: Iterable[Position] = ()) -> None:
        self.path = Path(path) if path is not None else None
        self._memory = [_plain(p) for p in seed]
        self._lock = threading.Lock()

    # -- Reading and writing -------------------------------------------------

    def _rows(self) -> list[Any]:
        """Every entry as it is written, whether or not it reads as a position."""
        if self.path is None:
            return list(self._memory)
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        except (OSError, UnicodeError) as err:
            raise StoreError(f"The positions file {self.path} could not be read ({type(err).__name__}).") from err
        if not text.strip():
            return []
        try:
            rows = json.loads(text)["positions"]
            if not isinstance(rows, list):
                raise TypeError
        except (ValueError, KeyError, TypeError):
            raise StoreError(
                f"The positions file {self.path} does not read as a list of positions. It has been left as it is, "
                "and nothing will be written to it until it is repaired or moved away."
            ) from None
        return rows

    def _write(self, rows: list[Any]) -> None:
        if self.path is None:
            self._memory = rows
            return
        payload = json.dumps({"version": 1, "positions": rows}, indent=2) + "\n"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Written beside the file and swapped into place, so a crash never leaves half a file.
            handle, temporary = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as file:
                    file.write(payload)
                os.replace(temporary, self.path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(temporary)
                raise
        except OSError as err:
            raise StoreError(f"The positions file {self.path} could not be written ({type(err).__name__}).") from err

    # -- What callers use ----------------------------------------------------

    def load(self) -> tuple[list[Position], int]:
        """The positions on file, in the order they were entered, and how many entries could not be read."""
        with self._lock:
            found = [_position(row) for row in self._rows()]
        return [p for p in found if p is not None], sum(1 for p in found if p is None)

    def add(self, position: Position) -> Position:
        with self._lock:
            rows = self._rows()
            if any((held := _position(row)) and held.id == position.id for row in rows):
                raise ValueError(f"there is already a position with the id {position.id}")
            self._write(rows + [_plain(position)])
        return position

    def remove(self, id: str) -> bool:
        """Take the position with this id off the file. False when there is none."""
        with self._lock:
            rows = self._rows()
            kept = [row for row in rows if not ((held := _position(row)) and held.id == id)]
            if len(kept) == len(rows):
                return False
            self._write(kept)
        return True


def open_store(settings, market=None) -> PositionStore:
    """The person's own file. For demo data, a store in memory that starts with the demo's made-up positions."""
    if market is not None and market.mode == "demo":
        return PositionStore(None, market.sample_positions())
    return PositionStore(settings.positions_file)


def unread_note(count: int, path: Path | None) -> str:
    """What to say when entries on file do not read as positions."""
    what = "1 entry" if count == 1 else f"{count} entries"
    verb = "does" if count == 1 else "do"
    return f"{what} in {path or 'the positions on file'} {verb} not read as a position, and {'is' if count == 1 else 'are'} left there untouched."
