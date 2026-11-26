"""The pool of stocks a type of event is compared across.

One stock rarely has more than a handful of events of a type, so a study of
what followed them rests on very little. Each stock whose news is read leaves
a small tally here: for every type, how many of its days carried an event, and
how many of those were big next to that stock's own usual day. A pooled study
is those tallies added up over the stocks covered about as much as the one
being looked at (`events.coverage`).

Nothing here fetches anything. A scan adds the stocks it shows, and
`Scanner.widen_pool` adds stocks nobody has scanned.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from app.analysis import events
from app.cache import DiskCache, MemoryCache, NullCache

PREFIX = "pool_"


class EventPool:
    # A tally older than this is left out. A stock nobody has looked at for months
    # is missing its newer events, and may no longer be covered as it was.
    KEEP = timedelta(days=120)

    def __init__(self, cache: DiskCache | MemoryCache | NullCache) -> None:
        self._cache = cache

    def add(self, symbol: str, group: str, tally: dict[str, list[int]]) -> None:
        """Keep one stock's tally, in place of any it left before."""
        self._cache.set(PREFIX + symbol, {"group": group, "kinds": tally})

    def totals(self) -> dict[str, dict[str, Any]]:
        """For each group: how many stocks are on record, and by type the summed counts and the stocks behind them.

        A record that is too old or does not read as a tally is passed over.
        """
        out: dict[str, dict[str, Any]] = {group: {"stocks": 0, "kinds": {}} for group in events.GROUPS}
        for key in self._cache.keys(PREFIX):
            found = self._cache.entry(key)
            if found is None or found[1] > self.KEEP or not isinstance(found[0], dict):
                continue
            group, kinds = found[0].get("group"), found[0].get("kinds")
            if group not in out or not isinstance(kinds, dict) or not all(_counts(row) for row in kinds.values()):
                continue
            out[group]["stocks"] += 1
            for kind, row in kinds.items():
                total = out[group]["kinds"].setdefault(kind, {"counts": [0] * len(events.TALLY), "stocks": 0})
                total["counts"] = [a + b for a, b in zip(total["counts"], row)]
                total["stocks"] += 1 if row[0] else 0
        return out


def _counts(row: Any) -> bool:
    """Whether `row` reads as one type's counts, in `events.TALLY` order, and they agree with each other."""
    if not isinstance(row, list) or len(row) != len(events.TALLY) or not all(type(n) is int and n >= 0 for n in row):
        return False
    days, big, up, down, sessions, big_sessions = row
    # Event days are among the sessions they are compared with, so a row with events has sessions.
    return big <= days and up + down <= days and big_sessions <= sessions and days <= sessions
