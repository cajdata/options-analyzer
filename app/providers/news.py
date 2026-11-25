"""News for one stock: headlines and SEC filings, kept on disk so the record builds up.

A source hands over raw items. What kind of event each one is gets decided
later, by the visible rules in analysis/events.py.
"""

from __future__ import annotations

from datetime import date, datetime, time as clock, timedelta, timezone
from typing import Any, Callable, Protocol

import httpx

from app.cache import DiskCache, NullCache
from app.models import NewsItem
from app.providers.earnings import Unsupported


class HeadlineSource(Protocol):
    name: str

    def headlines(self, symbol: str, since: datetime) -> list[NewsItem]: ...


class FilingSource(Protocol):
    name: str

    def filings(self, symbol: str) -> list[NewsItem]: ...


# What a source is expected to raise when it cannot answer. AlpacaError and
# Unsupported are both RuntimeErrors.
_FAILURES = (RuntimeError, httpx.HTTPError, ValueError, KeyError)


def _why(err: Exception) -> str:
    text = str(err).rstrip(".")
    return text if isinstance(err, RuntimeError) and text else type(err).__name__


class NewsService:
    """Combines the configured sources and caches what they return.

    Headlines are merged into the copy on disk, so a stock's record keeps
    growing even though the feed itself reaches back only some months. Only
    the headlines since the newest one on file are asked for.
    """

    REFRESH = timedelta(minutes=15)  # how long a stock's record is reused before asking again
    LOOKBACK = timedelta(days=400)  # how far back the first request for a stock's headlines reaches
    KEEP = timedelta(days=1200)  # a study cannot use an item older than the price history

    def __init__(
        self,
        cache: DiskCache | NullCache,
        *,
        headline_source: HeadlineSource | None = None,
        filing_source: FilingSource | None = None,
        names: Callable[[str], str | None] | None = None,
    ) -> None:
        self._cache = cache
        self._headline_source = headline_source
        self._filing_source = filing_source
        self._names = names  # looks up a company's name by ticker, where one is available

    def name(self, symbol: str) -> str | None:
        """The company's name, or None when there is nowhere to look it up or the lookup fails."""
        if self._names is None:
            return None
        try:
            return self._names(symbol)
        except _FAILURES:
            return None

    @property
    def sources(self) -> list[str]:
        out = []
        if self._headline_source:
            out.append(f"{self._headline_source.name} headlines")
        if self._filing_source:
            out.append(f"{self._filing_source.name} 8-K filings")
        return out

    def items(self, symbol: str, today: date) -> tuple[list[NewsItem], str | None, bool]:
        """Everything on record for `symbol`, newest first, a note, and whether a lookup failed.

        The note is set when a source failed or cannot answer for this company.
        Only the first is a failure: a company whose filings carry no item
        codes gets a note every time, and nothing went wrong.
        """
        headlines, first, first_failed = self._headlines(symbol, today)
        filings, second, second_failed = self._filings(symbol)
        note = " ".join(n for n in (first, second) if n) or None
        items = sorted(headlines + filings, key=lambda i: i.when, reverse=True)
        return items, note, first_failed or second_failed

    def _headlines(self, symbol: str, today: date) -> tuple[list[NewsItem], str | None, bool]:
        source = self._headline_source
        if source is None:
            return [], None, False
        key = f"news_headlines_{symbol}"
        found = self._cache.entry(key)
        on_disk = _items(found[0]) if found is not None else None
        if on_disk is not None and found[1] <= self.REFRESH:
            return on_disk, None, False
        kept = on_disk or []
        oldest = datetime.combine(today - self.LOOKBACK, clock.min, tzinfo=timezone.utc)
        since = max((i.when for i in kept), default=oldest)
        try:
            fresh = source.headlines(symbol, since)
        except _FAILURES as err:
            return kept, f"{source.name} headlines were unavailable ({_why(err)}).", True  # stale beats nothing
        floor = datetime.combine(today - self.KEEP, clock.min, tzinfo=timezone.utc)
        merged = {i.ref or f"{i.when.isoformat()} {i.title}": i for i in kept + fresh}
        items = sorted((i for i in merged.values() if i.when >= floor), key=lambda i: i.when, reverse=True)
        self._cache.set(key, [_plain(i) for i in items])
        return items, None, False

    def _filings(self, symbol: str) -> tuple[list[NewsItem], str | None, bool]:
        source = self._filing_source
        if source is None:
            return [], None, False
        key = f"news_filings_{symbol}"
        found = self._cache.entry(key)
        on_disk = _items(found[0].get("items")) if found is not None and isinstance(found[0], dict) else None
        if on_disk is not None and found[1] <= self.REFRESH:
            return on_disk, found[0].get("note"), False
        try:
            items, note = source.filings(symbol), None
        except Unsupported as err:
            items, note = [], str(err)
        except _FAILURES as err:
            return on_disk or [], f"{source.name} filings were unavailable ({_why(err)}).", True
        self._cache.set(key, {"items": [_plain(i) for i in items], "note": note})
        return items, note, False


def _plain(item: NewsItem) -> dict:
    return {
        "when": item.when.isoformat(),
        "kind": item.kind,
        "title": item.title,
        "source": item.source,
        "url": item.url,
        "ref": item.ref,
        "items": list(item.items),
    }


def _items(rows: Any) -> list[NewsItem] | None:
    """Items as stored on disk. None for a copy that does not parse, which is then fetched again."""
    if not isinstance(rows, list):
        return None
    try:
        return [
            NewsItem(
                when=datetime.fromisoformat(row["when"]),
                kind=row["kind"],
                title=row["title"],
                source=row.get("source", ""),
                url=row.get("url"),
                ref=row.get("ref", ""),
                items=tuple(row.get("items") or ()),
            )
            for row in rows
        ]
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
