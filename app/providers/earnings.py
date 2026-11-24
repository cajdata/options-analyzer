"""Earnings dates: SEC EDGAR for history, Alpha Vantage for the calendar and history."""

from __future__ import annotations

import csv
import io
import threading
import time
from datetime import date, datetime, time as clock, timedelta
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import httpx

from app.analysis.earnings import estimate_next
from app.cache import DiskCache
from app.models import EarningsEvent

ET = ZoneInfo("America/New_York")
MARKET_OPEN, MARKET_CLOSE = clock(9, 30), clock(16, 0)


class RateLimited(RuntimeError):
    pass


class HistorySource(Protocol):
    name: str

    def history(self, symbol: str) -> list[EarningsEvent]: ...


class CalendarSource(Protocol):
    name: str

    def calendar(self) -> dict[str, EarningsEvent]: ...


def _float(text: Any) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


class AlphaVantageEarnings:
    """Free tier allows roughly 25 requests a day, so every call here is cached upstream."""

    name = "Alpha Vantage"
    URL = "https://www.alphavantage.co/query"
    _TIMING = {"pre-market": "pre", "post-market": "post"}

    def __init__(self, key: str, client: httpx.Client | None = None, min_interval: float = 1.2) -> None:
        self._key = key
        self._client = client or httpx.Client(timeout=30.0)
        self._lock = threading.Lock()
        self._min_interval = min_interval
        self._last_call = 0.0
        self._exhausted = False

    def _fetch(self, **params: str) -> httpx.Response:
        if self._exhausted:
            raise RateLimited("Alpha Vantage request limit reached")
        with self._lock:
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            resp = self._client.get(self.URL, params={**params, "apikey": self._key})
            self._last_call = time.monotonic()
        resp.raise_for_status()
        return resp

    def _check(self, payload: dict) -> None:
        # Limits and key problems come back as HTTP 200 with a message instead of data.
        message = payload.get("Information") or payload.get("Note")
        if message:
            self._exhausted = True
            raise RateLimited(str(message))

    def calendar(self) -> dict[str, EarningsEvent]:
        resp = self._fetch(function="EARNINGS_CALENDAR", horizon="3month")
        text = resp.text.lstrip()
        if text.startswith("{"):
            self._check(resp.json())
            return {}
        out: dict[str, EarningsEvent] = {}
        for row in csv.DictReader(io.StringIO(text)):
            try:
                day = date.fromisoformat(row["reportDate"])
            except (KeyError, ValueError):
                continue
            symbol = (row.get("symbol") or "").upper()
            if symbol and (symbol not in out or day < out[symbol].day):
                timing = self._TIMING.get(row.get("timeOfTheDay") or "", "unknown")
                out[symbol] = EarningsEvent(day=day, timing=timing, confirmed=True, source=self.name)
        return out

    def history(self, symbol: str) -> list[EarningsEvent]:
        payload = self._fetch(function="EARNINGS", symbol=symbol).json()
        self._check(payload)
        events = []
        for row in payload.get("quarterlyEarnings") or []:
            try:
                day = date.fromisoformat(row["reportedDate"])
            except (KeyError, ValueError):
                continue
            events.append(
                EarningsEvent(
                    day=day,
                    timing=self._TIMING.get(row.get("reportTime") or "", "unknown"),
                    surprise_pct=_float(row.get("surprisePercentage")),
                    source=self.name,
                )
            )
        return events


class EdgarEarnings:
    """Past earnings dates from 8-K filings that report results (Item 2.02).

    The filing's acceptance time says whether results came out before the open
    or after the close. A company that files its 8-K hours after the press
    release can be off by one session; that is rare.
    """

    name = "SEC EDGAR"
    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/"
    MAX_OLDER_FILES = 4

    def __init__(self, user_agent: str, cache: DiskCache, client: httpx.Client | None = None) -> None:
        self._headers = {"User-Agent": user_agent}
        self._cache = cache
        self._client = client or httpx.Client(timeout=30.0)
        self._lock = threading.Lock()
        self._ciks: dict[str, int] | None = None

    def _get(self, url: str) -> Any:
        resp = self._client.get(url, headers=self._headers)
        resp.raise_for_status()
        return resp.json()

    def _cik(self, symbol: str) -> int | None:
        with self._lock:
            if self._ciks is None:
                table = self._cache.get("edgar_tickers", timedelta(days=7))
                if table is None:
                    rows = self._get(self.TICKERS_URL).values()
                    table = {str(r["ticker"]).upper(): int(r["cik_str"]) for r in rows}
                    self._cache.set("edgar_tickers", table)
                self._ciks = table
        return self._ciks.get(symbol.upper().replace(".", "-"))

    def history(self, symbol: str, want: int = 12) -> list[EarningsEvent]:
        cik = self._cik(symbol)
        if cik is None:
            return []
        data = self._get(f"{self.SUBMISSIONS_URL}CIK{cik:010d}.json")
        filings = data.get("filings") or {}
        events = self._events(filings.get("recent") or {})
        # Companies that file constantly (large banks) push earnings 8-Ks into older files.
        for older in (filings.get("files") or [])[: self.MAX_OLDER_FILES]:
            if len(events) >= want:
                break
            events += self._events(self._get(self.SUBMISSIONS_URL + older["name"]))
        unique = {e.day: e for e in events}
        return sorted(unique.values(), key=lambda e: e.day, reverse=True)

    def _events(self, table: dict) -> list[EarningsEvent]:
        forms, items, accepted = table.get("form") or [], table.get("items") or [], table.get("acceptanceDateTime") or []
        out = []
        for form, item, stamp in zip(forms, items, accepted):
            if form != "8-K" or "2.02" not in item.split(","):
                continue
            try:
                when = datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone(ET)
            except ValueError:
                continue
            # A release during market hours moves the stock the same day, like a pre-market one.
            timing = "post" if when.time() >= MARKET_CLOSE else "pre"
            out.append(EarningsEvent(day=when.date(), timing=timing, source=self.name))
        return out


class EarningsService:
    """Combines the configured sources, caches results and estimates missing dates."""

    HISTORY_TTL = timedelta(days=30)
    EMPTY_TTL = timedelta(days=1)
    CALENDAR_TTL = timedelta(hours=12)
    ENOUGH = 8  # reports a source must return before the next source is not consulted

    def __init__(
        self,
        cache: DiskCache,
        *,
        history_sources: list[HistorySource] | None = None,
        calendar_source: CalendarSource | None = None,
        today: date | None = None,
    ) -> None:
        self._cache = cache
        self._history_sources = history_sources or []
        self._calendar_source = calendar_source
        self._today = today or datetime.now(ET).date()
        self._calendar: dict[str, EarningsEvent] | None = None
        self._lock = threading.Lock()
        self.calendar_note: str | None = None

    @property
    def sources(self) -> list[str]:
        names = [s.name for s in self._history_sources]
        if self._calendar_source and self._calendar_source.name not in names:
            names.append(self._calendar_source.name)
        return names

    def history(self, symbol: str) -> tuple[list[EarningsEvent], str | None]:
        """Past reports, newest first, plus a note when something went wrong."""
        if not self._history_sources:
            return [], "No earnings data source is configured."
        key = f"earnings_history_{symbol}"
        found = self._cache.entry(key)
        if found is not None:
            events = [_event(e) for e in found[0]]
            if not self._stale(events, found[1]):
                return events, None

        events, note = [], None
        for source in self._history_sources:
            try:
                got = source.history(symbol)
            except RateLimited:
                note = f"{source.name} request limit reached; try again tomorrow."
                continue
            except (httpx.HTTPError, ValueError, KeyError) as err:
                note = f"{source.name} lookup failed: {type(err).__name__}."
                continue
            if len(got) > len(events):
                events = got
            if len(events) >= self.ENOUGH:
                break
        if events or note is None:
            self._cache.set(key, [_plain(e) for e in events])
        elif found is not None:
            events = [_event(e) for e in found[0]]  # stale beats nothing
        return events, note if not events else None

    def _stale(self, events: list[EarningsEvent], age: timedelta) -> bool:
        if not events:
            return age > self.EMPTY_TTL
        if age > self.HISTORY_TTL:
            return True
        # A new report has probably come out since the newest one on file.
        return events[0].day + timedelta(days=75) < self._today and age > timedelta(days=1)

    def calendar(self) -> dict[str, EarningsEvent]:
        with self._lock:
            if self._calendar is not None:
                return self._calendar
            self._calendar = {}
            if self._calendar_source is None:
                return self._calendar
            found = self._cache.entry("earnings_calendar")
            if found is not None and found[1] <= self.CALENDAR_TTL:
                self._calendar = {s: _event(e) for s, e in found[0].items()}
                return self._calendar
            try:
                self._calendar = self._calendar_source.calendar()
                self._cache.set("earnings_calendar", {s: _plain(e) for s, e in self._calendar.items()})
            except (RateLimited, httpx.HTTPError, ValueError) as err:
                self.calendar_note = (
                    f"{self._calendar_source.name} earnings calendar unavailable "
                    f"({type(err).__name__}); upcoming dates are estimates."
                )
                if found is not None:
                    self._calendar = {s: _event(e) for s, e in found[0].items()}
            return self._calendar

    def upcoming(self, symbol: str, history: list[EarningsEvent]) -> EarningsEvent | None:
        listed = self.calendar().get(symbol)
        if listed and listed.day >= self._today:
            return listed
        return estimate_next(history, self._today)


def _plain(e: EarningsEvent) -> dict:
    return {
        "day": e.day.isoformat(),
        "timing": e.timing,
        "surprise_pct": e.surprise_pct,
        "confirmed": e.confirmed,
        "source": e.source,
    }


def _event(d: dict) -> EarningsEvent:
    return EarningsEvent(
        day=date.fromisoformat(d["day"]),
        timing=d.get("timing", "unknown"),
        surprise_pct=d.get("surprise_pct"),
        confirmed=d.get("confirmed", True),
        source=d.get("source", ""),
    )
