"""Earnings dates: SEC EDGAR for history, Alpha Vantage for the calendar and history.

SEC EDGAR also lists a company's 8-K filings for the news layer (providers/news.py),
through the same paced client.
"""

from __future__ import annotations

import csv
import io
import threading
import time
from datetime import date, datetime, time as clock, timedelta
from typing import Any, Callable, Protocol
from zoneinfo import ZoneInfo

import httpx

from app.analysis.earnings import estimate_next
from app.cache import DiskCache
from app.models import EarningsEvent, NewsItem

ET = ZoneInfo("America/New_York")
MARKET_CLOSE = clock(16, 0)


class RateLimited(RuntimeError):
    pass


class Unsupported(RuntimeError):
    """The source cannot answer for this company, which is different from "no reports"."""


class HistorySource(Protocol):
    name: str

    def history(self, symbol: str) -> list[EarningsEvent]: ...


class CalendarSource(Protocol):
    name: str

    def calendar(self, today: date) -> dict[str, EarningsEvent]: ...


def _float(text: Any) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


class AlphaVantageEarnings:
    """Free tier allows roughly 25 requests a day, so every call here is cached upstream."""

    name = "Alpha Vantage"
    rate_limited = True  # too few requests a day to spend any on a stock nobody is scanning
    URL = "https://www.alphavantage.co/query"
    COOL_DOWN = 15 * 60.0  # seconds to stay quiet after the service says a limit was hit
    _TIMING = {"pre-market": "pre", "post-market": "post"}

    def __init__(
        self,
        key: str,
        client: httpx.Client | None = None,
        min_interval: float = 1.2,
        cool_down: float = COOL_DOWN,
    ) -> None:
        self._key = key
        self._client = client or httpx.Client(timeout=30.0)
        self._lock = threading.Lock()
        self._min_interval = min_interval
        self._cool_down = cool_down
        self._last_call = 0.0
        self._blocked_until = 0.0

    def _fetch(self, **params: str) -> httpx.Response:
        # The block is checked, and set, while holding the lock. Otherwise threads
        # already queued behind a request that hits the limit would each still fire.
        with self._lock:
            if time.monotonic() < self._blocked_until:
                raise RateLimited("Alpha Vantage request limit reached")
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            resp = self._client.get(self.URL, params={**params, "apikey": self._key})
            self._last_call = time.monotonic()
            resp.raise_for_status()
            message = self._limit_message(resp)
            if message:
                # Back off for a while, then try again. The burst limit clears within
                # a minute and the daily one overnight; a flag that never cleared
                # would outlive both in a server left running.
                self._blocked_until = time.monotonic() + self._cool_down
                raise RateLimited(message)
        return resp

    @staticmethod
    def _limit_message(resp: httpx.Response) -> str | None:
        """Limits and key problems come back as HTTP 200 with a message instead of data."""
        if not resp.text.lstrip().startswith("{"):
            return None
        try:
            payload = resp.json()
        except ValueError:
            return None
        message = payload.get("Information") or payload.get("Note") if isinstance(payload, dict) else None
        return str(message) if message else None

    def calendar(self, today: date) -> dict[str, EarningsEvent]:
        """The next report on or after `today` for every listed symbol."""
        resp = self._fetch(function="EARNINGS_CALENDAR", horizon="3month")
        text = resp.text.lstrip()
        if text.startswith("{"):  # JSON where the CSV should be: an error with no calendar in it
            return {}
        out: dict[str, EarningsEvent] = {}
        for row in csv.DictReader(io.StringIO(text)):
            try:
                day = date.fromisoformat(row["reportDate"])
            except (KeyError, TypeError, ValueError):  # a short row leaves the field as None
                continue
            symbol = (row.get("symbol") or "").upper()
            if day < today or not symbol:
                continue
            if symbol not in out or day < out[symbol].day:
                timing = self._TIMING.get(row.get("timeOfTheDay") or "", "unknown")
                out[symbol] = EarningsEvent(day=day, timing=timing, confirmed=True, source=self.name)
        return out

    def history(self, symbol: str) -> list[EarningsEvent]:
        payload = self._fetch(function="EARNINGS", symbol=symbol).json()
        events = []
        for row in payload.get("quarterlyEarnings") or []:
            try:
                day = date.fromisoformat(row["reportedDate"])
            except (KeyError, TypeError, ValueError):
                continue
            events.append(
                EarningsEvent(
                    day=day,
                    timing=self._TIMING.get(row.get("reportTime") or "", "unknown"),
                    surprise_pct=_float(row.get("surprisePercentage")),
                    source=self.name,
                )
            )
        return sorted(events, key=lambda e: e.day, reverse=True)


class EdgarEarnings:
    """Past earnings dates from 8-K filings that report results (Item 2.02).

    The filing's acceptance time says whether results came out before the open
    or after the close. A company that files its 8-K hours after the press
    release can be off by one session; that is rare.
    """

    name = "SEC EDGAR"
    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/"
    ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/"
    MAX_OLDER_FILES = 4
    MIN_INTERVAL = 0.12  # the SEC allows ten requests a second across all threads
    RETRIES = 2
    # Forms a foreign company files in place of the 10-K, 10-Q and 8-K.
    FOREIGN_FORMS = frozenset({"6-K", "20-F", "40-F"})

    def __init__(
        self,
        user_agent: str,
        cache: DiskCache,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._headers = {"User-Agent": user_agent}
        self._cache = cache
        self._client = client or httpx.Client(timeout=30.0)
        self._sleep = sleep
        self._lock = threading.Lock()
        self._pace = threading.Lock()
        self._last_call = 0.0
        self._companies: dict[str, list] | None = None  # ticker: [CIK number, company name]

    def _get(self, url: str) -> Any:
        for attempt in range(self.RETRIES + 1):
            with self._pace:  # space requests out, whichever thread makes them
                wait = self.MIN_INTERVAL - (time.monotonic() - self._last_call)
                if wait > 0:
                    self._sleep(wait)
                self._last_call = time.monotonic()
            resp = self._client.get(url, headers=self._headers)
            busy = resp.status_code == 429 or resp.status_code >= 500
            if busy and attempt < self.RETRIES:
                self._sleep(2.0 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        raise AssertionError("unreachable")

    def _company(self, symbol: str) -> list | None:
        """The SEC's [CIK number, company name] for a ticker, from its list of every listed company."""
        with self._lock:
            if self._companies is None:
                table = self._cache.get("edgar_companies", timedelta(days=7))
                if table is None:
                    rows = self._get(self.TICKERS_URL).values()
                    table = {str(r["ticker"]).upper(): [int(r["cik_str"]), str(r.get("title") or "")] for r in rows}
                    self._cache.set("edgar_companies", table)
                self._companies = table
        return self._companies.get(symbol.upper().replace(".", "-"))

    def _cik(self, symbol: str) -> int | None:
        found = self._company(symbol)
        return found[0] if found else None

    def company_name(self, symbol: str) -> str | None:
        """The company's name as the SEC lists it, e.g. "INSMED Inc". None for a ticker it does not list."""
        found = self._company(symbol)
        return found[1] or None if found else None

    def history(self, symbol: str, want: int = 12) -> list[EarningsEvent]:
        cik = self._cik(symbol)
        if cik is None:
            return []
        data = self._get(f"{self.SUBMISSIONS_URL}CIK{cik:010d}.json")
        filings = data.get("filings") or {}
        recent = filings.get("recent") or {}
        events = self._events(recent)
        if not events and self.FOREIGN_FORMS.intersection(recent.get("form") or []):
            # A foreign company announces results on Form 6-K, which carries no item
            # codes, so its reports cannot be picked out here. That is not the same
            # as an ETF, which has no reports at all, and must not look like it.
            raise Unsupported(f"{self.name} cannot date earnings for a company that reports on Form 6-K.")
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
            if form != "8-K" or "2.02" not in _codes(item):
                continue
            when = _accepted(stamp)
            if when is None:
                continue
            # A release during market hours moves the stock the same day, like a pre-market one.
            timing = "post" if when.time() >= MARKET_CLOSE else "pre"
            out.append(EarningsEvent(day=when.date(), timing=timing, source=self.name))
        return out

    def filings(self, symbol: str) -> list[NewsItem]:
        """Every 8-K among the company's recent filings, newest first, with its item codes.

        Only the newest part of the filing index is read. That is the last
        thousand filings or so: about ten years for most companies, and about
        one for a company that files constantly (a large bank).
        """
        cik = self._cik(symbol)
        if cik is None:
            return []
        data = self._get(f"{self.SUBMISSIONS_URL}CIK{cik:010d}.json")
        recent = (data.get("filings") or {}).get("recent") or {}
        forms = recent.get("form") or []

        def cell(column: str, row: int) -> Any:
            values = recent.get(column) or []
            return values[row] if row < len(values) else None

        out = []
        for row, form in enumerate(forms):
            when = _accepted(cell("acceptanceDateTime", row)) if form in ("8-K", "8-K/A") else None
            if when is None:
                continue
            accession = str(cell("accessionNumber", row) or "")
            folder = f"{self.ARCHIVE_URL}{cik}/{accession.replace('-', '')}/" if accession else None
            out.append(
                NewsItem(
                    when=when,
                    kind="filing",
                    title=form,
                    source=self.name,
                    url=folder + str(cell("primaryDocument", row) or "") if folder else None,
                    ref=accession,
                    items=_codes(cell("items", row)),
                )
            )
        if not out and self.FOREIGN_FORMS.intersection(forms):
            # As with earnings: a foreign company has filings, but none that say what they are about.
            raise Unsupported(f"{self.name} filings are not classified for a company that reports on Form 6-K.")
        return sorted(out, key=lambda f: f.when, reverse=True)


def _codes(item: Any) -> tuple[str, ...]:
    """An 8-K's item codes. They arrive as "2.02,9.01"; tolerate a list, padding or a missing value."""
    parts = item if isinstance(item, list) else str(item or "").split(",")
    return tuple(code for code in (str(part).strip() for part in parts) if code)


def _accepted(stamp: Any) -> datetime | None:
    """When the SEC accepted a filing, in New York time."""
    try:
        return datetime.fromisoformat(str(stamp).replace("Z", "+00:00")).astimezone(ET)
    except ValueError:
        return None


class EarningsService:
    """Combines the configured sources, caches results and estimates missing dates.

    One instance lives as long as the server does, so nothing here may assume the
    date, the calendar or a failure stays the same from one scan to the next.
    """

    HISTORY_TTL = timedelta(days=30)
    # Every source answered and none has a report on file: an ETF, most likely.
    EMPTY_TTL = timedelta(days=30)
    CALENDAR_TTL = timedelta(hours=12)  # how long a downloaded calendar is trusted
    CALENDAR_RECHECK = timedelta(minutes=30)  # how long the copy in memory is reused
    CALENDAR_RETRY = timedelta(minutes=15)  # how soon to try again after a failed download
    ENOUGH = 8  # reports a source must return before the next source is not consulted

    def __init__(
        self,
        cache: DiskCache,
        *,
        history_sources: list[HistorySource] | None = None,
        calendar_source: CalendarSource | None = None,
        today: date | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cache = cache
        self._history_sources = history_sources or []
        self._calendar_source = calendar_source
        self._fixed_today = today
        self._clock = clock
        self._calendar: dict[str, EarningsEvent] | None = None
        self._calendar_at = 0.0
        self._calendar_ok = False
        self._lock = threading.Lock()
        self.calendar_note: str | None = None
        # A different set of sources can answer differently, so it gets its own entries.
        self._tag = "-".join(s.name.split()[0].lower() for s in self._history_sources)

    @property
    def today(self) -> date:
        """The trading date now, unless one was fixed for a test or the demo."""
        return self._fixed_today or datetime.now(ET).date()

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
        key = f"earnings_history_{symbol}_{self._tag}"
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
                note = f"{source.name} request limit reached; try again later."
                continue
            except Unsupported as err:
                note = str(err)
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

    def reports(self, symbol: str) -> list[EarningsEvent]:
        """Past reports for a stock that is not being scanned, newest first. Empty when none can be had.

        What is on file answers. Otherwise the first source alone is asked,
        and its answer is kept nowhere: a stock added to a pool must not spend
        a request at a rate-limited source, and an answer from one source must
        not stand in later for the fuller one a scan would get.
        """
        found = self._cache.entry(f"earnings_history_{symbol}_{self._tag}")
        if found is not None and found[0]:
            return [_event(e) for e in found[0]]
        if not self._history_sources or getattr(self._history_sources[0], "rate_limited", False):
            return []
        try:
            return self._history_sources[0].history(symbol)
        except (RuntimeError, httpx.HTTPError, ValueError, KeyError):
            return []

    def _stale(self, events: list[EarningsEvent], age: timedelta) -> bool:
        if not events:
            return age > self.EMPTY_TTL
        if age > self.HISTORY_TTL:
            return True
        # A new report has probably come out since the newest one on file.
        return events[0].day + timedelta(days=75) < self.today and age > timedelta(days=1)

    def calendar(self) -> dict[str, EarningsEvent]:
        with self._lock:
            if self._calendar is not None:
                keep = self.CALENDAR_RECHECK if self._calendar_ok else self.CALENDAR_RETRY
                if self._clock() - self._calendar_at < keep.total_seconds():
                    return self._calendar
            self._calendar, self._calendar_ok = self._load_calendar()
            self._calendar_at = self._clock()
            return self._calendar

    def _load_calendar(self) -> tuple[dict[str, EarningsEvent], bool]:
        """The calendar and whether it is current. Sets or clears `calendar_note`."""
        if self._calendar_source is None:
            return {}, True
        found = self._cache.entry("earnings_calendar")
        if found is not None and found[1] <= self.CALENDAR_TTL:
            self.calendar_note = None
            return {s: _event(e) for s, e in found[0].items()}, True
        try:
            fresh = self._calendar_source.calendar(self.today)
        except (RateLimited, httpx.HTTPError, ValueError) as err:
            self.calendar_note = (
                f"{self._calendar_source.name} earnings calendar unavailable "
                f"({type(err).__name__}); upcoming dates are estimates."
            )
            stale = {s: _event(e) for s, e in found[0].items()} if found is not None else {}
            return stale, False
        self._cache.set("earnings_calendar", {s: _plain(e) for s, e in fresh.items()})
        self.calendar_note = None
        return fresh, True

    def upcoming(self, symbol: str, history: list[EarningsEvent]) -> EarningsEvent | None:
        today = self.today
        listed = self.calendar().get(symbol)
        if listed and listed.day >= today:
            return listed
        return estimate_next(history, today)


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
