"""Alpaca market data over REST. Read-only: no order endpoints are used.

Endpoints and field names follow https://docs.alpaca.markets/reference.
"""

from __future__ import annotations

import math
import time
from datetime import date, datetime, time as clock, timedelta, timezone
from typing import Any, Callable, Iterable
from zoneinfo import ZoneInfo

import httpx

from app import occ
from app.models import Bar, NewsItem, OptionQuote, StockQuote

DATA_URL = "https://data.alpaca.markets"
PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
ET = ZoneInfo("America/New_York")

MAX_CHAIN_PAGES = 15  # 1,000 contracts a page
NEWS_PAGE = 50  # the most headlines the news endpoint returns at once
MAX_NEWS_PAGES = 40
SUMMARY_CHARS = 400  # a headline's summary is one sentence; the longest seen was 281 characters
_RETRIES = 3
# Today's daily bar is final a little after the 4 p.m. close.
BAR_FINAL = clock(16, 20)
# A quote stamped before the open belongs to no trading session yet.
SESSION_OPEN = clock(9, 30)
# A chain whose newest quote is older than this has not been quoted in any session
# since. Four days leaves room for a three-day weekend, seen from before the open.
MAX_QUOTE_AGE_DAYS = 4
IEX_BARS_NOTE = (
    "Price history comes from the IEX exchange only, because this Alpaca "
    "plan does not include consolidated (SIP) history."
)


class AlpacaError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"Alpaca returned {status}: {message}")
        self.status = status


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _num(obj: Any, key: str) -> float | None:
    value = obj.get(key) if isinstance(obj, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


class AlpacaData:
    mode = "live"
    name = "Alpaca"

    def __init__(
        self,
        key: str,
        secret: str,
        *,
        options_feed: str = "indicative",
        stock_feed: str = "iex",
        paper: bool = True,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(ET),
    ) -> None:
        self._now = now  # New York time; replaceable so tests do not depend on the clock
        self.options_feed = options_feed
        self.stock_feed = stock_feed
        self.notes: list[str] = []
        self._bars_feed = "sip"
        self._trading_url = PAPER_URL if paper else LIVE_URL
        self._client = client or httpx.Client(timeout=20.0)
        self._headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self._sleep = sleep

    # -- HTTP ---------------------------------------------------------------

    def _get(self, url: str, params: dict[str, Any]) -> Any:
        for attempt in range(_RETRIES + 1):
            try:
                resp = self._client.get(url, params=params, headers=self._headers)
            except httpx.TransportError as err:  # timeouts and dropped connections
                if attempt < _RETRIES:
                    self._sleep(2.0 * (attempt + 1))
                    continue
                raise AlpacaError(0, f"could not reach Alpaca ({type(err).__name__})") from err
            busy = resp.status_code == 429 or resp.status_code >= 500
            if busy and attempt < _RETRIES:
                try:
                    wait = float(resp.headers.get("Retry-After", ""))
                except ValueError:
                    wait = 2.0 * (attempt + 1)
                # A header can say anything: a negative wait or "nan" would make the sleep itself fail.
                self._sleep(min(wait, 30.0) if 0 <= wait < math.inf else 2.0 * (attempt + 1))
                continue
            if resp.status_code >= 400:
                try:
                    body = resp.json()
                except ValueError:
                    body = None
                message = body.get("message", resp.text) if isinstance(body, dict) else resp.text
                raise AlpacaError(resp.status_code, str(message)[:200])
            return resp.json()
        raise AssertionError("unreachable")

    def _paged(self, url: str, params: dict[str, Any], max_pages: int) -> tuple[list[dict], bool]:
        """Follow next_page_token. Returns (pages, truncated)."""
        params = dict(params)
        pages: list[dict] = []
        for _ in range(max_pages):
            page = self._get(url, params)
            pages.append(page)
            token = page.get("next_page_token")
            if not token:
                return pages, False
            params["page_token"] = token
        return pages, True

    # -- Stocks -------------------------------------------------------------

    def stock_snapshots(self, symbols: list[str]) -> dict[str, StockQuote]:
        out: dict[str, StockQuote] = {}
        for chunk in _chunks(symbols, 100):
            try:
                self._snapshots(chunk, out)
            except AlpacaError as err:
                if err.status not in (400, 404, 422):
                    raise
                # One unknown symbol rejects the whole batch, so ask one at a time.
                for symbol in chunk:
                    try:
                        self._snapshots([symbol], out)
                    except AlpacaError:
                        continue
        return out

    def _snapshots(self, symbols: list[str], out: dict[str, StockQuote]) -> None:
        params = {"symbols": ",".join(symbols), "feed": self.stock_feed}
        data = self._get(f"{DATA_URL}/v2/stocks/snapshots", params)
        for symbol, snap in data.items():
            if not isinstance(snap, dict):
                continue
            previous = _num(snap.get("prevDailyBar"), "c")
            price = _num(snap.get("latestTrade"), "p") or _num(snap.get("dailyBar"), "c")
            if not price and previous:
                # Before the open, a thinly traded feed may have no trade yet today.
                price = previous
                self.notes.append(
                    f"Some stocks have not traded yet today on the {self.stock_feed} feed; "
                    "their price is the previous close."
                )
            if price:
                out[symbol] = StockQuote(symbol, price, previous)

    def daily_bars(self, symbols: list[str], start: date) -> dict[str, list[Bar]]:
        """Adjusted daily closes, oldest first. Today's bar is included once it is final."""
        if not symbols:
            return {}
        # Accounts without a real-time subscription cannot query the latest 15 minutes.
        return self._daily(symbols, start, datetime.now(timezone.utc) - timedelta(minutes=16), "all")

    def close_on(self, symbol: str, day: date) -> float | None:
        """A stock's close on one day, as it printed that day. None when that day has no final bar.

        Not adjusted. A dividend or a split since then lowers every earlier close in the adjusted
        history (COST's close of 1,097.91 on 2026-11-23 reads 1,067.91 there, after its special
        dividend), and an option that expired that day had a strike that was not lowered with it.
        """
        now = self._now()
        # The last 15 minutes cannot be asked for, and nothing past the day itself is wanted.
        end = min(datetime.combine(day, clock(23, 59, 59), ET), (now - timedelta(minutes=16)).astimezone(ET))
        if end.date() < day:
            return None
        return next((bar.close for bar in self._daily([symbol], day, end, "raw").get(symbol, []) if bar.day == day), None)

    def _daily(self, symbols: list[str], start: date, end: datetime, adjustment: str) -> dict[str, list[Bar]]:
        """Daily bars up to `end`, oldest first, with today's left out until it is final."""
        params = {
            "symbols": ",".join(symbols),
            "timeframe": "1Day",
            "start": start.isoformat(),
            "end": end.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 10000,
            "adjustment": adjustment,
            "sort": "asc",
            "feed": self._bars_feed,
        }
        try:
            pages, _ = self._paged(f"{DATA_URL}/v2/stocks/bars", params, max_pages=60)
        except AlpacaError as err:
            if self._bars_feed != "sip" or err.status not in (403, 422):
                raise
            self._bars_feed = params["feed"] = "iex"
            pages, _ = self._paged(f"{DATA_URL}/v2/stocks/bars", params, max_pages=60)
        if self._bars_feed == "iex":
            # Said on every scan, not only the one that discovered it.
            self.notes.append(IEX_BARS_NOTE)

        now = self._now()
        today = now.date()
        final = now.time() >= BAR_FINAL
        out: dict[str, list[Bar]] = {symbol: [] for symbol in symbols}
        for page in pages:
            for symbol, rows in (page.get("bars") or {}).items():
                for row in rows:
                    day = date.fromisoformat(row["t"][:10])
                    if day < today or (day == today and final):
                        out.setdefault(symbol, []).append(Bar(day, float(row["c"]), float(row.get("v", 0))))
        return out

    # -- Options ------------------------------------------------------------

    def option_chain(
        self, symbol: str, *, expiry_from: date, expiry_to: date, strike_from: float, strike_to: float
    ) -> list[OptionQuote]:
        params = {
            "feed": self.options_feed,
            "limit": 1000,
            "expiration_date_gte": expiry_from.isoformat(),
            "expiration_date_lte": expiry_to.isoformat(),
            "strike_price_gte": round(strike_from, 2),
            "strike_price_lte": round(strike_to, 2),
        }
        url = f"{DATA_URL}/v1beta1/options/snapshots/{symbol}"
        pages, truncated = self._paged(url, params, MAX_CHAIN_PAGES)
        if truncated:
            self.notes.append(f"{symbol}: the option chain was cut off at {MAX_CHAIN_PAGES * 1000:,} contracts.")
        root = symbol.replace(".", "")
        parsed = [
            (quote, snap)
            for page in pages
            for contract, snap in (page.get("snapshots") or {}).items()
            if (quote := _option_quote(symbol, root, contract, snap or {}))
        ]
        session = _session_day(q.quote_time for q, _ in parsed if q.quote_time)
        if session and (self._now().date() - session).days > MAX_QUOTE_AGE_DAYS:
            # The quotes agree with each other and are all old: a halted or delisted
            # stock, whose own last trade is just as old. Only the clock can tell.
            self.notes.append(f"{symbol}: the newest option quote is from {session:%b %d}, so none was used.")
            return []
        quotes = _from_session(parsed, session)
        old = len(parsed) - len(quotes)
        if old:
            what = "1 contract was" if old == 1 else f"{old} contracts were"
            self.notes.append(f"{symbol}: {what} left out because the latest quote is from an earlier day.")
        return quotes

    def option_quotes(self, contracts: dict[str, str]) -> dict[str, OptionQuote]:
        """The latest quote for each named contract, by OCC symbol. `contracts` maps each to its underlying.

        This is for positions already held. One request covers a hundred
        contracts whatever their underlyings, where a chain costs a request or
        more for every stock.

        The same two rules as for a chain decide what counts as a quote. Options
        on every stock trade the same sessions, so the newest quote among the
        contracts asked for says which day is current, and a quote from an
        earlier day is not used. With a single contract there is nothing to
        compare it with, and only the clock rule applies: a quote up to
        `MAX_QUOTE_AGE_DAYS` old gets through. Its time goes with it, so that
        can be seen.

        A chain says how many quotes it left out. Here each one belongs to a
        position somebody holds, so it is handed back marked `outdated`, with
        its time and without its prices. The position can then say when it was
        last quoted, where "no quote found" would send the person to check a
        strike that is perfectly right.
        """
        parsed = [
            (quote, snap)
            for chunk in _chunks(list(contracts), 100)
            for contract, snap in self._contract_snapshots(chunk).items()
            if contract in contracts
            and (quote := _option_quote(contracts[contract], contracts[contract].replace(".", ""), contract, snap or {}))
        ]
        session = _session_day(q.quote_time for q, _ in parsed if q.quote_time)
        current: dict[str, OptionQuote] = {}
        if session and (self._now().date() - session).days > MAX_QUOTE_AGE_DAYS:
            self.notes.append(f"The newest quote for the contracts held is from {session:%b %d}, so none was used.")
        else:
            current = {quote.symbol: quote for quote in _from_session(parsed, session)}
        for quote, _ in parsed:
            # Only a quote with a time can be from an earlier day. A snapshot with no quote in it
            # at all is left out, and the position says no quote was found.
            if quote.symbol not in current and quote.quote_time is not None:
                current[quote.symbol] = OptionQuote(
                    quote.symbol, quote.underlying, quote.kind, quote.strike, quote.expiry, quote_time=quote.quote_time, outdated=True
                )
        return current

    def _contract_snapshots(self, symbols: list[str]) -> dict[str, dict]:
        params = {"symbols": ",".join(symbols), "feed": self.options_feed, "limit": 1000}
        try:
            pages, _ = self._paged(f"{DATA_URL}/v1beta1/options/snapshots", params, MAX_CHAIN_PAGES)
        except AlpacaError as err:
            if err.status not in (400, 404, 422):
                raise
            # One unknown stock rejects a whole batch of stock snapshots. In case a
            # mistyped contract does the same here, ask one at a time and pass over it.
            if len(symbols) == 1:
                return {}
            return {contract: snap for symbol in symbols for contract, snap in self._contract_snapshots([symbol]).items()}
        return {
            contract: snap
            for page in pages
            for contract, snap in (page.get("snapshots") or {}).items()
            if isinstance(snap, dict)
        }

    def open_interest(
        self, symbol: str, *, expiry_from: date, expiry_to: date, strike_from: float, strike_to: float
    ) -> dict[str, int]:
        """Open interest lives on the trading API's contract list, not in the data feed."""
        params = {
            "underlying_symbols": symbol,
            "status": "active",
            "limit": 10000,
            "expiration_date_gte": expiry_from.isoformat(),
            "expiration_date_lte": expiry_to.isoformat(),
            "strike_price_gte": round(strike_from, 2),
            "strike_price_lte": round(strike_to, 2),
        }
        pages, _ = self._paged(f"{self._trading_url}/v2/options/contracts", params, max_pages=5)
        out: dict[str, int] = {}
        for page in pages:
            for row in page.get("option_contracts") or []:
                try:
                    out[row["symbol"]] = int(float(row["open_interest"]))
                except (KeyError, TypeError, ValueError):
                    continue
        return out

    # -- Discovery ----------------------------------------------------------

    def most_actives(self, top: int) -> list[str]:
        data = self._get(f"{DATA_URL}/v1beta1/screener/stocks/most-actives", {"by": "volume", "top": top})
        return [row["symbol"] for row in data.get("most_actives") or [] if row.get("symbol")]

    def movers(self, top: int) -> list[str]:
        data = self._get(f"{DATA_URL}/v1beta1/screener/stocks/movers", {"top": top})
        rows = (data.get("gainers") or []) + (data.get("losers") or [])
        return [row["symbol"] for row in rows if row.get("symbol")]

    # -- News ---------------------------------------------------------------

    def headlines(self, symbol: str, since: datetime) -> list[NewsItem]:
        """Headlines the feed tags with `symbol`, from `since` on, newest first.

        The tag is the feed's own. A headline can be about another company that
        only mentions this one, and an exchange-traded fund or a second share
        class (BRK.B) usually has none.
        """
        params = {
            "symbols": symbol,
            "start": since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": NEWS_PAGE,
            "sort": "desc",
            "include_content": "false",
        }
        pages, truncated = self._paged(f"{DATA_URL}/v1beta1/news", params, MAX_NEWS_PAGES)
        if truncated:
            self.notes.append(f"{symbol}: only the newest {MAX_NEWS_PAGES * NEWS_PAGE:,} headlines were loaded.")
        return [item for page in pages for row in page.get("news") or [] if (item := _headline(row))]


def _session_day(times: Iterable[datetime]) -> date | None:
    """The day of the latest trading session a chain has quotes from.

    A snapshot holds the last quote and the last daily bar Alpaca saw for a
    contract. For a contract that was not quoted or traded in the latest session
    those are an earlier day's. Seen live: when a special dividend or a split
    changes a stock's strikes, the contracts that stopped existing go on
    answering with their final quote, a day old or a year old. Priced against
    today's stock, such a quote can look like a bargain.

    The newest quote in the chain says which day is current, so the answer is the
    same after the close and at a weekend, and the clock is not needed. A quote
    stamped before the 9:30 open does not move the day on: until trading starts,
    the session that counts is still the one before.
    """
    local = (t.astimezone(ET) for t in times)
    return max((t.date() for t in local if t.time() >= SESSION_OPEN), default=None)


def _from_session(parsed: list[tuple[OptionQuote, dict]], session: date | None) -> list[OptionQuote]:
    """The quotes that are from the latest session, each with that session's volume."""
    quotes: list[OptionQuote] = []
    for quote, snap in parsed:
        if session and quote.quote_time and quote.quote_time.astimezone(ET).date() < session:
            continue  # not quoted in the latest session: see _session_day
        traded = _bar_day(snap)
        if session and traded and traded < session:
            quote.volume = 0  # the daily bar is the last day it traded, which was not this one
        quotes.append(quote)
    return quotes


def _bar_day(snap: dict | None) -> date | None:
    """The day a snapshot's daily bar belongs to: the last day the contract traded."""
    stamp = (snap or {}).get("dailyBar") or {}
    try:
        return date.fromisoformat(str(stamp.get("t"))[:10])
    except ValueError:
        return None


def _headline(row: Any) -> NewsItem | None:
    """One row of the news endpoint. A row with no headline or no usable time is skipped."""
    if not isinstance(row, dict):
        return None
    title = " ".join(str(row.get("headline") or "").split())
    try:
        when = datetime.fromisoformat(str(row.get("created_at")).replace("Z", "+00:00"))
    except ValueError:
        return None
    if not title or when.tzinfo is None:
        return None
    link = str(row.get("url") or "").strip()
    return NewsItem(
        when=when,
        kind="headline",
        title=title,
        source=str(row.get("source") or "Alpaca"),
        # The link comes from a third party and ends up on a page, so only a web address is kept.
        url=link if link.lower().startswith(("https://", "http://")) else None,
        ref=str(row.get("id") or ""),
        summary=" ".join(str(row.get("summary") or "").split())[:SUMMARY_CHARS],
    )


def _option_quote(underlying: str, root: str, contract: str, snap: dict) -> OptionQuote | None:
    try:
        contract_root, expiry, kind, strike = occ.parse(contract)
    except ValueError:
        return None
    if contract_root != root:
        return None  # adjusted contract with a non-standard deliverable
    quote, greeks = snap.get("latestQuote") or {}, snap.get("greeks") or {}
    volume = _num(snap.get("dailyBar"), "v")
    stamp = quote.get("t")
    try:
        quote_time = datetime.fromisoformat(str(stamp).replace("Z", "+00:00")) if stamp else None
    except ValueError:
        quote_time = None
    if quote_time is not None and quote_time.tzinfo is None:
        quote_time = None  # with no zone it cannot be set beside the other quotes' times
    return OptionQuote(
        symbol=contract,
        underlying=underlying,
        kind=kind,
        strike=strike,
        expiry=expiry,
        bid=_num(quote, "bp"),
        ask=_num(quote, "ap"),
        last=_num(snap.get("latestTrade"), "p"),
        volume=int(volume) if volume is not None else None,
        iv=_num(snap, "impliedVolatility"),
        delta=_num(greeks, "delta"),
        gamma=_num(greeks, "gamma"),
        theta=_num(greeks, "theta"),
        vega=_num(greeks, "vega"),
        quote_time=quote_time,
    )
