"""Alpaca market data over REST. Read-only: no order endpoints are used.

Endpoints and field names follow https://docs.alpaca.markets/reference.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

import httpx

from app import occ
from app.models import Bar, OptionQuote, StockQuote

DATA_URL = "https://data.alpaca.markets"
PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
ET = ZoneInfo("America/New_York")

MAX_CHAIN_PAGES = 15  # 1,000 contracts a page
_RETRIES = 3


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
    ) -> None:
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
            resp = self._client.get(url, params=params, headers=self._headers)
            busy = resp.status_code == 429 or resp.status_code >= 500
            if busy and attempt < _RETRIES:
                try:
                    wait = float(resp.headers.get("Retry-After", ""))
                except ValueError:
                    wait = 2.0 * (attempt + 1)
                self._sleep(min(wait, 30.0))
                continue
            if resp.status_code >= 400:
                try:
                    message = resp.json().get("message", resp.text)
                except ValueError:
                    message = resp.text
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
            price = _num(snap.get("latestTrade"), "p") or _num(snap.get("dailyBar"), "c")
            if price:
                out[symbol] = StockQuote(symbol, price, _num(snap.get("prevDailyBar"), "c"))

    def daily_bars(self, symbols: list[str], start: date) -> dict[str, list[Bar]]:
        """Adjusted daily closes, oldest first, excluding today's unfinished bar."""
        if not symbols:
            return {}
        # Accounts without a real-time subscription cannot query the latest 15 minutes.
        end = datetime.now(timezone.utc) - timedelta(minutes=16)
        params = {
            "symbols": ",".join(symbols),
            "timeframe": "1Day",
            "start": start.isoformat(),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": 10000,
            "adjustment": "all",
            "sort": "asc",
            "feed": self._bars_feed,
        }
        try:
            pages, _ = self._paged(f"{DATA_URL}/v2/stocks/bars", params, max_pages=60)
        except AlpacaError as err:
            if self._bars_feed != "sip" or err.status not in (403, 422):
                raise
            self._bars_feed = params["feed"] = "iex"
            self.notes.append(
                "Price history comes from the IEX exchange only, because this Alpaca "
                "plan does not include consolidated (SIP) history."
            )
            pages, _ = self._paged(f"{DATA_URL}/v2/stocks/bars", params, max_pages=60)

        today = datetime.now(ET).date()
        out: dict[str, list[Bar]] = {symbol: [] for symbol in symbols}
        for page in pages:
            for symbol, rows in (page.get("bars") or {}).items():
                for row in rows:
                    day = date.fromisoformat(row["t"][:10])
                    if day < today:
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
        quotes: list[OptionQuote] = []
        for page in pages:
            for contract, snap in (page.get("snapshots") or {}).items():
                quote = _option_quote(symbol, root, contract, snap or {})
                if quote:
                    quotes.append(quote)
        return quotes

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
        quote_time = datetime.fromisoformat(stamp.replace("Z", "+00:00")) if stamp else None
    except ValueError:
        quote_time = None
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
