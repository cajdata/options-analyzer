"""Orders on Alpaca's paper account.

This is the one module that can send an order, and the paper host is the only
host it knows. Nothing here can reach a live account: there is no setting for
the address, and Alpaca's paper host turns live keys away.

Four things are fixed, and `app/paper.py` relies on them:

* An order is a limit order for whole contracts of one option, good for the
  day. There is no market order, no order that outlives the session and no
  order with more than one leg.
* Sending is never retried. A request that timed out may have arrived, and a
  second one would be a second order. `BrokerError.refused` says which of the
  two it was: Alpaca answered and turned the order down, or nobody can tell.
* Every order carries an id this app made up (`client_id`), so an order whose
  answer was lost can be looked for afterwards.
* Reading is retried like any other lookup, and changes nothing.

Endpoints and field names follow https://docs.alpaca.markets/reference. The
account sends its numbers as text.
"""

from __future__ import annotations

import math
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

import httpx

from app import occ
from app.models import Order, PaperAccount, PaperHolding

PAPER_URL = "https://paper-api.alpaca.markets"
# Statuses after which an order does not change again. Anything else may still fill.
DONE = frozenset({"filled", "canceled", "expired", "rejected", "replaced"})
ORDER_PAGE = 500  # the most orders the list returns at once
MAX_ORDER_PAGES = 4  # past this many, `orders` says its list is not whole
_RETRIES = 2
_ORDER_ID = re.compile(r"[0-9A-Za-z-]{6,64}")
_INTENTS = {"buy_to_open": "open", "sell_to_open": "open", "buy_to_close": "close", "sell_to_close": "close"}


class BrokerError(RuntimeError):
    """The paper account could not do what was asked. The message is written for the user."""

    def __init__(self, message: str, *, status: int = 0, refused: bool = False) -> None:
        super().__init__(message)
        self.status = status
        # True: the account answered and said no, so nothing was placed or changed.
        # False: there was no clear answer, and whether it happened is not known.
        self.refused = refused


class Broker(Protocol):
    mode: str  # "paper" or "demo"

    def account(self) -> PaperAccount: ...

    def positions(self) -> list[PaperHolding]:
        """The option positions the account holds."""
        ...

    def orders(self, since: datetime) -> tuple[list[Order], bool]:
        """Every option order sent since then, whatever became of it, and whether that is all of them.

        False when the account has more than can be read at once. What is missing is then the newest, less
        the orders still working, which are asked for on their own.
        """
        ...

    def order(self, client_id: str) -> Order | None:
        """The order this app gave that id. None only when the account says it has no such order."""
        ...

    def submit(self, symbol: str, side: str, intent: str, quantity: int, limit: float, client_id: str) -> Order:
        """Send one limit order, good for the day. Raises BrokerError, and never sends twice."""
        ...

    def cancel(self, order_id: str) -> None: ...


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _number(row: dict, key: str) -> float | None:
    """A figure the account sends as text ("2.40"), or as a number. None when it is neither, or not finite."""
    try:
        value = float(row.get(key))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _when(row: dict, key: str) -> datetime | None:
    try:
        when = datetime.fromisoformat(_text(row.get(key)).replace("Z", "+00:00"))
    except ValueError:
        return None
    return when if when.tzinfo else None


def _is_option(row: dict) -> bool:
    asset = row.get("asset_class")
    if asset:
        return asset == "us_option"
    try:
        occ.parse(_text(row.get("symbol")))
    except ValueError:
        return False
    return True


def _order(row: Any) -> Order | None:
    """One order from the account. None when it is not an order for an option, or does not read as one."""
    if not isinstance(row, dict) or not _is_option(row):
        return None
    quantity, status = _number(row, "qty"), _text(row.get("status"))
    if not (row.get("id") and row.get("symbol") and quantity and status):
        return None
    return Order(
        id=str(row["id"]),
        client_id=_text(row.get("client_order_id")),
        symbol=str(row["symbol"]),
        side=_text(row.get("side")),
        intent=_INTENTS.get(_text(row.get("position_intent")), ""),
        quantity=int(quantity),
        limit=_number(row, "limit_price"),
        status=status,
        working=status not in DONE,
        filled=int(_number(row, "filled_qty") or 0),
        fill_price=_number(row, "filled_avg_price"),
        submitted=_when(row, "submitted_at") or _when(row, "created_at"),
        filled_at=_when(row, "filled_at"),
    )


def _said(response: httpx.Response) -> str:
    """What Alpaca said was wrong, cut short."""
    try:
        body = response.json()
    except ValueError:
        body = None
    message = body.get("message") if isinstance(body, dict) else None
    return str(message or response.text or f"status {response.status_code}")[:300]


class AlpacaPaper:
    """An Alpaca paper account. It holds pretend money, and its orders are never sent to an exchange."""

    mode = "paper"

    def __init__(
        self,
        key: str,
        secret: str,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client or httpx.Client(timeout=20.0)
        self._headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self._sleep = sleep

    # -- Reading ------------------------------------------------------------

    def _read(self, path: str, params: dict[str, Any] | None = None, missing: Any = ...) -> Any:
        """One GET, retried when the account is busy or cannot be reached. `missing` is the answer to a 404."""
        for attempt in range(_RETRIES + 1):
            try:
                response = self._client.get(f"{PAPER_URL}{path}", params=params, headers=self._headers)
            except httpx.TransportError as err:
                if attempt < _RETRIES:
                    self._sleep(2.0 * (attempt + 1))
                    continue
                raise BrokerError(f"Alpaca's paper account could not be reached ({type(err).__name__}).") from err
            if (response.status_code == 429 or response.status_code >= 500) and attempt < _RETRIES:
                self._sleep(2.0 * (attempt + 1))
                continue
            if response.status_code == 404 and missing is not ...:
                return missing
            if response.status_code in (401, 403):
                raise BrokerError(
                    f"Alpaca's paper account turned the keys away ({_said(response)}). Orders need the keys of a paper "
                    "account: a live account's keys are not accepted there.",
                    status=response.status_code,
                    refused=True,
                )
            if response.status_code >= 400:
                raise BrokerError(f"Alpaca's paper account answered {response.status_code}: {_said(response)}", status=response.status_code)
            try:
                return response.json()
            except ValueError:
                raise BrokerError("Alpaca's paper account sent an answer that does not read.") from None
        raise AssertionError("unreachable")

    def account(self) -> PaperAccount:
        row = self._read("/v2/account")
        row = row if isinstance(row, dict) else {}
        level = _number(row, "options_trading_level")
        return PaperAccount(int(level) if level is not None else None, _number(row, "options_buying_power"))

    def positions(self) -> list[PaperHolding]:
        rows = self._read("/v2/positions")
        held = []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict) or not _is_option(row):
                continue
            quantity, price = _number(row, "qty"), _number(row, "avg_entry_price")
            if not quantity or not price or not row.get("symbol"):
                continue
            # A short position is said in the side. Whether the count is negative as well is not documented.
            side = "short" if row.get("side") == "short" or quantity < 0 else "long"
            held.append(PaperHolding(str(row["symbol"]), side, abs(int(quantity)), price))
        return held

    def orders(self, since: datetime) -> tuple[list[Order], bool]:
        found: dict[str, Order] = {}

        def keep(rows: Any) -> list:
            rows = rows if isinstance(rows, list) else []
            for row in rows:
                if order := _order(row):
                    found[order.id] = order
            return rows

        after, whole = since, False
        for _ in range(MAX_ORDER_PAGES):
            params = {
                "status": "all",
                "limit": ORDER_PAGE,
                "direction": "asc",
                # To the microsecond, as the account stamps an order. Cut to the second, a page began again with
                # every order of its last one's second, and 500 orders in one second could not be got past.
                "after": after.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            }
            rows = keep(self._read("/v2/orders", params))
            if len(rows) < ORDER_PAGE:
                whole = True
                break
            # The list holds every order on the account, stocks too, and the next page starts after the newest one here.
            times = [when for row in rows if isinstance(row, dict) and (when := _when(row, "submitted_at") or _when(row, "created_at"))]
            if not times or max(times) <= after:
                break  # a full page with no later time to go on from: there is more, and no way to ask for it
            after = max(times)
        if not whole:
            # Read oldest first, so what was not reached is the newest. The orders still working are among
            # those, and they are the ones a person may want to cancel: they are asked for on their own.
            keep(self._read("/v2/orders", {"status": "open", "limit": ORDER_PAGE}))
        return list(found.values()), whole

    def order(self, client_id: str) -> Order | None:
        row = self._read("/v2/orders:by_client_order_id", {"client_order_id": client_id}, missing=None)
        if row is None:
            return None  # a 404: the account has no order by that name
        order = _order(row)
        if order is None:
            # The account does have it. Read as "no such order", it would be taken to have never arrived.
            raise BrokerError("Alpaca's paper account answered for that order, but its answer does not read as one.")
        return order

    # -- Changing -----------------------------------------------------------

    def submit(self, symbol: str, side: str, intent: str, quantity: int, limit: float, client_id: str) -> Order:
        check(symbol, side, intent, quantity, limit)
        body = {
            "symbol": symbol,
            "qty": str(quantity),
            "side": side,
            "type": "limit",
            "time_in_force": "day",
            "limit_price": f"{limit:.2f}",
            "position_intent": f"{side}_to_{intent}",
            "client_order_id": client_id,
        }
        try:
            # Sent once. A request that timed out may have arrived, and a second one would be a second order.
            response = self._client.post(f"{PAPER_URL}/v2/orders", json=body, headers=self._headers)
        except httpx.TransportError as err:
            raise BrokerError(
                f"Alpaca's paper account could not be reached ({type(err).__name__}), so it is not known whether the order arrived."
            ) from err
        if response.status_code >= 500:
            raise BrokerError(
                f"Alpaca's paper account answered {response.status_code}, so it is not known whether the order was taken.",
                status=response.status_code,
            )
        if response.status_code >= 400:
            raise BrokerError(f"Alpaca turned the order down: {_said(response)}", status=response.status_code, refused=True)
        try:
            order = _order(response.json())
        except ValueError:
            order = None
        if order is None:
            raise BrokerError("Alpaca took the request, but its answer does not read as an order.", status=response.status_code)
        return order

    def cancel(self, order_id: str) -> None:
        if not _ORDER_ID.fullmatch(order_id):
            raise BrokerError("That is not an order's id.", status=422, refused=True)
        try:
            response = self._client.delete(f"{PAPER_URL}/v2/orders/{order_id}", headers=self._headers)
        except httpx.TransportError as err:
            raise BrokerError(f"Alpaca's paper account could not be reached ({type(err).__name__}).") from err
        if response.status_code == 404:
            raise BrokerError("The paper account has no order with that id.", status=404, refused=True)
        if response.status_code == 422:
            raise BrokerError("That order can no longer be cancelled: it has filled or already ended.", status=422, refused=True)
        if response.status_code >= 400:
            raise BrokerError(f"Alpaca did not cancel the order: {_said(response)}", status=response.status_code)


def check(symbol: str, side: str, intent: str, quantity: int, limit: float) -> None:
    """The last look before an order leaves: one option, whole contracts, a price in whole cents."""
    occ.parse(symbol)  # raises ValueError unless it is an option's symbol
    if side not in ("buy", "sell") or intent not in ("open", "close"):
        raise ValueError("an order buys or sells, to open or to close")
    if type(quantity) is not int or quantity < 1:
        raise ValueError("an order is for a whole number of contracts")
    if not isinstance(limit, (int, float)) or isinstance(limit, bool) or not math.isfinite(limit) or limit < 0.01:
        raise ValueError("a limit price is at least one cent")
    if abs(limit * 100 - round(limit * 100)) > 1e-6:
        raise ValueError("a limit price is in whole cents")


class DemoBroker:
    """A made-up paper account for the demo. It lives in memory and sends nothing anywhere.

    An order fills at once when its limit reaches the demo's quote: a buy at the
    ask, a sale at the bid. Otherwise it waits, and since the demo's prices never
    move it waits until it is cancelled. Like the real account, it holds no
    shares, so it turns away a call sold to open. And it holds a contract one
    way or the other, so it turns away an order to open the other side of one
    it holds.
    """

    mode = "demo"

    def __init__(self, market, now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self._market, self._now = market, now
        self._orders: dict[str, Order] = {}
        self._held: dict[tuple[str, str], tuple[int, float]] = {}  # (contract, side) to (contracts, average price)

    def account(self) -> PaperAccount:
        return PaperAccount(options_level=3, buying_power=100_000.0)

    def hold(self, symbol: str, side: str, quantity: int, price: float) -> None:
        """Put a position in the account, as if an order had filled. For the demo's starting state."""
        have, average = self._held.get((symbol, side), (0, 0.0))
        self._held[(symbol, side)] = (have + quantity, (have * average + quantity * price) / (have + quantity))

    def positions(self) -> list[PaperHolding]:
        return [PaperHolding(symbol, side, count, round(price, 4)) for (symbol, side), (count, price) in self._held.items()]

    def orders(self, since: datetime) -> tuple[list[Order], bool]:
        return [o for o in self._orders.values() if o.submitted and o.submitted >= since], True

    def order(self, client_id: str) -> Order | None:
        return next((o for o in self._orders.values() if o.client_id == client_id), None)

    def submit(self, symbol: str, side: str, intent: str, quantity: int, limit: float, client_id: str) -> Order:
        check(symbol, side, intent, quantity, limit)
        root, _, kind, _ = occ.parse(symbol)
        if self.order(client_id):
            raise BrokerError("Alpaca turned the order down: client order id must be unique", status=422, refused=True)
        held = "long" if (side == "buy") == (intent == "open") else "short"
        if intent == "close" and self._held.get((symbol, held), (0, 0.0))[0] < quantity:
            raise BrokerError("Alpaca turned the order down: the account does not hold that many contracts to close", status=403, refused=True)
        if intent == "open" and self._held.get((symbol, "short" if held == "long" else "long"), (0, 0.0))[0]:
            # What Alpaca says to this has not been seen, so the words are the demo's own and are not passed off as Alpaca's.
            raise BrokerError(
                "The demo's account holds this contract the other way, and an account holds a contract one way or the other.",
                status=403,
                refused=True,
            )
        if intent == "open" and side == "sell" and kind == "call":
            raise BrokerError(
                f"Alpaca turned the order down: insufficient underlying qty available for covered call (required: {quantity * 100}, available: 0)",
                status=403,
                refused=True,
            )
        quote = self._market.option_quotes({symbol: root}).get(symbol)
        price = (quote.ask if side == "buy" else quote.bid) if quote else None
        fills = bool(price) and (limit >= price if side == "buy" else limit <= price)
        now = self._now()
        order = Order(
            id=str(uuid.uuid4()),
            client_id=client_id,
            symbol=symbol,
            side=side,
            intent=intent,
            quantity=quantity,
            limit=limit,
            status="filled" if fills else "new",
            working=not fills,
            filled=quantity if fills else 0,
            fill_price=price if fills else None,
            submitted=now,
            filled_at=now if fills else None,
        )
        self._orders[order.id] = order
        if fills and intent == "open":
            self.hold(symbol, held, quantity, price)
        elif fills:
            have, average = self._held[(symbol, held)]
            if have > quantity:
                self._held[(symbol, held)] = (have - quantity, average)
            else:
                del self._held[(symbol, held)]
        return order

    def cancel(self, order_id: str) -> None:
        order = self._orders.get(order_id)
        if order is None:
            raise BrokerError("The paper account has no order with that id.", status=404, refused=True)
        if not order.working:
            raise BrokerError("That order can no longer be cancelled: it has filled or already ended.", status=422, refused=True)
        self._orders[order_id] = Order(**{**order.__dict__, "status": "canceled", "working": False})
