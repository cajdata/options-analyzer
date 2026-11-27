"""Paper trading: the one module that sends an order, the journal, and the desk between them.

No test here reaches Alpaca. The paper account is a stand-in (`Account`), the demo's
made-up one, or `AlpacaPaper` on a transport that answers from the test.
"""

import json
import re
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest
from fastapi.testclient import TestClient

from app import broker, journal, paper, positions
from app.analysis import signals
from app.broker import AlpacaPaper, BrokerError, DemoBroker
from app.config import ConfigError, Settings, load_settings
from app.journal import Journal, StoreError
from app.models import ClosingOrder, Exit, Order, PaperAccount, PaperHolding, PaperReview, Trade
from app.paper import PaperDesk, PaperError, reconcile
from app.providers.demo import DemoMarket
from app.scanner import ScanRequest, build_scanner
from app.server import create_app

TODAY = date(2026, 11, 24)  # a Tuesday
ET = ZoneInfo("America/New_York")
NOW = datetime(2026, 11, 24, 15, 0, tzinfo=timezone.utc)  # 10 a.m. in New York
SHOWN = {"lens": "sell", "score": 72.0, "rating": "favorable", "pop": 0.78, "spans_earnings": False, "news": []}


@pytest.fixture(scope="module", autouse=True)
def demo_day_is_pinned():
    """Hold the demo market at TODAY: the command line and the dashboard build theirs from the wall clock."""
    wall_clock = DemoMarket.__init__

    def pinned(self, today=None):
        wall_clock(self, today or TODAY)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(DemoMarket, "__init__", pinned)
        yield


def trade(**changes) -> Trade:
    """A put sold to open with a limit of $2.00, 45 days from expiry, its order just sent."""
    base = dict(
        underlying="XYZ", kind="put", strike=95.0, expiry=TODAY + timedelta(days=45), side="short",
        quantity=1, limit=2.00, ordered=NOW, signals=dict(SHOWN),
    )  # fmt: skip
    return journal.new(**{**base, **changes})


def held(**changes) -> Trade:
    """That trade once its order has filled at the limit."""
    t = trade(**changes)
    t.state, t.entry_price, t.opened, t.open_order = "open", t.limit, t.ordered, f"opened-{t.id}"
    return journal.settle(t)


def order(t: Trade, intent: str = "open", status: str = "filled", **changes) -> Order:
    """An order the account reports for a trade: the one that opened it, or one that closes it."""
    opening = intent == "open"
    buys = (t.side == "long") == opening
    base = dict(
        id=f"{intent}-{t.id}",
        client_id=paper.open_id(t) if opening else (t.closing.client_id if t.closing else "typed-in-at-alpaca"),
        symbol=t.symbol,
        side="buy" if buys else "sell",
        intent=intent,
        quantity=t.quantity,
        limit=t.limit,
        status=status,
        working=status not in broker.DONE,
        filled=t.quantity if status == "filled" else 0,
        fill_price=t.limit if status == "filled" else None,
        submitted=NOW,
        filled_at=NOW + timedelta(minutes=1) if status == "filled" else None,
    )
    return Order(**{**base, **changes})


# -- The one module that sends an order ------------------------------------------------------------


def account_on(handler) -> tuple[AlpacaPaper, list[httpx.Request]]:
    """An Alpaca paper account whose answers come from `handler`, and the requests it was sent."""
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return AlpacaPaper("key", "secret", client=httpx.Client(transport=httpx.MockTransport(answer)), sleep=lambda _: None), seen


ROW = {
    "id": "61e69015-8549-4bfd-b9c3-01e75843f47d",
    "client_order_id": "sc-ab12cd34-o",
    "symbol": "XYZ270108P00095000",
    "asset_class": "us_option",
    "qty": "2",
    "filled_qty": "1",
    "filled_avg_price": "2.05",
    "limit_price": "2",
    "side": "sell",
    "type": "limit",
    "position_intent": "sell_to_open",
    "status": "partially_filled",
    "submitted_at": "2026-11-24T15:00:01.5Z",
    "filled_at": None,
}


def test_an_order_is_a_day_limit_order_sent_to_the_paper_host():
    account, seen = account_on(lambda request: httpx.Response(200, json=ROW))
    made = account.submit("XYZ270108P00095000", "sell", "open", 2, 2.0, "sc-ab12cd34-o")
    (request,) = seen
    assert request.method == "POST" and str(request.url) == "https://paper-api.alpaca.markets/v2/orders"
    assert json.loads(request.content) == {
        "symbol": "XYZ270108P00095000",
        "qty": "2",
        "side": "sell",
        "type": "limit",
        "time_in_force": "day",
        "limit_price": "2.00",
        "position_intent": "sell_to_open",
        "client_order_id": "sc-ab12cd34-o",
    }
    # The account sends its numbers as text.
    assert (made.quantity, made.filled, made.fill_price, made.limit) == (2, 1, 2.05, 2.0)
    assert (made.intent, made.status, made.working, made.client_id) == ("open", "partially_filled", True, "sc-ab12cd34-o")
    assert made.submitted == datetime(2026, 11, 24, 15, 0, 1, 500000, tzinfo=timezone.utc) and made.filled_at is None


def test_the_broker_knows_one_host_and_no_other_module_sends_an_order():
    # Nothing can reach a live account: the address is not a setting, and it is written in one place.
    source = Path(broker.__file__).read_text(encoding="utf-8")
    hosts = set(re.findall(r"https://[\w.-]+", source))
    assert hosts == {"https://paper-api.alpaca.markets", "https://docs.alpaca.markets"}
    assert "url" not in AlpacaPaper.__init__.__code__.co_varnames and "paper" not in AlpacaPaper.__init__.__code__.co_varnames
    for path in Path(broker.__file__).parent.rglob("*.py"):
        if path.name != "broker.py":
            text = path.read_text(encoding="utf-8")
            assert "/v2/orders" not in text and "client.post(" not in text and "client.delete(" not in text, path.name


@pytest.mark.parametrize(
    "answer, refused",
    [
        (httpx.ConnectTimeout("no answer"), False),  # it may have arrived
        (httpx.Response(503, text="try later"), False),  # and so may this one
        (httpx.Response(403, json={"code": 40310000, "message": "insufficient options buying power"}), True),
        (httpx.Response(422, json={"message": "qty must be integer for options trading"}), True),
    ],
)
def test_sending_is_never_retried_and_says_whether_the_order_may_have_arrived(answer, refused):
    def handler(request):
        if isinstance(answer, Exception):
            raise answer
        return answer

    account, seen = account_on(handler)
    with pytest.raises(BrokerError) as caught:
        account.submit("XYZ270108P00095000", "sell", "open", 1, 2.0, "sc-ab12cd34-o")
    assert len(seen) == 1  # a second request would be a second order
    assert caught.value.refused is refused
    assert ("turned the order down" in str(caught.value)) is refused
    assert ("not known" in str(caught.value)) is not refused


@pytest.mark.parametrize(
    "symbol, side, intent, quantity, limit",
    [
        ("XYZ", "sell", "open", 1, 2.0),  # a stock, not an option
        ("XYZ270108P00095000", "short", "open", 1, 2.0),
        ("XYZ270108P00095000", "sell", "roll", 1, 2.0),
        ("XYZ270108P00095000", "sell", "open", 0, 2.0),
        ("XYZ270108P00095000", "sell", "open", 1.5, 2.0),
        ("XYZ270108P00095000", "sell", "open", True, 2.0),
        ("XYZ270108P00095000", "sell", "open", 1, 0.0),
        ("XYZ270108P00095000", "sell", "open", 1, 2.005),  # finer than a cent
        ("XYZ270108P00095000", "sell", "open", 1, float("nan")),
    ],
)
def test_what_is_not_an_order_never_leaves(symbol, side, intent, quantity, limit):
    account, seen = account_on(lambda request: httpx.Response(200, json=ROW))
    with pytest.raises(ValueError):
        account.submit(symbol, side, intent, quantity, limit, "sc-ab12cd34-o")
    assert seen == []


def test_reading_is_retried_and_reads_what_the_account_sends():
    answers = iter([httpx.Response(503), httpx.Response(200, json=[ROW, {**ROW, "id": "a-stock", "asset_class": "us_equity", "symbol": "XYZ"}, "stray"])])
    account, seen = account_on(lambda request: next(answers))
    (found,), whole = account.orders(NOW - timedelta(hours=1))
    assert len(seen) == 2 and found.id == ROW["id"] and whole  # the stock order and the stray line are passed over
    assert seen[1].url.params["status"] == "all" and seen[1].url.params["after"] == "2026-11-24T14:00:00Z"

    rows = [
        {"symbol": "XYZ270108P00095000", "asset_class": "us_option", "qty": "2", "side": "short", "avg_entry_price": "2.05"},
        {"symbol": "XYZ270108C00105000", "asset_class": "us_option", "qty": "-1", "side": "long", "avg_entry_price": "1.10"},
        {"symbol": "XYZ", "asset_class": "us_equity", "qty": "100", "side": "long", "avg_entry_price": "98.00"},
    ]
    account, _ = account_on(lambda request: httpx.Response(200, json=rows))
    assert account.positions() == [
        PaperHolding("XYZ270108P00095000", "short", 2, 2.05),
        PaperHolding("XYZ270108C00105000", "short", 1, 1.10),  # a count below zero says short as well
    ]

    account, _ = account_on(lambda request: httpx.Response(200, json={"options_trading_level": 2, "options_buying_power": "9395.5"}))
    assert account.account() == PaperAccount(2, 9395.5)


def test_an_order_can_be_looked_for_by_the_id_this_app_gave_it():
    account, seen = account_on(lambda request: httpx.Response(200, json=ROW))
    assert account.order("sc-ab12cd34-o").id == ROW["id"]
    assert seen[0].url.path == "/v2/orders:by_client_order_id" and seen[0].url.params["client_order_id"] == "sc-ab12cd34-o"
    account, _ = account_on(lambda request: httpx.Response(404, json={"message": "order not found"}))
    assert account.order("sc-ab12cd34-o") is None  # the account never had it
    account, _ = account_on(lambda request: httpx.Response(500))
    with pytest.raises(BrokerError):
        account.order("sc-ab12cd34-o")  # which is not the same as not being able to ask


def test_live_keys_are_turned_away_in_words():
    account, _ = account_on(lambda request: httpx.Response(401, json={"message": "unauthorized."}))
    with pytest.raises(BrokerError, match="keys of a paper account"):
        account.positions()


def test_cancelling_says_when_it_is_too_late():
    account, seen = account_on(lambda request: httpx.Response(204))
    account.cancel(ROW["id"])
    assert seen[0].method == "DELETE" and seen[0].url.path == f"/v2/orders/{ROW['id']}"
    for status, said in ((422, "can no longer be cancelled"), (404, "no order with that id")):
        account, _ = account_on(lambda request, status=status: httpx.Response(status, json={"message": "no"}))
        with pytest.raises(BrokerError, match=said):
            account.cancel(ROW["id"])
    account, seen = account_on(lambda request: httpx.Response(204))
    with pytest.raises(BrokerError) as caught:
        account.cancel("../account")  # not an id, so nothing is asked
    assert seen == [] and (caught.value.status, caught.value.refused) == (422, True)
    with pytest.raises(PaperError) as caught:
        PaperDesk(Journal(None), None, account).cancel("../account")
    assert caught.value.status == 422  # what was asked is wrong. It used to answer 502, as if Alpaca had failed.


def test_the_demo_account_fills_at_its_quote_or_waits_and_holds_no_shares():
    market = DemoMarket(TODAY)
    (quote,) = [q for q in market.option_chain("CRUX", expiry_from=TODAY + timedelta(days=20), expiry_to=TODAY + timedelta(days=30), strike_from=394, strike_to=396) if q.kind == "put"][:1]
    account = DemoBroker(market, now=lambda: NOW)
    filled = account.submit(quote.symbol, "sell", "open", 2, quote.bid, "sc-a-o")
    assert (filled.status, filled.filled, filled.fill_price) == ("filled", 2, quote.bid)
    assert account.positions() == [PaperHolding(quote.symbol, "short", 2, quote.bid)]
    waiting = account.submit(quote.symbol, "buy", "close", 2, 0.01, "sc-a-c1")  # nobody sells at a cent
    assert waiting.working and account.positions()[0].quantity == 2
    account.cancel(waiting.id)
    assert not account.order("sc-a-c1").working
    with pytest.raises(BrokerError, match="can no longer"):
        account.cancel(waiting.id)
    account.submit(quote.symbol, "buy", "close", 2, quote.ask, "sc-a-c2")
    assert account.positions() == []
    for sent, said in (
        ((quote.symbol, "buy", "close", 1, quote.ask, "sc-a-c3"), "does not hold"),
        ((quote.symbol.replace("P", "C"), "sell", "open", 1, 1.0, "sc-b-o"), "covered call"),  # as Alpaca does without the shares
        ((quote.symbol, "sell", "open", 1, quote.bid, "sc-a-o"), "must be unique"),
    ):
        with pytest.raises(BrokerError, match=said) as caught:
            account.submit(*sent)
        assert caught.value.refused


# -- The journal -------------------------------------------------------------------------------


def test_a_trade_round_trips_through_the_file(tmp_path):
    book = Journal(tmp_path / "data" / "journal.json")
    assert book.load() == ([], 0)  # no file yet is no trades, and not an error
    first = book.add(trade())
    second = held(underlying=" brk-b ", kind="call", strike=512.5, side="long", quantity=3, limit=7.15)
    second.closing = ClosingOrder("sc-x-c1", 3, 9.0, NOW, ("time_stop",), "an-order")
    second.exits.append(Exit(1, 8.0, NOW + timedelta(days=2), "outside"))
    book.add(second)
    assert Journal(book.path).load() == ([first, second], 0)
    assert second.underlying == "BRK.B" and second.symbol == "BRKB270108C00512500"  # as a position is spelled

    # The file is meant to be read, and edited by hand if need be.
    on_file = json.loads(book.path.read_text())["trades"]
    assert on_file[0]["ordered"] == "2026-11-24T15:00:00+00:00" and on_file[0]["expiry"] == "2027-01-08"
    assert on_file[0]["state"] == "ordered" and on_file[0]["stands"] == "ordered, not filled yet"
    assert on_file[1]["closing"]["rules"] == ["time_stop"] and on_file[1]["exits"][0]["how"] == "outside"
    assert not list(tmp_path.glob("data/*.tmp"))


def test_a_result_is_worked_out_from_the_exits():
    # A seller keeps what the option cost less to buy back, across every contract, and a buyer the reverse.
    sold = held(quantity=2, limit=2.00)
    sold.exits.append(Exit(1, 0.90, NOW + timedelta(days=9), "order", ("profit_target",)))
    journal.settle(sold)
    assert (sold.state, sold.remaining, sold.pnl, sold.closed) == ("open", 1, None, None)  # one contract is still held
    sold.exits.append(Exit(1, 1.10, NOW + timedelta(days=12), "order", ("profit_target", "time_stop")))
    journal.settle(sold)
    assert (sold.state, sold.remaining, sold.exit_price) == ("closed", 0, 1.0)
    assert (sold.pnl, sold.pnl_pct, sold.closed) == (200.0, 0.5, NOW + timedelta(days=12))
    assert sold.stands == "at the profit target"

    bought = held(side="long", limit=3.00)
    bought.exits.append(Exit(1, 1.20, NOW + timedelta(days=5), "order", ()))
    # Only a seller has a profit target, so a bought option has one rule to be ahead of, and is not said to have two.
    assert (journal.settle(bought).pnl, bought.pnl_pct, bought.stands) == (-180.0, -0.6, "before the time stop")

    # With no price on record for an exit there is no result, and nothing is made up for one.
    gone = held()
    gone.exits.append(Exit(1, None, NOW, "unknown"))
    assert (journal.settle(gone).state, gone.pnl, gone.stands) == ("closed", None, journal.HOW["unknown"])


def test_a_result_is_an_amount_in_cents():
    # 100 * (1.03 - 0.52) is 51.00000000000001 in floating point.
    sold = held(limit=1.03)
    sold.exits.append(Exit(1, 0.52, NOW, "order", ()))
    assert journal.settle(sold).pnl == 51.0


@pytest.mark.parametrize("written", ["{half a file", "[]", json.dumps({"trades": "none"}), json.dumps({"version": 1})])
def test_a_journal_that_does_not_read_is_an_error_and_is_left_as_it_is(tmp_path, written):
    # As for the positions file. Read as empty, it would be written over by the next order.
    path = tmp_path / "journal.json"
    path.write_text(written)
    book = Journal(path)
    for use in (book.load, lambda: book.add(trade()), lambda: book.save([trade()]), lambda: book.remove("abc")):
        with pytest.raises(StoreError, match="journal.json"):
            use()
    assert path.read_text() == written


def test_an_entry_that_does_not_read_is_kept_and_what_was_added_by_hand_survives(tmp_path):
    path = tmp_path / "journal.json"
    mine = {**journal._plain(held(id="mine")), "why": "sold after the dip"}
    odd = {"id": "odd", "underlying": "XYZ", "strike": "about 95"}
    no_zone = {**journal._plain(trade(id="nozone")), "ordered": "2026-11-24T10:00:00"}  # cannot be set beside the broker's times
    no_price = {**journal._plain(held(id="noprice")), "entry_price": None}  # open, with nothing to measure a result against
    path.write_text(json.dumps({"trades": [odd, mine, no_zone, no_price, "a stray line"]}))
    book = Journal(path)
    found, unread = book.load()
    assert [t.id for t in found] == ["mine"] and unread == 4
    assert "4 entries" in journal.unread_note(unread, path) and "1 entry" in journal.unread_note(1, path)

    (t,) = found
    t.exits.append(Exit(1, 1.0, NOW + timedelta(days=3), "order", ("profit_target",)))
    book.save([t])
    on_file = json.loads(path.read_text())["trades"]
    assert on_file[0] == odd and on_file[2:] == [no_zone, no_price, "a stray line"]
    assert on_file[1]["why"] == "sold after the dip" and on_file[1]["state"] == "closed" and on_file[1]["pnl"] == 100.0
    with pytest.raises(ValueError, match="no trade"):
        book.save([trade(id="never-added")])
    assert not book.remove("odd")  # it does not read as a trade, so it is not one to remove


def test_the_result_on_file_is_never_trusted_over_the_exits(tmp_path):
    path = tmp_path / "journal.json"
    done = held(id="done")
    done.exits.append(Exit(1, 0.50, NOW + timedelta(days=3), "order", ()))
    path.write_text(json.dumps({"trades": [{**journal._plain(journal.settle(done)), "pnl": 99999.0, "stands": "a triumph"}]}))
    (found,), _ = Journal(path).load()
    assert (found.pnl, found.stands) == (150.0, "before either rule was reached")


def test_two_entries_with_one_id_are_never_changed_together(tmp_path):
    path = tmp_path / "journal.json"
    first, copy = journal._plain(held(id="same")), journal._plain(held(id="same", strike=90.0))
    path.write_text(json.dumps({"trades": [first, copy]}))
    book = Journal(path)
    found, unread = book.load()
    assert [(t.id, t.strike) for t in found] == [("same", 95.0)] and unread == 1
    with pytest.raises(ValueError, match="already"):
        book.add(trade(id="same"))
    assert book.remove("same") and json.loads(path.read_text())["trades"] == [copy]


def test_the_journal_has_its_own_file_and_the_demo_has_none(tmp_path, monkeypatch):
    monkeypatch.setenv("JOURNAL_FILE", str(tmp_path / "mine.json"))
    assert load_settings().journal_file == tmp_path / "mine.json"
    monkeypatch.delenv("JOURNAL_FILE")
    assert load_settings().journal_file == Settings().journal_file != Settings().positions_file
    assert Settings().journal_file.parent != Settings().cache_dir  # nothing that clears the cache may take it
    settings = Settings(journal_file=tmp_path / "journal.json")
    assert journal.open_journal(settings).path == settings.journal_file
    demo = journal.open_journal(settings, demo=True)
    demo.add(trade())
    assert demo.path is None and len(demo.load()[0]) == 1 and not settings.journal_file.exists()


# -- Bringing the journal up to date with the account ------------------------------------------------


def test_an_opening_order_that_fills_opens_the_trade_at_the_average_fill():
    t = trade(quantity=3)
    filled = order(t, filled=2, fill_price=2.0312, status="canceled")  # two of the three, and then the day ended
    assert reconcile([t], [filled], None, TODAY, NOW) == [t]
    assert (t.state, t.quantity, t.entry_price, t.opened, t.open_order) == ("open", 2, 2.0312, filled.submitted, filled.id)
    assert "2 of the 3" in t.note


@pytest.mark.parametrize("status", ["canceled", "expired", "rejected"])
def test_an_opening_order_that_ends_with_nothing_filled_is_unfilled(status):
    t = trade()
    reconcile([t], [order(t, status=status)], None, TODAY, NOW)
    assert (t.state, t.entry_price) == ("unfilled", None) and status in t.note


def test_an_order_still_working_changes_nothing_however_much_has_filled():
    t = trade(quantity=3)
    working = order(t, status="partially_filled", filled=2, fill_price=2.0)
    assert reconcile([t], [working], [PaperHolding(t.symbol, "short", 2, 2.0)], TODAY, NOW) == [t]  # only its id was learned
    assert (t.state, t.quantity, t.open_order) == ("ordered", 3, working.id)
    assert reconcile([t], [working], [PaperHolding(t.symbol, "short", 2, 2.0)], TODAY, NOW) == []


def test_an_order_whose_answer_was_lost_is_asked_after_and_never_guessed_at():
    # The account confirmed this one, and the list does not have it. It is asked after by name, since a long
    # list can be cut short. But the account took it once, so "no such order" is no answer: it is waited for.
    confirmed = trade()
    confirmed.open_order = "the-account-has-it"
    asked: list[str] = []

    def never_arrived(client_id):
        asked.append(client_id)
        return None

    assert reconcile([confirmed], [], None, TODAY, NOW, lookup=never_arrived) == [] and asked == [paper.open_id(confirmed)]
    assert (confirmed.state, confirmed.note) == ("ordered", None)
    reconcile([confirmed], [], None, TODAY, NOW, lookup=lambda client_id: order(confirmed))
    assert (confirmed.state, confirmed.entry_price) == ("open", 2.0)  # found by name, though the list did not have it
    asked.clear()

    # No answer came back for this one. Only when the account says it never had it is it taken to have never arrived.
    lost = trade()
    assert reconcile([lost], [], None, TODAY, NOW) == [] and lost.state == "ordered"  # nobody to ask

    def cannot_ask(client_id):
        raise BrokerError("could not be reached")

    assert reconcile([lost], [], None, TODAY, NOW, lookup=cannot_ask) == [] and lost.state == "ordered"
    reconcile([lost], [], None, TODAY, NOW, lookup=never_arrived)
    assert asked == [paper.open_id(lost)] and lost.state == "unfilled" and "never arrived" in lost.note

    found = trade()
    reconcile([found], [], None, TODAY, NOW, lookup=lambda client_id: order(found))
    assert (found.state, found.entry_price) == ("open", 2.0)


def test_a_closing_order_that_fills_is_an_exit_with_the_rules_it_was_sent_under():
    t = held(quantity=2)
    t.closing = ClosingOrder("sc-x-c1", 2, 0.95, NOW, ("profit_target",))
    closed = order(t, "close", fill_price=0.9, id="the-close")
    assert reconcile([t], [closed], [], TODAY, NOW) == [t]
    assert t.exits == [Exit(2, 0.9, closed.filled_at, "order", ("profit_target",), "the-close")]
    assert (t.state, t.closing, t.pnl, t.stands) == ("closed", None, 220.0, "at the profit target")


def test_a_closing_order_that_ends_unfilled_leaves_the_trade_open():
    t = held()
    t.closing = ClosingOrder("sc-x-c1", 1, 0.10, NOW, ())
    reconcile([t], [order(t, "close", status="expired")], [PaperHolding(t.symbol, "short", 1, 2.0)], TODAY, NOW)
    assert (t.state, t.closing, t.exits) == ("open", None, [])

    # And one that filled in part closes that part.
    two = held(quantity=2)
    two.closing = ClosingOrder("sc-y-c1", 2, 0.95, NOW, ())
    reconcile([two], [order(two, "close", status="canceled", filled=1, fill_price=0.95)], [PaperHolding(two.symbol, "short", 1, 2.0)], TODAY, NOW)
    assert (two.state, two.remaining, two.closing, two.pnl) == ("open", 1, None, None)


def test_a_closing_order_still_working_keeps_the_account_from_being_read_as_short_of_contracts():
    t = held(quantity=2)
    t.closing = ClosingOrder("sc-x-c1", 2, 0.95, NOW, (), "the-close")
    half = order(t, "close", status="partially_filled", filled=1, fill_price=0.95, id="the-close")
    assert reconcile([t], [half], [PaperHolding(t.symbol, "short", 1, 2.0)], TODAY, NOW) == []
    assert (t.state, t.exits) == ("open", [])  # the missing contract is that order's, and it has not ended


def test_contracts_closed_by_an_order_this_app_did_not_send_take_that_orders_price():
    t = held(quantity=2)
    elsewhere = order(t, "close", quantity=1, filled=1, fill_price=1.25, id="typed-in")
    before = order(t, "close", filled=2, fill_price=0.01, id="older", filled_at=t.opened - timedelta(days=1))  # not this trade's
    reconcile([t], [before, elsewhere], [PaperHolding(t.symbol, "short", 1, 2.0)], TODAY, NOW)
    assert t.exits == [Exit(1, 1.25, elsewhere.filled_at, "outside", None, "typed-in")] and (t.state, t.remaining) == ("open", 1)
    # The same order is never counted twice.
    assert reconcile([t], [before, elsewhere], [PaperHolding(t.symbol, "short", 1, 2.0)], TODAY, NOW) == []


def test_a_trade_held_to_expiry_is_settled_at_what_exercising_paid_that_day():
    gone = TODAY - timedelta(days=4)
    closes = {"XYZ": 93.50}
    asked = []

    def close_on(underlying, day):
        asked.append((underlying, day))
        return closes.get(underlying)

    put = held(expiry=gone, ordered=NOW - timedelta(days=40))
    reconcile([put], [], [], TODAY, NOW, close_on=close_on)
    assert asked == [("XYZ", gone)]
    assert put.exits == [Exit(1, 1.5, datetime(2026, 11, 20, 16, 0, tzinfo=ET), "expiry")]
    assert (put.state, put.pnl, put.stands) == ("closed", 50.0, "held to expiry")  # sold for 2.00, worth 1.50 at the end
    assert "$93.50" in put.note and "shares" in put.note

    # Out of the money it is worth nothing: a seller keeps it all, and a buyer loses it all.
    call = held(kind="call", strike=100.0, side="long", expiry=gone, ordered=NOW - timedelta(days=40), limit=3.0)
    reconcile([call], [], [], TODAY, NOW, close_on=close_on)
    assert (call.exit_price, call.pnl, call.pnl_pct) == (0.0, -300.0, -1.0) and "worthless" in call.note

    # Until that day's close is in, the trade waits as it is.
    waiting = held(underlying="ABC", expiry=TODAY, ordered=NOW - timedelta(days=40))
    assert reconcile([waiting], [], [], TODAY, NOW, close_on=close_on) == [] and waiting.state == "open"
    assert reconcile([held(expiry=gone, ordered=NOW - timedelta(days=40))], [], [], TODAY, NOW) == []  # nothing to ask


def test_contracts_that_left_with_no_order_on_record_close_with_no_result():
    t = held()
    reconcile([t], [], [], TODAY, NOW)
    assert t.exits == [Exit(1, None, NOW, "unknown")] and (t.state, t.pnl) == ("closed", None)
    assert "Resetting" in t.note and not signals.counted(t)


def test_nothing_is_concluded_when_the_accounts_positions_could_not_be_read():
    t = held()
    assert reconcile([t], [], None, TODAY, NOW) == [] and t.state == "open"
    # Nor when the account holds as many as the journal says, or more.
    assert reconcile([t], [], [PaperHolding(t.symbol, "short", 5, 2.0)], TODAY, NOW) == []
    # The other side of the same contract is not this trade's.
    reconcile([t], [], [PaperHolding(t.symbol, "long", 1, 2.0)], TODAY, NOW)
    assert t.state == "closed"


def test_with_two_trades_on_one_contract_the_older_is_closed_first():
    older, newer = held(ordered=NOW - timedelta(days=5)), held(ordered=NOW - timedelta(days=1))
    reconcile([newer, older], [], [PaperHolding(older.symbol, "short", 1, 2.0)], TODAY, NOW)
    assert (older.state, newer.state) == ("closed", "open")


# -- Which signals worked --------------------------------------------------------------------------


def closed(pnl_per_share: float, **shown) -> Trade:
    """A closed trade that made that much a share, with those readings from the scan."""
    lens = shown.get("lens", "sell")
    t = held(side="long" if lens == "buy" else "short", signals={**SHOWN, **shown})
    price = t.entry_price + pnl_per_share if lens == "buy" else t.entry_price - pnl_per_share
    t.exits.append(Exit(1, price, NOW + timedelta(days=5), "order", shown.pop("rules", ("profit_target",))))
    return journal.settle(t)


def test_the_readings_a_trade_is_sorted_by():
    read = lambda name, **shown: signals.GROUPS[name][1](closed(1.0, **shown))  # noqa: E731
    assert [read("score", score=s) for s in (70.0, 69.9, 50.0, 49.9)] == ["70 or more", "50 to 70", "50 to 70", "under 50"]
    assert [read("pop", pop=p) for p in (0.7, 0.55, 0.39)] == ["70% or more", "40% to 70%", "under 40%"]
    assert read("score", score=None) is None and read("pop", pop="high") is None  # no reading, so no row
    assert read("rating", rating=None) == "not rated" and read("rating", rating="balanced") == "balanced"
    assert read("earnings", spans_earnings=True) == "open over a report"
    assert read("earnings", earnings_unknown=True) == "report date unknown" and read("earnings") == "clear of a report"
    assert read("news", news=["takeover"]) == "a flagged event in the news" and read("news") == "no flagged event"
    assert read("ended") == "at the profit target"


def test_bought_and_sold_are_never_counted_together_and_every_row_has_its_count():
    trades = [closed(1.0), closed(1.0, rating="balanced"), closed(-3.0), closed(0.5, lens="buy"), closed(-1.0, lens="buy")]
    trades += [held(), trade()]  # still open, and not filled yet: neither has a result
    found = signals.summary(trades)
    assert list(found) == ["buy", "sell"]
    everything = found["sell"][0]
    assert (everything.group, everything.trades, everything.won, everything.lost) == ("all", 3, 2, 1)
    assert (everything.total, everything.mean_pct) == (-100.0, pytest.approx(-1 / 6, abs=1e-6))  # a seller can lose more than was collected
    ratings = {row.bucket: row.trades for row in found["sell"] if row.group == "rating"}
    assert ratings == {"favorable": 2, "balanced": 1}
    assert (found["buy"][0].trades, found["buy"][0].won, found["buy"][0].total) == (2, 1, -50.0)
    assert signals.summary([held(), trade()]) == {}  # nothing closed, nothing to say
    # Every surface says "too few" below this many, as an event study does.
    assert signals.MIN_TRADES == 10 and everything.trades < signals.MIN_TRADES


def test_a_trade_with_no_result_or_no_readings_is_counted_nowhere():
    typed_in = closed(1.0)
    typed_in.signals = {}
    assert not signals.counted(typed_in) and signals.summary([typed_in]) == {}


# -- The desk ----------------------------------------------------------------------------------------


class Account:
    """A stand-in for the paper account. It takes every order, and can be told to turn one down or to lose its answer."""

    mode = "paper"

    def __init__(self) -> None:
        self.sent: list[tuple] = []  # every order submitted, as it was
        self.fills, self.turns_down, self.loses_the_answer, self.arrives = True, None, False, True
        self.readable = True
        self.whole = True  # False: the list of orders was cut short
        self.read: list[str] = []  # what a review read, in the order it read it
        self.listed: dict[str, Order] = {}
        self.holds: dict[tuple[str, str], int] = {}
        self.paid: dict[tuple[str, str], float] = {}  # what each position was opened at
        self.before = lambda: None  # run as an order arrives

    def account(self) -> PaperAccount:
        return PaperAccount(3, 50_000.0)

    def positions(self) -> list[PaperHolding]:
        self.read.append("positions")
        if not self.readable:
            raise BrokerError("could not be reached")
        return [PaperHolding(*key, count, self.paid.get(key, 2.0)) for key, count in self.holds.items() if count]

    def orders(self, since: datetime) -> tuple[list[Order], bool]:
        self.read.append("orders")
        return list(self.listed.values()), self.whole

    def order(self, client_id: str) -> Order | None:
        return next((o for o in self.listed.values() if o.client_id == client_id), None)

    def submit(self, symbol: str, side: str, intent: str, quantity: int, limit: float, client_id: str) -> Order:
        self.before()
        self.sent.append((symbol, side, intent, quantity, limit, client_id))
        if self.turns_down:
            raise BrokerError(f"Alpaca turned the order down: {self.turns_down}", status=403, refused=True)
        made = Order(
            f"order-{len(self.sent)}", client_id, symbol, side, intent, quantity, limit,
            "filled" if self.fills else "new", not self.fills, quantity if self.fills else 0, limit if self.fills else None, NOW, NOW if self.fills else None,
        )  # fmt: skip
        if self.arrives:
            self.listed[made.id] = made
            if self.fills:
                key = (symbol, "long" if (side == "buy") == (intent == "open") else "short")
                self.holds[key] = self.holds.get(key, 0) + (quantity if intent == "open" else -quantity)
                if intent == "open":
                    self.paid[key] = limit
        if self.loses_the_answer:
            raise BrokerError("Alpaca's paper account could not be reached (ReadTimeout), so it is not known whether the order arrived.")
        return made

    def cancel(self, order_id: str) -> None:
        self.listed[order_id] = replace(self.listed[order_id], status="canceled", working=False)


@pytest.fixture(scope="module")
def scanner():
    return build_scanner(Settings(), demo=True)


@pytest.fixture(scope="module")
def scan(scanner):
    """Puts to sell on one demo stock, a month or so out."""
    return scanner.scan(ScanRequest(symbols=["CRUX"], lens="sell", kind="put", dte_min=25, dte_max=45))


@pytest.fixture()
def desk(scanner, tmp_path):
    made = PaperDesk(Journal(tmp_path / "journal.json"), scanner, Account(), clock=lambda: made.now)
    made.now = NOW
    return made


def test_writing_an_order_out_sends_nothing_and_writes_nothing(desk, scan):
    idea = scan.ideas[0]
    ticket = desk.write_open(scan, idea.symbol, 2, 4.10)
    assert ticket.text == f"Sell to open 2 CRUX {idea.expiry:%b %d} {idea.strike:g} put at $4.10 a share or more, good for today."
    assert "$820" in ticket.notes[0] and f"${idea.strike * 200:,.0f}" in ticket.notes[0]  # what it brings in, and the cash set aside
    assert any("pretend money" in note for note in ticket.notes)
    assert (ticket.order_side, ticket.side, ticket.action, ticket.made) == ("sell", "short", "open", NOW)
    # The journal will keep what the scan showed for the contract, copied as plain values.
    assert ticket.signals["score"] == idea.score and ticket.signals["rating"] == idea.risk_reward.rating
    assert ticket.signals["rank"] == 1 and ticket.signals["lens"] == "sell" and ticket.signals["scan"] == scan.id
    json.dumps(ticket.signals, allow_nan=False)
    assert desk.broker.sent == [] and desk.journal.load() == ([], 0) and not desk.journal.path.exists()


def test_a_written_out_order_is_sent_once_as_it_was_shown(desk, scan):
    idea = scan.ideas[0]
    ticket = desk.write_open(scan, idea.symbol, 2, 4.10)
    made = desk.send(ticket.id)
    assert desk.broker.sent == [(idea.symbol, "sell", "open", 2, 4.10, f"sc-{made.id}-o")]
    assert (made.state, made.quantity, made.entry_price, made.open_order) == ("open", 2, 4.10, "order-1")
    assert made.signals == ticket.signals and desk.journal.load() == ([made], 0)
    with pytest.raises(PaperError, match="no longer waiting") as caught:
        desk.send(ticket.id)
    assert caught.value.status == 409 and len(desk.broker.sent) == 1
    with pytest.raises(PaperError, match="no longer waiting"):
        desk.send("never-written")


def test_an_order_left_too_long_is_not_sent(desk, scan):
    ticket = desk.write_open(scan, scan.ideas[0].symbol, 1, 4.10)
    desk.now = NOW + timedelta(minutes=paper.TICKET_MINUTES, seconds=1)
    with pytest.raises(PaperError, match="Write it out again"):
        desk.send(ticket.id)
    assert desk.broker.sent == [] and desk.journal.load() == ([], 0)


def test_the_journal_has_the_trade_before_the_order_leaves(desk, scan):
    on_file = []
    desk.broker.before = lambda: on_file.extend(t.state for t in desk.journal.load()[0])
    desk.send(desk.write_open(scan, scan.ideas[0].symbol, 1, 4.10).id)
    assert on_file == ["ordered"]


def test_an_order_the_account_turns_down_leaves_nothing_behind(desk, scan):
    desk.broker.turns_down = "insufficient options buying power for cash-secured put (required: 40500, available: 9395)"
    with pytest.raises(PaperError, match="required: 40500") as caught:
        desk.send(desk.write_open(scan, scan.ideas[0].symbol, 1, 4.10).id)
    assert caught.value.status == 422 and len(desk.broker.sent) == 1 and desk.journal.load() == ([], 0)


def test_an_order_whose_answer_was_lost_is_kept_and_then_found(desk, scan):
    desk.broker.loses_the_answer = True
    with pytest.raises(PaperError, match="next refresh asks the account") as caught:
        desk.send(desk.write_open(scan, scan.ideas[0].symbol, 1, 4.10).id)
    ((kept,), _) = desk.journal.load()
    assert caught.value.status == 502 and (kept.state, kept.open_order) == ("ordered", "")
    review = desk.review()  # the order did arrive, and filled
    assert [(t.state, t.entry_price, t.open_order) for t in review.trades] == [("open", 4.10, "order-1")]
    assert len(desk.broker.sent) == 1  # looking sends nothing

    desk.broker.arrives = False
    with pytest.raises(PaperError):
        desk.send(desk.write_open(scan, scan.ideas[1].symbol, 1, 4.10).id)
    lost = next(t for t in desk.review().trades if t.symbol == scan.ideas[1].symbol)
    assert lost.state == "unfilled" and "never arrived" in lost.note


def test_closing_records_the_rules_that_were_reached_and_ends_the_trade(desk, scan):
    idea = scan.ideas[0]
    # Sold for three times what it costs to buy back now, so the profit target is reached.
    made = desk.send(desk.write_open(scan, idea.symbol, 1, round(idea.ask * 3, 2)).id)
    ticket = desk.write_close(made.id, idea.ask)
    assert ticket.text == f"Buy to close 1 CRUX {idea.expiry:%b %d} {idea.strike:g} put at ${idea.ask:.2f} a share or less, good for today."
    assert f"sold at ${made.entry_price:.2f}" in ticket.notes[0] and "ends up" in ticket.notes[0]
    assert len(desk.broker.sent) == 1  # written out, not sent

    done = desk.send(ticket.id)
    assert desk.broker.sent[1][:5] == (idea.symbol, "buy", "close", 1, idea.ask) and desk.broker.sent[1][5].startswith(f"sc-{made.id}-c")
    assert (done.state, done.closing, done.exit_price) == ("closed", None, idea.ask)
    assert done.exits[0].rules == ("profit_target",) and done.stands == "at the profit target"
    assert done.pnl == pytest.approx((made.entry_price - idea.ask) * 100) and desk.journal.load()[0] == [done]
    with pytest.raises(PaperError, match="is closed"):
        desk.write_close(made.id, idea.ask)
    with pytest.raises(PaperError, match="no trade with that id") as caught:
        desk.write_close("nope", idea.ask)
    assert caught.value.status == 404


def test_one_closing_order_at_a_time_and_one_that_is_turned_down_leaves_the_trade_open(desk, scan):
    idea = scan.ideas[0]
    made = desk.send(desk.write_open(scan, idea.symbol, 1, idea.bid).id)
    first, second = desk.write_close(made.id, 0.05), desk.write_close(made.id, 0.06)

    desk.broker.turns_down = "position intent mismatch"
    with pytest.raises(PaperError, match="position intent mismatch"):
        desk.send(first.id)
    ((t,), _) = desk.journal.load()
    assert (t.state, t.closing) == ("open", None)

    desk.broker.turns_down, desk.broker.fills = None, False
    working = desk.send(second.id)
    assert working.state == "open" and working.closing.order == "order-3" and working.closing.limit == 0.06
    with pytest.raises(PaperError, match="already working"):
        desk.write_close(made.id, 0.07)
    review = desk.review()
    assert [o.id for o in review.orders] == ["order-3"] and review.trades[0].stands == "open, closing order working"

    desk.cancel("order-3")
    after = desk.review()
    assert after.orders == [] and (after.trades[0].state, after.trades[0].closing) == ("open", None)
    desk.write_close(made.id, 0.07)  # and now it can be written out again


def test_an_order_written_for_a_trade_that_has_since_changed_is_not_sent(desk, scan):
    idea = scan.ideas[0]
    made = desk.send(desk.write_open(scan, idea.symbol, 1, idea.bid).id)
    stale, used = desk.write_close(made.id, idea.ask), desk.write_close(made.id, idea.ask)
    desk.send(used.id)
    with pytest.raises(PaperError, match="has changed") as caught:
        desk.send(stale.id)
    assert caught.value.status == 409 and len(desk.broker.sent) == 2


@pytest.mark.parametrize("quantity, limit", [(0, 1.0), (101, 1.0), (1.5, 1.0), (True, 1.0), (1, 0.0), (1, 0.005), (1, 1.005), (1, 10_000.01), (1, float("inf")), (1, "1.00")])
def test_what_is_not_an_order_is_not_written_out(desk, scan, quantity, limit):
    with pytest.raises(PaperError):
        desk.write_open(scan, scan.ideas[0].symbol, quantity, limit)
    with pytest.raises(PaperError, match="not in the scan"):
        desk.write_open(scan, "CRUX270108P00001000", 1, 1.0)


def test_a_call_sold_to_open_says_it_needs_the_shares(scanner, tmp_path):
    desk = PaperDesk(Journal(None), scanner, DemoBroker(scanner.market))
    calls = scanner.scan(ScanRequest(symbols=["CRUX"], lens="sell", kind="call", dte_min=25, dte_max=45))
    ticket = desk.write_open(calls, calls.ideas[0].symbol, 1, calls.ideas[0].bid)
    assert "100 shares of CRUX" in ticket.notes[0] and "does not buy shares" in ticket.notes[0]
    with pytest.raises(PaperError, match="covered call"):
        desk.send(ticket.id)  # as Alpaca does, and nothing is left in the journal
    assert desk.journal.load() == ([], 0)


def test_looking_never_sends_or_cancels_an_order(desk, scan):
    idea = scan.ideas[0]
    desk.send(desk.write_open(scan, idea.symbol, 1, round(idea.ask * 3, 2)).id)
    desk.broker.cancel = lambda order_id: pytest.fail("a review cancelled an order")
    for _ in range(3):
        review = desk.review()
    (holding,) = review.positions.holdings
    # Its profit target is reached, and that is a fact on the page: nothing acts on it.
    assert {c.rule: c.state for c in holding.checks}["profit_target"] == "reached"
    assert len(desk.broker.sent) == 1 and review.trades[0].state == "open"


def test_a_review_sets_what_the_account_holds_against_the_exit_rules(desk, scan):
    idea = scan.ideas[0]
    made = desk.send(desk.write_open(scan, idea.symbol, 2, idea.bid).id)
    desk.broker.holds[("CRUX270108C00999000", "long")] = 1  # bought at Alpaca, not here
    review = desk.review()
    assert (review.mode, review.can_order, review.account) == ("demo", True, PaperAccount(3, 50_000.0))
    mine, other = review.positions.holdings
    assert (mine.position.id, mine.position.symbol, mine.position.side, mine.position.quantity) == (f"{idea.symbol}:short", idea.symbol, "short", 2)
    assert mine.position.opened == TODAY and mine.close_price == idea.ask and mine.plan is not None
    assert review.opened_by == {f"{idea.symbol}:short": [made.id], "CRUX270108C00999000:long": []}
    assert other.note and other.position.underlying == "CRUX"  # listed all the same, with no quote to price it
    assert "CRUX" in review.positions.underlyings and review.positions.warnings == []
    json.dumps(asdict(review), default=str, allow_nan=False)  # what a review hands out goes through a strict encoder


def test_a_share_class_goes_by_the_stocks_own_spelling(desk):
    # An option's symbol has no dot in it, so the stock's spelling comes from the journal.
    t = held(underlying="BRK.B")
    assert desk._position(t.symbol, "short", 1, 2.0, [t]).underlying == "BRK.B"
    assert desk._position(t.symbol, "short", 1, 2.0, []).underlying == "BRKB"


def test_an_expired_paper_position_is_not_said_to_need_removing(desk):
    gone = TODAY - timedelta(days=4)
    desk.broker.holds[(held(underlying="CRUX", expiry=gone, ordered=NOW - timedelta(days=40)).symbol, "short")] = 1
    (holding,) = desk.review().positions.holdings
    assert holding.dte == -4 and "Alpaca takes it off" in holding.checks[0].detail and "Remove it" not in holding.checks[0].detail


def test_the_stocks_close_on_a_day_comes_from_its_price_history(desk):
    friday = TODAY - timedelta(days=4)
    bars = {bar.day: bar.close for bar in desk.scanner.market.daily_bars(["CRUX"], friday)["CRUX"]}
    assert desk._close_on("CRUX", friday) == bars[friday]
    assert desk._close_on("CRUX", TODAY) is None  # today's bar is not final
    assert desk._close_on("NOPE", friday) is None


def test_what_cannot_be_read_is_said_and_concludes_nothing(desk, scan):
    made = desk.send(desk.write_open(scan, scan.ideas[0].symbol, 1, scan.ideas[0].bid).id)
    desk.broker.readable = False
    review = desk.review()
    assert review.positions is None and "positions could not be read" in review.warnings[0]
    assert [(t.id, t.state) for t in review.trades] == [(made.id, "open")]  # not read as gone


def test_with_the_keys_of_a_live_account_there_is_no_broker_at_all(scanner, tmp_path, scan):
    live = Settings(alpaca_key="k", alpaca_secret="s", alpaca_paper=False, journal_file=tmp_path / "journal.json")
    desk = paper.build_desk(live, scanner)
    assert desk.broker is None and not desk.available and "live account" in desk.why
    for use in (lambda: desk.write_open(scan, scan.ideas[0].symbol, 1, 1.0), lambda: desk.write_close("x", 1.0), lambda: desk.send("x"), lambda: desk.cancel("x")):
        with pytest.raises(PaperError, match="paper account only") as caught:
            use()
        assert caught.value.status == 503
    Journal(live.journal_file).add(held())
    review = desk.review()  # the journal is still there to read
    assert (review.can_order, review.positions, len(review.trades)) == (False, None, 1) and review.warnings == [paper.LIVE_KEYS]

    assert isinstance(paper.build_desk(replace(live, alpaca_paper=True), scanner).broker, AlpacaPaper)
    with_no_keys = paper.build_desk(live, None, setup_error="Alpaca keys are not set.")
    assert (with_no_keys.mode, with_no_keys.broker) == ("unconfigured", None)
    assert with_no_keys.review().warnings == ["Alpaca keys are not set."] and len(with_no_keys.review().trades) == 1


def test_the_demo_starts_with_a_made_up_account_and_journal_in_memory(scanner, tmp_path):
    settings = Settings(journal_file=tmp_path / "journal.json")
    desk = paper.build_desk(settings, scanner, demo=True)
    review = desk.review()
    assert isinstance(desk.broker, DemoBroker) and desk.journal.path is None and not settings.journal_file.exists()
    assert {h.position.side for h in review.positions.holdings} == {"long", "short"}
    assert sorted(review.opened_by.values()) == [["demob1"], ["demos1"]]
    assert set(review.summary) == {"buy", "sell"} and all(rows[0].trades >= signals.MIN_TRADES for rows in review.summary.values())
    assert any(row.trades < signals.MIN_TRADES for rows in review.summary.values() for row in rows)  # and some rows are too few
    assert len(desk.review().trades) == len(review.trades)  # made once, not on every look


# -- The API and the page ----------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    settings = Settings(positions_file=tmp_path / "positions.json", journal_file=tmp_path / "journal.json")
    return TestClient(create_app(settings, demo=True))


def scanned(client, **body) -> tuple[dict, dict]:
    result = client.post("/api/scan", json={"symbols": ["CRUX"], "lens": "sell", "kind": "put", "dte_min": 25, "dte_max": 45, **body}).json()
    return result, result["ideas"][0]


def test_a_paper_trade_from_the_scan_to_the_journal_through_the_api(client, tmp_path):
    status = client.get("/api/status").json()["paper"]
    assert status["available"] and status["file"] is None and status["why"] is None
    assert status["min_trades"] == 10 and status["signals"]["rating"] == "Rating if history repeats"
    before = client.get("/api/paper").json()

    result, idea = scanned(client)
    written = client.post("/api/paper/tickets", json={"action": "open", "scan_id": result["id"], "contract": idea["symbol"], "quantity": 1, "limit": idea["bid"]})
    assert written.status_code == 200 and written.json()["text"].startswith("Sell to open 1 CRUX")
    assert len(client.get("/api/paper").json()["trades"]) == len(before["trades"])  # nothing has happened yet

    sent = client.post("/api/paper/orders", json={"ticket": written.json()["id"]})
    made = sent.json()
    assert sent.status_code == 201 and (made["state"], made["entry_price"], made["signals"]["score"]) == ("open", idea["bid"], idea["score"])
    assert client.post("/api/paper/orders", json={"ticket": written.json()["id"]}).status_code == 409

    now = client.get("/api/paper").json()
    assert now["trades"][0]["id"] == made["id"] and made["id"] in now["opened_by"][idea["symbol"] + ":short"]
    holding = next(h for h in now["positions"]["holdings"] if h["position"]["symbol"] == idea["symbol"])
    assert holding["close_price"] == idea["ask"] and holding["checks"]

    closing = client.post("/api/paper/tickets", json={"action": "close", "trade": made["id"], "limit": idea["ask"]}).json()
    done = client.post("/api/paper/orders", json={"ticket": closing["id"]}).json()
    assert (done["state"], done["exit_price"], done["stands"]) == ("closed", idea["ask"], "before either rule was reached")
    assert done["pnl"] == pytest.approx((idea["bid"] - idea["ask"]) * 100)  # the spread, paid on the way out
    assert not (tmp_path / "journal.json").exists()  # the demo's journal never reaches the disk


def test_an_order_that_waits_can_be_cancelled_through_the_api(client):
    result, idea = scanned(client)
    body = {"action": "open", "scan_id": result["id"], "contract": idea["symbol"], "quantity": 1, "limit": round(idea["ask"] * 3, 2)}
    made = client.post("/api/paper/orders", json={"ticket": client.post("/api/paper/tickets", json=body).json()["id"]}).json()
    assert made["state"] == "ordered" and made["stands"] == "ordered, not filled yet"
    (waiting,) = client.get("/api/paper").json()["orders"]
    assert (waiting["symbol"], waiting["intent"], waiting["working"]) == (idea["symbol"], "open", True)
    assert client.delete(f"/api/paper/orders/{waiting['id']}").json() == {"cancelled": waiting["id"]}
    assert client.delete(f"/api/paper/orders/{waiting['id']}").status_code == 422  # it has already ended
    assert client.delete("/api/paper/orders/nope").status_code == 404
    after = client.get("/api/paper").json()
    assert after["orders"] == [] and (after["trades"][0]["id"], after["trades"][0]["state"]) == (made["id"], "unfilled")


@pytest.mark.parametrize(
    "body",
    [
        {"action": "open", "limit": 1.0},  # no scan, no contract
        {"action": "open", "scan_id": "x", "contract": "CRUX261218P00395000", "limit": 1.0},  # no number of contracts
        {"action": "open", "scan_id": "x", "contract": "CRUX261218P00395000", "quantity": 0, "limit": 1.0},
        {"action": "open", "scan_id": "x", "contract": "CRUX261218P00395000", "quantity": 1, "limit": 0},
        {"action": "open", "scan_id": "x", "contract": "CRUX261218P00395000", "quantity": 1, "limit": 1e9},
        {"action": "close", "limit": 1.0},  # no trade
        {"action": "roll", "trade": "x", "limit": 1.0},
        {"action": "close", "trade": "x"},  # no limit: the server makes no price up
    ],
)
def test_the_api_turns_away_what_is_not_an_order(client, body):
    assert client.post("/api/paper/tickets", json=body).status_code == 422


def test_an_order_needs_a_scan_that_is_still_in_memory_and_a_contract_from_it(client):
    result, idea = scanned(client)
    body = {"action": "open", "scan_id": "gone", "contract": idea["symbol"], "quantity": 1, "limit": 1.0}
    assert client.post("/api/paper/tickets", json=body).status_code == 404
    assert client.post("/api/paper/tickets", json={**body, "scan_id": result["id"], "contract": "CRUX270108P00001000"}).status_code == 404
    assert client.post("/api/paper/tickets", json={**body, "scan_id": result["id"], "limit": 1.005}).status_code == 422
    assert client.post("/api/paper/tickets", json={"action": "close", "trade": "nope", "limit": 1.0}).status_code == 404


def test_a_page_on_another_site_cannot_write_an_order_out_or_send_one(client):
    # A form on another site can post text without asking the browser's leave, but not JSON.
    result, idea = scanned(client)
    body = {"action": "open", "scan_id": result["id"], "contract": idea["symbol"], "quantity": 1, "limit": idea["bid"]}
    as_text = {"Content-Type": "text/plain"}
    assert client.post("/api/paper/tickets", content=json.dumps(body), headers=as_text).status_code == 422
    ticket = client.post("/api/paper/tickets", json=body).json()["id"]
    before = len(client.get("/api/paper").json()["trades"])
    assert client.post("/api/paper/orders", content=json.dumps({"ticket": ticket}), headers=as_text).status_code == 422
    assert len(client.get("/api/paper").json()["trades"]) == before


def test_the_journal_can_be_read_with_no_keys_and_no_order_can_be_sent(tmp_path):
    path = tmp_path / "journal.json"
    Journal(path).add(held())
    client = TestClient(create_app(Settings(positions_file=tmp_path / "positions.json", journal_file=path), demo=False))
    status = client.get("/api/status").json()["paper"]
    assert not status["available"] and "ALPACA_API_KEY" in status["why"] and status["file"] == str(path)
    body = client.get("/api/paper").json()
    assert (body["mode"], body["can_order"], body["positions"], len(body["trades"])) == ("unconfigured", False, None, 1)
    assert client.post("/api/paper/tickets", json={"action": "close", "trade": body["trades"][0]["id"], "limit": 1.0}).status_code == 503

    path.write_text("{half a file")
    spoiled = client.get("/api/paper")
    assert spoiled.status_code == 500 and "does not read as a list of trades" in spoiled.json()["detail"]
    assert path.read_text() == "{half a file"


def test_the_page_guards_how_an_order_is_sent(client):
    # The page has no test runner, so these hold the order flow in place by its text.
    page = client.get("/").text
    assert 'id="tab-paper"' in page and 'data-view="paper"' in page
    assert page.count('api("/api/paper/orders", { ticket: ticket.id })') == 1  # the one place an order is sent from
    assert "send.disabled = back.disabled = true;" in page  # one order, however many clicks
    assert "Nothing has been sent." in page and "Write the order out…" in page
    assert "if (ticket !== paperLoads) return;" in page  # an answer that is not the newest load's is dropped
    assert "row.trades < rules.min_trades" in page and "too few to compare" in page
    assert "It cannot place orders" not in page and "It cannot place a real one." in page


# -- The command line ----------------------------------------------------------------------------------


def test_paper_trading_on_the_command_line(capsys):
    from app.__main__ import main

    assert main(["paper", "--demo"]) == 0
    out = capsys.readouterr().out
    assert "Paper trading   Data: SYNTHETIC DEMO DATA" in out and "options level 3" in out
    assert "demos1" in out and "profit target reached" in out and "Journal: " in out and "2 open" in out
    assert "Which signals worked" in out and "Sold to open" in out and "too few to compare" in out
    assert "Every closed trade" in out and "It is not a forecast" in out and "nothing here sends or cancels one for you" in out
    assert "-0%" not in out and "+$0" not in out

    assert main(["paper", "--demo", "--json"]) == 0
    document = json.loads(capsys.readouterr().out, parse_constant=pytest.fail)  # strict JSON: no NaN, no Infinity
    assert document["can_order"] and len(document["positions"]["holdings"]) == 2 and document["summary"]["sell"][0]["group"] == "all"


def test_an_order_on_the_command_line_is_only_written_out_until_send_is_said(capsys):
    from app.__main__ import main

    quoted = build_scanner(Settings(), demo=True).scan(ScanRequest(symbols=["CRUX"], lens="sell", kind="put", dte_min=25, dte_max=45)).ideas[0]
    order_for = ["paper", "--demo", "open", "crux", quoted.expiry.isoformat(), f"{quoted.strike:g}", "put", "--sell"]
    assert main(order_for) == 0
    out = capsys.readouterr().out
    assert f"Sell to open 1 CRUX {quoted.expiry:%b %d} {quoted.strike:g} put at ${quoted.bid:.2f} a share or more" in out
    assert "Nothing was sent." in out and "Sent" not in out

    assert main(order_for + ["--qty", "2", "--send"]) == 0
    assert f"Sent, and filled: 2 at {quoted.bid:.2f}." in capsys.readouterr().out
    assert main(order_for + ["--limit", "99", "--send"]) == 0
    assert "waiting to fill" in capsys.readouterr().out

    assert main(["paper", "--demo", "close", "demos1"]) == 0
    assert "Buy to close 1" in (out := capsys.readouterr().out) and "Nothing was sent." in out
    assert main(["paper", "--demo", "close", "demos1", "--send"]) == 0
    assert "Trade demos1 is closed: +$" in capsys.readouterr().out

    for refused, said in (
        (["paper", "--demo", "open", "CRUX", quoted.expiry.isoformat(), "1", "put", "--sell"], "does not rank"),
        (["paper", "--demo", "open", "CRUX", "someday", "395", "put", "--sell"], "expiry must be a date"),
        (["paper", "--demo", "open", "CRUX", "2020-01-17", "395", "put", "--sell"], "has expired"),
        (["paper", "--demo", "open", "CRUX", quoted.expiry.isoformat(), f"{quoted.strike:g}", "call", "--sell", "--send"], "covered call"),
        (order_for + ["--limit", "1.005"], "whole cents"),
        (["paper", "--demo", "close", "nope", "--limit", "1"], "no trade with that id"),
        (["paper", "--demo", "cancel", "nope"], "no order with that id"),
    ):
        assert main(refused) == 2
        assert said in capsys.readouterr().err
    with pytest.raises(SystemExit):  # whether it is bought or sold has to be said
        main(["paper", "--demo", "open", "CRUX", quoted.expiry.isoformat(), "395", "put"])


def test_a_quote_becomes_a_limit_in_whole_cents():
    from app.__main__ import _cents

    # Up to buy and down to sell, and a price that is already whole cents is left alone (3.10 * 100 is 310.00000000000006).
    assert (_cents(3.10, up=True), _cents(3.10, up=False)) == (3.10, 3.10)
    assert (_cents(8.109, up=True), _cents(8.109, up=False)) == (8.11, 8.10)
    assert _cents(0.004, up=False) == 0.01  # never below a cent


# -- From the outside review of phase 4 (2026-11-26) -----------------------------------------------------
#
# Every point was reproduced on main before anything was changed. Each test below failed there.


def test_a_trade_found_open_is_not_looked_for_among_positions_read_before_it_filled():
    # The positions are read before the orders. An opening order that fills between the two reads is in the
    # list as filled, with no position yet to show for it, and the new trade was closed on the spot as gone.
    t = trade()
    assert reconcile([t], [order(t)], [], TODAY, NOW) == [t]
    assert (t.state, t.exits, t.stands) == ("open", [], "open")  # and where it stands is worded for the state it is in now
    assert reconcile([t], [order(t)], [PaperHolding(t.symbol, "short", 1, 2.0)], TODAY, NOW) == []
    # Only a trade found open in that same look is spared. Missing at the next one, it is accounted for as ever.
    reconcile([t], [order(t)], [], TODAY, NOW)
    assert (t.state, t.exits[0].how) == ("closed", "unknown")


def test_a_review_reads_the_positions_before_the_orders_and_an_order_that_fills_in_between_costs_nothing(desk, scan):
    idea = scan.ideas[0]
    desk.broker.fills = False
    made = desk.send(desk.write_open(scan, idea.symbol, 1, idea.bid).id)
    desk.broker.read.clear()
    # The positions were read, and showed nothing. Then the order filled, and the list of orders says so.
    (waiting,) = desk.broker.listed.values()
    desk.broker.listed[waiting.id] = replace(waiting, status="filled", working=False, filled=1, fill_price=idea.bid, filled_at=NOW)
    first = desk.review()
    assert desk.broker.read == ["positions", "orders"]  # so whatever the positions show has already reached the orders
    assert [(t.id, t.state, t.exits) for t in first.trades] == [(made.id, "open", [])]
    desk.broker.holds[(idea.symbol, "short")] = 1
    assert desk.review().opened_by == {paper.held_id(idea.symbol, "short"): [made.id]}


def test_a_rule_that_cannot_be_read_is_not_on_record_and_never_reads_as_not_reached(desk, scan, monkeypatch):
    idea = scan.ideas[0]
    made = desk.send(desk.write_open(scan, idea.symbol, 1, round(idea.ask * 3, 2)).id)  # the profit target is reached
    ticket = desk.write_close(made.id, idea.ask)

    def down(wanted):
        raise RuntimeError("quotes are down")

    monkeypatch.setattr(desk.scanner.market, "option_quotes", down)
    done = desk.send(ticket.id)
    # With no ask the profit target is "unknown". Left out of the list it read as not reached, and a trade that
    # kept two thirds of its premium was filed as closed "before either rule was reached".
    assert done.exits[0].rules is None and done.stands == "rules not on record"
    assert len(desk.broker.sent) == 2 and done.state == "closed"  # the order went all the same: it is the person's


def test_the_rules_are_read_without_the_news_while_an_order_waits_to_leave(desk, scan, monkeypatch):
    idea = scan.ideas[0]
    made = desk.send(desk.write_open(scan, idea.symbol, 1, idea.bid).id)
    ticket = desk.write_close(made.id, idea.ask)
    with monkeypatch.context() as patch:
        patch.setattr(desk.scanner, "_add_news", lambda *a, **k: pytest.fail("news was loaded while an order waited to leave"))
        done = desk.send(ticket.id)
    assert done.exits[0].rules == () and done.stands == "before either rule was reached"  # read, and none reached
    held_position = desk._position(idea.symbol, "short", 1, idea.bid, [])
    assert desk.scanner.review([held_position], news=False).underlyings["CRUX"].news == []
    assert desk.scanner.review([held_position]).underlyings["CRUX"].news  # a review for the page still has it


def many_orders(total: int, stamped: str = "submitted_at") -> tuple[AlpacaPaper, list[httpx.Request], datetime]:
    """An account with that many orders a second apart, and one more that is still working."""
    start = datetime(2026, 11, 1, tzinfo=timezone.utc)
    rows = [{**ROW, "id": f"id-{n}", "client_order_id": f"x-{n}", "status": "filled", "submitted_at": None,
             stamped: (start + timedelta(seconds=n + 1)).strftime("%Y-%m-%dT%H:%M:%SZ")} for n in range(total)]  # fmt: skip

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params["status"] == "open":
            return httpx.Response(200, json=[{**ROW, "id": "working"}])
        after = datetime.fromisoformat(request.url.params["after"].replace("Z", "+00:00"))
        return httpx.Response(200, json=[row for row in rows if datetime.fromisoformat(row[stamped].replace("Z", "+00:00")) > after][:500])

    account, seen = account_on(handler)
    return account, seen, start


def test_a_list_of_orders_that_was_cut_short_says_so_and_still_has_the_ones_working():
    # Read oldest first, four pages of 500. Past that the newest were dropped and the list looked whole.
    account, seen, start = many_orders(2600)
    found, whole = account.orders(start)
    assert not whole and len(found) == 2001 and found[-1].id == "working" and found[-1].working
    assert [request.url.params["status"] for request in seen] == ["all"] * 4 + ["open"]

    account, seen, start = many_orders(1200)
    found, whole = account.orders(start)
    assert whole and len(found) == 1200 and len(seen) == 3  # the last page came back short, so that is all of them

    # A row with no time it was submitted goes by the time it was created, as the order read from it does.
    account, seen, start = many_orders(1200, stamped="created_at")
    found, whole = account.orders(start)
    assert whole and len(found) == 1200

    # A full page with no time on it at all leaves nowhere to go on from: there is more, and that is said.
    account, seen = account_on(lambda request: httpx.Response(200, json=[{**ROW, "id": f"id-{n}", "submitted_at": None} for n in range(500)]))
    assert account.orders(start)[1] is False


def test_with_part_of_the_orders_unread_nothing_is_concluded_about_contracts_that_have_gone(desk, scan):
    made = desk.send(desk.write_open(scan, scan.ideas[0].symbol, 1, scan.ideas[0].bid).id)
    desk.broker.holds.clear()  # closed at Alpaca, by an order in the part of the list that was not read
    desk.broker.whole = False
    review = desk.review()
    assert paper.CUT_SHORT in review.warnings and [(t.id, t.state) for t in review.trades] == [(made.id, "open")]
    desk.broker.whole = True
    after = desk.review()
    assert paper.CUT_SHORT not in after.warnings and after.trades[0].stands == journal.HOW["unknown"]


def test_an_answer_for_an_order_that_does_not_read_is_not_the_order_never_arriving():
    # The account answered 200: it has an order by that name. Read as "no such order", the trade was marked
    # as never having arrived, with the order still out there.
    account, _ = account_on(lambda request: httpx.Response(200, json={**ROW, "status": ""}))
    with pytest.raises(BrokerError, match="does not read as one"):
        account.order("sc-ab12cd34-o")
    lost = trade()
    assert reconcile([lost], [], None, TODAY, NOW, lookup=account.order) == [] and lost.state == "ordered"


def test_each_side_of_a_contract_is_a_row_of_its_own(desk):
    # An account holds a contract one way or the other. A stand-in that says both must still not have one
    # row's journal entries shown under the other.
    short, long = held(underlying="CRUX", id="short1"), held(underlying="CRUX", side="long", id="long1")
    for t in (short, long):
        desk.journal.add(t)
        desk.broker.holds[(t.symbol, t.side)] = 1
    review = desk.review()
    assert review.opened_by == {f"{short.symbol}:short": ["short1"], f"{short.symbol}:long": ["long1"]}
    assert sorted(h.position.id for h in review.positions.holdings) == sorted(review.opened_by)


def test_the_other_side_of_a_contract_that_is_held_is_not_opened_from_here(desk, scanner):
    wide = dict(symbols=["CRUX"], kind="put", dte_min=25, dte_max=45, per_side=10_000, top_n=10_000)
    sells, buys = scanner.scan(ScanRequest(lens="sell", **wide)), scanner.scan(ScanRequest(lens="buy", **wide))
    both = sorted({c.symbol for c in sells.ideas} & {c.symbol for c in buys.ideas})
    sold = next(c for c in sells.ideas if c.symbol == both[0])
    made = desk.send(desk.write_open(sells, sold.symbol, 1, sold.bid).id)
    # Bought to open, it would be set against the trade that is there, and the journal could not tell which had ended.
    with pytest.raises(PaperError, match="one way or the other"):
        desk.write_open(buys, sold.symbol, 1, sold.ask)
    desk.write_open(sells, sold.symbol, 1, sold.bid)  # more of the same side is another trade on the contract
    desk.send(desk.write_close(made.id, sold.ask).id)
    desk.write_open(buys, sold.symbol, 1, sold.ask)  # and once that trade has ended, the other side can be opened


def test_the_demos_clock_is_on_the_demos_day_whatever_day_it_really_is(scanner, scan, tmp_path, monkeypatch):
    # The demo's market is held at TODAY for the tests, and its contracts expire within weeks of that. With the
    # desk on the wall clock, every test that sends a demo order was due to fail once those weeks had passed.
    class Later(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2027, 6, 1, 15, 0, tzinfo=timezone.utc).astimezone(tz)

    monkeypatch.setattr(paper, "datetime", Later)
    monkeypatch.setattr(broker, "datetime", Later)
    desk = paper.build_desk(Settings(journal_file=tmp_path / "journal.json"), scanner, demo=True)
    made = desk.send(desk.write_open(scan, scan.ideas[0].symbol, 1, scan.ideas[0].bid).id)
    assert made.state == "open" and made.ordered.astimezone(ET).date() == made.opened.astimezone(ET).date() == TODAY
    assert desk.review().generated_at.astimezone(ET).date() == TODAY


def test_a_price_on_file_has_a_positions_bounds_and_a_figure_that_is_not_a_number_does_not_read(tmp_path):
    done = held(id="done")
    done.exits.append(Exit(1, 0.5, NOW + timedelta(days=3), "order", ()))
    plain = journal._plain(journal.settle(done))
    worthless = {**plain, "id": "worthless", "exits": [{**plain["exits"][0], "price": 0.0}]}
    rows = [
        {**plain, "id": "tiny", "entry_price": 1e-310},  # every result against it was infinite
        {**plain, "id": "nan", "signals": {**SHOWN, "score": float("nan")}},  # Python reads NaN from a file
        {**plain, "id": "closing", "state": "open", "exits": [], "closing": {"client_id": "sc-x-c1", "quantity": 1, "limit": 0, "sent": NOW.isoformat()}},
        worthless,  # an exit at nothing is a price: an option can expire worth nothing
        plain,
    ]
    path = tmp_path / "journal.json"
    path.write_text(json.dumps({"trades": rows}))
    found, unread = Journal(path).load()
    assert [t.id for t in found] == ["worthless", "done"] and unread == 3
    assert (found[0].pnl, found[0].pnl_pct) == (200.0, 1.0)
    json.dumps([asdict(t) for t in found], default=str, allow_nan=False)

    # And nothing this app writes holds such a figure. The entry is refused before the file is touched.
    other = Journal(tmp_path / "other.json")
    with pytest.raises(ValueError):
        other.add(held(signals={**SHOWN, "pop": float("inf")}))
    assert not other.path.exists()


def test_an_order_that_filled_with_no_price_on_it_is_waited_for():
    # It was read as "ended filled with nothing filled", with the contracts in the account. And a closing
    # order that filled so was dropped, and its contracts written off as gone with no order on record.
    t = trade()
    assert reconcile([t], [order(t, fill_price=None)], [], TODAY, NOW) == [t]
    assert (t.state, t.entry_price) == ("ordered", None) and t.note.startswith(paper.NO_PRICE)
    assert reconcile([t], [order(t, fill_price=None)], [], TODAY, NOW) == []  # said once
    # A fill at less than a position can cost is no price either. Rounded, it was an entry price of nothing,
    # and the journal could not read its own entry back.
    assert reconcile([t], [order(t, fill_price=0.00004)], [], TODAY, NOW) == [] and t.state == "ordered"
    reconcile([t], [order(t)], None, TODAY, NOW)
    assert (t.state, t.entry_price, t.note) == ("open", 2.0, None)

    c = held()
    c.closing = ClosingOrder("sc-x-c1", 1, 0.95, NOW, ())
    reconcile([c], [order(c, "close", fill_price=None)], [], TODAY, NOW)
    assert (c.state, c.exits) == ("open", []) and c.closing and c.note.startswith(paper.NO_PRICE)
    reconcile([c], [order(c, "close", fill_price=0.9)], [], TODAY, NOW)
    assert (c.state, c.pnl, c.note) == ("closed", 110.0, None)


def test_a_reading_typed_into_the_journal_by_hand_never_breaks_the_summary():
    read = lambda name, **shown: signals.GROUPS[name][1](closed(1.0, **shown))  # noqa: E731
    assert read("rating", rating=["favorable"]) == "not rated" and read("rating", rating=3) == "not rated"
    assert read("score", score=float("nan")) is None and read("pop", pop=float("inf")) is None  # NaN fell in the lowest band
    odd = [closed(1.0, rating=["favorable"]), closed(1.0, score=float("nan")), closed(1.0, lens=["sell"]), closed(1.0)]
    found = signals.summary(odd)  # a list is no key for a table, and this raised TypeError
    assert found["sell"][0].trades == 3 and not signals.counted(odd[2])  # with no lens to go by it is counted nowhere
    assert {row.bucket: row.trades for row in found["sell"] if row.group == "rating"} == {"favorable": 2, "not rated": 1}


def test_an_outside_order_from_before_the_trade_is_not_its_exit_though_it_has_no_fill_time():
    # An order that fills in part and is then cancelled has no time it filled. Taken as "now", it passed for
    # an order from after the trade opened.
    t = held()
    old = order(t, "close", status="canceled", filled=1, fill_price=0.01, id="older", filled_at=None, submitted=t.opened - timedelta(days=1))
    reconcile([t], [old], [], TODAY, NOW)
    assert [(e.how, e.price) for e in t.exits] == [("unknown", None)] and t.pnl is None
    # One sent after the trade opened is its exit, at the time it was sent.
    since = held()
    part = order(since, "close", status="canceled", filled=1, fill_price=1.25, id="later", filled_at=None, submitted=since.opened + timedelta(hours=1))
    reconcile([since], [part], [], TODAY, NOW)
    assert since.exits == [Exit(1, 1.25, since.opened + timedelta(hours=1), "outside", None, "later")]


def test_an_outside_order_that_opens_is_not_taken_as_one_that_closes():
    # A put sold here left the account, and the same contract was then bought to open at Alpaca.
    t = held()
    opens = replace(order(t, "close", id="bought-to-open", fill_price=1.25), intent="open")
    reconcile([t], [opens], [], TODAY, NOW)
    assert [(e.how, e.order) for e in t.exits] == [("unknown", "")]
    # When the account does not say which it was, the side is all there is to go by.
    unsaid = held()
    reconcile([unsaid], [replace(order(unsaid, "close", id="typed-in", fill_price=1.25), intent="")], [], TODAY, NOW)
    assert [(e.how, e.order) for e in unsaid.exits] == [("outside", "typed-in")]


def test_an_outside_order_taken_in_part_is_there_for_the_rest():
    a = held(quantity=2, ordered=NOW - timedelta(days=5), id="aaaa")
    b = trade(quantity=2, ordered=NOW - timedelta(days=1), id="bbbb")
    outside = order(a, "close", filled=2, fill_price=1.25, id="typed-in")
    one_held = [PaperHolding(a.symbol, "short", 1, 2.0)]
    # B's opening order has filled one so far, so the account holds one where A's two used to be.
    reconcile([a, b], [outside, order(b, status="partially_filled", filled=1, fill_price=2.0)], one_held, TODAY, NOW)
    assert [(e.quantity, e.how) for e in a.exits] == [(1, "outside")] and a.state == "open"
    # B's order ends with that one filled. A's other contract is the rest of the same outside order, where it
    # used to be written off with no price because the order had "been used".
    reconcile([a, b], [outside, order(b, status="canceled", filled=1, fill_price=2.0)], one_held, TODAY, NOW)
    reconcile([a, b], [outside, order(b, status="canceled", filled=1, fill_price=2.0)], one_held, TODAY, NOW)
    assert [(e.quantity, e.how, e.price) for e in a.exits] == [(1, "outside", 1.25), (1, "outside", 1.25)]
    assert (a.state, a.pnl, b.state, b.quantity) == ("closed", 150.0, "open", 1)


def test_no_order_is_written_or_sent_for_a_contract_that_has_expired(desk, scan):
    # The account lists an expired contract until it has settled, and an order to close it was written and sent.
    gone = held(underlying="CRUX", expiry=TODAY - timedelta(days=4), ordered=NOW - timedelta(days=40), id="gone")
    desk.journal.add(gone)
    desk.broker.holds[(gone.symbol, "short")] = 1
    with pytest.raises(PaperError, match="has expired"):
        desk.write_close("gone", 0.05)

    # Written out late on a contract's last day and confirmed after midnight in New York: the order is no
    # longer one to send. It used to fail inside the journal, which the server answered with a bare 500.
    ticket = desk.write_open(scan, scan.ideas[0].symbol, 1, scan.ideas[0].bid)
    desk.now = datetime(2026, 11, 25, 5, 1, tzinfo=timezone.utc)  # 12:01 a.m. on the 25th
    desk._tickets[ticket.id] = replace(ticket, expiry=TODAY, made=desk.now - timedelta(minutes=5))
    with pytest.raises(PaperError, match="expired since") as caught:
        desk.send(ticket.id)
    assert caught.value.status == 409 and desk.broker.sent == [] and [t.id for t in desk.journal.load()[0]] == ["gone"]


def test_the_stocks_close_is_read_in_turn_with_scans(desk, monkeypatch):
    # The provider and its notes are shared, and this one read went round the lock that keeps scans apart.
    friday, real, locked = TODAY - timedelta(days=4), desk.scanner.market.daily_bars, []

    def bars(symbols, start):
        locked.append(desk.scanner._lock.locked())
        desk.scanner.market.notes.append("a note with no reader")
        return real(symbols, start)

    monkeypatch.setattr(desk.scanner.market, "daily_bars", bars)
    assert desk._close_on("CRUX", friday) is not None
    assert locked == [True] and desk.scanner.market.notes == [] and not desk.scanner._lock.locked()


def test_what_closing_at_the_limit_would_give_is_worded_after_rounding(desk):
    # $2.004 against $2.00 is 40 cents on a contract: it ended "up $0", or "down $0".
    for id, side in (("sold", "short"), ("bought", "long"), ("gain", "short")):
        desk.journal.add(held(underlying="CRUX", side=side, limit=2.004 if id != "gain" else 3.0, id=id))
    said = {id: desk.write_close(id, 2.0).notes[0] for id in ("sold", "bought", "gain")}
    assert said["sold"].endswith("the trade ends even.") and said["bought"].endswith("the trade ends even.")
    assert said["gain"].endswith("the trade ends up $100.")


def test_the_page_keeps_what_became_of_an_order_in_sight(client):
    page = client.get("/").text
    # A closing order's answer has a line of its own, since reading the account afresh redraws the pane it was sent from.
    assert 'id="paper-said" role="status"' in page and "(sent) => { say(sentText(sent, false)); loadPaper(); }" in page
    assert 'if (err.status !== 422) { say("The closing order: " + err.message, true); loadPaper(); }' in page
    # A send from the scan that failed may have left an entry in the journal, so the paper view is read afresh.
    assert "() => { paper = null; }));" in page and "failure.status = response.status;" in page
    assert page.count("whichTrade(trade, mine)") == 3  # two trades on one contract are told apart, twice, and the function itself
    assert 'day: "numeric", timeZone: "America/New_York" }) : "–";' in page  # a day in the journal is New York's
    assert "} else if (holding.dte < 0) {" in page and "This contract has expired, so no order can close it." in page
    assert 'problem.textContent = "Sending…";' in page


def test_what_is_typed_for_an_order_is_checked_before_anything_is_scanned(capsys, monkeypatch):
    from app.__main__ import main
    from app.scanner import Scanner

    expiry = (TODAY + timedelta(days=24)).isoformat()
    assert main(["paper", "--demo", "close", "nope"]) == 2  # with no --limit it used to say there was no quote
    assert "no trade with that id" in capsys.readouterr().err

    monkeypatch.setattr(Scanner, "scan", lambda self, request: pytest.fail("a scan was run"))
    for typed, said in (
        (["   ", expiry, "395", "put", "--sell"], "not a ticker symbol"),  # blank, it was a scan of every stock
        (["CRUX", expiry, "inf", "put", "--sell"], "strike must be"),  # a traceback
        (["CRUX", expiry, "nan", "put", "--sell"], "strike must be"),
        (["CRUX", expiry, "-5", "put", "--sell"], "strike must be"),  # CRUX270101P-0005000
        (["CRUX", expiry, "395", "put", "--sell", "--qty", "0"], "whole number of contracts"),
        (["CRUX", expiry, "395", "put", "--sell", "--limit", "1.005"], "whole cents"),
    ):
        assert main(["paper", "--demo", "open", *typed]) == 2
        assert said in capsys.readouterr().err


def test_with_the_keys_of_a_live_account_the_command_line_says_so_before_it_scans(scanner, tmp_path, capsys, monkeypatch):
    import argparse

    from app.__main__ import _open_ticket, print_paper

    live = Settings(alpaca_key="k", alpaca_secret="s", alpaca_paper=False, journal_file=tmp_path / "journal.json")
    desk = paper.build_desk(live, scanner)
    typed = argparse.Namespace(underlying="CRUX", expiry=(TODAY + timedelta(days=24)).isoformat(), strike=395.0, kind="put", lens="sell", qty=1, limit=None)
    with monkeypatch.context() as patch:
        patch.setattr(scanner, "scan", lambda request: pytest.fail("a scan was run for an order that cannot be sent"))
        with pytest.raises(PaperError, match="paper account only"):
            _open_ticket(typed, desk, scanner)

    # And the account is said to be missing, not to have failed to read.
    print_paper(PaperReview(NOW, "live", False, PaperAccount(None, None), None, {}, [], [], {}, [paper.LIVE_KEYS]))
    out = capsys.readouterr().out
    assert "Data: no paper account" in out and "No paper account is set up" in out
    assert "could not be read" not in out and "Account:" not in out


@pytest.mark.parametrize(
    "written, read",
    [("true", True), ("1", True), ("YES", True), ("on", True), ("false", False), ("0", False), ("no", False), ("Off", False), ("", True), ("maybe", None)],
)
def test_whether_the_keys_are_a_paper_accounts_is_yes_or_no(monkeypatch, written, read):
    # Every word but "0", "false" and "no" used to count as yes, "off" among them.
    monkeypatch.setenv("ALPACA_PAPER", written)
    if read is None:
        with pytest.raises(ConfigError, match="ALPACA_PAPER must be true or false"):
            load_settings()
    else:
        assert load_settings().alpaca_paper is read


# -- From the second outside review of phase 4 (2026-11-27) ----------------------------------------------
#
# It read the fixes above and found five more points. Each was reproduced on `main` first, and each test here failed there.


def test_the_other_side_is_asked_about_again_when_an_order_is_sent(desk, scanner):
    wide = dict(symbols=["CRUX"], kind="put", dte_min=25, dte_max=45, per_side=10_000, top_n=10_000)
    sells, buys = scanner.scan(ScanRequest(lens="sell", **wide)), scanner.scan(ScanRequest(lens="buy", **wide))
    both = sorted({c.symbol for c in sells.ideas} & {c.symbol for c in buys.ideas})
    idea = next(c for c in sells.ideas if c.symbol == both[0])
    # Both orders are written out before either is sent: two tabs, or the command line while a ticket waits.
    # Asked only when an order was written out, both were sent, and the journal had the contract open both ways.
    sell, buy = desk.write_open(sells, idea.symbol, 1, idea.bid), desk.write_open(buys, idea.symbol, 1, idea.ask)
    again = desk.write_open(sells, idea.symbol, 1, idea.bid)
    made = desk.send(sell.id)
    with pytest.raises(PaperError, match="one way or the other.*It was not sent") as caught:
        desk.send(buy.id)
    assert caught.value.status == 409 and len(desk.broker.sent) == 1 and desk.journal.load() == ([made], 0)
    with pytest.raises(PaperError, match="no longer waiting"):
        desk.send(buy.id)  # and the order is used up, like any other that was not sent
    assert desk.send(again.id).side == "short"  # more of the same side is another trade on the contract, as before

    # The demo's account does not hold a contract both ways either.
    account = DemoBroker(scanner.market, now=lambda: NOW)
    account.submit(idea.symbol, "sell", "open", 1, 0.01, "sc-a-o")
    with pytest.raises(BrokerError, match="one way or the other") as turned_down:
        account.submit(idea.symbol, "buy", "open", 1, 10_000.0, "sc-b-o")
    assert turned_down.value.refused and "Alpaca" not in str(turned_down.value)  # what Alpaca says has not been seen
    assert [(h.side, h.quantity) for h in account.positions()] == [("short", 1)]
    account.submit(idea.symbol, "sell", "open", 1, 0.01, "sc-c-o")
    assert [(h.side, h.quantity) for h in account.positions()] == [("short", 2)]


def test_the_demos_clock_only_runs_forward(scanner, scan, tmp_path, monkeypatch):
    class Wall(datetime):
        at = datetime(2026, 12, 1, 23, 55, tzinfo=ET)

        @classmethod
        def now(cls, tz=None):
            return cls.at.astimezone(tz)

    monkeypatch.setattr(paper, "datetime", Wall)
    monkeypatch.setattr(broker, "datetime", Wall)
    first, second = scan.ideas[0], scan.ideas[1]
    desk = paper.build_desk(Settings(journal_file=tmp_path / "journal.json"), scanner, demo=True)
    started = desk._clock()
    assert (started.astimezone(ET).date(), started.astimezone(ET).time()) == (TODAY, Wall.at.time())  # the demo's day, at this time
    waiting = desk.write_open(scan, first.symbol, 1, first.bid)
    opened = desk.send(desk.write_open(scan, second.symbol, 1, second.bid).id)

    # Ten minutes on it is past midnight. Read as the demo's day at the time on the wall, the clock went back a day.
    Wall.at = datetime(2026, 12, 2, 0, 5, tzinfo=ET)
    assert desk._clock() - started == timedelta(minutes=10)
    shut = desk.send(desk.write_close(opened.id, second.ask + 5).id)
    assert shut.state == "closed" and shut.closed - shut.opened == timedelta(minutes=10)  # it was closed the day before it opened

    Wall.at = datetime(2026, 12, 2, 0, 15, tzinfo=ET)
    with pytest.raises(PaperError, match="no longer waiting"):
        desk.send(waiting.id)  # written out twenty minutes ago, and it was taken as written 23 hours 40 minutes from now

    # The hour that comes twice when the clocks go back: 1:30 a.m., and forty minutes later 1:10 a.m.
    Wall.at = datetime(2026, 11, 1, 5, 30, tzinfo=timezone.utc)
    fall = paper.build_desk(Settings(journal_file=tmp_path / "fall.json"), scanner, demo=True)
    before = fall._clock()
    Wall.at = datetime(2026, 11, 1, 6, 10, tzinfo=timezone.utc)
    assert fall._clock() - before == timedelta(minutes=40)


def test_a_fill_above_what_a_position_can_cost_is_no_price_either(tmp_path):
    # The floor was checked and the ceiling was not. The trade opened at $2,000,000 a share, the journal wrote
    # that down, and then could not read its own entry back.
    t = trade()
    reconcile([t], [order(t, fill_price=2_000_000.0)], [], TODAY, NOW)
    assert (t.state, t.entry_price) == ("ordered", None) and paper.unpriced(t)
    most = trade()
    reconcile([most], [order(most, fill_price=float(positions.MAX_PRICE))], None, TODAY, NOW)
    assert (most.state, most.entry_price) == ("open", positions.MAX_PRICE)
    # An order this app did not send, at such a price, is not an exit at it.
    gone = held()
    reconcile([gone], [order(gone, "close", fill_price=2_000_000.0, id="outside")], [], TODAY, NOW)
    assert [(e.how, e.price) for e in gone.exits] == [("unknown", None)]
    # Whatever the account says, the journal reads back what it was given.
    kept = Journal(tmp_path / "journal.json")
    for one in (t, most, gone):
        kept.add(one)
    assert kept.load() == ([t, most, gone], 0)


class Unpriced(Account):
    """A stand-in that can fill an order and give no price for the fill."""

    priced = True

    def submit(self, *sent) -> Order:
        made = super().submit(*sent)
        if not self.priced:
            made = self.listed[made.id] = replace(made, fill_price=None)
        return made


def test_an_order_filled_with_no_price_is_not_said_to_be_waiting_or_working(desk, scan, client):
    from app.__main__ import _sent

    account = desk.broker = Unpriced()
    stands = lambda t: next((found.state, found.stands, found.note) for found in desk.review().trades if found.id == t.id)  # noqa: E731

    account.priced = False
    made = desk.send(desk.write_open(scan, scan.ideas[0].symbol, 1, 4.10).id)
    assert made.state == "ordered" and paper.unpriced(made)
    said = _sent(made, "open")
    assert made.note in said and "waiting to fill" not in said  # it filled, and that was all the account said
    # Where it stands is worded beside that remark, and said "ordered, not filled yet".
    assert made.stands == "ordered, waiting for the price it filled at" and stands(made)[1] == made.stands
    account.listed["order-1"] = replace(account.listed["order-1"], fill_price=4.10)
    assert stands(made) == ("open", "open", None)  # the price came, and the words and the remark went with the wait

    account.priced = True
    opened = desk.send(desk.write_open(scan, scan.ideas[1].symbol, 1, 4.10).id)
    assert not paper.unpriced(opened) and "filled: 1 at 4.10" in _sent(opened, "open")
    account.priced = False
    shut = desk.send(desk.write_close(opened.id, 1.00).id)
    assert shut.state == "open" and shut.closing and paper.unpriced(shut)
    said = _sent(shut, "close")
    assert shut.note in said and "is working" not in said
    assert shut.stands == "open, waiting for the price its closing order filled at" and stands(shut)[1] == shut.stands
    account.listed["order-3"] = replace(account.listed["order-3"], fill_price=1.00)
    state, words, remark = stands(shut)
    assert (state, remark) == ("closed", None) and "waiting" not in words

    # An order that is working, or not filled yet, is still said to be.
    assert journal.settle(trade()).stands == "ordered, not filled yet"
    working = held()
    working.closing = ClosingOrder("sc-x-c1", 1, 0.95, NOW, ())
    assert journal.settle(working).stands == "open, closing order working"

    # The page words all of it from the same remark, and tells the remark by how Python begins it.
    page = client.get("/").text
    assert client.get("/api/status").json()["paper"]["no_price"] == paper.NO_PRICE == journal.NO_PRICE
    assert "const unpriced = (trade) => Boolean(trade.note && trade.note.startsWith(status.paper.no_price));" in page
    assert 'if (unpriced(trade)) return "Sent. " + trade.note;' in page
    # Under "Close it" it was said to be working, and to be in the list of orders, where it is not.
    assert '? "The order that opened it has ended, and it cannot be closed from here yet. " + trade.note' in page
    assert '? "The closing order at " + exact(trade.closing.limit) + " has ended. " + trade.note' in page


def test_cancelling_an_order_clears_what_was_said_about_the_last_one(client):
    # "Sent. The closing order is working." stayed above the lists after that very order was cancelled.
    page = client.get("/").text
    start = page.index("function renderOrders()")
    cancelling = page[start : page.index("\nfunction ", start + 1)]
    assert re.search(r'"DELETE"\);\s*(//[^\n]*\s*)?say\(""\);\s*await loadPaper\(\);', cancelling)
    # A cancel the account turns down means the order has filled or ended, so the list it was pressed in is
    # out of date. The reason was put beside the button and nothing was read afresh.
    turned_down = cancelling[cancelling.index("} catch (err) {") :]
    assert re.search(r"if \(err\.status === 404 \|\| err\.status === 422\) \{\s*(//[^\n]*\s*)*say\(err\.message, true\);\s*await loadPaper\(\);\s*return;", turned_down)
    assert "cancel.disabled = false;" in turned_down  # with no answer to go by, the button can be tried again
