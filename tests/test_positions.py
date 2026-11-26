"""Positions: what a person holds, kept on file and set against the exit rules.

The tests marked as coming from the outside review of phase 3 were each first reproduced
against the code as a failure.
"""

import json
import threading
from dataclasses import asdict
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import httpx
import pytest
from fastapi.testclient import TestClient

from app import positions, pricing
from app.analysis import contracts as metrics
from app.analysis import exits, holdings, risk
from app.cache import NullCache
from app.config import Settings, load_settings
from app.models import EarningsEvent, Holding, ImpliedMove, OptionQuote, Position, Reaction, UnderlyingContext
from app.positions import PositionStore, StoreError
from app.providers.alpaca import AlpacaData
from app.providers.demo import SPECS, DemoMarket
from app.providers.earnings import EarningsService
from app.scanner import ESTIMATE_MARGIN, Scanner, build_scanner
from app.server import create_app

TODAY = date(2026, 11, 24)  # a Tuesday
ET = ZoneInfo("America/New_York")
RATE = 0.04


def position(**changes) -> Position:
    """A put sold ten days ago for $2.00, 45 days from expiry, unless told otherwise."""
    base = dict(
        underlying="XYZ",
        kind="put",
        strike=95.0,
        expiry=TODAY + timedelta(days=45),
        side="short",
        quantity=1,
        entry_price=2.00,
        opened=TODAY - timedelta(days=10),
    )
    return positions.make(**{**base, **changes})


def quoted(p: Position, bid: float | None, ask: float | None) -> OptionQuote:
    return OptionQuote(p.symbol, p.underlying, p.kind, p.strike, p.expiry, bid=bid, ask=ask)


def context(**overrides) -> UnderlyingContext:
    base = dict(symbol="XYZ", price=100.0, change_pct=0.0, realized_vol_20d=0.3, realized_vol_60d=0.3, atm_iv=None)
    return UnderlyingContext(**{**base, **overrides})


@pytest.fixture(scope="module", autouse=True)
def demo_day_is_pinned():
    """Hold the demo market at TODAY: the command line and the dashboard build theirs from the wall clock."""
    wall_clock = DemoMarket.__init__

    def pinned(self, today=None):
        wall_clock(self, today or TODAY)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(DemoMarket, "__init__", pinned)
        yield


# -- The file ------------------------------------------------------------------------


def test_a_position_round_trips_through_the_file(tmp_path):
    store = PositionStore(tmp_path / "data" / "positions.json")
    assert store.load() == ([], 0)  # no file yet is no positions, and not an error
    first = store.add(position())
    second = store.add(position(underlying=" brk-b ", kind="call", strike=512.5, side="long", quantity=3, entry_price=7.15))
    assert PositionStore(store.path).load() == ([first, second], 0)

    assert first.symbol == "XYZ270108P00095000"
    # A share class is written both ways, and its options are listed with no dot at all.
    assert second.underlying == "BRK.B" and second.symbol == "BRKB270108C00512500"
    assert first.id != second.id and len(first.id) == 8

    # The file is meant to be read, and edited by hand if need be.
    on_file = json.loads(store.path.read_text())
    assert on_file["positions"][0] == {
        "id": first.id,
        "underlying": "XYZ",
        "kind": "put",
        "strike": 95.0,
        "expiry": "2027-01-08",
        "side": "short",
        "quantity": 1,
        "entry_price": 2.0,
        "opened": "2026-11-14",
    }
    assert not list(tmp_path.glob("data/*.tmp"))


@pytest.mark.parametrize(
    "changes",
    [
        {"underlying": "DROP TABLE"},
        {"underlying": ""},
        {"underlying": 5},
        {"kind": "straddle"},
        {"side": "hold"},
        {"strike": 0},
        {"strike": float("nan")},
        {"strike": 100_000},
        {"strike": True},
        {"quantity": 0},
        {"quantity": 1.5},
        {"quantity": True},
        {"entry_price": -1.0},
        {"entry_price": float("inf")},
        {"entry_price": 1e-310},  # a slip of the hand that turned every result measured against it into infinity
        {"entry_price": 0.00004},
        {"entry_price": 1e308},
        {"expiry": "next Friday"},
        {"expiry": datetime(2026, 12, 18, 16, 0)},  # a time of day cannot be compared with a date
        {"opened": TODAY + timedelta(days=60)},  # after the contract has expired
        {"id": " "},
    ],
)
def test_what_is_not_a_position_is_turned_away(changes):
    with pytest.raises(ValueError):
        position(**changes)


@pytest.mark.parametrize("written", ["{half a file", "[]", json.dumps({"positions": "none"}), json.dumps({"version": 1})])
def test_a_file_that_does_not_read_is_an_error_and_is_left_as_it_is(tmp_path, written):
    # The opposite of the cache's rule. Read as "no positions", the file would be
    # written over by the next entry, and what a person typed in cannot be fetched again.
    path = tmp_path / "positions.json"
    path.write_text(written)
    store = PositionStore(path)
    for use in (store.load, lambda: store.add(position()), lambda: store.remove("abc")):
        with pytest.raises(StoreError, match="positions.json"):
            use()
    assert path.read_text() == written


def test_an_empty_file_is_no_positions(tmp_path):
    path = tmp_path / "positions.json"
    path.write_text("\n")
    store = PositionStore(path)
    assert store.load() == ([], 0)
    store.add(position())
    assert len(store.load()[0]) == 1


def test_an_entry_that_does_not_read_is_kept_and_counted(tmp_path):
    path = tmp_path / "positions.json"
    odd = {"id": "odd", "underlying": "XYZ", "strike": "about 95"}
    mine = {**json.loads(json.dumps(positions._plain(position(id="mine")))), "why": "sold after the dip"}
    path.write_text(json.dumps({"positions": [odd, mine, "a stray line"]}))
    store = PositionStore(path)

    found, unread = store.load()
    assert [p.id for p in found] == ["mine"] and unread == 2
    assert "2 entries" in positions.unread_note(unread, path) and "1 entry" in positions.unread_note(1, path)

    # Adding and removing rewrite the file. What could not be read is still in it, and so is
    # anything a person added to an entry by hand.
    added = store.add(position())
    assert store.remove(added.id)
    assert json.loads(path.read_text())["positions"] == [odd, mine, "a stray line"]
    assert store.remove("mine")
    assert json.loads(path.read_text())["positions"] == [odd, "a stray line"]
    assert not store.remove("odd")  # it does not read as a position, so it is not one to remove


def test_a_write_that_fails_is_an_error(tmp_path):
    (tmp_path / "taken").write_text("a file where the folder should be")
    with pytest.raises(StoreError):
        PositionStore(tmp_path / "taken" / "positions.json").add(position())


def test_an_entry_price_is_kept_to_a_hundredth_of_a_cent():
    assert position(entry_price=1.2550000001).entry_price == 1.255  # an average of two fills
    assert position(entry_price=positions.MIN_PRICE).entry_price == 0.0001
    assert position(entry_price=positions.MAX_PRICE).entry_price == 1_000_000


def test_a_file_saved_with_a_byte_order_mark_still_reads(tmp_path):
    # An older Notepad and PowerShell's Set-Content both put one in front, and the file is meant to be edited by hand.
    path = tmp_path / "positions.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"positions": [positions._plain(position(id="mine"))]}).encode())
    store = PositionStore(path)
    assert [held.id for held in store.load()[0]] == ["mine"]
    store.add(position())
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf") and len(store.load()[0]) == 2


def test_an_entry_that_repeats_an_id_is_never_removed_with_the_first(tmp_path):
    # An entry copied by hand keeps its id unless that is changed too. Removing one used to take both.
    path = tmp_path / "positions.json"
    first, copy = positions._plain(position(id="same")), positions._plain(position(id="same", strike=90.0))
    path.write_text(json.dumps({"positions": [first, copy]}))
    store = PositionStore(path)
    found, unread = store.load()
    assert [(held.id, held.strike) for held in found] == [("same", 95.0)] and unread == 1
    assert "repeats another entry's id" in positions.unread_note(unread, path)
    with pytest.raises(ValueError, match="already"):
        store.add(position(id="same", strike=80.0))

    assert store.remove("same")
    assert json.loads(path.read_text())["positions"] == [copy]  # the copy is still there, and now it reads
    assert [(held.id, held.strike) for held in store.load()[0]] == [("same", 90.0)] and store.load()[1] == 0


def test_removing_takes_one_entry_and_says_when_there_is_none(tmp_path):
    store = PositionStore(tmp_path / "positions.json")
    first, second = store.add(position()), store.add(position(strike=90.0))
    assert store.remove(first.id) and not store.remove(first.id)
    assert store.load() == ([second], 0)
    with pytest.raises(ValueError, match="already"):
        store.add(second)


def test_the_demo_keeps_its_positions_in_memory(tmp_path):
    settings = Settings(positions_file=tmp_path / "positions.json")
    store = positions.open_store(settings, DemoMarket(TODAY))
    assert store.path is None
    made_up, unread = store.load()
    assert {p.underlying for p in made_up} <= set(SPECS) and len(made_up) == 5 and unread == 0
    store.add(position())
    assert len(store.load()[0]) == 6 and store.remove(made_up[0].id)
    assert not settings.positions_file.exists()  # demo data never reaches the disk
    # Without a demo market it is the person's own file.
    assert positions.open_store(settings).path == settings.positions_file


def test_the_file_can_be_kept_somewhere_else(tmp_path, monkeypatch):
    monkeypatch.setenv("POSITIONS_FILE", str(tmp_path / "mine.json"))
    assert load_settings().positions_file == tmp_path / "mine.json"
    monkeypatch.delenv("POSITIONS_FILE")
    assert load_settings().positions_file == Settings().positions_file
    assert Settings().positions_file.parent != Settings().cache_dir  # nothing that clears the cache may take it


# -- The rules -----------------------------------------------------------------------


def test_closing_now_is_priced_at_the_bid_for_a_holder_and_the_ask_to_buy_back():
    # The spread is counted, as it is in the exit plan: neither side closes at the mid.
    holder = position(side="long", quantity=3)
    assert holdings.result(holder, quoted(holder, 2.50, 2.70)) == pytest.approx((2.50, 150.0, 0.25))
    seller = position(side="short", quantity=2)
    assert holdings.result(seller, quoted(seller, 1.00, 1.20)) == pytest.approx((1.20, 160.0, 0.40))
    # With the side that closing needs missing there is no result, whatever the other side says.
    assert holdings.result(holder, quoted(holder, None, 2.70)) is None
    assert holdings.result(seller, quoted(seller, 1.00, 0.0)) is None
    assert holdings.result(seller, None) is None


def test_the_profit_target_is_measured_at_the_ask():
    p = position(entry_price=2.00)  # the 50% target is to buy back at $1.00 or less
    reached = holdings.profit_target(p, 1.00)
    assert reached.state == "reached" and "keeps 50% of the $2.00 collected" in reached.detail
    short_of_it = holdings.profit_target(p, 1.01)
    assert short_of_it.state == "clear" and "would keep 49.5%" in short_of_it.detail and "$1.00 or less" in short_of_it.detail
    losing = holdings.profit_target(p, 2.60)
    assert losing.state == "clear" and "costs $0.60 a share more" in losing.detail
    # A bid alone says nothing about what buying back would cost.
    assert holdings.profit_target(p, None).state == "unknown"
    assert exits.PROFIT_TARGET == 0.50


def test_the_profit_target_never_reads_as_reached_when_it_is_not():
    # From the outside review. Half of $1.03 is $0.515. Rounded for show, an ask of $0.52 "would keep 50%"
    # against a target of "$0.52 or less", and the check beside those words said not reached.
    p = position(entry_price=1.03)
    short_of_it = holdings.profit_target(p, 0.52)
    assert short_of_it.state == "clear"
    assert short_of_it.detail == "Buying back at $0.52 would keep 49.5% of the $1.03 collected. The 50% target is $0.515 or less."
    there = holdings.profit_target(p, 0.515)  # the free feed's quotes are not always whole cents
    assert there.state == "reached" and there.detail.startswith("Buying back at $0.515 keeps 50% of the $1.03 collected.")
    assert holdings.profit_target(p, 0.51).state == "reached"

    assert [holdings.dollars(v) for v in (2.0, 4.19, 0.515, 0.05, 1234.5)] == ["$2.00", "$4.19", "$0.515", "$0.05", "$1234.50"]
    # Cut off, never rounded up: 8.38 / 2 is 4.19 exactly, and 1.2555 / 2 is 0.62775.
    assert [holdings.at_most(v) for v in (4.19, 0.515, 0.62775, 1.0)] == ["$4.19", "$0.515", "$0.6277", "$1.00"]


def test_a_quote_out_of_date_against_the_stock_gives_no_result():
    # From the outside review. A put sold for $2.00 at a $95 strike, and the stock has dropped to $75 after the
    # close: exercising would pay $20. The option's quote is still the close's, 0.40 x 0.50. Read as it stands,
    # the position showed +$150 and "profit target reached" on the night it was deepest under water.
    p = position()
    stale = holdings.assess(p, TODAY, spot=75.0, quote=quoted(p, 0.40, 0.50), ctx=context(price=75.0))
    assert stale.pnl is None and stale.pnl_pct is None and stale.close_price is None
    assert (stale.bid, stale.ask) == (0.40, 0.50)  # the quote is still shown, as what it is
    states = {c.rule: c.state for c in stale.checks}
    assert states["profit_target"] == "unknown" and states["strike"] == "reached"
    assert "out of date against the stock" in stale.note and "$0.50" in stale.note and "$20.00" in stale.note
    assert "not a price to buy back at" in stale.checks[0].detail

    # A holder's result goes the same way: the whole quote is under exercise value.
    owned = position(side="long")
    assert holdings.assess(owned, TODAY, spot=75.0, quote=quoted(owned, 0.40, 0.50)).pnl is None

    # A bid under exercise value with the ask above it is an ordinary deep in-the-money quote. It stands.
    deep = quoted(p, 19.20, 20.60)
    assert holdings.unusable(p, deep, 75.0) is None
    assert holdings.assess(p, TODAY, spot=75.0, quote=deep).pnl == pytest.approx((2.00 - 20.60) * 100)
    assert holdings.assess(owned, TODAY, spot=75.0, quote=quoted(owned, 19.20, 20.60)).close_price == 19.20
    # So does a quote at exercise value to the cent, and one with no stock price to hold it against.
    assert holdings.unusable(p, quoted(p, 19.90, 20.00), 75.0) is None
    assert holdings.assess(p, TODAY, quote=quoted(p, 0.40, 0.50)).pnl == pytest.approx(150.0)


def test_a_crossed_quote_gives_no_result():
    # From the outside review: a bid of 5.00 over an ask of 0.50 gave a seller "profit target reached".
    for side in ("short", "long"):
        p = position(side=side)
        crossed = holdings.assess(p, TODAY, spot=105.0, quote=quoted(p, 5.00, 0.50), ctx=context(price=105.0))
        assert crossed.pnl is None and "crossed" in crossed.note
    assert holdings.assess(position(), TODAY, spot=105.0, quote=quoted(position(), 5.00, 0.50)).checks[0].state == "unknown"
    # No bid at all is not crossed: it is what a nearly worthless option looks like.
    assert holdings.unusable(position(), quoted(position(), None, 0.05), 105.0) is None
    assert holdings.unusable(position(), None, 105.0) is None


def test_only_a_seller_has_a_profit_target():
    sold = holdings.assess(position(side="short"), TODAY)
    bought = holdings.assess(position(side="long"), TODAY)
    assert [c.rule for c in sold.checks] == ["profit_target", "time_stop", "earnings", "strike"]
    assert [c.rule for c in bought.checks] == ["time_stop", "earnings", "strike"]
    assert {c.rule for c in sold.checks} <= set(holdings.RULES)


def test_the_time_stop_is_the_last_twenty_one_days():
    outside = holdings.time_stop(position(expiry=TODAY + timedelta(days=22)), TODAY)
    assert outside.state == "clear" and "The 21-day time stop is Nov 25." in outside.detail
    for side in ("short", "long"):  # it is counted for a holder too: decay is fastest there
        inside = holdings.time_stop(position(expiry=TODAY + timedelta(days=21), side=side), TODAY)
        assert inside.state == "reached" and "21 days to expiry" in inside.detail
    last_day = holdings.time_stop(position(expiry=TODAY, opened=TODAY), TODAY)
    assert last_day.state == "reached" and last_day.detail.startswith("Expires today.")


REPORT_IN = lambda days, timing="post", confirmed=True: EarningsEvent(TODAY + timedelta(days=days), timing, confirmed=confirmed)  # noqa: E731


@pytest.mark.parametrize(
    "event, expires_in, state, label",
    [
        (REPORT_IN(7), 45, "reached", "earnings in 7d"),
        (REPORT_IN(0), 45, "reached", "earnings today"),  # after today's close: tomorrow is the day it is felt
        (REPORT_IN(8), 45, "clear", "report before expiry"),
        # A report after the close moves the stock the next day, so a contract expiring that day misses it.
        (REPORT_IN(5, "post"), 5, "clear", "expires before earnings"),
        (REPORT_IN(5, "pre"), 5, "reached", "earnings in 5d"),
        (REPORT_IN(5, "unknown"), 5, "reached", "earnings in 5d"),  # it may come before the open
        (REPORT_IN(0, "pre"), 45, "clear", "report already out"),
        (REPORT_IN(60), 45, "clear", "expires before earnings"),
        # An estimated date is given a week of room, as the skip-earnings filter gives it.
        (REPORT_IN(14, confirmed=False), 45, "reached", "earnings in 14d (est.)"),
        (REPORT_IN(15, confirmed=False), 45, "clear", "report before expiry"),
        (REPORT_IN(50, confirmed=False), 45, "unknown", "earnings near expiry (est.)"),
        (REPORT_IN(53, confirmed=False), 45, "clear", "expires before earnings"),
    ],
)
def test_a_report_is_pointed_out_when_it_is_close_and_the_contract_is_still_open(event, expires_in, state, label):
    p = position(expiry=TODAY + timedelta(days=expires_in))
    check = holdings.earnings(p, context(next_earnings=event), TODAY, ESTIMATE_MARGIN)
    assert (check.state, check.label) == (state, label)
    assert holdings.EARNINGS_SOON == 7 and ESTIMATE_MARGIN == timedelta(days=7)


def test_a_report_that_is_out_and_not_yet_felt_is_not_called_today():
    # From the outside review. No source hands the app such a date today (EarningsService.upcoming gives none
    # that is past), but the check is plain arithmetic and has to be right for whatever it is handed.
    saturday = date(2026, 11, 21)
    p = position(expiry=saturday + timedelta(days=30), opened=saturday - timedelta(days=5))
    check = holdings.earnings(p, context(next_earnings=EarningsEvent(date(2026, 11, 20), "post")), saturday)
    assert (check.state, check.label) == ("reached", "earnings out, not yet felt")
    assert check.detail == "Reported Nov 20, after the close. The first session to reflect it is Nov 23, and this contract is still open then."


def test_no_earnings_date_is_never_read_as_no_report():
    # As with skip-earnings, this fails closed: "cannot tell" is said, and is not a pass.
    unknown = holdings.earnings(position(), context(earnings_unknown=True), TODAY)
    assert unknown.state == "unknown" and "cannot be ruled out" in unknown.detail
    assert holdings.earnings(position(), None, TODAY).state == "unknown"  # the stock could not be looked up at all
    fund = holdings.earnings(position(), context(), TODAY)
    assert fund.state == "clear" and "normal for a fund" in fund.detail


@pytest.mark.parametrize(
    "kind, side, spot, state, label",
    [
        ("put", "short", 94.99, "reached", "through the strike"),
        ("put", "short", 95.00, "clear", "strike not reached"),
        ("call", "short", 95.01, "reached", "through the strike"),
        ("call", "short", 90.00, "clear", "strike not reached"),
        # In the money is what a holder is waiting for, so it is said and not pointed out.
        ("call", "long", 99.00, "clear", "in the money"),
        ("put", "long", 99.00, "clear", "out of the money"),
        ("put", "short", None, "unknown", "stock price unknown"),
    ],
)
def test_only_a_short_option_is_pointed_out_for_being_through_its_strike(kind, side, spot, state, label):
    check = holdings.strike(position(kind=kind, side=side), spot)
    assert (check.state, check.label) == (state, label)


def test_the_strike_is_measured_the_way_the_exit_plan_measures_it():
    check = holdings.strike(position(strike=95.0), 100.0)
    assert check.detail == "The $95 strike is 5.0% below the stock price ($100.00)."  # strike / stock - 1, as in ExitPlan
    assert "can be assigned" in holdings.strike(position(strike=95.0), 90.0).detail


def test_a_position_with_nothing_to_go_on_is_still_listed():
    p = position()
    bare = holdings.assess(p, TODAY)
    assert bare.position is p and bare.dte == 45 and bare.pnl is None and bare.close_price is None and bare.plan is None
    assert {c.rule: c.state for c in bare.checks} == {
        "profit_target": "unknown",
        "time_stop": "clear",  # this one needs only the date
        "earnings": "unknown",
        "strike": "unknown",
    }


def test_an_expired_position_gets_one_check_and_no_price():
    p = position(expiry=TODAY - timedelta(days=4), opened=TODAY - timedelta(days=30))
    gone = holdings.assess(p, TODAY, spot=100.0, quote=quoted(p, 1.0, 1.2), ctx=context())
    assert [(c.rule, c.state) for c in gone.checks] == [("expired", "reached")]
    assert "Expired on Nov 20" in gone.checks[0].detail and gone.pnl is None and gone.dte == -4
    assert len(holdings.assess(position(expiry=TODAY, opened=TODAY), TODAY).checks) == 4  # the last day is not expired


# -- The path ahead, from the price it was opened at -----------------------------------------


def test_an_exit_plan_for_a_held_contract_is_measured_against_its_entry_price():
    fair = pricing.price("put", 100.0, 95.0, 45 / 365, RATE, 0.30)
    quote = OptionQuote("X", "XYZ", "put", 95.0, TODAY + timedelta(days=45), bid=fair - 0.05, ask=fair + 0.05, iv=0.30)
    c = metrics.analyze_contract(quote, 100.0, TODAY, RATE)
    opening_now = exits.build(c, context(), "sell", RATE, TODAY)
    held = exits.build(c, context(), "sell", RATE, TODAY, entry=4.00)

    assert opening_now.entry_price == c.bid and held.entry_price == 4.00
    assert held.target_price == pytest.approx(2.00)
    # The path is the same one, and still starts at today's quoted mid. Only the results differ.
    assert [(k.day, k.value, k.close_price) for k in held.checkpoints] == [(k.day, k.value, k.close_price) for k in opening_now.checkpoints]
    assert held.checkpoints[-1].label == "at expiry" and held.checkpoints[-1].pnl == pytest.approx(400.0)
    assert held.checkpoints[0].pnl == pytest.approx((4.00 - held.checkpoints[0].close_price) * 100)
    assert held.checkpoints[0].pnl_pct == pytest.approx(held.checkpoints[0].pnl / 400.0)
    assert exits.build(c, context(), "sell", RATE, TODAY, entry=0.0) is None

    # From the outside review: "decay alone gets there" answered tomorrow for a target met today.
    not_yet = exits.build(c, context(), "sell", RATE, TODAY, entry=1.5 * c.ask)  # the ask is above three quarters of itself
    assert not_yet.target_day is None or not_yet.target_day > TODAY
    already = exits.build(c, context(), "sell", RATE, TODAY, entry=2 * c.ask)
    assert c.ask <= already.target_price and already.target_day == TODAY
    assert opening_now.target_day != TODAY  # opened now it never is: the ask is above the bid it was sold at


def test_a_step_that_history_cannot_size_is_not_blamed_on_the_options():
    # From the outside review. A review reads no chain, so the options give no reading. When past reports then
    # size the step at nothing, the note said "the options price no extra move", which nobody had looked at.
    fair = pricing.price("put", 100.0, 95.0, 45 / 365, RATE, 0.30)
    quote = OptionQuote("X", "XYZ", "put", 95.0, TODAY + timedelta(days=45), bid=fair - 0.05, ask=fair + 0.05, iv=0.30)
    c = metrics.analyze_contract(quote, 100.0, TODAY, RATE)
    c.spans_earnings = True
    report = EarningsEvent(TODAY + timedelta(days=9), "post")
    tiny = [Reaction(TODAY - timedelta(days=91 * n), "unknown", 0.0005, sessions=2) for n in range(1, 6)]
    assert risk.event_variance(tiny, 0.30) == 0.0

    unread = exits.build(c, context(next_earnings=report, event_var=0.0), "sell", RATE, TODAY)
    assert unread.earnings is None
    assert unread.notes[0] == "Past reports were too small to size a step for the coming one, so none is shown."
    # When the options were read and price nothing, that is still what is said.
    priced = ImpliedMove(c.expiry, None, 0.0, 0.30, "test", event_var=0.0)
    read = exits.build(c, context(next_earnings=report, implied_move=priced, event_var=0.01), "sell", RATE, TODAY)
    assert read.notes[0] == "The options price no extra move for the coming report, so no step is shown for it."


# -- A review on the demo market ---------------------------------------------------------------


@pytest.fixture()
def scanner() -> Scanner:
    return build_scanner(Settings(), demo=True)


def flagged(review) -> dict[str, list[str]]:
    return {h.position.underlying: [c.rule for c in h.checks if c.state != "clear"] for h in review.holdings}


def test_a_review_sets_each_demo_position_against_the_rules(scanner):
    held = scanner.market.sample_positions()
    review = scanner.review(held)
    assert review.mode == "demo" and review.warnings == []
    assert [h.position for h in review.holdings] == held  # every one, in the order entered

    # The demo holds one position for each thing the page checks, and one with nothing to say.
    assert flagged(review) == {"CRUX": ["profit_target"], "ORBT": ["time_stop"], "BOLT": ["earnings"], "EMBR": ["strike"], "FLUX": []}
    for h in review.holdings:
        p = h.position
        assert h.bid < h.ask and h.close_price == (h.bid if p.side == "long" else h.ask)
        gain = h.close_price - p.entry_price if p.side == "long" else p.entry_price - h.close_price
        assert h.pnl == pytest.approx(gain * 100 * p.quantity) and h.pnl_pct == pytest.approx(gain / p.entry_price)
        assert h.note is None and h.contract.symbol == p.symbol and h.contract.lens == holdings.LENS[p.side]
        assert h.plan.entry_price == p.entry_price and h.plan.lens == h.contract.lens
        assert h.plan.checkpoints[-1].label == "at expiry"
    assert review.underlyings["EMBR"].contracts_scanned == 1 and review.underlyings["EMBR"].reasons == ["held"]
    # CRUX is at its target now, and its plan says so too.
    assert review.holdings[0].plan.target_day == TODAY
    # What a review hands out is plain JSON: no figure in it is infinite or not a number.
    json.dumps(asdict(review), default=str, allow_nan=False)


def test_a_review_carries_the_earnings_and_the_news_of_what_is_held(scanner):
    review = scanner.review(scanner.market.sample_positions())
    by_stock = {h.position.underlying: h for h in review.holdings}
    bolt, ctx = by_stock["BOLT"], review.underlyings["BOLT"]
    assert ctx.next_earnings.day == date(2026, 11, 30) and ctx.earnings_stats.count == 12
    assert bolt.contract.spans_earnings and bolt.plan.earnings.report_day == date(2026, 12, 1)
    assert "Reports Nov 30, after the close (in 6 days)" in next(c.detail for c in bolt.checks if c.rule == "earnings")
    # No chain was read, so what the options price for the report is not known, and the plan says
    # where the size of its step came from.
    assert ctx.implied_move is None and ctx.atm_iv is None
    assert any("sized from this stock's past reports" in note for note in bolt.plan.notes)
    crux = by_stock["CRUX"]
    assert not crux.contract.spans_earnings and crux.plan.earnings is None
    assert not any("past reports" in note for note in crux.plan.notes)
    # A held stock gets the news section a scanned one gets, flagged events and all.
    assert review.underlyings["CRUX"].news and "split" in review.underlyings["CRUX"].recent_events
    assert review.underlyings["CRUX"].pooled_studies


def test_a_review_pulls_no_chain(scanner):
    held = scanner.market.sample_positions()
    asked = []
    real = scanner.market.option_quotes

    def counted(contracts):
        asked.append(dict(contracts))
        return real(contracts)

    def no_chain(*args, **kwargs):
        raise AssertionError("a review asks for the held contracts by name")

    scanner.market.option_quotes, scanner.market.option_chain, scanner.market.open_interest = counted, no_chain, no_chain
    review = scanner.review(held)
    assert asked == [{p.symbol: p.underlying for p in held}]  # one request, whatever the stocks
    assert all(h.pnl is not None for h in review.holdings)


def test_a_contract_with_no_quote_is_still_listed(scanner):
    far_out = position(underlying="ACME", kind="call", strike=600.0)  # nobody bids for it
    review = scanner.review([far_out])
    (h,) = review.holdings
    assert h.bid is None and h.pnl is None and h.contract is None and h.plan is None
    assert h.note == "No quote was found for this contract. Check its expiry and strike."
    # What does not need the option's quote is still checked.
    assert {c.rule: c.state for c in h.checks} == {"profit_target": "unknown", "time_stop": "clear", "earnings": "clear", "strike": "clear"}
    assert "ACME" in review.underlyings


def test_a_quote_that_cannot_be_priced_still_gives_a_result(scanner):
    p = position(underlying="ACME", strike=170.0, side="short", entry_price=3.00)
    scanner.market.option_quotes = lambda contracts: {p.symbol: quoted(p, None, 0.05)}  # no bid: nearly worthless
    (h,) = scanner.review([p]).holdings
    assert h.contract is None and h.plan is None and "cannot be priced" in h.note
    # Buying back needs only the ask, and at five cents the target is plainly reached.
    assert h.close_price == 0.05 and h.pnl == pytest.approx(295.0)
    assert next(c for c in h.checks if c.rule == "profit_target").state == "reached"


def test_a_stale_quote_in_a_review_is_named_as_stale(scanner):
    # From the outside review, through the whole review: the reason is the out-of-date quote, not a general
    # "cannot be priced", and nothing is shown as a result.
    p = position(underlying="ACME", strike=220.0, side="short", entry_price=3.00)  # the demo's ACME is at 184
    scanner.market.option_quotes = lambda contracts: {p.symbol: quoted(p, 0.40, 0.50)}
    (h,) = scanner.review([p]).holdings
    assert h.pnl is None and h.contract is None and h.note.startswith("This quote is out of date against the stock.")
    assert {c.rule: c.state for c in h.checks}["profit_target"] == "unknown"


def test_a_contract_last_quoted_on_an_earlier_day_says_when(scanner):
    # From the outside review. The provider's answer for such a contract used to be nothing at all, and the
    # position was told to check a strike that was perfectly right.
    old, fresh = position(underlying="ACME", strike=170.0), position(underlying="ACME", strike=175.0)
    real = scanner.market.option_quotes
    friday = datetime(2026, 11, 20, 15, 59, tzinfo=ET)

    def one_is_old(contracts):
        quotes = real(contracts)
        quotes[old.symbol] = OptionQuote(old.symbol, "ACME", "put", 170.0, old.expiry, quote_time=friday, outdated=True)
        return quotes

    scanner.market.option_quotes = one_is_old
    stale, live = scanner.review([old, fresh]).holdings
    assert stale.note == "This contract was last quoted on Nov 20. A quote from an earlier day is not used, so the position is not priced."
    assert stale.pnl is None and stale.bid is None and stale.contract is None and stale.quote_time == friday
    assert {c.rule: c.state for c in stale.checks}["strike"] == "clear"  # what needs no option quote is still checked
    assert live.pnl is not None and live.note is None


def test_every_position_is_listed_when_a_source_fails(scanner):
    held = scanner.market.sample_positions()

    def down(*args, **kwargs):
        raise httpx.ConnectError("no route")

    scanner.market.option_quotes = down
    review = scanner.review(held)
    assert len(review.holdings) == 5 and all(h.pnl is None and h.plan is None for h in review.holdings)
    assert all(h.note == "Option quotes could not be loaded, so this position is not priced." for h in review.holdings)
    assert any(w.startswith("Option quotes could not be loaded:") for w in review.warnings)
    # The stock prices did arrive, so the checks that rest on them were still made.
    assert flagged(review)["EMBR"] == ["profit_target", "strike"]

    scanner.market.stock_snapshots = down
    review = scanner.review(held)
    assert len(review.holdings) == 5 and review.underlyings == {}
    assert sum(w.startswith("Stock prices could not be loaded:") for w in review.warnings) == 1
    assert all({c.rule: c.state for c in h.checks}["strike"] == "unknown" for h in review.holdings)


def test_one_stock_that_fails_does_not_hide_the_others(scanner):
    real = scanner.earnings.history

    def flaky(symbol):
        if symbol == "BOLT":
            raise RuntimeError("lookup broke")
        return real(symbol)

    scanner.earnings.history = flaky
    review = scanner.review(scanner.market.sample_positions())
    assert "BOLT: its earnings dates could not be looked up (lookup broke)." in review.warnings
    assert flagged(review)["CRUX"] == ["profit_target"]

    # From the outside review. The first version dropped the stock whole and told the person "There is no
    # stock price for BOLT", with the price in hand. Only the dates are lost, and the checks say so.
    bolt, ctx = next(h for h in review.holdings if h.position.underlying == "BOLT"), review.underlyings["BOLT"]
    assert ctx.price == SPECS["BOLT"]["price"] and ctx.earnings_unknown and ctx.next_earnings is None
    assert ctx.earnings_note == "Earnings lookup failed: RuntimeError."
    assert bolt.note is None and bolt.pnl is not None and bolt.contract is not None and bolt.plan is not None
    states = {c.rule: c.state for c in bolt.checks}
    assert states["strike"] == "clear" and states["earnings"] == "unknown"
    assert any("No earnings date is known" in note for note in bolt.plan.notes)
    assert ctx.news  # and its news is still read


def test_a_stock_that_cannot_be_looked_up_is_not_said_to_have_no_price(scanner, monkeypatch):
    # Whatever else goes wrong for one stock, the note names what happened, and the strike is still set
    # against the price that did arrive.
    real = metrics.realized_vol

    def breaks_for_bolt(bars, window, exclude=frozenset()):
        if bars and bars[-1].close == round(SPECS["BOLT"]["price"] / (1 + SPECS["BOLT"]["change"]), 2):
            raise ZeroDivisionError("bad bar")
        return real(bars, window, exclude)

    monkeypatch.setattr(metrics, "realized_vol", breaks_for_bolt)
    review = scanner.review(scanner.market.sample_positions())
    bolt = next(h for h in review.holdings if h.position.underlying == "BOLT")
    assert "BOLT" not in review.underlyings and "BOLT: bad bar" in review.warnings
    assert bolt.note == "BOLT could not be looked up, so the Greeks and the path ahead are left out."
    assert {c.rule: c.state for c in bolt.checks}["strike"] == "clear" and bolt.pnl is not None
    assert review.holdings[0].pnl is not None


def test_a_calendar_note_from_an_earlier_scan_is_not_repeated_by_a_review_that_asked_nothing(scanner):
    # From the outside review. The note stays on the earnings service between scans. With the stock prices
    # down no stock's dates are looked up, so nothing here has asked the calendar.
    held = scanner.market.sample_positions()

    def timed_out(today):
        raise httpx.ReadTimeout("slow")

    scanner.market.calendar = timed_out  # the demo market is the demo's calendar
    left_over = "demo data earnings calendar unavailable (ReadTimeout); upcoming dates are estimates."
    assert left_over in scanner.review(held).warnings  # dates were looked up just now, and the note is about them
    assert scanner.earnings.calendar_note == left_over

    def down(*args, **kwargs):
        raise httpx.ConnectError("no route")

    scanner.market.stock_snapshots = down
    review = scanner.review(held)
    assert review.underlyings == {} and review.warnings == ["Stock prices could not be loaded: no route"]


def test_an_unknown_stock_is_named_and_its_position_kept(scanner):
    review = scanner.review([position(underlying="NOPE")])
    assert "NOPE: no stock quote found." in review.warnings
    assert len(review.holdings) == 1 and review.holdings[0].pnl is None


def test_expired_positions_cost_no_request(scanner):
    def no_request(*args, **kwargs):
        raise AssertionError("nothing is quoted for an expired contract")

    for name in ("stock_snapshots", "option_quotes", "daily_bars", "headlines"):
        setattr(scanner.market, name, no_request)
    gone = position(underlying="ACME", expiry=TODAY - timedelta(days=1), opened=TODAY - timedelta(days=40))
    review = scanner.review([gone])
    assert review.warnings == [] and review.holdings[0].checks[0].rule == "expired" and review.holdings[0].note is None
    assert scanner.review([]).holdings == []


def test_a_held_stock_with_no_earnings_source_cannot_be_shown_clear_of_a_report():
    market = DemoMarket(TODAY)
    bare = Scanner(market, EarningsService(NullCache(), today=TODAY), Settings(), today=TODAY)
    (h,) = bare.review([position(underlying="ACME", strike=170.0)]).holdings
    assert next(c for c in h.checks if c.rule == "earnings").state == "unknown"
    assert h.plan is not None and any("No earnings date is known" in note for note in h.plan.notes)


def test_a_review_waits_its_turn_with_a_scan(scanner):
    # The provider and its notes are shared, so a review and a scan never run together.
    held, done = scanner.market.sample_positions()[:1], threading.Event()

    def run():
        scanner.review(held)
        done.set()

    worker = threading.Thread(target=run)
    with scanner._lock:  # a scan in progress
        worker.start()
        assert not done.wait(0.2)
    worker.join(10)
    assert done.is_set()

    # And a review starts with a clean slate and leaves one, as a scan does.
    scanner.market.notes.append("left over from something else")
    assert scanner.review(held).warnings == [] and scanner.market.notes == []


# -- Alpaca's snapshots by contract ------------------------------------------------------------

NOW = datetime(2026, 11, 24, 17, 0, tzinfo=ET)


def alpaca(handler, now=NOW) -> AlpacaData:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return AlpacaData("key-id", "secret", client=client, sleep=lambda _: None, now=lambda: now)


def snapshot(stamp: str, bid: float = 6.1, ask: float = 6.3, volume: int = 40) -> dict:
    return {
        "greeks": {"delta": -0.31, "gamma": 0.02, "theta": -0.09, "vega": 0.18},
        "impliedVolatility": 0.28,
        "latestQuote": {"t": stamp, "bp": bid, "ap": ask},
        "dailyBar": {"t": stamp[:10] + "T05:00:00Z", "v": volume},
    }


def test_held_contracts_are_asked_for_by_name_in_one_request():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "snapshots": {
                    "AAPL261218P00200000": snapshot("2026-11-24T20:59:59Z"),
                    "BRKB270115C00500000": snapshot("2026-11-24T20:58:00Z", bid=9.0, ask=9.4),
                    "MSFT261218C00500000": snapshot("2026-11-24T20:59:00Z"),  # nobody asked for this one
                },
                "next_page_token": None,
            },
        )

    wanted = {"AAPL261218P00200000": "AAPL", "BRKB270115C00500000": "BRK.B"}
    quotes = alpaca(handler).option_quotes(wanted)
    assert len(seen) == 1 and seen[0].url.path == "/v1beta1/options/snapshots"
    assert seen[0].url.params["symbols"] == "AAPL261218P00200000,BRKB270115C00500000"
    assert seen[0].url.params["feed"] == "indicative"
    assert set(quotes) == set(wanted)
    put = quotes["AAPL261218P00200000"]
    assert (put.underlying, put.kind, put.strike, put.expiry) == ("AAPL", "put", 200.0, date(2026, 12, 18))
    assert (put.bid, put.ask, put.iv, put.delta, put.volume) == (6.1, 6.3, 0.28, -0.31, 40)
    assert put.quote_time == datetime.fromisoformat("2026-11-24T20:59:59+00:00")
    assert quotes["BRKB270115C00500000"].underlying == "BRK.B"  # the stock's own spelling, not the option root's


def test_a_held_contracts_quote_from_an_earlier_day_is_not_a_quote():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "snapshots": {
                    "AAPL261218P00200000": snapshot("2026-11-24T20:59:59Z"),
                    "COST261218C00900000": snapshot("2026-11-20T20:59:59Z"),  # its strike went with a special dividend
                    "NVDA261218C00150000": snapshot("2026-11-24T20:40:00Z", volume=7) | {"dailyBar": {"t": "2026-11-19T05:00:00Z", "v": 7}},
                }
            },
        )

    market = alpaca(handler)
    quotes = market.option_quotes({"AAPL261218P00200000": "AAPL", "COST261218C00900000": "COST", "NVDA261218C00150000": "NVDA"})
    # Options on every stock trade the same sessions, so the newest quote among them sets the day.
    assert not quotes["AAPL261218P00200000"].outdated and not quotes["NVDA261218C00150000"].outdated
    assert quotes["NVDA261218C00150000"].volume == 0  # its last trade was on an earlier day
    # From the outside review: the old quote used to vanish without a word. It comes back marked, with
    # the time it was made and none of its prices, so the position can say when it was last quoted.
    old = quotes["COST261218C00900000"]
    assert old.outdated and (old.bid, old.ask, old.iv, old.delta, old.volume) == (None, None, None, None, None)
    assert old.quote_time == datetime.fromisoformat("2026-11-20T20:59:59+00:00") and old.underlying == "COST"


def test_held_contracts_nobody_has_quoted_for_days_are_left_out_whole():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"snapshots": {"AAPL261218P00200000": snapshot("2026-11-17T20:59:59Z")}})

    market = alpaca(handler)
    (old,) = market.option_quotes({"AAPL261218P00200000": "AAPL"}).values()
    assert old.outdated and old.bid is None and old.ask is None and old.quote_time.day == 17
    assert market.notes == ["The newest quote for the contracts held is from Nov 17, so none was used."]
    # Within four days there is nothing to compare a single contract with, so it gets through with its time.
    friday = alpaca(lambda request: httpx.Response(200, json={"snapshots": {"AAPL261218P00200000": snapshot("2026-11-20T20:59:59Z")}}))
    kept = friday.option_quotes({"AAPL261218P00200000": "AAPL"})["AAPL261218P00200000"]
    assert kept.quote_time.day == 20 and not kept.outdated and kept.bid == 6.1


def test_one_mistyped_contract_does_not_cost_the_others_their_quotes():
    asked = []

    def handler(request: httpx.Request) -> httpx.Response:
        symbols = request.url.params["symbols"].split(",")
        asked.append(symbols)
        if "AAPL261218P00999999" in symbols:
            return httpx.Response(400, json={"message": "invalid symbol"})
        return httpx.Response(200, json={"snapshots": {s: snapshot("2026-11-24T20:59:59Z") for s in symbols}})

    wanted = {"AAPL261218P00200000": "AAPL", "AAPL261218P00999999": "AAPL"}
    assert set(alpaca(handler).option_quotes(wanted)) == {"AAPL261218P00200000"}
    assert asked == [list(wanted), ["AAPL261218P00200000"], ["AAPL261218P00999999"]]
    assert alpaca(handler).option_quotes({}) == {}


# -- The API, the page and the command line ------------------------------------------------------

ENTRY = {"underlying": "acme", "kind": "put", "strike": 170, "expiry": "2026-12-18", "side": "short", "quantity": 3, "entry_price": 2.4}


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(Settings(positions_file=tmp_path / "positions.json"), demo=True))


def test_positions_reach_the_api_and_the_page(client, tmp_path):
    status = client.get("/api/status").json()["positions"]
    assert status == {"rules": holdings.RULES, "earnings_soon_days": 7, "file": None}
    page = client.get("/").text
    assert 'id="position-form"' in page and 'id="tab-positions"' in page and "/api/positions" in page

    body = client.get("/api/positions").json()
    assert body["mode"] == "demo" and len(body["holdings"]) == 5
    first = body["holdings"][0]
    assert first["position"]["expiry"] == "2026-12-18" and first["position"]["symbol"] == "CRUX261218P00395000"
    assert first["checks"][0] == {
        "rule": "profit_target",
        "state": "reached",
        "label": "profit target reached",
        "detail": "Buying back at $3.61 keeps 57% of the $8.38 collected. The 50% target is $4.19 or less.",
    }
    assert first["pnl"] == pytest.approx(477.0) and first["plan"]["target_price"] == pytest.approx(4.19)
    assert first["contract"]["delta"] < 0 and "CRUX" in body["underlyings"]

    added = client.post("/api/positions", json=ENTRY)
    assert added.status_code == 201
    made = added.json()
    assert made["underlying"] == "ACME" and made["symbol"] == "ACME261218P00170000" and made["opened"] == "2026-11-24"
    listed = client.get("/api/positions").json()["holdings"]
    assert len(listed) == 6 and listed[-1]["position"]["id"] == made["id"] and listed[-1]["pnl"] is not None

    assert client.delete(f"/api/positions/{made['id']}").json() == {"removed": made["id"]}
    assert client.delete(f"/api/positions/{made['id']}").status_code == 404
    assert len(client.get("/api/positions").json()["holdings"]) == 5
    assert not (tmp_path / "positions.json").exists()  # the demo's positions never reach the disk


@pytest.mark.parametrize(
    "changes",
    [
        {"underlying": "DROP TABLE"},
        {"strike": 0},
        {"side": "hold"},
        {"kind": "both"},
        {"quantity": 0},
        {"entry_price": -2},
        {"entry_price": 1e-310},  # from the outside review: accepted, stored, and infinite in every answer after
        {"expiry": "soon"},
        {"opened": "2027-03-01"},  # after the contract expires
    ],
)
def test_the_api_turns_away_what_is_not_a_position(client, changes):
    assert client.post("/api/positions", json={**ENTRY, **changes}).status_code == 422
    assert len(client.get("/api/positions").json()["holdings"]) == 5


def test_the_page_guards_what_the_review_found_in_it(client):
    # From the outside review. The page has no test runner, so these hold the fixes in place by their text.
    page = client.get("/").text
    assert "[...new Set(lines.concat(notices[view]))]" in page  # one setup note, not the status's and the review's
    assert "if (ticket !== loads) return;" in page  # an answer that is not the newest load's is dropped
    assert "could not be loaded. Press Refresh to try again." in page  # the loading line does not stay up after a failure
    assert "buttons(true);" in page  # one remove request, however many clicks
    assert 'id="p_entry_price" type="number" min="0.0001" step="any"' in page  # an average of fills is not a whole cent
    assert page.count("holding.dte < 0 ? null : review.underlyings[p.underlying]") == 2  # an expired row has no stock context
    assert "atMost(plan.target_price)" in page  # the plan's target is cut off as the check's is, never rounded up
    assert page.count("exact(p.entry_price)") == 2 and "exact(plan.entry_price)" in page  # $1.255 is not shown as $1.25


def test_a_page_on_another_site_cannot_add_a_position(client):
    # A form on another site can post text without asking the browser's leave, but not JSON.
    sent = client.post("/api/positions", content=json.dumps(ENTRY), headers={"Content-Type": "text/plain"})
    assert sent.status_code == 422
    assert len(client.get("/api/positions").json()["holdings"]) == 5


def test_positions_can_be_kept_and_read_with_no_market_data(tmp_path):
    path = tmp_path / "positions.json"
    client = TestClient(create_app(Settings(positions_file=path), demo=False))  # no keys
    assert client.get("/api/status").json()["positions"]["file"] == str(path)
    assert client.get("/api/positions").json()["holdings"] == []

    made = client.post("/api/positions", json={**ENTRY, "opened": "2026-11-20"}).json()
    assert [p.id for p in PositionStore(path).load()[0]] == [made["id"]]
    body = client.get("/api/positions").json()
    assert body["mode"] == "unconfigured" and "ALPACA_API_KEY" in body["warnings"][0]
    (h,) = body["holdings"]
    assert h["position"]["id"] == made["id"] and h["pnl"] is None
    assert {c["rule"]: c["state"] for c in h["checks"]}["earnings"] == "unknown"
    assert client.delete(f"/api/positions/{made['id']}").status_code == 200 and PositionStore(path).load() == ([], 0)


def test_a_spoiled_file_is_reported_by_the_api_and_not_written_over(tmp_path):
    path = tmp_path / "positions.json"
    path.write_text("{half a file")
    client = TestClient(create_app(Settings(positions_file=path), demo=False))
    for response in (client.get("/api/positions"), client.post("/api/positions", json=ENTRY), client.delete("/api/positions/abc")):
        assert response.status_code == 500 and "does not read as a list of positions" in response.json()["detail"]
    assert path.read_text() == "{half a file"

    path.write_text(json.dumps({"positions": [{"id": "odd"}]}))
    assert "1 entry" in client.get("/api/positions").json()["warnings"][0]


def test_positions_on_the_command_line(capsys, tmp_path, monkeypatch):
    from app.__main__ import main

    assert main(["positions", "--demo"]) == 0
    out = capsys.readouterr().out
    assert "Positions   Data: SYNTHETIC DEMO DATA" in out and "What the rules say" in out
    assert "CRUX Dec 18 395 put, sold" in out and "+$477 (+57%)" in out and "profit target reached" in out
    assert "through the strike" in out and "earnings in 6d" in out and "inside time stop" in out
    assert "News on what is held" in out and "not advice" in out and "delayed" not in out

    assert main(["positions", "--demo", "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert len(document["holdings"]) == 5 and document["holdings"][0]["checks"][0]["state"] == "reached"

    # Adding and removing touch the file alone, so they need no keys.
    path = tmp_path / "positions.json"
    monkeypatch.setenv("POSITIONS_FILE", str(path))
    assert main(["positions", "add", "acme", "2026-12-18", "170", "put", "--sold", "--price", "2.40", "--qty", "3"]) == 0
    assert "ACME Dec 18 170 put, sold" in capsys.readouterr().out
    ((made,), unread) = PositionStore(path).load()
    assert (made.symbol, made.side, made.quantity, made.entry_price, unread) == ("ACME261218P00170000", "short", 3, 2.40, 0)
    assert main(["positions", "add", "ACME", "someday", "170", "put", "--bought", "--price", "1"]) == 2
    assert "expiry must be a date" in capsys.readouterr().err
    assert main(["positions", "remove", made.id]) == 0 and PositionStore(path).load() == ([], 0)
    assert main(["positions", "remove", made.id]) == 2
    assert main(["positions", "--demo", "remove", made.id]) == 2
    assert "kept in memory" in capsys.readouterr().err
    with pytest.raises(SystemExit):  # which side it is has to be said
        main(["positions", "add", "ACME", "2026-12-18", "170", "put", "--price", "1"])


def test_a_result_that_rounds_to_nothing_has_no_sign():
    # From the outside review: a loss of thirty cents printed as "+$0 (-0%)".
    from app.__main__ import _result

    blank = lambda pnl, pct: Holding(position(), 45, [], pnl=pnl, pnl_pct=pct)  # noqa: E731
    assert _result(blank(-0.30, -0.001)) == "$0 (0%)" and _result(blank(0.0, 0.0)) == "$0 (0%)"
    assert _result(blank(477.0, 0.569)) == "+$477 (+57%)" and _result(blank(-1110.6, -0.726)) == "-$1,111 (-73%)"
    assert _result(blank(None, None)) == "-"


def test_an_expired_position_is_not_given_its_stocks_news(scanner, capsys):
    # From the outside review. An expired position alone has no stock beside it. With another position open on
    # the same stock it picked that one's news chips up.
    from app.__main__ import print_positions

    gone = position(underlying="CRUX", strike=395.0, expiry=TODAY - timedelta(days=4), opened=TODAY - timedelta(days=40))
    live = position(underlying="CRUX", strike=395.0)
    review = scanner.review([gone, live])
    assert "split" in review.underlyings["CRUX"].recent_events
    print_positions(review)
    rows = [line for line in capsys.readouterr().out.splitlines() if "CRUX" in line and "395 put, sold" in line and "$" not in line[:60]]
    expired_row, live_row = rows[0], rows[1]
    assert expired_row.rstrip().endswith("expired") and "news:" not in expired_row
    assert "news: split" in live_row
