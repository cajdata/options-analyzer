"""Provider parsing, checked against the services' documented shapes and against responses captured live."""

import json
from datetime import date, datetime, time as clock, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest

from app.cache import DiskCache
from app.models import EarningsEvent
from app.providers.alpaca import AlpacaData, AlpacaError
from app.providers.earnings import AlphaVantageEarnings, EarningsService, EdgarEarnings, RateLimited

FIXTURES = Path(__file__).parent / "fixtures"
ET = ZoneInfo("America/New_York")
WINDOW = dict(expiry_from=date(2026, 12, 1), expiry_to=date(2027, 1, 31), strike_from=170.0, strike_to=230.0)


def alpaca(handler, **kwargs) -> AlpacaData:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return AlpacaData("key-id", "secret", client=client, sleep=lambda _: None, **kwargs)


# -- Alpaca -------------------------------------------------------------------

SNAPSHOT = {
    "greeks": {"delta": 0.5123, "gamma": 0.0211, "rho": 0.0412, "theta": -0.0931, "vega": 0.1877},
    "impliedVolatility": 0.2843,
    "latestQuote": {"t": "2026-11-24T15:59:59.5Z", "bx": "C", "bp": 6.1, "bs": 12, "ap": 6.3, "as": 9, "ax": "A", "c": " "},
    "latestTrade": {"t": "2026-11-24T15:41:02Z", "x": "C", "p": 6.2, "s": 3, "c": "I"},
    "dailyBar": {"t": "2026-11-24T05:00:00Z", "o": 6.0, "h": 6.5, "l": 5.9, "c": 6.2, "v": 1412, "n": 210, "vw": 6.21},
}


def test_option_chain_parses_snapshots_and_follows_pages():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        assert request.url.path == "/v1beta1/options/snapshots/AAPL"
        if "page_token" not in request.url.params:
            page = {"snapshots": {"AAPL261218C00200000": SNAPSHOT}, "next_page_token": "abc"}
        else:
            page = {
                "snapshots": {
                    "AAPL261218P00200000": {"latestQuote": {"bp": 5.0, "ap": 5.2}},  # no greeks, no trades
                    "AAPL1261218C00200000": SNAPSHOT,  # adjusted contract: different root
                    "garbage": SNAPSHOT,
                },
                "next_page_token": None,
            }
        return httpx.Response(200, json=page)

    chain = alpaca(handler).option_chain("AAPL", **WINDOW)

    first = seen[0]
    assert first.headers["APCA-API-KEY-ID"] == "key-id" and first.headers["APCA-API-SECRET-KEY"] == "secret"
    assert first.url.params["feed"] == "indicative"
    assert first.url.params["expiration_date_gte"] == "2026-12-01"
    assert first.url.params["strike_price_lte"] == "230.0"
    assert seen[1].url.params["page_token"] == "abc"

    assert [q.symbol for q in chain] == ["AAPL261218C00200000", "AAPL261218P00200000"]
    call, put = chain
    assert (call.kind, call.strike, call.expiry) == ("call", 200.0, date(2026, 12, 18))
    assert (call.bid, call.ask, call.last, call.volume) == (6.1, 6.3, 6.2, 1412)
    assert (call.iv, call.delta, call.theta, call.vega) == (0.2843, 0.5123, -0.0931, 0.1877)
    assert call.quote_time.year == 2026
    assert (put.bid, put.ask, put.iv, put.delta, put.volume, put.open_interest) == (5.0, 5.2, None, None, None, None)


def test_open_interest_comes_from_the_trading_api():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "paper-api.alpaca.markets"
        assert request.url.path == "/v2/options/contracts"
        assert request.url.params["underlying_symbols"] == "AAPL"
        rows = [
            {"symbol": "AAPL261218C00200000", "open_interest": "6168", "open_interest_date": "2026-11-23"},
            {"symbol": "AAPL261218P00200000", "open_interest": None},
        ]
        return httpx.Response(200, json={"option_contracts": rows, "next_page_token": None})

    assert alpaca(handler).open_interest("AAPL", **WINDOW) == {"AAPL261218C00200000": 6168}


def test_live_keys_use_the_live_trading_host():
    hosts = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(200, json={"option_contracts": []})

    alpaca(handler, paper=False).open_interest("AAPL", **WINDOW)
    assert hosts == ["api.alpaca.markets"]


def test_stock_snapshots_fall_back_to_single_symbols_when_a_batch_is_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        symbols = request.url.params["symbols"].split(",")
        if len(symbols) > 1 or symbols == ["NOPE"]:
            return httpx.Response(400, json={"message": "invalid symbol"})
        snap = {"latestTrade": {"p": 201.5}, "dailyBar": {"c": 201.0}, "prevDailyBar": {"c": 199.0}}
        return httpx.Response(200, json={symbols[0]: snap})

    quotes = alpaca(handler).stock_snapshots(["AAPL", "NOPE"])
    assert list(quotes) == ["AAPL"]
    assert quotes["AAPL"].price == 201.5
    assert quotes["AAPL"].change_pct == pytest.approx(201.5 / 199.0 - 1)


def test_daily_bars_fall_back_to_iex_and_drop_todays_unfinished_bar():
    today = datetime.now(ET).date()
    feeds = []

    def handler(request: httpx.Request) -> httpx.Response:
        feeds.append(request.url.params["feed"])
        if request.url.params["feed"] == "sip":
            return httpx.Response(403, json={"message": "subscription does not permit querying recent SIP data"})
        assert request.url.params["adjustment"] == "all" and request.url.params["timeframe"] == "1Day"
        bars = [
            {"t": "2026-01-05T05:00:00Z", "o": 1, "h": 2, "l": 1, "c": 100.5, "v": 9000, "n": 1, "vw": 1},
            {"t": "2026-01-06T05:00:00Z", "o": 1, "h": 2, "l": 1, "c": 101.25, "v": 8000, "n": 1, "vw": 1},
            {"t": f"{today.isoformat()}T05:00:00Z", "o": 1, "h": 2, "l": 1, "c": 999.0, "v": 1, "n": 1, "vw": 1},
        ]
        return httpx.Response(200, json={"bars": {"AAPL": bars}, "next_page_token": None})

    midday = lambda: datetime.combine(today, clock(12, 0), ET)  # noqa: E731
    market = alpaca(handler, now=midday)
    bars = market.daily_bars(["AAPL"], date(2026, 1, 1))["AAPL"]
    assert feeds == ["sip", "iex"]
    assert [(b.day, b.close) for b in bars] == [(date(2026, 1, 5), 100.5), (date(2026, 1, 6), 101.25)]
    assert len(market.notes) == 1 and "IEX" in market.notes[0]

    # The fallback is remembered, and the caveat is repeated on every later scan,
    # not only the one that discovered it.
    market.notes.clear()
    market.daily_bars(["AAPL"], date(2026, 1, 1))
    assert feeds == ["sip", "iex", "iex"]
    assert len(market.notes) == 1 and "IEX" in market.notes[0]

    # Once the session is over, today's bar is final and belongs in the history.
    evening = lambda: datetime.combine(today, clock(17, 0), ET)  # noqa: E731
    closed = alpaca(handler, now=evening).daily_bars(["AAPL"], date(2026, 1, 1))["AAPL"]
    assert [b.close for b in closed] == [100.5, 101.25, 999.0]


def test_rate_limits_are_retried_and_other_errors_raise():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if len(calls) < 3:
            return httpx.Response(429, headers={"Retry-After": "1"}, json={"message": "too many requests"})
        return httpx.Response(200, json={"most_actives": [{"symbol": "NVDA", "volume": 1, "trade_count": 1}]})

    assert alpaca(handler).most_actives(5) == ["NVDA"]
    assert len(calls) == 3

    denied = alpaca(lambda request: httpx.Response(403, json={"message": "forbidden."}))
    with pytest.raises(AlpacaError) as err:
        denied.movers(5)
    assert err.value.status == 403 and "forbidden" in str(err.value)


def test_movers_combines_gainers_and_losers():
    body = {"gainers": [{"symbol": "UP", "percent_change": 12.0}], "losers": [{"symbol": "DOWN", "percent_change": -9.0}]}
    assert alpaca(lambda request: httpx.Response(200, json=body)).movers(5) == ["UP", "DOWN"]


# -- Alpaca, against responses captured live (the fixture is trimmed) -------------

LIVE = json.loads((FIXTURES / "alpaca_live.json").read_text(encoding="utf-8"))
COST_WINDOW = dict(expiry_from=date(2026, 12, 1), expiry_to=date(2026, 12, 31), strike_from=900.0, strike_to=1250.0)


def live(seen: list | None = None, **kwargs) -> AlpacaData:
    """Alpaca answering every request with the response captured for that path."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json=LIVE[request.url.path])

    return alpaca(handler, **kwargs)


def test_a_live_option_chain_parses_down_to_its_nanosecond_stamps():
    chain = {q.symbol: q for q in live().option_chain("COST", **COST_WINDOW)}
    call = chain["COST261204C01085000"]
    assert (call.kind, call.strike, call.expiry) == ("call", 1085.0, date(2026, 12, 4))
    assert (call.bid, call.ask, call.last, call.volume) == (17.55, 23.95, 22.4, 17)
    assert (call.iv, call.delta, call.gamma, call.theta, call.vega) == (0.2421, 0.566, 0.009, -0.924, 0.71)
    # Alpaca stamps to the nanosecond. The part finer than a microsecond is dropped.
    assert call.quote_time == datetime(2026, 11, 24, 20, 59, 59, 429737, tzinfo=timezone.utc)
    # A contract nobody bids for comes with no greeks or volatility, and one that
    # has never traded comes with no trade and no bars.
    no_bid, untraded = chain["COST261204C01165000"], chain["COST261204C01052500"]
    assert (no_bid.bid, no_bid.ask, no_bid.iv, no_bid.delta) == (0.0, 1.87, None, None)
    assert (untraded.bid, untraded.delta, untraded.last, untraded.volume) == (40.76, 0.7944, None, None)


def test_quotes_from_an_earlier_day_are_left_out_and_the_scan_says_so():
    market = live()
    chain = market.option_chain("COST", **COST_WINDOW)
    # A special dividend changed the strikes that morning. Two contracts that had
    # stopped existing still answered with the previous close, one of them near the
    # money. Set beside the new stock price, a quote like that can look like a bargain.
    assert [q.symbol for q in chain] == [
        "COST261204C01085000",
        "COST261204C01165000",
        "COST261204C00930000",
        "COST261204C01052500",
    ]
    assert market.notes == ["COST: 2 contracts were left out because the latest quote is from an earlier day."]


def test_volume_counts_only_trades_from_the_latest_session():
    chain = {q.symbol: q for q in live().option_chain("COST", **COST_WINDOW)}
    assert chain["COST261204C01085000"].volume == 17  # traded that day
    assert chain["COST261204C00930000"].volume == 0  # its daily bar is the last day it traded, four days before
    assert chain["COST261204C01052500"].volume is None  # has never traded


def test_a_quote_before_the_open_does_not_make_the_last_close_old():
    def chain(*stamps):
        snaps = {
            f"AAPL261218C00{200 + n}000": {"latestQuote": {"bp": 5.0, "ap": 5.2, "t": stamp}}
            for n, stamp in enumerate(stamps)
        }
        market = alpaca(lambda request: httpx.Response(200, json={"snapshots": snaps, "next_page_token": None}))
        return [q.symbol for q in market.option_chain("AAPL", **WINDOW)], market.notes

    # In New York these are 15:59 on the 23rd, then 08:05 and 09:31 on the 24th.
    close, early, opened = "2026-11-23T20:59:59.5Z", "2026-11-24T13:05:00Z", "2026-11-24T14:31:00Z"

    # Before the open, the session that counts is still the day before.
    quotes, notes = chain(close, close, early)
    assert len(quotes) == 3 and notes == []
    # Once trading has started, a contract still showing the last close has not been quoted today.
    quotes, notes = chain(close, close, early, opened)
    assert quotes == ["AAPL261218C00202000", "AAPL261218C00203000"]
    assert notes == ["AAPL: 2 contracts were left out because the latest quote is from an earlier day."]
    quotes, notes = chain(close, opened)
    assert notes == ["AAPL: 1 contract was left out because the latest quote is from an earlier day."]
    # With no time stamps there is nothing to judge by, so nothing is left out.
    quotes, notes = chain(None, None)
    assert len(quotes) == 2 and notes == []


def test_live_contract_rows_give_open_interest_as_whole_numbers():
    # The figure arrives as text. A contract listed since Alpaca's last update has none.
    assert live().open_interest("COST", **COST_WINDOW) == {
        "COST261204C01085000": 91,
        "COST261204C01165000": 61,
        "COST261204C00930000": 6,
    }


def test_live_stock_snapshot_and_price_history():
    seen = []
    noon = lambda: datetime(2026, 11, 24, 12, 0, tzinfo=ET)  # noqa: E731
    market = live(seen, now=noon)
    quote = market.stock_snapshots(["COST"])["COST"]
    assert (quote.price, quote.prev_close) == (1091.44, 1097.8)

    bars = market.daily_bars(["COST"], date(2023, 8, 12))["COST"]
    # A bar is stamped at midnight in New York: 04:00 or 05:00 UTC by season. The
    # fourth bar is the 24th itself, unfinished at noon.
    assert [(b.day, b.close) for b in bars] == [
        (date(2023, 8, 14), 523.08),
        (date(2026, 7, 1), 896.65),
        (date(2026, 11, 23), 1067.91),
    ]
    # The free plan answered the request for consolidated history, so no IEX caveat.
    assert seen[-1].url.params["feed"] == "sip" and market.notes == []


def test_live_screener_lists():
    market = live()
    assert market.most_actives(20) == ["SOXL", "NVDA"]
    assert market.movers(20) == ["JAGX", "BENF", "FGIWW", "EVRL"]


# -- Alpha Vantage (fixtures are trimmed real responses) -----------------------


def alpha_vantage(handler, **kwargs) -> AlphaVantageEarnings:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return AlphaVantageEarnings("demo", client=client, min_interval=0, **kwargs)


def test_alpha_vantage_calendar():
    text = (FIXTURES / "av_calendar.csv").read_text(encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["function"] == "EARNINGS_CALENDAR"
        assert request.url.params["apikey"] == "demo"
        return httpx.Response(200, text=text)

    calendar = alpha_vantage(handler).calendar(date(2026, 11, 24))
    assert calendar["ADI"] == EarningsEvent(date(2026, 11, 24), "pre", source="Alpha Vantage")
    assert calendar["ADSK"].timing == "post"
    assert calendar["AFJK"].timing == "unknown"  # the calendar leaves the time blank


def test_alpha_vantage_history():
    payload = json.loads((FIXTURES / "av_earnings_ibm.json").read_text(encoding="utf-8"))
    events = alpha_vantage(lambda request: httpx.Response(200, json=payload)).history("IBM")
    assert len(events) == 6
    assert events[0].day == date(2026, 10, 21)
    assert events[0].timing == "post"
    assert events[0].surprise_pct == pytest.approx(2.4306)


def test_alpha_vantage_reports_its_request_limit_once_then_stops_calling():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json={"Information": "rate limit is 25 requests per day"})

    source = alpha_vantage(handler)
    for _ in range(3):
        with pytest.raises(RateLimited):
            source.history("IBM")
    assert len(calls) == 1


# -- SEC EDGAR ------------------------------------------------------------------


def edgar(tmp_path, submissions: dict, seen: list | None = None) -> EdgarEarnings:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if request.url.path == "/files/company_tickers.json":
            return httpx.Response(200, json={"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}})
        assert request.url.path == "/submissions/CIK0000320193.json"
        return httpx.Response(200, json=submissions)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return EdgarEarnings("Test Person test@example.com", DiskCache(tmp_path), client=client, sleep=lambda _: None)


def test_edgar_history_from_a_real_filing_index(tmp_path):
    submissions = json.loads((FIXTURES / "edgar_aapl.json").read_text(encoding="utf-8"))
    seen = []
    events = edgar(tmp_path, submissions, seen).history("AAPL")

    assert seen[0].headers["User-Agent"] == "Test Person test@example.com"
    assert len(events) == 12
    assert [e.day for e in events[:3]] == [date(2026, 11, 2), date(2026, 7, 30), date(2026, 4, 30)]
    assert {e.timing for e in events} == {"post"}  # Apple reports after the close
    assert events == sorted(events, key=lambda e: e.day, reverse=True)


def test_edgar_timing_uses_the_acceptance_time_in_new_york(tmp_path):
    table = {
        "form": ["8-K", "8-K", "8-K", "10-Q", "8-K"],
        "items": ["2.02,9.01", "2.02", "2.02,9.01", "", "5.02"],
        "acceptanceDateTime": [
            "2026-10-15T11:05:00.000Z",  # 07:05 in New York: before the open
            "2026-07-16T22:10:00.000Z",  # 18:10: after the close (EDGAR dates the filing the next day)
            "2026-04-16T17:00:00.000Z",  # 13:00: during the session, moves the stock that day
            "2026-04-20T20:30:00.000Z",
            "2026-03-01T21:30:00.000Z",  # an 8-K that is not about results
        ],
    }
    events = edgar(tmp_path, {"filings": {"recent": table, "files": []}}).history("AAPL")
    assert [(e.day, e.timing) for e in events] == [
        (date(2026, 10, 15), "pre"),
        (date(2026, 7, 16), "post"),
        (date(2026, 4, 16), "pre"),
    ]


def test_edgar_unknown_ticker_and_cached_ticker_table(tmp_path):
    seen = []
    source = edgar(tmp_path, {"filings": {"recent": {}, "files": []}}, seen)
    assert source.history("ZZZZ") == []
    assert source.history("ZZZZ") == []
    assert [r.url.path for r in seen] == ["/files/company_tickers.json"]  # fetched once, then cached


# -- Earnings service -----------------------------------------------------------

TODAY = date(2026, 11, 24)


class FakeSource:
    def __init__(self, name, events=(), error=None, calendar=None):
        self.name, self.events, self.error, self.listed, self.calls = name, list(events), error, calendar or {}, 0

    def history(self, symbol):
        self.calls += 1
        if self.error:
            raise self.error
        return self.events

    def calendar(self, today=None):
        self.calls += 1
        if self.error:
            raise self.error
        return self.listed


def quarterly(count, last=date(2026, 10, 29), **kwargs):
    return [EarningsEvent(last - timedelta(days=91 * n), "post", **kwargs) for n in range(count)]


def test_history_is_cached_between_services(tmp_path):
    source = FakeSource("A", quarterly(8))
    first = EarningsService(DiskCache(tmp_path), history_sources=[source], today=TODAY)
    assert first.history("AAPL") == (quarterly(8), None)
    second = EarningsService(DiskCache(tmp_path), history_sources=[source], today=TODAY)
    assert second.history("AAPL") == (quarterly(8), None)
    assert source.calls == 1


def test_cached_history_is_refreshed_once_a_new_report_is_due(tmp_path):
    cache, source = DiskCache(tmp_path), FakeSource("A", quarterly(8))
    EarningsService(cache, history_sources=[source], today=TODAY).history("AAPL")
    # Three months on, the newest report on file is old enough that another has surely happened.
    stale_path = next(tmp_path.glob("earnings_history_AAPL*"))
    raw = json.loads(stale_path.read_text())
    raw["stored_at"] = (datetime.now(ET) - timedelta(days=2)).isoformat()
    stale_path.write_text(json.dumps(raw))
    EarningsService(cache, history_sources=[source], today=date(2027, 2, 20)).history("AAPL")
    assert source.calls == 2


def test_second_source_fills_in_when_the_first_has_too_little(tmp_path):
    thin, full = FakeSource("EDGAR", quarterly(1)), FakeSource("AV", quarterly(8, surprise_pct=1.5))
    service = EarningsService(DiskCache(tmp_path), history_sources=[thin, full], today=TODAY)
    events, note = service.history("JPM")
    assert len(events) == 8 and events[0].surprise_pct == 1.5 and note is None
    assert service.sources == ["EDGAR", "AV"]


def test_a_rate_limited_source_leaves_a_note_and_is_not_cached(tmp_path):
    limited = FakeSource("Alpha Vantage", error=RateLimited("limit"))
    service = EarningsService(DiskCache(tmp_path), history_sources=[limited], today=TODAY)
    events, note = service.history("AAPL")
    assert events == [] and "request limit" in note
    assert list(tmp_path.iterdir()) == []
    assert EarningsService(DiskCache(tmp_path), today=TODAY).history("AAPL")[1] == "No earnings data source is configured."


def test_upcoming_prefers_the_calendar_and_otherwise_estimates(tmp_path):
    listed = EarningsEvent(date(2026, 12, 3), "post", source="cal")
    calendar = FakeSource("cal", calendar={"AAPL": listed, "OLD": EarningsEvent(date(2026, 11, 1))})
    service = EarningsService(DiskCache(tmp_path), calendar_source=calendar, today=TODAY)
    history = quarterly(6)
    assert service.upcoming("AAPL", history) == listed
    for symbol in ("MSFT", "OLD"):  # not listed, or listed with a date already past
        guess = service.upcoming(symbol, history)
        assert (guess.day, guess.confirmed) == (date(2027, 1, 28), False)
    assert service.upcoming("ETF", []) is None
    assert calendar.calls == 1  # one calendar download serves every symbol


def test_calendar_failure_is_reported_not_raised(tmp_path):
    broken = FakeSource("Alpha Vantage", error=RateLimited("limit"))
    service = EarningsService(DiskCache(tmp_path), calendar_source=broken, today=TODAY)
    assert service.calendar() == {}
    assert "estimates" in service.calendar_note


# -- Found in review: a server left running ------------------------------------


def test_the_service_follows_the_date_instead_of_freezing_it(tmp_path):
    listed = EarningsEvent(date(2026, 12, 3), "post", source="cal")
    calendar = FakeSource("cal", calendar={"AAPL": listed})
    service = EarningsService(DiskCache(tmp_path), calendar_source=calendar, today=TODAY)
    history = quarterly(6)
    assert service.upcoming("AAPL", history) == listed
    # The day after the report, a server that never restarted must stop announcing it.
    service._fixed_today = date(2026, 12, 4)
    assert service.upcoming("AAPL", history).day == date(2027, 1, 28)
    # With no date fixed, the service reads the clock every time it is asked.
    assert EarningsService(DiskCache(tmp_path)).today == datetime.now(ET).date()


def test_a_failed_calendar_download_is_retried_and_the_note_clears(tmp_path):
    now = [0.0]
    source = FakeSource("Alpha Vantage", error=RateLimited("limit"), calendar={"AAPL": EarningsEvent(date(2026, 12, 3))})
    service = EarningsService(DiskCache(tmp_path), calendar_source=source, today=TODAY, clock=lambda: now[0])
    assert service.calendar() == {} and service.calendar_note
    assert service.calendar() == {} and source.calls == 1  # not hammered while it is failing
    source.error = None
    now[0] += 16 * 60  # a quarter of an hour later the download works
    assert "AAPL" in service.calendar() and service.calendar_note is None
    assert source.calls == 2


def test_alpha_vantage_tries_again_after_its_cool_down():
    payload = json.loads((FIXTURES / "av_earnings_ibm.json").read_text(encoding="utf-8"))
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        body = {"Note": "burst limit reached"} if len(calls) == 1 else payload
        return httpx.Response(200, json=body)

    source = alpha_vantage(handler, cool_down=0.0)
    with pytest.raises(RateLimited):
        source.history("IBM")
    assert len(source.history("IBM")) == 6  # the limit cleared and so did the block
    assert len(calls) == 2


def test_a_stock_with_no_reports_is_not_looked_up_again_each_day(tmp_path):
    etf = FakeSource("SEC EDGAR", [])
    for _ in range(2):
        events, note = EarningsService(DiskCache(tmp_path), history_sources=[etf], today=TODAY).history("SPY")
        assert events == [] and note is None
    path = next(tmp_path.glob("earnings_history_SPY*"))
    raw = json.loads(path.read_text())
    raw["stored_at"] = (datetime.now(ET) - timedelta(days=2)).isoformat()
    path.write_text(json.dumps(raw))
    EarningsService(DiskCache(tmp_path), history_sources=[etf], today=TODAY).history("SPY")
    assert etf.calls == 1  # a scarce request is not spent on the same ETF every day


def test_adding_a_source_does_not_reuse_answers_from_the_old_set(tmp_path):
    thin = FakeSource("SEC EDGAR", [])
    EarningsService(DiskCache(tmp_path), history_sources=[thin], today=TODAY).history("SHOP")
    full = FakeSource("Alpha Vantage", quarterly(8))
    events, _ = EarningsService(DiskCache(tmp_path), history_sources=[thin, full], today=TODAY).history("SHOP")
    assert len(events) == 8


# -- Found in review: hardening --------------------------------------------------


def test_alpha_vantage_calendar_skips_past_dates_and_malformed_rows():
    text = (
        "symbol,name,reportDate,fiscalDateEnding,estimate,currency,timeOfTheDay\n"
        "AAA,Alpha,2026-11-20,2026-09-30,1.0,USD,pre-market\n"  # already past
        "AAA,Alpha,2027-02-19,2026-12-31,1.1,USD,post-market\n"
        "BBB\n"  # a short row: no date field at all
        "CCC,Gamma,not-a-date,,,USD,\n"
    )
    calendar = alpha_vantage(lambda request: httpx.Response(200, text=text)).calendar(date(2026, 11, 24))
    assert set(calendar) == {"AAA"}
    assert (calendar["AAA"].day, calendar["AAA"].timing) == (date(2027, 2, 19), "post")


def test_alpha_vantage_history_is_sorted_newest_first():
    rows = [
        {"reportedDate": "2026-01-20", "reportTime": "post-market"},
        {"reportedDate": "2026-10-21", "reportTime": "post-market"},
        {"reportedDate": None},
        {"reportedDate": "2026-07-22"},
    ]
    source = alpha_vantage(lambda request: httpx.Response(200, json={"quarterlyEarnings": rows}))
    assert [e.day for e in source.history("IBM")] == [date(2026, 10, 21), date(2026, 7, 22), date(2026, 1, 20)]


def test_edgar_backs_off_and_retries_when_told_to_slow_down(tmp_path):
    submissions = json.loads((FIXTURES / "edgar_aapl.json").read_text(encoding="utf-8"))
    statuses, waits = iter([429, 503, 200]), []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/files/company_tickers.json":
            return httpx.Response(200, json={"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}})
        status = next(statuses)
        return httpx.Response(status, json=submissions if status == 200 else {})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    source = EdgarEarnings("Test Person test@example.com", DiskCache(tmp_path), client=client, sleep=waits.append)
    assert len(source.history("AAPL")) == 12
    assert [w for w in waits if w >= 1] == [2.0, 4.0]  # two back-offs, then through


def test_edgar_reads_item_codes_in_any_shape(tmp_path):
    stamps = ["2026-10-15T21:00:00.000Z", "2026-07-16T21:00:00.000Z", "2026-04-16T21:00:00.000Z",
              "2026-01-15T21:00:00.000Z", "2025-10-15T21:00:00.000Z"]
    table = {"form": ["8-K"] * 5, "items": ["2.02, 9.01", ["2.02", "9.01"], None, "", "9.01"], "acceptanceDateTime": stamps}
    events = edgar(tmp_path, {"filings": {"recent": table, "files": []}}).history("AAPL")
    assert [e.day for e in events] == [date(2026, 10, 15), date(2026, 7, 16)]


def test_dropped_connections_are_retried():
    calls = []

    def flaky(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ConnectError("connection reset")
        return httpx.Response(200, json={"most_actives": [{"symbol": "NVDA"}]})

    assert alpaca(flaky).most_actives(5) == ["NVDA"]

    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(AlpacaError, match="could not reach Alpaca"):
        alpaca(dead).most_actives(5)


def test_a_stock_with_no_trade_yet_today_uses_the_previous_close():
    snaps = {
        "THIN": {"latestTrade": None, "dailyBar": None, "prevDailyBar": {"c": 41.5}},
        "GONE": {"latestTrade": None, "dailyBar": None, "prevDailyBar": None},
    }
    market = alpaca(lambda request: httpx.Response(200, json=snaps))
    quotes = market.stock_snapshots(["THIN", "GONE"])
    assert list(quotes) == ["THIN"]
    assert (quotes["THIN"].price, quotes["THIN"].change_pct) == (41.5, 0.0)
    assert "previous close" in market.notes[0]


def test_cache_writes_are_whole_and_never_fatal(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    cache.set("a/b", {"x": 1})
    assert cache.get("a/b", timedelta(minutes=1)) == {"x": 1}
    assert [p.suffix for p in (tmp_path / "cache").iterdir()] == [".json"]  # no temporary file left behind

    blocker = tmp_path / "not-a-folder"
    blocker.write_text("x")
    broken = DiskCache(blocker / "cache")
    broken.set("key", 1)  # cannot be written: skipped, not raised
    assert broken.entry("key") is None


def test_cache_survives_threads_writing_the_same_key(tmp_path):
    import threading

    cache, errors = DiskCache(tmp_path), []

    def hammer(n: int) -> None:
        try:
            for i in range(40):
                cache.set("shared", {"writer": n, "i": i})
                found = cache.entry("shared")
                assert found is None or set(found[0]) == {"writer", "i"}  # never half a file
        except Exception as err:  # pragma: no cover - the failure this guards against
            errors.append(err)

    threads = [threading.Thread(target=hammer, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert cache.entry("shared") is not None
    assert not list(tmp_path.glob("*.tmp"))


# -- Found in the follow-up review ------------------------------------------------


def test_edgar_tells_a_foreign_filer_from_a_security_with_no_reports(tmp_path):
    from app.providers.earnings import Unsupported

    def filings(forms):
        stamps = ["2026-10-15T21:00:00.000Z"] * len(forms)
        return {"filings": {"recent": {"form": forms, "items": [""] * len(forms), "acceptanceDateTime": stamps}, "files": []}}

    # A foreign company (the real index for TSM is hundreds of 6-Ks and no 8-K at all).
    with pytest.raises(Unsupported, match="Form 6-K"):
        edgar(tmp_path, filings(["6-K", "6-K", "20-F", "4"])).history("AAPL")
    # An ETF files neither, and genuinely has no earnings reports.
    assert edgar(tmp_path, filings(["497", "NPORT-P", "N-CSR"])).history("AAPL") == []
    # A company that moved from foreign to US forms is read from its 8-Ks.
    moved = filings(["8-K", "6-K", "10-K"])
    moved["filings"]["recent"]["items"] = ["2.02,9.01", "", ""]
    assert len(edgar(tmp_path, moved).history("AAPL")) == 1


def test_a_source_that_cannot_answer_leaves_a_note_and_defers_to_the_next(tmp_path):
    from app.providers.earnings import Unsupported

    cannot = FakeSource("SEC EDGAR", error=Unsupported("SEC EDGAR cannot date earnings for a company that reports on Form 6-K."))
    alone = EarningsService(DiskCache(tmp_path / "a"), history_sources=[cannot], today=TODAY)
    events, note = alone.history("TSM")
    assert events == [] and "Form 6-K" in note  # reported as unknown, not as "no reports"
    assert not list((tmp_path / "a").glob("*"))  # and not cached as an empty answer

    other = FakeSource("Alpha Vantage", quarterly(8))
    both = EarningsService(DiskCache(tmp_path / "b"), history_sources=[cannot, other], today=TODAY)
    events, note = both.history("TSM")
    assert len(events) == 8 and note is None


def test_threads_queued_behind_a_rate_limited_request_do_not_fire(tmp_path):
    import threading
    import time as real_time

    calls, outcomes = [], []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        real_time.sleep(0.05)  # long enough for the other threads to queue up behind this one
        return httpx.Response(200, json={"Information": "rate limit is 25 requests per day"})

    source = alpha_vantage(handler)

    def ask() -> None:
        try:
            source.history("IBM")
        except RateLimited:
            outcomes.append("limited")

    threads = [threading.Thread(target=ask) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes == ["limited"] * 4
    assert len(calls) == 1  # the three that were waiting saw the block and stood down
