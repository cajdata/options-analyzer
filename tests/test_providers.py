"""Provider parsing, checked against response shapes from the services' own docs and samples."""

import json
from datetime import date, datetime, timedelta
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

    market = alpaca(handler)
    bars = market.daily_bars(["AAPL"], date(2026, 1, 1))["AAPL"]
    assert feeds == ["sip", "iex"]
    assert [(b.day, b.close) for b in bars] == [(date(2026, 1, 5), 100.5), (date(2026, 1, 6), 101.25)]
    assert len(market.notes) == 1 and "IEX" in market.notes[0]

    market.daily_bars(["AAPL"], date(2026, 1, 1))
    assert feeds == ["sip", "iex", "iex"]  # the fallback is remembered


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


# -- Alpha Vantage (fixtures are trimmed real responses) -----------------------


def alpha_vantage(handler) -> AlphaVantageEarnings:
    return AlphaVantageEarnings("demo", client=httpx.Client(transport=httpx.MockTransport(handler)), min_interval=0)


def test_alpha_vantage_calendar():
    text = (FIXTURES / "av_calendar.csv").read_text(encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["function"] == "EARNINGS_CALENDAR"
        assert request.url.params["apikey"] == "demo"
        return httpx.Response(200, text=text)

    calendar = alpha_vantage(handler).calendar()
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
    return EdgarEarnings("Test Person test@example.com", DiskCache(tmp_path), client=client)


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

    def calendar(self):
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
