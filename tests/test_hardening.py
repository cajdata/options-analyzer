"""Things an outside review found after phase 2: spoiled cache files, stale and partial answers, and odd provider replies.

Each case here was first reproduced against the code as a failure.
"""

import json
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest

from app.analysis import earnings as earnings_math
from app.cache import DiskCache
from app.config import Settings, tidy
from app.models import EarningsEvent
from app.pool import EventPool
from app.providers.alpaca import AlpacaData, AlpacaError, _option_quote
from app.providers.earnings import EarningsService, EdgarEarnings, RateLimited, Unsupported, _accepted

TODAY = date(2026, 11, 24)


# -- The cache ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "written",
    [
        "[]",  # valid JSON of the wrong shape
        json.dumps({"stored_at": "2026-11-24T10:00:00+00:00"}),  # no value
        json.dumps({"stored_at": "2026-11-24T10:00:00", "value": 1}),  # a time with no zone
        json.dumps({"stored_at": "not a time", "value": 1}),
        json.dumps({"value": 1}),
        "{half a file",
    ],
)
def test_a_cache_file_that_does_not_read_as_an_entry_is_a_miss(tmp_path, written):
    (tmp_path / "key.json").write_text(written)
    cache = DiskCache(tmp_path)
    assert cache.entry("key") is None and cache.get("key", timedelta(days=1)) is None
    # And it can be written over like any other key.
    cache.set("key", {"fine": True})
    assert cache.entry("key")[0] == {"fine": True}


def test_one_spoiled_record_does_not_empty_the_pool(tmp_path):
    pool = EventPool(DiskCache(tmp_path))
    pool.add("GOOD", "light", {"financing": [4, 2, 1, 3, 80, 9]})
    (tmp_path / "pool_SPOILED.json").write_text("[]")
    (tmp_path / "pool_NOZONE.json").write_text(json.dumps({"stored_at": "2026-11-24T10:00:00", "value": {"group": "light", "kinds": {}}}))
    assert pool.totals()["light"] == {"stocks": 1, "kinds": {"financing": {"counts": [4, 2, 1, 3, 80, 9], "stocks": 1}}}


@pytest.mark.parametrize(
    "row",
    [
        [25, 5, 10, 15, 0, 0],  # event days with no sessions to compare them with
        [4, 5, 1, 3, 80, 9],  # more big days than days
        [4, 2, 3, 3, 80, 9],  # more up and down days than days
        [4, 2, 1, 3, 80, 90],  # more big sessions than sessions
        [True, False, True, False, True, False],  # truth values are not counts
        [4, 2, 1, 3, 80.0, 9],
    ],
)
def test_a_tally_whose_counts_disagree_is_passed_over(tmp_path, row):
    cache = DiskCache(tmp_path)
    cache.set("pool_ODD", {"group": "light", "kinds": {"financing": row}})
    assert EventPool(cache).totals()["light"] == {"stocks": 0, "kinds": {}}


def test_a_pooled_line_with_nothing_to_compare_with_still_prints():
    from app.__main__ import _pooled_line
    from app.models import PooledStudy

    line = _pooled_line(PooledStudy("financing", "Stock or debt offering", "light", 1, 1, 25, 5, 10, 15, 0, 0))
    assert line == 'across 1 lightly covered stock, 25 "financing" days: 20% moved over 2x the stock\'s usual day, 10 up 15 down'


# -- Earnings history -----------------------------------------------------------------


class Source:
    def __init__(self, answer, name="SEC EDGAR", rate_limited=False):
        self.answer, self.name, self.rate_limited, self.calls = answer, name, rate_limited, 0

    def history(self, symbol):
        self.calls += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def reports(count: int, newest: date = date(2026, 11, 3)) -> list[EarningsEvent]:
    return [EarningsEvent(day=newest - timedelta(days=91 * n), timing="post", source="SEC EDGAR") for n in range(count)]


def age(path, days: int) -> None:
    raw = json.loads(path.read_text())
    raw["stored_at"] = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    path.write_text(json.dumps(raw))


def test_a_short_answer_is_not_kept_when_a_source_could_not_be_asked(tmp_path):
    few, limited = Source(reports(3)), Source(RateLimited("limit"), name="Alpha Vantage")
    service = EarningsService(DiskCache(tmp_path), history_sources=[few, limited], today=TODAY)
    # Three reports are used for this scan. They are not kept: the second source might have had twelve.
    assert service.history("ACME") == (reports(3), None)
    assert list(tmp_path.glob("earnings_history_*")) == []
    # So the next scan asks again, and once the limit has passed the fuller answer is the one kept.
    limited.answer = reports(12)
    assert len(service.history("ACME")[0]) == 12 and limited.calls == 2
    assert len(service.history("ACME")[0]) == 12 and limited.calls == 2  # now from the file

    # The same goes for a source that failed outright.
    down = Source(httpx.ConnectError("down"), name="Alpha Vantage")
    other = EarningsService(DiskCache(tmp_path / "b"), history_sources=[Source(reports(3)), down], today=TODAY)
    assert len(other.history("ACME")[0]) == 3 and list((tmp_path / "b").glob("*")) == []


def test_a_short_answer_is_kept_when_no_source_was_cut_short(tmp_path):
    # A source that cannot answer for this company at all will not answer tomorrow either. Asking the
    # other one on every scan would spend its few requests a day for nothing.
    foreign = Source(Unsupported("SEC EDGAR cannot date earnings for a company that reports on Form 6-K."))
    second = Source(reports(5), name="Alpha Vantage")
    service = EarningsService(DiskCache(tmp_path), history_sources=[foreign, second], today=TODAY)
    assert len(service.history("TSM")[0]) == 5 and len(service.history("TSM")[0]) == 5
    assert second.calls == 1
    # Enough reports are kept whatever happened to a later source.
    full = EarningsService(DiskCache(tmp_path / "b"), history_sources=[Source(reports(8)), Source(RateLimited("x"), name="AV")], today=TODAY)
    assert len(full.history("ACME")[0]) == 8 and len(list((tmp_path / "b").glob("*"))) == 1


def test_reports_on_file_that_do_not_read_as_reports_are_asked_for_again(tmp_path):
    cache = DiskCache(tmp_path)
    source = Source(reports(8))
    service = EarningsService(cache, history_sources=[source], today=TODAY)
    for spoiled in ([{"no": "day"}], [{"day": "not a date"}], "a string", [["a list"]], {"a": "dict"}):
        cache.set("earnings_history_ACME_sec", spoiled)
        assert service.history("ACME") == (reports(8), None)  # not an error, and not "no reports"
    cache.set("earnings_history_BOLT_sec", [{"no": "day"}])
    assert service.reports("BOLT") == reports(8)


def test_stale_reports_are_not_used_for_a_pooled_stock(tmp_path):
    cache = DiskCache(tmp_path)
    source = Source(reports(8))
    service = EarningsService(cache, history_sources=[source], today=TODAY)
    # On file from a scan months ago: its newest report is from last year.
    cache.set("earnings_history_ACME_sec", [{"day": "2025-08-05", "timing": "post"}])
    age(tmp_path / "earnings_history_ACME_sec.json", 200)
    # Used as it is, this year's report days would not be left out, and their moves would count as events.
    assert service.reports("ACME") == reports(8) and source.calls == 1
    assert json.loads((tmp_path / "earnings_history_ACME_sec.json").read_text())["value"] == [{"day": "2025-08-05", "timing": "post"}]  # kept nowhere

    # A fresh "no reports" answers for a fund without a request each time. An old one is asked about again.
    cache.set("earnings_history_ETF_sec", [])
    assert service.reports("ETF") == [] and source.calls == 1
    age(tmp_path / "earnings_history_ETF_sec.json", 45)
    assert service.reports("ETF") == reports(8) and source.calls == 2

    # Stale reports, and nobody to ask: nothing, so the stock is passed over.
    alone = EarningsService(cache, history_sources=[Source(reports(8), rate_limited=True)], today=TODAY)
    assert alone.reports("ACME") == []


def test_the_usual_report_time_is_settled_the_same_way_every_run():
    # Two before the open and two after it, newest first. The newest decides, whatever order a set would give.
    history = [
        EarningsEvent(day=date(2026, 11, 3) - timedelta(days=91 * n), timing=timing, source="t")
        for n, timing in enumerate(["post", "pre", "pre", "post"])
    ]
    guesses = {earnings_math.estimate_next(history, TODAY).timing for _ in range(20)}
    assert guesses == {"post"}
    flipped = [EarningsEvent(day=e.day, timing="pre" if e.timing == "post" else "post", source="t") for e in history]
    assert earnings_math.estimate_next(flipped, TODAY).timing == "pre"


# -- Providers ------------------------------------------------------------------------


def alpaca(handler, slept=None) -> AlpacaData:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return AlpacaData("key-id", "secret", client=client, sleep=(slept.append if slept is not None else lambda _: None))


@pytest.mark.parametrize("header, waited", [("-5", 2.0), ("nan", 2.0), ("inf", 2.0), ("soon", 2.0), ("7", 7.0), ("900", 30.0)])
def test_a_wait_the_server_asks_for_is_a_sane_number_of_seconds(header, waited):
    replies = [httpx.Response(429, headers={"Retry-After": header}), httpx.Response(200, json={"most_actives": []})]
    slept = []
    assert alpaca(lambda request: replies.pop(0), slept).most_actives(5) == []
    assert slept == [waited]


@pytest.mark.parametrize("body", ['["not", "an", "object"]', '"just text"', "not json at all", '{"message": "forbidden"}'])
def test_an_error_reply_of_any_shape_is_reported_not_raised_as_something_else(body):
    with pytest.raises(AlpacaError) as err:
        alpaca(lambda request: httpx.Response(403, text=body)).most_actives(5)
    assert err.value.status == 403 and ("forbidden" in str(err.value) or body.strip('"')[:8] in str(err.value))


@pytest.mark.parametrize("stamp, has_time", [("2026-11-24T20:59:59Z", True), ("2026-11-24T20:59:59", False), (20261124, False), ("", False), (None, False)])
def test_an_option_quote_keeps_only_a_time_with_a_zone(stamp, has_time):
    snap = {"latestQuote": {"bp": 1.0, "ap": 1.2, "t": stamp}, "greeks": {}}
    quote = _option_quote("ACME", "ACME", "ACME261218C00100000", snap)
    assert quote is not None and (quote.quote_time is not None) is has_time


def test_edgar_tries_again_after_a_dropped_connection(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow")
        return httpx.Response(200, json={"0": {"ticker": "ACME", "cik_str": 1, "title": "Acme Corp"}})

    slept = []
    edgar = EdgarEarnings("test agent", DiskCache(tmp_path), client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=slept.append)
    assert edgar.company_name("ACME") == "Acme Corp" and len(calls) == 2 and 2.0 in slept

    # After the last try the failure is the caller's to handle, as it always was.
    def always(request):
        raise httpx.ConnectError("down")

    dead = EdgarEarnings("test agent", DiskCache(tmp_path / "b"), client=httpx.Client(transport=httpx.MockTransport(always)), sleep=lambda _: None)
    with pytest.raises(httpx.ConnectError):
        dead.company_name("ACME")


def test_a_filing_time_with_no_zone_is_no_time():
    assert _accepted("2026-11-24T21:05:12.000Z").hour == 16  # New York time
    assert _accepted("2026-11-24T21:05:12") is None and _accepted("yesterday") is None and _accepted(None) is None


def test_edgar_notes_the_names_a_company_had_before(tmp_path):
    def handler(request):
        if request.url.path.endswith("company_tickers.json"):
            return httpx.Response(200, json={"0": {"ticker": "SRTM", "cik_str": 77, "title": "Stratum Holdings Corp."}})
        former = [{"name": "La Rosa Holdings Corp.", "from": "2021-08-01", "to": "2026-09-30"}, {"name": ""}, "odd", {"name": "Older Name Inc"}]
        return httpx.Response(200, json={"formerNames": former, "filings": {"recent": {"form": ["8-K"], "items": ["5.02"], "acceptanceDateTime": ["2026-11-20T21:05:12.000Z"], "accessionNumber": ["0001-26-000001"], "primaryDocument": ["a.htm"]}}})

    edgar = EdgarEarnings("test agent", DiskCache(tmp_path), client=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda _: None)
    assert edgar.former_names("SRTM") == []  # nothing on file yet, and it asks for nothing
    assert len(edgar.filings("SRTM")) == 1
    assert edgar.former_names("SRTM") == ["La Rosa Holdings Corp.", "Older Name Inc"]
    # Read back by another instance, with no request.
    again = EdgarEarnings("test agent", DiskCache(tmp_path), client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))), sleep=lambda _: None)
    assert again.former_names("SRTM") == ["La Rosa Holdings Corp.", "Older Name Inc"]


# -- Symbols and aliases ----------------------------------------------------------------


def test_symbols_are_tidied_the_same_way_wherever_they_come_from(tmp_path):
    assert tidy([" brk-b ", "BRK.B", "aapl", "", "AAPL", "nvda"]) == ["BRK.B", "AAPL", "NVDA"]
    universe = tmp_path / "universe.txt"
    universe.write_text("# a comment\nbrk-b\nAAPL  # trailing\n\naapl\n")
    assert Settings(universe_file=universe).universe() == ["BRK.B", "AAPL"]
    assert Settings(universe_file=tmp_path / "missing.txt").universe() == []

    import os

    from app import config

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(config, "load_dotenv", lambda *a, **k: None)
        patch.setitem(os.environ, "WATCHLIST", "brk-b, AAPL; aapl ,BRK.B")
        assert config.load_settings().watchlist == ("BRK.B", "AAPL")


def test_other_names_a_company_goes_by_are_read_from_a_file(tmp_path):
    listed = tmp_path / "aliases.txt"
    listed.write_text("# brands\nC: Citibank\nnu : Nubank # the bank\nMETA: Facebook, Instagram ,\nno colon here\nEMPTY:\n: nameless\nbrk-b: Berkshire\n")
    assert Settings(aliases_file=listed).aliases() == {
        "C": ("Citibank",),
        "NU": ("Nubank",),
        "META": ("Facebook", "Instagram"),
        "BRK.B": ("Berkshire",),
    }
    assert Settings(aliases_file=tmp_path / "missing.txt").aliases() == {}
    # The file that ships with the app reads cleanly.
    shipped = Settings().aliases()
    assert shipped["NU"] == ("Nubank",) and all(names and all(names) for names in shipped.values())
