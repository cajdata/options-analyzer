"""Pooling a type of event across stocks: what one stock adds, how stocks are grouped, and where it is kept."""

import json
from datetime import date, datetime, time as clock, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx
import pytest

from app import ai
from app.analysis import events
from app.cache import DiskCache, MemoryCache, NullCache
from app.config import Settings
from app.models import Bar, EarningsEvent, NewsItem, PooledStudy
from app.pool import EventPool
from app.providers.demo import NAMES, SPECS, DemoMarket
from app.providers.earnings import EarningsService
from app.providers.news import NewsService
from app.scanner import ConfigError, Scanner, ScanRequest, build_scanner

ET = ZoneInfo("America/New_York")
TODAY = date(2026, 11, 24)


def sessions(count: int, start: date = date(2026, 6, 1)) -> list[date]:
    days, day = [], start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def bars_from(returns: list[float]) -> list[Bar]:
    closes = [100.0]
    for r in returns:
        closes.append(closes[-1] * (1 + r))
    return [Bar(day, close) for day, close in zip(sessions(len(closes)), closes)]


def news(day: date, kind: str, title: str = "headline", item_kind: str = "headline") -> NewsItem:
    when = datetime.combine(day, clock(10, 0), tzinfo=ET)
    return NewsItem(when=when, kind=item_kind, title=title, source="test", ref=f"{when}{kind}{title}", event=kind)


# -- What one stock adds ----------------------------------------------------------


def test_a_stock_adds_its_event_days_counted_against_its_own_usual_day():
    # Every day moves 1% one way or the other. Sessions 20, 40, 60 and 80 carry an offering.
    returns = [0.01 if n % 2 else -0.01 for n in range(100)]
    for n, size in ((20, -0.05), (40, 0.012), (60, -0.03), (80, 0.008)):
        returns[n - 1] = size  # returns[n - 1] is the move on session n
    bars = bars_from(returns)
    days = [b.day for b in bars]
    items = [news(days[n], "financing") for n in (20, 40, 60, 80)] + [news(days[50], "calendar")]

    tally = events.tally(items, bars, set())
    assert events.TALLY == ("events", "big", "up", "down", "sessions", "big_sessions")
    assert set(tally) == {"financing"}  # a notice of a date is not an event
    counted = dict(zip(events.TALLY, tally["financing"]))
    # The usual day is the average over sessions 20 to 100: about 1.07%. Two of the four are over twice that.
    assert counted == {"events": 4, "big": 2, "up": 2, "down": 2, "sessions": 81, "big_sessions": 2}

    # A day that carries an earnings report is neither an event day nor one of the days it is compared with.
    with_report = dict(zip(events.TALLY, events.tally(items, bars, {days[20]})["financing"]))
    assert with_report == {"events": 3, "big": 1, "up": 2, "down": 1, "sessions": 61, "big_sessions": 1}


def test_a_stock_that_never_moved_has_no_usual_day_and_adds_nothing():
    bars = bars_from([0.0] * 90)
    assert events.tally([news(bars[50].day, "financing")], bars, set()) == {}
    assert events.tally([], bars, set()) == {}


def test_a_stock_is_heavily_covered_when_the_feed_carries_its_own_news_most_days():
    def run(count: int, kind: str = "product", days_back: int = 30, item_kind: str = "headline") -> list[NewsItem]:
        return [news(TODAY - timedelta(days=days_back), kind, f"item {n}", item_kind) for n in range(count)]

    assert (events.COVERAGE_DAYS, events.HEAVY_HEADLINES) == (180, 90)
    assert events.coverage(run(90), TODAY) == "heavy"
    assert events.coverage(run(89), TODAY) == "light"
    # Another company's news, filings, and headlines from too long ago say nothing of how much it is covered.
    assert events.coverage(run(89) + run(50, kind="another"), TODAY) == "light"
    assert events.coverage(run(89) + run(50, item_kind="filing"), TODAY) == "light"
    assert events.coverage(run(89) + run(50, days_back=181), TODAY) == "light"
    assert events.coverage([], TODAY) == "light"
    assert set(events.GROUPS) == {"light", "heavy"}


# -- Where it is kept ----------------------------------------------------------------


def test_the_pool_adds_up_the_stocks_in_each_group(tmp_path):
    pool = EventPool(DiskCache(tmp_path))
    pool.add("AAA", "light", {"financing": [4, 2, 1, 3, 80, 9], "analyst": [10, 1, 5, 5, 90, 10]})
    pool.add("BBB", "light", {"financing": [6, 1, 2, 4, 120, 13], "legal": [0, 0, 0, 0, 0, 0]})
    pool.add("CCC", "heavy", {"financing": [30, 3, 15, 15, 300, 33]})
    pool.add("BRK.B", "heavy", {})  # a stock with no event still counts as one on record

    totals = pool.totals()
    assert totals["light"]["stocks"] == 2 and totals["heavy"]["stocks"] == 2
    assert totals["light"]["kinds"]["financing"] == {"counts": [10, 3, 3, 7, 200, 22], "stocks": 2}
    assert totals["light"]["kinds"]["analyst"] == {"counts": [10, 1, 5, 5, 90, 10], "stocks": 1}
    assert totals["light"]["kinds"]["legal"]["stocks"] == 0
    assert totals["heavy"]["kinds"]["financing"]["counts"][0] == 30

    # A stock's newer tally takes the place of its older one, and it can change group.
    pool.add("AAA", "heavy", {"financing": [5, 2, 2, 3, 100, 11]})
    again = EventPool(DiskCache(tmp_path)).totals()  # read back from disk
    assert again["light"]["stocks"] == 1 and again["heavy"]["stocks"] == 3
    assert again["light"]["kinds"]["financing"]["counts"] == [6, 1, 2, 4, 120, 13]


def test_the_pool_passes_over_what_is_old_or_does_not_read_as_a_tally(tmp_path):
    cache = DiskCache(tmp_path)
    pool = EventPool(cache)
    pool.add("GOOD", "light", {"financing": [4, 2, 1, 3, 80, 9]})
    cache.set("pool_BADGROUP", {"group": "medium", "kinds": {}})
    cache.set("pool_BADROW", {"group": "light", "kinds": {"financing": [4, 2, 1]}})
    cache.set("pool_BADNUM", {"group": "light", "kinds": {"financing": [4, "2", 1, 3, 80, 9]}})
    cache.set("pool_BADSHAPE", ["not a tally"])
    cache.set("news_headlines2_GOOD", [])  # another kind of record altogether
    pool.add("OLD", "light", {"financing": [100, 100, 100, 0, 100, 100]})
    path = tmp_path / "pool_OLD.json"
    raw = json.loads(path.read_text())
    raw["stored_at"] = (datetime.now(timezone.utc) - EventPool.KEEP - timedelta(days=1)).isoformat()
    path.write_text(json.dumps(raw))

    totals = pool.totals()
    assert totals["light"] == {"stocks": 1, "kinds": {"financing": {"counts": [4, 2, 1, 3, 80, 9], "stocks": 1}}}
    assert cache.keys("pool_") == ["pool_BADGROUP", "pool_BADNUM", "pool_BADROW", "pool_BADSHAPE", "pool_GOOD", "pool_OLD"]

    # A cache that keeps nothing gives an empty pool, and one in memory works like the one on disk.
    assert EventPool(NullCache()).totals() == {"light": {"stocks": 0, "kinds": {}}, "heavy": {"stocks": 0, "kinds": {}}}
    memory = EventPool(MemoryCache())
    memory.add("AAA", "light", {"financing": [4, 2, 1, 3, 80, 9]})
    assert memory.totals()["light"]["kinds"]["financing"]["stocks"] == 1


def test_a_pooled_study_is_one_groups_figures_for_the_types_asked_about():
    totals = {
        "light": {"stocks": 34, "kinds": {"financing": {"counts": [26, 8, 9, 17, 4721, 520], "stocks": 14}, "split": {"counts": [0, 0, 0, 0, 0, 0], "stocks": 0}}},
        "heavy": {"stocks": 39, "kinds": {"financing": {"counts": [140, 11, 64, 74, 5630, 676], "stocks": 23}}},
    }
    found = events.pooled(totals, "light", ["calendar", "financing", "split", "guidance", "financing"])
    # Only types the pool has events of, and never one that is not studied.
    assert list(found) == ["financing"]
    assert found["financing"] == PooledStudy(
        event="financing", label="Stock or debt offering", group="light", group_stocks=34, stocks=14,
        events=26, big=8, up=9, down=17, sessions=4721, big_sessions=520,
    )
    assert events.pooled(totals, "heavy", ["financing"])["financing"].group_stocks == 39
    assert events.pooled({}, "light", ["financing"]) == {}
    assert (events.BIG_MOVE, events.MIN_POOLED) == (2.0, 20)


# -- In a scan -----------------------------------------------------------------------


def pooled_scanner(pool: EventPool, today: date = TODAY) -> Scanner:
    market = DemoMarket(today)
    earnings = EarningsService(NullCache(), history_sources=[market], calendar_source=market, today=today)
    feed = NewsService(NullCache(), headline_source=market, filing_source=market, names=NAMES.get)
    return Scanner(market, earnings, Settings(), universe=list(SPECS), today=today, news=feed, pool=pool)


def test_a_scan_adds_its_stocks_to_the_pool_and_shows_each_its_group():
    pool = EventPool(MemoryCache())
    scanner = pooled_scanner(pool)
    result = scanner.scan(ScanRequest(symbols=["ACME", "BOLT"]))
    assert {c.underlying for c in result.ideas} == {"ACME", "BOLT"}

    totals = pool.totals()
    assert totals["light"]["stocks"] == 2 and totals["heavy"]["stocks"] == 0  # the demo's stocks are lightly covered
    acme = result.underlyings["ACME"]
    assert acme.coverage == "light"
    # A pooled study for each type in its recent news that is studied at all, in the same order.
    wanted = [kind for kind in acme.recent_events if kind not in events.NOT_STUDIED]
    assert list(acme.pooled_studies) == wanted and wanted
    for kind, study in acme.pooled_studies.items():
        assert (study.event, study.group, study.group_stocks) == (kind, "light", 2)
        assert study.events == totals["light"]["kinds"][kind]["counts"][0] >= 1
        assert 1 <= study.stocks <= 2 and study.big <= study.events and study.up + study.down <= study.events
        assert study.sessions > study.events and 0 < study.big_sessions < study.sessions

    # A later scan of other stocks draws on the ones already there.
    later = scanner.scan(ScanRequest(symbols=["CRUX"]))
    crux = later.underlyings["CRUX"]
    assert pool.totals()["light"]["stocks"] == 3
    assert all(study.group_stocks == 3 for study in crux.pooled_studies.values()) and crux.pooled_studies


def test_a_stock_with_no_report_days_adds_nothing_to_the_pool():
    class NoReports(DemoMarket):
        def history(self, symbol):
            return []

    market = NoReports(TODAY)
    earnings = EarningsService(NullCache(), history_sources=[market], calendar_source=market, today=TODAY)
    feed = NewsService(NullCache(), headline_source=market, filing_source=market, names=NAMES.get)
    pool = EventPool(MemoryCache())
    result = Scanner(market, earnings, Settings(), universe=list(SPECS), today=TODAY, news=feed, pool=pool).scan(ScanRequest(symbols=["ACME"]))
    # Its own news is shown as ever. But a move that belongs to a report cannot be told from an event's.
    assert result.underlyings["ACME"].news and pool.totals()["light"]["stocks"] == 0
    assert result.underlyings["ACME"].pooled_studies == {}


def test_a_pool_that_cannot_be_read_does_not_sink_the_scan():
    class Broken(EventPool):
        def totals(self):
            raise OSError("disk gone")

    result = pooled_scanner(Broken(MemoryCache())).scan(ScanRequest(symbols=["ACME"]))
    acme = result.underlyings["ACME"]
    assert result.ideas and acme.news and acme.event_studies and acme.pooled_studies == {}
    assert not any("News" in w for w in result.warnings)

    # With no pool at all, a scan is as it was before pooling.
    plain = pooled_scanner(None).scan(ScanRequest(symbols=["ACME"]))
    assert plain.underlyings["ACME"].coverage == "light" and plain.underlyings["ACME"].pooled_studies == {}


# -- Widening the pool -----------------------------------------------------------------


def test_widening_the_pool_adds_stocks_nobody_has_scanned():
    pool = EventPool(MemoryCache())
    scanner = pooled_scanner(pool)
    seen = []
    found = scanner.widen_pool(["acme", "BOLT", "bolt", "ZZZZ"], progress=lambda n, of, symbol: seen.append((n, of, symbol)))
    # Tidied like a scan's symbols. A stock the feed has nothing for is passed over, and said to be.
    assert seen == [(1, 3, "ACME"), (2, 3, "BOLT"), (3, 3, "ZZZZ")]
    assert found["asked"] == 3 and found["added"] == {"light": 2, "heavy": 0}
    assert found["passed_over"] == {"no headlines": 1}
    assert found["totals"] == pool.totals() and pool.totals()["light"]["stocks"] == 2
    assert scanner.market.notes == []

    # With no symbols it takes the universe and the day's most active stocks and movers.
    everything = scanner.widen_pool()
    assert everything["asked"] == len(SPECS) and everything["added"]["light"] == len(SPECS)
    assert sum(pool.totals()["light"]["kinds"]["analyst"]["counts"][:1]) >= events.MIN_POOLED

    # What it adds for a stock is what a scan of that stock adds.
    scanned = EventPool(MemoryCache())
    pooled_scanner(scanned).scan(ScanRequest(symbols=["ACME"]))
    only = EventPool(MemoryCache())
    pooled_scanner(only).widen_pool(["ACME"])
    assert scanned.totals() == only.totals()


def test_widening_needs_a_news_source_and_a_pool():
    market = DemoMarket(TODAY)
    earnings = EarningsService(NullCache(), history_sources=[market], calendar_source=market, today=TODAY)
    with pytest.raises(ConfigError):
        Scanner(market, earnings, Settings(), today=TODAY).widen_pool(["ACME"])


def test_report_days_for_a_pooled_stock_cost_no_request_at_a_rate_limited_source(tmp_path):
    report = EarningsEvent(day=date(2026, 8, 5), timing="post", source="SEC EDGAR")

    class Source:
        name = "SEC EDGAR"

        def __init__(self, answer, rate_limited=False):
            self.answer, self.calls, self.rate_limited = answer, 0, rate_limited

        def history(self, symbol):
            self.calls += 1
            if isinstance(self.answer, Exception):
                raise self.answer
            return self.answer

    first, second = Source([report]), Source([report, report])
    service = EarningsService(DiskCache(tmp_path), history_sources=[first, second], today=TODAY)
    # Nothing on file: the first source alone is asked, and its answer is kept nowhere.
    assert service.reports("ACME") == [report] and (first.calls, second.calls) == (1, 0)
    assert list(tmp_path.glob("earnings_history_*")) == []
    assert service.reports("ACME") == [report] and first.calls == 2

    # Once a scan has put the fuller history on file, that answers and no source is asked.
    assert service.history("ACME")[0] and (first.calls, second.calls) == (3, 1)
    assert service.reports("ACME") == service.history("ACME")[0] and (first.calls, second.calls) == (3, 1)

    # A source with only a few requests a day is never asked on a pooled stock's account.
    scarce = Source([report], rate_limited=True)
    assert EarningsService(DiskCache(tmp_path / "b"), history_sources=[scarce], today=TODAY).reports("ACME") == [] and scarce.calls == 0
    # A source that fails, and no source at all, give nothing.
    down = Source(httpx.ConnectError("down"))
    assert EarningsService(DiskCache(tmp_path / "c"), history_sources=[down], today=TODAY).reports("ACME") == []
    assert EarningsService(DiskCache(tmp_path / "d"), today=TODAY).reports("ACME") == []


# -- On every surface ------------------------------------------------------------------


@pytest.fixture()
def demo_day_is_pinned():
    wall_clock = DemoMarket.__init__

    def pinned(self, today=None):
        wall_clock(self, today or TODAY)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(DemoMarket, "__init__", pinned)
        yield


def test_pooled_studies_reach_the_api_the_page_and_the_command_line(demo_day_is_pinned, capsys):
    from fastapi.testclient import TestClient

    from app.__main__ import main
    from app.server import create_app

    client = TestClient(create_app(Settings(), demo=True))
    rules = client.get("/api/status").json()["events"]["pool"]
    assert rules == {
        "groups": {"light": "lightly covered", "heavy": "heavily covered"},
        "big_move": 2.0,
        "min_events": 20,
        "coverage_days": 180,
        "heavy_headlines": 90,
    }

    acme = client.post("/api/scan", json={"symbols": ["ACME"], "top_n": 4}).json()["underlyings"]["ACME"]
    assert acme["coverage"] == "light" and acme["pooled_studies"]
    study = next(iter(acme["pooled_studies"].values()))
    # The demo starts with every demo stock in its pool, kept in memory.
    assert study["group_stocks"] == len(SPECS) and study["events"] >= study["big"] >= 0 and study["sessions"] > 0
    assert set(study) == {"event", "label", "group", "group_stocks", "stocks", "events", "big", "up", "down", "sessions", "big_sessions"}

    page = client.get("/").text
    assert "pooledBlock(ctx)" in page and "innerHTML" not in page

    assert main(["scan", "ACME", "--demo", "--top", "4"]) == 0
    out = capsys.readouterr().out
    assert f"across {len(SPECS)} lightly covered stocks, " in out and "moved over 2x the stock's usual day" in out

    assert main(["pool", "--demo"]) == 0
    printed = capsys.readouterr()
    assert f"Asked for {len(SPECS)} stocks. Added {len(SPECS)} lightly covered, 0 heavily covered." in printed.out
    assert "Lightly covered: 6 stocks on record" in printed.out and "it is not a forecast" in printed.out
    assert "reading news: 1 of 6" in printed.err


def test_the_brief_is_not_given_the_pooled_studies(demo_day_is_pinned):
    # What the model reads is unchanged: pooled figures would need the prompt to say what they are.
    scanner = build_scanner(Settings(), demo=True)
    result = scanner.scan(ScanRequest(symbols=["ACME"]))
    assert result.underlyings["ACME"].pooled_studies
    payload = json.dumps(ai.scan_payload(result))
    assert "pooled" not in payload and "lightly covered" not in payload and "big_sessions" not in payload


# -- What may join the pool (found by an outside review) ------------------------------------


def test_what_may_join_the_pool_is_one_rule():
    bars = bars_from([0.01] * 80)
    day = bars[40].day
    heads, files = [news(day, "product")], [news(day, "management", item_kind="filing")]
    assert events.unpoolable(heads + files, bars, {day}) is None
    assert events.unpoolable(files, bars, {day}) == "no headlines"
    assert events.unpoolable(heads, bars[:59], {day}) == "too little price history"
    assert events.unpoolable(heads, bars, set()) == "no report days"
    assert events.BASELINE_SESSIONS == 60


def test_a_failed_lookup_does_not_take_the_place_of_a_stocks_full_tally():
    class NoFeed(DemoMarket):
        def headlines(self, symbol, since):
            raise httpx.ConnectError("down")

    market = NoFeed(TODAY)
    earnings = EarningsService(NullCache(), history_sources=[market], calendar_source=market, today=TODAY)
    feed = NewsService(NullCache(), headline_source=market, filing_source=market, names=NAMES.get)
    pool = EventPool(MemoryCache())
    full = {"financing": [9, 1, 4, 5, 100, 11]}
    pool.add("ACME", "heavy", full)
    scanner = Scanner(market, earnings, Settings(), universe=list(SPECS), today=TODAY, news=feed, pool=pool)

    result = scanner.scan(ScanRequest(symbols=["ACME"]))
    assert any("News could not be loaded for ACME" in w for w in result.warnings)
    # Only its filings came through. Pooled from those it would have become a lightly covered stock.
    assert result.underlyings["ACME"].coverage == "light"
    assert pool.totals()["heavy"] == {"stocks": 1, "kinds": {"financing": {"counts": full["financing"], "stocks": 1}}}
    assert pool.totals()["light"]["stocks"] == 0

    # Widening treats it the same way, and says why.
    found = scanner.widen_pool(["ACME", "BOLT"])
    assert found["added"] == {"light": 0, "heavy": 0} and found["passed_over"] == {"a news lookup that failed": 2}
    assert pool.totals()["heavy"]["stocks"] == 1


def test_a_stock_with_filings_and_no_headlines_is_pooled_by_neither_a_scan_nor_widening():
    # A second share class has 8-K filings and report days, and the feed carries nothing for it.
    class NoHeadlines(DemoMarket):
        def headlines(self, symbol, since):
            return []

    market = NoHeadlines(TODAY)
    earnings = EarningsService(NullCache(), history_sources=[market], calendar_source=market, today=TODAY)
    feed = NewsService(NullCache(), headline_source=market, filing_source=market, names=NAMES.get)
    pool = EventPool(MemoryCache())
    scanner = Scanner(market, earnings, Settings(), universe=list(SPECS), today=TODAY, news=feed, pool=pool)
    result = scanner.scan(ScanRequest(symbols=["ACME"]))
    assert result.underlyings["ACME"].news  # its filings are shown as ever
    assert pool.totals()["light"]["stocks"] == 0
    assert scanner.widen_pool(["ACME"])["passed_over"] == {"no headlines": 1}


def test_reports_that_fall_outside_the_price_history_do_not_qualify_a_stock():
    class OldReports(DemoMarket):
        def history(self, symbol):
            return [EarningsEvent(day=date(2019, 2, 5), timing="post", source="demo")]

    market = OldReports(TODAY)
    earnings = EarningsService(NullCache(), history_sources=[market], calendar_source=market, today=TODAY)
    feed = NewsService(NullCache(), headline_source=market, filing_source=market, names=NAMES.get)
    pool = EventPool(MemoryCache())
    scanner = Scanner(market, earnings, Settings(), universe=list(SPECS), today=TODAY, news=feed, pool=pool)
    # It has reports on file, but none within the three years of prices, so no day can be left out.
    assert scanner.widen_pool(["ACME"])["passed_over"] == {"no report days": 1}
    assert pool.totals()["light"]["stocks"] == 0


def test_the_scan_says_when_most_of_a_stocks_headlines_never_name_it(capsys):
    from app.__main__ import print_scan

    class Feed:
        name = "test"

        def headlines(self, symbol, since):
            def at(n, title, summary):
                when = datetime.combine(TODAY - timedelta(days=1 + n % 9), clock(10, 0), tzinfo=ET)
                return NewsItem(when, "headline", title, "test", ref=str(n), summary=summary)

            # Twelve headlines under a name that is not the company's, and three under its own.
            other = [at(n, f"Brandco Launches Product {n}", "Brandco said it launched a product.") for n in range(12)]
            own = [at(100 + n, f"Acme Opens Office {n}", "Acme opened an office.") for n in range(3)]
            return other + own

    class NoFilings:
        name = "test"

        def filings(self, symbol):
            return []

    market = DemoMarket(TODAY)
    earnings = EarningsService(NullCache(), history_sources=[market], calendar_source=market, today=TODAY)
    feed = NewsService(NullCache(), headline_source=Feed(), filing_source=NoFilings(), names=NAMES.get)
    result = Scanner(market, earnings, Settings(), universe=list(SPECS), today=TODAY, news=feed).scan(ScanRequest(symbols=["ACME"]))
    acme = result.underlyings["ACME"]
    assert acme.news_note == (
        "12 of the 15 headlines the feed tags with ACME never name Acme, and are set aside. "
        "If it goes by another name, add that to data/aliases.txt."
    )
    assert acme.recent_events["another"] == 12 and len(acme.news) == 3
    # It is a note about one stock, not a failed lookup, so it is not among the scan's warnings.
    assert not any("never name" in w for w in result.warnings)
    # The command line says it too, as it now does any note about a stock's news.
    print_scan(result)
    out = " ".join(capsys.readouterr().out.split())
    assert "12 of the 15 headlines the feed tags with ACME never name Acme" in out

    # A listed alias settles it: the same headlines are the company's own, and nothing is said.
    scanner = Scanner(market, earnings, Settings(), universe=list(SPECS), today=TODAY, news=feed)
    scanner._also = {"ACME": ("Brandco",)}
    named = scanner.scan(ScanRequest(symbols=["ACME"])).underlyings["ACME"]
    assert named.news_note is None and "another" not in named.recent_events and len(named.news) == events.MAX_RECENT
