"""News layer: event types, what followed past events, and the sources behind them."""

import json
from datetime import date, datetime, time as clock, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest

from app.analysis import events
from app.cache import DiskCache, NullCache
from app.models import Bar, NewsItem
from app.providers.alpaca import MAX_NEWS_PAGES, AlpacaData, AlpacaError
from app.providers.earnings import EdgarEarnings, Unsupported
from app.providers.news import NewsService

FIXTURES = Path(__file__).parent / "fixtures"
NEWS = json.loads((FIXTURES / "alpaca_news.json").read_text(encoding="utf-8"))
ET = ZoneInfo("America/New_York")
TODAY = date(2026, 11, 24)


# -- Rules ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "headline, kind",
    [
        ("Acme Reports Third Quarter 2026 Financial Results", "earnings"),
        ("Acme Reports Loss on Plant Closure but Lifts 2026 Financial Forecast", "earnings"),
        ("Acme Raises Full-Year 2026 Revenue Guidance", "guidance"),
        ("Acme Warns of Margin Pressures in the December Quarter", "guidance"),
        ("Quillfeather Research Raises Acme Price Target to $229", "analyst"),
        ("Marrowgate Capital Initiates Coverage of Acme with Buy Rating", "analyst"),
        ("Acme Agrees to Be Acquired by Tidewell Systems for $4 Billion", "takeover"),
        ("Acme Receives Unsolicited Acquisition Proposal", "takeover"),
        ("Acme Acquires Tidewell Systems for Up to $2.3 Billion", "acquisition"),
        ("Acme Announces 4-for-1 Stock Split", "split"),
        ("Acme Declares Special Cash Dividend", "split"),
        ("Acme Announces Pricing of Upsized $2.6 Billion Offering of Convertible Senior Notes", "financing"),
        ("Acme Announces 10% Dividend Increase", "payout"),
        ("Acme CFO Sells Shares Worth Over $1 Million", "insider"),
        ("Longview Partners Increases Stake in Acme to 6.8%", "insider"),
        ("Acme Files for Chapter 11 Bankruptcy Protection", "distress"),
        ("Acme Reports Cyberattack on Third-Party Vendors", "incident"),
        ("Acme Announces Positive Phase 3 Results for Its Lung Cancer Drug", "drug"),
        ("FDA Approves Acme's Combination Treatment", "drug"),
        ("Acme Recalls Over 565,000 Vehicles Due to Fire Risk", "legal"),
        ("Acme Faces Civil Trial Over Product Claims", "legal"),
        ("Acme Conducts Targeted Layoffs in Its Fitness Division", "restructuring"),
        ("Acme Shareholders Approve Charter Amendment", "governance"),
        ("Acme Appoints Dana Whitlow as Chief Financial Officer", "management"),
        ("Acme Software Chief Jon Marlowe Departs", "management"),
        ("Acme and Tidewell Systems Announce Multi-Year Partnership", "agreement"),
        ("Acme Launches Next-Generation Platform", "product"),
        ("Acme to Present at Lakeshore Technology Conference", "calendar"),
        ("Acme Publishes Annual Sustainability Report", "other"),
    ],
)
def test_headlines_are_sorted_into_event_types_by_keyword(headline, kind):
    assert events.headline_type(headline) == kind


@pytest.mark.parametrize(
    "headline, kind",
    [
        # Raising a price target is an analyst's note, not the company's guidance.
        ("Quillfeather Research Raises Acme Price Target Ahead of Q1 Earnings", "analyst"),
        # A notice of a date is not the results.
        ("Acme Announces Webcast for Third Quarter 2026 Earnings Conference Call", "calendar"),
        ("Acme Schedules Third Quarter 2026 Financial Results Release", "calendar"),
        # "to Hold" is a rating here, not a meeting being held.
        ("Marrowgate Capital Downgrades Acme to Hold From Buy", "analyst"),
        # Weight loss is not a loss.
        ("Acme Reports Its New Pill Showed Greater Weight Loss Than a Rival in Data Analysis", "other"),
        # A study the company publishes about its market is not its own outlook.
        ("Acme Releases 2026 Commercial Market Outlook", "product"),
        # The lender is not the one raising money.
        ("Acme Bank Leads $660 Million Credit Facility for Tidewell Systems", "agreement"),
        ("Acme Ventures Participates in Tidewell's Series A Financing Round", "agreement"),
        # A bond leaving an exchange is routine; a stock leaving one is not.
        ("Acme Delists 1.625% Notes Due 2026 from NYSE", "other"),
        ("Acme Receives Nasdaq Delisting Notice", "distress"),
        # A clinical trial is not a court case.
        ("Acme Initiates Phase 3 Trial for Its Flu Shot", "drug"),
        # An executive selling shares is a trade, not a change of management.
        ("Acme Director Files Form 144 to Propose Sale of 6.42 Million Shares", "insider"),
    ],
)
def test_the_order_of_the_rules_settles_a_headline_that_fits_two(headline, kind):
    assert events.headline_type(headline) == kind


@pytest.mark.parametrize(
    "symbol, name, expected",
    [
        ("INSM", "INSMED Inc", ("INSM", "insmed")),
        ("JPM", "JPMORGAN CHASE & CO", ("JPM", "jpmorgan")),
        ("COST", "COSTCO WHOLESALE CORP /NEW", ("COST", "costco")),
        ("CRM", "Salesforce, Inc.", ("CRM", "salesforce")),
        ("F", "Ford Motor Co", ("ford",)),  # a one-letter ticker would match far too much
        ("GM", "General Motors Co", ("general motors",)),  # "general" alone names nobody
        ("BRK.B", "BERKSHIRE HATHAWAY INC", ("berkshire",)),
        ("ZZZZ", None, ("ZZZZ",)),
        ("T", "", ()),
    ],
)
def test_a_company_goes_by_its_ticker_and_the_first_word_of_its_name(symbol, name, expected):
    assert events.aliases(symbol, name) == expected


def test_the_company_being_bought_is_told_from_the_buyer_by_name():
    insmed, sanofi = ("INSM", "insmed"), ("SNY", "sanofi")
    deal = "Sanofi to Acquire Insmed for $41 Billion"
    # The feed tags both companies with this headline. Only one of them is being taken over.
    assert events.headline_type(deal, insmed) == "takeover"
    assert events.headline_type(deal, sanofi) == "acquisition"
    assert events.headline_type(deal) == "acquisition"  # with no name to go on, the keyword rules decide
    assert events.headline_type("Sanofi Reportedly Considering Acquisition of Insmed", insmed) == "takeover"
    assert events.headline_type("Ademi LLP Investigates Proposed Acquisition of Insmed", insmed) == "takeover"
    assert events.headline_type("Tidewell Systems Makes $9 Billion Offer for INSM", insmed) == "takeover"
    # Insmed as the buyer, and a deal that only mentions it, are not takeovers of Insmed.
    assert events.headline_type("Insmed to Acquire Tidewell Systems", insmed) == "acquisition"
    assert events.headline_type("Berenberg Reaffirms Sanofi Buy Rating Amid Ongoing Acquisition Rumors", insmed) == "analyst"

    # A ticker that is also a word has to appear as a ticker.
    costco = ("COST", "costco")
    assert events.headline_type("Tidewell Systems Wins Deal for Cost Monitoring Software", costco) == "agreement"
    assert events.headline_type("Tidewell Systems to Buy COST in $400 Billion Deal", costco) == "takeover"
    assert events.classify(headline(1, NOON, deal), insmed).event == "takeover"


def test_every_rule_points_at_a_type_with_a_label():
    known = set(events.EVENT_TYPES)
    assert {kind for kind, _ in events.HEADLINE_RULES} <= known
    assert {kind for kind, _ in events.FILING_ITEMS.values()} <= known
    assert events.NOT_STUDIED <= known and events.FLAGGED <= known and set(events.CAUTIONS) <= events.FLAGGED
    assert not events.FLAGGED & events.NOT_STUDIED


def filing(items, when=datetime(2026, 11, 20, 16, 30, tzinfo=ET), form="8-K", ref="f1"):
    return NewsItem(when=when, kind="filing", title=form, source="SEC EDGAR", ref=ref, items=tuple(items))


NOON = datetime(2026, 11, 20, 17, 0, tzinfo=timezone.utc)


def headline(n: int, when: datetime, title: str = "Acme Launches Next-Generation Platform") -> NewsItem:
    return NewsItem(when=when, kind="headline", title=title, source="fiscal.ai", url=f"https://example.com/{n}", ref=str(n))


def test_a_filing_takes_its_type_from_its_most_telling_item():
    both = events.classify(filing(["7.01", "5.02", "9.01"]))
    assert both.event == "management"
    assert both.title == "8-K: Departure or appointment of directors or officers; Regulation FD disclosure"

    # Results come with other items attached, and it is still an earnings report.
    assert events.classify(filing(["2.02", "7.01", "9.01"])).event == "earnings"
    assert events.classify(filing(["8.01", "1.03"])).event == "distress"

    amended = events.classify(filing(["5.02"], form="8-K/A"))
    assert amended.title == "8-K/A: Departure or appointment of directors or officers"

    # Exhibits alone, or a code the table does not know, say nothing about the event.
    paperwork = events.classify(filing(["9.01", "6.02"]))
    assert (paperwork.event, paperwork.title) == ("other", "8-K")


# -- What followed ----------------------------------------------------------------


def sessions(count: int, start: date = date(2026, 6, 1)) -> list[date]:
    days, day = [], start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def bars_from(returns: list[float], start: date = date(2026, 6, 1)) -> list[Bar]:
    """Closes on consecutive weekdays: 100 on `start`, then one return a session."""
    closes = [100.0]
    for r in returns:
        closes.append(closes[-1] * (1 + r))
    return [Bar(day, close) for day, close in zip(sessions(len(closes), start), closes)]


def news(day: date, at: clock, kind: str, title: str = "headline", ref: str = "") -> NewsItem:
    when = datetime.combine(day, at, tzinfo=ET)
    return NewsItem(when=when, kind="headline", title=title, source="test", ref=ref or f"{when}{title}", event=kind)


def test_an_event_is_measured_on_the_first_session_that_reflects_it():
    #        Mon Jun 1 (100), Tue +1%, Wed +2%, Thu -3%, Fri +4%, Mon -5%, then flat
    bars = bars_from([0.01, 0.02, -0.03, 0.04, -0.05] + [0.0] * 10)
    wed, sat = date(2026, 6, 3), date(2026, 6, 6)
    items = [
        news(wed, clock(10, 0), "guidance"),  # during the session: Wednesday's own move
        news(wed, clock(16, 0), "legal"),  # at the close or after it: Thursday's
        news(sat, clock(9, 0), "management"),  # at a weekend: Friday's close to Monday's
        # The same instant as the second one, given in UTC as a feed does.
        NewsItem(datetime(2026, 6, 3, 20, 0, tzinfo=timezone.utc), "headline", "utc", "test", ref="u", event="drug"),
    ]
    found = events.study(items, bars, set(), ["guidance", "legal", "management", "drug"])
    moves = {kind: study.moves[0] for kind, study in found.items()}
    assert (moves["guidance"].day, moves["guidance"].move_pct) == (wed, pytest.approx(0.02))
    assert (moves["legal"].day, moves["legal"].move_pct) == (date(2026, 6, 4), pytest.approx(-0.03))
    assert (moves["management"].day, moves["management"].move_pct) == (date(2026, 6, 8), pytest.approx(-0.05))
    assert (moves["drug"].day, moves["drug"].move_pct) == (date(2026, 6, 4), pytest.approx(-0.03))
    # What followed: that session's close to the close five sessions later.
    assert moves["guidance"].after_pct == pytest.approx(0.97 * 1.04 * 0.95 - 1)


def test_event_days_are_set_beside_the_stocks_usual_day():
    # Most days move 1% either way. Four event days move 5%.
    returns = [0.01 if n % 2 else -0.01 for n in range(100)]
    for n, size in ((20, 0.05), (40, -0.05), (60, 0.05), (80, 0.05)):
        returns[n - 1] = size  # returns[n - 1] is the move on session n
    bars = bars_from(returns)
    days = [b.day for b in bars]
    items = [news(days[n], clock(10, 0), "guidance", f"event {n}") for n in (20, 40, 60, 80)]

    study = events.study(items, bars, set(), ["guidance"])["guidance"]
    assert (study.event, study.label) == ("guidance", "Guidance or outlook")
    assert (study.count, study.up, study.down, study.since) == (4, 3, 1, days[20])
    assert study.mean_abs == pytest.approx(0.05)
    # Every session from the first event on, the event days among them: 4 of 5% and 77 of 1%.
    assert study.ordinary_abs == pytest.approx((4 * 0.05 + 77 * 0.01) / 81)
    assert [m.title for m in study.moves] == ["event 80", "event 60", "event 40", "event 20"]  # newest first

    closes = [b.close for b in bars]
    assert study.after_count == 4
    assert study.mean_abs_after == pytest.approx(sum(abs(closes[n + 5] / closes[n] - 1) for n in (20, 40, 60, 80)) / 4)
    # Every five-session span from the first event on, whether or not it holds an event.
    spans = [abs(closes[i + 5] / closes[i] - 1) for i in range(20, len(closes) - 5)]
    assert study.ordinary_abs_after == pytest.approx(sum(spans) / len(spans))


def test_a_day_that_carries_an_earnings_report_is_left_out():
    returns = [0.01 if n % 2 else -0.01 for n in range(100)]
    returns[19], returns[39], returns[59] = 0.05, -0.05, 0.20  # sessions 20 and 40, and a report on 60
    bars = bars_from(returns)
    days = [b.day for b in bars]
    items = [news(days[n], clock(10, 0), "guidance") for n in (20, 40, 60)]

    study = events.study(items, bars, {days[60]}, ["guidance"])["guidance"]
    # The 20% belongs to the report: it is neither an event of this type nor part of the usual day.
    assert study.count == 2 and study.mean_abs == pytest.approx(0.05)
    assert study.ordinary_abs == pytest.approx((2 * 0.05 + 78 * 0.01) / 80)
    assert days[60] not in {m.day for m in study.moves}


def test_several_items_on_one_session_are_one_event():
    bars = bars_from([0.01] * 30)
    days = [b.day for b in bars]
    items = [
        news(days[9], clock(17, 0), "analyst", "the evening before"),  # after the close: felt on session 10
        news(days[10], clock(8, 0), "analyst", "before the open"),
        news(days[10], clock(11, 0), "analyst", "mid-morning"),
        news(days[10], clock(11, 0), "legal", "another type the same day"),
    ]
    found = events.study(items, bars, set(), ["analyst", "legal"])
    assert found["analyst"].count == 1 and found["analyst"].moves[0].title == "the evening before"
    assert found["legal"].count == 1


def test_an_event_is_counted_only_once_the_session_that_reflects_it_has_closed():
    bars = bars_from([0.01] * 30)
    days = [b.day for b in bars]
    items = [
        news(days[-1], clock(16, 30), "guidance", "after the last close on record"),
        news(days[-1] + timedelta(days=1), clock(9, 0), "guidance", "the next morning"),
        news(days[-3], clock(10, 0), "guidance", "three sessions ago"),
    ]
    study = events.study(items, bars, set(), ["guidance"])["guidance"]
    assert [m.title for m in study.moves] == ["three sessions ago"]
    # Too recent for five sessions to have followed, so nothing is said about what followed.
    assert study.moves[0].after_pct is None
    assert (study.after_count, study.mean_abs_after, study.ordinary_abs_after) == (0, None, None)

    # With nothing but unfinished events there is no study at all.
    assert events.study(items[:2], bars, set(), ["guidance"]) == {}


def test_the_usual_day_is_measured_over_at_least_a_quarter():
    # Two events in the last week, which was wild. Before it: quiet sessions, and wilder ones earlier still.
    returns = [0.10] * 40 + [0.01] * 55 + [0.04] * 5
    bars = bars_from(returns)
    days = [b.day for b in bars]
    items = [news(days[-4], clock(10, 0), "legal"), news(days[-2], clock(10, 0), "legal")]
    study = events.study(items, bars, set(), ["legal"])["legal"]
    assert study.mean_abs == pytest.approx(0.04)
    # Not just the days since the first event (all 4%), and not the stretch before the last 60 sessions.
    assert events.BASELINE_SESSIONS == 60
    assert study.ordinary_abs == pytest.approx((55 * 0.01 + 5 * 0.04) / 60)


def test_types_that_are_not_events_get_no_study():
    bars = bars_from([0.01] * 30)
    day = bars[10].day
    items = [news(day, clock(10, 0), kind) for kind in ("earnings", "calendar", "other")]
    assert events.study(items, bars, set(), ["earnings", "calendar", "other", "guidance"]) == {}


def test_a_hole_in_the_price_history_drops_the_event():
    early, late = bars_from([0.01] * 5, date(2026, 6, 1)), bars_from([0.01] * 5, date(2026, 7, 1))
    in_the_hole = news(date(2026, 6, 17), clock(10, 0), "legal")
    assert events.study([in_the_hole], early + late, set(), ["legal"]) == {}


def test_context_lists_recent_news_and_studies_only_the_types_in_it():
    bars = bars_from([0.01] * 120, date(2026, 6, 1))
    days = [b.day for b in bars]
    last = days[-1]
    today = last + timedelta(days=1)

    def raw(day, title, n):
        return NewsItem(datetime.combine(day, clock(10, 0), tzinfo=ET), "headline", title, "test", ref=str(n))

    recent = [raw(last - timedelta(days=n % 5), f"Acme Raises Full-Year Revenue Guidance {n}", n) for n in range(12)]
    older = [
        raw(days[30], "Acme Raises Full-Year Revenue Guidance", 100),
        raw(days[50], "Acme Faces Class Action Lawsuit", 101),  # a type with nothing recent
        raw(today - timedelta(days=events.RECENT_DAYS + 1), "Acme Launches Next-Generation Platform", 102),  # a day too old
    ]
    extra = [raw(last, "Acme to Present at Lakeshore Technology Conference", 200), filing(["5.02"], when=datetime.combine(last, clock(8, 0), tzinfo=ET))]

    shown, counts, studies = events.context(recent + older + extra, bars, set(), today)
    assert len(shown) == events.MAX_RECENT and shown == sorted(shown, key=lambda i: i.when, reverse=True)
    assert all(item.event for item in shown)  # classified on the way through
    # Every recent item is counted, not only the ones listed, in the order the types are defined.
    assert counts == {"guidance": 12, "management": 1, "calendar": 1}
    # A study draws on the whole record, and there is none for a type with nothing recent.
    assert set(studies) == {"guidance", "management"}
    assert studies["guidance"].since == days[30]


# -- Alpaca headlines -------------------------------------------------------------


def alpaca(handler) -> AlpacaData:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return AlpacaData("key-id", "secret", client=client, sleep=lambda _: None)


def test_headlines_parse_a_captured_response_and_follow_its_pages():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=NEWS["second" if "page_token" in request.url.params else "first"])

    items = alpaca(handler).headlines("ACME", datetime(2026, 11, 2, tzinfo=ET))

    assert seen[0].url.path == "/v1beta1/news"
    assert dict(seen[0].url.params) == {
        "symbols": "ACME",
        "start": "2026-11-02T05:00:00Z",  # midnight in New York, sent in UTC
        "limit": "50",
        "sort": "desc",
        "include_content": "false",
    }
    assert seen[1].url.params["page_token"] == NEWS["first"]["next_page_token"] and len(seen) == 2
    assert len(items) == 6 and items == sorted(items, key=lambda i: i.when, reverse=True)
    assert items[0] == NewsItem(
        when=datetime(2026, 11, 24, 19, 2, 45, tzinfo=timezone.utc),
        kind="headline",
        title="Acme Director Sells Shares Worth $2 Million",
        source="fiscal.ai",
        url="https://news.example.com/acme/director-share-sale?source=feed&campaign=1",
        ref="129880987",
    )
    # Two stories can share a time stamp. Their ids keep them apart.
    assert items[2].when == items[3].when and items[2].ref != items[3].ref
    assert [events.classify(i).event for i in items] == ["insider", "agreement", "analyst", "product", "guidance", "other"]


def test_headline_rows_that_cannot_be_used_are_skipped_and_only_web_links_are_kept():
    rows = [
        {"id": 1, "headline": "  Acme   Opens\nNew Office ", "created_at": "2026-11-20T15:00:00Z", "url": "javascript:alert(1)"},
        {"id": 2, "headline": "", "created_at": "2026-11-20T15:00:00Z"},
        {"id": 3, "headline": "No time", "created_at": None},
        {"id": 4, "headline": "No zone", "created_at": "2026-11-20T15:00:00"},
        "not a row",
        {"id": 5, "headline": "Plain link", "created_at": "2026-11-19T15:00:00Z", "url": "HTTPS://example.com/a"},
    ]
    items = alpaca(lambda request: httpx.Response(200, json={"news": rows, "next_page_token": None})).headlines("ACME", datetime(2026, 11, 1, tzinfo=ET))
    assert [(i.ref, i.title, i.url, i.source) for i in items] == [
        ("1", "Acme Opens New Office", None, "Alpaca"),
        ("5", "Plain link", "HTTPS://example.com/a", "Alpaca"),
    ]


def test_a_very_long_run_of_headlines_is_cut_off_and_the_scan_says_so():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        row = {"id": len(calls), "headline": "More", "created_at": "2026-11-20T15:00:00Z"}
        return httpx.Response(200, json={"news": [row], "next_page_token": "more"})

    market = alpaca(handler)
    assert len(market.headlines("ACME", datetime(2026, 1, 1, tzinfo=ET))) == MAX_NEWS_PAGES == len(calls)
    assert market.notes == ["ACME: only the newest 2,000 headlines were loaded."]


# -- SEC filings ------------------------------------------------------------------


def edgar(tmp_path, recent: dict) -> EdgarEarnings:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/files/company_tickers.json":
            return httpx.Response(200, json={"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}})
        return httpx.Response(200, json={"filings": {"recent": recent, "files": []}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return EdgarEarnings("Test Person test@example.com", DiskCache(tmp_path), client=client, sleep=lambda _: None)


def test_edgar_lists_each_8k_with_its_items_and_a_link(tmp_path):
    recent = {
        "form": ["8-K", "10-Q", "8-K/A", "4", "8-K"],
        "items": ["1.01,7.01,9.01", "", "5.02", "", "2.02,9.01"],
        "acceptanceDateTime": [
            "2026-11-24T11:04:08.000Z",
            "2026-11-05T21:10:00.000Z",
            "2026-09-01T20:30:35.000Z",
            "2026-08-03T22:00:00.000Z",
            "2026-07-30T20:30:40.000Z",
        ],
        "accessionNumber": ["0000320193-26-000105", "0000320193-26-000104", "0000320193-26-000103", "0000320193-26-000102", ""],
        "primaryDocument": ["agreement_8k.htm", "q3_10q.htm", "officers_8ka.htm", "form4.xml", "results_8k.htm"],
    }
    found = edgar(tmp_path, recent).filings("AAPL")

    assert [(f.title, f.items) for f in found] == [
        ("8-K", ("1.01", "7.01", "9.01")),
        ("8-K/A", ("5.02",)),
        ("8-K", ("2.02", "9.01")),
    ]
    first = found[0]
    assert first.when == datetime(2026, 11, 24, 6, 4, 8, tzinfo=ET) and (first.kind, first.source) == ("filing", "SEC EDGAR")
    assert first.url == "https://www.sec.gov/Archives/edgar/data/320193/000032019326000105/agreement_8k.htm"
    assert first.ref == "0000320193-26-000105"
    assert found[2].url is None  # no accession number, so nothing to link to
    assert [events.classify(f).event for f in found] == ["agreement", "management", "earnings"]


def test_edgar_filings_from_a_real_filing_index(tmp_path):
    recent = json.loads((FIXTURES / "edgar_aapl.json").read_text(encoding="utf-8"))["filings"]["recent"]
    found = edgar(tmp_path, recent).filings("AAPL")
    assert len(found) == 27 and found == sorted(found, key=lambda f: f.when, reverse=True)
    kinds = [events.classify(f).event for f in found]
    assert kinds.count("earnings") == 12 and kinds.count("management") == 7
    # This capture carries no document names, so each link goes to the filing's folder.
    assert all(f.url.startswith("https://www.sec.gov/Archives/edgar/data/320193/") and f.url.endswith("/") for f in found)


def test_edgar_knows_a_company_by_name(tmp_path):
    source = edgar(tmp_path, {})
    assert source.company_name("AAPL") == "Apple Inc." and source.company_name("ZZZZ") is None
    assert source._cik("AAPL") == 320193  # the same table still finds a company's filings
    # The names came with the list of tickers, which is on disk for the next run.
    again = EdgarEarnings("Test Person test@example.com", DiskCache(tmp_path), client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    assert again.company_name("AAPL") == "Apple Inc."


def test_edgar_tells_a_foreign_filer_from_a_company_with_no_8k(tmp_path):
    foreign = {"form": ["6-K", "20-F", "6-K"], "items": ["", "", ""], "acceptanceDateTime": ["2026-11-01T12:00:00.000Z"] * 3}
    with pytest.raises(Unsupported, match="Form 6-K"):
        edgar(tmp_path, foreign).filings("AAPL")
    fund = {"form": ["N-CSR", "497"], "items": ["", ""], "acceptanceDateTime": ["2026-11-01T12:00:00.000Z"] * 2}
    assert edgar(tmp_path, fund).filings("AAPL") == []
    assert edgar(tmp_path, {}).filings("ZZZZ") == []  # a ticker the SEC does not list


# -- News service -----------------------------------------------------------------


class Feed:
    name = "Alpaca"

    def __init__(self, items=(), error=None):
        self.items, self.error, self.asked = list(items), error, []

    def headlines(self, symbol, since):
        self.asked.append(since)
        if self.error:
            raise self.error
        return [i for i in self.items if i.when >= since]


class Filings:
    name = "SEC EDGAR"

    def __init__(self, items=(), error=None):
        self.items, self.error, self.calls = list(items), error, 0

    def filings(self, symbol):
        self.calls += 1
        if self.error:
            raise self.error
        return self.items


def age(tmp_path, minutes: int) -> None:
    """Make every cached file look as if it was stored `minutes` ago."""
    for path in tmp_path.glob("*.json"):
        raw = json.loads(path.read_text())
        raw["stored_at"] = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
        path.write_text(json.dumps(raw))


def test_the_record_on_disk_keeps_growing_and_only_newer_headlines_are_asked_for(tmp_path):
    old, mid = headline(1, NOON - timedelta(days=200)), headline(2, NOON - timedelta(days=3))
    feed = Feed([old, mid])
    service = NewsService(DiskCache(tmp_path), headline_source=feed)
    assert service.items("ACME", TODAY) == ([mid, old], None, False)
    assert feed.asked == [datetime(2025, 10, 20, tzinfo=timezone.utc)]  # the first request reaches back 400 days

    # Asked again within a quarter of an hour: the copy on disk answers.
    assert NewsService(DiskCache(tmp_path), headline_source=feed).items("ACME", TODAY)[0] == [mid, old]
    assert len(feed.asked) == 1

    # Later, the feed no longer reaches back to the oldest story, and has a new one.
    new = headline(3, NOON)
    feed.items = [mid, new]
    age(tmp_path, 20)
    items, note, failed = service.items("ACME", TODAY)
    assert (items, note, failed) == ([new, mid, old], None, False)  # nothing lost, nothing twice
    assert feed.asked[1] == mid.when  # only from the newest one on file


def test_a_failed_lookup_keeps_the_old_record_and_says_so(tmp_path):
    kept = headline(1, NOON)
    feed = Feed([kept])
    service = NewsService(DiskCache(tmp_path), headline_source=feed)
    service.items("ACME", TODAY)
    age(tmp_path, 20)

    feed.error = AlpacaError(403, "forbidden")
    assert service.items("ACME", TODAY) == ([kept], "Alpaca headlines were unavailable (Alpaca returned 403: forbidden).", True)
    feed.error = httpx.ConnectError("down")
    assert service.items("ACME", TODAY)[1] == "Alpaca headlines were unavailable (ConnectError)."

    # Filings fail on their own, and the headlines still come through.
    both = NewsService(DiskCache(tmp_path / "other"), headline_source=Feed([kept]), filing_source=Filings(error=httpx.ReadTimeout("slow")))
    assert both.items("ACME", TODAY) == ([kept], "SEC EDGAR filings were unavailable (ReadTimeout).", True)


def test_a_company_whose_filings_cannot_be_classified_gets_a_note_and_is_not_a_failure(tmp_path):
    source = Filings(error=Unsupported("SEC EDGAR filings are not classified for a company that reports on Form 6-K."))
    service = NewsService(DiskCache(tmp_path), filing_source=source)
    answer = ([], "SEC EDGAR filings are not classified for a company that reports on Form 6-K.", False)
    assert service.items("TSM", TODAY) == answer
    assert service.items("TSM", TODAY) == answer and source.calls == 1  # remembered like any other answer


def test_filings_and_headlines_come_back_as_one_list_newest_first(tmp_path):
    report = filing(["2.02", "9.01"], when=datetime(2026, 11, 19, 16, 5, tzinfo=ET), ref="0001")
    before, after = headline(1, NOON - timedelta(days=5)), headline(2, NOON)
    service = NewsService(DiskCache(tmp_path), headline_source=Feed([before, after]), filing_source=Filings([report]))
    assert service.sources == ["Alpaca headlines", "SEC EDGAR 8-K filings"]
    assert service.items("ACME", TODAY)[0] == [after, report, before]
    # Read back from disk, a filing still has its item codes.
    assert NewsService(DiskCache(tmp_path)).sources == []
    again = NewsService(DiskCache(tmp_path), headline_source=Feed(), filing_source=Filings())
    assert again.items("ACME", TODAY)[0] == [after, report, before]


def test_news_with_no_source_or_a_spoiled_cache_is_harmless(tmp_path):
    assert NewsService(NullCache()).items("ACME", TODAY) == ([], None, False)

    cache, feed = DiskCache(tmp_path), Feed([headline(1, NOON)])
    cache.set("news_headlines_ACME", [{"title": "a row from some other version"}])
    cache.set("news_filings_ACME", ["not what is expected"])
    service = NewsService(cache, headline_source=feed, filing_source=Filings())
    assert service.items("ACME", TODAY) == ([headline(1, NOON)], None, False)  # fetched afresh, not an error


def test_a_company_name_is_an_extra_that_may_be_missing():
    def down(symbol):
        raise httpx.ConnectError("down")

    assert NewsService(NullCache()).name("ACME") is None
    assert NewsService(NullCache(), names=down).name("ACME") is None
    assert NewsService(NullCache(), names={"ACME": "Acme Corp"}.get).name("ACME") == "Acme Corp"


def test_headlines_older_than_the_price_history_are_let_go(tmp_path):
    ancient, recent = headline(1, NOON - timedelta(days=1300)), headline(2, NOON)
    service = NewsService(DiskCache(tmp_path), headline_source=Feed([recent]))
    DiskCache(tmp_path).set("news_headlines_ACME", [{"when": ancient.when.isoformat(), "kind": "headline", "title": ancient.title, "ref": "1"}])
    age(tmp_path, 20)
    assert [i.ref for i in service.items("ACME", TODAY)[0]] == ["2"]
