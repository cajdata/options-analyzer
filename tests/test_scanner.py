"""End-to-end scans on the synthetic provider, plus the API and the AI brief plumbing."""

import json
import math
import re
from datetime import date, timedelta
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import ai
from app.cache import NullCache
from app.config import Settings
from app.providers.demo import SPECS, DemoMarket
from app.providers.earnings import EarningsService
from app.scanner import PRESETS, ConfigError, Scanner, ScanRequest, build_scanner
from app.server import ScanBody, create_app

TODAY = date(2026, 11, 24)


def demo_scanner(**settings) -> Scanner:
    market = DemoMarket(TODAY)
    earnings = EarningsService(NullCache(), history_sources=[market], calendar_source=market, today=TODAY)
    return Scanner(market, earnings, Settings(**settings), universe=list(SPECS), today=TODAY)


@pytest.fixture(scope="module", autouse=True)
def demo_day_is_pinned():
    """Hold the demo market at TODAY for the command line and the dashboard.

    Both build their own demo market, which starts from the wall clock, and several tests
    read a date or a rating off their output. Module scope puts this in place before the
    `client` fixture builds its app.
    """
    wall_clock = DemoMarket.__init__

    def pinned(self, today=None):
        wall_clock(self, today or TODAY)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(DemoMarket, "__init__", pinned)
        yield


@pytest.fixture(scope="module")
def discovery():
    return demo_scanner().scan(ScanRequest())


def test_discovery_returns_a_varied_ranked_list(discovery):
    ideas = discovery.ideas
    assert discovery.mode == "demo" and discovery.lens == "buy"
    assert len(ideas) >= 8
    assert [c.score for c in ideas] == sorted((c.score for c in ideas), reverse=True)
    sides = [(c.underlying, c.kind) for c in ideas]
    assert len(sides) == len(set(sides))  # at most one call and one put per stock
    assert {c.underlying for c in ideas} <= set(SPECS)
    for c in ideas:
        assert 7 <= c.dte <= 60
        assert 0 <= c.score <= 100 and 0 < c.pop < 1
        assert c.bid <= c.mid <= c.ask


def test_discovery_explains_why_each_stock_was_scanned(discovery):
    bolt = discovery.underlyings["BOLT"]
    assert "most active today" in bolt.reasons and "top mover today" in bolt.reasons
    assert any(r.startswith("earnings in") for r in bolt.reasons)
    assert any(r.startswith("moved +6.4%") for r in bolt.reasons)


def test_earnings_context(discovery):
    acme = discovery.underlyings["ACME"]
    assert acme.next_earnings.confirmed and acme.days_to_earnings == 9
    assert acme.earnings_stats.count == 12 and len(acme.reactions) == 12
    assert acme.earnings_stats.up + acme.earnings_stats.down == 12
    # The synthetic chains price each report at a known size (ACME 70% of its
    # typical move, BOLT 140%). The scanner should recover that from quotes alone:
    # the report-day move is the priced-in jump plus one ordinary day.
    def built_in(symbol):
        spec = SPECS[symbol]
        jump, ordinary = spec["jump"] * spec["pricing"], spec["vol"] * 1.08
        return math.sqrt(jump**2 + ordinary**2 / 252) * math.sqrt(2 / math.pi)

    assert acme.implied_move.event_move_pct == pytest.approx(built_in("ACME"), rel=0.02)
    assert acme.implied_move.baseline_source == "the next expiry after it"
    bolt = discovery.underlyings["BOLT"]
    assert bolt.implied_move.event_move_pct == pytest.approx(built_in("BOLT"), rel=0.02)
    assert acme.implied_move.event_move_pct < acme.earnings_stats.mean_abs  # priced below its history

    # EMBR reports in 71 days: beyond the calendar and beyond every scanned expiry.
    embr = discovery.underlyings["EMBR"]
    assert not embr.next_earnings.confirmed
    assert embr.implied_move is None
    assert not any(c.spans_earnings for c in discovery.ideas if c.underlying == "EMBR")
    assert any(c.spans_earnings for c in discovery.ideas if c.underlying == "ACME")


def test_volatility_and_greeks_are_computed_when_the_provider_omits_them(discovery):
    embr = [c for c in discovery.ideas if c.underlying == "EMBR"]
    assert embr and all(c.iv_source == "computed" and c.greeks_source == "computed" for c in embr)
    assert all(0.3 < c.iv < 0.8 for c in embr)  # the synthetic chain is built around 52%
    acme = [c for c in discovery.ideas if c.underlying == "ACME"]
    assert all(c.iv_source == "provider" for c in acme)


def test_requested_symbols_get_depth_and_filters_apply():
    result = demo_scanner().scan(ScanRequest(symbols=["ACME"], lens="sell", kind="put", top_n=5, dte_max=30))
    assert len(result.ideas) == 5
    assert {(c.underlying, c.kind) for c in result.ideas} == {("ACME", "put")}
    assert all(c.dte <= 30 for c in result.ideas)
    assert all(c.pop == pytest.approx(1 - c.pop_long) for c in result.ideas)
    assert set(result.ideas[0].components) == {"liquidity", "probability", "vol_value", "yield", "delta_fit"}
    assert result.underlyings["ACME"].reasons == ["requested"]


def test_unknown_symbols_are_reported_not_fatal():
    result = demo_scanner().scan(ScanRequest(symbols=["ACME", "NOPE"]))
    assert "NOPE: no stock quote found." in result.warnings
    assert {c.underlying for c in result.ideas} == {"ACME"}


def test_watchlist_symbols_survive_the_symbol_cap():
    result = demo_scanner(watchlist=("ORBT",)).scan(ScanRequest(max_symbols=2))
    assert "ORBT" in result.underlyings and len(result.underlyings) == 2
    assert "watchlist" in result.underlyings["ORBT"].reasons


def test_a_failing_symbol_does_not_sink_the_scan():
    scanner = demo_scanner()
    real = scanner.market.option_chain

    def flaky(symbol, **window):
        if symbol == "BOLT":
            raise RuntimeError("chain unavailable")
        return real(symbol, **window)

    scanner.market.option_chain = flaky
    result = scanner.scan(ScanRequest())
    assert "BOLT: chain unavailable" in result.warnings
    assert "BOLT" not in result.underlyings and result.ideas


def test_stocks_with_no_option_quotes_are_named_in_one_note():
    scanner = demo_scanner()
    real = scanner.market.option_chain
    scanner.market.option_chain = lambda symbol, **window: [] if symbol in ("ACME", "BOLT") else real(symbol, **window)
    result = scanner.scan(ScanRequest(symbols=["BOLT", "ACME", "FLUX"]))
    # A live discovery scan turns up several of these at once (warrants, tiny companies).
    assert "No two-sided option quotes in range for ACME, BOLT." in result.warnings
    assert {c.underlying for c in result.ideas} == {"FLUX"}


def test_live_mode_needs_alpaca_keys():
    with pytest.raises(ConfigError, match="--demo"):
        build_scanner(Settings())


# -- HTTP API ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(Settings(), demo=True))


def test_api_status_and_page(client):
    status = client.get("/api/status").json()
    assert status["mode"] == "demo" and status["setup_error"] is None
    assert status["earnings_sources"] == ["demo data"]
    assert set(status["lenses"]) == {"buy", "sell"}
    assert "Strike Check" in client.get("/").text


def test_api_scan_returns_json_the_page_can_render(client):
    response = client.post("/api/scan", json={"symbols": ["acme", " flux "], "lens": "buy", "top_n": 6})
    assert response.status_code == 200
    body = response.json()
    assert len(body["ideas"]) == 6 and set(body["underlyings"]) == {"ACME", "FLUX"}
    idea = body["ideas"][0]
    assert idea["expiry"][:2] == "20" and isinstance(idea["components"], dict)
    assert body["underlyings"]["ACME"]["next_earnings"]["day"]


@pytest.mark.parametrize("body", [{"symbols": ["DROP TABLE"]}, {"lens": "hold"}, {"dte_min": 40, "dte_max": 10}, {"top_n": 0}])
def test_api_rejects_bad_input(client, body):
    assert client.post("/api/scan", json=body).status_code == 422


def test_api_without_keys_explains_the_setup():
    client = TestClient(create_app(Settings(), demo=False))
    assert "ALPACA_API_KEY" in client.get("/api/status").json()["setup_error"]
    assert client.post("/api/scan", json={}).status_code == 503


def test_brief_for_an_unknown_scan_is_a_404(client):
    assert client.post("/api/brief/nope").status_code == 404


# -- AI brief ---------------------------------------------------------------------


class FakeClaude:
    """Stands in for the Anthropic client; records the request and returns a canned answer."""

    def __init__(self, brief, stop_reason="end_turn"):
        self.requests = []
        answer = SimpleNamespace(stop_reason=stop_reason, parsed_output=brief, model="claude-opus-5-5")

        def parse(**request):
            self.requests.append(request)
            return answer

        self.messages = SimpleNamespace(parse=parse)
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=parse))


def pick(contract):
    return ai.Pick(contract=contract, headline="h", rationale="r", risks="k", check_first="c")


def test_scan_payload_is_compact_json_with_the_caveats(discovery):
    payload = json.loads(json.dumps(ai.scan_payload(discovery)))
    assert payload["lens"] == "buy" and "synthetic" in payload["data"]
    assert len(payload["ranked_contracts"]) == len(discovery.ideas)
    acme = payload["underlyings"]["ACME"]
    assert acme["next_earnings"]["date_is_confirmed"] is True
    assert acme["past_earnings_moves"]["reports"] == 12
    assert "implied_one_day_abs_move_pct" in acme["options_pricing_for_next_earnings"]
    assert payload["underlyings"]["EMBR"]["next_earnings"]["date_is_confirmed"] is False


def test_brief_sends_the_scan_and_keeps_only_real_contracts(discovery):
    real = discovery.ideas[0].symbol
    fake = FakeClaude(ai.Brief(summary="Two setups stand out.", picks=[pick(real), pick("MADEUP261218C00100000")]))
    brief = ai.write_brief(discovery, client=fake)

    assert brief.summary == "Two setups stand out." and brief.model == "claude-opus-5-5"
    assert [p["contract"] for p in brief.picks] == [real]  # the invented contract is dropped

    request = fake.requests[0]
    assert request["model"] == "claude-opus-5-5" and request["output_format"] is ai.Brief
    assert request["fallbacks"] == "default" and request["betas"] == [ai.FALLBACK_BETA]
    assert real in request["messages"][0]["content"]
    assert "thinking" not in request and "temperature" not in request


def test_brief_on_another_model_skips_the_fallback_parameters(discovery):
    fake = FakeClaude(ai.Brief(summary="s", picks=[]))
    ai.write_brief(discovery, model="claude-haiku-5-5", client=fake)
    assert "fallbacks" not in fake.requests[0] and "betas" not in fake.requests[0]


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_brief_surfaces_declined_or_truncated_answers(discovery, stop_reason):
    fake = FakeClaude(ai.Brief(summary="s", picks=[]), stop_reason=stop_reason)
    with pytest.raises(ai.BriefError):
        ai.write_brief(discovery, client=fake)


def test_brief_without_ideas_makes_no_model_call(discovery):
    empty = SimpleNamespace(ideas=[])
    assert "nothing to write up" in ai.write_brief(empty, client=None).summary


# -- Command line -----------------------------------------------------------------


def test_cli_scan_prints_a_table_and_earnings_context(capsys):
    from app.__main__ import main

    assert main(["scan", "ACME", "BOLT", "--demo", "--top", "4"]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC DEMO DATA" in out and "Earnings context" in out
    assert "ACME: past 12 reports averaged" in out and "options price about" in out
    assert out.count(" call ") + out.count(" put ") == 4


def test_cli_without_keys_points_to_the_setup(capsys, monkeypatch):
    from app import __main__ as cli
    from app.config import Settings

    monkeypatch.setattr(cli, "load_settings", lambda: Settings())
    assert cli.main(["scan"]) == 2
    assert "ALPACA_API_KEY" in capsys.readouterr().err


# -- Risk and reward ------------------------------------------------------------


def test_every_idea_is_rated_and_the_rating_agrees_with_the_volatility_score(discovery):
    for c in discovery.ideas:
        rr = c.risk_reward
        assert rr is not None and rr.rating in {"favorable", "balanced", "unfavorable"}
        assert 0 < rr.win_prob < 1 and rr.avg_win > 0 and rr.avg_loss > 0
        assert rr.max_loss == pytest.approx(c.outlay)  # a buyer can lose what the contract cost, no more
        assert rr.history_vol == c.history_vol
        if rr.edge_pct > 0:  # worth more than the ask only if the options are priced below history
            assert c.iv < c.history_vol


def test_rating_reflects_how_each_report_is_priced():
    # ACME's options price its report at 70% of the usual move, BOLT's at 140%.
    buy = demo_scanner().scan(ScanRequest(symbols=["ACME", "BOLT"], kind="call", dte_max=20))
    spanning = {s: [c for c in buy.ideas if c.underlying == s and c.spans_earnings] for s in ("ACME", "BOLT")}
    assert spanning["ACME"] and spanning["BOLT"]
    assert all(c.risk_reward.rating == "favorable" for c in spanning["ACME"])
    assert all(c.risk_reward.rating == "unfavorable" for c in spanning["BOLT"])

    sell = demo_scanner().scan(ScanRequest(symbols=["BOLT"], lens="sell", kind="put", dte_max=20))
    rated = [c.risk_reward for c in sell.ideas if c.spans_earnings]
    assert rated and all(r.rating == "favorable" for r in rated)
    assert all(r.bad_case_loss > r.avg_win for r in rated)  # the tail is still larger than a typical win


def test_risk_reward_reaches_the_api_the_brief_and_the_command_line(client, discovery, capsys):
    from app.__main__ import main

    idea = client.post("/api/scan", json={"symbols": ["ACME"], "top_n": 3}).json()["ideas"][0]
    assert set(idea["risk_reward"]) >= {"rating", "ratio", "win_prob", "avg_win", "avg_loss", "expected_value", "max_loss"}

    block = ai.scan_payload(discovery)["ranked_contracts"][0]["risk_reward_if_history_repeats"]
    assert block["rating"] in {"favorable", "balanced", "unfavorable"} and block["reward_to_risk"] > 0
    assert "risk_reward_if_history_repeats" in ai.SYSTEM

    # Short-dated ACME calls span a report the options price cheaply, so they rate favorable.
    assert main(["scan", "ACME", "--demo", "--type", "call", "--dte-max", "20", "--top", "3"]) == 0
    out = capsys.readouterr().out
    assert "R:R" in out and "Rating" in out
    assert re.search(r"\bfavorable\b", out) and "unfavorable" not in out


# -- Budget -----------------------------------------------------------------------


def test_budget_keeps_only_contracts_a_buyer_can_afford(discovery):
    result = demo_scanner().scan(ScanRequest(budget=100))
    assert result.budget == 100 and result.ideas
    for c in result.ideas:
        assert c.outlay == pytest.approx(c.ask * 100) and c.outlay <= 100
    assert max(c.outlay for c in discovery.ideas) > 1000  # the same scan without a budget is far pricier
    assert all(c.outlay == pytest.approx(c.ask * 100) for c in discovery.ideas)
    # Cheap contracts sit far from the money, and the scan says so.
    assert any(c.risk_reward.shape == "long shot" for c in result.ideas)
    assert max(c.score for c in result.ideas) < max(c.score for c in discovery.ideas)


def test_budget_for_a_seller_is_the_collateral():
    puts = demo_scanner().scan(ScanRequest(lens="sell", kind="put", budget=3000))
    assert puts.ideas and {c.underlying for c in puts.ideas} == {"EMBR"}  # the only stock under $30
    assert all(c.outlay == pytest.approx(c.strike * 100) for c in puts.ideas)
    assert all(c.outlay <= 3000 for c in puts.ideas)

    calls = demo_scanner().scan(ScanRequest(symbols=["EMBR"], lens="sell", kind="call", budget=3000))
    assert calls.ideas and all(c.outlay == pytest.approx(27.80 * 100) for c in calls.ideas)  # 100 shares


def test_a_budget_nothing_fits_says_what_the_cheapest_contract_costs():
    result = demo_scanner().scan(ScanRequest(budget=5))
    assert result.ideas == []
    note = next(w for w in result.warnings if w.startswith("Nothing fits a $5 budget."))
    assert "The cheapest contract that passed the other filters needs $" in note


def test_discovery_under_a_budget_leans_toward_cheaper_stocks():
    unlimited = demo_scanner().scan(ScanRequest(max_symbols=2))
    assert "EMBR" not in unlimited.underlyings
    limited = demo_scanner().scan(ScanRequest(max_symbols=2, budget=60))
    assert "EMBR" in limited.underlyings  # at $27.80 it is the only stock with near-the-money contracts under $60


def test_budget_reaches_the_api_the_brief_and_the_command_line(client, capsys):
    from app.__main__ import main

    body = client.post("/api/scan", json={"budget": 100}).json()
    assert body["budget"] == 100 and body["ideas"] and all(i["outlay"] <= 100 for i in body["ideas"])
    for bad in (0, -50, "cheap"):
        assert client.post("/api/scan", json={"budget": bad}).status_code == 422

    result = demo_scanner().scan(ScanRequest(budget=100))
    payload = ai.scan_payload(result)
    assert payload["budget_per_contract_usd"] == 100
    first = payload["ranked_contracts"][0]
    assert first["cash_to_open_one_contract_usd"] <= 100
    assert first["risk_reward_if_history_repeats"]["payoff_shape"] in {"ordinary", "long shot", "tail risk"}

    assert main(["scan", "--demo", "--budget", "100", "--top", "5"]) == 0
    out = capsys.readouterr().out
    assert "Budget: $100 per contract" in out and "Outlay" in out and "long shot" in out


# -- Found in review ----------------------------------------------------------------


def test_duplicate_and_untidy_symbols_are_scanned_once():
    result = demo_scanner().scan(ScanRequest(symbols=["ACME", " acme ", "ACME"], top_n=50))
    contracts = [c.symbol for c in result.ideas]
    assert contracts and len(contracts) == len(set(contracts))


def test_strike_window_widens_for_a_volatile_stock():
    request = ScanRequest(symbols=["BOLT", "ORBT"], lens="sell", kind="put", dte_min=45, dte_max=60, top_n=500, per_side=500)
    result = demo_scanner().scan(request)

    def lowest_strike(symbol):
        return min(c.strike for c in result.ideas if c.underlying == symbol) / result.underlyings[symbol].price

    assert lowest_strike("BOLT") < 0.75  # 55% volatility: the search reaches well past 15% below the price
    assert lowest_strike("ORBT") > 0.84  # 18% volatility: the 15% minimum is plenty
    bolt = [abs(c.delta) for c in result.ideas if c.underlying == "BOLT"]
    assert min(bolt) < 0.15  # the whole 0.15 to 0.35 delta band a seller looks for is in reach


class _Source:
    """A stand-in earnings source that answers, fails, or has nothing."""

    name = "SEC EDGAR"

    def __init__(self, answer):
        self.answer = answer

    def history(self, symbol):
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _scan_with(sources, symbols=("ACME", "BOLT")):
    earnings = EarningsService(NullCache(), history_sources=sources, today=TODAY)
    scanner = Scanner(DemoMarket(TODAY), earnings, Settings(), universe=list(SPECS), today=TODAY)
    return scanner.scan(ScanRequest(symbols=list(symbols)))


def test_ratings_say_when_no_earnings_date_is_known():
    # ACME reports in nine days. If the lookup fails, a rating cannot see that coming.
    failed = _scan_with([_Source(httpx.ConnectError("down"))])
    assert all(u.earnings_unknown for u in failed.underlyings.values())
    assert any(w.startswith("No earnings date is known for ACME, BOLT") for w in failed.warnings)
    assert failed.ideas and all(c.risk_reward for c in failed.ideas)  # still rated, with the caveat attached
    assert ai.scan_payload(failed)["underlyings"]["ACME"]["earnings_date_unknown"] is True

    unconfigured = _scan_with([])
    assert "No earnings data source is configured, so no rating accounts for an upcoming report." in unconfigured.warnings

    # A security with no reports on file at all (an ETF) is a different case and is not flagged.
    none_on_file = _scan_with([_Source([])])
    assert not any(u.earnings_unknown for u in none_on_file.underlyings.values())
    assert not any("earnings" in w.lower() for w in none_on_file.warnings)


def test_a_stock_with_nothing_to_price_is_not_looked_up_any_further():
    asked, interest = [], []

    class Counting(_Source):
        def history(self, symbol):
            asked.append(symbol)
            return super().history(symbol)

    # The lookup fails, which would put a stock in the "no earnings date is known" note.
    earnings = EarningsService(NullCache(), history_sources=[Counting(httpx.ConnectError("down"))], today=TODAY)
    scanner = Scanner(DemoMarket(TODAY), earnings, Settings(), universe=list(SPECS), today=TODAY)
    real = scanner.market.option_chain

    def chain(symbol, **window):
        quotes = [] if symbol == "BOLT" else real(symbol, **window)  # BOLT: no options at all
        for q in quotes:
            q.open_interest = None  # as a live feed sends it, so the scanner has to ask
            if symbol == "FLUX":
                q.bid = 0.0  # FLUX: contracts, but nobody bids for any of them
        return quotes

    def open_interest(symbol, **window):
        interest.append(symbol)
        return {}

    scanner.market.option_chain, scanner.market.open_interest = chain, open_interest
    result = scanner.scan(ScanRequest(symbols=["ACME", "BOLT", "FLUX"]))

    # A live discovery scan picks up warrants and tiny companies like these. Nothing is
    # shown for them, so they must not cost an earnings lookup (Alpha Vantage allows
    # about 25 a day) or an open-interest request, nor be named in a note about ratings.
    assert asked == ["ACME"] and interest == ["ACME"]
    assert "No two-sided option quotes in range for BOLT, FLUX." in result.warnings
    assert "No earnings date is known for ACME, so ratings there do not account for a report." in result.warnings
    assert {c.underlying for c in result.ideas} == {"ACME"} and all(c.open_interest is None for c in result.ideas)


def test_a_provider_caveat_is_reported_on_every_scan():
    scanner = demo_scanner()
    real = scanner.market.daily_bars

    def with_caveat(symbols, start):
        scanner.market.notes.append("price history caveat")
        return real(symbols, start)

    scanner.market.daily_bars = with_caveat
    for _ in range(2):
        assert scanner.scan(ScanRequest(symbols=["ACME"])).warnings.count("price history caveat") == 1


def test_an_open_interest_rejection_points_at_the_paper_setting():
    scanner = demo_scanner()
    real = scanner.market.option_chain

    def without_interest(symbol, **window):
        chain = real(symbol, **window)
        for q in chain:
            q.open_interest = None
        return chain

    def rejected(symbol, **window):
        err = RuntimeError("unauthorized")
        err.status = 401
        raise err

    scanner.market.option_chain, scanner.market.open_interest = without_interest, rejected
    result = scanner.scan(ScanRequest(symbols=["ACME"]))
    note = next(w for w in result.warnings if w.startswith("Open interest was unavailable"))
    assert "ALPACA_PAPER" in note and result.ideas


@pytest.mark.parametrize(
    "flags", [["--top", "0"], ["--budget", "0"], ["--budget", "-5"], ["--dte-min", "40", "--dte-max", "10"], ["--dte-min", "-1"]]
)
def test_cli_rejects_impossible_options(flags, capsys):
    from app.__main__ import main

    with pytest.raises(SystemExit) as stop:
        main(["scan", "--demo", *flags])
    assert stop.value.code == 2
    capsys.readouterr()


def test_cli_json_with_a_brief_is_one_valid_document(capsys, monkeypatch):
    from app import __main__ as cli

    written = ai.BriefResult(summary="Two setups.", picks=[], model="test-model")
    monkeypatch.setattr(cli.ai, "write_brief", lambda result, **options: written)
    assert cli.main(["scan", "ACME", "--demo", "--top", "2", "--json", "--brief"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert len(document["ideas"]) == 2 and document["brief"]["summary"] == "Two setups."

    def no_key(result, **options):
        raise ai.BriefError("no key")

    monkeypatch.setattr(cli.ai, "write_brief", no_key)
    assert cli.main(["scan", "ACME", "--demo", "--top", "2", "--json", "--brief"]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["brief"] is None and "no key" in captured.err


def test_settings_are_cleaned_up_and_bad_ones_reported(capsys, monkeypatch):
    from app.__main__ import main
    from app.config import ConfigError, load_settings

    monkeypatch.setenv("ALPACA_OPTIONS_FEED", "OPRA")
    monkeypatch.setenv("AI_EFFORT", "None")
    settings = load_settings()
    assert settings.options_feed == "opra" and settings.ai_effort is None

    for bad in ("four percent", "4"):  # not a number, and 400% where 4% was meant
        monkeypatch.setenv("RISK_FREE_RATE", bad)
        with pytest.raises(ConfigError, match="RISK_FREE_RATE"):
            load_settings()
    assert main(["scan", "--demo"]) == 2
    assert "RISK_FREE_RATE" in capsys.readouterr().err


def test_api_accepts_share_classes_and_rejects_non_tickers():
    assert ScanBody(symbols=["brk-b", "BRK.B", " f "]).symbols == ["BRK.B", "F"]
    for bad in ("..A", "TOOLONGX", "A.BCDE", "A B", "A/B"):
        with pytest.raises(ValidationError):
            ScanBody(symbols=[bad])


def test_brief_is_off_without_credentials(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    assert ai.available() is False  # the button stays hidden, as .env.example promises
    assert ai.available("sk-test") is True
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token")
    assert ai.available() is True


def test_brief_effort_can_be_left_out_for_a_model_that_rejects_it(discovery):
    fake = FakeClaude(ai.Brief(summary="s", picks=[]))
    ai.write_brief(discovery, client=fake, effort=None)
    assert "output_config" not in fake.requests[0]
    ai.write_brief(discovery, client=fake, effort="low")
    assert fake.requests[1]["output_config"] == {"effort": "low"}


# -- Found in the follow-up review ----------------------------------------------------


def test_share_classes_are_normalized_on_every_way_in():
    # The command line passes symbols straight through, so the scanner must tidy them itself.
    scanner = demo_scanner()
    asked = []
    real = scanner.market.stock_snapshots

    def record(symbols):
        asked.extend(symbols)
        return real(symbols)

    scanner.market.stock_snapshots = record
    result = scanner.scan(ScanRequest(symbols=["brk-b", "BRK.B", " acme "]))
    assert asked == ["BRK.B", "ACME"]
    assert "BRK.B: no stock quote found." in result.warnings


def test_a_foreign_filer_is_flagged_not_treated_like_an_etf():
    from app.providers.earnings import Unsupported

    foreign = _scan_with([_Source(Unsupported("SEC EDGAR cannot date earnings for a company that reports on Form 6-K."))])
    assert all(u.earnings_unknown for u in foreign.underlyings.values())
    assert any(w.startswith("No earnings date is known for ACME, BOLT") for w in foreign.warnings)
    assert "Form 6-K" in foreign.underlyings["ACME"].earnings_note


def test_a_blank_credential_does_not_turn_the_brief_on(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    assert ai.available() is False


# -- Exit plan and the premium-selling preset ------------------------------------------


def test_every_idea_gets_an_exit_plan(discovery):
    assert set(discovery.exit_plans) == {c.symbol for c in discovery.ideas}
    for c in discovery.ideas:
        plan = discovery.exit_plans[c.symbol]
        assert plan.lens == "buy" and plan.entry_price == c.ask
        assert plan.checkpoints[-1].label == "at expiry" and plan.checkpoints[-1].day == c.expiry
        assert [k.day for k in plan.checkpoints] == sorted(k.day for k in plan.checkpoints)
        assert plan.target_price is None  # a seller's rule
        # Standing still, a buyer only loses; and selling early never beats the price paid.
        assert all(k.pnl < 0 for k in plan.checkpoints)

    spanning = [c for c in discovery.ideas if c.underlying == "ACME" and c.spans_earnings]
    assert spanning
    for c in spanning:
        step = discovery.exit_plans[c.symbol].earnings
        assert step is not None and step.crush_pct < 0
        assert step.report_day == date(2026, 12, 4)  # ACME reports after the close on 3 December
    quiet = [c for c in discovery.ideas if c.underlying == "EMBR"]
    assert quiet and all(discovery.exit_plans[c.symbol].earnings is None for c in quiet)


def test_context_carries_what_a_later_repricing_needs(discovery):
    acme = discovery.underlyings["ACME"]
    assert acme.ordinary_vol == pytest.approx(SPECS["ACME"]["vol"], rel=0.25)  # realized, with earnings days removed
    squares = [math.log1p(m.move_pct) ** 2 for m in acme.reactions]
    assert acme.event_var == pytest.approx(sum(squares) / len(squares))  # variance of past report-day moves
    assert acme.implied_move.event_var > 0
    points = acme.term_structure
    assert [p.expiry for p in points] == sorted(p.expiry for p in points) and len(points) >= 5
    assert all(7 <= p.dte <= 60 and 0.1 < p.atm_iv < 1.5 for p in points)
    # The report lifts every expiry that holds it, the nearest ones most.
    assert points[0].spans_earnings and points[0].atm_iv > points[-1].atm_iv


def test_premium_selling_preset():
    settings = PRESETS["sell-premium"]["settings"]
    result = demo_scanner().scan(ScanRequest(**settings))
    assert result.lens == "sell" and result.ideas
    for c in result.ideas:
        assert 30 <= c.dte <= 45
        assert 0.16 <= abs(c.delta) <= 0.30
        assert not c.spans_earnings
        plan = result.exit_plans[c.symbol]
        assert plan.target_price == pytest.approx(c.bid / 2)
        assert plan.time_stop_day == c.expiry - timedelta(days=21)
    # Stocks reporting inside the window have nothing left to offer under this preset.
    assert not {"ACME", "BOLT", "FLUX"} & {c.underlying for c in result.ideas}


def test_delta_and_earnings_filters_work_on_their_own():
    scanner = demo_scanner()
    narrow = scanner.scan(ScanRequest(symbols=["ACME"], delta_min=0.40, delta_max=0.50, top_n=50))
    assert narrow.ideas and all(0.40 <= abs(c.delta) <= 0.50 for c in narrow.ideas)

    everything = scanner.scan(ScanRequest(symbols=["ACME"], dte_min=1, top_n=50, per_side=50))
    quiet = scanner.scan(ScanRequest(symbols=["ACME"], dte_min=1, top_n=50, per_side=50, avoid_earnings=True))
    assert any(c.spans_earnings for c in everything.ideas)
    assert quiet.ideas and not any(c.spans_earnings for c in quiet.ideas)
    assert all(c.dte <= 9 for c in quiet.ideas)  # only what expires before the report in nine days


def test_exit_plan_and_preset_reach_the_api(client):
    status = client.get("/api/status").json()
    assert status["exit_rules"] == {"profit_target": 0.5, "time_stop_dte": 21}
    preset = status["presets"]["sell-premium"]
    assert preset["settings"]["avoid_earnings"] is True and preset["label"] and preset["about"]

    body = client.post("/api/scan", json=preset["settings"]).json()
    assert body["ideas"] and set(body["exit_plans"]) == {i["symbol"] for i in body["ideas"]}
    plan = body["exit_plans"][body["ideas"][0]["symbol"]]
    assert plan["lens"] == "sell" and plan["target_price"] > 0 and plan["checkpoints"][-1]["label"] == "at expiry"
    assert all(0.16 <= abs(i["delta"]) <= 0.30 for i in body["ideas"])

    assert client.post("/api/scan", json={"delta_min": 0.5, "delta_max": 0.2}).status_code == 422
    assert client.post("/api/scan", json={"delta_max": 1.5}).status_code == 422


def test_exit_plan_and_preset_on_the_command_line(capsys):
    from app.__main__ import main

    assert main(["scan", "--demo", "--preset", "sell-premium", "--top", "3", "--exits"]) == 0
    out = capsys.readouterr().out
    assert "Lens: sell premium" in out and "Exit plan, if the stock stands still" in out
    assert "buy back at $" in out and "time stop" in out and "They are not probabilities" in out

    # An option given alongside a preset overrides it.
    assert main(["scan", "--demo", "--preset", "sell-premium", "--dte-min", "20", "--dte-max", "29", "--json"]) == 0
    ideas = json.loads(capsys.readouterr().out)["ideas"]
    assert ideas and all(20 <= i["dte"] <= 29 and i["lens"] == "sell" for i in ideas)

    assert main(["scan", "ACME", "--demo", "--top", "2", "--exits"]) == 0
    out = capsys.readouterr().out
    assert "half the time value is gone by" in out and "earnings Dec 04" in out

    for flags in (["--delta-min", "0.5", "--delta-max", "0.2"], ["--delta-max", "1.5"]):
        with pytest.raises(SystemExit):
            main(["scan", "--demo", *flags])
    capsys.readouterr()


def test_the_brief_is_given_the_exit_plan(discovery):
    payload = json.loads(json.dumps(ai.scan_payload(discovery)))
    plan = payload["ranked_contracts"][0]["exit_plan"]
    assert plan["if_the_stock_stands_still"][-1]["when"] == "at expiry"
    assert "time_stop_date" in plan and "strike_vs_stock_pct" in plan
    spanning = next(c for c in payload["ranked_contracts"] if c["expires_after_next_earnings"])
    assert spanning["exit_plan"]["across_the_earnings_report"]["price_change_with_stock_unchanged_pct"] < 0
    assert "exit_plan" in ai.SYSTEM and "never the chance of getting there" in ai.SYSTEM

    sold = demo_scanner().scan(ScanRequest(**PRESETS["sell-premium"]["settings"]))
    plan = ai.scan_payload(sold)["ranked_contracts"][0]["exit_plan"]
    assert plan["profit_target_buy_back_at_or_below"] > 0 and "decay_alone_reaches_target_on" in plan


# -- Found in the exit-plan review -------------------------------------------------------


def test_discovery_does_not_spend_its_slots_on_stocks_skip_earnings_will_discard():
    # BOLT, ACME and FLUX report within 16 days, so under the preset (30 to 45 days out,
    # nothing open over earnings) every contract of theirs is discarded. They used to be
    # picked first, because an upcoming report earned extra points.
    preset = PRESETS["sell-premium"]["settings"]
    result = demo_scanner().scan(ScanRequest(**preset, max_symbols=2))
    assert result.ideas  # the two slots went to stocks that have something to offer
    assert not {"BOLT", "ACME", "FLUX"} & set(result.underlyings)
    assert any(w.startswith("Skipped ACME, BOLT, FLUX: they report before any contract") for w in result.warnings)

    # CRUX reports in 38 days: its nearest expiry in range comes first, so it stays.
    everything = demo_scanner().scan(ScanRequest(**preset))
    assert "CRUX" in everything.underlyings
    assert all(c.expiry < date(2027, 1, 1) for c in everything.ideas if c.underlying == "CRUX")

    # Without the filter, an upcoming report is still a reason to look.
    plain = demo_scanner().scan(ScanRequest(max_symbols=2))
    assert "BOLT" in plain.underlyings


def test_skip_earnings_leaves_out_a_stock_whose_report_date_is_unknown():
    preset = PRESETS["sell-premium"]["settings"]

    def scan(sources):
        earnings = EarningsService(NullCache(), history_sources=sources, today=TODAY)
        scanner = Scanner(DemoMarket(TODAY), earnings, Settings(), universe=list(SPECS), today=TODAY)
        return scanner.scan(ScanRequest(**preset, symbols=["ACME", "EMBR"]))

    # ACME reports in nine days. With the lookup down, nothing says so, and its contracts used to pass.
    failed = scan([_Source(httpx.ConnectError("down"))])
    assert failed.ideas == []
    assert any(w.startswith("Left out ACME, EMBR: no earnings date is known") for w in failed.warnings)
    assert not any("ratings there do not account" in w for w in failed.warnings)  # said once, not twice

    unconfigured = scan([])
    assert unconfigured.ideas == []
    assert any(w.startswith("Nothing can be shown as clear of an earnings report") for w in unconfigured.warnings)

    # A security with no reports on file at all (an ETF) has nothing to avoid and still passes.
    assert scan([_Source([])]).ideas


def test_skip_earnings_explains_a_requested_stock_with_nothing_left():
    preset = PRESETS["sell-premium"]["settings"]
    result = demo_scanner().scan(ScanRequest(**preset, symbols=["ACME"]))
    assert result.ideas == []
    assert "Left out ACME: every contract in this range is still open over the next earnings report." in result.warnings


def test_skip_earnings_gives_an_estimated_date_room():
    # ORBT's next report is an estimate: 18 January, after the close. A contract expiring three
    # days before an estimate may well be open over the real report, so skip-earnings drops it.
    scanner = demo_scanner()
    request = dict(symbols=["ORBT"], lens="sell", dte_min=40, dte_max=60, top_n=200, per_side=200)
    every = scanner.scan(ScanRequest(**request))
    assert not every.underlyings["ORBT"].next_earnings.confirmed
    assert every.underlyings["ORBT"].next_earnings.day == date(2027, 1, 18)
    assert {c.expiry for c in every.ideas} == {date(2027, 1, 8), date(2027, 1, 15), date(2027, 1, 22)}
    assert {c.expiry for c in every.ideas if c.spans_earnings} == {date(2027, 1, 22)}

    careful = scanner.scan(ScanRequest(**request, avoid_earnings=True))
    assert {c.expiry for c in careful.ideas} == {date(2027, 1, 8)}

    # A confirmed date needs no such room: CRUX reports before the open on 1 January,
    # and a contract expiring a week before that is kept.
    crux = scanner.scan(ScanRequest(symbols=["CRUX"], lens="sell", dte_min=25, dte_max=45, avoid_earnings=True, top_n=200, per_side=200))
    assert crux.underlyings["CRUX"].next_earnings.confirmed
    assert {c.expiry for c in crux.ideas} == {date(2026, 12, 25)}


def test_a_date_from_the_calendar_is_known_even_if_the_history_lookup_failed():
    market = DemoMarket(TODAY)
    earnings = EarningsService(NullCache(), history_sources=[_Source(httpx.ConnectError("down"))], calendar_source=market, today=TODAY)
    result = Scanner(market, earnings, Settings(), universe=list(SPECS), today=TODAY).scan(ScanRequest(symbols=["ACME"]))
    acme = result.underlyings["ACME"]
    assert acme.next_earnings.confirmed and acme.days_to_earnings == 9
    assert acme.earnings_unknown is False and acme.earnings_note  # the date is known; the note says history is missing
    assert not any("No earnings date is known" in w for w in result.warnings)


def test_cli_exit_line_shows_the_time_stop_result_and_speaks_for_the_seller(capsys):
    from app.__main__ import _exit_line, main
    from app.analysis import exits

    # A 35-day contract: the time stop lands 14 days out, on a date the path also samples.
    scanner = demo_scanner()
    result = scanner.scan(ScanRequest(symbols=["EMBR"], lens="sell", kind="put", dte_min=30, dte_max=45, top_n=200, per_side=200))
    c = next(c for c in result.ideas if c.dte == 31)
    c.dte, c.expiry = 35, TODAY + timedelta(days=35)
    ctx = result.underlyings["EMBR"]
    plan = exits.build(c, ctx, "sell", 0.04, TODAY)
    assert [k.label for k in plan.checkpoints] == ["in 7 days", "time stop", "in 21 days", "at expiry"]
    line = _exit_line(c, plan, ctx)
    stop = plan.checkpoints[1]
    assert f"time stop {stop.day:%b %d} {stop.pnl:+,.0f}" in line  # its result used to be dropped
    assert line.count("time stop") == 1

    # Across earnings, a seller is told what the position absorbs, not what a buyer needs.
    bolt = scanner.scan(ScanRequest(symbols=["BOLT"], lens="sell", kind="put", dte_min=20, dte_max=30))
    sold = bolt.ideas[0]
    text = _exit_line(sold, bolt.exit_plans[sold.symbol], bolt.underlyings["BOLT"])
    assert "buying back the next day costs less than was collected unless the stock moves more than -" in text
    held = scanner.scan(ScanRequest(symbols=["ACME"], kind="call", dte_max=30))
    bought = held.ideas[0]
    text = _exit_line(bought, held.exit_plans[bought.symbol], held.underlyings["ACME"])
    assert "the stock has to move +" in text and "to be back at the entry price" in text

    # No cushion at all: buying back already costs what was collected.
    plan = bolt.exit_plans[sold.symbol]
    stuck = type(plan.earnings)(plan.earnings.report_day, 1.0, 1.0, 0.0, 0.0)
    plan.earnings = stuck
    assert "already costs at least what was collected" in _exit_line(sold, plan, bolt.underlyings["BOLT"])

    # A preset's skip-earnings setting can be switched off from the command line.
    assert main(["scan", "ACME", "--demo", "--preset", "sell-premium", "--no-avoid-earnings", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ideas"]
    assert main(["scan", "ACME", "--demo", "--preset", "sell-premium", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ideas"] == []
