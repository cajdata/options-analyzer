"""End-to-end scans on the synthetic provider, plus the API and the AI brief plumbing."""

import json
import math
import re
from datetime import date
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import ai
from app.cache import NullCache
from app.config import Settings
from app.providers.demo import SPECS, DemoMarket
from app.providers.earnings import EarningsService
from app.scanner import ConfigError, Scanner, ScanRequest, build_scanner
from app.server import create_app

TODAY = date(2026, 11, 24)


def demo_scanner(**settings) -> Scanner:
    market = DemoMarket(TODAY)
    earnings = EarningsService(NullCache(), history_sources=[market], calendar_source=market, today=TODAY)
    return Scanner(market, earnings, Settings(**settings), universe=list(SPECS), today=TODAY)


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
        return math.sqrt(jump**2 + ordinary**2 / 365) * math.sqrt(2 / math.pi)

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
    assert "Options Analyzer" in client.get("/").text


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
        assert rr.max_loss == pytest.approx(c.mid * 100)  # a buyer can lose the premium and no more
        assert rr.history_vol == c.history_vol
        cheap = c.iv < c.history_vol
        assert (rr.edge_pct > 0) == cheap  # priced below history means positive expected value for a buyer


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
