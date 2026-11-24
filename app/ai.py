"""Optional AI brief: Claude reads a scan and writes up the setups worth a closer look.

Needs the `anthropic` package (pip install -e ".[ai]") and credentials the SDK
can find, normally ANTHROPIC_API_KEY in .env.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from app.models import ContractAnalysis, ScanResult, UnderlyingContext

DEFAULT_MODEL = "claude-opus-5-5"
# Models that can hand a declined request to another model inside the same call.
FALLBACK_MODELS = {"claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_PICKS = 5

SYSTEM = """\
You write the brief for a personal options screener. The reader trades their own \
account and makes their own decisions. The screener's job is to surface setups \
worth a closer look and explain them plainly, so the reader can decide what to \
research further.

You receive one scan as JSON: contracts ranked by the screener's rules, each with \
its metrics, plus context for every underlying stock: price, realized volatility, \
the next earnings date, how the stock moved on past reports, and the size of move \
the options are pricing for the next one. The "lens" says whether the scan ranks \
contracts to buy or contracts to sell.

Choose up to five contracts from the scan that make the most coherent case under \
that lens. A setup where several independent numbers agree is stronger than one \
with a single standout metric. Returning fewer than five, or none, is the right \
answer when the scan is weak; say so in the summary.

For each pick, explain what the setup is, which numbers support it, what would \
make it a bad trade, and what the reader should check before acting.

The scan is all you know. You have no news, no fundamentals and no view of the \
reader's portfolio or risk tolerance, so quote only numbers that appear in the \
scan, describe tradeoffs rather than telling the reader to buy or sell, and leave \
position size to them. Where the scan flags a limitation (an estimated earnings \
date, synthetic demo data, a delayed or indicative quote feed), carry that caveat \
into what you write. The "contract" field must be the contract symbol exactly as \
it appears in the scan."""


class Pick(BaseModel):
    contract: str
    headline: str
    rationale: str
    risks: str
    check_first: str


class Brief(BaseModel):
    summary: str
    picks: list[Pick]


class BriefError(RuntimeError):
    pass


@dataclass
class BriefResult:
    summary: str
    picks: list[dict[str, str]] = field(default_factory=list)
    model: str = ""


def available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _r(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def _contract(c: ContractAnalysis) -> dict[str, Any]:
    return {
        "contract": c.symbol,
        "underlying": c.underlying,
        "type": c.kind,
        "strike": c.strike,
        "expiry": c.expiry.isoformat(),
        "days_to_expiry": c.dte,
        "score": c.score,
        "score_components": c.components,
        "mid": _r(c.mid, 2),
        "spread_pct_of_mid": _r(c.spread_pct),
        "open_interest": c.open_interest,
        "volume": c.volume,
        "implied_vol": _r(c.iv),
        "delta": _r(c.delta),
        "theta_pct_of_price_per_day": _r(c.theta_pct_per_day),
        "breakeven": _r(c.breakeven, 2),
        "breakeven_move_pct": _r(c.breakeven_move_pct),
        "expected_move_to_expiry_pct": _r(c.expected_move_pct),
        "breakeven_vs_expected_move": _r(c.breakeven_vs_expected, 2),
        "probability_of_profit": _r(c.pop, 3),
        "annualized_yield_if_sold": _r(c.annualized_yield),
        "expires_after_next_earnings": c.spans_earnings,
    }


def _underlying(u: UnderlyingContext) -> dict[str, Any]:
    out: dict[str, Any] = {
        "price": u.price,
        "change_today_pct": _r(u.change_pct),
        "realized_vol_20d": _r(u.realized_vol_20d),
        "realized_vol_60d": _r(u.realized_vol_60d),
        "at_the_money_implied_vol": _r(u.atm_iv),
        "why_scanned": u.reasons,
    }
    if u.next_earnings:
        out["next_earnings"] = {
            "date": u.next_earnings.day.isoformat(),
            "days_away": u.days_to_earnings,
            "timing": u.next_earnings.timing,
            "date_is_confirmed": u.next_earnings.confirmed,
        }
    if u.earnings_stats:
        s = u.earnings_stats
        out["past_earnings_moves"] = {
            "reports": s.count,
            "average_abs_move_pct": _r(s.mean_abs),
            "median_abs_move_pct": _r(s.median_abs),
            "largest_abs_move_pct": _r(s.max_abs),
            "up": s.up,
            "down": s.down,
            "recent": [{"date": m.day.isoformat(), "move_pct": _r(m.move_pct)} for m in u.reactions[:8]],
        }
    if u.implied_move:
        m = u.implied_move
        out["options_pricing_for_next_earnings"] = {
            "first_expiry_after_report": m.expiry.isoformat(),
            "straddle_pct_of_price": _r(m.straddle_pct),
            "implied_one_day_abs_move_pct": _r(m.event_move_pct),
            "ordinary_vol_taken_from": m.baseline_source,
        }
    if u.earnings_note:
        out["earnings_data_note"] = u.earnings_note
    return out


def scan_payload(result: ScanResult) -> dict[str, Any]:
    """The scan as compact JSON for the model."""
    symbols = {c.underlying for c in result.ideas}
    return {
        "lens": result.lens,
        "data": "synthetic demo data, not real markets" if result.mode == "demo" else "live market data",
        "options_quote_feed": result.options_feed,
        "generated_at": result.generated_at.isoformat(timespec="seconds"),
        "warnings": result.warnings,
        "ranked_contracts": [_contract(c) for c in result.ideas],
        "underlyings": {s: _underlying(u) for s, u in result.underlyings.items() if s in symbols},
    }


def write_brief(result: ScanResult, model: str | None = None, client: Any = None) -> BriefResult:
    if not result.ideas:
        return BriefResult(summary="The scan returned no contracts, so there is nothing to write up.")
    try:
        import anthropic
    except ImportError as err:
        raise BriefError('The AI brief needs the anthropic package: pip install -e ".[ai]"') from err

    model = model or DEFAULT_MODEL
    request: dict[str, Any] = dict(
        model=model,
        max_tokens=16000,
        output_config={"effort": "high"},
        system=SYSTEM,
        messages=[{"role": "user", "content": json.dumps(scan_payload(result))}],
        output_format=Brief,
    )
    try:
        client = client or anthropic.Anthropic()
        if model in FALLBACK_MODELS:
            # If a safety classifier declines the request, the API retries it on another model.
            response = client.beta.messages.parse(betas=[FALLBACK_BETA], fallbacks="default", **request)
        else:
            response = client.messages.parse(**request)
    except anthropic.AuthenticationError as err:
        raise BriefError("Anthropic rejected the credentials. Check ANTHROPIC_API_KEY in .env.") from err
    except anthropic.PermissionDeniedError as err:
        raise BriefError(f"This Anthropic key cannot use {model}.") from err
    except anthropic.RateLimitError as err:
        raise BriefError("Anthropic rate limit reached. Try again in a minute.") from err
    except anthropic.APIStatusError as err:
        raise BriefError(f"Anthropic returned an error ({err.status_code}): {err.message}") from err
    except anthropic.APIConnectionError as err:
        raise BriefError("Could not reach the Anthropic API.") from err
    except anthropic.AnthropicError as err:
        raise BriefError(f"No usable Anthropic credentials: {err}") from err

    if response.stop_reason == "refusal":
        raise BriefError("The model declined to write this brief.")
    brief = response.parsed_output
    if response.stop_reason == "max_tokens" or brief is None:
        raise BriefError("The model's answer was cut off before it finished.")

    # Keep only picks that name a contract from this scan.
    known = {c.symbol for c in result.ideas}
    picks = [p.model_dump() for p in brief.picks if p.contract in known][:MAX_PICKS]
    return BriefResult(summary=brief.summary, picks=picks, model=response.model)
