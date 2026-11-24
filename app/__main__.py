"""Command line: `python -m app scan ...` or `python -m app serve`."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from app import ai
from app.config import load_settings
from app.models import ContractAnalysis, ScanResult, UnderlyingContext
from app.scanner import ConfigError, ScanRequest, build_scanner


_RATING = 22  # width of the rating column, e.g. "unfavorable, long shot"


def _pct(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def _earnings(ctx: UnderlyingContext, c: ContractAnalysis) -> str:
    if ctx.days_to_earnings is None:
        return "-"
    text = f"in {ctx.days_to_earnings}d" + ("" if ctx.next_earnings.confirmed else " (est.)")
    return text + (", spans" if c.spans_earnings else "")


def print_scan(result: ScanResult) -> None:
    feed = "SYNTHETIC DEMO DATA" if result.mode == "demo" else f"Alpaca, {result.options_feed} option quotes"
    budget = f"   Budget: ${result.budget:,.0f} per contract" if result.budget else ""
    print(f"\nLens: {result.lens} premium   Data: {feed}{budget}\n")
    if not result.ideas:
        print("No contracts passed the filters.")
    else:
        header = (
            f"{'#':>2}  {'Score':>5}  {'Contract':<24} {'DTE':>3}  {'Mid':>7}  {'Outlay':>8}  {'Spread':>6}  "
            f"{'Delta':>6}  {'IV':>6}  {'POP':>5}  {'B/E move':>8}  {'vs exp.':>7}  {'R:R':>5}  {'Rating':<{_RATING}}  Earnings"
        )
        print(header)
        print("-" * len(header))
        for i, c in enumerate(result.ideas, 1):
            ctx = result.underlyings[c.underlying]
            name = f"{c.underlying} {c.expiry:%b %d} {c.strike:g} {c.kind}"
            ratio = "-" if c.breakeven_vs_expected is None else f"{c.breakeven_vs_expected:.2f}x"
            reward = f"{'-':>5}  {'-':<{_RATING}}"
            if c.risk_reward:
                label = c.risk_reward.rating + (f", {c.risk_reward.shape}" if c.risk_reward.shape else "")
                reward = f"{c.risk_reward.ratio:>5.2f}  {label:<{_RATING}}"
            print(
                f"{i:>2}  {c.score:>5.1f}  {name:<24} {c.dte:>3}  {c.mid:>7.2f}  {f'${c.outlay:,.0f}':>8}  "
                f"{_pct(c.spread_pct):>6}  "
                f"{c.delta:>+6.2f}  {_pct(c.iv):>6}  {_pct(c.pop):>5}  {_pct(c.breakeven_move_pct, True):>8}  "
                f"{ratio:>7}  {reward}  {_earnings(ctx, c)}"
            )

    shown = {c.underlying for c in result.ideas}
    lines = []
    for symbol in sorted(shown):
        ctx = result.underlyings[symbol]
        stats, implied = ctx.earnings_stats, ctx.implied_move
        if not stats:
            continue
        line = f"  {symbol}: past {stats.count} reports averaged a {_pct(stats.mean_abs)} move (largest {_pct(stats.max_abs)})"
        if implied and implied.event_move_pct is not None:
            line += f"; options price about {_pct(implied.event_move_pct)} for the next one"
        lines.append(line)
    if lines:
        print("\nEarnings context")
        print("\n".join(lines))
    if result.warnings:
        print("\nNotes")
        for warning in result.warnings:
            print(f"  - {warning}")
    print("\nOutlay is the cash to open one contract: the ask x 100 to buy, or the collateral to sell.")
    print("R:R is average win over average loss, and the rating compares the price with the stock's")
    print("own history. Ranked by fixed, visible rules. Not a forecast and not advice.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app", description="Read-only options scanner.")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="scan symbols, or discover candidates when none are given")
    scan.add_argument("symbols", nargs="*", help="tickers to scan; leave empty to discover")
    scan.add_argument("--lens", choices=["buy", "sell"], default="buy", help="rank contracts to buy or to sell")
    scan.add_argument("--type", dest="kind", choices=["call", "put", "both"], default="both")
    scan.add_argument("--dte-min", type=int, default=7)
    scan.add_argument("--dte-max", type=int, default=60)
    scan.add_argument("--top", type=int, default=20)
    scan.add_argument("--budget", type=float, help="most cash to put into one contract, in dollars")
    scan.add_argument("--brief", action="store_true", help="add an AI write-up (uses the Claude API)")
    scan.add_argument("--json", action="store_true", help="print the full result as JSON")
    scan.add_argument("--demo", action="store_true", help="use synthetic data; no keys needed")

    serve = sub.add_parser("serve", help="start the local dashboard")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--demo", action="store_true", help="use synthetic data; no keys needed")

    args = parser.parse_args(argv)
    settings = load_settings()

    if args.command == "serve":
        import uvicorn

        from app.server import create_app

        print(f"Dashboard: http://127.0.0.1:{args.port}")
        uvicorn.run(create_app(settings, demo=args.demo), host="127.0.0.1", port=args.port, log_level="warning")
        return 0

    try:
        scanner = build_scanner(settings, demo=args.demo)
    except ConfigError as err:
        print(err, file=sys.stderr)
        return 2
    request = ScanRequest(
        symbols=[s.upper() for s in args.symbols],
        lens=args.lens,
        kind=args.kind,
        dte_min=args.dte_min,
        dte_max=args.dte_max,
        top_n=args.top,
        budget=args.budget,
    )
    result = scanner.scan(request)
    if args.json:
        print(json.dumps(asdict(result), default=str, indent=2))
    else:
        print_scan(result)

    if args.brief:
        try:
            brief = ai.write_brief(result, model=settings.ai_model)
        except ai.BriefError as err:
            print(f"\nAI brief unavailable: {err}", file=sys.stderr)
            return 1
        print(f"\nAI brief ({brief.model or 'no model call'})\n{brief.summary}")
        for pick in brief.picks:
            print(f"\n* {pick['contract']}: {pick['headline']}")
            print(f"  Why: {pick['rationale']}")
            print(f"  Risks: {pick['risks']}")
            print(f"  Check first: {pick['check_first']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
