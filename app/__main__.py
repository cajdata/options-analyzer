"""Command line: `python -m app scan ...` or `python -m app serve`."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from app import ai
from app.config import ConfigError, load_settings
from app.analysis import exits
from app.models import ContractAnalysis, ExitPlan, ScanResult, UnderlyingContext
from app.scanner import PRESETS, ScanRequest, build_scanner

_RATING = 22  # width of the rating column, e.g. "unfavorable, long shot"


def _pct(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def _earnings(ctx: UnderlyingContext, c: ContractAnalysis) -> str:
    if ctx.days_to_earnings is None:
        return "unknown" if ctx.earnings_unknown else "-"
    text = "today" if ctx.days_to_earnings == 0 else f"in {ctx.days_to_earnings}d"
    text += "" if ctx.next_earnings.confirmed else " (est.)"
    return text + (", spans" if c.spans_earnings else "")


def print_scan(result: ScanResult) -> None:
    feed = "SYNTHETIC DEMO DATA" if result.mode == "demo" else f"Alpaca, {result.options_feed} option quotes"
    budget = f"   Budget: ${result.budget:,.0f} per contract" if result.budget is not None else ""
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
    print("R:R is average win over average loss, priced at the ask to buy or the bid to sell, and the")
    print("rating compares that price with the stock's own history. Ranked by fixed, visible rules.")
    print("Not a forecast and not advice.")


def _exit_line(c: ContractAnalysis, plan: ExitPlan, ctx: UnderlyingContext) -> str:
    """One contract's exit plan in a line or two."""
    name = f"{c.underlying} {c.expiry:%b %d} {c.strike:g} {c.kind}"
    parts = []
    if plan.lens == "sell":
        when = f"decay alone gets there about {plan.target_day:%b %d}" if plan.target_day else "decay alone does not get there before expiry"
        parts.append(f"buy back at ${plan.target_price:.2f} or less to keep {exits.PROFIT_TARGET:.0%} ({when})")
    elif plan.half_life_day:
        parts.append(f"half the time value is gone by {plan.half_life_day:%b %d}")
    if plan.time_stop_day:
        parts.append(f"time stop {plan.time_stop_day:%b %d}")
    for k in plan.checkpoints:
        if k.label.startswith("in ") or k.label == "at expiry":
            parts.append(f"{k.label} {k.pnl:+,.0f} ({k.pnl_pct:+.0%})")
    text = f"  {name}: " + "; ".join(parts)
    report = plan.earnings
    if report:
        move = report.move_to_breakeven_pct
        back = "no plausible move gets back to the entry price" if move is None else f"{move:+.1%} in the stock gets back to the entry price"
        text += f"\n      earnings {report.report_day:%b %d}: {report.crush_pct:+.0%} with the stock unchanged; {back}"
        if ctx.earnings_stats:
            text += f" (past reports averaged {ctx.earnings_stats.mean_abs:.1%})"
    for note in plan.notes:
        text += f"\n      note: {note}"
    return text


def print_exits(result: ScanResult) -> None:
    if not result.exit_plans:
        return
    print("\nExit plan, if the stock stands still (dollars per contract, closing at the bid or ask)")
    for c in result.ideas:
        plan = result.exit_plans.get(c.symbol)
        if plan:
            print(_exit_line(c, plan, result.underlyings[c.underlying]))
    print("  These dates are where time alone takes the price. They are not probabilities, and the")
    print(f"  {exits.PROFIT_TARGET:.0%} target and {exits.TIME_STOP_DTE}-day time stop are rules of thumb, not signals.")


def _print_brief(brief: ai.BriefResult) -> None:
    print(f"\nAI brief ({brief.model or 'no model call'})\n{brief.summary}")
    for pick in brief.picks:
        print(f"\n* {pick['contract']}: {pick['headline']}")
        print(f"  Why: {pick['rationale']}")
        print(f"  Risks: {pick['risks']}")
        print(f"  Check first: {pick['check_first']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app", description="Read-only options scanner.")
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="scan symbols, or discover candidates when none are given")
    scan.add_argument("symbols", nargs="*", help="tickers to scan; leave empty to discover")
    scan.add_argument("--preset", choices=sorted(PRESETS), help="start from a named setup; other options override it")
    scan.add_argument("--lens", choices=["buy", "sell"], help="rank contracts to buy (default) or to sell")
    scan.add_argument("--type", dest="kind", choices=["call", "put", "both"], default="both")
    scan.add_argument("--dte-min", type=int, help="fewest days to expiry (default 7)")
    scan.add_argument("--dte-max", type=int, help="most days to expiry (default 60)")
    scan.add_argument("--delta-min", type=float, help="smallest delta, ignoring its sign")
    scan.add_argument("--delta-max", type=float, help="largest delta, ignoring its sign")
    scan.add_argument("--avoid-earnings", action="store_true", default=None, help="skip contracts open over the next report")
    scan.add_argument("--exits", action="store_true", help="print an exit plan for each contract")
    scan.add_argument("--top", type=int, default=20)
    scan.add_argument("--budget", type=float, help="most cash to put into one contract, in dollars")
    scan.add_argument("--brief", action="store_true", help="add an AI write-up (uses the Claude API)")
    scan.add_argument("--json", action="store_true", help="print the full result as one JSON document")
    scan.add_argument("--demo", action="store_true", help="use synthetic data; no keys needed")

    serve = sub.add_parser("serve", help="start the local dashboard")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--demo", action="store_true", help="use synthetic data; no keys needed")

    args = parser.parse_args(argv)
    if args.command == "scan":
        # A preset fills in whatever was not given on the command line.
        chosen = {"lens": "buy", "dte_min": 7, "dte_max": 60, "delta_min": None, "delta_max": None, "avoid_earnings": False}
        chosen.update(PRESETS[args.preset]["settings"] if args.preset else {})
        for name in chosen:
            if getattr(args, name) is None:
                setattr(args, name, chosen[name])
        if args.top < 1:
            parser.error("--top must be at least 1")
        if args.dte_min < 0 or args.dte_max < args.dte_min:
            parser.error("--dte-min must be 0 or more, and no greater than --dte-max")
        if args.budget is not None and args.budget <= 0:
            parser.error("--budget must be more than 0")
        for value in (args.delta_min, args.delta_max):
            if value is not None and not 0 <= value <= 1:
                parser.error("--delta-min and --delta-max must be between 0 and 1")
        if args.delta_min is not None and args.delta_max is not None and args.delta_max < args.delta_min:
            parser.error("--delta-min must be no greater than --delta-max")

    try:
        settings = load_settings()
    except ConfigError as err:
        print(err, file=sys.stderr)
        return 2

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
        symbols=args.symbols,
        lens=args.lens,
        kind=args.kind,
        dte_min=args.dte_min,
        dte_max=args.dte_max,
        top_n=args.top,
        budget=args.budget,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        avoid_earnings=args.avoid_earnings,
    )
    result = scanner.scan(request)

    brief, brief_error = None, None
    if args.brief:
        try:
            brief = ai.write_brief(
                result, model=settings.ai_model, api_key=settings.anthropic_key, effort=settings.ai_effort
            )
        except ai.BriefError as err:
            brief_error = str(err)
            print(f"AI brief unavailable: {err}", file=sys.stderr)

    if args.json:
        # One document, so the output can be piped straight into a JSON parser.
        document = asdict(result)
        if args.brief:
            document["brief"] = asdict(brief) if brief else None
        print(json.dumps(document, default=str, indent=2))
    else:
        print_scan(result)
        if args.exits:
            print_exits(result)
        if brief:
            _print_brief(brief)
    return 1 if brief_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
