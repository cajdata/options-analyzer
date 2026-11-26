"""Command line: `python -m app scan ...`, `python -m app positions` or `python -m app serve`."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import asdict
from datetime import datetime

from app import ai, positions
from app.config import ConfigError, load_settings
from app.analysis import events, exits, holdings
from app.models import (
    ContractAnalysis,
    EventStudy,
    ExitPlan,
    Holding,
    PooledStudy,
    Position,
    PositionsReview,
    ScanResult,
    UnderlyingContext,
)
from app.scanner import ET, PRESETS, ScanRequest, build_scanner

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


def _study_line(study: EventStudy) -> str:
    """What followed a stock's past events of one type, in a line."""
    what = f'past {study.count} "{study.event}" event{"" if study.count == 1 else "s"}'
    if study.count < events.MIN_EVENTS:
        return f"{what}: too few to compare with its usual day"
    text = f"{what}: {study.mean_abs:.1%} average move that day"
    if study.ordinary_abs is not None:
        text += f" ({study.ordinary_abs:.1%} on any day)"
    text += f", {study.up} up {study.down} down"
    if study.after_count >= events.MIN_EVENTS:
        text += (
            f"; {_pct(study.mean_abs_after)} over the next {events.FOLLOW_SESSIONS} sessions "
            f"({_pct(study.ordinary_abs_after)} over any {events.FOLLOW_SESSIONS})"
        )
    return text


def _stocks(count: int) -> str:
    return "stock" if count == 1 else "stocks"


def _pooled_line(pooled: PooledStudy, lead: bool = True) -> str:
    """One type of event across the stocks covered about as much as this one, in a line."""
    what = f'{pooled.events} "{pooled.event}" days'
    if lead:
        what = f"across {pooled.group_stocks} {events.GROUPS[pooled.group]} {_stocks(pooled.group_stocks)}, {what}"
    if pooled.events < events.MIN_POOLED:
        return f"{what}: too few to compare"
    any_day = f" ({pooled.big_sessions / pooled.sessions:.0%} of all days)" if pooled.sessions else ""
    return (
        f"{what}: {pooled.big / pooled.events:.0%} moved over {events.BIG_MOVE:g}x the stock's usual day"
        f"{any_day}, {pooled.up} up {pooled.down} down"
    )


def print_pool(found: dict) -> None:
    """What `Scanner.widen_pool` did, and what the pool now holds for the types that are pointed out."""
    added = found["added"]
    print(f"\nAsked for {found['asked']} {_stocks(found['asked'])}. Added " + ", ".join(f"{n} {events.GROUPS[g]}" for g, n in added.items()) + ".")
    if found["passed_over"]:
        print("Passed over: " + ", ".join(f"{n} with {why}" for why, n in found["passed_over"].items()) + ".")
    for group, label in events.GROUPS.items():
        totals = found["totals"][group]
        print(f"\n{label.capitalize()}: {totals['stocks']} {_stocks(totals['stocks'])} on record")
        pooled = events.pooled(found["totals"], group, [kind for kind in events.EVENT_TYPES if kind in events.FLAGGED])
        for study in pooled.values():
            print(f"  {_pooled_line(study, lead=False)}")
    print("\nA day is counted against the stock's own usual day, earnings days left out. Types are keyword")
    print("guesses, and this describes the past: it is not a forecast.")


NEWS_LINES = 3  # headlines printed for one stock


def _news_lines(symbol: str, ctx: UnderlyingContext) -> list[str]:
    """A stock's latest classified news, and what followed its past events of those types.

    There is room for a few lines, and a headline behind a caution is always
    one of them (`events.shortlist`).
    """
    telling = events.shortlist((item for item in ctx.news if item.event not in events.NOT_STUDIED), NEWS_LINES)
    lines = []
    for n, item in enumerate(telling):
        when = item.when.astimezone(events.ET)
        lines.append(f"  {symbol if n == 0 else '':<6}{when:%b %d}  {item.event:<13}  {item.title[:88]}")
    for kind in dict.fromkeys(item.event for item in telling):
        if kind in ctx.event_studies:
            lines.append(f"        {_study_line(ctx.event_studies[kind])}")
        if kind in ctx.pooled_studies:
            lines.append(f"        {_pooled_line(ctx.pooled_studies[kind])}")
    for kind in ctx.recent_events:
        if kind in events.CAUTIONS:
            caution = f"{events.EVENT_TYPES[kind]}: {events.CAUTIONS[kind]}"
            lines.append(textwrap.fill(caution, 110, initial_indent=" " * 8, subsequent_indent=" " * 8))
    for note in (ctx.news_note, ctx.naming_note):
        if note:
            # With no headline line above to carry the ticker, the note carries it.
            lead = " " * 8 if lines else f"  {symbol:<6}"
            lines.append(textwrap.fill(note, 110, initial_indent=lead, subsequent_indent=" " * 8))
    return lines


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
    news = [line for symbol in sorted(shown) for line in _news_lines(symbol, result.underlyings[symbol])]
    if news:
        print(f"\nNews and filings, last {events.RECENT_DAYS} days")
        print("\n".join(news))
        print("  Types come from keyword rules on the headline and from the item codes on an 8-K, so some are")
        print("  wrong. What followed past events describes the stock's history; it is not a forecast.")
    if result.warnings:
        print("\nNotes")
        for warning in result.warnings:
            print(f"  - {warning}")
    print("\nOutlay is the cash to open one contract: the ask x 100 to buy, or the collateral to sell.")
    print("R:R is average win over average loss, priced at the ask to buy or the bid to sell, and the")
    print("rating compares that price with the stock's own history. Ranked by fixed, visible rules.")
    print("Not a forecast and not advice.")


def _after_report(plan: ExitPlan) -> str:
    """What the session after the report asks of the stock, in the holder's or the seller's terms."""
    move = plan.earnings.move_to_breakeven_pct
    if move is None:
        return "no plausible move in the stock gets back to the entry price"
    none_needed = abs(move) < 0.0005
    if plan.lens == "sell":
        if none_needed:
            return "buying back the next day already costs at least what was collected"
        return f"buying back the next day costs less than was collected unless the stock moves more than {move:+.1%}"
    if none_needed:
        return "selling the next day already returns the entry price"
    return f"the stock has to move {move:+.1%} to be back at the entry price the next day"


def _exit_line(c: ContractAnalysis, plan: ExitPlan, ctx: UnderlyingContext) -> str:
    """One contract's exit plan in a line or two."""
    name = f"{c.underlying} {c.expiry:%b %d} {c.strike:g} {c.kind}"
    parts = []
    if plan.lens == "sell":
        when = f"decay alone gets there about {plan.target_day:%b %d}" if plan.target_day else "decay alone does not get there before expiry"
        parts.append(f"buy back at {holdings.at_most(plan.target_price)} or less to keep {exits.PROFIT_TARGET:.0%} ({when})")
    elif plan.half_life_day:
        parts.append(f"half the time value is gone by {plan.half_life_day:%b %d}")
    stop_shown = False
    for k in plan.checkpoints:
        if k.label in ("before earnings", "after earnings"):
            continue  # the earnings line below covers these
        stop_shown = stop_shown or k.label == "time stop"
        when = f"time stop {k.day:%b %d}" if k.label == "time stop" else k.label
        parts.append(f"{when} {k.pnl:+,.0f} ({k.pnl_pct:+.0%})")
    if plan.time_stop_day and not stop_shown:
        parts.append(f"time stop {plan.time_stop_day:%b %d}")
    text = f"  {name}: " + "; ".join(parts)
    report = plan.earnings
    if report:
        text += (
            f"\n      earnings {report.report_day:%b %d}: {report.crush_pct:+.0%} "
            f"with the stock unchanged; {_after_report(plan)}"
        )
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


def _held(p: Position) -> str:
    """A position in a few words: FLUX Jan 08 87.5 put, sold."""
    return f"{p.underlying} {p.expiry:%b %d} {p.strike:g} {p.kind}, {'bought' if p.side == 'long' else 'sold'}"


def _figure(price: float) -> str:
    """A price as it is, with no dollar sign: to the cent, or finer when it is not a whole cent (1.255)."""
    return holdings.dollars(price)[1:]


def _result(h: Holding) -> str:
    """A result in whole dollars and percent. Signs are settled after rounding, so nothing prints as "+$0" or "-0%"."""
    if h.pnl is None:
        return "-"
    whole, share = round(h.pnl), round(h.pnl_pct * 100)
    money = f"{'-' if whole < 0 else '+' if whole > 0 else ''}${abs(whole):,}"
    return f"{money} ({'+' if share > 0 else ''}{share}%)"


def print_positions(review: PositionsReview) -> None:
    """Each position with what closing now would give, and what the exit rules say about it."""
    feed = {"demo": "SYNTHETIC DEMO DATA", "live": f"Alpaca, {review.options_feed} option quotes"}.get(review.mode, "no market data")
    print(f"\nPositions   Data: {feed}\n")
    if not review.holdings:
        print("No positions on file. Add one with:")
        print("  python -m app positions add SYMBOL YYYY-MM-DD STRIKE call|put --sold --price 1.25")
    else:
        header = f"{'ID':<9} {'Position':<32} {'Qty':>3}  {'Entry':>7}  {'Close at':>8}  {'Result':>15}  {'DTE':>4}  To look at"
        print(header)
        print("-" * len(header))
        for h in review.holdings:
            p = h.position
            # An expired position is not set beside its stock's news, whether or not something else on the stock is held.
            ctx = review.underlyings.get(p.underlying) if h.dte >= 0 else None
            look = [c.label for c in h.checks if c.state != "clear"]
            look += [f"news: {kind}" for kind in (ctx.recent_events if ctx else {}) if kind in events.FLAGGED]
            close = "-" if h.close_price is None else _figure(h.close_price)
            print(
                f"{p.id:<9} {_held(p):<32} {p.quantity:>3}  {_figure(p.entry_price):>7}  {close:>8}  {_result(h):>15}  "
                f"{h.dte:>4}  {', '.join(look) or '-'}"
            )

        said = False
        for h in review.holdings:
            lines = [f"{holdings.RULES[c.rule]}: {c.detail}" for c in h.checks if c.state != "clear"]
            lines += [h.note] if h.note else []
            if not lines:
                continue
            if not said:
                print("\nWhat the rules say")
                said = True
            print(f"  {_held(h.position)}")
            for line in lines:
                print(textwrap.fill(line, 110, initial_indent=" " * 6, subsequent_indent=" " * 6))

    # Only the stocks with a recent event of a type that tends to move a stock in one jump. That
    # is what a holder has to see, and every held stock's news would bury it.
    telling = [s for s, ctx in sorted(review.underlyings.items()) if any(kind in events.FLAGGED for kind in ctx.recent_events)]
    news = [line for symbol in telling for line in _news_lines(symbol, review.underlyings[symbol])]
    if news:
        print(f"\nNews on what is held, last {events.RECENT_DAYS} days, where an event of a flagged type is among it")
        print("\n".join(news))
    if review.warnings:
        print("\nNotes")
        for warning in review.warnings:
            print(f"  - {warning}")
    print("\nResult is what closing now would give across every contract: selling at the bid, or buying back")
    print("at the ask. A rule that is reached is a fact about the position, not advice, and the")
    delayed = " Quotes on this feed are delayed." if review.options_feed == "indicative" else ""
    print(f"{exits.PROFIT_TARGET:.0%} target and {exits.TIME_STOP_DTE}-day time stop are rules of thumb.{delayed}")


def _change_positions(args, settings) -> int:
    """Add or remove an entry. This touches the positions file only, so it needs no keys."""
    if args.demo:
        print("The demo's positions are kept in memory, so there is nothing to add to or remove from.", file=sys.stderr)
        return 2
    store = positions.PositionStore(settings.positions_file)
    try:
        if args.action == "remove":
            if not store.remove(args.id):
                print(f"There is no position with the id {args.id}.", file=sys.stderr)
                return 2
            print(f"Removed {args.id} from {store.path}.")
            return 0
        opened = args.opened or datetime.now(ET).date()
        made = store.add(positions.make(args.underlying, args.kind, args.strike, args.expiry, args.side, args.qty, args.price, opened))
    except (ValueError, positions.StoreError) as err:
        print(err, file=sys.stderr)
        return 2
    print(f"Added {made.id}: {made.quantity} x {_held(made)} at {_figure(made.entry_price)} ({made.symbol}), in {store.path}.")
    return 0


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
    scan.add_argument(
        "--avoid-earnings",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="skip contracts open over the next report (--no-avoid-earnings turns a preset's setting off)",
    )
    scan.add_argument("--exits", action="store_true", help="print an exit plan for each contract")
    scan.add_argument("--top", type=int, default=20)
    scan.add_argument("--budget", type=float, help="most cash to put into one contract, in dollars")
    scan.add_argument("--brief", action="store_true", help="add an AI write-up (uses the Claude API)")
    scan.add_argument("--json", action="store_true", help="print the full result as one JSON document")
    scan.add_argument("--demo", action="store_true", help="use synthetic data; no keys needed")

    serve = sub.add_parser("serve", help="start the local dashboard")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--demo", action="store_true", help="use synthetic data; no keys needed")

    pool = sub.add_parser("pool", help="add stocks to the pool that news events are compared across")
    pool.add_argument("symbols", nargs="*", help="tickers to add; leave empty for the universe and the day's most active stocks and movers")
    pool.add_argument("--demo", action="store_true", help="use synthetic data; no keys needed")

    held = sub.add_parser("positions", help="check the positions you hold against the exit rules")
    held.add_argument("--json", action="store_true", help="print the full result as one JSON document")
    held.add_argument("--demo", action="store_true", help="use synthetic data and made-up positions; no keys needed")
    actions = held.add_subparsers(dest="action")
    add = actions.add_parser("add", help="record a position you hold")
    add.add_argument("underlying", help="the stock's ticker")
    add.add_argument("expiry", help="the contract's expiry, as YYYY-MM-DD")
    add.add_argument("strike", type=float)
    add.add_argument("kind", choices=["call", "put"])
    side = add.add_mutually_exclusive_group(required=True)
    side.add_argument("--bought", dest="side", action="store_const", const="long", help="you bought it (long)")
    side.add_argument("--sold", dest="side", action="store_const", const="short", help="you sold it (short)")
    add.add_argument("--price", type=float, required=True, help="what you paid or collected, per share")
    add.add_argument("--qty", type=int, default=1, help="contracts (default 1)")
    add.add_argument("--opened", help="the day it was opened, as YYYY-MM-DD (default today)")
    remove = actions.add_parser("remove", help="take a position off the file")
    remove.add_argument("id", help="its id, as the list shows it")

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

    if args.command == "positions" and args.action:
        return _change_positions(args, settings)

    try:
        scanner = build_scanner(settings, demo=args.demo)
    except ConfigError as err:
        print(err, file=sys.stderr)
        return 2
    if args.command == "positions":
        store = positions.open_store(settings, scanner.market)
        try:
            found, unread = store.load()
        except positions.StoreError as err:
            print(err, file=sys.stderr)
            return 2
        review = scanner.review(found)
        if unread:
            review.warnings.insert(0, positions.unread_note(unread, store.path))
        if args.json:
            print(json.dumps(asdict(review), default=str, indent=2))
        else:
            print_positions(review)
        return 0
    if args.command == "pool":
        def progress(n: int, of: int, symbol: str) -> None:
            if n == 1 or n % 25 == 0 or n == of:
                print(f"  reading news: {n} of {of} ({symbol})", file=sys.stderr)

        try:
            found = scanner.widen_pool(args.symbols, progress=progress)
        except ConfigError as err:
            print(err, file=sys.stderr)
            return 2
        print_pool(found)
        return 0
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
