"""What a scan showed for a contract when it was paper traded, and which of those readings went on to make money.

Two halves:

* `snapshot` copies the scan's numbers for one contract into plain values, at
  the moment its order is written. The journal keeps that copy, so a later
  change to a weight or a threshold never rewrites what was shown at the time.
* `summary` sorts the closed trades by each reading (the rating, the score,
  the chance of profit, earnings, flagged news, how it was closed) and counts
  how many made money.

Four things keep the second half honest:

* It describes these trades and nothing else. Below `MIN_TRADES` a row says
  "too few" on every surface, as an event study does.
* Bought and sold options are never counted together. A buyer can lose all of
  the premium and no more, a seller can lose several times it, and a "favorable"
  rating means a different thing to each.
* Every row is set beside the same lens's row for every closed trade, because
  a reading only says something by how it differs from that.
* Nothing here feeds the score or the rating. The journal is beside them.

The fills are a paper account's: no order moved a market, and none waited in
a queue. A result here is likely kinder than the same order would have had
with real money.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable

from app import journal
from app.analysis import events
from app.models import ContractAnalysis, ScanResult, SignalRow, Trade

MIN_TRADES = 10  # below this many closed trades, a row is not compared
SCORE_BANDS = (70.0, 50.0)  # "70 or more", "50 to 70", "under 50"
POP_BANDS = (0.70, 0.40)  # the market-implied chance of profit
LENS = {"buy": "Bought to open", "sell": "Sold to open"}
ALL = "all"


def snapshot(result: ScanResult, c: ContractAnalysis) -> dict:
    """The scan's reading of one ranked contract, as plain values that can be written to a file."""
    ctx = result.underlyings[c.underlying]
    rated = c.risk_reward
    report = ctx.next_earnings
    return {
        "scan": result.id,
        "scanned": result.generated_at.isoformat(),
        "mode": result.mode,
        "feed": result.options_feed,
        "lens": result.lens,
        "rank": next((n for n, idea in enumerate(result.ideas, 1) if idea.symbol == c.symbol), None),
        "of": len(result.ideas),
        "score": c.score,
        "components": dict(c.components),
        "rating": rated.rating if rated else None,
        "shape": rated.shape if rated else None,
        "edge_pct": rated.edge_pct if rated else None,
        "win_prob": rated.win_prob if rated else None,  # if the stock is as volatile as its history
        "pop": c.pop,  # as the options price it
        "delta": c.delta,
        "dte": c.dte,
        "iv": c.iv,
        "history_vol": c.history_vol,
        "spread_pct": c.spread_pct,
        "bid": c.bid,
        "ask": c.ask,
        "stock": ctx.price,
        "spans_earnings": c.spans_earnings,
        "days_to_earnings": ctx.days_to_earnings,
        "earnings_estimated": bool(report and not report.confirmed),
        "earnings_unknown": ctx.earnings_unknown,
        # Recent events of the types a scan points out. A keyword match, as everywhere: not a judgment.
        "news": sorted(kind for kind in ctx.recent_events if kind in events.FLAGGED),
    }


# -- Which readings went on to make money --------------------------------------------------


def _band(value, bands: tuple[float, float], text: Callable[[float], str]) -> str | None:
    # A journal entry can be edited by hand, so a reading is whatever was typed. One that is not a number has
    # no band: NaN compares false with everything, and fell into the lowest.
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return None
    high, low = bands
    return f"{text(high)} or more" if value >= high else f"{text(low)} to {text(high)}" if value >= low else f"under {text(low)}"


def _rating(t: Trade) -> str | None:
    rating = t.signals.get("rating")
    return rating if isinstance(rating, str) and rating else "not rated"  # a list here would be a key no table can hold


def _score(t: Trade) -> str | None:
    return _band(t.signals.get("score"), SCORE_BANDS, lambda v: f"{v:g}")


def _pop(t: Trade) -> str | None:
    return _band(t.signals.get("pop"), POP_BANDS, lambda v: f"{v:.0%}")


def _earnings(t: Trade) -> str | None:
    if t.signals.get("spans_earnings"):
        return "open over a report"
    return "report date unknown" if t.signals.get("earnings_unknown") else "clear of a report"


def _news(t: Trade) -> str | None:
    return "a flagged event in the news" if t.signals.get("news") else "no flagged event"


# Each signal: what it is called, and the reading a trade had of it. The order is the order shown.
GROUPS: dict[str, tuple[str, Callable[[Trade], str | None]]] = {
    "rating": ("Rating if history repeats", _rating),
    "score": ("Score", _score),
    "pop": ("Chance of profit, as priced", _pop),
    "earnings": ("Earnings", _earnings),
    "news": ("News", _news),
    "ended": ("Closed", journal.ended),
}
LABELS = {ALL: "Every closed trade", **{name: label for name, (label, _) in GROUPS.items()}}


def counted(t: Trade) -> bool:
    """Whether a trade has a result and the scan's readings to set it beside."""
    lens = t.signals.get("lens")
    return t.state == "closed" and t.pnl is not None and t.pnl_pct is not None and isinstance(lens, str) and lens in LENS


def _row(group: str, bucket: str, trades: list[Trade]) -> SignalRow:
    return SignalRow(
        group=group,
        bucket=bucket,
        trades=len(trades),
        won=sum(1 for t in trades if t.pnl > 0),
        lost=sum(1 for t in trades if t.pnl < 0),
        mean_pct=round(sum(t.pnl_pct for t in trades) / len(trades), 6),
        total=round(sum(t.pnl for t in trades), 2),
    )


def summary(trades: Iterable[Trade]) -> dict[str, list[SignalRow]]:
    """Closed trades by each reading, for each lens that has any. The first row of a lens is every closed trade."""
    out: dict[str, list[SignalRow]] = {}
    closed = [t for t in trades if counted(t)]
    for lens in LENS:
        mine = [t for t in closed if t.signals["lens"] == lens]
        if not mine:
            continue
        rows = [_row(ALL, LABELS[ALL], mine)]
        for group, (_, reading) in GROUPS.items():
            buckets: dict[str, list[Trade]] = {}
            for t in mine:
                if (bucket := reading(t)) is not None:
                    buckets.setdefault(bucket, []).append(t)
            rows += [_row(group, bucket, found) for bucket, found in sorted(buckets.items(), key=lambda kv: -len(kv[1]))]
        out[lens] = rows
    return out
