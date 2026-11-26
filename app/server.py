"""Local web dashboard and JSON API."""

from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict
from dataclasses import asdict
from datetime import date, datetime, timezone
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from app import ai, positions
from app.analysis import events, exits, holdings, scoring
from app.config import ROOT, ConfigError, Settings, load_settings
from app.models import Position, PositionsReview, ScanResult
from app.scanner import ET, PRESETS, Scanner, ScanRequest, build_scanner

log = logging.getLogger(__name__)

KEPT_SCANS = 10
# One to six letters or digits, optionally a class suffix: AAPL, F, BRK.B.
_SYMBOL = re.compile(r"[A-Z0-9]{1,6}(\.[A-Z0-9]{1,3})?")


class ScanBody(BaseModel):
    symbols: list[str] = Field(default_factory=list, max_length=40)
    lens: Literal["buy", "sell"] = "buy"
    kind: Literal["call", "put", "both"] = "both"
    dte_min: int = Field(7, ge=0, le=730)
    dte_max: int = Field(60, ge=1, le=730)
    top_n: int = Field(20, ge=1, le=100)
    budget: float | None = Field(None, gt=0, le=10_000_000)
    delta_min: float | None = Field(None, ge=0, le=1)
    delta_max: float | None = Field(None, ge=0, le=1)
    avoid_earnings: bool = False

    @field_validator("symbols")
    @classmethod
    def clean(cls, symbols: list[str]) -> list[str]:
        # Share classes are written both ways (BRK-B, BRK.B); the data provider wants the dot.
        cleaned = [s.strip().upper().replace("-", ".") for s in symbols if s.strip()]
        for symbol in cleaned:
            if not _SYMBOL.fullmatch(symbol):
                raise ValueError(f"not a ticker symbol: {symbol!r}")
        return list(dict.fromkeys(cleaned))


class PositionBody(BaseModel):
    """A position as the person enters it. `positions.make` has the last word on what is one."""

    underlying: str = Field(max_length=12)
    kind: Literal["call", "put"]
    strike: float = Field(gt=0, lt=100_000)
    expiry: date
    side: Literal["long", "short"]
    quantity: int = Field(1, ge=1, le=positions.MAX_QUANTITY)
    entry_price: float = Field(gt=0, le=1_000_000)
    opened: date | None = None  # today, when left out


def create_app(settings: Settings | None = None, demo: bool = False) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="Strike Check", docs_url=None, redoc_url=None)
    scans: OrderedDict[str, ScanResult] = OrderedDict()
    scans_lock = threading.Lock()  # request handlers run on several threads
    scanner: Scanner | None = None
    setup_error: str | None = None
    try:
        scanner = build_scanner(settings, demo=demo)
    except ConfigError as err:
        setup_error = str(err)
    # Positions can be entered and read without market data, so the store does not wait on the keys.
    store = positions.open_store(settings, scanner.market if scanner else None)

    def today() -> date:
        return scanner.today if scanner else datetime.now(ET).date()

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        # Always revalidate, so the page matches the code after an update.
        return FileResponse(ROOT / "public" / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/status")
    def status() -> dict:
        return {
            "mode": scanner.market.mode if scanner else "unconfigured",
            "options_feed": scanner.market.options_feed if scanner else None,
            "earnings_sources": scanner.earnings.sources if scanner else [],
            "ai_available": ai.available(settings.anthropic_key),
            "setup_error": setup_error,
            "watchlist": list(settings.watchlist),
            "lenses": {name: cfg["weights"] for name, cfg in scoring.LENSES.items()},
            "presets": PRESETS,
            "exit_rules": {"profit_target": exits.PROFIT_TARGET, "time_stop_dte": exits.TIME_STOP_DTE},
            "positions": {
                "rules": holdings.RULES,
                "earnings_soon_days": holdings.EARNINGS_SOON,
                # Where they are kept. None: in memory, as the demo's are.
                "file": str(store.path) if store.path else None,
            },
            "news_sources": scanner.news.sources if scanner and scanner.news else [],
            "events": {
                "labels": events.EVENT_TYPES,
                "flagged": sorted(events.FLAGGED),
                "cautions": events.CAUTIONS,
                "elsewhere": events.ELSEWHERE,
                "pool": {
                    "groups": events.GROUPS,
                    "big_move": events.BIG_MOVE,
                    "min_events": events.MIN_POOLED,
                    "coverage_days": events.COVERAGE_DAYS,
                    "heavy_headlines": events.HEAVY_HEADLINES,
                },
                "recent_days": events.RECENT_DAYS,
                "follow_sessions": events.FOLLOW_SESSIONS,
                "min_events": events.MIN_EVENTS,
            },
        }

    @app.post("/api/scan")
    def scan(body: ScanBody) -> ScanResult:
        if scanner is None:
            raise HTTPException(503, setup_error)
        if body.dte_max < body.dte_min:
            raise HTTPException(422, "dte_max must not be below dte_min")
        if body.delta_min is not None and body.delta_max is not None and body.delta_max < body.delta_min:
            raise HTTPException(422, "delta_max must not be below delta_min")
        request = ScanRequest(
            symbols=body.symbols,
            lens=body.lens,
            kind=body.kind,
            dte_min=body.dte_min,
            dte_max=body.dte_max,
            top_n=body.top_n,
            budget=body.budget,
            delta_min=body.delta_min,
            delta_max=body.delta_max,
            avoid_earnings=body.avoid_earnings,
        )
        try:
            result = scanner.scan(request)
        except Exception as err:  # data-provider failure: report it rather than a bare 500
            log.exception("scan failed")
            raise HTTPException(502, f"Scan failed: {err}") from err
        with scans_lock:
            scans[result.id] = result
            while len(scans) > KEPT_SCANS:
                scans.popitem(last=False)
        return result

    @app.post("/api/brief/{scan_id}")
    def brief(scan_id: str) -> dict:
        with scans_lock:
            result = scans.get(scan_id)
        if result is None:
            raise HTTPException(404, "That scan is no longer in memory. Run it again.")
        try:
            written = ai.write_brief(
                result, model=settings.ai_model, api_key=settings.anthropic_key, effort=settings.ai_effort
            )
        except ai.BriefError as err:
            raise HTTPException(502, str(err)) from err
        return asdict(written)

    @app.get("/api/positions")
    def held() -> PositionsReview:
        """Every position on file beside today's numbers. They are listed even when nothing can be priced."""
        try:
            found, unread = store.load()
        except positions.StoreError as err:
            raise HTTPException(500, str(err)) from err
        if scanner is None:
            listed = [holdings.assess(p, today()) for p in found]
            # No data source: the positions are still there to read, with the checks that need only the date.
            review = PositionsReview(
                datetime.now(timezone.utc), "unconfigured", None, listed, warnings=[setup_error] if setup_error else []
            )
        else:
            try:
                review = scanner.review(found)
            except Exception as err:  # as for a scan: report it rather than a bare 500
                log.exception("review failed")
                raise HTTPException(502, f"Positions could not be checked: {err}") from err
        if unread:
            review.warnings.insert(0, positions.unread_note(unread, store.path))
        return review

    @app.post("/api/positions", status_code=201)
    def add_position(body: PositionBody) -> Position:
        try:
            made = positions.make(
                body.underlying,
                body.kind,
                body.strike,
                body.expiry,
                body.side,
                body.quantity,
                body.entry_price,
                body.opened or today(),
            )
            return store.add(made)
        except ValueError as err:
            raise HTTPException(422, str(err)) from err
        except positions.StoreError as err:
            raise HTTPException(500, str(err)) from err

    @app.delete("/api/positions/{position_id}")
    def remove_position(position_id: str) -> dict:
        try:
            removed = store.remove(position_id)
        except positions.StoreError as err:
            raise HTTPException(500, str(err)) from err
        if not removed:
            raise HTTPException(404, "There is no position with that id.")
        return {"removed": position_id}

    return app
