"""Local web dashboard and JSON API."""

from __future__ import annotations

import logging
import re
import threading
from collections import OrderedDict
from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from app import ai
from app.analysis import scoring
from app.config import ROOT, ConfigError, Settings, load_settings
from app.models import ScanResult
from app.scanner import Scanner, ScanRequest, build_scanner

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

    @field_validator("symbols")
    @classmethod
    def clean(cls, symbols: list[str]) -> list[str]:
        # Share classes are written both ways (BRK-B, BRK.B); the data provider wants the dot.
        cleaned = [s.strip().upper().replace("-", ".") for s in symbols if s.strip()]
        for symbol in cleaned:
            if not _SYMBOL.fullmatch(symbol):
                raise ValueError(f"not a ticker symbol: {symbol!r}")
        return list(dict.fromkeys(cleaned))


def create_app(settings: Settings | None = None, demo: bool = False) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="Options Analyzer", docs_url=None, redoc_url=None)
    scans: OrderedDict[str, ScanResult] = OrderedDict()
    scans_lock = threading.Lock()  # request handlers run on several threads
    scanner: Scanner | None = None
    setup_error: str | None = None
    try:
        scanner = build_scanner(settings, demo=demo)
    except ConfigError as err:
        setup_error = str(err)

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
        }

    @app.post("/api/scan")
    def scan(body: ScanBody) -> ScanResult:
        if scanner is None:
            raise HTTPException(503, setup_error)
        if body.dte_max < body.dte_min:
            raise HTTPException(422, "dte_max must not be below dte_min")
        request = ScanRequest(
            symbols=body.symbols,
            lens=body.lens,
            kind=body.kind,
            dte_min=body.dte_min,
            dte_max=body.dte_max,
            top_n=body.top_n,
            budget=body.budget,
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

    return app
