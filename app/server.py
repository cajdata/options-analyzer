"""Local web dashboard and JSON API."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from app import ai
from app.analysis import scoring
from app.config import ROOT, Settings, load_settings
from app.models import ScanResult
from app.scanner import ConfigError, Scanner, ScanRequest, build_scanner

KEPT_SCANS = 10


class ScanBody(BaseModel):
    symbols: list[str] = Field(default_factory=list, max_length=40)
    lens: Literal["buy", "sell"] = "buy"
    kind: Literal["call", "put", "both"] = "both"
    dte_min: int = Field(7, ge=0, le=730)
    dte_max: int = Field(60, ge=1, le=730)
    top_n: int = Field(20, ge=1, le=100)

    @field_validator("symbols")
    @classmethod
    def clean(cls, symbols: list[str]) -> list[str]:
        cleaned = [s.strip().upper() for s in symbols if s.strip()]
        for symbol in cleaned:
            if len(symbol) > 8 or not symbol.replace(".", "").isalnum():
                raise ValueError(f"not a ticker symbol: {symbol!r}")
        return list(dict.fromkeys(cleaned))


def create_app(settings: Settings | None = None, demo: bool = False) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="Options Analyzer", docs_url=None, redoc_url=None)
    scans: OrderedDict[str, ScanResult] = OrderedDict()
    scanner: Scanner | None = None
    setup_error: str | None = None
    try:
        scanner = build_scanner(settings, demo=demo)
    except ConfigError as err:
        setup_error = str(err)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(ROOT / "public" / "index.html")

    @app.get("/api/status")
    def status() -> dict:
        return {
            "mode": scanner.market.mode if scanner else "unconfigured",
            "options_feed": scanner.market.options_feed if scanner else None,
            "earnings_sources": scanner.earnings.sources if scanner else [],
            "ai_available": ai.available(),
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
        )
        try:
            result = scanner.scan(request)
        except Exception as err:  # data-provider failure: report it rather than a bare 500
            raise HTTPException(502, f"Scan failed: {err}") from err
        scans[result.id] = result
        while len(scans) > KEPT_SCANS:
            scans.popitem(last=False)
        return result

    @app.post("/api/brief/{scan_id}")
    def brief(scan_id: str) -> dict:
        result = scans.get(scan_id)
        if result is None:
            raise HTTPException(404, "That scan is no longer in memory. Run it again.")
        try:
            return asdict(ai.write_brief(result, model=settings.ai_model))
        except ai.BriefError as err:
            raise HTTPException(502, str(err)) from err

    return app
