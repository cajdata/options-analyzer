"""Settings read from the environment and an optional .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def _env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _symbols(text: str | None) -> list[str]:
    return [s.strip().upper() for s in (text or "").replace(";", ",").split(",") if s.strip()]


@dataclass(frozen=True)
class Settings:
    alpaca_key: str | None = None
    alpaca_secret: str | None = None
    alpaca_paper: bool = True
    options_feed: str = "indicative"
    stock_feed: str = "iex"
    alphavantage_key: str | None = None
    sec_user_agent: str | None = None
    anthropic_key: str | None = None
    ai_model: str | None = None
    watchlist: tuple[str, ...] = ()
    risk_free_rate: float = 0.04
    cache_dir: Path = ROOT / ".cache"
    universe_file: Path = ROOT / "data" / "universe.txt"

    @property
    def has_alpaca(self) -> bool:
        return bool(self.alpaca_key and self.alpaca_secret)

    def universe(self) -> list[str]:
        try:
            lines = self.universe_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        symbols = (line.split("#", 1)[0].strip().upper() for line in lines)
        return [s for s in symbols if s]


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    return Settings(
        alpaca_key=_env("ALPACA_API_KEY", "APCA_API_KEY_ID"),
        alpaca_secret=_env("ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY"),
        alpaca_paper=(_env("ALPACA_PAPER") or "true").lower() not in ("0", "false", "no"),
        options_feed=_env("ALPACA_OPTIONS_FEED") or "indicative",
        stock_feed=_env("ALPACA_STOCK_FEED") or "iex",
        alphavantage_key=_env("ALPHAVANTAGE_API_KEY"),
        sec_user_agent=_env("SEC_USER_AGENT"),
        anthropic_key=_env("ANTHROPIC_API_KEY"),
        ai_model=_env("AI_MODEL"),
        watchlist=tuple(_symbols(_env("WATCHLIST"))),
        risk_free_rate=float(_env("RISK_FREE_RATE") or 0.04),
    )
