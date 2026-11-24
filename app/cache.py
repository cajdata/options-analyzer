"""Small JSON-on-disk cache so slow or rate-limited lookups are not repeated."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class NullCache:
    """Stores nothing. Used for demo data, which must never be mistaken for real data."""

    def entry(self, key: str) -> None:
        return None

    def get(self, key: str, max_age: timedelta) -> None:
        return None

    def set(self, key: str, value: Any) -> None:
        return None


class DiskCache:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / (re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".json")

    def entry(self, key: str) -> tuple[Any, timedelta] | None:
        """Return (value, age) for `key`, or None if it was never stored."""
        try:
            raw = json.loads(self._path(key).read_text(encoding="utf-8"))
            stored = datetime.fromisoformat(raw["stored_at"])
        except (OSError, ValueError, KeyError):
            return None
        return raw["value"], datetime.now(timezone.utc) - stored

    def get(self, key: str, max_age: timedelta) -> Any | None:
        found = self.entry(key)
        if found is None or found[1] > max_age:
            return None
        return found[0]

    def set(self, key: str, value: Any) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"stored_at": datetime.now(timezone.utc).isoformat(), "value": value}
        self._path(key).write_text(json.dumps(payload), encoding="utf-8")
