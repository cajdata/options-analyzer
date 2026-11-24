"""Small JSON-on-disk cache so slow or rate-limited lookups are not repeated."""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
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
        """Store `value`. A cache that cannot be written is skipped, never fatal.

        Scans write from several threads. The value goes to a temporary file that
        is then swapped into place, so a reader never sees half a file.
        """
        payload = json.dumps({"stored_at": datetime.now(timezone.utc).isoformat(), "value": value})
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(dir=self.root, suffix=".tmp")
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as file:
                    file.write(payload)
                os.replace(temporary, self._path(key))
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(temporary)
                raise
        except OSError:
            return
