"""Small JSON-on-disk cache so slow or rate-limited lookups are not repeated."""

from __future__ import annotations

import contextlib
import glob
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

    def keys(self, prefix: str) -> list[str]:
        return []


class MemoryCache:
    """Keeps values for as long as the program runs, and nowhere else.

    The demo's pool of stocks lives here: it has to be kept between scans to be
    of any use, and demo data must never reach the disk.
    """

    def __init__(self) -> None:
        self._values: dict[str, tuple[Any, datetime]] = {}

    def entry(self, key: str) -> tuple[Any, timedelta] | None:
        found = self._values.get(key)
        return None if found is None else (found[0], datetime.now(timezone.utc) - found[1])

    def get(self, key: str, max_age: timedelta) -> Any | None:
        found = self.entry(key)
        return None if found is None or found[1] > max_age else found[0]

    def set(self, key: str, value: Any) -> None:
        self._values[key] = (value, datetime.now(timezone.utc))

    def keys(self, prefix: str) -> list[str]:
        return sorted(key for key in self._values if key.startswith(prefix))


class DiskCache:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / (re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".json")

    def keys(self, prefix: str) -> list[str]:
        """Every key stored that starts with `prefix`, as it is spelled on disk."""
        return sorted(path.stem for path in self.root.glob(glob.escape(prefix) + "*.json"))

    def entry(self, key: str) -> tuple[Any, timedelta] | None:
        """Return (value, age) for `key`, or None if it was never stored or does not read as an entry.

        A file of the wrong shape, one with no value, and one stamped with a time
        that has no zone are all "never stored". Every reader treats None as a
        miss, so one spoiled file costs a fresh lookup and nothing else.
        """
        try:
            raw = json.loads(self._path(key).read_text(encoding="utf-8"))
            age = datetime.now(timezone.utc) - datetime.fromisoformat(raw["stored_at"])
            return raw["value"], age
        except (OSError, ValueError, KeyError, TypeError):
            return None

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
