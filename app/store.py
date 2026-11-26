"""A list of entries in one JSON file that is the person's own: their positions, their journal.

Such a file cannot be fetched again, so it is handled the opposite way to the
cache:

* A file that does not read is an error, never an empty list. Read as empty,
  it would be written over by the next entry.
* A write that fails is an error, never skipped.

What an entry means is the caller's business (`app/positions.py`,
`app/journal.py`). Here an entry is whatever JSON was written, and it goes back
as it came, so one that does not read stays in the file untouched.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


class StoreError(RuntimeError):
    """A file of the person's own cannot be read or written. The message is written for the user."""


class ListFile:
    """Entries under one key of a JSON file, or in memory only when no path is given.

    Demo data lives in memory: it must never reach the disk, and the person's
    own file must never be shown as demo data.
    """

    def __init__(self, path: Path | str | None, key: str, what: str, seed: Iterable[Any] = ()) -> None:
        self.path = Path(path) if path is not None else None
        self.key = key  # "positions", "trades"
        self.what = what  # how the file is named to the person: "positions file", "journal file"
        self._memory = list(seed)

    def rows(self) -> list[Any]:
        """Every entry as it is written, whether or not it reads as one."""
        if self.path is None:
            return list(self._memory)
        try:
            # "utf-8-sig" reads a file with or without a byte order mark. An older Notepad and
            # PowerShell's Set-Content both put one in front, and these files are meant to be edited by hand.
            text = self.path.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            return []
        except (OSError, UnicodeError) as err:
            raise StoreError(f"The {self.what} {self.path} could not be read ({type(err).__name__}).") from err
        if not text.strip():
            return []
        try:
            rows = json.loads(text)[self.key]
            if not isinstance(rows, list):
                raise TypeError
        except (ValueError, KeyError, TypeError):
            raise StoreError(
                f"The {self.what} {self.path} does not read as a list of {self.key}. It has been left as it is, "
                "and nothing will be written to it until it is repaired or moved away."
            ) from None
        return rows

    def write(self, rows: list[Any]) -> None:
        if self.path is None:
            self._memory = rows
            return
        payload = json.dumps({"version": 1, self.key: rows}, indent=2) + "\n"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Written beside the file and swapped into place, so a crash never leaves half a file.
            handle, temporary = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as file:
                    file.write(payload)
                os.replace(temporary, self.path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(temporary)
                raise
        except OSError as err:
            raise StoreError(f"The {self.what} {self.path} could not be written ({type(err).__name__}).") from err
