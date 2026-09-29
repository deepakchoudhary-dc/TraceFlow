"""Disposable, rebuildable derived state.

TraceFlow's architecture separates two kinds of stored data:

* **Evidence** — the event log and the baseline snapshots. Immutable, and never
  rewritten. Losing it loses information that cannot be recovered.
* **Derived state** — anything computed from evidence. Rebuildable, and therefore
  disposable.

This module holds the second kind. Parsing a Python file is the expensive part of
analysis, so results are cached against the hash of the file's contents. When the
analyzer improves, the fix is to delete ``.traceflow/derived/`` and let it re-derive —
no migration, and no sessions carrying conclusions drawn by an older, worse analyzer.

Keying on content rather than path has two useful consequences: renaming a file keeps
its cache entry, and a baseline snapshot is already cached, because the blob store
digest *is* the cache key.

The ``kind`` namespacing exists so a change to the analyzer invalidates its own
entries. Without it, improving the parser would keep serving results the old parser
produced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DERIVED_DIRNAME = "derived"
ANALYSIS_DIRNAME = "analysis"


class AnalysisCache:
    """A content-addressed cache of analysis results, stored as JSON."""

    def __init__(self, state_dir: Path) -> None:
        self._root = state_dir / DERIVED_DIRNAME / ANALYSIS_DIRNAME

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, kind: str, digest: str) -> Path:
        return self._root / kind / digest[:2] / f"{digest}.json"

    def get(self, kind: str, digest: str) -> dict[str, Any] | None:
        """Return a cached payload, or ``None`` when there is none to trust.

        A corrupt entry is treated as a miss rather than an error: the cache is
        derived state, so the worst case of discarding it is recomputing it.
        """
        path = self.path_for(kind, digest)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def put(self, kind: str, digest: str, payload: dict[str, Any]) -> None:
        path = self.path_for(kind, digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
