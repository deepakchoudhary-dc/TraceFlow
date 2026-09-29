"""Sessions and their on-disk store (plan.md §13, §14, §35).

A session is one coherent episode of repository change. It is the unit of analysis
for everything TraceFlow does — deliberately not a commit and not a pull request.
An agent may make dozens of edits, run tests, fail, and retry before any commit
exists; all of that belongs to one session, and a commit-scoped tool cannot see it.

Two storage decisions:

* ``events.jsonl`` is **append-only**. Events are the raw evidence and are never
  rewritten, so a later improvement to the analysis engine can re-derive everything
  downstream without having destroyed its inputs.
* ``sessions/<id>/session.json`` is the per-session record described by plan.md §35.

Baseline dirtiness is recorded on the session itself because plan.md §14 makes it a
correctness requirement: a repository that is already dirty when the watcher starts
has *pre-existing* changes, and those must never be attributed to the session. The
counts captured here are what let Phase 2 draw that line.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from traceflow.config import STATE_DIRNAME
from traceflow.stamps import new_stamp_id, now_iso

EVENTS_FILENAME = "events.jsonl"
SESSIONS_DIRNAME = "sessions"
SESSION_FILENAME = "session.json"
CHANGES_FILENAME = "changes.json"
BASELINE_FILENAME = "baseline.json"
SYMBOLS_FILENAME = "symbols.json"
IMPACT_FILENAME = "impact.json"
INTENT_FILENAME = "intent.json"
TESTS_FILENAME = "tests.json"

CURRENT_BASELINE_FILENAME = "current-baseline.json"
"""Where the repository last settled, independent of any session.

Distinct from a session's ``baseline.json``: that one is frozen at the moment the session
began, while this one advances every time the repository settles. ``traceflow analyze``
measures against it, which is what lets a manual analysis report "nothing changed" on a
second run instead of recording the same session twice.
"""

EVENT_WATCH_STARTED = "watch_started"
EVENT_WATCH_STOPPED = "watch_stopped"
EVENT_SESSION_STABILIZED = "session_stabilized"
EVENT_BASELINE_CAPTURED = "baseline_captured"

# ``now_iso`` and ``new_stamp_id`` are re-exported so callers working with sessions
# have one import to reach for. Declaring them here keeps that intent explicit
# rather than leaving them looking like an unused import.
__all__ = [
    "BASELINE_FILENAME",
    "CHANGES_FILENAME",
    "CURRENT_BASELINE_FILENAME",
    "EVENTS_FILENAME",
    "EVENT_BASELINE_CAPTURED",
    "EVENT_SESSION_STABILIZED",
    "EVENT_WATCH_STARTED",
    "EVENT_WATCH_STOPPED",
    "IMPACT_FILENAME",
    "INTENT_FILENAME",
    "SESSIONS_DIRNAME",
    "SESSION_FILENAME",
    "STATUS_STABILIZED",
    "SYMBOLS_FILENAME",
    "TESTS_FILENAME",
    "Session",
    "SessionEvent",
    "SessionStatus",
    "SessionStore",
    "new_session_id",
    "new_stamp_id",
    "now_iso",
]


class SessionStatus(str, Enum):
    """The lifecycle status of a session.

    Only ``STABILIZED`` exists, and no others are planned. An earlier version of this
    docstring said "later phases add the statuses that follow from analysis"; they never
    arrived, because a session is written *after* its analysis has run. The record therefore
    never describes a session that is mid-analysis, and a status for that moment would be a
    value nothing could ever observe.
    """

    STABILIZED = "stabilized"


STATUS_STABILIZED = SessionStatus.STABILIZED


@dataclass(frozen=True)
class Session:
    """One coherent episode of repository change (plan.md §13)."""

    session_id: str
    repository: str
    root: str
    started_at: str
    stabilized_at: str | None = None
    baseline_commit: str | None = None
    baseline_dirty: bool = False
    baseline_tracked_changes: int = 0
    baseline_untracked_files: int = 0
    baseline_id: str | None = None
    status: str = STATUS_STABILIZED.value

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SessionEvent:
    """A single append-only record in the event log."""

    at: str
    type: str
    session_id: str | None = None
    detail: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def new_session_id(moment: datetime | None = None, suffix: str | None = None) -> str:
    """A session identifier, in the form ``2026-09-25T14-40-12-a1b2c3``.

    Named here rather than at the call site so the session module's vocabulary
    stays readable; the mechanism itself lives in :mod:`traceflow.stamps`.
    """
    return new_stamp_id(moment, suffix)


def _read_json(path: Path) -> dict[str, object] | None:
    """Read a JSON object, or ``None`` when it is absent, unreadable or not an object."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


class SessionStore:
    """Reads and writes TraceFlow's state directory inside a repository."""

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    @property
    def state_dir(self) -> Path:
        return self._root / STATE_DIRNAME

    @property
    def events_path(self) -> Path:
        return self.state_dir / EVENTS_FILENAME

    @property
    def sessions_dir(self) -> Path:
        return self.state_dir / SESSIONS_DIRNAME

    def session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def ensure_state_dir(self) -> None:
        """Create the state directory if it is missing.

        Called defensively by the write paths as well: the directory can be removed
        by hand between runs, and losing a whole session because of that would be
        worse than one extra ``mkdir``.
        """
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def append_event(self, event: SessionEvent) -> None:
        """Append one event to the immutable log.

        ``ensure_ascii`` is left on deliberately. Paths can contain lone surrogates
        after ``surrogateescape`` decoding of unusual filenames, and escaping keeps
        the log writable and exactly round-trippable instead of raising mid-write.
        """
        self.ensure_state_dir()
        line = json.dumps(event.to_json(), sort_keys=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def write_session(self, session: Session) -> Path:
        """Write a session record and return the path it was written to."""
        self.ensure_state_dir()
        target_dir = self.session_dir(session.session_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / SESSION_FILENAME
        path.write_text(
            json.dumps(session.to_json(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def write_artifact(self, session_id: str, filename: str, payload: dict[str, object]) -> Path:
        """Write one JSON artifact into a session's directory (plan.md §35).

        Deliberately generic: the store owns the directory layout, while the modules
        that produce baselines and change sets own their own serialisation. Keeping
        it that way means the git evidence package never has to import the watcher
        package just to record what it found.
        """
        self.ensure_state_dir()
        target_dir = self.session_dir(session_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / filename
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def read_artifact(self, session_id: str, filename: str) -> dict[str, object] | None:
        """Read a JSON artifact back, or ``None`` if it is absent or unreadable."""
        return _read_json(self.session_dir(session_id) / filename)

    def write_state(self, filename: str, payload: dict[str, object]) -> Path:
        """Write one JSON file at the root of the state directory.

        For state that outlives a session. Kept separate from :meth:`write_artifact` so
        the two layouts cannot be confused: an artifact is addressed by session id, state
        is not.
        """
        self.ensure_state_dir()
        path = self.state_dir / filename
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def read_state(self, filename: str) -> dict[str, object] | None:
        """Read a JSON state file back, or ``None`` if it is absent or unreadable."""
        return _read_json(self.state_dir / filename)

    def list_sessions(self) -> list[Session]:
        """Return every readable session, oldest first.

        Unreadable records are skipped rather than raised. A single corrupt file
        must not make the whole history unavailable (plan.md §46), and the cost of
        skipping is that one session is missing from a listing — not that the tool
        stops working.
        """
        if not self.sessions_dir.is_dir():
            return []

        sessions: list[Session] = []
        for path in sorted(self.sessions_dir.glob(f"*/{SESSION_FILENAME}")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                sessions.append(Session(**payload))
            except (OSError, json.JSONDecodeError, TypeError):
                continue

        sessions.sort(key=lambda session: session.started_at)
        return sessions
