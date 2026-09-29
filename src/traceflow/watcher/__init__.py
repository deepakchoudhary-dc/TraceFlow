"""Repository watching: activity detection, stability, and session lifecycle."""

from traceflow.watcher.activity import ActivitySource, PollingActivitySource
from traceflow.watcher.quiescence import QuiescenceDetector, WatchState
from traceflow.watcher.session import (
    BASELINE_FILENAME,
    CHANGES_FILENAME,
    EVENT_SESSION_STABILIZED,
    EVENT_WATCH_STARTED,
    EVENT_WATCH_STOPPED,
    STATUS_STABILIZED,
    SYMBOLS_FILENAME,
    Session,
    SessionEvent,
    SessionStatus,
    SessionStore,
)

__all__ = [
    "BASELINE_FILENAME",
    "CHANGES_FILENAME",
    "EVENT_SESSION_STABILIZED",
    "EVENT_WATCH_STARTED",
    "EVENT_WATCH_STOPPED",
    "STATUS_STABILIZED",
    "SYMBOLS_FILENAME",
    "ActivitySource",
    "PollingActivitySource",
    "QuiescenceDetector",
    "Session",
    "SessionEvent",
    "SessionStatus",
    "SessionStore",
    "WatchState",
]
