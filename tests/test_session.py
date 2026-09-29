"""Session records and the append-only store."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from traceflow.config import STATE_DIRNAME
from traceflow.watcher.session import (
    EVENT_SESSION_STABILIZED,
    EVENT_WATCH_STARTED,
    SESSION_FILENAME,
    STATUS_STABILIZED,
    Session,
    SessionEvent,
    SessionStore,
    new_session_id,
    now_iso,
)


def make_session(session_id: str = "2026-09-25T10-00-00-abcdef", **overrides: object) -> Session:
    payload: dict[str, object] = {
        "session_id": session_id,
        "repository": "sample-repo",
        "root": "/tmp/sample-repo",
        "started_at": "2026-09-25T10:00:00+00:00",
        "stabilized_at": "2026-09-25T10:00:12+00:00",
        "baseline_commit": "a" * 40,
        "baseline_dirty": False,
        "baseline_tracked_changes": 0,
        "baseline_untracked_files": 0,
        "status": STATUS_STABILIZED.value,
    }
    payload.update(overrides)
    return Session(**payload)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- identifiers


def test_session_id_is_sortable_and_readable() -> None:
    moment = datetime(2026, 9, 25, 14, 40, 12, tzinfo=timezone.utc)
    assert new_session_id(moment, suffix="a1b2c3") == "2026-09-25T14-40-12-a1b2c3"


def test_session_id_matches_the_documented_shape() -> None:
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-[0-9a-f]{6}$")
    assert pattern.match(new_session_id())


def test_session_ids_in_the_same_second_differ() -> None:
    moment = datetime(2026, 9, 25, 14, 40, 12, tzinfo=timezone.utc)
    assert new_session_id(moment) != new_session_id(moment)


def test_now_iso_carries_an_explicit_offset() -> None:
    stamp = now_iso()
    assert stamp.endswith(("Z", "+00:00")) or "+" in stamp[10:] or "-" in stamp[10:]


# --------------------------------------------------------------------------- store


def test_state_dir_lives_under_the_repository_root(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    assert store.state_dir == tmp_path / STATE_DIRNAME
    assert store.events_path.name == "events.jsonl"
    assert store.sessions_dir.name == "sessions"


def test_write_session_creates_the_expected_layout(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session = make_session()

    path = store.write_session(session)

    assert path == tmp_path / STATE_DIRNAME / "sessions" / session.session_id / SESSION_FILENAME
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8"))["session_id"] == session.session_id


def test_write_session_creates_missing_directories(tmp_path: Path) -> None:
    """The state directory can be removed by hand between runs."""
    store = SessionStore(tmp_path)
    assert not store.state_dir.exists()

    store.write_session(make_session())
    assert store.state_dir.is_dir()


def test_events_are_appended_not_replaced(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)

    store.append_event(SessionEvent(at=now_iso(), type=EVENT_WATCH_STARTED))
    store.append_event(SessionEvent(at=now_iso(), type=EVENT_SESSION_STABILIZED, session_id="abc"))

    lines = store.events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["type"] == EVENT_WATCH_STARTED
    assert json.loads(lines[1])["session_id"] == "abc"


def test_appended_events_survive_a_second_store_instance(tmp_path: Path) -> None:
    """Append-only is only meaningful across processes."""
    SessionStore(tmp_path).append_event(SessionEvent(at=now_iso(), type=EVENT_WATCH_STARTED))
    SessionStore(tmp_path).append_event(SessionEvent(at=now_iso(), type=EVENT_WATCH_STARTED))

    lines = SessionStore(tmp_path).events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_list_sessions_is_empty_before_anything_is_written(tmp_path: Path) -> None:
    assert SessionStore(tmp_path).list_sessions() == []


def test_list_sessions_returns_them_oldest_first(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.write_session(
        make_session("2026-09-25T12-00-00-cccccc", started_at="2026-09-25T12:00:00+00:00")
    )
    store.write_session(
        make_session("2026-09-25T10-00-00-aaaaaa", started_at="2026-09-25T10:00:00+00:00")
    )
    store.write_session(
        make_session("2026-09-25T11-00-00-bbbbbb", started_at="2026-09-25T11:00:00+00:00")
    )

    started = [session.started_at for session in store.list_sessions()]
    assert started == [
        "2026-09-25T10:00:00+00:00",
        "2026-09-25T11:00:00+00:00",
        "2026-09-25T12:00:00+00:00",
    ]


def test_list_sessions_round_trips_every_field(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    original = make_session(
        baseline_dirty=True,
        baseline_tracked_changes=3,
        baseline_untracked_files=2,
    )
    store.write_session(original)

    assert store.list_sessions() == [original]


def test_unreadable_session_records_are_skipped_not_raised(tmp_path: Path) -> None:
    """plan.md §46: a corrupt file must not make the whole history unavailable."""
    store = SessionStore(tmp_path)
    store.write_session(make_session("2026-09-25T10-00-00-aaaaaa"))

    broken_dir = store.sessions_dir / "2026-09-25T11-00-00-broken"
    broken_dir.mkdir(parents=True)
    (broken_dir / SESSION_FILENAME).write_text("{not json", encoding="utf-8")

    sessions = store.list_sessions()
    assert [session.session_id for session in sessions] == ["2026-09-25T10-00-00-aaaaaa"]


def test_session_record_with_unknown_fields_is_skipped(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    store.write_session(make_session("2026-09-25T10-00-00-aaaaaa"))

    odd_dir = store.sessions_dir / "2026-09-25T11-00-00-odd"
    odd_dir.mkdir(parents=True)
    (odd_dir / SESSION_FILENAME).write_text('{"unexpected": true}', encoding="utf-8")

    assert len(store.list_sessions()) == 1


def test_session_json_is_serialisable_with_unusual_paths(tmp_path: Path) -> None:
    """Filenames decoded with surrogateescape must not break the event log write."""
    store = SessionStore(tmp_path)
    store.append_event(
        SessionEvent(
            at=now_iso(),
            type=EVENT_SESSION_STABILIZED,
            session_id="abc",
            detail={"path": "weird-\udcff-name.py"},
        )
    )
    assert store.events_path.read_text(encoding="utf-8").strip()
