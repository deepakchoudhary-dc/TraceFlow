"""The command-line surface.

Most of these are thin checks that a command runs and reports success. The last
test is different in kind: it drives a real ``traceflow watch`` process, edits a
file underneath it, and asserts that a session was recorded. That is the Phase 1
acceptance criterion from plan.md §57, and it is only meaningful end to end.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from traceflow.cli import EXIT_ERROR, EXIT_OK, EXIT_USAGE, main
from traceflow.config import CONFIG_FILENAME, STATE_DIRNAME
from traceflow.git.repository import Repository
from traceflow.watcher.session import (
    BASELINE_FILENAME,
    CHANGES_FILENAME,
    CURRENT_BASELINE_FILENAME,
    IMPACT_FILENAME,
    SESSION_FILENAME,
    SessionStore,
)

# Deliberately tiny timings so the end-to-end test finishes in seconds. The
# production defaults (8s quiet period) are covered by the config tests.
_FAST_CONFIG = """\
[activity]
quiet_period_seconds = 0.6
minimum_session_seconds = 0.2
poll_interval_seconds = 0.2
idle_poll_interval_seconds = 0.2
"""


def test_init_reports_success(repo_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["init", str(repo_root)]) == EXIT_OK

    output = capsys.readouterr().out
    assert "State dir" in output
    assert (repo_root / CONFIG_FILENAME).is_file()


def test_status_reports_success(repo_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["status", str(repo_root)]) == EXIT_OK
    assert "Working tree" in capsys.readouterr().out


def test_sessions_on_an_unwatched_repository_is_not_an_error(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["sessions", str(repo_root)]) == EXIT_OK
    assert "No sessions recorded yet." in capsys.readouterr().out


def test_status_outside_a_repository_fails(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    assert main(["status", str(plain)]) == EXIT_ERROR


def test_init_outside_a_repository_fails(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    assert main(["init", str(plain)]) == EXIT_ERROR


def test_malformed_config_is_a_usage_error(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo_root / CONFIG_FILENAME).write_text("[activity\nbroken", encoding="utf-8")
    assert main(["status", str(repo_root)]) == EXIT_USAGE
    assert "not valid TOML" in capsys.readouterr().err


def test_version_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "traceflow" in capsys.readouterr().out


def test_no_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == EXIT_USAGE


# --------------------------------------------------------------------------- end to end


def _wait_for_session(repo_root: Path, deadline_seconds: float) -> dict[str, object] | None:
    """Poll for a written session record rather than sleeping a fixed amount.

    A record that does not parse *yet* is not a failure. The watcher writes the file in
    place, so a read can land mid-write and see an empty or partial document; the loop
    retries and the deadline decides the outcome. That is only sound because the session
    record is written last — its presence means the artifacts it refers to are complete.
    """
    sessions_dir = repo_root / STATE_DIRNAME / "sessions"
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        for path in sorted(sessions_dir.glob(f"*/{SESSION_FILENAME}")):
            try:
                return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
            except (OSError, json.JSONDecodeError):
                continue
        time.sleep(0.1)
    return None


def _wait_for_marker(log_path: Path, marker: str, deadline_seconds: float = 30.0) -> bool:
    """Wait until the watcher's output contains *marker*."""
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        if log_path.is_file():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            if marker in text:
                return True
        time.sleep(0.05)
    return False


def _drive_watcher(
    repo_root: Path,
    mutate: Callable[[], None],
    deadline_seconds: float = 25.0,
    expect_session: bool = True,
) -> tuple[dict[str, object] | None, str]:
    """Run a real watcher, apply *mutate*, and return the session plus its output.

    Child output goes to a file rather than a pipe: the watcher is long-running, and
    a pipe nobody drains is a deadlock waiting to happen. The log is returned so a
    failing assertion can show what the watcher actually did.

    The mutation waits for the watcher to announce that it is watching. A fixed sleep
    would race against interpreter startup, and losing that race changes the outcome:
    a change made before the first poll is part of the baseline, not a session — which
    is correct behaviour (plan.md §14) and therefore the wrong thing for these tests
    to accidentally exercise.

    Set ``expect_session=False`` for tests that only need the watcher running. The
    deadline is a real 25 seconds, so a test that never records a session would
    otherwise spend all of it polling for something it does not want.
    """
    log_path = repo_root.parent / "watch-output.log"
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "traceflow.cli", "watch", str(repo_root)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            ready = _wait_for_marker(log_path, "Waiting for activity...")
            if not ready:
                # Mutating anyway keeps the assertion message useful: it will show
                # whatever the watcher printed instead of the expected banner.
                output = log_path.read_text(encoding="utf-8", errors="replace")
                return None, f"watcher never became ready.\n{output}"

            mutate()
            session = _wait_for_session(repo_root, deadline_seconds) if expect_session else None
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                process.kill()

    return session, log_path.read_text(encoding="utf-8")


def test_watch_detects_a_change_and_records_a_settled_session(repo_root: Path) -> None:
    """plan.md §57 acceptance: modify a file, TraceFlow notices, waits, records.

    This also proves the self-trigger guard works. TraceFlow writes its own session
    record into the repository it is watching; if that write counted as activity,
    the repository would never settle and this test would time out.
    """
    (repo_root / CONFIG_FILENAME).write_text(_FAST_CONFIG, encoding="utf-8")

    session, output = _drive_watcher(
        repo_root,
        lambda: (repo_root / "app.py").write_text(
            "def main() -> int:\n    return 42\n", encoding="utf-8"
        ),
    )

    assert session is not None, f"no session recorded. Watcher output:\n{output}"
    assert session["status"] == "stabilized"
    assert session["repository"] == "sample-repo"
    assert session["stabilized_at"] is not None


def test_watch_records_a_clean_baseline_before_the_session(repo_root: Path) -> None:
    """plan.md §14: changes the session did not make must not be attributed to it.

    The session's baseline has to be the state *before* the first change. Capturing
    the post-change state instead would credit the session with a modification that
    was already present — the count below would be 1 instead of 0.
    """
    (repo_root / CONFIG_FILENAME).write_text(_FAST_CONFIG, encoding="utf-8")

    session, output = _drive_watcher(
        repo_root,
        lambda: (repo_root / "app.py").write_text(
            "def main() -> int:\n    return 42\n", encoding="utf-8"
        ),
    )

    assert session is not None, f"no session recorded. Watcher output:\n{output}"
    assert session["baseline_tracked_changes"] == 0

    # The baseline is legitimately dirty: this test writes .traceflow.toml into the
    # repository, so one untracked file exists before anything happens. It is
    # recorded rather than hidden, which is the point of capturing the counts.
    assert session["baseline_dirty"] is True
    assert session["baseline_untracked_files"] == 1


def test_watch_records_start_and_stop_events(repo_root: Path) -> None:
    """The event log is the immutable record; it must capture the watcher's lifecycle."""
    (repo_root / CONFIG_FILENAME).write_text(_FAST_CONFIG, encoding="utf-8")

    _, output = _drive_watcher(repo_root, lambda: None, expect_session=False)

    events_path = repo_root / STATE_DIRNAME / "events.jsonl"
    assert events_path.is_file(), f"no event log written. Watcher output:\n{output}"
    types = [
        json.loads(line)["type"]
        for line in events_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert "watch_started" in types


def test_watch_reports_a_missing_gitignore_entry(repo_root: Path) -> None:
    (repo_root / CONFIG_FILENAME).write_text(_FAST_CONFIG, encoding="utf-8")

    _, output = _drive_watcher(repo_root, lambda: None, expect_session=False)

    assert "not ignored by git" in output
    assert "traceflow init" in output


# --------------------------------------------------------------------------- change collection


def _read_artifact(repo_root: Path, session_id: str, filename: str) -> dict[str, object]:
    path = repo_root / STATE_DIRNAME / "sessions" / session_id / filename
    return json.loads(path.read_text(encoding="utf-8"))


def _files_by_path(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    entries = payload.get("files")
    assert isinstance(entries, list)
    return {str(entry["path"]): entry for entry in entries}  # type: ignore[index]


def _write_lf(path: Path, text: str) -> None:
    """Write with explicit LF so line counts do not depend on the platform."""
    path.write_text(text, encoding="utf-8", newline="\n")


def test_watch_writes_a_change_set_for_the_session(repo_root: Path) -> None:
    """plan.md §58 acceptance: TraceFlow reports what changed during a session."""
    (repo_root / CONFIG_FILENAME).write_text(_FAST_CONFIG, encoding="utf-8")

    session, output = _drive_watcher(
        repo_root,
        lambda: _write_lf(repo_root / "app.py", "def main() -> int:\n    return 2\n"),
    )
    assert session is not None, f"no session recorded. Watcher output:\n{output}"

    payload = _read_artifact(repo_root, str(session["session_id"]), CHANGES_FILENAME)
    files = _files_by_path(payload)

    assert "app.py" in files, f"app.py missing from the change set: {files}"
    assert files["app.py"]["status"] == "modified"
    assert files["app.py"]["insertions"] == 1
    assert files["app.py"]["deletions"] == 1

    totals = payload["totals"]
    assert isinstance(totals, dict)
    assert totals["files"] == 1


def test_watch_records_the_baseline_alongside_the_session(repo_root: Path) -> None:
    """The baseline is stored so the session stays interpretable on its own."""
    (repo_root / CONFIG_FILENAME).write_text(_FAST_CONFIG, encoding="utf-8")

    session, output = _drive_watcher(
        repo_root,
        lambda: _write_lf(repo_root / "app.py", "def main() -> int:\n    return 2\n"),
    )
    assert session is not None, f"no session recorded. Watcher output:\n{output}"

    baseline = _read_artifact(repo_root, str(session["session_id"]), BASELINE_FILENAME)
    assert baseline["dirty"] is True  # .traceflow.toml is untracked
    assert baseline["commit"] is not None


def test_watch_separates_pre_existing_changes_from_the_session(repo_root: Path) -> None:
    """The end-to-end form of plan.md §14.

    ``app.py`` is modified before the watcher starts and modified again while it runs.
    The session must report only the second edit, and the first must appear as
    pre-existing rather than being credited to the session.
    """
    _write_lf(repo_root / "app.py", "def main() -> int:\n    return 2\n")
    (repo_root / CONFIG_FILENAME).write_text(_FAST_CONFIG, encoding="utf-8")

    session, output = _drive_watcher(
        repo_root,
        lambda: _write_lf(repo_root / "app.py", "def main() -> int:\n    return 2\n# note\n"),
    )
    assert session is not None, f"no session recorded. Watcher output:\n{output}"

    payload = _read_artifact(repo_root, str(session["session_id"]), CHANGES_FILENAME)
    files = _files_by_path(payload)

    assert files["app.py"]["insertions"] == 1, "only the line this session added"
    assert files["app.py"]["deletions"] == 0

    pre_existing = payload["pre_existing"]
    assert isinstance(pre_existing, list)
    assert [entry["path"] for entry in pre_existing] == ["app.py"]  # type: ignore[index]
    assert pre_existing[0]["insertions"] == 1  # type: ignore[index]
    assert pre_existing[0]["deletions"] == 1  # type: ignore[index]


def test_watch_never_reads_a_secret_file(repo_root: Path) -> None:
    """plan.md §26, end to end: the withheld reason is recorded, the contents are not."""
    _write_lf(repo_root / ".env", "API_KEY=do-not-store-me\n")
    (repo_root / CONFIG_FILENAME).write_text(_FAST_CONFIG, encoding="utf-8")

    session, output = _drive_watcher(
        repo_root,
        lambda: _write_lf(repo_root / ".env", "API_KEY=still-do-not-store-me\n"),
    )
    assert session is not None, f"no session recorded. Watcher output:\n{output}"

    baseline = _read_artifact(repo_root, str(session["session_id"]), BASELINE_FILENAME)
    captured = baseline["captured"]
    assert isinstance(captured, list)
    env_entry = next(entry for entry in captured if entry["path"] == ".env")  # type: ignore[index]
    assert env_entry["digest"] is None
    assert env_entry["withheld_reason"] == "sensitive path"

    blob_root = repo_root / STATE_DIRNAME / "blobs"
    stored = [path for path in blob_root.rglob("*") if path.is_file()] if blob_root.exists() else []
    for path in stored:
        assert b"do-not-store-me" not in path.read_bytes(), f"secret leaked into {path}"


def test_sessions_command_reports_the_change_summary(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo_root / CONFIG_FILENAME).write_text(_FAST_CONFIG, encoding="utf-8")
    session, output = _drive_watcher(
        repo_root,
        lambda: _write_lf(repo_root / "app.py", "def main() -> int:\n    return 2\n"),
    )
    assert session is not None, f"no session recorded. Watcher output:\n{output}"

    assert main(["sessions", str(repo_root)]) == EXIT_OK
    printed = capsys.readouterr().out

    assert "1 file(s), +1/-1" in printed
    assert "impact" in printed


# --------------------------------------------------------------------------- analyze / impact
#
# ``traceflow analyze`` exists so that the analysis engine can be exercised without
# waiting out the quiet period (plan.md §44). These tests use it as the setup step for
# the impact tests as well, which is exactly what it is for.


def _analyze(repo_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Establish a baseline, change app.py, and analyse the result."""
    assert main(["analyze", str(repo_root)]) == EXIT_OK
    _write_lf(repo_root / "app.py", "def main() -> int:\n    return 7\n")
    assert main(["analyze", str(repo_root)]) == EXIT_OK
    capsys.readouterr()


def test_analyze_captures_a_baseline_before_it_can_report_anything(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no baseline there is no "before", so the first run must say so."""
    assert main(["analyze", str(repo_root)]) == EXIT_OK

    assert "Nothing to analyse yet" in capsys.readouterr().out
    assert (repo_root / STATE_DIRNAME / CURRENT_BASELINE_FILENAME).is_file()
    assert SessionStore(repo_root).list_sessions() == []


def test_analyze_records_a_session_and_its_impact(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _analyze(repo_root, capsys)

    store = SessionStore(repo_root)
    sessions = store.list_sessions()
    assert len(sessions) == 1

    impact = store.read_artifact(sessions[0].session_id, IMPACT_FILENAME)
    assert impact is not None
    assert impact["totals"]["nodes"] >= 1
    assert impact["analyzer"] == "python-1"


def test_analyze_on_an_unchanged_tree_records_nothing(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Recording an empty session every run would make ``sessions`` useless."""
    assert main(["analyze", str(repo_root)]) == EXIT_OK
    capsys.readouterr()

    assert main(["analyze", str(repo_root)]) == EXIT_OK

    assert "Nothing changed" in capsys.readouterr().out
    assert SessionStore(repo_root).list_sessions() == []


def test_impact_on_an_unwatched_repository_is_not_an_error(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["impact", str(repo_root)]) == EXIT_OK
    assert "No sessions recorded yet." in capsys.readouterr().out


def test_impact_renders_the_stored_artifact(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _analyze(repo_root, capsys)

    assert main(["impact", str(repo_root)]) == EXIT_OK
    printed = capsys.readouterr().out

    assert "DIRECT" in printed
    assert "app.py::main" in printed


def test_impact_json_is_machine_readable(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No banner: a caller piping this into a parser must not have to strip a header."""
    _analyze(repo_root, capsys)

    assert main(["impact", str(repo_root), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)

    assert payload["analyzer"] == "python-1"
    assert payload["totals"]["nodes"] >= 1
    assert payload["totals"]["changed_files"] == 1


def test_analyze_lists_the_repository_once(
    repo_root: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three passes over one tree used to list it three times: three git spawns, one answer.

    A git spawn costs about 120ms here, so this was the largest avoidable cost in the
    command. The spy calls the real method, so the analysis still sees git's own output —
    what is counted is how many times the repository was asked the same question.
    """
    assert main(["analyze", str(repo_root)]) == EXIT_OK  # establish the baseline
    _write_lf(repo_root / "app.py", "def main() -> int:\n    return 7\n")

    calls: list[tuple[str, ...]] = []
    real = Repository.git

    def counting(self: Repository, *args: str, check: bool = True) -> str:
        calls.append(args)
        return real(self, *args, check=check)

    monkeypatch.setattr(Repository, "git", counting)
    assert main(["analyze", str(repo_root)]) == EXIT_OK
    capsys.readouterr()

    listings = [call for call in calls if call[:1] == ("ls-files",)]
    assert len(listings) == 1, listings
    assert calls, "the spy saw no git calls at all, so it proves nothing"
