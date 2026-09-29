"""Configurable test execution (plan.md §22, §23, §64).

The safety rules are tested as behaviour, not as intentions: nothing runs when the
config is off, the command comes from config and is split without a shell, paths
follow a ``--`` separator, and a result is a fact about a run — never a verdict on
the change. The end-to-end test runs a real passing and a real failing pytest inside
a fixture repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from traceflow import testing as tf_testing
from traceflow.cli import EXIT_OK, main
from traceflow.config import Config, default_config, parse_config
from traceflow.git.repository import Repository
from traceflow.testing import (
    TestRun as _TestRun,
)
from traceflow.testing import (
    TestRunStatus as _TestRunStatus,
)
from traceflow.testing import (
    run_session_tests,
)
from traceflow.ui.delivery import load_delivery
from traceflow.watcher.session import SessionStore


def select_paths(
    impacted_paths: tuple[str, ...], changed_paths: tuple[str, ...]
) -> tuple[str, ...]:
    """Wrapper so pytest does not collect the helper as a test (its name starts with ``test_``)."""
    return tf_testing.test_paths_from(impacted_paths, changed_paths)


def run_from_json(payload: dict[str, object] | None) -> _TestRun | None:
    """Wrapper for the same reason: not a test, despite the name."""
    return tf_testing.test_run_from_json(payload)


# --------------------------------------------------------------------------- config


def test_tests_are_disabled_by_default() -> None:
    """plan.md §22: never run a repository's tests without explicit configuration."""
    tests = default_config().tests
    assert tests.enabled is False
    assert tests.command == "pytest"
    assert tests.timeout_seconds == 300.0


def test_tests_section_is_accepted_and_parsed() -> None:
    config = parse_config(
        {"tests": {"enabled": True, "command": "python -m pytest -q", "timeout_seconds": 12.5}}
    )
    assert config.tests.enabled is True
    assert config.tests.command == "python -m pytest -q"
    assert config.tests.timeout_seconds == 12.5


def test_enabled_must_be_a_boolean() -> None:
    with pytest.raises(Exception, match=r"\[tests\]\.enabled"):
        parse_config({"tests": {"enabled": "yes"}})


def test_a_non_numeric_timeout_is_rejected() -> None:
    with pytest.raises(Exception, match=r"\[tests\]\.timeout_seconds"):
        parse_config({"tests": {"timeout_seconds": "soon"}})


def test_an_empty_command_is_rejected() -> None:
    with pytest.raises(Exception, match=r"\[tests\]\.command"):
        parse_config({"tests": {"command": "   "}})


def test_the_rendered_config_round_trips_with_the_tests_section() -> None:
    from traceflow.compat import tomllib
    from traceflow.config import render_default_config

    assert parse_config(tomllib.loads(render_default_config())) == default_config()


# --------------------------------------------------------------------------- selection


def test_only_test_paths_are_selected() -> None:
    paths = select_paths(
        impacted_paths=("src/app.py", "tests/test_app.py"),
        changed_paths=("tests/test_new.py", "src/other.py"),
    )
    assert paths == ("tests/test_new.py", "tests/test_app.py")


def test_a_changed_test_comes_first_and_duplicates_collapse() -> None:
    paths = select_paths(
        impacted_paths=("tests/test_a.py",),
        changed_paths=("tests/test_a.py", "tests/test_b.py"),
    )
    assert paths == ("tests/test_a.py", "tests/test_b.py")


def test_windows_separators_are_normalised() -> None:
    paths = select_paths(impacted_paths=(), changed_paths=("tests" + chr(92) + "test_a.py",))
    assert paths == ("tests/test_a.py",)


# --------------------------------------------------------------------------- the runner


def _repo_config(command: str = "pytest", enabled: bool = True) -> Config:
    return parse_config({"tests": {"enabled": enabled, "command": command}})


def test_a_disabled_config_returns_none_without_running_anything(repo: Repository) -> None:
    assert (
        run_session_tests(
            repo, default_config(), impacted_paths=("tests/test_a.py",), changed_paths=()
        )
        is None
    )


def test_no_reached_tests_is_recorded_as_not_run(repo: Repository) -> None:
    run = run_session_tests(repo, _repo_config(), impacted_paths=("src/app.py",), changed_paths=())
    assert run is not None
    assert run.status is _TestRunStatus.NOT_RUN
    assert not run.ran


def test_paths_go_after_the_separator(repo: Repository, repo_root: Path) -> None:
    """A filename can never become an option: paths follow ``--`` and no shell runs."""
    (repo_root / "tests").mkdir()
    (repo_root / "tests" / "test_ok.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8", newline="\n"
    )
    run = run_session_tests(
        repo, _repo_config(), impacted_paths=("tests/test_ok.py",), changed_paths=()
    )
    assert run is not None
    assert run.command[-2] == "--"
    assert run.command[-1] == "tests/test_ok.py"
    assert run.status is _TestRunStatus.PASSED
    assert run.passed == 1


def test_a_failing_run_is_a_fact_about_the_run(repo: Repository, repo_root: Path) -> None:
    (repo_root / "tests").mkdir()
    (repo_root / "tests" / "test_bad.py").write_text(
        "def test_bad():\n    assert False\n", encoding="utf-8", newline="\n"
    )
    run = run_session_tests(
        repo, _repo_config(), impacted_paths=("tests/test_bad.py",), changed_paths=()
    )
    assert run is not None
    assert run.status is _TestRunStatus.FAILED
    assert run.failed == 1
    assert run.output_tail is not None and "test_bad" in run.output_tail


def test_a_missing_executable_is_an_error_not_a_crash(repo: Repository) -> None:
    run = run_session_tests(
        repo,
        _repo_config(command="definitely-not-a-real-command-xyz"),
        impacted_paths=("tests/test_a.py",),
        changed_paths=(),
    )
    assert run is not None
    assert run.status is _TestRunStatus.ERROR
    assert "PATH" in run.note


# --------------------------------------------------------------------------- artifact


def test_the_run_round_trips_through_json() -> None:
    run = _TestRun(
        status=_TestRunStatus.PASSED,
        command=("pytest", "--", "tests/test_a.py"),
        exit_code=0,
        duration_seconds=1.5,
        output_tail="1 passed",
        passed=1,
    )
    payload = run.to_json()
    restored = run_from_json(json.loads(json.dumps(payload)))
    assert restored == run


def test_a_damaged_payload_degrades_to_none() -> None:
    assert run_from_json(None) is None
    assert run_from_json({}) is None
    assert run_from_json({"status": "exploded"}) is None
    assert run_from_json({"status": "not_run"}) is None


# --------------------------------------------------------------------------- end to end


@pytest.fixture
def tested_repo(repo_root: Path) -> Path:
    # The conftest.py at the root is what puts the repository on sys.path when pytest
    # runs the tests — exactly what a real repository the command targets would have.
    (repo_root / "conftest.py").write_text("", encoding="utf-8", newline="\n")
    (repo_root / "app.py").write_text(
        "def value() -> int:\n    return 1\n", encoding="utf-8", newline="\n"
    )
    (repo_root / "tests").mkdir()
    (repo_root / "tests" / "test_app.py").write_text(
        "from app import value\n\n\ndef test_value():\n    assert value() == 1\n",
        encoding="utf-8",
        newline="\n",
    )
    from tests.conftest import commit

    commit(repo_root, "app with a test")
    return repo_root


def test_analyze_records_a_passing_test_run(
    tested_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tested_repo / ".traceflow.toml").write_text(
        '[tests]\nenabled = true\ncommand = "pytest"\n', encoding="utf-8"
    )
    assert main(["analyze", str(tested_repo)]) == EXIT_OK  # baseline
    capsys.readouterr()

    (tested_repo / "app.py").write_text(
        "def value() -> int:\n    return 2\n", encoding="utf-8", newline="\n"
    )
    (tested_repo / "tests" / "test_app.py").write_text(
        "from app import value\n\n\ndef test_value():\n    assert value() == 2\n",
        encoding="utf-8",
        newline="\n",
    )

    assert main(["analyze", str(tested_repo)]) == EXIT_OK
    output = capsys.readouterr().out
    assert "passed" in output

    store = SessionStore(tested_repo)
    session_id = store.list_sessions()[-1].session_id
    payload = store.read_artifact(session_id, "tests.json")
    assert payload is not None
    assert payload["status"] == "passed"


def test_the_delivery_shows_the_run_and_the_concern_on_failure(
    tested_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from traceflow.ui import render

    (tested_repo / ".traceflow.toml").write_text(
        '[tests]\nenabled = true\ncommand = "pytest"\n', encoding="utf-8"
    )
    assert main(["analyze", str(tested_repo)]) == EXIT_OK  # baseline
    capsys.readouterr()

    (tested_repo / "tests" / "test_app.py").write_text(
        "from app import value\n\n\ndef test_value():\n    assert value() == 999\n",
        encoding="utf-8",
        newline="\n",
    )
    assert main(["analyze", str(tested_repo)]) == EXIT_OK
    capsys.readouterr()

    store = SessionStore(tested_repo)
    session_id = store.list_sessions()[-1].session_id
    delivery = load_delivery(store, session_id)

    assert delivery is not None
    assert delivery.test_run is not None
    assert delivery.test_run.run.status is _TestRunStatus.FAILED
    assert any("Tests failed" in item.title for item in delivery.concerns)

    page = render.render_delivery(delivery)
    assert "Tests failed" in page
    # plan.md §23: the run's result, never a verdict on the change.
    assert "Change is safe" not in page


def test_a_disabled_repository_states_the_fact_without_a_run(
    tested_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from traceflow.ui import render

    assert main(["analyze", str(tested_repo)]) == EXIT_OK  # baseline
    (tested_repo / "app.py").write_text(
        "def value() -> int:\n    return 3\n", encoding="utf-8", newline="\n"
    )
    assert main(["analyze", str(tested_repo)]) == EXIT_OK
    capsys.readouterr()

    store = SessionStore(tested_repo)
    session_id = store.list_sessions()[-1].session_id
    delivery = load_delivery(store, session_id)

    assert delivery is not None
    assert delivery.test_run is None
    page = render.render_delivery(delivery)
    assert "No test run is recorded" in page
