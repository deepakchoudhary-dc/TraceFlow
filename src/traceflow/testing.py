"""Configurable test execution (plan.md §22, §23, §64).

Test running is the first thing TraceFlow does that *executes* rather than reads, so
this module is built around three safety rules the plan states outright:

**Nothing runs unless the repository's configuration says so.** plan.md §22: "Do NOT
automatically run an entire massive production test suite without user configuration.
Running tests can be expensive or destructive." The ``[tests]`` section defaults to
disabled, and even when it is enabled, this module runs only the tests a session's own
impact analysis reaches — a handful of files, never the whole suite. plan.md §64 adds
"do not execute arbitrary commands automatically"; this module never runs anything the
repository's own configuration file did not name.

**The configured command is a program and arguments, not a script.** It is read from
configuration only — never from a request, an artifact, or anywhere the repository's
contents could put words in — split with :mod:`shlex`, and executed without a shell.
Test paths are appended after a ``--`` separator, so a filename beginning with ``-``
could start an option even if one ever reached this far. A run is bounded by
``timeout_seconds`` and stopped when it exceeds it.

**A result is a fact about a run, not a verdict on the change.** plan.md §23: the
delivery says "Tests passed", never "Change is safe". When tests do not run, the
record says why — disabled by configuration, no tests reached, or the configured
command missing — rather than pretending to a result it does not have (plan.md §69).
"""

from __future__ import annotations

import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from enum import Enum

from traceflow.analysis.models import is_test_path
from traceflow.config import Config
from traceflow.git.repository import Repository

#: How much captured output is kept. The artifact holds the tail — the summary is
#: what a reader wants and what pytest writes last — and the note says so when the
#: head was cut. A hundred kilobytes of pytest output is not evidence anyone reads.
_MAX_OUTPUT_CHARS = 20_000


class TestRunStatus(str, Enum):
    """What happened when TraceFlow tried to run tests (plan.md §23)."""

    PASSED = "passed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    ERROR = "error"
    """The configured command could not be run at all — missing executable, and the like."""
    NOT_RUN = "not_run"
    """Tests were enabled but none of the session's changes reached any."""


@dataclass(frozen=True)
class TestRun:
    """The evidence of one attempt to run the tests a session reaches."""

    status: TestRunStatus
    command: tuple[str, ...] = ()
    """The exact program and arguments, as run. No shell joined them."""

    exit_code: int | None = None
    duration_seconds: float | None = None
    output_tail: str | None = None
    output_truncated: bool = False
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    note: str = ""
    """Why there is no result, when there is no result (plan.md §69)."""

    @property
    def ran(self) -> bool:
        """True when tests actually executed — not a skip, a refusal, or a timeout."""
        return self.status in (TestRunStatus.PASSED, TestRunStatus.FAILED)

    def to_json(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "command": list(self.command),
            "exit_code": self.exit_code,
            "duration_seconds": self.duration_seconds,
            "output_tail": self.output_tail,
            "output_truncated": self.output_truncated,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "note": self.note,
        }


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def test_run_from_json(payload: dict[str, object] | None) -> TestRun | None:
    """Rebuild a stored run, or ``None`` when there is none to read.

    A damaged artifact degrades to "no test record" rather than a broken delivery,
    the same rule every other artifact follows (plan.md §46). A run that never
    happened is also ``None``: a status of not-run with no command behind it would
    be a record of nothing, presented as though it were evidence.
    """
    if payload is None:
        return None
    raw_status = payload.get("status")
    if not isinstance(raw_status, str):
        return None
    try:
        status = TestRunStatus(raw_status)
    except ValueError:
        return None
    if status is TestRunStatus.NOT_RUN and not payload.get("note"):
        return None

    raw_command = payload.get("command")
    command = tuple(
        part
        for part in (raw_command if isinstance(raw_command, list) else ())
        if isinstance(part, str)
    )

    return TestRun(
        status=status,
        command=command,
        exit_code=_optional_int(payload.get("exit_code")),
        duration_seconds=_optional_float(payload.get("duration_seconds")),
        output_tail=_optional_text(payload.get("output_tail")),
        output_truncated=bool(payload.get("output_truncated", False)),
        passed=_optional_int(payload.get("passed")) or 0,
        failed=_optional_int(payload.get("failed")) or 0,
        skipped=_optional_int(payload.get("skipped")) or 0,
        note=_optional_text(payload.get("note")) or "",
    )


def test_paths_from(
    impacted_paths: tuple[str, ...],
    changed_paths: tuple[str, ...],
) -> tuple[str, ...]:
    """The test files a session reaches: tests it changed, plus tests it affects.

    A test the session *changed* is always included — its own contents changed, which
    makes it the first thing to run. A test the walk *reached* is included because the
    change runs through it. Deterministic order, duplicates collapsed: a test both
    changed and affected is one file, not an argument twice.
    """
    seen: dict[str, None] = {}
    for path in (*changed_paths, *impacted_paths):
        normalised = path.replace("\\", "/").strip("/")
        if normalised and normalised not in seen and is_test_path(normalised):
            seen[normalised] = None
    return tuple(seen)


_SUMMARY_WORD = re.compile(r"(\d+) (passed|failed|errors?|skipped)")


def _parse_pytest_summary(output: str) -> tuple[int, int, int]:
    """The final counts line of a pytest run: passed, failed, skipped.

    Returns zeros when the counts cannot be read — an unknown runner, a crash before
    the summary. Zeros are the honest default here: the run's *status* comes from the
    exit code, not from these counts, and the artifact records the output tail, so
    nothing is silently claimed either way.
    """
    for line in reversed(output.splitlines()):
        if "=" not in line:
            continue
        found = _SUMMARY_WORD.findall(line)
        if not found:
            continue
        passed = failed = skipped = 0
        for count, word in found:
            number = int(count)
            if word == "passed":
                passed += number
            elif word == "skipped":
                skipped += number
            else:
                # "failed" and "errors" share a bucket: either is a red result.
                failed += number
        return passed, failed, skipped
    return 0, 0, 0


def run_session_tests(
    repository: Repository,
    config: Config,
    impacted_paths: tuple[str, ...],
    changed_paths: tuple[str, ...],
) -> TestRun | None:
    """Run the tests *impacted_paths* reaches, when the repository's config allows it.

    Returns ``None`` when test execution is disabled (the default) — the caller then
    knows there is deliberately no record, which the delivery states in the Tests
    section rather than inventing a result for.
    """
    if not config.tests.enabled:
        return None

    paths = test_paths_from(impacted_paths, changed_paths)
    if not paths:
        return TestRun(
            status=TestRunStatus.NOT_RUN,
            note="No test file is among the files this session changed or reached.",
        )

    try:
        base = shlex.split(config.tests.command)
    except ValueError as exc:
        return TestRun(
            status=TestRunStatus.ERROR,
            note=f"The configured tests.command could not be parsed: {exc}",
        )
    if not base:
        return TestRun(
            status=TestRunStatus.ERROR,
            note="The configured tests.command is empty.",
        )

    command = (*base, "--", *paths)
    started_at_mono = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=str(repository.root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=config.tests.timeout_seconds,
            check=False,
            shell=False,
        )
    except FileNotFoundError:
        return TestRun(
            status=TestRunStatus.ERROR,
            command=command,
            note=f"'{base[0]}' is not on PATH. Fix tests.command in the configuration.",
        )
    except subprocess.TimeoutExpired:
        return TestRun(
            status=TestRunStatus.TIMED_OUT,
            command=command,
            duration_seconds=config.tests.timeout_seconds,
            note=f"The run was stopped after {config.tests.timeout_seconds:g}s. "
            "Raise tests.timeout_seconds if it needs longer.",
        )
    except OSError as exc:
        return TestRun(
            status=TestRunStatus.ERROR,
            command=command,
            note=f"The test command could not be run: {exc}",
        )

    duration = time.monotonic() - started_at_mono
    output = (completed.stdout or "") + (completed.stderr or "")
    truncated = len(output) > _MAX_OUTPUT_CHARS
    tail = output[-_MAX_OUTPUT_CHARS:] if truncated else output
    passed, failed, skipped = _parse_pytest_summary(output)

    # The exit code is the fact; the counts are supporting detail read from the
    # output. When they disagree, the exit code wins and the tail is on the record.
    status = TestRunStatus.FAILED if completed.returncode != 0 else TestRunStatus.PASSED
    return TestRun(
        status=status,
        command=command,
        exit_code=completed.returncode,
        duration_seconds=duration,
        output_tail=tail,
        output_truncated=truncated,
        passed=passed,
        failed=failed,
        skipped=skipped,
    )
