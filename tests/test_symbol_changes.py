"""Symbol-level change detection (plan.md §16, §59).

The tests that matter most are the attribution ones. Comparing symbols against the last
commit instead of against the session's baseline silently credits the session with
whatever was already edited — the same mistake plan.md §14 forbids at file level, one
layer down.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import run_git
from traceflow.analysis.symbols import (
    SymbolReport,
    baseline_source,
    collect_symbol_changes,
)
from traceflow.blobs import BlobStore
from traceflow.config import STATE_DIRNAME, Config
from traceflow.derived import AnalysisCache
from traceflow.git.baseline import Baseline, capture_baseline
from traceflow.git.diff import collect_changes
from traceflow.git.repository import Repository
from traceflow.languages.base import (
    SymbolChangeKind,
    diff_module_analysis,
)
from traceflow.languages.python.analyzer import analyze_python


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


@pytest.fixture
def blobs(repo_root: Path) -> BlobStore:
    return BlobStore(repo_root / STATE_DIRNAME)


@pytest.fixture
def cache(repo_root: Path) -> AnalysisCache:
    return AnalysisCache(repo_root / STATE_DIRNAME)


def snapshot(repo: Repository, blobs: BlobStore, config: Config) -> Baseline:
    return capture_baseline(repo, blobs, repo.working_tree_state(config.ignore), config)


def report(
    repo: Repository, baseline: Baseline, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> SymbolReport:
    changes = collect_changes(repo, baseline, repo.working_tree_state(config.ignore), blobs, config)
    return collect_symbol_changes(repo, baseline, changes, blobs, cache, config)


# --------------------------------------------------------------------------- the diff itself


def test_everything_is_added_when_there_is_no_before() -> None:
    after = analyze_python("m.py", b"def f():\n    return 1\n")
    changes = diff_module_analysis(None, after)

    assert [item.change for item in changes.changes] == [SymbolChangeKind.ADDED]
    assert changes.imports_added == ()


def test_an_unchanged_module_produces_no_changes() -> None:
    before = analyze_python("m.py", b"def f():\n    return 1\n")
    after = analyze_python("m.py", b"def f():\n    return 1\n")

    assert diff_module_analysis(before, after).has_changes is False


def test_a_signature_change_is_reported_as_such() -> None:
    before = analyze_python("m.py", b"def f(a):\n    return a\n")
    after = analyze_python("m.py", b"def f(a, b):\n    return a\n")

    changes = diff_module_analysis(before, after)

    assert [item.change for item in changes.changes] == [SymbolChangeKind.SIGNATURE_CHANGED]
    assert changes.signature_changes[0].qualified_name == "f"


def test_a_body_only_change_is_reported_as_such() -> None:
    before = analyze_python("m.py", b"def f(a):\n    return a\n")
    after = analyze_python("m.py", b"def f(a):\n    return a + 1\n")

    changes = diff_module_analysis(before, after)

    assert [item.change for item in changes.changes] == [SymbolChangeKind.BODY_CHANGED]


def test_a_signature_change_wins_over_a_body_change() -> None:
    """Both changed; only the fact that forces callers to be re-examined is reported."""
    before = analyze_python("m.py", b"def f(a):\n    return a\n")
    after = analyze_python("m.py", b"def f(a, b):\n    return b\n")

    changes = diff_module_analysis(before, after)

    assert [item.change for item in changes.changes] == [SymbolChangeKind.SIGNATURE_CHANGED]


def test_a_removed_symbol_is_reported() -> None:
    before = analyze_python("m.py", b"def f():\n    return 1\n\ndef g():\n    return 2\n")
    after = analyze_python("m.py", b"def f():\n    return 1\n")

    changes = diff_module_analysis(before, after)

    assert [item.qualified_name for item in changes.changes] == ["g"]
    assert changes.changes[0].change is SymbolChangeKind.REMOVED


def test_an_added_symbol_is_reported() -> None:
    before = analyze_python("m.py", b"def f():\n    return 1\n")
    after = analyze_python("m.py", b"def f():\n    return 1\n\ndef g():\n    return 2\n")

    changes = diff_module_analysis(before, after)

    assert [(item.qualified_name, item.change) for item in changes.changes] == [
        ("g", SymbolChangeKind.ADDED)
    ]


def test_import_changes_are_reported() -> None:
    before = analyze_python("m.py", b"import os\n")
    after = analyze_python("m.py", b"import sys\n")

    changes = diff_module_analysis(before, after)

    assert changes.imports_added == ("sys",)
    assert changes.imports_removed == ("os",)


def test_reformatting_a_module_reports_nothing() -> None:
    """The payoff of fingerprinting the tree rather than the text."""
    before = analyze_python("m.py", b"def f(a):\n    return a\n")
    after = analyze_python("m.py", b"def f(\n    a,\n):\n\n    # a note\n    return a\n")

    assert diff_module_analysis(before, after).has_changes is False


def test_a_parse_error_is_carried_into_the_result() -> None:
    before = analyze_python("m.py", b"def f():\n    return 1\n")
    after = analyze_python("m.py", b"def f(:\n")

    changes = diff_module_analysis(before, after)

    assert changes.parse_error is not None


# --------------------------------------------------------------------------- attribution


def test_a_modified_python_file_is_analysed(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    baseline = snapshot(repo, blobs, config)
    _write(repo_root / "app.py", "def main() -> int:\n    return 2\n")

    result = report(repo, baseline, blobs, cache, config)

    assert len(result.body_changes) == 1
    assert result.body_changes[0].qualified_name == "main"


def test_a_new_python_file_reports_all_of_its_symbols_as_added(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    baseline = snapshot(repo, blobs, config)
    _write(repo_root / "fresh.py", "def one():\n    return 1\n\n\ndef two():\n    return 2\n")

    result = report(repo, baseline, blobs, cache, config)

    assert sorted(item.qualified_name for item in result.added) == ["one", "two"]


def test_a_deleted_python_file_reports_all_of_its_symbols_as_removed(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    baseline = snapshot(repo, blobs, config)
    (repo_root / "app.py").unlink()

    result = report(repo, baseline, blobs, cache, config)

    assert [item.qualified_name for item in result.removed] == ["main"]


def test_a_non_python_file_is_not_analysed(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    baseline = snapshot(repo, blobs, config)
    _write(repo_root / "README.md", "# changed\n")

    result = report(repo, baseline, blobs, cache, config)

    assert result.modules == ()


def test_a_pre_existing_signature_change_is_not_attributed_to_the_session(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """The case that separates this from a naive implementation.

    ``main`` already gained a parameter before the session began. Comparing against the
    commit would report that as this session's signature change; comparing against the
    baseline snapshot correctly reports only the body edit the session actually made.
    """
    _write(repo_root / "app.py", "def main(a, b):\n    return a\n")
    baseline = snapshot(repo, blobs, config)

    _write(repo_root / "app.py", "def main(a, b):\n    return b\n")

    result = report(repo, baseline, blobs, cache, config)

    assert result.signature_changes == (), "the earlier edit is not this session's doing"
    assert [item.qualified_name for item in result.body_changes] == ["main"]


def test_a_wholly_pre_existing_symbol_change_is_not_reported(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    _write(repo_root / "app.py", "def main(a, b):\n    return a\n")
    baseline = snapshot(repo, blobs, config)

    # The session touches a different file; app.py is left exactly as it was.
    _write(repo_root / "other.py", "def other():\n    return 1\n")

    result = report(repo, baseline, blobs, cache, config)

    assert "app.py" not in {module.path for module in result.modules}


def test_a_dirty_file_compared_against_the_commit_would_be_wrong(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """Demonstrates the difference directly, so the previous test cannot pass by accident.

    The same two versions of ``app.py`` compared against the *committed* content yield a
    signature change; compared against the session's baseline they yield a body change.
    """
    _write(repo_root / "app.py", "def main(a, b):\n    return a\n")
    baseline = snapshot(repo, blobs, config)
    _write(repo_root / "app.py", "def main(a, b):\n    return b\n")

    committed = analyze_python("app.py", b"def main(a) -> int:\n    return 1\n")
    current = analyze_python("app.py", b"def main(a, b):\n    return b\n")
    naive = diff_module_analysis(committed, current)

    assert naive.signature_changes, "against the commit this looks like a signature change"
    assert report(repo, baseline, blobs, cache, config).signature_changes == ()


def test_the_report_records_which_analyzer_produced_it(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """A session artifact should state the basis of its own conclusions."""
    baseline = snapshot(repo, blobs, config)
    _write(repo_root / "app.py", "def main() -> int:\n    return 2\n")

    assert report(repo, baseline, blobs, cache, config).analyzer == "python-1"


def test_the_report_round_trips_through_json(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    baseline = snapshot(repo, blobs, config)
    _write(repo_root / "app.py", "def main(extra) -> int:\n    return 2\n")

    payload = report(repo, baseline, blobs, cache, config).to_json()

    assert payload["analyzer"] == "python-1"
    totals = payload["totals"]
    assert isinstance(totals, dict)
    assert totals["signature_changes"] == 1
    assert totals["modules"] == 1


def test_a_broken_file_is_reported_and_does_not_stop_the_analysis(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    baseline = snapshot(repo, blobs, config)
    _write(repo_root / "broken.py", "def f(:\n")
    _write(repo_root / "app.py", "def main() -> int:\n    return 2\n")

    result = report(repo, baseline, blobs, cache, config)

    assert any("broken.py" in error for error in result.parse_errors)
    assert [item.qualified_name for item in result.body_changes] == ["main"]


def test_the_analysis_is_cached(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    baseline = snapshot(repo, blobs, config)
    _write(repo_root / "app.py", "def main() -> int:\n    return 2\n")

    report(repo, baseline, blobs, cache, config)

    assert [path for path in cache.root.rglob("*.json") if path.is_file()]


def test_a_withheld_file_is_not_recovered_from_git(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    """plan.md §26 says its contents are never read — including through a second door.

    The baseline records *why* it stored nothing for a sensitive path. A later caller
    asking for that file's pre-session content must honour that record rather than falling
    back to git, which would read exactly what the policy said not to.
    """
    _write(repo_root / ".env", "API_KEY=do-not-read-me\n")
    run_git(repo_root, "add", "-f", ".env")
    run_git(repo_root, "commit", "-q", "-m", "track env")
    # It has to be *dirty* for the baseline to capture it at all, which is the only case
    # where a snapshot would otherwise stand in for a committed version.
    _write(repo_root / ".env", "API_KEY=changed-since\n")
    baseline = snapshot(repo, blobs, config)

    assert baseline.was_withheld(".env") is True
    assert baseline_source(repo, baseline, ".env", blobs) is None


def test_an_ordinary_file_is_still_recovered_from_git(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    """The withheld check must not stop a clean file being recovered the ordinary way."""
    baseline = snapshot(repo, blobs, config)

    assert baseline.was_withheld("app.py") is False
    assert baseline_source(repo, baseline, "app.py", blobs) is not None
