"""Repository discovery, porcelain parsing, and activity fingerprinting."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.conftest import run_git
from traceflow.config import DEFAULT_IGNORE
from traceflow.git.repository import (
    Repository,
    is_ignored_path,
    parse_porcelain_v2,
)

# --------------------------------------------------------------------------- discovery


def test_discover_finds_repository_from_root(repo_root: Path) -> None:
    repository = Repository.discover(repo_root)
    assert repository is not None
    assert repository.root == repo_root.resolve()
    assert repository.name == "sample-repo"


def test_discover_finds_repository_from_nested_directory(repo_root: Path) -> None:
    nested = repo_root / "src" / "package"
    nested.mkdir(parents=True)

    repository = Repository.discover(nested)
    assert repository is not None
    assert repository.root == repo_root.resolve()


def test_discover_returns_none_outside_a_repository(tmp_path: Path) -> None:
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    assert Repository.discover(plain) is None


def test_discover_returns_none_for_missing_path(tmp_path: Path) -> None:
    assert Repository.discover(tmp_path / "does-not-exist") is None


def test_head_commit_returns_the_committed_revision(repo: Repository) -> None:
    head = repo.head_commit()
    assert head is not None
    assert len(head) == 40


def test_head_commit_is_none_without_commits(tmp_path: Path) -> None:
    empty = tmp_path / "empty-repo"
    empty.mkdir()
    run_git(empty, "init", "-q")

    repository = Repository.discover(empty)
    assert repository is not None
    assert repository.head_commit() is None


# --------------------------------------------------------------------------- fingerprinting


def test_clean_repository_reports_clean(repo: Repository) -> None:
    state = repo.working_tree_state()
    assert state.is_clean
    assert state.tracked_change_count == 0
    assert state.untracked_count == 0


def test_token_is_stable_when_nothing_changes(repo: Repository) -> None:
    assert repo.working_tree_state().token == repo.working_tree_state().token


def test_modifying_a_tracked_file_changes_the_token(repo: Repository) -> None:
    before = repo.working_tree_state().token

    (repo.root / "app.py").write_text("def main() -> int:\n    return 2\n", encoding="utf-8")

    after = repo.working_tree_state()
    assert after.token != before
    assert after.tracked_change_count == 1


def test_new_untracked_file_changes_the_token(repo: Repository) -> None:
    before = repo.working_tree_state().token

    (repo.root / "extra.py").write_text("x = 1\n", encoding="utf-8")

    after = repo.working_tree_state()
    assert after.token != before
    assert after.untracked_count == 1


def test_modifying_an_already_modified_file_changes_the_token(repo: Repository) -> None:
    """git's porcelain output describes *which* files differ, not *how much*.

    Without a content-sensitive component in the fingerprint, editing a file that is
    already dirty leaves the status output byte for byte identical and the activity is
    never seen — which is exactly the case the baseline exists to handle.
    """
    (repo.root / "app.py").write_text("def main() -> int:\n    return 222\n", encoding="utf-8")
    before = repo.working_tree_state().token

    (repo.root / "app.py").write_text("def main() -> int:\n    return 3333\n", encoding="utf-8")

    assert repo.working_tree_state().token != before


def test_modifying_an_untracked_file_changes_the_token(repo: Repository) -> None:
    """Untracked files never appear in a diff, so only the stat signature can see them."""
    (repo.root / "scratch.py").write_text("one\n", encoding="utf-8")
    before = repo.working_tree_state().token

    (repo.root / "scratch.py").write_text("one\ntwo\n", encoding="utf-8")

    assert repo.working_tree_state().token != before


def test_a_size_preserving_edit_is_still_detected(repo: Repository) -> None:
    """The modification time catches an edit that leaves the size unchanged."""
    path = repo.root / "app.py"
    path.write_text("def main() -> int:\n    return 222\n", encoding="utf-8")
    before = repo.working_tree_state().token

    path.write_text("def main() -> int:\n    return 333\n", encoding="utf-8")
    # Set explicitly rather than relying on the clock ticking between two writes.
    os.utime(path, ns=(1_000_000_000, 2_000_000_000))

    assert repo.working_tree_state().token != before


def test_second_file_inside_a_new_directory_still_registers(repo: Repository) -> None:
    """The reason --untracked-files=all is used instead of the cheaper default.

    With ``normal``, a new directory is reported as a single entry, so adding a
    second file inside it leaves the output unchanged and the activity is missed —
    exactly when an agent is scaffolding new files.
    """
    package = repo.root / "package"
    package.mkdir()
    (package / "first.py").write_text("a = 1\n", encoding="utf-8")
    first_token = repo.working_tree_state().token

    (package / "second.py").write_text("b = 2\n", encoding="utf-8")
    second_token = repo.working_tree_state().token

    assert second_token != first_token


def test_committing_registers_as_activity(repo: Repository) -> None:
    """HEAD is part of the fingerprint, so a commit counts even if the tree was clean."""
    before = repo.working_tree_state().token

    (repo.root / "app.py").write_text("def main() -> int:\n    return 3\n", encoding="utf-8")
    run_git(repo.root, "commit", "-q", "-am", "change")

    assert repo.working_tree_state().token != before


def test_ignored_paths_are_excluded_from_the_token(repo: Repository) -> None:
    """TraceFlow's own state must never register as activity.

    Without this, writing a session record changes the working tree, which looks
    like activity, which starts another session — an unbounded loop.
    """
    before = repo.working_tree_state(ignore=(".traceflow",)).token

    state_dir = repo.root / ".traceflow" / "sessions"
    state_dir.mkdir(parents=True)
    (state_dir / "session.json").write_text("{}\n", encoding="utf-8")

    after = repo.working_tree_state(ignore=(".traceflow",))
    assert after.token == before
    assert after.is_clean


def test_state_directory_registers_when_not_ignored(repo: Repository) -> None:
    """Confirms the previous test is meaningful: without the filter it would show up."""
    before = repo.working_tree_state().token

    state_dir = repo.root / ".traceflow"
    state_dir.mkdir()
    (state_dir / "events.jsonl").write_text("{}\n", encoding="utf-8")

    assert repo.working_tree_state().token != before


def test_is_git_busy_detects_the_index_lock(repo: Repository) -> None:
    assert repo.is_git_busy() is False

    lock = repo.git_dir / "index.lock"
    lock.write_text("", encoding="utf-8")
    try:
        assert repo.is_git_busy() is True
    finally:
        lock.unlink()


# --------------------------------------------------------------------------- ignore matching


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (".traceflow", True),
        (".traceflow/", True),
        (".traceflow/events.jsonl", True),
        (".traceflow\\events.jsonl", True),
        ("src/.traceflow/file.py", False),
        ("traceflow.py", False),
        ("src/traceflow.py", False),
    ],
)
def test_is_ignored_path(path: str, expected: bool) -> None:
    assert is_ignored_path(path, (".traceflow",)) is expected


def test_is_ignored_path_with_empty_ignore_list() -> None:
    assert is_ignored_path(".traceflow/events.jsonl", ()) is False


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("ProU Assessment.docx", True),
        ("docs/report.pdf", True),
        ("assets/logo.PNG", True),  # suffix matching is case-insensitive
        ("deep/nested/photo.jpeg", True),
        ("bundle.Zip", True),
        ("notes.txt", False),
        ("report.pdfx", False),  # a suffix is not a substring
        ("src/main.ts", False),
        ("docx", False),  # the glob needs the dot
    ],
)
def test_is_ignored_path_matches_suffix_globs(path: str, expected: bool) -> None:
    assert is_ignored_path(path, ("*.docx", "*.pdf", "*.png", "*.jpeg", "*.zip")) is expected


def test_default_ignore_covers_the_binary_noise_a_session_cannot_explain() -> None:
    """The kinds the ProU run surfaced — .docx/.pdf — are ignored out of the box."""
    for path in ("assessment.docx", "spec.PDF", "icon.ico", "font.woff2", ".traceflow/x"):
        assert is_ignored_path(path, DEFAULT_IGNORE) is True
    # Source files are never ignored by default, whatever their name looks like.
    for path in ("main.py", "index.ts", "TaskForm.tsx", "report.md"):
        assert is_ignored_path(path, DEFAULT_IGNORE) is False


# --------------------------------------------------------------------------- porcelain parsing


def _records(*records: str) -> str:
    """Join porcelain v2 records the way git emits them: NUL-terminated.

    Built with an explicit join rather than adjacent string literals. A formatter
    that folds adjacent literals together would turn a ``\\0`` terminator followed
    by a ``1`` record into the single octal escape ``\\01`` — silently changing the
    payload rather than failing loudly.
    """
    return "".join(f"{record}\0" for record in records)


def test_parse_porcelain_v2_reads_the_branch_header() -> None:
    parsed = parse_porcelain_v2(_records("# branch.oid abc123", "# branch.head main"))
    assert parsed.branch_oid == "abc123"
    assert parsed.entries == ()


def test_parse_porcelain_v2_treats_initial_branch_as_no_commit() -> None:
    assert parse_porcelain_v2(_records("# branch.oid (initial)")).branch_oid is None


def test_parse_porcelain_v2_handles_changed_and_untracked() -> None:
    parsed = parse_porcelain_v2(
        _records(
            "# branch.oid deadbeef",
            "1 .M N... 100644 100644 100644 aaaa bbbb app.py",
            "? notes.txt",
        )
    )

    assert parsed.branch_oid == "deadbeef"
    kinds = {(entry.kind, entry.path) for entry in parsed.entries}
    assert kinds == {("changed", "app.py"), ("untracked", "notes.txt")}


def test_parse_porcelain_v2_consumes_the_extra_rename_field() -> None:
    """Rename records carry the original path as a second NUL-separated field."""
    parsed = parse_porcelain_v2(
        _records(
            "2 R. N... 100644 100644 100644 aaaa bbbb R100 new.py",
            "old.py",
            "? after.txt",
        )
    )

    assert len(parsed.entries) == 2
    renamed, untracked = parsed.entries
    assert renamed.kind == "renamed"
    assert renamed.path == "new.py"
    assert renamed.original_path == "old.py"
    assert untracked.path == "after.txt"


def test_parse_porcelain_v2_marks_tracked_changes() -> None:
    parsed = parse_porcelain_v2(
        _records(
            "1 .M N... 100644 100644 100644 aaaa bbbb app.py",
            "? notes.txt",
        )
    )
    changed, untracked = parsed.entries
    assert changed.is_tracked_change is True
    assert untracked.is_tracked_change is False


def test_parse_porcelain_v2_keeps_paths_containing_spaces() -> None:
    parsed = parse_porcelain_v2(_records("1 .M N... 100644 100644 100644 aaaa bbbb my file.py"))
    assert parsed.entries[0].path == "my file.py"


def test_parse_porcelain_v2_handles_unmerged_records() -> None:
    parsed = parse_porcelain_v2(
        _records("u UU N... 100644 100644 100644 100644 aaaa bbbb cccc conflict.py")
    )
    assert parsed.entries[0].kind == "unmerged"
    assert parsed.entries[0].path == "conflict.py"


def test_parse_porcelain_v2_ignores_unknown_record_types() -> None:
    """A future git version adding a record type must not crash a running watcher."""
    parsed = parse_porcelain_v2(_records("? known.txt", "Z something unexpected"))
    assert [entry.path for entry in parsed.entries] == ["known.txt"]


def test_parse_porcelain_v2_handles_empty_output() -> None:
    parsed = parse_porcelain_v2("")
    assert parsed.branch_oid is None
    assert parsed.entries == ()
