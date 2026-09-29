"""Parsing git's diff output and counting lines.

The payloads below are built with an explicit NUL join rather than adjacent string
literals. A formatter that folds adjacent literals together would turn a ``\\0``
terminator followed by a ``1`` into the single octal escape ``\\01``, silently
changing the fixture instead of failing.
"""

from __future__ import annotations

from traceflow.git.diff import (
    ChangeStatus,
    count_line_changes,
    parse_name_status,
    parse_numstat,
)


def _records(*records: str) -> str:
    """Join records the way git emits them under ``-z``: NUL-terminated."""
    return "".join(f"{record}\0" for record in records)


# --------------------------------------------------------------------------- name-status


def test_name_status_reads_a_modification() -> None:
    changes = parse_name_status(_records("M", "app.py"))
    assert len(changes) == 1
    assert changes[0].status is ChangeStatus.MODIFIED
    assert changes[0].path == "app.py"
    assert changes[0].original_path is None


def test_name_status_reads_every_common_status() -> None:
    changes = parse_name_status(
        _records("A", "added.py", "M", "changed.py", "D", "gone.py", "T", "link.py")
    )
    assert [(change.status, change.path) for change in changes] == [
        (ChangeStatus.ADDED, "added.py"),
        (ChangeStatus.MODIFIED, "changed.py"),
        (ChangeStatus.DELETED, "gone.py"),
        (ChangeStatus.TYPE_CHANGED, "link.py"),
    ]


def test_name_status_reads_a_rename_with_its_original_path() -> None:
    """A rename contributes three fields, so a pair-wise walk would desynchronise."""
    changes = parse_name_status(_records("R050", "old.py", "new.py", "M", "other.py"))

    assert len(changes) == 2
    renamed, modified = changes
    assert renamed.status is ChangeStatus.RENAMED
    assert renamed.path == "new.py"
    assert renamed.original_path == "old.py"
    assert modified.path == "other.py"


def test_name_status_reads_a_copy() -> None:
    changes = parse_name_status(_records("C100", "source.py", "duplicate.py"))
    assert changes[0].status is ChangeStatus.COPIED
    assert changes[0].path == "duplicate.py"
    assert changes[0].original_path == "source.py"


def test_name_status_maps_an_unrecognised_code_to_unknown() -> None:
    """A future git status letter must not crash a running watcher."""
    changes = parse_name_status(_records("Z", "mystery.py"))
    assert changes[0].status is ChangeStatus.UNKNOWN
    assert changes[0].path == "mystery.py"


def test_name_status_handles_empty_output() -> None:
    assert parse_name_status("") == []


def test_name_status_keeps_paths_containing_spaces() -> None:
    changes = parse_name_status(_records("M", "my file.py"))
    assert changes[0].path == "my file.py"


# --------------------------------------------------------------------------- numstat


def test_numstat_reads_line_counts() -> None:
    counts = parse_numstat(_records("17\t4\tauth/service.py"))
    assert counts == {"auth/service.py": (17, 4)}


def test_numstat_reports_binary_files_as_having_no_counts() -> None:
    """git writes ``-`` for binary content; a line count there would be a fabrication."""
    counts = parse_numstat(_records("-\t-\tlogo.png"))
    assert counts == {"logo.png": (None, None)}


def test_numstat_keys_a_rename_by_its_new_path() -> None:
    """The path field is empty for a rename; the two paths follow as their own fields."""
    counts = parse_numstat(_records("2\t1\t", "old.py", "new.py", "3\t0\tadded.py"))

    assert counts == {"new.py": (2, 1), "added.py": (3, 0)}


def test_numstat_handles_empty_output() -> None:
    assert parse_numstat("") == {}


def test_numstat_ignores_malformed_records() -> None:
    assert parse_numstat(_records("not a numstat record")) == {}


# --------------------------------------------------------------------------- line counting


def test_identical_content_produces_no_change() -> None:
    assert count_line_changes(b"a\nb\n", b"a\nb\n") == (0, 0)


def test_a_changed_line_counts_as_one_of_each() -> None:
    assert count_line_changes(b"a\nb\nc\n", b"a\nB\nc\n") == (1, 1)


def test_an_inserted_line_counts_once() -> None:
    assert count_line_changes(b"a\nb\n", b"a\nb\nc\n") == (1, 0)


def test_a_deleted_line_counts_once() -> None:
    assert count_line_changes(b"a\nb\nc\n", b"a\nc\n") == (0, 1)


def test_everything_added_to_nothing_is_all_insertions() -> None:
    assert count_line_changes(b"", b"one\ntwo\nthree\n") == (3, 0)


def test_everything_removed_is_all_deletions() -> None:
    assert count_line_changes(b"one\ntwo\n", b"") == (0, 2)


def test_binary_content_has_no_line_count() -> None:
    assert count_line_changes(b"text\n", b"bin\x00ary\n") is None
    assert count_line_changes(b"bin\x00ary\n", b"text\n") is None


def test_a_file_without_a_trailing_newline_still_counts_its_lines() -> None:
    assert count_line_changes(b"", b"one\ntwo") == (2, 0)


def test_whitespace_only_changes_are_counted() -> None:
    assert count_line_changes(b"a\n", b"  a\n") == (1, 1)
