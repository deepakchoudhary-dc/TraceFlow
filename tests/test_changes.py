"""Baseline capture and change collection (plan.md §14, §15).

The tests that matter most here are the attribution ones. A change set is only
useful if "what this session did" and "what was already lying around" never get
mixed up, so those cases are pinned precisely rather than approximately.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import run_git
from traceflow.blobs import BlobStore
from traceflow.config import STATE_DIRNAME, Config, default_config
from traceflow.git.baseline import Baseline, capture_baseline
from traceflow.git.diff import ChangeSet, ChangeStatus, FileChange, collect_changes
from traceflow.git.repository import Repository


@pytest.fixture
def blobs(repo_root: Path) -> BlobStore:
    return BlobStore(repo_root / STATE_DIRNAME)


def snapshot(repo: Repository, blobs: BlobStore, config: Config) -> Baseline:
    return capture_baseline(repo, blobs, repo.working_tree_state(config.ignore), config)


def collect(repo: Repository, baseline: Baseline, blobs: BlobStore, config: Config) -> ChangeSet:
    return collect_changes(repo, baseline, repo.working_tree_state(config.ignore), blobs, config)


def find(change_set: ChangeSet, path: str) -> FileChange | None:
    for change in change_set.files:
        if change.path == path:
            return change
    return None


def write(root: Path, name: str, text: str) -> None:
    """Write a fixture file with explicit LF endings so counts are platform-independent."""
    (root / name).write_text(text, encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- baseline


def test_a_clean_baseline_captures_nothing(
    repo: Repository, blobs: BlobStore, config: Config
) -> None:
    baseline = snapshot(repo, blobs, config)

    assert baseline.dirty is False
    assert baseline.tracked_changes == 0
    assert baseline.untracked_files == 0
    assert baseline.captured == ()
    assert baseline.commit is not None
    assert baseline.base_revision == baseline.commit


def test_a_modified_tracked_file_is_captured(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    write(repo_root, "app.py", "def main() -> int:\n    return 99\n")

    baseline = snapshot(repo, blobs, config)

    assert baseline.dirty is True
    assert baseline.tracked_changes == 1
    captured = baseline.captured_digests()
    assert "app.py" in captured
    assert blobs.get(captured["app.py"]) == b"def main() -> int:\n    return 99\n"
    assert baseline.was_tracked("app.py") is True


def test_an_untracked_file_is_captured(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    write(repo_root, "scratch.py", "draft = True\n")

    baseline = snapshot(repo, blobs, config)

    assert baseline.untracked_files == 1
    assert baseline.was_tracked("scratch.py") is False
    assert blobs.get(baseline.captured_digests()["scratch.py"]) == b"draft = True\n"


def test_baseline_round_trips_through_json(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    write(repo_root, "scratch.py", "draft = True\n")
    payload = snapshot(repo, blobs, config).to_json()

    assert isinstance(payload["captured"], list)
    assert payload["captured"][0]["path"] == "scratch.py"  # type: ignore[index]


# --------------------------------------------------------------------------- session changes


def test_a_clean_session_reports_nothing(
    repo: Repository, blobs: BlobStore, config: Config
) -> None:
    baseline = snapshot(repo, blobs, config)
    changes = collect(repo, baseline, blobs, config)

    assert changes.files == ()
    assert changes.pre_existing == ()
    assert changes.insertions == 0
    assert changes.deletions == 0


def test_a_modified_tracked_file_is_reported(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    baseline = snapshot(repo, blobs, config)
    write(repo_root, "app.py", "def main() -> int:\n    return 2\n")

    changes = collect(repo, baseline, blobs, config)
    change = find(changes, "app.py")

    assert change is not None
    assert change.status is ChangeStatus.MODIFIED
    assert change.insertions == 1
    assert change.deletions == 1
    assert change.changed_lines == 2
    assert change.tracked is True


def test_a_deleted_file_is_reported(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    baseline = snapshot(repo, blobs, config)
    (repo_root / "app.py").unlink()

    change = find(collect(repo, baseline, blobs, config), "app.py")

    assert change is not None
    assert change.status is ChangeStatus.DELETED
    assert change.deletions == 2


def test_a_renamed_file_keeps_its_original_path(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    baseline = snapshot(repo, blobs, config)
    run_git(repo_root, "mv", "README.md", "GUIDE.md")

    changes = collect(repo, baseline, blobs, config)
    renamed = next(
        (change for change in changes.files if change.status is ChangeStatus.RENAMED), None
    )

    assert renamed is not None
    assert renamed.path == "GUIDE.md"
    assert renamed.original_path == "README.md"


def test_a_staged_new_file_is_reported_as_added(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    baseline = snapshot(repo, blobs, config)
    write(repo_root, "feature.py", "one\ntwo\n")
    run_git(repo_root, "add", "feature.py")

    change = find(collect(repo, baseline, blobs, config), "feature.py")

    assert change is not None
    assert change.status is ChangeStatus.ADDED
    assert change.insertions == 2
    assert change.deletions == 0


def test_a_new_untracked_file_is_reported_as_added(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    baseline = snapshot(repo, blobs, config)
    write(repo_root, "fresh.py", "alpha\nbeta\ngamma\n")

    change = find(collect(repo, baseline, blobs, config), "fresh.py")

    assert change is not None
    assert change.status is ChangeStatus.ADDED
    assert change.insertions == 3
    assert change.tracked is False


def test_an_untracked_file_modified_during_the_session_uses_the_snapshot(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    """The case git cannot help with: no committed version exists to diff against."""
    write(repo_root, "scratch.py", "one\ntwo\nthree\n")
    baseline = snapshot(repo, blobs, config)

    write(repo_root, "scratch.py", "one\nTWO\nthree\nfour\n")

    change = find(collect(repo, baseline, blobs, config), "scratch.py")

    assert change is not None
    assert change.status is ChangeStatus.MODIFIED
    assert change.insertions == 2
    assert change.deletions == 1
    assert change.tracked is False


def test_an_untracked_file_left_alone_is_not_reported(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    write(repo_root, "scratch.py", "unchanged\n")
    baseline = snapshot(repo, blobs, config)

    assert find(collect(repo, baseline, blobs, config), "scratch.py") is None


def test_an_untracked_file_deleted_during_the_session_is_reported(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    """git never knew about the file, so its disappearance is only visible against the snapshot."""
    write(repo_root, "scratch.py", "temporary\n")
    baseline = snapshot(repo, blobs, config)

    (repo_root / "scratch.py").unlink()

    change = find(collect(repo, baseline, blobs, config), "scratch.py")

    assert change is not None
    assert change.status is ChangeStatus.DELETED
    assert change.tracked is False


# --------------------------------------------------------------------------- attribution


def test_a_pre_existing_change_is_excluded_from_the_session(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    """plan.md §14: what was already modified is not this session's doing."""
    write(repo_root, "app.py", "def main() -> int:\n    return 1\n")
    baseline = snapshot(repo, blobs, config)

    changes = collect(repo, baseline, blobs, config)

    assert changes.files == ()
    assert changes.pre_existing_count == 0  # nothing was captured as dirty


def test_a_pre_existing_change_is_reported_separately(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    write(repo_root, "app.py", "def main() -> int:\n    return 77\n")
    baseline = snapshot(repo, blobs, config)

    changes = collect(repo, baseline, blobs, config)

    assert changes.files == ()
    assert [change.path for change in changes.pre_existing] == ["app.py"]
    assert changes.pre_existing[0].insertions == 1
    assert changes.pre_existing[0].deletions == 1


def test_a_session_change_to_a_dirty_file_is_measured_from_the_snapshot(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    """The precise case §14 exists for.

    ``app.py`` was already modified when the session began, so the session's own
    contribution is the difference between the snapshot and the final content — not
    the difference between the commit and the final content.
    """
    write(repo_root, "app.py", "def main() -> int:\n    return 77\n")
    baseline = snapshot(repo, blobs, config)

    write(repo_root, "app.py", "def main() -> int:\n    return 77\n# added\n")

    changes = collect(repo, baseline, blobs, config)
    change = find(changes, "app.py")

    assert change is not None
    assert change.insertions == 1, "only the line this session added"
    assert change.deletions == 0
    assert [pre.path for pre in changes.pre_existing] == ["app.py"]
    assert changes.pre_existing[0].insertions == 1, "the pre-existing edit is counted separately"
    assert changes.pre_existing[0].deletions == 1


def test_a_commit_during_the_session_does_not_lose_changes(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    """The baseline is the commit the session started from, not whatever HEAD is now.

    Diffing against a moved HEAD would silently drop everything the agent had already
    committed and report an empty session.
    """
    baseline = snapshot(repo, blobs, config)

    write(repo_root, "app.py", "def main() -> int:\n    return 5\n")
    run_git(repo_root, "commit", "-q", "-am", "agent commit")

    change = find(collect(repo, baseline, blobs, config), "app.py")

    assert change is not None
    assert change.insertions == 1
    assert change.deletions == 1


def test_an_untracked_file_committed_during_the_session_is_still_reported(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    write(repo_root, "scratch.py", "one\n")
    baseline = snapshot(repo, blobs, config)

    write(repo_root, "scratch.py", "one\ntwo\n")
    run_git(repo_root, "add", "scratch.py")
    run_git(repo_root, "commit", "-q", "-m", "add scratch")

    change = find(collect(repo, baseline, blobs, config), "scratch.py")

    assert change is not None
    assert change.insertions == 1
    assert change.deletions == 0


# --------------------------------------------------------------------------- secrets


def test_an_untracked_secret_file_is_never_read(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    """plan.md §26: the strongest guarantee is not collecting the contents at all."""
    write(repo_root, ".env", "API_KEY=super-secret-value\n")

    baseline = snapshot(repo, blobs, config)

    entry = baseline.captured[0]
    assert entry.path == ".env"
    assert entry.digest is None
    assert entry.withheld_reason == "sensitive path"
    stored = [path for path in blobs.root.rglob("*") if path.is_file()]
    assert stored == [], "the contents must not reach the blob store"


def test_a_modified_untracked_secret_file_is_not_reported(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    """Reporting a change that could not be determined would be a fabrication."""
    write(repo_root, ".env", "API_KEY=one\n")
    baseline = snapshot(repo, blobs, config)

    write(repo_root, ".env", "API_KEY=two\n")

    assert find(collect(repo, baseline, blobs, config), ".env") is None


def test_a_tracked_secret_file_is_reported_with_its_contents_withheld(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    """git supplies status and counts for a tracked file without anyone reading it."""
    write(repo_root, ".env", "API_KEY=one\n")
    run_git(repo_root, "add", "-f", ".env")
    run_git(repo_root, "commit", "-q", "-m", "track env")

    baseline = snapshot(repo, blobs, config)
    write(repo_root, ".env", "API_KEY=one\nAPI_KEY2=two\n")

    change = find(collect(repo, baseline, blobs, config), ".env")

    assert change is not None
    assert change.contents_withheld is True
    assert change.insertions == 1


def test_a_new_secret_file_is_reported_as_added_without_being_read(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    """Its absence from the baseline is the evidence; no content read is needed."""
    baseline = snapshot(repo, blobs, config)
    write(repo_root, ".env", "API_KEY=brand-new\n")

    change = find(collect(repo, baseline, blobs, config), ".env")

    assert change is not None
    assert change.status is ChangeStatus.ADDED
    assert change.contents_withheld is True


def test_a_path_matching_no_pattern_is_still_read(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    """Confirms the previous tests are meaningful rather than always withholding."""
    write(repo_root, "environment.py", "SETTINGS = {}\n")
    baseline = snapshot(repo, blobs, config)

    assert baseline.captured[0].digest is not None


# --------------------------------------------------------------------------- binary and edge cases


def test_binary_files_have_no_line_counts(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    (repo_root / "logo.png").write_bytes(b"\x89PNG\x00\x01\x02")
    run_git(repo_root, "add", "logo.png")
    run_git(repo_root, "commit", "-q", "-m", "add image")

    baseline = snapshot(repo, blobs, config)
    (repo_root / "logo.png").write_bytes(b"\x89PNG\x00\x03\x04\x05")

    change = find(collect(repo, baseline, blobs, config), "logo.png")

    assert change is not None
    assert change.binary is True
    assert change.insertions is None
    assert change.changed_lines is None


def test_a_repository_without_commits_falls_back_to_the_empty_tree(
    tmp_path: Path, config: Config
) -> None:
    root = tmp_path / "fresh"
    root.mkdir()
    run_git(root, "init", "-q")

    repository = Repository.discover(root)
    assert repository is not None
    store = BlobStore(root / STATE_DIRNAME)

    baseline = snapshot(repository, store, config)
    assert baseline.commit is None

    write(root, "new.py", "x = 1\ny = 2\n")

    change = find(collect(repository, baseline, store, config), "new.py")

    assert change is not None
    assert change.status is ChangeStatus.ADDED
    assert change.insertions == 2


def test_the_change_set_records_the_revision_it_was_measured_against(
    repo: Repository, blobs: BlobStore, config: Config
) -> None:
    baseline = snapshot(repo, blobs, config)
    changes = collect(repo, baseline, blobs, config)

    assert changes.base_revision == baseline.base_revision
    assert changes.baseline_commit == baseline.commit


def test_the_change_set_round_trips_through_json(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    baseline = snapshot(repo, blobs, config)
    write(repo_root, "app.py", "def main() -> int:\n    return 3\n")

    payload = collect(repo, baseline, blobs, config).to_json()

    totals = payload["totals"]
    assert isinstance(totals, dict)
    assert totals["files"] == 1
    assert totals["insertions"] == 1
    assert payload["files"][0]["path"] == "app.py"  # type: ignore[index]


def test_a_large_file_is_reported_but_not_read(
    repo: Repository, repo_root: Path, blobs: BlobStore
) -> None:
    config = default_config()
    tiny = Config(
        repository=config.repository,
        activity=config.activity,
        analysis=type(config.analysis)(max_file_size_mb=0.000001),
        secrets=config.secrets,
        tests=config.tests,
        ignore=config.ignore,
    )

    write(repo_root, "huge.py", "x" * 5000 + "\n")
    baseline = snapshot(repo, blobs, tiny)

    assert baseline.captured[0].digest is None
    assert baseline.captured[0].withheld_reason is not None


# --------------------------------------------------------------------------- deleted at baseline


def test_a_file_deleted_before_the_session_is_not_the_session_s_deletion(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    """plan.md §14. A deletion that pre-dates the session belongs to the pre-existing list.

    ``git status`` reports a tracked file deleted from the working tree as a *change*, so it
    is captured — and the capture used to record "unreadable", which is indistinguishable
    from "deliberately withheld". The session then reported the deletion as its own work, and
    because the diff base is still the commit it was deleted from, every later session
    reported it again. Forever.
    """
    write(repo_root, "gone.py", "def old():\n    return None\n")
    write(repo_root, "kept.py", "def kept():\n    return 1\n")
    run_git(repo_root, "add", "-A")
    run_git(repo_root, "commit", "-q", "-m", "initial")

    (repo_root / "gone.py").unlink()  # before the session begins
    baseline = snapshot(repo, blobs, config)

    captured = next(item for item in baseline.captured if item.path == "gone.py")
    assert captured.absent is True
    assert captured.withheld_reason is None, "nothing was withheld — the file was gone"
    assert baseline.was_withheld("gone.py") is False
    assert baseline.was_absent("gone.py") is True

    write(repo_root, "kept.py", "def kept():\n    return 2\n")
    change_set = collect(repo, baseline, blobs, config)

    assert find(change_set, "gone.py") is None, "the session did not delete it"

    pre_existing = next(item for item in change_set.pre_existing if item.path == "gone.py")
    assert pre_existing.status is ChangeStatus.DELETED


def test_a_file_deleted_during_the_session_is_the_session_s_deletion(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    """The other half of the same rule: a deletion the session *did* make is reported."""
    write(repo_root, "gone.py", "def old():\n    return None\n")
    run_git(repo_root, "add", "-A")
    run_git(repo_root, "commit", "-q", "-m", "initial")
    baseline = snapshot(repo, blobs, config)

    (repo_root / "gone.py").unlink()
    change_set = collect(repo, baseline, blobs, config)

    change = find(change_set, "gone.py")
    assert change is not None
    assert change.status is ChangeStatus.DELETED


def test_a_file_that_reappears_after_being_deleted_is_an_addition(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    """Absent at baseline and present now is an addition, whatever git's index says."""
    write(repo_root, "gone.py", "def old():\n    return None\n")
    run_git(repo_root, "add", "-A")
    run_git(repo_root, "commit", "-q", "-m", "initial")

    (repo_root / "gone.py").unlink()
    baseline = snapshot(repo, blobs, config)

    write(repo_root, "gone.py", "def back():\n    return 1\n")
    change_set = collect(repo, baseline, blobs, config)

    change = find(change_set, "gone.py")
    assert change is not None
    assert change.status is ChangeStatus.ADDED
    assert change.insertions == 2
