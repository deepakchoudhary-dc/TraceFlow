"""Repository preparation: what ``traceflow init`` creates and modifies."""

from __future__ import annotations

from pathlib import Path

from traceflow.config import CONFIG_FILENAME, STATE_DIRNAME
from traceflow.scaffold import (
    GITIGNORE_FILENAME,
    ensure_gitignore_entry,
    gitignore_has_entry,
    initialise,
)


def test_initialise_creates_config_state_dir_and_ignore_entry(repo_root: Path) -> None:
    report = initialise(repo_root)

    assert report.config_created is True
    assert report.config_path == repo_root / CONFIG_FILENAME
    assert report.config_path.is_file()
    assert report.state_dir == repo_root / STATE_DIRNAME
    assert report.state_dir.is_dir()
    assert report.gitignore_updated is True
    assert report.gitignore_present is True


def test_initialise_is_idempotent(repo_root: Path) -> None:
    initialise(repo_root)
    before = (repo_root / CONFIG_FILENAME).read_text(encoding="utf-8")
    gitignore_before = (repo_root / GITIGNORE_FILENAME).read_text(encoding="utf-8")

    report = initialise(repo_root)

    assert report.config_created is False
    assert report.gitignore_updated is False
    assert (repo_root / CONFIG_FILENAME).read_text(encoding="utf-8") == before
    assert (repo_root / GITIGNORE_FILENAME).read_text(encoding="utf-8") == gitignore_before


def test_initialise_preserves_existing_gitignore_content(repo_root: Path) -> None:
    gitignore = repo_root / GITIGNORE_FILENAME
    original = "node_modules/\n*.log\n"
    gitignore.write_text(original, encoding="utf-8")

    initialise(repo_root)

    content = gitignore.read_text(encoding="utf-8")
    assert content.startswith(original)
    assert f"{STATE_DIRNAME}/" in content


def test_existing_gitignore_without_trailing_newline_is_handled(repo_root: Path) -> None:
    gitignore = repo_root / GITIGNORE_FILENAME
    gitignore.write_text("node_modules/", encoding="utf-8")

    ensure_gitignore_entry(repo_root, STATE_DIRNAME)

    lines = gitignore.read_text(encoding="utf-8").splitlines()
    assert "node_modules/" in lines
    assert f"{STATE_DIRNAME}/" in lines


def test_gitignore_entry_is_not_duplicated(repo_root: Path) -> None:
    ensure_gitignore_entry(repo_root, STATE_DIRNAME)
    assert ensure_gitignore_entry(repo_root, STATE_DIRNAME) is False

    content = (repo_root / GITIGNORE_FILENAME).read_text(encoding="utf-8")
    assert content.count(f"{STATE_DIRNAME}/") == 1


def test_gitignore_has_entry_ignores_comments_and_blank_lines(repo_root: Path) -> None:
    (repo_root / GITIGNORE_FILENAME).write_text(
        "# .traceflow/ is mentioned here but not ignored\n\n", encoding="utf-8"
    )
    assert gitignore_has_entry(repo_root, STATE_DIRNAME) is False


def test_gitignore_has_entry_tolerates_a_trailing_slash(repo_root: Path) -> None:
    (repo_root / GITIGNORE_FILENAME).write_text(f"{STATE_DIRNAME}/\n", encoding="utf-8")
    assert gitignore_has_entry(repo_root, STATE_DIRNAME) is True


def test_gitignore_has_entry_is_false_when_the_file_is_missing(tmp_path: Path) -> None:
    assert gitignore_has_entry(tmp_path, STATE_DIRNAME) is False


def test_init_does_not_touch_any_other_file(repo_root: Path) -> None:
    """plan.md §46: TraceFlow creates its own state and changes nothing else."""
    before = {
        path.relative_to(repo_root): path.read_bytes()
        for path in repo_root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }

    initialise(repo_root)

    after = {
        path.relative_to(repo_root): path.read_bytes()
        for path in repo_root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }
    changed = {name for name in before if name in after and before[name] != after[name]}
    created = set(after) - set(before)

    # The fixture repository has no .gitignore, so init creates one rather than
    # modifying it. Nothing that already existed may be altered.
    assert changed == set()
    assert created == {Path(CONFIG_FILENAME), Path(GITIGNORE_FILENAME)}


def test_init_only_appends_to_an_existing_gitignore(repo_root: Path) -> None:
    """The one pre-existing file init touches, and only by appending."""
    gitignore = repo_root / GITIGNORE_FILENAME
    gitignore.write_text("node_modules/\n", encoding="utf-8")

    before = {
        path.relative_to(repo_root): path.read_bytes()
        for path in repo_root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }
    initialise(repo_root)
    after = {
        path.relative_to(repo_root): path.read_bytes()
        for path in repo_root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }

    changed = {name for name in before if name in after and before[name] != after[name]}
    assert changed == {Path(GITIGNORE_FILENAME)}
    assert gitignore.read_text(encoding="utf-8").startswith("node_modules/\n")
