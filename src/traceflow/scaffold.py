"""Preparing a repository for TraceFlow.

This module owns every file TraceFlow creates inside someone else's repository, and
it is kept separate from the CLI so that the behaviour is testable without going
through argument parsing.

The only file TraceFlow ever modifies that it did not create is ``.gitignore``, and
it does so for one specific reason: ``plan.md`` §35 writes diffs into
``.traceflow/``, so leaving that directory untracked-but-unignored would let a
session artifact — including a diff of a ``.env`` file — be committed by accident.
``plan.md`` §46 forbids overwriting user files unexpectedly, so the entry is only
added by an explicit ``traceflow init``, is appended rather than rewritten, and is
reported to the user.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from traceflow.config import CONFIG_FILENAME, STATE_DIRNAME, render_default_config
from traceflow.watcher.session import SessionStore

GITIGNORE_FILENAME = ".gitignore"


@dataclass(frozen=True)
class InitReport:
    """What ``traceflow init`` actually did, so the CLI can report it honestly."""

    config_path: Path
    config_created: bool
    state_dir: Path
    gitignore_path: Path
    gitignore_updated: bool
    gitignore_present: bool


def _read_gitignore(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def gitignore_has_entry(root: Path, entry: str) -> bool:
    """True when *.gitignore* already ignores *entry*."""
    wanted = entry.strip().rstrip("/")
    for line in _read_gitignore(root / GITIGNORE_FILENAME).splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and stripped.rstrip("/") == wanted:
            return True
    return False


def ensure_gitignore_entry(root: Path, entry: str) -> bool:
    """Append *entry* to the repository's ``.gitignore``. Returns True if it changed.

    The file is appended to, never rewritten, so existing user content is preserved
    byte for byte.
    """
    if gitignore_has_entry(root, entry):
        return False

    path = root / GITIGNORE_FILENAME
    existing = _read_gitignore(path)
    separator = "" if not existing else ("" if existing.endswith("\n") else "\n")
    block = f"{separator}\n# TraceFlow state directory\n{entry}/\n"

    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(block)
    return True


def initialise(root: Path) -> InitReport:
    """Prepare *root* for TraceFlow, creating only what is missing.

    Idempotent: running it twice reports ``config_created=False`` and
    ``gitignore_updated=False`` the second time and changes nothing.
    """
    config_path = root / CONFIG_FILENAME
    config_created = not config_path.is_file()
    if config_created:
        config_path.write_text(render_default_config(), encoding="utf-8", newline="\n")

    store = SessionStore(root)
    store.ensure_state_dir()

    gitignore_path = root / GITIGNORE_FILENAME
    gitignore_updated = ensure_gitignore_entry(root, STATE_DIRNAME)

    return InitReport(
        config_path=config_path,
        config_created=config_created,
        state_dir=store.state_dir,
        gitignore_path=gitignore_path,
        gitignore_updated=gitignore_updated,
        gitignore_present=gitignore_has_entry(root, STATE_DIRNAME),
    )
