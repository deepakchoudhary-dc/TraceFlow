"""Shared test fixtures.

Every fixture that touches git builds a real repository in a temporary directory.
Nothing is mocked: the whole point of the watcher is how it behaves against real git
output, and a fake would only test the fake.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from traceflow.config import Config, default_config
from traceflow.git.repository import Repository

# Configured per-invocation rather than relying on the developer's global git config, so
# the suite behaves identically on a machine with commit signing enabled.
#
# The identity is passed here rather than written with two `git config` calls in the
# fixture. That is two fewer process spawns for every test that needs a repository, and on
# Windows a git spawn costs ~120ms — so the saving is real time in every run, not tidiness.
_GIT_CONFIG = (
    "-c",
    "core.autocrlf=false",
    "-c",
    "commit.gpgsign=false",
    "-c",
    "init.defaultBranch=main",
    "-c",
    "protocol.file.allow=always",
    "-c",
    "user.email=traceflow@example.invalid",
    "-c",
    "user.name=TraceFlow Tests",
)


def run_git(root: Path, *args: str) -> str:
    """Run git in *root*, asserting success, and return stdout."""
    result = subprocess.run(
        ["git", *_GIT_CONFIG, *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
        check=False,
    )
    assert result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr}"
    return result.stdout


def write_file(root: Path, relative: str, text: str) -> None:
    """Write a file with explicit LF endings so line numbers are platform-independent."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def commit(root: Path, message: str = "change") -> None:
    run_git(root, "add", "-A")
    run_git(root, "commit", "-q", "-m", message)


def changed_callers_repo(root: Path) -> None:
    """A declaration with two callers the session will also edit, and one it will not.

    This is the shape that hid the impact engine's duplicate-finding defect: the walk
    reaches all three callers, but two of them are already recorded as changes, so the
    obligation reached them and was dropped. Shared between `test_impact.py`, which asserts
    the artifact, and `test_ui.py`, which asserts that the count above a list matches it —
    two files describing one scenario is how the two descriptions drift apart.
    """
    write_file(root, "auth/__init__.py", "")
    write_file(root, "auth/service.py", "def authenticate(user, password):\n    return user\n")
    write_file(
        root,
        "auth/routes.py",
        "from auth.service import authenticate\n\n\ndef login(user, password):\n"
        "    return authenticate(user, password)\n",
    )
    write_file(
        root,
        "auth/admin.py",
        "from auth.service import authenticate\n\n\ndef admin_login(u, p):\n"
        "    return authenticate(u, p)\n",
    )
    write_file(
        root,
        "tests/test_auth.py",
        "from auth.service import authenticate\n\n\ndef test_authenticate():\n"
        "    return authenticate('a', 'b')\n",
    )
    commit(root, "add auth")


def move_the_declaration(root: Path) -> None:
    """Move the declaration, and edit both of the callers the session already touched."""
    write_file(
        root,
        "auth/service.py",
        "def authenticate(user, password, *, strict=False):\n    return user\n",
    )
    write_file(
        root,
        "auth/routes.py",
        "from auth.service import authenticate\n\n\ndef login(user, password):\n"
        "    return authenticate(user, password) or False\n",
    )
    write_file(
        root,
        "auth/admin.py",
        "from auth.service import authenticate\n\n\ndef admin_login(u, p):\n"
        "    return authenticate(u, p) and True\n",
    )


class FakeClock:
    """A monotonic clock under test control.

    Time-dependent behaviour is tested by advancing this, never by sleeping, so the
    whole suite stays fast and deterministic.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def config() -> Config:
    """The built-in configuration, independent of anything on disk."""
    return default_config()


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    """A git repository with one commit and a clean working tree.

    Files are written with explicit LF endings. Without that, Windows would store
    CRLF and every test that rewrites a fixture file would produce a whole-file diff,
    making the line counts depend on the platform rather than on the change.
    """
    root = tmp_path / "sample-repo"
    root.mkdir()

    run_git(root, "init", "-q")

    (root / "app.py").write_text(
        "def main() -> int:\n    return 1\n", encoding="utf-8", newline="\n"
    )
    (root / "README.md").write_text("# Sample\n", encoding="utf-8", newline="\n")

    run_git(root, "add", ".")
    run_git(root, "commit", "-q", "-m", "initial commit")
    return root


@pytest.fixture
def repo(repo_root: Path) -> Repository:
    discovered = Repository.discover(repo_root)
    assert discovered is not None, "fixture repository should be discoverable"
    return discovered
