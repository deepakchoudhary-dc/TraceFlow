"""Repository discovery and working-tree state.

Two design notes that shape everything here:

1. **Git is the source of truth for what changed.** The watcher only decides *when*
   to look; git decides *what* is there. This is what makes the watcher robust:
   even if an activity signal is missed, the next successful status call returns
   the complete truth. It also means git's own ``.gitignore`` handling replaces a
   hand-maintained ignore list, which would inevitably drift from the real one.

2. **One call answers two questions.** ``git status`` reports both the current
   state and whether it differs from the last one, so activity detection and
   change collection share a single operation.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

GIT_TIMEOUT_SECONDS = 30.0

# Git writes this while it is mutating the index. A present lock means the working
# tree is in flux and no conclusion about stability can be drawn yet.
_INDEX_LOCK_NAME = "index.lock"


class GitError(RuntimeError):
    """Raised when git cannot be run, times out, or fails unexpectedly."""


@dataclass(frozen=True)
class StatusEntry:
    """One changed path as reported by ``git status --porcelain=v2``."""

    kind: str
    """One of: ``changed``, ``renamed``, ``unmerged``, ``untracked``, ``ignored``."""

    path: str
    original_path: str | None = None
    """For renames, the path the file had before the rename."""

    @property
    def is_tracked_change(self) -> bool:
        """True for modifications to files git already tracks."""
        return self.kind in {"changed", "renamed", "unmerged"}


@dataclass(frozen=True)
class ParsedStatus:
    """The result of parsing a porcelain v2 status payload."""

    branch_oid: str | None
    entries: tuple[StatusEntry, ...]


@dataclass(frozen=True)
class WorkingTreeState:
    """A fingerprint of the repository's working tree at one instant.

    ``token`` is stable: the same working tree always produces the same token, and
    any change to tracked or untracked content produces a different one. Comparing
    successive tokens is how activity is detected, which means the watcher never
    needs to understand filesystem events at all.

    ``head_oid`` is carried alongside because HEAD is already part of the status
    output — recording it here means the commit at any observed instant is available
    without a second git call.
    """

    token: str
    entries: tuple[StatusEntry, ...]
    head_oid: str | None = None

    @property
    def is_clean(self) -> bool:
        return not self.entries

    @property
    def tracked_change_count(self) -> int:
        return sum(1 for entry in self.entries if entry.is_tracked_change)

    @property
    def untracked_count(self) -> int:
        return sum(1 for entry in self.entries if entry.kind == "untracked")


def _path_after(record: str, field_count: int) -> str:
    """Return the path field of a porcelain record.

    Paths may contain spaces, so the record is split at most ``field_count`` times
    and the final remainder is the path.
    """
    parts = record.split(" ", field_count)
    return parts[field_count] if len(parts) > field_count else ""


def parse_porcelain_v2(output: str) -> ParsedStatus:
    """Parse ``git status --porcelain=v2 -z`` output.

    Records are NUL-separated rather than newline-separated. Rename records carry a
    second NUL-separated field holding the original path, so this cannot be a naive
    split-and-filter — the parser walks the fields and consumes that extra field
    when it appears.
    """
    fields = output.split("\0")
    entries: list[StatusEntry] = []
    branch_oid: str | None = None

    index = 0
    while index < len(fields):
        record = fields[index].lstrip("\n")
        index += 1

        if not record:
            continue
        if record.startswith("#"):
            if record.startswith("# branch.oid "):
                candidate = record[len("# branch.oid ") :].strip()
                # A repository with no commits reports "(initial)".
                branch_oid = None if candidate == "(initial)" else candidate
            continue

        kind = record[0]
        if kind == "?":
            entries.append(StatusEntry(kind="untracked", path=record[2:]))
        elif kind == "!":
            entries.append(StatusEntry(kind="ignored", path=record[2:]))
        elif kind == "1":
            entries.append(StatusEntry(kind="changed", path=_path_after(record, 8)))
        elif kind == "2":
            original = fields[index] if index < len(fields) else ""
            index += 1
            entries.append(
                StatusEntry(
                    kind="renamed",
                    path=_path_after(record, 9),
                    original_path=original or None,
                )
            )
        elif kind == "u":
            entries.append(StatusEntry(kind="unmerged", path=_path_after(record, 10)))
        # Unknown record types are skipped deliberately: a future git version
        # adding a new record type must not crash a running watcher.

    return ParsedStatus(branch_oid=branch_oid, entries=tuple(entries))


def _normalise(path: str) -> str:
    return path.replace("\\", "/").strip("/")


def is_ignored_path(path: str, ignore: tuple[str, ...]) -> bool:
    """True when *path* falls under one of the ignored prefixes."""
    normalised = _normalise(path)
    for prefix in ignore:
        cleaned = _normalise(prefix)
        if cleaned and (normalised == cleaned or normalised.startswith(f"{cleaned}/")):
            return True
    return False


class Repository:
    """A discovered git repository.

    Construction performs discovery once so that later calls do not repeat it.
    """

    def __init__(self, root: Path, git_dir: Path) -> None:
        self._root = root
        self._git_dir = git_dir

    @property
    def root(self) -> Path:
        """Absolute path to the repository working tree."""
        return self._root

    @property
    def name(self) -> str:
        return self._root.name

    @property
    def git_dir(self) -> Path:
        return self._git_dir

    @classmethod
    def discover(cls, path: Path) -> Repository | None:
        """Locate the repository containing *path*, or return ``None``.

        Returning ``None`` rather than raising lets callers report "not a git
        repository" as an ordinary outcome instead of an exception.
        """
        candidate = path.resolve()
        if not candidate.is_dir():
            return None

        toplevel = run_git(["rev-parse", "--show-toplevel"], cwd=candidate, check=False)
        if toplevel.returncode != 0:
            return None

        git_dir = run_git(["rev-parse", "--absolute-git-dir"], cwd=candidate, check=False)
        if git_dir.returncode != 0:
            return None

        return cls(
            root=Path(toplevel.stdout.strip()),
            git_dir=Path(git_dir.stdout.strip()),
        )

    def head_commit(self) -> str | None:
        """Return the current HEAD commit, or ``None`` in a repository with no commits."""
        result = run_git(["rev-parse", "HEAD"], cwd=self._root, check=False)
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    def is_git_busy(self) -> bool:
        """True while git is mid-operation and the working tree cannot be trusted as settled."""
        return (self._git_dir / _INDEX_LOCK_NAME).exists()

    def git(self, *args: str, check: bool = True) -> str:
        """Run git inside this repository and return stdout.

        This is the primitive the evidence modules build on. ``check=False`` exists
        for the cases where a non-zero exit is a meaningful answer rather than a
        failure — ``git diff --no-index`` exits 1 whenever two files differ, which
        is the normal outcome, not an error.
        """
        return run_git(list(args), cwd=self._root, check=check).stdout

    def working_tree_state(self, ignore: tuple[str, ...] = ()) -> WorkingTreeState:
        """Fingerprint the working tree, excluding *ignore* prefixes.

        ``--untracked-files=all`` is used rather than ``normal`` on purpose. With
        ``normal`` a brand-new directory is reported as a single entry, so adding a
        second file inside it leaves the output unchanged — activity would be
        missed exactly when an agent is creating new files, which is the common
        case. Correctness wins over the extra directory walk.
        """
        result = run_git(
            [
                "-c",
                "core.quotepath=false",
                "status",
                "--porcelain=v2",
                "-z",
                "--untracked-files=all",
                "--branch",
            ],
            cwd=self._root,
        )

        parsed = parse_porcelain_v2(result.stdout)
        entries = tuple(
            entry for entry in parsed.entries if not is_ignored_path(entry.path, ignore)
        )

        token = _token_for(parsed.branch_oid, entries, _stat_signature(self._root, entries))
        return WorkingTreeState(token=token, entries=entries, head_oid=parsed.branch_oid)


def _stat_signature(root: Path, entries: tuple[StatusEntry, ...]) -> str:
    """Size and modification time for every changed or untracked path.

    ``git status --porcelain=v2`` describes *which* files differ from the index, but
    not *how much* they differ. Editing a file that is already modified therefore
    leaves the porcelain output byte for byte identical, and a token built from it
    alone would report a repository that never settles and never changes again.

    That is precisely the case the baseline exists for — a file already dirty when a
    session begins — so the fingerprint has to include something that moves when the
    content does. Size and modification time are two cheap stats that between them
    catch every ordinary write, and unlike reading the file they cost nothing on a
    large dirty tree.
    """
    parts: list[str] = []
    for entry in entries:
        try:
            info = (root / entry.path).stat()
            parts.append(f"{entry.path}\0{info.st_size}\0{info.st_mtime_ns}")
        except OSError:
            # A path git still reports but which is gone from disk. Recorded rather
            # than skipped so its disappearance is itself a change.
            parts.append(f"{entry.path}\0missing")
    return "\n".join(parts)


def _token_for(
    branch_oid: str | None, entries: tuple[StatusEntry, ...], stat_signature: str
) -> str:
    """Build the stable fingerprint used to detect activity.

    HEAD is part of the fingerprint so that a commit registers as activity even when
    it leaves the working tree clean.
    """
    digest = hashlib.sha256()
    digest.update((branch_oid or "-").encode("utf-8", "surrogateescape"))
    for entry in entries:
        digest.update(b"\0")
        digest.update(entry.kind.encode("utf-8", "surrogateescape"))
        digest.update(b"\0")
        digest.update(entry.path.encode("utf-8", "surrogateescape"))
        if entry.original_path is not None:
            digest.update(b"\0")
            digest.update(entry.original_path.encode("utf-8", "surrogateescape"))
    digest.update(b"\0")
    digest.update(stat_signature.encode("utf-8", "surrogateescape"))
    return digest.hexdigest()


def run_git(args: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run git and return the completed process.

    ``surrogateescape`` decoding keeps unusual filenames from raising on the way
    through; the bytes survive the round trip unchanged.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="surrogateescape",
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GitError("git executable not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"git {args[0]} timed out after {GIT_TIMEOUT_SECONDS:g}s") from exc

    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or "no output"
        raise GitError(f"git {' '.join(args)} failed ({result.returncode}): {detail}")

    return result
