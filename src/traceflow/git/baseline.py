"""Baseline capture (plan.md §14).

A session's baseline is the state the repository was in *before* the change began.
Capturing it correctly is the difference between "these files changed during this
session" and "these files are changed" — and only the first is useful.

The baseline is built from two things:

* **The commit the session started from.** Diffing against that commit yields the
  session's changes to every file that was *clean* when the session began. Using the
  recorded commit rather than ``HEAD`` matters: an agent that commits partway
  through a session moves ``HEAD``, and diffing against the moved ``HEAD`` would
  silently lose everything it had done so far.
* **Content snapshots of files git cannot supply a "before" for.** A file that was
  already modified, or already untracked, has no committed version matching what was
  on disk. Its bytes are copied into TraceFlow's own content-addressed store, and
  the session's changes to that file are then measured against the snapshot rather
  than against the commit.

An earlier design used ``git stash create`` for the second part. It works, but it
writes unreachable objects — the full content of every dirty file — into the user's
object database on every session. A tool that promises to isolate its own generated
artifacts should not leave litter in someone else's repository, so the snapshot goes
into TraceFlow's own store instead. It also means one mechanism covers both cases.

Contents are read here and almost nowhere else, which is why the guards below sit in
this module: a file whose contents were never read cannot leak, whatever later code
does with the result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from traceflow.blobs import BlobStore
from traceflow.config import Config
from traceflow.git.repository import Repository, StatusEntry, WorkingTreeState
from traceflow.secrets import matches_secret_path
from traceflow.stamps import new_stamp_id, now_iso

#: Git's well-known empty tree object, used as the diff base in a repository that
#: has no commits and therefore no HEAD.
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

WITHHELD_SENSITIVE = "sensitive path"
WITHHELD_TOO_LARGE = "exceeds analysis.max_file_size_mb"
WITHHELD_UNREADABLE = "unreadable"

#: A tracked file that was already deleted from the working tree when the session began.
DELETED_AT_BASELINE = "deleted before this session began"

_TRACKED_CHANGE_KINDS = frozenset({"changed", "renamed", "unmerged"})


@dataclass(frozen=True)
class CapturedFile:
    """What TraceFlow recorded about one file whose "before" git cannot supply."""

    path: str
    size: int
    tracked: bool
    """True when git tracked the file (and it was modified); false when untracked."""

    digest: str | None = None
    """Content digest, or ``None`` when the contents were deliberately not stored."""

    withheld_reason: str | None = None

    absent: bool = False
    """True when the file was not on disk when the baseline was taken.

    A tracked file deleted but not yet staged is reported by ``git status`` as a change, and
    there is nothing to read. Recorded separately from ``withheld_reason`` because the two
    mean opposite things: nothing was withheld, the file was gone. Conflating them made a
    deletion that *pre-dated* the session look like the session's own work — and, because the
    diff base is still the commit it was deleted from, made every later session report it
    again.
    """

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Baseline:
    """The state a session is measured against."""

    baseline_id: str
    captured_at: str
    commit: str | None
    base_revision: str
    """What a later ``git diff`` should compare against: the commit, or the empty tree."""

    dirty: bool
    tracked_changes: int
    untracked_files: int
    captured: tuple[CapturedFile, ...] = ()

    def captured_digests(self) -> dict[str, str]:
        """Map path to stored digest, skipping files whose contents were withheld."""
        return {item.path: item.digest for item in self.captured if item.digest is not None}

    def captured_paths(self) -> frozenset[str]:
        """Every path whose baseline content TraceFlow holds or deliberately withheld."""
        return frozenset(item.path for item in self.captured)

    def was_absent(self, path: str) -> bool:
        """True when this path was recorded and was absent from disk at baseline."""
        for item in self.captured:
            if item.path == path:
                return item.absent
        return False

    def was_tracked(self, path: str) -> bool:
        for item in self.captured:
            if item.path == path:
                return item.tracked
        return True

    def was_withheld(self, path: str) -> bool:
        """True when this path was captured and its contents deliberately not stored.

        The record exists so the decision survives the process that made it. Without it, a
        later caller asking for the file's pre-session content would fall back to git and
        read the committed version — honouring the policy in one code path and breaking it
        through another (plan.md §26).
        """
        for item in self.captured:
            if item.path == path:
                return item.withheld_reason is not None
        return False

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["captured"] = [item.to_json() for item in self.captured]
        return payload


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _flag(value: object, default: bool = False) -> bool:
    """A boolean from stored JSON. Anything else takes *default*.

    ``bool("false")`` is true, so coercing would read a hand-edited string as a flag
    that is set.
    """
    return value if isinstance(value, bool) else default


def _int_or(value: object, default: int) -> int:
    """A whole number from stored JSON, or *default*.

    ``bool`` is excluded explicitly because it is a subclass of ``int``, so a stored
    ``true`` would otherwise be read as the number 1.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def baseline_from_json(payload: dict[str, object]) -> Baseline | None:
    """Rebuild a :class:`Baseline` from a stored payload, or ``None`` if it is malformed.

    Needed because ``traceflow analyze`` measures against the baseline the last run
    recorded, so the baseline has to survive a process boundary. Returning ``None`` rather
    than raising keeps the same rule the rest of the state directory follows: a damaged
    file is a missing one, never a crash (plan.md §46).
    """
    baseline_id = _optional_str(payload.get("baseline_id"))
    base_revision = _optional_str(payload.get("base_revision"))
    if baseline_id is None or base_revision is None:
        return None

    raw_captured = payload.get("captured")
    captured: list[CapturedFile] = []
    if isinstance(raw_captured, list):
        for item in raw_captured:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            if not isinstance(path, str):
                continue
            captured.append(
                CapturedFile(
                    path=path,
                    size=_int_or(item.get("size"), 0),
                    tracked=bool(item.get("tracked", True)),
                    digest=_optional_str(item.get("digest")),
                    withheld_reason=_optional_str(item.get("withheld_reason")),
                    absent=_flag(item.get("absent")),
                )
            )

    return Baseline(
        baseline_id=baseline_id,
        captured_at=_optional_str(payload.get("captured_at")) or "",
        commit=_optional_str(payload.get("commit")),
        base_revision=base_revision,
        dirty=_flag(payload.get("dirty")),
        tracked_changes=_int_or(payload.get("tracked_changes"), 0),
        untracked_files=_int_or(payload.get("untracked_files"), 0),
        captured=tuple(captured),
    )


def _capture_one(
    repository: Repository,
    blobs: BlobStore,
    entry: StatusEntry,
    tracked: bool,
    config: Config,
) -> CapturedFile:
    """Copy one file's current contents into the store, or record why not."""
    absolute = repository.root / entry.path

    try:
        size = absolute.stat().st_size
    except OSError:
        # Not unreadable — absent. A tracked file deleted from the working tree is
        # reported by git as a change, and the distinction between "we did not read it"
        # and "there was nothing to read" decides whether the session is credited with
        # a deletion it did not make.
        return CapturedFile(entry.path, 0, tracked, None, None, absent=True)

    if matches_secret_path(entry.path, config.secrets.exclude_paths):
        return CapturedFile(entry.path, size, tracked, None, WITHHELD_SENSITIVE)

    if size > config.analysis.max_file_size_bytes:
        return CapturedFile(entry.path, size, tracked, None, WITHHELD_TOO_LARGE)

    try:
        content = absolute.read_bytes()
    except OSError:
        return CapturedFile(entry.path, size, tracked, None, WITHHELD_UNREADABLE)

    return CapturedFile(entry.path, size, tracked, blobs.put(content))


def capture_baseline(
    repository: Repository,
    blobs: BlobStore,
    state: WorkingTreeState,
    config: Config,
    baseline_id: str | None = None,
) -> Baseline:
    """Capture the baseline for the session that is about to begin.

    *state* is the working-tree fingerprint the watcher already sampled, so the
    baseline and the fingerprint agree by construction rather than by two reads that
    could straddle a change.
    """
    captured: list[CapturedFile] = []
    for entry in state.entries:
        if entry.kind in _TRACKED_CHANGE_KINDS:
            captured.append(_capture_one(repository, blobs, entry, tracked=True, config=config))
        elif entry.kind == "untracked":
            captured.append(_capture_one(repository, blobs, entry, tracked=False, config=config))

    return Baseline(
        baseline_id=baseline_id or new_stamp_id(),
        captured_at=now_iso(),
        commit=state.head_oid,
        base_revision=state.head_oid or EMPTY_TREE,
        dirty=not state.is_clean,
        tracked_changes=state.tracked_change_count,
        untracked_files=state.untracked_count,
        captured=tuple(captured),
    )
