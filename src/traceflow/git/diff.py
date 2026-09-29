"""Change collection (plan.md §15, §16 groundwork).

Given a baseline and the repository's current state, work out precisely what changed
during the session, and keep the pre-existing modifications out of the answer.

Three sources are combined, each used where it is the authority:

1. **Files that were clean when the session began** are diffed by git against the
   commit the session started from. Git's line counts, rename detection and binary
   handling are the reference implementation, and reimplementing them would only
   produce numbers that disagree with ``git diff``. Git reports these without reading
   the files at all, which is also what keeps secret files out of memory.
2. **Files whose baseline content TraceFlow snapshotted** — already modified, or
   already untracked — are compared against that snapshot with :mod:`difflib`,
   because git has no record of what they looked like at that moment.
3. **Files that became untracked during the session** are new, and their contents
   are counted directly.

A file appears in exactly one of these, so no path is ever counted twice.

On withheld contents: TraceFlow refuses to read files matching the sensitive-path
policy. For a *tracked* such file git still supplies status and line counts without
anyone reading the bytes, so it is reported normally. For an *untracked* one there is
no such source, so it is recorded in the baseline with its reason and left out of the
change set. Reporting a change that could not be determined would be a fabrication.
"""

from __future__ import annotations

import difflib
from dataclasses import asdict, dataclass
from enum import Enum

from traceflow.blobs import BlobStore, looks_binary
from traceflow.config import Config
from traceflow.git.baseline import (
    DELETED_AT_BASELINE,
    WITHHELD_SENSITIVE,
    WITHHELD_TOO_LARGE,
    WITHHELD_UNREADABLE,
    Baseline,
    CapturedFile,
)
from traceflow.git.repository import Repository, WorkingTreeState, run_git
from traceflow.secrets import matches_secret_path

UNKNOWN_LINE_COUNTS = "line statistics unavailable"
UNAVAILABLE_BASELINE = "baseline contents unavailable"
PRE_EXISTING_COUNTS_UNAVAILABLE = "pre-existing line counts unavailable"


class ChangeStatus(str, Enum):
    """How a file changed between the baseline and now."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    COPIED = "copied"
    TYPE_CHANGED = "type_changed"
    UNKNOWN = "unknown"


_STATUS_BY_CODE: dict[str, ChangeStatus] = {
    "A": ChangeStatus.ADDED,
    "M": ChangeStatus.MODIFIED,
    "D": ChangeStatus.DELETED,
    "R": ChangeStatus.RENAMED,
    "C": ChangeStatus.COPIED,
    "T": ChangeStatus.TYPE_CHANGED,
}


@dataclass(frozen=True)
class RawChange:
    """One record from ``git diff --name-status``."""

    status: ChangeStatus
    path: str
    original_path: str | None = None


@dataclass(frozen=True)
class FileChange:
    """One file's change during a session."""

    path: str
    status: ChangeStatus
    original_path: str | None = None
    insertions: int | None = None
    deletions: int | None = None
    binary: bool = False
    tracked: bool = True
    """True when git tracked this file at the start of the session."""

    contents_withheld: bool = False
    """True when the path matched the sensitive-path policy and its contents were not read."""

    note: str | None = None
    """Why line counts are missing, when the reason is neither binary nor size."""

    @property
    def changed_lines(self) -> int | None:
        """plan.md §15's "changed lines": insertions plus deletions."""
        if self.insertions is None or self.deletions is None:
            return None
        return self.insertions + self.deletions

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ChangeSet:
    """Everything that changed during one session, plus what did not."""

    files: tuple[FileChange, ...]
    pre_existing: tuple[FileChange, ...]
    base_revision: str | None
    baseline_commit: str | None
    untracked_at_baseline: int

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def insertions(self) -> int:
        return sum(change.insertions or 0 for change in self.files)

    @property
    def deletions(self) -> int:
        return sum(change.deletions or 0 for change in self.files)

    @property
    def changed_lines(self) -> int:
        return self.insertions + self.deletions

    def count_of(self, status: ChangeStatus) -> int:
        return sum(1 for change in self.files if change.status is status)

    @property
    def pre_existing_count(self) -> int:
        return len(self.pre_existing)

    @property
    def withheld_count(self) -> int:
        return sum(1 for change in self.files if change.contents_withheld)

    def to_json(self) -> dict[str, object]:
        return {
            "base_revision": self.base_revision,
            "baseline_commit": self.baseline_commit,
            "untracked_at_baseline": self.untracked_at_baseline,
            "totals": {
                "files": self.file_count,
                "insertions": self.insertions,
                "deletions": self.deletions,
                "changed_lines": self.changed_lines,
                "pre_existing_files": self.pre_existing_count,
                "contents_withheld": self.withheld_count,
            },
            "files": [change.to_json() for change in self.files],
            "pre_existing": [change.to_json() for change in self.pre_existing],
        }


# --------------------------------------------------------------------------- parsing


def _parse_count(value: str) -> int | None:
    """git writes ``-`` for a binary file, where a line count is meaningless."""
    if value == "-":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_name_status(output: str) -> list[RawChange]:
    """Parse ``git diff --name-status -z`` output.

    Records are NUL-separated, and the status is its own field rather than being
    tab-joined to the path. A rename or copy contributes three fields — status,
    pre-image path, post-image path — so this cannot be a naive pair-wise walk.
    """
    fields = output.split("\0")
    changes: list[RawChange] = []

    index = 0
    while index < len(fields):
        status_field = fields[index]
        index += 1
        if not status_field:
            continue

        code = status_field[0]
        status = _STATUS_BY_CODE.get(code, ChangeStatus.UNKNOWN)

        path = fields[index] if index < len(fields) else ""
        index += 1

        original: str | None = None
        if code in {"R", "C"}:
            original = path
            path = fields[index] if index < len(fields) else ""
            index += 1

        changes.append(RawChange(status=status, path=path, original_path=original))

    return changes


def parse_numstat(output: str) -> dict[str, tuple[int | None, int | None]]:
    """Parse ``git diff --numstat -z`` output, keyed by post-image path.

    For a rename the third tab-separated field is empty and the pre-image and
    post-image paths follow as their own NUL-separated fields.
    """
    fields = output.split("\0")
    counts: dict[str, tuple[int | None, int | None]] = {}

    index = 0
    while index < len(fields):
        record = fields[index]
        index += 1
        if not record:
            continue

        parts = record.split("\t")
        if len(parts) < 3:
            continue

        insertions = _parse_count(parts[0])
        deletions = _parse_count(parts[1])
        path = parts[2]

        if not path:
            index += 1  # discard the pre-image path
            path = fields[index] if index < len(fields) else ""
            index += 1

        if path:
            counts[path] = (insertions, deletions)

    return counts


# --------------------------------------------------------------------------- line counting


def count_line_changes(before: bytes, after: bytes) -> tuple[int, int] | None:
    """Line-level insertions and deletions between two byte strings.

    Returns ``None`` when either side looks binary, because a line count for binary
    content is not a meaningful number and reporting one would be a fabrication.
    """
    if looks_binary(before) or looks_binary(after):
        return None

    before_lines = before.decode("utf-8", errors="replace").splitlines()
    after_lines = after.decode("utf-8", errors="replace").splitlines()

    matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines, autojunk=False)
    insertions = 0
    deletions = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"replace", "delete"}:
            deletions += i2 - i1
        if tag in {"replace", "insert"}:
            insertions += j2 - j1

    return insertions, deletions


# --------------------------------------------------------------------------- collection


def _collect_git_changes(
    repository: Repository, base_revision: str, config: Config
) -> dict[str, FileChange]:
    """Diff *base_revision* against the working tree and describe each changed file."""
    status_output = repository.git("diff", "--name-status", "-z", base_revision)
    counts = parse_numstat(repository.git("diff", "--numstat", "-z", base_revision))

    changes: dict[str, FileChange] = {}
    for raw in parse_name_status(status_output):
        counts_for_path = counts.get(raw.path)
        if counts_for_path is None:
            insertions: int | None = None
            deletions: int | None = None
            binary = False
            note: str | None = UNKNOWN_LINE_COUNTS
        else:
            insertions, deletions = counts_for_path
            binary = insertions is None
            note = None

        changes[raw.path] = FileChange(
            path=raw.path,
            status=raw.status,
            original_path=raw.original_path,
            insertions=insertions,
            deletions=deletions,
            binary=binary,
            tracked=True,
            contents_withheld=matches_secret_path(raw.path, config.secrets.exclude_paths),
            note=note,
        )

    return changes


def _read_current(
    repository: Repository, path: str, config: Config
) -> tuple[bytes | None, int | None, str | None]:
    """Read a file for comparison, honouring the size policy.

    Returns ``(content, size, withheld_reason)``. ``content`` is ``None`` when the
    file could not be read or was deliberately not read.
    """
    absolute = repository.root / path
    try:
        size = absolute.stat().st_size
    except OSError:
        return None, None, None  # gone; the caller decides what that means

    if size > config.analysis.max_file_size_bytes:
        return None, size, WITHHELD_TOO_LARGE

    try:
        return absolute.read_bytes(), size, None
    except OSError:
        return None, size, WITHHELD_UNREADABLE


def _describe_snapshotted(
    repository: Repository,
    item: CapturedFile,
    blobs: BlobStore,
    config: Config,
) -> FileChange | None:
    """Describe a file whose baseline content came from TraceFlow's own store.

    Returns ``None`` when the file is unchanged, so it is not reported at all.
    """
    path = item.path
    tracked = item.tracked
    sensitive = matches_secret_path(path, config.secrets.exclude_paths)
    content, _size, withheld = _read_current(repository, path, config)

    if item.absent:
        # The file was already gone when the session began, so there is no "before" to
        # compare against — only "nothing". Anything it holds now appeared during the
        # session; if it is still gone, the session did not do it.
        #
        # Without this branch the file was reported as *this session's* deletion, and
        # because the diff base is still the commit it was deleted from, every later session
        # reported it again. A deletion the session did not make, claimed indefinitely.
        if content is None:
            return None
        if sensitive:
            return FileChange(
                path=path,
                status=ChangeStatus.ADDED,
                tracked=tracked,
                contents_withheld=True,
                note=WITHHELD_SENSITIVE,
            )
        if withheld is not None:
            return FileChange(path=path, status=ChangeStatus.ADDED, tracked=tracked, note=withheld)
        counts = count_line_changes(b"", content)
        if counts is None:
            return FileChange(path=path, status=ChangeStatus.ADDED, tracked=tracked, binary=True)
        insertions, deletions = counts
        return FileChange(
            path=path,
            status=ChangeStatus.ADDED,
            insertions=insertions,
            deletions=deletions,
            tracked=tracked,
        )

    if sensitive:
        # The contents were deliberately never read, so this session's contribution
        # to the file cannot be determined. It still appears among the pre-existing
        # changes if git tracks it; guessing at a number here would be fabrication.
        return None

    if withheld is not None:
        return FileChange(path=path, status=ChangeStatus.MODIFIED, tracked=tracked, note=withheld)

    if content is None:
        return FileChange(path=path, status=ChangeStatus.DELETED, tracked=tracked)

    if item.digest is None:
        return FileChange(
            path=path, status=ChangeStatus.MODIFIED, tracked=tracked, note=UNAVAILABLE_BASELINE
        )

    before = blobs.get(item.digest)
    if before is None:
        return FileChange(
            path=path, status=ChangeStatus.MODIFIED, tracked=tracked, note=UNAVAILABLE_BASELINE
        )

    counts = count_line_changes(before, content)
    if counts is None:
        return FileChange(path=path, status=ChangeStatus.MODIFIED, tracked=tracked, binary=True)

    insertions, deletions = counts
    if insertions == 0 and deletions == 0:
        return None

    return FileChange(
        path=path,
        status=ChangeStatus.MODIFIED,
        insertions=insertions,
        deletions=deletions,
        tracked=tracked,
    )


def _describe_new_untracked(repository: Repository, path: str, config: Config) -> FileChange | None:
    """Describe a file that git did not track at baseline and does not track now.

    Its existence is the change: it was not there when the session began, so this is
    an addition and its whole contents count as insertions.
    """
    if matches_secret_path(path, config.secrets.exclude_paths):
        return FileChange(
            path=path,
            status=ChangeStatus.ADDED,
            tracked=False,
            contents_withheld=True,
            note=WITHHELD_SENSITIVE,
        )

    content, _size, withheld = _read_current(repository, path, config)

    if withheld is not None:
        return FileChange(path=path, status=ChangeStatus.ADDED, tracked=False, note=withheld)

    if content is None:
        return None  # vanished while collecting; the next settle will see the truth

    counts = count_line_changes(b"", content)
    if counts is None:
        return FileChange(path=path, status=ChangeStatus.ADDED, tracked=False, binary=True)

    insertions, deletions = counts
    return FileChange(
        path=path,
        status=ChangeStatus.ADDED,
        insertions=insertions,
        deletions=deletions,
        tracked=False,
    )


def committed_content(repository: Repository, revision: str, path: str) -> bytes | None:
    """Return a file's content at *revision*, or ``None`` when it is not in that revision."""
    result = run_git(["show", f"{revision}:{path}"], cwd=repository.root, check=False)
    if result.returncode != 0:
        return None
    # surrogateescape round-trips the bytes exactly, including non-UTF-8 content.
    return result.stdout.encode("utf-8", "surrogateescape")


def _pre_existing_change(
    repository: Repository, baseline: Baseline, item: CapturedFile, blobs: BlobStore
) -> FileChange | None:
    """The modification that was already present when the session began (plan.md §14).

    Measured from the committed version to the baseline snapshot — deliberately *not*
    to the current working tree. Diffing to the working tree would fold this session's
    own edits into the pre-existing number and overstate it, which is precisely the
    confusion the baseline exists to prevent.

    Returns ``None`` when the snapshot turns out to match the commit, which happens
    when git reported a file dirty on stat alone.
    """
    if item.absent:
        # Already deleted when the session began. Reported among the pre-existing changes —
        # which is where a change that pre-dates the session belongs — rather than as this
        # session's work, and with no line count because there is no content on either side
        # to compare.
        return FileChange(
            path=item.path,
            status=ChangeStatus.DELETED,
            tracked=True,
            note=DELETED_AT_BASELINE,
        )

    if item.digest is None:
        # Contents were withheld, so whether it changed cannot be determined. It was
        # dirty at baseline, and saying that much is honest.
        return FileChange(
            path=item.path,
            status=ChangeStatus.MODIFIED,
            tracked=True,
            contents_withheld=item.withheld_reason == WITHHELD_SENSITIVE,
            note=item.withheld_reason or UNAVAILABLE_BASELINE,
        )

    snapshot_content = blobs.get(item.digest)
    committed = (
        None
        if baseline.commit is None
        else committed_content(repository, baseline.commit, item.path)
    )
    if committed is None or snapshot_content is None:
        return FileChange(
            path=item.path,
            status=ChangeStatus.MODIFIED,
            tracked=True,
            note=PRE_EXISTING_COUNTS_UNAVAILABLE,
        )

    counts = count_line_changes(committed, snapshot_content)
    if counts is None:
        return FileChange(path=item.path, status=ChangeStatus.MODIFIED, tracked=True, binary=True)

    insertions, deletions = counts
    if insertions == 0 and deletions == 0:
        return None

    return FileChange(
        path=item.path,
        status=ChangeStatus.MODIFIED,
        insertions=insertions,
        deletions=deletions,
        tracked=True,
    )


def collect_changes(
    repository: Repository,
    baseline: Baseline,
    final_state: WorkingTreeState,
    blobs: BlobStore,
    config: Config,
) -> ChangeSet:
    """Collect everything that changed between *baseline* and *final_state*."""
    captured_paths = baseline.captured_paths()
    git_changes = _collect_git_changes(repository, baseline.base_revision, config)

    session: dict[str, FileChange] = {}
    for path, git_change in git_changes.items():
        if path in captured_paths:
            # Described from the snapshot below instead, so that the session's own
            # contribution can be separated from what was already there.
            continue
        session[path] = git_change

    for item in sorted(baseline.captured, key=lambda captured: captured.path):
        snapshotted = _describe_snapshotted(repository, item, blobs, config)
        if snapshotted is not None:
            session[item.path] = snapshotted

    for entry in final_state.entries:
        if entry.kind != "untracked" or entry.path in captured_paths:
            continue
        added = _describe_new_untracked(repository, entry.path, config)
        if added is not None:
            session[entry.path] = added

    pre_existing = [
        change
        for item in baseline.captured
        if item.tracked
        for change in [_pre_existing_change(repository, baseline, item, blobs)]
        if change is not None
    ]

    return ChangeSet(
        files=tuple(session[path] for path in sorted(session)),
        pre_existing=tuple(sorted(pre_existing, key=lambda change: change.path)),
        base_revision=baseline.base_revision,
        baseline_commit=baseline.commit,
        untracked_at_baseline=baseline.untracked_files,
    )
