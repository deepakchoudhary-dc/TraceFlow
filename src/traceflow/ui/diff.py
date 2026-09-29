"""Session diffs for the delivery view (plan.md §26, §46, §61).

"View Diff" is the only part of the dashboard that shows file *contents*, which makes it
the second place the secret policy has to hold. Analysis never reads a sensitive file, so
the diff must not either — and it refuses the path outright rather than redacting it,
because a redaction rule is a rule that can be got wrong, while a file that was never
read cannot leak (plan.md §26).

Two honesty properties matter as much as the diff itself.

**A diff is measured against the session's own baseline, never against `HEAD`.** That is
the same rule the change set follows, and for the same reason: an agent that commits
partway through a session moves `HEAD`, and diffing against it would show nothing for
everything already committed.

**The file may have moved on since the session was recorded.** The session stored a digest
of each module as it was, so drift is detectable rather than merely possible. When it is
detected the diff still renders — the reader asked for it — but it says so, because a
diff silently mixing a later edit into a session's record is exactly the kind of quiet
misattribution this project exists to prevent.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from traceflow.analysis.symbols import baseline_source, current_source
from traceflow.blobs import BlobStore, digest_of, looks_binary
from traceflow.config import Config
from traceflow.git.baseline import Baseline
from traceflow.git.repository import Repository
from traceflow.secrets import matches_secret_path

#: How much of a diff is rendered. A four-hundred-line cap is not a limitation of the
#: engine, it is a limit on what a page can usefully show; the artifact on disk still
#: holds the full picture, and the note says the view was cut short.
MAX_DIFF_LINES = 400

DIFF_CONTEXT_LINES = 3

AVAILABLE = "available"
WITHHELD = "withheld"
BINARY = "binary"
TOO_LARGE = "too_large"
UNAVAILABLE = "unavailable"

_ADDED = "added"
_DELETED = "deleted"


@dataclass(frozen=True)
class DiffRequest:
    """Which file to diff, and what the session recorded about it."""

    path: str
    status: str
    original_path: str | None = None
    """Where the file was before the session, when it was renamed."""

    recorded_digest: str | None = None
    """The digest of the file's content at the end of the session, from `symbols.json`.

    ``None`` for a file the analyzer does not handle, in which case drift cannot be
    detected — which is stated by silence rather than by a claim.
    """


@dataclass(frozen=True)
class FileDiff:
    """A rendered diff, or the reason there is not one."""

    path: str
    status: str
    lines: tuple[str, ...] = ()
    note: str = ""
    truncated: bool = False
    drifted: bool = False

    @property
    def is_available(self) -> bool:
        return self.status == AVAILABLE

    @property
    def insertions(self) -> int:
        return sum(1 for line in self.lines if line.startswith("+") and not line.startswith("+++"))

    @property
    def deletions(self) -> int:
        return sum(1 for line in self.lines if line.startswith("-") and not line.startswith("---"))


def _decode(content: bytes) -> list[str]:
    return content.decode("utf-8", errors="replace").splitlines()


def _unavailable(request: DiffRequest, note: str, status: str = UNAVAILABLE) -> FileDiff:
    return FileDiff(path=request.path, status=status, note=note)


def file_diff(
    repository: Repository,
    baseline: Baseline,
    blobs: BlobStore,
    config: Config,
    request: DiffRequest,
) -> FileDiff:
    """Render *request*'s diff against the session baseline, or explain why it cannot.

    Never raises: every failure to produce a diff is a status with a reason attached,
    because a dashboard that 500s on one awkward file is less useful than one that says
    which file it could not show (plan.md §46).
    """
    if matches_secret_path(request.path, config.secrets.exclude_paths):
        return _unavailable(
            request,
            "This path matches the sensitive-path policy, so its contents were never "
            "read — not during analysis, and not here.",
            status=WITHHELD,
        )

    # The size policy is applied before anything is read, so an oversized file is never
    # even asked for — and says so, rather than being reported as generically unavailable.
    absolute = repository.root / request.path
    exists = absolute.is_file()
    if exists:
        try:
            size = absolute.stat().st_size
        except OSError:
            size = 0
        if size > config.analysis.max_file_size_bytes:
            limit = config.analysis.max_file_size_mb
            return _unavailable(
                request,
                f"The file is {size} bytes, above analysis.max_file_size_mb ({limit:g} MB), "
                f"so its contents were never read.",
                status=TOO_LARGE,
            )

    before = baseline_source(repository, baseline, request.original_path or request.path, blobs)
    if before is None:
        if request.status != _ADDED:
            return _unavailable(request, "The baseline contents for this file are unavailable.")
        before = b""

    if exists:
        after, reason = current_source(repository, request.path, config)
        if after is None:
            # The file is there but could not be read, which is a different fact from a
            # deletion — so it gets a different answer.
            return _unavailable(request, f"The file could not be read: {reason}.")
    elif request.status == _DELETED:
        after = b""
    else:
        return _unavailable(request, "The file no longer exists on disk.")

    if looks_binary(before) or looks_binary(after):
        return _unavailable(
            request, "One side is binary, where a line diff has no meaning.", status=BINARY
        )

    drifted = request.recorded_digest is not None and digest_of(after) != request.recorded_digest

    rendered = tuple(
        difflib.unified_diff(
            _decode(before),
            _decode(after),
            fromfile=f"a/{request.original_path or request.path}",
            tofile=f"b/{request.path}",
            n=DIFF_CONTEXT_LINES,
            lineterm="",
        )
    )

    truncated = len(rendered) > MAX_DIFF_LINES
    note = ""
    if drifted:
        note = (
            "This file has changed since the session was recorded, so the diff below "
            "includes edits that were not part of it."
        )
    if truncated:
        cut = f"Showing the first {MAX_DIFF_LINES} of {len(rendered)} diff lines."
        note = f"{note} {cut}".strip()

    return FileDiff(
        path=request.path,
        status=AVAILABLE,
        lines=rendered[:MAX_DIFF_LINES],
        note=note,
        truncated=truncated,
        drifted=drifted,
    )
