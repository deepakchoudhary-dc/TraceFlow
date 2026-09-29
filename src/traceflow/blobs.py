"""Content-addressed storage for file contents.

Most of what changed in a session can be recovered from git: the committed version
of a tracked file is always available, and a working-tree snapshot covers the
modified ones. Two cases have no recoverable "before" at all:

* a file that was **already modified** when the session began, and
* a file that was **already untracked** when the session began.

For those, the bytes on disk at the moment the session started are the only record
of what the file looked like beforehand. This module stores them.

Storage is content-addressed, so identical content is stored once no matter how
many paths or sessions refer to it. That property also means a rename costs
nothing, and it is the same mechanism Phase 3 will use to cache parsed syntax
trees.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

BLOBS_DIRNAME = "blobs"

# git's own heuristic for deciding a file is binary: a NUL byte in the first
# 8000 bytes. Reusing it means TraceFlow and git agree about what "binary" means
# instead of each having a slightly different opinion.
_BINARY_SNIFF_BYTES = 8000


def looks_binary(content: bytes) -> bool:
    """True when *content* appears to be binary rather than text."""
    return b"\0" in content[:_BINARY_SNIFF_BYTES]


def digest_of(content: bytes) -> str:
    """Return the content-addressed key for *content*."""
    return hashlib.sha256(content).hexdigest()


class BlobStore:
    """A content-addressed store rooted at ``<state_dir>/blobs``.

    Objects are sharded by the first two hex characters of their digest so a
    repository with many captured files does not put them all in one directory.
    """

    def __init__(self, state_dir: Path) -> None:
        self._root = state_dir / BLOBS_DIRNAME

    @property
    def root(self) -> Path:
        return self._root

    def path_for(self, digest: str) -> Path:
        return self._root / digest[:2] / digest

    def put(self, content: bytes) -> str:
        """Store *content* and return its digest. Storing twice is free."""
        digest = digest_of(content)
        target = self.path_for(digest)
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return digest

    def has(self, digest: str) -> bool:
        return self.path_for(digest).is_file()

    def get(self, digest: str) -> bytes | None:
        """Return the stored content, or ``None`` if it is not present."""
        target = self.path_for(digest)
        if not target.is_file():
            return None
        return target.read_bytes()
