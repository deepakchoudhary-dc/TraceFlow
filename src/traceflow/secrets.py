"""Which files TraceFlow must never read the contents of.

plan.md §26 requires that TraceFlow never exposes secrets in logs, generated JSON,
the dashboard, diagrams, or error messages. The strongest way to guarantee that is
to **not collect the contents in the first place** — a value that was never read
cannot leak through a redaction bug, a log line, or a stack trace.

This matters in Phase 2 specifically because the baseline snapshot reads file
contents off disk. A ``.env`` file that was already modified when a session began
would otherwise have its contents copied into TraceFlow's state directory.

The match is on **paths**, not on content patterns. Content scanning is a second
line of defence for files that were not excluded by path, and belongs with the
phase that renders diffs. Path exclusion is the primary guarantee.

Matching is deliberately conservative: a false positive costs one file's line
statistics (the file is still reported, just without its contents), while a false
negative costs a leaked credential.
"""

from __future__ import annotations

from fnmatch import fnmatch

#: Default sensitive-path patterns, applied to both the full repository-relative
#: path and the bare filename. A bare name with no separator (``.ssh``) also
#: matches any path segment, so it covers everything beneath that directory.
DEFAULT_SECRET_PATHS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.env",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "*.ppk",
    ".ssh",
    ".aws",
    ".gnupg",
    ".docker",
    ".kube",
    ".netrc",
    "_netrc",
    ".npmrc",
    ".pypirc",
    ".htpasswd",
    "credentials",
    "credentials.*",
    "secrets.*",
    "secret.*",
    "*.secrets",
)


def matches_secret_path(path: str, patterns: tuple[str, ...]) -> bool:
    """True when *path* is one TraceFlow must not read the contents of.

    A pattern is tested three ways: against the whole repository-relative path,
    against the bare filename, and — when it contains no separator — as a path
    segment, so ``.ssh`` covers ``.ssh/id_rsa`` without also matching ``my.sshfile``.

    Matching is case-insensitive on every platform. Git paths are case-sensitive,
    but the filesystems underneath are not always, and erring towards excluding a
    file costs only its line statistics.
    """
    normalised = path.replace("\\", "/").strip("/").lower()
    if not normalised:
        return False

    name = normalised.rsplit("/", 1)[-1]
    segments = f"/{normalised}/"

    for pattern in patterns:
        candidate = pattern.replace("\\", "/").strip("/").lower()
        if not candidate:
            continue
        if fnmatch(normalised, candidate) or fnmatch(name, candidate):
            return True
        if "/" not in candidate and f"/{candidate}/" in segments:
            return True

    return False
