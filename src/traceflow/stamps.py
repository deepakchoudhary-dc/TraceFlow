"""Timestamps and generated identifiers.

These live in their own module so that a caller which needs only a timestamp or a
name does not have to import the session store. Without this, the git evidence
modules would depend on the watcher package — the dependency graph pointing the
wrong way, which gets harder to unwind as later phases add more modules that need
the same two helpers.
"""

from __future__ import annotations

import secrets
from datetime import datetime


def now_iso(moment: datetime | None = None) -> str:
    """Render a local-time ISO 8601 timestamp with an explicit UTC offset."""
    return (moment or datetime.now().astimezone()).astimezone().isoformat(timespec="seconds")


def new_stamp_id(moment: datetime | None = None, suffix: str | None = None) -> str:
    """Build a sortable, readable identifier such as ``2026-09-25T14-40-12-a1b2c3``.

    The timestamp makes records sortable and legible in a directory listing; the
    short random suffix keeps two records created in the same second distinct.
    """
    stamp = (moment or datetime.now().astimezone()).strftime("%Y-%m-%dT%H-%M-%S")
    return f"{stamp}-{suffix or secrets.token_hex(3)}"
