"""Standard-library compatibility shims.

TraceFlow targets Python 3.10 because that is the version available on the
development machine. The only gap that matters to this project is ``tomllib``,
which became part of the standard library in 3.11. Rather than require a newer
interpreter, we fall back to the third-party ``tomli`` package, which is the
library ``tomllib`` was derived from.

Nothing else in the project needs a version shim.
"""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib

__all__ = ["tomllib"]
