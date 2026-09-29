"""The language-analyzer contract.

plan.md §6 requires a language adapter interface rather than pretending one parser
handles every language. This module defines that interface and the models that cross
it, so a second language can be added without touching the analysis pipeline.

The models are deliberately language-neutral. ``Symbol`` knows a qualified name, a
kind, a line range and two fingerprints — it does not know what a decorator or a base
class *means* in any particular language, only that the analyzer reported them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Protocol


class SymbolKind(str, Enum):
    """What kind of thing a symbol is."""

    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


class SymbolChangeKind(str, Enum):
    """How a symbol changed between two versions of a file.

    The split between a signature change and a body change is the point of the whole
    module. A signature change alters how the symbol must be *called*, so every caller
    is affected; a body change alters what it *does*, so callers may be affected
    without any of them needing to be edited. Conflating the two throws away the most
    useful distinction available from static analysis.
    """

    ADDED = "added"
    REMOVED = "removed"
    SIGNATURE_CHANGED = "signature_changed"
    BODY_CHANGED = "body_changed"


@dataclass(frozen=True)
class ImportRef:
    """One import statement."""

    module: str
    """The dotted module being imported from. Empty for ``from . import x``."""

    name: str | None
    """The imported name, or ``None`` for a plain ``import x``."""

    alias: str | None
    level: int
    """Zero for an absolute import, otherwise the number of leading dots."""

    line: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CallRef:
    """One call expression, as written in the source."""

    name: str
    """The dotted name at the call site, e.g. ``service.authenticate``."""

    line: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Symbol:
    """A class, function, or method."""

    qualified_name: str
    """Name within its module, including the enclosing class, e.g. ``Service.login``."""

    name: str
    kind: SymbolKind
    line_start: int
    line_end: int
    signature: str = ""
    """The declaration rendered canonically, e.g. ``(user, password, mfa=False)``.

    Stored alongside the fingerprint so a reported signature change can actually be
    read. "The signature changed" is not evidence on its own (plan.md §33); showing
    what it changed from and to is."""

    signature_fingerprint: str = ""
    """Hash of how the symbol is declared: name, parameters, annotations, decorators."""

    body_fingerprint: str = ""
    """Hash of what the symbol contains, excluding nested definitions."""

    parent: str | None = None
    decorators: tuple[str, ...] = ()
    bases: tuple[str, ...] = ()

    occurrences: int = 1
    """How many definitions in the file share this qualified name.

    More than one when a name is defined in more than one place — the two arms of a version
    shim, for instance. The fingerprints cover every definition, so a change to any of them
    is reported, and this records that the fold happened rather than leaving the reader to
    assume the file defines the name once.
    """

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["decorators"] = list(self.decorators)
        payload["bases"] = list(self.bases)
        return payload


@dataclass(frozen=True)
class ModuleAnalysis:
    """Everything the analyzer could determine about one file."""

    path: str
    digest: str
    module_name: str | None
    imports: tuple[ImportRef, ...] = ()
    symbols: tuple[Symbol, ...] = ()
    calls: tuple[CallRef, ...] = ()
    parse_error: str | None = None
    """Set when the file could not be parsed. Malformed source is expected (plan.md
    §46) and must degrade to "nothing known", never to an exception."""

    def symbol(self, qualified_name: str) -> Symbol | None:
        for candidate in self.symbols:
            if candidate.qualified_name == qualified_name:
                return candidate
        return None

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "digest": self.digest,
            "module_name": self.module_name,
            "imports": [item.to_json() for item in self.imports],
            "symbols": [item.to_json() for item in self.symbols],
            "calls": [item.to_json() for item in self.calls],
            "parse_error": self.parse_error,
        }


@dataclass(frozen=True)
class SymbolChange:
    """One symbol's change between two versions of a file."""

    qualified_name: str
    kind: SymbolKind
    change: SymbolChangeKind
    line_start: int | None = None
    line_end: int | None = None

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["change"] = self.change.value
        return payload


@dataclass(frozen=True)
class ModuleSymbolChanges:
    """What changed inside one module during a session."""

    path: str
    before_digest: str | None
    after_digest: str
    changes: tuple[SymbolChange, ...] = ()
    imports_added: tuple[str, ...] = ()
    imports_removed: tuple[str, ...] = ()
    parse_error: str | None = None

    @property
    def has_changes(self) -> bool:
        return bool(self.changes or self.imports_added or self.imports_removed)

    @property
    def signature_changes(self) -> tuple[SymbolChange, ...]:
        return self.by_kind(SymbolChangeKind.SIGNATURE_CHANGED)

    def by_kind(self, change_kind: SymbolChangeKind) -> tuple[SymbolChange, ...]:
        return tuple(item for item in self.changes if item.change is change_kind)

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
            "changes": [item.to_json() for item in self.changes],
            "imports_added": list(self.imports_added),
            "imports_removed": list(self.imports_removed),
            "parse_error": self.parse_error,
        }


class LanguageAnalyzer(Protocol):
    """A parser for one language (plan.md §6)."""

    name: str
    extensions: tuple[str, ...]
    version: str

    def can_analyze(self, path: str) -> bool:
        """True when this analyzer handles *path*."""
        ...

    def analyze(self, path: str, source: bytes, module_name: str | None) -> ModuleAnalysis:
        """Parse *source* and describe it. Must never raise on malformed input."""
        ...


def render_import(reference: ImportRef) -> str:
    """Render an import as a single comparable string.

    Used to diff import statements, so it has to be stable and unambiguous rather than
    pretty. The subtlety is that the leading dots already stand for "this package", so
    ``from . import x`` is ``.x`` and not ``..x``:

    ============================  ==================
    Source                        Rendered
    ============================  ==================
    ``import os``                 ``os``
    ``from a import b``           ``a.b``
    ``from . import x``           ``.x``
    ``from .mod import z``        ``.mod.z``
    ``from .. import y``          ``..y``
    ``from ..pkg import z``       ``..pkg.z``
    ============================  ==================
    """
    dots = "." * reference.level
    if reference.name is None:
        return f"{dots}{reference.module}"

    prefix = f"{reference.module}." if reference.module else ""
    return f"{dots}{prefix}{reference.name}"


def _import_set(analysis: ModuleAnalysis) -> set[str]:
    return {render_import(item) for item in analysis.imports}


def diff_module_analysis(
    before: ModuleAnalysis | None, after: ModuleAnalysis
) -> ModuleSymbolChanges:
    """Compare two versions of the same module.

    A signature change takes precedence over a body change: when a symbol's declaration
    changed, that is the fact a caller needs, and reporting "body changed" as well would
    bury it.
    """
    if before is None:
        added = tuple(
            SymbolChange(
                qualified_name=symbol.qualified_name,
                kind=symbol.kind,
                change=SymbolChangeKind.ADDED,
                line_start=symbol.line_start,
                line_end=symbol.line_end,
            )
            for symbol in after.symbols
        )
        return ModuleSymbolChanges(
            path=after.path,
            before_digest=None,
            after_digest=after.digest,
            changes=added,
            imports_added=tuple(sorted(_import_set(after))),
            parse_error=after.parse_error,
        )

    before_symbols = {symbol.qualified_name: symbol for symbol in before.symbols}
    after_symbols = {symbol.qualified_name: symbol for symbol in after.symbols}

    changes: list[SymbolChange] = []
    for name, symbol in after_symbols.items():
        previous = before_symbols.get(name)
        if previous is None:
            changes.append(
                SymbolChange(
                    qualified_name=name,
                    kind=symbol.kind,
                    change=SymbolChangeKind.ADDED,
                    line_start=symbol.line_start,
                    line_end=symbol.line_end,
                )
            )
        elif previous.signature_fingerprint != symbol.signature_fingerprint:
            changes.append(
                SymbolChange(
                    qualified_name=name,
                    kind=symbol.kind,
                    change=SymbolChangeKind.SIGNATURE_CHANGED,
                    line_start=symbol.line_start,
                    line_end=symbol.line_end,
                )
            )
        elif previous.body_fingerprint != symbol.body_fingerprint:
            changes.append(
                SymbolChange(
                    qualified_name=name,
                    kind=symbol.kind,
                    change=SymbolChangeKind.BODY_CHANGED,
                    line_start=symbol.line_start,
                    line_end=symbol.line_end,
                )
            )

    for name, symbol in before_symbols.items():
        if name not in after_symbols:
            changes.append(
                SymbolChange(
                    qualified_name=name,
                    kind=symbol.kind,
                    change=SymbolChangeKind.REMOVED,
                )
            )

    before_imports = _import_set(before)
    after_imports = _import_set(after)

    changes.sort(key=lambda item: (item.line_start or 0, item.qualified_name))

    return ModuleSymbolChanges(
        path=after.path,
        before_digest=before.digest,
        after_digest=after.digest,
        changes=tuple(changes),
        imports_added=tuple(sorted(after_imports - before_imports)),
        imports_removed=tuple(sorted(before_imports - after_imports)),
        parse_error=after.parse_error or before.parse_error,
    )


# --------------------------------------------------------------------------- cache round-trip


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _number(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _text_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def module_analysis_from_json(
    payload: dict[str, Any], *, path: str, module_name: str | None
) -> ModuleAnalysis:
    """Rebuild a :class:`ModuleAnalysis` from a cached payload.

    *path* and *module_name* come from the caller rather than the payload: the cache is
    content-addressed, so identical content at two paths shares one entry and the
    stored path would be wrong for whichever caller read it second.

    Raises ``ValueError`` when the payload is not shaped like an analysis, so a damaged
    entry can be treated as a miss instead of being trusted.
    """
    raw_symbols = payload.get("symbols")
    raw_imports = payload.get("imports")
    raw_calls = payload.get("calls")
    if not isinstance(raw_symbols, list):
        raise ValueError("payload has no symbol list")
    if not isinstance(raw_imports, list):
        raise ValueError("payload has no import list")
    if not isinstance(raw_calls, list):
        raise ValueError("payload has no call list")

    symbols = tuple(
        Symbol(
            qualified_name=_text(item.get("qualified_name")),
            name=_text(item.get("name")),
            kind=SymbolKind(_text(item.get("kind"))),
            line_start=_number(item.get("line_start")),
            line_end=_number(item.get("line_end")),
            signature=_text(item.get("signature")),
            signature_fingerprint=_text(item.get("signature_fingerprint")),
            body_fingerprint=_text(item.get("body_fingerprint")),
            parent=_optional_text(item.get("parent")),
            decorators=_text_tuple(item.get("decorators")),
            bases=_text_tuple(item.get("bases")),
            occurrences=_number(item.get("occurrences")) or 1,
        )
        for item in raw_symbols
        if isinstance(item, dict)
    )

    imports = tuple(
        ImportRef(
            module=_text(item.get("module")),
            name=_optional_text(item.get("name")),
            alias=_optional_text(item.get("alias")),
            level=_number(item.get("level")),
            line=_number(item.get("line")),
        )
        for item in raw_imports
        if isinstance(item, dict)
    )

    calls = tuple(
        CallRef(name=_text(item.get("name")), line=_number(item.get("line")))
        for item in raw_calls
        if isinstance(item, dict)
    )

    return ModuleAnalysis(
        path=path,
        digest=_text(payload.get("digest")),
        module_name=module_name,
        imports=imports,
        symbols=symbols,
        calls=calls,
        parse_error=_optional_text(payload.get("parse_error")),
    )
