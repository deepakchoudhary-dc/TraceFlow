"""The C-family analysis engine (plan.md §66): Go, Java, Rust and C# over one machine.

One pass, the same architecture as the TypeScript engine: every ``{`` in code is
classified by reading **backward** over its header — an optional parameter list, an
optional generic or result list, then the name, then the keyword wall before the name
— and the :class:`~traceflow.languages.cfamily.profile.LanguageProfile` supplies the
vocabulary that decides what each token means in this language. Nothing here names a
language outside a profile lookup.

Why backward from the brace, again: the four languages put an unbounded modifier wall
in front of every declaration — Go has none but Rust has ``pub async unsafe extern
" C" fn``, Java has seven-deep ``public static final`` stacks, C# adds ``required
partial``. Reading forward from a keyword would mean reimplementing each wall; reading
backward from the brace meets the *name* first, then climbs the wall one word at a
time until the declaration head appears — and the head is what the profile knows.

Symbols come out with qualified names (``Repo.Save``, ``Wrapper::persist``), both
fingerprints, and the span the diff needs. Imports come out as
:class:`~traceflow.languages.base.ImportRef` records at the analyzer's own per-statement
granularity — every form these languages declare is captured, and resolution to files
happens in :mod:`traceflow.languages.cfamily.graph`.
"""

from __future__ import annotations

from traceflow.blobs import digest_of
from traceflow.languages.base import CallRef, ImportRef, ModuleAnalysis, Symbol, SymbolKind
from traceflow.languages.cfamily.classifier import classify
from traceflow.languages.cfamily.profile import (
    LanguageProfile,
    csharp_profile,
    go_profile,
    java_profile,
    rust_profile,
)
from traceflow.languages.cfamily.scanner import SourceView
from traceflow.languages.textops import (
    fingerprint,
    match_paren_forward,
    normalise_whitespace,
    skip_space_forward,
)

#: Bumped whenever the analysis output changes shape or meaning; part of the cache key.
ANALYZER_VERSION = "2"

_MAX_PARSE_ERROR_LENGTH = 300

# Modifier words any profile accepts in a climb: declaration-adjacent tokens the
# wall may contain regardless of language. Per-language extras ride on the profile.
_UNIVERSAL_MODIFIERS = frozenset(
    {"const", "static", "async", "unsafe", "extern", "public", "private", "protected"}
)


class CFamilyAnalyzer:
    """One engine over the profile: ``CFamilyAnalyzer(go_profile())`` analyses Go."""

    def __init__(self, profile: LanguageProfile) -> None:
        self._profile = profile
        self.name = profile.name
        self.version = ANALYZER_VERSION
        self.extensions = profile.extensions

    @property
    def profile(self) -> LanguageProfile:
        return self._profile

    @property
    def cache_kind(self) -> str:
        """The cache namespace for this analyzer and version."""
        return f"{self.name}-{self.version}"

    def can_analyze(self, path: str) -> bool:
        return path.endswith(self.extensions)

    def analyze(self, path: str, source: bytes, module_name: str | None) -> ModuleAnalysis:
        digest = digest_of(source)

        try:
            text = source.decode("utf-8")
        except UnicodeDecodeError:
            return ModuleAnalysis(
                path=path,
                digest=digest,
                module_name=module_name,
                parse_error=(
                    "UnicodeDecodeError: source is not valid UTF-8"[:_MAX_PARSE_ERROR_LENGTH]
                ),
            )

        view = SourceView(text)
        symbols = _extract_symbols(view, self._profile)
        return ModuleAnalysis(
            path=path,
            digest=digest,
            module_name=module_name,
            imports=_imports_from(view, self._profile),
            symbols=symbols,
            calls=_calls_from(view, self._profile),
        )


# --------------------------------------------------------------------------- imports


def _imports_from(view: SourceView, profile: LanguageProfile) -> tuple[ImportRef, ...]:
    """Every import the language's line forms declare.

    Two shapes exist across the family. The *line* form — ``import x`` (Go),
    ``use a::b`` (Rust) — ends at its ``;``. The *block* form — Go's
    ``import ( … )`` — is a list of quoted paths, so its parenthesised body is
    read directly rather than as one statement; treating the block as a single
    statement would swallow half the file past it, which is what a naive
    ``;``-terminated scan did. Java/C# ``import a.b.c;`` are line statements.
    """
    imports: list[ImportRef] = []

    if profile.name == "go":
        imports.extend(_go_imports(view))
    else:
        for statement in _statements_beginning_with(view, profile.import_line_starts):
            line = view.line_of(statement[0])
            body = _normalise_statement(view, statement)
            # The statement's first word is the keyword itself (import/use/using);
            # the path is what follows it. Keeping the keyword would make every
            # module name start with it, and no resolver matches that.
            parts = body.split(None, 1)
            if parts and parts[0] in profile.import_line_starts:
                body = parts[1] if len(parts) > 1 else ""
            if profile.uses_colon_imports:
                path_text = body.split("::", 1)[-1] if body.startswith("::") else body
                path_text = path_text.rstrip(";").strip()
                path_text = path_text.split("{", 1)[0].strip().rstrip(";")
                module = path_text.replace("::", ".")
                alias = module.rsplit(".", 1)[-1] if module else None
                imports.append(ImportRef(module=module, name=None, alias=alias, level=0, line=line))
            else:
                path_text = body.rstrip(";").strip()
                quoted = path_text.startswith("'") and path_text.endswith("'")
                quoted = quoted or (path_text.startswith('"') and path_text.endswith('"'))
                if quoted:
                    path_text = path_text[1:-1]
                alias = None
                if " as " in path_text:
                    path_text, alias = path_text.rsplit(" as ", 1)
                    path_text, alias = path_text.strip(), alias.strip()
                module = path_text.replace("::", ".")
                imports.append(ImportRef(module=module, name=None, alias=alias, level=0, line=line))

    imports.sort(key=lambda item: (item.line, item.module))
    return tuple(imports)


def _go_imports(view: SourceView) -> tuple[ImportRef, ...]:
    """Go's import forms: single ``import "p"`` and the ``import ( ... )`` block.

    The paths are string literals, and the scanner masks string literals
    *including their quotes* -- so the walker never searches for a code-offset
    quote. It finds the keyword, takes the string spans the scanner recorded
    after it (one for the single form, every span up to the block's closing
    paren for the block form), and reads each span's inner content.
    """
    text = view.text
    imports: list[ImportRef] = []
    index = 0
    length = view.length
    while index < length:
        if not view.in_code(index):
            index += 1
            continue
        word = view.identifier_at(index)
        if word != "import":
            index = index + len(word) if word else index + 1
            continue
        after = index + len(word)
        if after < length and view.is_word_char(after):
            index = after
            continue
        line = view.line_of(index)
        cursor = skip_space_forward(view, after)
        if cursor < length and text[cursor] == "(":
            # Block form: every string span up to the closing paren is a path.
            depth = 0
            body_end = cursor
            walk = cursor
            while walk < length:
                if view.in_code(walk):
                    char = text[walk]
                    if char == "(":
                        depth += 1
                    elif char == ")":
                        depth -= 1
                        if depth == 0:
                            body_end = walk
                            break
                walk += 1
            for start, end in view.string_spans_between(cursor, body_end):
                imports.append(
                    ImportRef(
                        module=text[start:end],
                        name=None,
                        alias=None,
                        level=0,
                        line=view.line_of(start),
                    )
                )
            index = body_end + 1
            continue
        # Single form: the one string span right after the keyword.
        content = view.string_content_at(cursor)
        if content is not None:
            imports.append(
                ImportRef(
                    module=text[content[0] : content[1]],
                    name=None,
                    alias=None,
                    level=0,
                    line=line,
                )
            )
            index = content[1] + 1
            continue
        index = after
    return tuple(imports)


def _statements_beginning_with(view: SourceView, starts: tuple[str, ...]) -> list[tuple[int, int]]:
    """Offset pairs ``(start, end)`` of every statement whose first word is in *starts*.

    A statement ends at the first top-level ``;`` — Go, Java, Rust, C# all end
    simple statements with one, and the import forms this engine parses all do.
    """
    found: list[tuple[int, int]] = []
    index = 0
    length = view.length
    while index < length:
        if not view.in_code(index):
            index += 1
            continue
        word = view.identifier_at(index)
        if not word or word not in starts:
            index = max(index + 1, index + len(word) if word else index + 1)
            continue
        # Whole-word check: `imported` must not match `import`.
        after = index + len(word)
        if after < length and view.is_word_char(after):
            index = after
            continue
        end = index
        while end < length:
            if not view.in_code(end):
                if view.text[end] == "\n" and view.in_comment(end - 1):
                    pass
                end += 1
                continue
            if view.text[end] == ";":
                break
            end += 1
        found.append((index, min(end + 1, length)))
        index = end + 1
    return found


def _normalise_statement(view: SourceView, span: tuple[int, int]) -> str:
    """The statement's code characters as one whitespace-normalised string."""
    start, end = span
    pieces: list[str] = []
    for index in range(start, min(end, view.length)):
        if view.in_code(index):
            pieces.append(view.text[index])
        elif view.text[index] == "\n":
            pieces.append(" ")
    return normalise_whitespace("".join(pieces))


# --------------------------------------------------------------------------- symbols


class _Record:
    """A mutable symbol under construction; finalised into a frozen Symbol."""

    __slots__ = ("body", "kind", "line_end", "line_start", "name", "qualified", "signature")

    def __init__(
        self, name: str, kind: SymbolKind, qualified: str, line_start: int, signature: str = ""
    ) -> None:
        self.name = name
        self.kind = kind
        self.qualified = qualified
        self.line_start = line_start
        self.line_end = line_start
        self.signature = signature
        self.body = ""


class _Scope:
    """One open brace: its kind, its qualified prefix, and the symbol it builds."""

    __slots__ = ("brace_index", "is_type_scope", "qualified", "record")

    def __init__(
        self,
        brace_index: int,
        is_type_scope: bool,
        qualified: str,
        record: _Record | None,
    ) -> None:
        self.brace_index = brace_index
        self.is_type_scope = is_type_scope
        self.qualified = qualified
        self.record = record


class _ScopeStack:
    """The open braces above the current position, and every symbol built so far."""

    def __init__(self) -> None:
        self.stack: list[_Scope] = []
        self.records: list[_Record] = []

    @property
    def enclosing(self) -> _Scope | None:
        return self.stack[-1] if self.stack else None

    @property
    def prefix(self) -> str:
        enclosing = self.enclosing
        return enclosing.qualified if enclosing is not None else ""

    @property
    def in_type_scope(self) -> bool:
        return bool(self.stack) and self.stack[-1].is_type_scope

    def push(self, scope: _Scope) -> None:
        self.stack.append(scope)


_JOINER = "::"


def _classify_brace(
    view: SourceView,
    brace_index: int,
    scopes: _ScopeStack,
    profile: LanguageProfile,
    floor_override: int | None = None,
) -> _Scope:
    """The classifier's verdict, made into a scope."""
    enclosing = scopes.enclosing
    if floor_override is not None:
        floor = floor_override
    else:
        floor = enclosing.brace_index if enclosing is not None else -1
    prefix = scopes.prefix

    verdict = classify(view, brace_index, floor, profile, scopes.in_type_scope)

    if verdict.is_type_scope and verdict.name:
        qualified = f"{prefix}{_JOINER}{verdict.name}" if prefix else verdict.name
        record = _Record(
            name=verdict.name,
            kind=SymbolKind.CLASS,
            qualified=qualified,
            line_start=(
                view.line_of(verdict.name_offset)
                if verdict.name_offset >= 0
                else view.line_of(brace_index)
            ),
        )
        return _Scope(brace_index, is_type_scope=True, qualified=qualified, record=record)

    if verdict.is_type_scope:
        # `impl` opens method scope without declaring anything of its own.
        return _Scope(brace_index, is_type_scope=True, qualified=prefix, record=None)

    if verdict.name and verdict.kind in ("method", "function"):
        kind = SymbolKind.METHOD if verdict.kind == "method" else SymbolKind.FUNCTION
        # A Go receiver method is qualified by its receiver type — `Repo.Save` —
        # which is how Go programmers say the name and how a changed method's
        # callers are found. The enclosing prefix is for every other case.
        qualifier = verdict.receiver or prefix
        qualified = f"{qualifier}{_JOINER}{verdict.name}" if qualifier else verdict.name
        record = _Record(
            name=verdict.name,
            kind=kind,
            qualified=qualified,
            line_start=(
                view.line_of(verdict.name_offset)
                if verdict.name_offset >= 0
                else view.line_of(brace_index)
            ),
            signature=verdict.signature,
        )
        return _Scope(brace_index, is_type_scope=False, qualified=qualified, record=record)

    return _Scope(brace_index, is_type_scope=False, qualified=prefix, record=None)


def _params_text(view: SourceView, paren_open: int | None, paren_close: int | None) -> str:
    """The parameter list as written, whitespace-normalised; ``""`` when unknown."""
    if paren_open is None or paren_close is None:
        return ""
    return normalise_whitespace(view.text[paren_open : paren_close + 1])


def _extract_symbols(view: SourceView, profile: LanguageProfile) -> tuple[Symbol, ...]:
    """One backward-classifying walk over every code brace in the file.

    The floor handed to the classifier is the later of the enclosing scope's
    brace and the most recently *closed* brace. Starting from the enclosing brace
    alone would make every header contain all its completed siblings — and a
    Java method after a constructor would be named by the constructor's
    parameter list, a Go function after a type by the type's body. A header
    belongs to the statement(s) written since the last block ended; this bound
    is what makes that true.
    """
    text = view.text
    length = view.length
    scopes = _ScopeStack()
    last_close = -1

    index = 0
    while index < length:
        if not view.in_code(index):
            index += 1
            continue
        char = text[index]
        if char == "{":
            enclosing = scopes.enclosing
            scope_floor = enclosing.brace_index if enclosing is not None else -1
            scopes.push(_classify_brace(view, index, scopes, profile, max(scope_floor, last_close)))
            index += 1
            continue
        if char == "}":
            if scopes.stack:
                scope = scopes.stack.pop()
                record = scope.record
                if record is not None:
                    record.line_end = view.line_of(index)
                    record.body = normalise_whitespace(text[scope.brace_index + 1 : index])
                    scopes.records.append(record)
            last_close = index
            index += 1
            continue
        index += 1

    # Open scopes at end of file: a half-written file still owns the symbols it
    # managed to open (the same rule as the TypeScript engine).
    for scope in scopes.stack:
        record = scope.record
        if record is not None:
            record.line_end = view.line_of(length - 1) if length else record.line_start
            record.body = normalise_whitespace(text[scope.brace_index + 1 :])
            scopes.records.append(record)

    return _finalise(scopes.records)


def _finalise(records: list[_Record]) -> tuple[Symbol, ...]:
    """Fold same-named records and freeze into Symbols with both fingerprints."""
    folded: dict[str, _Record] = {}
    order: list[str] = []
    for record in records:
        existing = folded.get(record.qualified)
        if existing is None:
            folded[record.qualified] = record
            order.append(record.qualified)
            continue
        existing.line_start = min(existing.line_start, record.line_start)
        existing.line_end = max(existing.line_end, record.line_end)
        if record.signature and record.signature != existing.signature:
            existing.signature = f"{existing.signature} | {record.signature}"
        existing.body = f"{existing.body} {record.body}".strip()

    symbols: list[Symbol] = []
    for qualified in order:
        record = folded[qualified]
        symbols.append(
            Symbol(
                qualified_name=record.qualified,
                name=record.name,
                kind=record.kind,
                line_start=record.line_start,
                line_end=record.line_end,
                signature=record.signature,
                signature_fingerprint=fingerprint(record.kind.value, record.name, record.signature),
                body_fingerprint=fingerprint(record.body),
            )
        )
    symbols.sort(key=lambda item: (item.line_start, item.qualified_name))
    return tuple(symbols)


# --------------------------------------------------------------------------- calls


def _calls_from(view: SourceView, profile: LanguageProfile) -> tuple[CallRef, ...]:
    """Every call expression in code: `name(…)`, `receiver.method(…)`, `Type::new(…)`.

    Declaration and definition headers are excluded by the same terminator rule as
    the TypeScript engine — a body or a `where`/`throws`/`->` clause follows a
    declaration; a call is followed by anything else.
    """
    text = view.text
    length = view.length
    calls: list[CallRef] = []

    index = 0
    while index < length:
        if not view.in_code(index):
            index += 1
            continue
        char = text[index]
        if not (char.isalpha() or char == "_"):
            index += 1
            continue
        word = view.identifier_at(index)
        if not word:
            index += 1
            continue
        end = index + len(word)

        if word in profile.control_keywords or word in profile.stop_words:
            index = end
            continue

        following = skip_space_forward(view, end)
        if following < length and text[following] == "(":
            close = match_paren_forward(view, following)
            if close is not None:
                after = skip_space_forward(view, close + 1)
                if _follows_declaration_tail(view, after):
                    # A body, a result type or a throws clause follows: the header
                    # is a declaration, and its own name is not a call site.
                    index = end
                    continue
                name = _chain_back(view, index, profile)
                calls.append(CallRef(name=name, line=view.line_of(index)))
                index = end
                continue
        index = end

    calls.sort(key=lambda item: (item.line, item.name))
    return tuple(calls)


def _follows_declaration_tail(view: SourceView, index: int) -> bool:
    """True when what follows a closed paren group marks a declaration header.

    A call is followed by anything else; these languages follow a declaration's
    parameter list with a body (``{``), a result type (Rust's ``-> T {``) or a
    throws/where clause (``throws IOException {``, ``where T: Clone {``). The
    body marker alone misreads ``fn start() -> bool {`` as a call of ``start`` —
    its own declaration — so the tails are checked by shape, not by position.
    """
    text = view.text
    length = view.length
    if index >= length:
        return False
    if text[index] == "{":
        return True
    if text.startswith("->", index):
        return True
    word = view.identifier_at(index)
    return word in ("throws", "where")


def _chain_back(view: SourceView, word_start: int, profile: LanguageProfile) -> str:
    """The receiver chain ending at *word_start*: ``a.b.c`` or ``a::b::c``.

    Reading stops at a call boundary — ``users.find().then`` keeps ``then`` — the
    same place the Python and TypeScript chain renderers stop.
    """
    text = view.text
    first = view.identifier_at(word_start)
    if not first:
        return ""
    parts = [first]
    cursor = word_start
    separators = profile.call_separators
    while cursor > 0 and view.in_code(cursor - 1) and text[cursor - 1] in separators:
        end = cursor - 2
        if end < 0 or not view.in_code(end) or not view.is_word_char(end):
            break
        start = end
        while start > 0 and view.in_code(start - 1) and view.is_word_char(start - 1):
            start -= 1
        parts.append(text[start : end + 1])
        cursor = start
    parts.reverse()
    return _JOINER.join(parts) if "::" in separators else ".".join(parts)


__all__ = [
    "ANALYZER_VERSION",
    "CFamilyAnalyzer",
    "csharp_profile",
    "go_profile",
    "java_profile",
    "rust_profile",
]
