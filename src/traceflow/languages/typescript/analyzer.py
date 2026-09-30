"""TypeScript and JavaScript source analysis (plan.md §66, Phase 10).

A **scanner-based analyzer**, not a compiler. It recognises the constructs that carry
dependency and symbol information — import statements, declarations, call expressions —
by matching the small set of syntactic shapes that express them, rather than parsing
TypeScript to completion.

That decision has a precise scope, stated here because plan.md §70 forbids presenting a
guess as knowledge. Recognised and resolved: ES modules (``import … from``, ``export …
from``, ``import()`), CommonJS ``require()`` with single or destructured bindings, ES
import attributes, JSX (treated as expression syntax), generic parameter lists,
decorators, arrow functions, ``async``, ``declare``, accessors, overload signatures.
Beyond the scanner's grammar: conditional types, declaration merging across files,
computed member names. The module-walk guarantees a symbol found anywhere is never
reported as removed even when its declaration head is one the classifier mis-reads.

CommonJS interop is kept because a large part of the watched ecosystem still uses it
(Express codebases, test runners, older configs) and because ``require()`` calls are
indistinguishable from ordinary calls without interop — the call scanner would see
them as noise and the graph would lose real edges.

Symbols come from **brace classification**, in one pass. Every ``{`` in code is
classified by scanning *backward* over its header — the parenthesised parameter list,
the name, the keyword or binding before it — and deciding what the brace opens: a
class, a function, a method, or an anonymous block. A block is pushed and forgotten; a
named scope records a symbol whose span closes at its matching ``}``. Working backward
from each brace rather than forward from keywords is what makes class members
robust: ``render() {`` inside a class body is a method whoever wrote it and whatever
modifiers precede it, because the brace is what asks the question.
"""

from __future__ import annotations

import re

from traceflow.blobs import digest_of
from traceflow.languages.base import (
    CallRef,
    ImportRef,
    ModuleAnalysis,
    Symbol,
    SymbolKind,
)
from traceflow.languages.typescript.scanner import SourceView

#: Bumped whenever the analysis output changes shape or meaning; part of the cache key.
ANALYZER_VERSION = "1"

_MAX_PARSE_ERROR_LENGTH = 300

_TS_EXTENSIONS = (".ts", ".tsx", ".mts", ".cts", ".jsx", ".mjs", ".cjs")

_IMPORT_RE = re.compile(r"\bimport\s+(type\s+)?([\s\S]*?)\s+from\s*(['\"])(?P<module>[^'\"]+)\3")
_SIDE_EFFECT_RE = re.compile(r"\bimport\s*(['\"])(?P<module>[^'\"]+)\1")
# `export … from '…'` re-exports. The clause is anchored to what an export clause can
# actually contain (names in braces, a star, a name), so the `from` this matches is the
# one the clause owns — an unanchored lazy match could skip across unrelated code and
# bind an export to a specifier that belongs to another statement.
_EXPORT_FROM_RE = re.compile(
    r"\bexport\s+(?:type\s+)?(?:\{[^}]*\}|\*(?:\s+as\s+[A-Za-z_$][\w$]*)?"
    r"|[A-Za-z_$][\w$]*)\s*from\s*(['\"])(?P<module>[^'\"]+)\1"
)
_REQUIRE_RE = re.compile(
    r"\b(?:const|let|var)\s+(?P<binding>[\w$]+|\{[^}]*\})"
    r"(?:\s*:\s*[^=;]+)?\s*=\s*require\s*\(\s*(['\"])(?P<module>[^'\"]+)\2\s*\)"
)
_BARE_REQUIRE_RE = re.compile(r"\brequire\s*\(\s*(['\"])(?P<module>[^'\"]+)\1\s*\)")
_DYNAMIC_IMPORT_RE = re.compile(r"\bimport\s*\(\s*(['\"])(?P<module>[^'\"]+)\1\s*\)")

# Words that name control flow rather than a callable: a name preceded by one of
# these is a keyword's operand, not a declaration.
_CONTROL_KEYWORDS = frozenset(
    {"if", "for", "while", "switch", "catch", "with", "return", "typeof", "delete", "void", "new"}
)

# Modifiers that may sit between a member's name and its declaration keyword.
_MEMBER_MODIFIERS = frozenset(
    {
        "public",
        "private",
        "protected",
        "static",
        "readonly",
        "abstract",
        "override",
        "async",
        "declare",
        "get",
        "set",
        "export",
        "default",
    }
)

# Words a binding name can never be: reading backward over ``const x = …`` must
# stop at keywords, or ``= (a) => …`` would bind to ``const``.
_BINDING_STOP_WORDS = frozenset(
    {
        "const",
        "let",
        "var",
        "return",
        "await",
        "yield",
        "case",
        "else",
        "do",
        "try",
        "function",
        "class",
        "import",
        "export",
    }
)


def _is_word_char(char: str) -> bool:
    return char.isalnum() or char in "$_#"


class TypeScriptAnalyzer:
    """Analyses TypeScript, TSX and JavaScript source with a dependency-free scanner."""

    name = "typescript"
    version = ANALYZER_VERSION
    extensions = _TS_EXTENSIONS

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
        if view.unclosed_template_nesting() > 0:
            # A ``${`` opened and never closed: the token stream from that point is
            # text the scanner cannot trust. Half-written is a normal state while an
            # agent types (plan.md §46), so this is recorded and the analysis stops
            # here rather than inventing symbols from a leaked template body.
            return ModuleAnalysis(
                path=path,
                digest=digest,
                module_name=module_name,
                parse_error="SyntaxError: unterminated template literal"[:_MAX_PARSE_ERROR_LENGTH],
            )

        symbols = _extract_symbols(view)
        return ModuleAnalysis(
            path=path,
            digest=digest,
            module_name=module_name,
            imports=_imports_from(view),
            symbols=symbols,
            calls=_calls_from(view),
        )


# --------------------------------------------------------------------------- imports


def _imports_from(view: SourceView) -> tuple[ImportRef, ...]:
    """Every import in the file: ES modules, re-exports, dynamic imports and CommonJS.

    Suppression comments are deliberately not consulted: ``@ts-ignore`` tells the
    compiler a line's types cannot be checked; it does not unmake the runtime fact
    that the import exists. An edge in the graph is a dependency that exists, and
    that is what the graph is for.

    Granularity is one ImportRef **per binding**, not per statement. A session that
    removes ``createElement`` from ``import { createElement, useState } from 'react'``
    is a real, reviewable fact; a per-statement record would report "nothing changed"
    for an edit that visibly removed a dependency.
    """
    text = view.text
    imports: list[ImportRef] = []

    for match in _IMPORT_RE.finditer(text):
        if not view.in_code(match.start()):
            continue
        module = match.group("module")
        clause = match.group(2) or ""
        line = view.line_of(match.start())
        if match.group(1) is not None:
            # `import type { … } from` — a compile-time-only dependency.
            imports.append(ImportRef(module=module, name=None, alias=None, level=0, line=line))
            continue
        if clause.strip():
            imports.extend(_clause_imports(module, clause, line))
        else:
            imports.append(ImportRef(module=module, name=None, alias=None, level=0, line=line))

    for match in _SIDE_EFFECT_RE.finditer(text):
        if view.in_code(match.start()):
            imports.append(
                ImportRef(
                    module=match.group("module"),
                    name=None,
                    alias=None,
                    level=0,
                    line=view.line_of(match.start()),
                )
            )

    for match in _EXPORT_FROM_RE.finditer(text):
        if view.in_code(match.start()):
            imports.append(
                ImportRef(
                    module=match.group("module"),
                    name=None,
                    alias=None,
                    level=0,
                    line=view.line_of(match.start()),
                )
            )

    for match in _DYNAMIC_IMPORT_RE.finditer(text):
        if view.in_code(match.start()):
            imports.append(
                ImportRef(
                    module=match.group("module"),
                    name=None,
                    alias=None,
                    level=0,
                    line=view.line_of(match.start()),
                )
            )

    for match in _REQUIRE_RE.finditer(text):
        if not view.in_code(match.start()):
            continue
        module = match.group("module")
        line = view.line_of(match.start())
        binding = match.group("binding")
        if binding.startswith("{"):
            for name, alias in _names_from_braces(binding):
                imports.append(ImportRef(module=module, name=name, alias=alias, level=0, line=line))
        else:
            imports.append(ImportRef(module=module, name=None, alias=binding, level=0, line=line))

    # A bare require with no binding at all — `require('./side-effect')`. A hit
    # inside a binding declaration's span is that declaration, already recorded.
    bound_spans = [match.span() for match in _REQUIRE_RE.finditer(text)]
    for match in _BARE_REQUIRE_RE.finditer(text):
        if not view.in_code(match.start()):
            continue
        if any(start <= match.start() < end for start, end in bound_spans):
            continue
        imports.append(
            ImportRef(
                module=match.group("module"),
                name=None,
                alias=None,
                level=0,
                line=view.line_of(match.start()),
            )
        )

    imports.sort(key=lambda item: (item.line, item.module, item.name or ""))
    return tuple(imports)


def _clause_imports(module: str, clause: str, line: int) -> list[ImportRef]:
    """Split an import clause into one ImportRef per binding.

    Handles default bindings, namespace bindings and named bindings in one clause:
    ``import React, { useState as use, type FC } from 'react'`` yields three records.
    """
    refs: list[ImportRef] = []
    for name, alias in _names_from_braces(clause):
        refs.append(ImportRef(module=module, name=name, alias=alias, level=0, line=line))

    stripped = _strip_braces(clause)
    namespace = re.search(r"\*\s+as\s+([A-Za-z_$][\w$]*)", stripped)
    if namespace:
        refs.append(
            ImportRef(module=module, name="*", alias=namespace.group(1), level=0, line=line)
        )

    without_namespace = re.sub(r"\*\s+as\s+[A-Za-z_$][\w$]*", "", stripped)
    default = re.match(r"\s*([A-Za-z_$][\w$]*)\s*,?", without_namespace)
    if default and default.group(1) not in ("type",):
        refs.append(ImportRef(module=module, name=None, alias=default.group(1), level=0, line=line))
    return refs


def _names_from_braces(clause: str) -> list[tuple[str, str | None]]:
    """``{ a as b, type c, d }`` -> ``[(a, b), (c, None), (d, None)]``.

    A clause with no braces — a bare default import like ``prisma`` — yields nothing:
    the default binding is extracted separately, and treating the bare name as a
    brace member would record the same import twice under different shapes.
    """
    if "{" not in clause:
        return []
    inner = clause[clause.index("{") :]
    if inner.endswith("}"):
        inner = inner[:-1]
    if inner.startswith("{"):
        inner = inner[1:]

    names: list[tuple[str, str | None]] = []
    for piece in inner.split(","):
        piece = re.sub(r"\btype\s+", "", piece.strip())
        # Inline member annotations: `{ status: 'active' as const }` shapes do not
        # appear in import clauses, but `{ a: string }`-looking garbage can when a
        # regex over-matches a type argument. Only plain identifiers are bindings.
        piece = piece.strip()
        if not piece:
            continue
        as_match = re.fullmatch(r"([A-Za-z_$][\w$]*)\s+as\s+([A-Za-z_$][\w$]*)", piece)
        if as_match:
            names.append((as_match.group(1), as_match.group(2)))
            continue
        single = re.fullmatch(r"([A-Za-z_$][\w$]*)", piece)
        if single:
            names.append((single.group(1), None))
    return names


def _strip_braces(clause: str) -> str:
    """The clause text outside any ``{…}`` group — the default and namespace part."""
    if "{" in clause and "}" in clause:
        open_index = clause.index("{")
        close_index = clause.rindex("}")
        if close_index > open_index:
            return clause[:open_index] + " " + clause[close_index + 1 :]
    return clause


# --------------------------------------------------------------------------- symbols


class _Record:
    """A mutable symbol under construction; finalised into a frozen Symbol."""

    __slots__ = ("body", "kind", "line_end", "line_start", "name", "qualified", "signature")

    def __init__(
        self,
        name: str,
        kind: SymbolKind,
        qualified: str,
        line_start: int,
        signature: str = "",
    ) -> None:
        self.name = name
        self.kind = kind
        self.qualified = qualified
        self.line_start = line_start
        self.line_end = line_start
        self.signature = signature
        self.body = ""


class _Scope:
    """One open brace: what it opened, and the symbol it is building, if any."""

    __slots__ = ("brace_index", "is_class", "qualified", "record")

    def __init__(
        self,
        brace_index: int,
        is_class: bool,
        qualified: str,
        record: _Record | None,
    ) -> None:
        self.brace_index = brace_index
        self.is_class = is_class
        self.qualified = qualified
        self.record = record


def _skip_space_back(view: SourceView, index: int, floor: int) -> int:
    """The last code, non-whitespace offset at or before *index*; ``floor`` when none."""
    while index > floor:
        if not view.in_code(index):
            index -= 1
            continue
        if not view.text[index].isspace():
            return index
        index -= 1
    return floor


def _skip_space_forward(view: SourceView, index: int) -> int:
    """The first code, non-whitespace offset at or after *index*; length when none."""
    length = view.length
    while index < length:
        if not view.in_code(index):
            index += 1
            continue
        if not view.text[index].isspace():
            return index
        index += 1
    return length


def _read_word_back(view: SourceView, end: int, floor: int) -> tuple[str, int]:
    """The identifier ending at (and including) offset *end*, and where it starts.

    Returns ``("", end + 1)`` when no identifier ends there.
    """
    if end <= floor or not view.in_code(end) or not _is_word_char(view.text[end]):
        return "", end + 1
    start = end
    while start > floor and view.in_code(start - 1) and _is_word_char(view.text[start - 1]):
        start -= 1
    return view.text[start : end + 1], start


def _match_paren_backward(view: SourceView, close: int, floor: int) -> int | None:
    """The ``(`` matching the ``)`` at *close*, or ``None``."""
    depth = 0
    index = close
    while index > floor:
        if view.in_code(index):
            char = view.text[index]
            if char == ")":
                depth += 1
            elif char == "(":
                depth -= 1
                if depth == 0:
                    return index
        index -= 1
    return None


def _match_angle_backward(view: SourceView, close: int, floor: int) -> int | None:
    """The ``<`` matching the ``>`` at *close*, or ``None`` when unbalanced.

    Angle brackets are ambiguous with comparison, so the caller only invokes this
    where a generic parameter list is plausible — immediately before a ``(`` or ``{``.
    """
    depth = 0
    index = close
    while index > floor:
        if view.in_code(index):
            char = view.text[index]
            if char == ">":
                depth += 1
            elif char == "<":
                depth -= 1
                if depth == 0:
                    return index
        index -= 1
    return None


def _match_paren_forward(view: SourceView, open_index: int) -> int | None:
    """The ``)`` matching the ``(`` at *open_index*, or ``None``."""
    depth = 0
    index = open_index
    length = view.length
    while index < length:
        if view.in_code(index):
            char = view.text[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return index
        index += 1
    return None


def _normalise_whitespace(text: str) -> str:
    return " ".join(text.split())


class _ScopeStack:
    """The open braces above the current position, and every symbol built so far."""

    def __init__(self) -> None:
        self.stack: list[_Scope] = []
        self.records: list[_Record] = []

    @property
    def enclosing(self) -> _Scope | None:
        return self.stack[-1] if self.stack else None

    @property
    def enclosing_is_class(self) -> bool:
        return bool(self.stack) and self.stack[-1].is_class

    @property
    def prefix(self) -> str:
        enclosing = self.enclosing
        return enclosing.qualified if enclosing is not None else ""

    def push(self, scope: _Scope) -> None:
        self.stack.append(scope)


def _classify_brace(view: SourceView, brace_index: int, scopes: _ScopeStack) -> _Scope:
    """Decide what the ``{`` at *brace_index* opens, by reading its header backward.

    The full decision tree lives in the module docstring's companion notes; the shape
    of it: read backward over an optional parameter list, then an optional generic
    list, then the name, then the keyword or binding character before the name — and
    let those tokens say whether this is a class, a function, a method, or a block
    that merely brackets an expression.
    """
    text = view.text
    enclosing = scopes.enclosing
    floor = enclosing.brace_index if enclosing is not None else -1
    prefix = scopes.prefix

    # ---- step 1: optional parenthesised parameter list immediately before the brace.
    last = _skip_space_back(view, brace_index - 1, floor)
    paren_open: int | None = None
    paren_close: int | None = None
    before = last
    if last > floor and text[last] == ")":
        opened = _match_paren_backward(view, last, floor)
        if opened is not None and opened > floor:
            paren_open, paren_close = opened, last
            before = _skip_space_back(view, opened - 1, floor)

    # ---- step 2: optional generic parameter list (`map<U>(…) {`, `class Foo<T> {`).
    generic_open: int | None = None
    if before > floor and text[before] == ">":
        opened = _match_angle_backward(view, before, floor)
        if opened is not None and opened > floor:
            generic_open = opened
            before = _skip_space_back(view, opened - 1, floor)

    # ---- step 3: the name, if a word ends at *before*.
    word, word_start = _read_word_back(view, before, floor)

    # ---- step 4: what precedes the name — the keyword or binding that decides.
    prev_end = word_start - 1 if word else before
    prev_pos = _skip_space_back(view, prev_end, floor)
    prev_word, prev_start = _read_word_back(view, prev_pos, floor)

    def _prev_prev() -> str:
        pos = _skip_space_back(view, prev_start - 1, floor)
        found, _ = _read_word_back(view, pos, floor)
        return found

    # Arrow with no parenthesised parameter: `x => {`.
    if word == "" and brace_index - 1 > floor:
        probe = _skip_space_back(view, brace_index - 1, floor)
        if probe - 1 > floor and text[probe] == ">" and text[probe - 1] == "=":
            # An arrow-function body. The binding — if the arrow is a declaration
            # at all — sits behind the parameter list (``const handler = (req, res)
            # => { … }``) or behind the lone parameter (``const f = x => { … }``).
            # Extracting it is what makes that shape a changed function rather
            # than an anonymous block, and modern TS is written in exactly it.
            binder, binder_start = "", brace_index
            # *probe* is the arrow's ``>``; the char under *joiner* is its ``=``.
            # What sits before the joiner decides the shape, and each shape finds
            # its binder one step further back:
            #   ``handler = (req, res) => {``  paren group, then the ``=``/``:``
            #                                  joiner, then the binder;
            #   ``handler: async (x) => {``    the same, with ``async`` in between;
            #   ``handler = x => {``           a lone parameter where a group would
            #                                  be — the binder is behind the ``=``
            #                                  that precedes the parameter.
            joiner = _skip_space_back(view, probe - 1, floor)
            if joiner > floor and text[joiner] in ("=", ":"):
                candidate = _skip_space_back(view, joiner - 1, floor)
                if candidate > floor and text[candidate] == ")":
                    arrow_paren = _match_paren_backward(view, candidate, floor)
                    if arrow_paren is not None and arrow_paren > floor:
                        candidate = _skip_space_back(view, arrow_paren - 1, floor)
                if text[candidate] in ("=", ":"):
                    binder, binder_start = _read_word_back(
                        view, _skip_space_back(view, candidate - 1, floor), floor
                    )
                else:
                    head, head_start = _read_word_back(view, candidate, floor)
                    if head == "async":
                        async_pos = _skip_space_back(view, head_start - 1, floor)
                        if async_pos > floor and text[async_pos] in ("=", ":"):
                            binder, binder_start = _read_word_back(
                                view, _skip_space_back(view, async_pos - 1, floor), floor
                            )
                    elif head and head not in _BINDING_STOP_WORDS:
                        # A lone parameter. The binder hides behind the ``=`` or
                        # ``:`` that precedes the parameter itself.
                        param_pos = _skip_space_back(view, head_start - 1, floor)
                        if param_pos > floor and text[param_pos] in ("=", ":"):
                            binder, binder_start = _read_word_back(
                                view, _skip_space_back(view, param_pos - 1, floor), floor
                            )
            if binder and binder not in _BINDING_STOP_WORDS:
                qualified = f"{prefix}.{binder}" if prefix else binder
                kind = SymbolKind.METHOD if scopes.enclosing_is_class else SymbolKind.FUNCTION
                record = _Record(
                    name=binder,
                    kind=kind,
                    qualified=qualified,
                    line_start=view.line_of(binder_start),
                )
                return _Scope(brace_index, is_class=False, qualified=qualified, record=record)
            return _Scope(brace_index, is_class=False, qualified=prefix, record=None)

    # A lone operator or keyword with no name: `try {`, `else {`, `= {`, `: {`.
    # Every arrow shape was decided above; what reaches here brackets an
    # expression or a statement, and brackets carry no declaration.
    if word == "":
        return _Scope(brace_index, is_class=False, qualified=prefix, record=None)

    # `class {` — anonymous class.
    if word == "class" and prev_word in ("", "default", "export"):
        return _Scope(brace_index, is_class=True, qualified=prefix, record=None)

    # `class Name {`, `interface Name {`, `enum Name {`, `namespace Name {`.
    if prev_word in ("class", "interface", "enum", "namespace", "module"):
        qualified = f"{prefix}.{word}" if prefix else word
        record = _Record(
            name=word,
            kind=SymbolKind.CLASS,
            qualified=qualified,
            line_start=view.line_of(word_start),
        )
        return _Scope(brace_index, is_class=True, qualified=qualified, record=record)

    # `class Foo extends Bar {`, `interface Foo extends Bar {`: the tokens closest
    # to the brace name the *superclass*; the declared name is one clause further
    # back, before `extends`/`implements`. Without this climb the class is misread
    # as an anonymous block and every one of its members is misattributed.
    if prev_word in ("extends", "implements"):
        declared_pos = _skip_space_back(view, prev_start - 1, floor)
        declared, declared_start = _read_word_back(view, declared_pos, floor)
        if declared and declared not in _BINDING_STOP_WORDS:
            qualified = f"{prefix}.{declared}" if prefix else declared
            record = _Record(
                name=declared,
                kind=SymbolKind.CLASS,
                qualified=qualified,
                line_start=view.line_of(declared_start),
            )
            return _Scope(brace_index, is_class=True, qualified=qualified, record=record)
        return _Scope(brace_index, is_class=True, qualified=prefix, record=None)

    # `type Name = { … }` — a type-alias body: scope for braces, not a symbol.
    if prev_word == "type":
        return _Scope(brace_index, is_class=False, qualified=prefix, record=None)

    # `function { … }` cannot happen; `foo(function () { … })` reaches here with
    # word == "function": an anonymous function expression.
    if word == "function":
        if prev_word in ("=", ":"):
            binder_pos = _skip_space_back(view, prev_start - 1, floor)
            binder, binder_start = _read_word_back(view, binder_pos, floor)
            if binder and binder not in _BINDING_STOP_WORDS:
                qualified = f"{prefix}.{binder}" if prefix else binder
                kind = SymbolKind.METHOD if scopes.enclosing_is_class else SymbolKind.FUNCTION
                record = _Record(
                    name=binder,
                    kind=kind,
                    qualified=qualified,
                    line_start=view.line_of(binder_start),
                )
                return _Scope(brace_index, is_class=False, qualified=qualified, record=record)
        return _Scope(brace_index, is_class=False, qualified=prefix, record=None)

    # `function name(…) {` — a named function (declaration or named expression).
    if prev_word == "function":
        qualified = f"{prefix}.{word}" if prefix else word
        record = _Record(
            name=word,
            kind=SymbolKind.FUNCTION,
            qualified=qualified,
            line_start=view.line_of(word_start),
            signature=_params_text(view, paren_open, paren_close),
        )
        return _Scope(brace_index, is_class=False, qualified=qualified, record=record)

    # Control flow: `if (…) {`, `for (…) {`, `switch (…) {`, `catch (…) {`.
    if word in _CONTROL_KEYWORDS or prev_word in _CONTROL_KEYWORDS:
        return _Scope(brace_index, is_class=False, qualified=prefix, record=None)

    # A plain name. Three ways it can be a declaration:
    if paren_open is not None or generic_open is not None or paren_close is not None:
        # (a) `name(…) {` / `name<T>(…) {` — a function or a class member.
        # (b) `= name(…) {`? no — that case has word as the binding already.
        # (c) modifier chains: `static async name(…) {`, `get name() {`.
        if prev_word in _MEMBER_MODIFIERS or prev_word in ("=", ":"):
            # Climb over modifiers to the binding context.
            climber = prev_word
            climber_start = prev_start
            binder = ""
            binder_start = word_start
            guard = 0
            while climber in _MEMBER_MODIFIERS and guard < 8:
                guard += 1
                pos = _skip_space_back(view, climber_start - 1, floor)
                climber, climber_start = _read_word_back(view, pos, floor)
                if climber not in _MEMBER_MODIFIERS and climber not in ("", "=", ":"):
                    # `get name` / `static name`: `name` was the member.
                    binder = ""
                    break
                if climber in ("=", ":"):
                    binder_pos = _skip_space_back(view, climber_start - 1, floor)
                    binder, binder_start = _read_word_back(view, binder_pos, floor)
                    if binder in _BINDING_STOP_WORDS:
                        binder = ""
                    break
            if binder and binder not in _BINDING_STOP_WORDS:
                # `name = async (…) => {`, `handler: function(…) {`.
                qualified = f"{prefix}.{binder}" if prefix else binder
                kind = SymbolKind.METHOD if scopes.enclosing_is_class else SymbolKind.FUNCTION
                record = _Record(
                    name=binder,
                    kind=kind,
                    qualified=qualified,
                    line_start=view.line_of(binder_start),
                    signature=_params_text(view, paren_open, paren_close),
                )
                return _Scope(brace_index, is_class=False, qualified=qualified, record=record)
            # `static async name(…) {` / `get value() {`: the member is *word*.
            kind = SymbolKind.METHOD if scopes.enclosing_is_class else SymbolKind.FUNCTION
            qualified = f"{prefix}.{word}" if prefix else word
            record = _Record(
                name=word,
                kind=kind,
                qualified=qualified,
                line_start=view.line_of(word_start),
                signature=_params_text(view, paren_open, paren_close),
            )
            return _Scope(
                brace_index,
                is_class=False,
                qualified=qualified,
                record=record,
            )

        # Plain `name(…) {`.
        kind = SymbolKind.METHOD if scopes.enclosing_is_class else SymbolKind.FUNCTION
        qualified = f"{prefix}.{word}" if prefix else word
        record = _Record(
            name=word,
            kind=kind,
            qualified=qualified,
            line_start=view.line_of(word_start),
            signature=_params_text(view, paren_open, paren_close),
        )
        return _Scope(brace_index, is_class=False, qualified=qualified, record=record)

    # A name with no parameter list directly before the brace: an object literal,
    # a destructuring remainder, JSX — brackets, not a declaration.
    return _Scope(brace_index, is_class=False, qualified=prefix, record=None)


def _params_text(view: SourceView, paren_open: int | None, paren_close: int | None) -> str:
    """The parameter list as written, whitespace-normalised; ``""`` when unknown."""
    if paren_open is None or paren_close is None:
        return ""
    return _normalise_whitespace(view.text[paren_open : paren_close + 1])


def _extract_symbols(view: SourceView) -> tuple[Symbol, ...]:
    """One backward-classifying walk over every code brace in the file."""
    text = view.text
    length = view.length
    scopes = _ScopeStack()

    index = 0
    while index < length:
        if not view.in_code(index):
            index += 1
            continue
        char = text[index]
        if char == "{":
            scopes.push(_classify_brace(view, index, scopes))
            index += 1
            continue
        if char == "}":
            if scopes.stack:
                scope = scopes.stack.pop()
                record = scope.record
                if record is not None:
                    record.line_end = view.line_of(index)
                    record.body = _normalise_whitespace(text[scope.brace_index + 1 : index])
                    scopes.records.append(record)
            index += 1
            continue
        index += 1

    # A file cut off mid-body — the normal state while an agent types — leaves
    # scopes open at end of input. Their declarations are still real: the class
    # was written before the editor lost it, and reporting nothing would tell the
    # user their half-file has no symbols in it. Each open record is closed at the
    # end of the text, with whatever body was written as its body.
    for scope in scopes.stack:
        record = scope.record
        if record is not None:
            record.line_end = view.line_of(length - 1) if length else record.line_start
            record.body = _normalise_whitespace(text[scope.brace_index + 1 :])
            scopes.records.append(record)

    return _finalise(scopes.records)


def _finalise(records: list[_Record]) -> tuple[Symbol, ...]:
    """Fold same-named records and freeze into Symbols with both fingerprints.

    Folding exists because a name can genuinely be declared more than once — a
    getter and a setter pair, two arms of an environment shim. The fingerprints of
    the folded symbol cover every definition, so a change to any of them is
    reported, and ``occurrences`` records that the fold happened.
    """
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
                signature_fingerprint=_fingerprint(
                    record.kind.value, record.name, record.signature
                ),
                body_fingerprint=_fingerprint(record.body),
                occurrences=1,
            )
        )
    symbols.sort(key=lambda item: (item.line_start, item.qualified_name))
    return tuple(symbols)


def _fingerprint(*parts: str) -> str:
    """A short, stable hash of the parts — how a symbol declares, what it contains."""
    import hashlib

    joined = "\x1f".join(parts)
    return hashlib.sha256(joined.encode("utf-8", "replace")).hexdigest()[:16]


# --------------------------------------------------------------------------- calls


def _calls_from(view: SourceView) -> tuple[CallRef, ...]:
    """Every call expression in code, as a dotted name and the line it appears on.

    A declaration is not a call: ``function process(…) {``, ``run(a) {`` inside a
    class body and ``interface Handler(…)``-shaped signatures are excluded by
    checking what follows the parameter list — a declaration is followed by a body
    or a type annotation, a call by anything else.
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
        if not (char.isalpha() or char in "$_"):
            index += 1
            continue
        word = view.identifier_at(index)
        if not word:
            index += 1
            continue
        end = index + len(word)

        if word in _CONTROL_KEYWORDS or word in (
            "function",
            "class",
            "interface",
            "enum",
            "namespace",
            "const",
            "let",
            "var",
            "else",
            "do",
            "try",
            "get",
            "set",
            "await",
            "yield",
        ):
            index = end
            continue

        following = _skip_space_forward(view, end)
        if following < length and text[following] == "(":
            close = _match_paren_forward(view, following)
            if close is not None:
                after = _skip_space_forward(view, close + 1)
                next_char = text[after] if after < length else ""
                # A body or type annotation follows the parameter list: declaration
                # or signature, not a call site. A ``)`` is deliberately *not* a
                # terminator — `outer(inner(1))` puts one after every nested call,
                # and skipping it would drop the calls that matter most. The
                # semicolon case is likewise exempted: `foo();` is the commonest
                # call form there is, and the price of keeping it is that an
                # overload signature (which ends in `;` inside a class or declare
                # block) may be recorded as a call. One benign row against a lost
                # impact finding is a trade worth making twice.
                if next_char in ("{", ":"):
                    index = end
                    continue
                name = _dotted_name_back(view, index)
                calls.append(CallRef(name=name, line=view.line_of(index)))
                # Continue just past the name, not past the argument list: calls
                # nest (`outer(inner(x))`), and skipping the arguments would drop
                # every call inside them.
                index = end
                continue
        index = end

    calls.sort(key=lambda item: (item.line, item.name))
    return tuple(calls)


def _dotted_name_back(view: SourceView, word_start: int) -> str:
    """The receiver chain ending at *word_start*: ``a.b.c`` from the ``c`` call site.

    Read *backward*, because a call is written ``receiver.method(…)`` and the
    method is the word the call scanner stopped on. A receiver that is itself a
    call — ``users.find().then(…)`` — is where reading stops, which is the same
    place Python's dotted-name rendering stops.
    """
    text = view.text
    first = view.identifier_at(word_start)
    if not first:
        return ""
    parts = [first]
    cursor = word_start
    while cursor > 0 and view.in_code(cursor - 1) and text[cursor - 1] == ".":
        end = cursor - 2
        if end < 0 or not view.in_code(end) or not _is_word_char(text[end]):
            break
        start = end
        while start > 0 and view.in_code(start - 1) and _is_word_char(text[start - 1]):
            start -= 1
        parts.append(text[start : end + 1])
        cursor = start
    parts.reverse()
    return ".".join(parts)
