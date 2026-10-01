"""Classifying a C-family brace by its header (plan.md §66).

The rules live here rather than in the engine so the engine stays a walk and the
language knowledge stays a set of decisions. Every decision is annotated with the
construct that forced it, because each one was written for a real shape and the
shape is its justification.

The method is **forward tokenisation with group awareness**: the header — every
code character between the previous brace and this one — is tokenised with its
paren and angle groups recorded as extents, and decisions read words *with their
offsets*. That is what lets ``impl<T> Foo<T>`` name ``Foo`` (skip the generic
groups), ``func (r *Repo) Save`` name ``Save`` (the word between two paren groups),
and ``Runnable r = () -> {`` stay anonymous (``=`` outside any paren group).

False positives are the enemy here, not false negatives: a block misreported as a
method puts a phantom symbol into every diff, while a method reported as nothing
merely goes unanalysed. Every rule prefers "not a declaration" when unsure.
"""

from __future__ import annotations

from dataclasses import dataclass

from traceflow.languages.cfamily.profile import LanguageProfile
from traceflow.languages.cfamily.scanner import SourceView
from traceflow.languages.textops import normalise_whitespace


@dataclass(frozen=True)
class Verdict:
    """What the brace opens, and the symbol it declares, if any."""

    is_type_scope: bool
    name: str | None = None
    kind: str | None = None  # "class" | "method" | "function"
    signature: str = ""
    name_offset: int = -1
    """Where the declared name begins — the symbol's reported line comes from it."""

    receiver: str = ""
    """Go only: the receiver type a method is declared on (``Repo`` from
    ``func (r *Repo) Save``). The engine qualifies the method under it, which is
    how Go programmers say the name: ``Repo.Save``."""


@dataclass(frozen=True)
class _Header:
    """The tokenised header: words with offsets, plus the groups between them."""

    words: tuple[str, ...]
    offsets: tuple[int, ...]
    """Each word's start offset, parallel to *words*."""

    paren_groups: tuple[tuple[int, int, int], ...]
    """``(open, close, depth_free)`` — simply ``(open, close)`` plus padding."""

    angle_groups: tuple[tuple[int, int], ...]
    """Every top-level ``< … >`` group in the header."""

    end: int
    """The brace offset this header belongs to."""


def _tokenise(view: SourceView, brace_index: int, floor: int) -> _Header:
    text = view.text
    words: list[str] = []
    offsets: list[int] = []
    parens: list[tuple[int, int, int]] = []
    angles: list[tuple[int, int]] = []

    index = floor + 1
    while index < brace_index:
        if not view.in_code(index):
            index += 1
            continue
        char = text[index]
        if char.isalpha() or char == "_":
            end = index
            while (
                end < brace_index
                and view.in_code(end)
                and (text[end].isalnum() or text[end] == "_")
            ):
                end += 1
            words.append(text[index:end])
            offsets.append(index)
            index = end
            continue
        if char == "(":
            depth = 0
            cursor = index
            while cursor < brace_index:
                if view.in_code(cursor):
                    if text[cursor] == "(":
                        depth += 1
                    elif text[cursor] == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    elif text[cursor].isalpha() or text[cursor] == "_":
                        # Group contents are tokenised too: Go's receiver type
                        # lives *inside* the first paren group (`r *Repo`), and
                        # the classifier reads it from there. The words are
                        # marked covered, so they never reach the top-level
                        # skeleton — they exist only for the receiver rule.
                        word_end = cursor
                        while (
                            word_end < brace_index
                            and view.in_code(word_end)
                            and (text[word_end].isalnum() or text[word_end] == "_")
                        ):
                            word_end += 1
                        words.append(text[cursor:word_end])
                        offsets.append(cursor)
                        cursor = word_end
                        continue
                cursor += 1
            if depth == 0 and cursor < brace_index:
                parens.append((index, cursor, 0))
                index = cursor + 1
                continue
            index += 1
            continue
        if char == "<":
            depth = 0
            cursor = index
            while cursor < brace_index:
                if view.in_code(cursor):
                    if text[cursor] == "<":
                        depth += 1
                    elif text[cursor] == ">":
                        depth -= 1
                        if depth == 0:
                            break
                cursor += 1
            if depth == 0 and cursor < brace_index:
                angles.append((index, cursor))
                index = cursor + 1
                continue
            index += 1
            continue
        index += 1

    return _Header(
        words=tuple(words),
        offsets=tuple(offsets),
        paren_groups=tuple(parens),
        angle_groups=tuple(angles),
        end=brace_index,
    )


def _in_group(offset: int, groups: tuple[tuple[int, int], ...]) -> bool:
    return any(start < offset < close for start, close in groups)


def _covered(offset: int, header: _Header) -> bool:
    """True when the word at *offset* sits inside a paren or angle group."""
    parens: tuple[tuple[int, int], ...] = tuple(
        (open_, close) for open_, close, _ in header.paren_groups
    )
    return _in_group(offset, parens) or _in_group(offset, header.angle_groups)


def _top_level_words(header: _Header) -> tuple[str, ...]:
    """Words outside every group — the declaration skeleton."""
    return tuple(
        word
        for word, offset in zip(header.words, header.offsets, strict=True)
        if not _covered(offset, header)
    )


def _has_top_level_equals(view: SourceView, header: _Header) -> bool:
    """True when ``=`` appears outside every group — an assignment, not a declaration.

    ``Runnable r = () -> {`` and ``var handler = (req) => {`` are bindings of a
    function value; the function itself is anonymous. A method header never
    contains ``=``, so this one character separates the two shapes — except inside
    paren groups, where Java annotations (``@Test(timeout = 5)``) legitimately
    carry equals signs in front of a perfectly good method declaration.
    """
    text = view.text
    index = header.end - 1
    floor = 0 if not header.words else header.offsets[0]
    while index > floor:
        if not view.in_code(index):
            index -= 1
            continue
        if text[index] == "=" and not _in_group(index, header.angle_groups):
            parens: tuple[tuple[int, int], ...] = tuple(
                (open_, close) for open_, close, _ in header.paren_groups
            )
            if not _in_group(index, parens):
                return True
        index -= 1
    return False


def classify(
    view: SourceView,
    brace_index: int,
    floor: int,
    profile: LanguageProfile,
    in_type_scope: bool,
) -> Verdict:
    """Decide what the ``{`` at *brace_index* opens, in *profile*'s terms."""
    header = _tokenise(view, brace_index, floor)
    top = _top_level_words(header)

    if not top:
        return Verdict(is_type_scope=False)

    if profile.name == "go":
        return _classify_go(view, header, top, in_type_scope)
    if profile.name == "rust":
        return _classify_rust(view, header, top, in_type_scope)
    return _classify_java_like(view, header, top, profile, in_type_scope)


# --------------------------------------------------------------------------- Go


def _classify_go(
    view: SourceView,
    header: _Header,
    top: tuple[str, ...],
    in_type_scope: bool,
) -> Verdict:
    # A file's first declaration follows the ``package x`` clause and possibly
    # ``import`` statements, and a header read from the file start therefore
    # begins with them. They are context, not part of the declaration — and the
    # rules below read position-sensitive words (the name is the word *after*
    # ``func``) — so the clauses are dropped before they run. Clauses can only
    # open a header, never sit inside one: they have no brace, and a later
    # declaration's floor starts after the previous block. Each clause's own
    # words (the package name, an import alias) are dropped with it; the next
    # declaration keyword is where they stop, which is also what keeps a quoted
    # import — whose path contributes no word at all — from swallowing ``func``.
    words: list[str] = list(header.words)
    offsets: list[int] = list(header.offsets)
    while words and words[0] in ("package", "import"):
        words.pop(0)
        offsets.pop(0)
        while words and words[0] not in ("func", "type", "var", "const", "import", "package"):
            words.pop(0)
            offsets.pop(0)
    if len(words) != len(header.words):
        header = _Header(
            words=tuple(words),
            offsets=tuple(offsets),
            paren_groups=header.paren_groups,
            angle_groups=header.angle_groups,
            end=header.end,
        )
        top = _top_level_words(header)

    # `func (r *Repo) Save(ctx, u) error {` — the name lives between the first
    # and second paren groups; that placement is Go's receiver syntax and nothing
    # else in the language produces it. The receiver's type (the last word inside
    # the first group, `Repo` from `r *Repo`) is what the method is qualified
    # under — Verdict carries it in `receiver` and the engine joins with `::`.
    if top and top[0] == "func" and len(header.paren_groups) >= 2:
        first_open, first_close, _ = header.paren_groups[0]
        receiver_words = [
            word
            for word, offset in zip(header.words, header.offsets, strict=True)
            if first_open < offset < first_close
        ]
        receiver = receiver_words[-1] if receiver_words else ""
        index = first_close + 1
        while index < header.end:
            if view.in_code(index) and (view.text[index].isalpha() or view.text[index] == "_"):
                name, _end = _word_at(view, index)
                if name:
                    return Verdict(
                        is_type_scope=False,
                        name=name,
                        kind="method",
                        signature=_last_group_text(view, header),
                        name_offset=index,
                        receiver=receiver,
                    )
            index += 1
        return Verdict(is_type_scope=False)

    # `func Save(ctx, u) error {` — the name follows the keyword in the *full*
    # word list. The top-level list would hand back the return type (`*Repo`):
    # everything after the parameter group is top-level, so taking the name from
    # there named half the file's functions after their return types, and those
    # symbols then folded into the type of the same name.
    if top and top[0] == "func" and len(header.words) >= 2:
        return Verdict(
            is_type_scope=False,
            name=header.words[1],
            kind="method" if in_type_scope else "function",
            signature=_last_group_text(view, header),
            name_offset=header.offsets[1] if len(header.offsets) > 1 else -1,
        )

    # `type Name struct {` / `type Name interface {` — the type keyword is the
    # last word; the declared name is right before it. `type ( Repo struct {`
    # inside a grouped declaration reaches the same rule through the same shape.
    if len(top) >= 3 and top[-1] in ("struct", "interface"):
        return Verdict(is_type_scope=True, name=top[-2], kind="class", name_offset=-1)

    # `if … {`, `for … {`, `switch … {`, composite literals, var blocks: brackets.
    return Verdict(is_type_scope=False)


# --------------------------------------------------------------------------- Rust


def _classify_rust(
    view: SourceView,
    header: _Header,
    top: tuple[str, ...],
    in_type_scope: bool,
) -> Verdict:
    # `impl Display for Wrapper {` / `impl<T> Foo<T> {` — opens method scope. The
    # declared type is the first top-level word after `impl`, and when `for`
    # follows it, the type is the next top-level word after that. Reading the
    # *top-level* words is what makes the generic forms work: `impl<T> Foo<T>`
    # contributes `impl Foo` once the generic contents are dropped.
    if top[0] == "impl":
        rest = top[1:]
        if rest and rest[0] == "for" and len(rest) >= 2:
            return Verdict(is_type_scope=True, name=rest[1])
        if rest:
            return Verdict(is_type_scope=True, name=rest[0])
        return Verdict(is_type_scope=True)

    # `fn name(…) … {` — the name immediately after `fn`, wherever the modifier
    # wall put it: `pub async unsafe extern "C" fn save`.
    if "fn" in top:
        position = top.index("fn") + 1
        if position < len(top):
            return Verdict(
                is_type_scope=False,
                name=top[position],
                kind="method" if in_type_scope else "function",
                signature=_last_group_text(view, header),
            )
        return Verdict(is_type_scope=False)

    # `struct Name {` / `pub enum Name {` / `trait Name {` — the name follows the
    # keyword; generics after it (`Name<T>`) are not top-level words.
    for position, word in enumerate(top):
        if word in ("struct", "enum", "trait", "union") and position + 1 < len(top):
            return Verdict(is_type_scope=True, name=top[position + 1], kind="class")
    if top[0] == "mod":
        return Verdict(is_type_scope=False)

    # `match x {`, `if let … {`, `for x in y {`, match arms (`Pat => {`),
    # closures (`|x| {`), blocks: brackets, all of them.
    return Verdict(is_type_scope=False)


# --------------------------------------------------------------------------- Java / C#


def _classify_java_like(
    view: SourceView,
    header: _Header,
    top: tuple[str, ...],
    profile: LanguageProfile,
    in_type_scope: bool,
) -> Verdict:
    # An assignment of a lambda or anonymous class is a binding, not a
    # declaration: `Runnable r = () -> {`, `var h = (req) => {`. Checked before
    # anything else because `new Foo() {` satisfies every method shape otherwise.
    if _has_top_level_equals(view, header):
        return Verdict(is_type_scope=False)
    if "new" in top:
        return Verdict(is_type_scope=False)

    # `class Foo extends Bar {` / `interface Foo … {` / `enum Foo {` /
    # `record Foo(…) {` / `static class Builder {`: the name always *follows*
    # the keyword, wherever the modifier wall put the keyword.
    for position, word in enumerate(top):
        if word in profile.type_keywords and position + 1 < len(top):
            return Verdict(is_type_scope=True, name=top[position + 1], kind="class")

    # Control flow opens with the keyword: `if (…) {`, `for (User u : users) {`,
    # `while (…) {`, `switch (…) {`, `catch (…) {`, `synchronized (lock) {`.
    if top[0] in profile.control_keywords:
        return Verdict(is_type_scope=False)
    # A lone modifier before the brace — `static {` initialiser blocks.
    if len(top) == 1 and top[0] in profile.modifier_words:
        return Verdict(is_type_scope=False)

    # A method: a paren group closes the header, and the word just before it is
    # the declared name. `public Set<String> keySet() {` contributes
    # `public keySet` once generics drop out; `public boolean isEmpty() {`
    # contributes `public isEmpty`; the constructor `public UserService(Store s) {`
    # contributes `public UserService`. All are methods by the same rule.
    if header.paren_groups and len(top) >= 2:
        last_close = header.paren_groups[-1][1]
        # The name is the last word before the final paren group.
        name = ""
        name_offset = -1
        for word, offset in zip(reversed(header.words), reversed(header.offsets), strict=True):
            if offset < last_close and not _covered(offset, header):
                name = word
                name_offset = offset
                break
        if name and name not in profile.control_keywords and name not in profile.stop_words:
            return Verdict(
                is_type_scope=False,
                name=name,
                kind="method" if in_type_scope else "function",
                signature=_last_group_text(view, header),
                name_offset=name_offset,
            )

    # `else {`, `do {`, try-with-resources remainders, array initialisers.
    return Verdict(is_type_scope=False)


# --------------------------------------------------------------------------- helpers


def _word_at(view: SourceView, index: int) -> tuple[str, int]:
    text = view.text
    end = index
    while end < view.length and view.in_code(end) and (text[end].isalnum() or text[end] == "_"):
        end += 1
    return text[index:end], end


def _last_group_text(view: SourceView, header: _Header) -> str:
    """The final paren group's source text — the declared parameters."""
    if not header.paren_groups:
        return ""
    open_index, close_index, _ = header.paren_groups[-1]
    return normalise_whitespace(view.text[open_index : close_index + 1])


__all__ = ["Verdict", "classify"]
