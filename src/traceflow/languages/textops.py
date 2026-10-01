"""Token primitives shared by the scanner-based analyzers (plan.md §6, §66).

Reading a declaration *backward* from its brace — over a parenthesised parameter
list, an optional generic list, then the name, then the keyword before it — is the
mechanism both the TypeScript and the C-family engine use, and these are its pieces.
They live here so the two engines cannot drift: a fix to paren-matching is one fix,
not two that disagree.

Every function takes a :class:`~traceflow.languages.typescript.scanner.SourceView`
and treats "in code" the same way that class defines it: outside strings, templates,
comments and regex literals.
"""

from __future__ import annotations

import hashlib
from typing import Protocol


class MaskedView(Protocol):
    """What the shared helpers need of a scanner: the TS and C-family views both."""

    text: str
    length: int

    def in_code(self, index: int) -> bool: ...


def skip_space_back(view: MaskedView, index: int, floor: int) -> int:
    """The last code, non-whitespace offset at or before *index*; ``floor`` when none."""
    while index > floor:
        if not view.in_code(index):
            index -= 1
            continue
        if not view.text[index].isspace():
            return index
        index -= 1
    return floor


def skip_space_forward(view: MaskedView, index: int) -> int:
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


def read_word_back(view: MaskedView, end: int, floor: int) -> tuple[str, int]:
    """The identifier ending at (and including) offset *end*, and where it starts.

    Returns ``("", end + 1)`` when no identifier ends there.
    """
    if end <= floor or end >= view.length:
        return "", end + 1
    if not view.in_code(end) or not is_word_char(view.text[end]):
        return "", end + 1
    start = end
    while start > floor and view.in_code(start - 1) and is_word_char(view.text[start - 1]):
        start -= 1
    return view.text[start : end + 1], start


def is_word_char(char: str) -> bool:
    """True when *char* can appear inside an identifier in any supported language."""
    return char.isalnum() or char in "$_#"


def match_paren_backward(view: MaskedView, close: int, floor: int) -> int | None:
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


def match_angle_backward(view: MaskedView, close: int, floor: int) -> int | None:
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


def match_paren_forward(view: MaskedView, open_index: int) -> int | None:
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


def normalise_whitespace(text: str) -> str:
    """Collapse every whitespace run to one space — how signatures are compared."""
    return " ".join(text.split())


def fingerprint(*parts: str) -> str:
    """A short, stable hash of the parts — how a symbol declares, what it contains."""
    joined = "\x1f".join(parts)
    return hashlib.sha256(joined.encode("utf-8", "replace")).hexdigest()[:16]
