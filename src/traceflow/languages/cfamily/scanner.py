"""The token-level scanner beneath the C-family analyzer (plan.md §66).

The sibling of :mod:`traceflow.languages.typescript.scanner`, for the languages the
TypeScript grammar cannot be stretched over: Go, Java, Rust, C#. The difference that
matters is one mask class: **character literals**. In this family a quote tick is a
character, not a string — ``b'('`` in Go, ``'\\''`` in Java, ``'c'`` in C# — and a
scanner without a mode for it reads the ``(`` inside the literal as real code, then
paren-matches against it and misclassifies every declaration after it.

What is otherwise tracked mirrors the TypeScript scanner, minus what this family
lacks: no template literals, and regular expressions only in Rust's raw-string form
(``r"…"``), which is consumed as a string. Raw strings in Go backticks and Rust
``r#"…"#`` are consumed as strings of their kind.

The scanner deliberately does not build a syntax tree. Each of these languages has a
specification larger than one scanner file; the constructs that carry dependency and
symbol information — declarations, imports, calls — are a small set of shapes over
braces, parentheses and angle brackets, and recognising those shapes degrades to
"nothing known" on the rest rather than crashing on a half-written file (plan.md §46).
"""

from __future__ import annotations

from bisect import bisect_right

_CODE = ord("_")
_STRING = ord("s")
_CHAR = ord("c")
_COMMENT = ord("m")
_RAW = ord("r")


class SourceView:
    """A C-family source file with cheap, position-aware classification of every offset.

    The mask vocabulary: ``_`` code, ``s`` string, ``c`` character literal,
    ``m`` comment, ``r`` raw string. Identical query interface to the TypeScript
    scanner, so :mod:`traceflow.languages.textops` serves both unchanged.
    """

    __slots__ = ("_line_starts", "_mask", "_raw_string_depth", "length", "text")

    def __init__(self, text: str) -> None:
        self.text = text
        self.length = len(text)

        starts = [0]
        for index, char in enumerate(text):
            if char == "\n":
                starts.append(index + 1)
        self._line_starts = starts

        mask = bytearray(b"_" * self.length)
        self._mask = mask
        self._raw_string_depth = 0

        index = 0
        while index < self.length:
            char = text[index]
            if char in "\"'":
                index = self._consume_quoted(index, char)
                continue
            if char == "/" and text.startswith("//", index):
                index = self._consume_line_comment(index)
                continue
            if char == "/" and text.startswith("/*", index):
                index = self._consume_block_comment(index)
                continue
            index = self._advance_plain(index)

        # Deliberate: the TypeScript engine reports an unterminated template as a
        # parse error because its token stream is untrustworthy past that point.
        # This family has no construct with the same failure mode — an unclosed
        # string or comment ends cleanly at the newline or end of file — so there
        # is no truncation signal to propagate, and a half-written file yields
        # every symbol it managed to close.

    # ------------------------------------------------------------------ consumption

    def _consume_quoted(self, start: int, quote: str) -> int:
        """Consume a string (``"``) or a character literal (``'``).

        Go backtick raw strings are handled by the caller before this runs; a
        backtick here is consumed as ordinary code, which is what it is in none
        of these languages.
        """
        kind = _STRING if quote == '"' else _CHAR
        self._mask[start] = kind
        index = start + 1
        while index < self.length:
            char = self.text[index]
            self._mask[index] = kind
            if char == "\\":
                if index + 1 < self.length:
                    self._mask[index + 1] = kind
                index += 2
                continue
            if char == quote:
                return index + 1
            if char == "\n":
                # Unterminated: the newline ends it. The languages disagree on
                # whether this is legal (Go says no), and the scanner's answer is
                # the same either way — the literal ended, the newline is code.
                return index
            index += 1
        return self.length

    def _consume_go_raw(self, start: int) -> int:
        """Consume a Go backtick raw string; they never escape, so the next backtick ends it."""
        index = start
        while index < self.length:
            if self.text[index] == "`":
                self._mask[index] = _RAW
                return index + 1
            if self.text[index] == "\n":
                # Raw strings may span lines; keep going.
                self._mask[index] = _RAW
                index += 1
                continue
            self._mask[index] = _RAW
            index += 1
        self._raw_string_depth += 1
        return self.length

    def _consume_rust_raw(self, start: int) -> int:
        """Consume a Rust raw string ``r"…"`` / ``r#"…"#`` / byte forms ``br"…"``.

        The closing quote must be followed by exactly as many ``#`` as the opener,
        so ``r#"a "quote" b"#`` contains quotes unmolested. Not a raw string at
        all? The rewind treats the ``r`` as an identifier, which is what it is.
        """
        index = start
        hashes = 0
        while index < self.length and self.text[index] == "#":
            hashes += 1
            index += 1
        if index >= self.length or self.text[index] != '"':
            # `r` beginning an identifier, or `r#` a raw identifier: plain code.
            return start + 1
        self._mask[start] = _RAW
        end_marker = '"' + "#" * hashes
        body_start = index + 1
        cursor = body_start
        while cursor < self.length:
            self._mask[cursor] = _RAW
            if self.text.startswith(end_marker, cursor):
                for offset in range(1, hashes + 1):
                    self._mask[cursor + offset] = _RAW
                return cursor + 1 + hashes
            cursor += 1
        self._raw_string_depth += 1
        return self.length

    def _consume_line_comment(self, start: int) -> int:
        end = self.text.find("\n", start)
        if end == -1:
            end = self.length
        for index in range(start, end):
            self._mask[index] = _COMMENT
        return end

    def _consume_block_comment(self, start: int) -> int:
        end = self.text.find("*/", start + 2)
        end = self.length if end == -1 else end + 2
        for index in range(start, end):
            self._mask[index] = _COMMENT
        return end

    # ------------------------------------------------------------------ main walk

    def _advance_plain(self, index: int) -> int:
        """One main-loop step for a character with no construct of its own."""
        char = self.text[index]
        if char == "`":
            return self._consume_go_raw(index)
        if char == "r" and (self.text.startswith('r"', index) or self.text.startswith("r#", index)):
            return self._consume_rust_raw(index)
        if char == "b" and self.text.startswith('br"', index):
            return self._consume_rust_raw(index + 1)
        return index + 1

    # ------------------------------------------------------------------ queries

    def in_code(self, index: int) -> bool:
        """True when the offset is real code: not inside a string, char or comment."""
        return 0 <= index < self.length and self._mask[index] == _CODE

    def in_comment(self, index: int) -> bool:
        return 0 <= index < self.length and self._mask[index] == _COMMENT

    def line_of(self, index: int) -> int:
        """The 1-based line containing *index*."""
        return bisect_right(self._line_starts, index)

    def line_start(self, line: int) -> int:
        """The offset of the beginning of *line* (1-based), clamped to the file."""
        position = min(max(line, 1), len(self._line_starts)) - 1
        return self._line_starts[position]

    def raw_string_left_open(self) -> bool:
        """A raw string that never closed — evidence the file was cut off mid-write."""
        return self._raw_string_depth > 0

    def string_content_at(self, index: int) -> tuple[int, int] | None:
        """The inner-content span of the string run starting at *index*, if it is one.

        The run includes both quotes — the scanner masks the whole literal — so
        the content is the run minus its first and last character. This accessor
        exists because an import path lives inside a masked string, and a walker
        that searched for a code-offset quote would search forever: there are no
        code-offset quotes inside a literal, by the scanner's own design.
        """
        if index >= self.length or self._mask[index] != _STRING:
            return None
        end = index
        while end < self.length and self._mask[end] == _STRING:
            end += 1
        return index + 1, end - 1

    def string_spans_between(self, start: int, end: int) -> tuple[tuple[int, int], ...]:
        """Every string literal's inner-content span within ``[start, end)``.

        Ordered, non-overlapping, exclusive of the quotes. Go's import block is
        a list of these — the paths are strings, and this is how they are read.
        """
        spans: list[tuple[int, int]] = []
        index = start
        while index < end:
            found = self.string_content_at(index)
            if found is None:
                index += 1
                continue
            spans.append(found)
            index = found[1] + 1
        return tuple(spans)

    def identifier_at(self, index: int) -> str | None:
        """The identifier starting at *index*, or ``None``."""
        if index >= self.length or self._mask[index] != _CODE:
            return None
        char = self.text[index]
        if not (char.isalpha() or char == "_"):
            return None
        end = index + 1
        while end < self.length:
            char = self.text[end]
            if self._mask[end] != _CODE:
                break
            if char.isalnum() or char == "_":
                end += 1
                continue
            break
        return self.text[index:end]

    def is_word_char(self, index: int) -> bool:
        """True when the offset continues an identifier."""
        if index >= self.length or index < 0:
            return False
        if self._mask[index] != _CODE:
            return False
        char = self.text[index]
        return char.isalnum() or char == "_"
