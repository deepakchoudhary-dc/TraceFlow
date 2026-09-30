"""The token-level scanner beneath the TypeScript analyzer (plan.md §66, §70).

A scanner, not a compiler. It answers one question for every offset in a source file:
*what kind of thing is this character?* Strings, template literals, comments and regular
expressions are the answer's whole point — a dependency hidden inside a comment or a
URL inside a string must not become an edge in the graph.

The scanner deliberately does not build a syntax tree. TypeScript's grammar is large
and its edge cases hostile ( decorators, conditional types, declaration merging); a
scanner that recognises the handful of shapes which carry dependency and symbol
information covers the repositories that matter, degrades to "nothing known" on the
rest, and never crashes on a half-written file — which, while an agent is editing, is
a normal thing to encounter (plan.md §46).

What is tracked, and why:

* strings, template literals and comments — so nothing inside them is misread;
* template-literal nesting — ``${`a${`b`}c`}`` nests, and getting the depth wrong
  ends the outer template early and leaks its body into the token stream;
* regular expressions — ``/https?:\\/\\//`` would otherwise be scanned for symbols,
  and worse, an unbalanced ``)`` inside a pattern would corrupt every paren-matching
  decision made after it. Division and regex are ambiguous at the same character; the
  heuristic is the standard one (see ``_regex_allowed``), and a mis-fire *rewinds*:
  a division misread as an unterminated regex is unmasked rather than trusted, so a
  wrong guess costs a few stray tokens, not the rest of the file;
* brace, bracket and paren depth — declaration context and call context are decided
  by depth, not by line position;

One deliberate non-feature: TypeScript's ``@ts-ignore`` / ``@ts-expect-error``
suppressions are *not* consulted. They tell the compiler a line's types cannot be
checked; they do not unmake the runtime fact that the import or the declaration
exists. TraceFlow records dependencies and declarations — facts about what the
program does — so a suppression comment neither adds nor removes an edge. The one
honest use of suppression tracking would be to report it, and the analyzer's report
already carries the import it annotates, which is the same evidence in a stronger
form.
"""

from __future__ import annotations

from bisect import bisect_right


class SourceView:
    """A source file with cheap, position-aware classification of every offset."""

    __slots__ = ("_line_starts", "_mask", "_template_depth", "length", "text")

    def __init__(self, text: str) -> None:
        self.text = text
        self.length = len(text)

        starts = [0]
        for index, char in enumerate(text):
            if char == "\n":
                starts.append(index + 1)
        self._line_starts = starts

        # _mask[i] is "_" outside every string/template/comment/regex, and one of
        # "s" (string), "t" (template), "c" (comment), "r" (regex body) inside one.
        # Regex bodies are masked like strings: a pattern's ``)`` or ``{`` must not
        # reach the paren and brace counters, and no import was ever written inside
        # a regular expression.
        mask = bytearray(b"_" * self.length)
        self._mask = mask
        self._template_depth = 0

        index = 0
        while index < self.length:
            char = text[index]
            if char in "\"'":
                index = self._consume_string(index, char)
                continue
            if char == "`":
                index = self._consume_template(index)
                continue
            if char == "/" and text.startswith("//", index):
                index = self._consume_line_comment(index)
                continue
            if char == "/" and text.startswith("/*", index):
                index = self._consume_block_comment(index)
                continue
            if char == "/" and self._regex_allowed(index):
                index = self._consume_regex(index)
                continue
            index += 1

    # ------------------------------------------------------------------ consumption

    def _consume_string(self, start: int, quote: str) -> int:
        self._mask[start] = ord("s")
        index = start + 1
        while index < self.length:
            char = self.text[index]
            self._mask[index] = ord("s")
            if char == "\\":
                # An escape consumes the next character whatever it is, so `\"`
                # does not end the string.
                if index + 1 < self.length:
                    self._mask[index + 1] = ord("s")
                index += 2
                continue
            if char == quote:
                return index + 1
            if char == "\n":
                # An unterminated single-line string: the newline ends it. JS
                # calls this a syntax error; the scanner calls it the end of the
                # string and lets the caller decide what the file deserves.
                return index
            index += 1
        return self.length

    def _consume_template(self, start: int) -> int:
        """Consume one whole template literal, including nested ``${ ... }`` zones.

        The subtlety is that ``${`` can contain another template, which can contain
        another ``${`` — the zones nest, and only matching braces unwind them. A
        scanner that treated the first ``}`` as the end would leak the rest of the
        nesting into the stream as if it were top-level code.

        A template that never finds its closing backtick raises
        :attr:`_template_depth` — the truncation signal the analyzer reports — and
        a closed one leaves it untouched, which is why ordinary template use is
        not mistaken for a broken file.
        """
        # Start *after* the opening backtick: the loop reads depth 0 as "inside
        # the template's own text", where a backtick means "closed". Beginning
        # on the opening delimiter would end the template before it began and
        # leak its entire body into the token stream as code.
        index = start + 1
        depth = 0
        while index < self.length:
            char = self.text[index]
            if char == "\\":
                self._mask[index] = ord("t")
                if index + 1 < self.length:
                    self._mask[index + 1] = ord("t")
                index += 2
                continue
            if depth == 0:
                self._mask[index] = ord("t")
                if char == "`":
                    return index + 1
                if char == "$" and self.text.startswith("${", index):
                    depth = 1
                    self._mask[index] = ord("t")
                    self._mask[index + 1] = ord("t")
                    index += 2
                    continue
                index += 1
                continue
            # Inside a ${ ... } zone: code, scanned for strings/templates of its own.
            if char in "\"'":
                index = self._consume_string(index, char)
                continue
            if char == "`":
                index = self._consume_template(index)
                continue
            if char == "/" and self.text.startswith("//", index):
                index = self._consume_line_comment(index)
                continue
            if char == "/" and self.text.startswith("/*", index):
                index = self._consume_block_comment(index)
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    # Back into the template's own text.
                    self._mask[index] = ord("t")
                    index += 1
                    continue
            # Zone code stays unmasked, so it scans as ordinary code; strings,
            # templates and comments inside it were masked by their consumers.
            index += 1
        # Ran off the end without the closing backtick: the file was cut off
        # inside this template. One is enough to signal truncation.
        self._template_depth += 1
        return self.length

    def _previous_code_char(self, index: int) -> tuple[int, str] | None:
        """The nearest preceding unmasked, non-whitespace character, if any."""
        cursor = index - 1
        while cursor >= 0:
            if self._mask[cursor] != ord("_"):
                cursor -= 1
                continue
            char = self.text[cursor]
            if not char.isspace():
                return cursor, char
            cursor -= 1
        return None

    def _word_before(self, index: int) -> str:
        """The identifier whose last code character sits at *index*.

        The contract is "the word this position ends", because every caller has
        just landed on a word's final character and wants to know what it read.
        An empty string when no identifier ends there.
        """
        if index < 0 or index >= self.length or self._mask[index] != ord("_"):
            return ""
        if not (self.text[index].isalnum() or self.text[index] in ("$", "_")):
            return ""
        end = index + 1
        start = end
        while start > 0:
            char = self.text[start - 1]
            if self._mask[start - 1] != ord("_") or not (char.isalnum() or char in ("$", "_")):
                break
            start -= 1
        word = self.text[start:end]
        return word if word.isidentifier() else ""

    #: A regex literal may begin after any of these characters, or at file start.
    _REGEX_AFTER = frozenset("([{,;=:!&|?+-*/%^<>~")

    #: Keywords after which a ``/`` starts a regex rather than division — ``return /x/.test(y)``.
    _REGEX_KEYWORDS = frozenset(
        {
            "return",
            "typeof",
            "case",
            "in",
            "of",
            "new",
            "delete",
            "void",
            "do",
            "else",
            "yield",
            "await",
            "throw",
        }
    )

    def _regex_allowed(self, index: int) -> bool:
        """The standard division-versus-regex heuristic.

        A ``/`` begins a regex when the previous code character is an opening
        bracket, an operator, or a statement keyword; it is a division when a
        value could end there (``a / b``, ``(x) / 2``). JavaScript itself resolves
        this with full parsing; this approximation misfires only on constructions
        rare enough that the rewind in :meth:`_consume_regex` makes them harmless.
        """
        previous = self._previous_code_char(index)
        if previous is None:
            return True
        position, char = previous
        if char in self._REGEX_AFTER:
            return True
        if char.isalnum() or char in ("$", "_", ")", "]", ".", '"', "'", "`"):
            # A value ends here, so division is likelier — unless the word is a
            # keyword, after which a regex can start.
            return self._word_before(position) in self._REGEX_KEYWORDS
        return False

    def _consume_regex(self, start: int) -> int:
        """Consume a regex literal; *rewind* when it turns out to be a division.

        The rewind is the load-bearing part. A ``/`` misclassified as a regex and
        run to end-of-line would mask real code; hitting a newline before the
        closing ``/`` means the guess was wrong, so every masked offset is unmasked
        and the ``/`` is treated as the division it probably was.
        """
        touched: list[int] = []
        index = start
        in_class = False
        while index < self.length:
            char = self.text[index]
            if char == "\n":
                # Unterminated on this line: division, not regex. Rewind.
                for position in touched:
                    self._mask[position] = ord("_")
                return start + 1
            touched.append(index)
            self._mask[index] = ord("r")
            if char == "\\" and index + 1 < self.length:
                touched.append(index + 1)
                self._mask[index + 1] = ord("r")
                index += 2
                continue
            if char == "[":
                in_class = True
            elif char == "]":
                in_class = False
            elif char == "/" and not in_class:
                return index + 1
            index += 1
        # Ran off the end: unterminated. Rewind — same reasoning as the newline case.
        for position in touched:
            self._mask[position] = ord("_")
        return start + 1

    def _consume_line_comment(self, start: int) -> int:
        end = self.text.find("\n", start)
        if end == -1:
            end = self.length
        for index in range(start, end):
            self._mask[index] = ord("c")
        return end

    def _consume_block_comment(self, start: int) -> int:
        end = self.text.find("*/", start + 2)
        end = self.length if end == -1 else end + 2
        for index in range(start, end):
            self._mask[index] = ord("c")
        return end

    # ------------------------------------------------------------------ queries

    def in_code(self, index: int) -> bool:
        """True when the offset is real code: not inside a string, template or comment."""
        return 0 <= index < self.length and self._mask[index] == ord("_")

    def in_comment(self, index: int) -> bool:
        return 0 <= index < self.length and self._mask[index] == ord("c")

    def line_of(self, index: int) -> int:
        """The 1-based line containing *index*."""
        return bisect_right(self._line_starts, index)

    def line_start(self, line: int) -> int:
        """The offset of the beginning of *line* (1-based), clamped to the file."""
        position = min(max(line, 1), len(self._line_starts)) - 1
        return self._line_starts[position]

    def unclosed_template_nesting(self) -> int:
        """``${`` zones still open at end of file — evidence of a truncated file."""
        return self._template_depth

    # ------------------------------------------------------------------ token access

    def skip_space(self, index: int) -> int:
        """The first offset at or after *index* that is code and not whitespace."""
        while index < self.length:
            if self._mask[index] != ord("_"):
                return index
            char = self.text[index]
            if not char.isspace():
                return index
            index += 1
        return self.length

    def identifier_at(self, index: int) -> str | None:
        """The identifier (or keyword) starting at *index*, or ``None``."""
        if index >= self.length or self._mask[index] != ord("_"):
            return None
        char = self.text[index]
        # An identifier starts with a letter, ``$`` or ``_`` — never a digit. The
        # continuation loop below is what allows digits.
        if not (char.isalpha() or char in ("$", "_")):
            return None
        end = index + 1
        while end < self.length:
            char = self.text[end]
            if self._mask[end] != ord("_"):
                break
            if char.isalnum() or char in ("$", "_", "#"):
                end += 1
                continue
            break
        return self.text[index:end]

    def is_word_char(self, index: int) -> bool:
        """True when the offset continues an identifier (used to reject prefixes)."""
        if index >= self.length or index < 0:
            return False
        if self._mask[index] != ord("_"):
            return False
        char = self.text[index]
        return char.isalnum() or char in ("$", "_", "#")


class DepthTracker:
    """Brace, bracket and paren depth at every code offset.

    Symbol declarations live at (or near) statement depth; call expressions inside a
    body live deeper. The analyzer uses the two facts together — a name followed by
    ``(`` at *any* depth is a call unless a declaration keyword put it there — and
    the tracker is what keeps template-literal braces from disturbing the count.
    """

    _OPEN = "([{"
    _CLOSE = ")]}"

    def __init__(self, view: SourceView) -> None:
        depths = [0] * view.length
        depth = 0
        index = 0
        while index < view.length:
            if not view.in_code(index):
                index += 1
                continue
            char = view.text[index]
            if char in self._OPEN:
                depth += 1
            elif char in self._CLOSE and depth > 0:
                depth -= 1
            depths[index] = depth
            index += 1
        self._depths = depths

    def at(self, index: int) -> int:
        """The nesting depth after the character at *index* (0 outside the file)."""
        if 0 <= index < len(self._depths):
            return self._depths[index]
        return 0
