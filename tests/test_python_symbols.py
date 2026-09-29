"""Symbol extraction and fingerprinting.

The property that matters most here is the one that keeps the analysis *useful*:
a fingerprint is computed from the syntax tree, so reformatting a file, reindenting a
block or editing a comment does not register as a change. Only a change to the
program's structure does.
"""

from __future__ import annotations

import ast

from traceflow.languages.base import Symbol, SymbolKind
from traceflow.languages.python.symbols import extract_symbols


def symbols_of(source: str) -> dict[str, Symbol]:
    return {symbol.qualified_name: symbol for symbol in extract_symbols(ast.parse(source))}


def signature_of(source: str, name: str) -> str:
    return symbols_of(source)[name].signature_fingerprint


def body_of(source: str, name: str) -> str:
    return symbols_of(source)[name].body_fingerprint


# --------------------------------------------------------------------------- extraction


def test_a_top_level_function_is_found() -> None:
    symbol = symbols_of("def run(a, b):\n    return a + b\n")["run"]

    assert symbol.kind is SymbolKind.FUNCTION
    assert symbol.name == "run"
    assert symbol.parent is None
    assert symbol.line_start == 1
    assert symbol.line_end == 2
    assert symbol.signature == "a, b"


def test_a_class_is_found_with_its_bases() -> None:
    symbol = symbols_of("class Service(Base, Mixin):\n    pass\n")["Service"]

    assert symbol.kind is SymbolKind.CLASS
    assert symbol.bases == ("Base", "Mixin")
    assert symbol.signature == "(Base, Mixin)"


def test_methods_are_distinguished_from_functions() -> None:
    source = (
        "class Service:\n    def login(self):\n        return 1\n\n\ndef helper():\n    return 2\n"
    )
    symbols = symbols_of(source)

    assert symbols["Service.login"].kind is SymbolKind.METHOD
    assert symbols["Service.login"].parent == "Service"
    assert symbols["helper"].kind is SymbolKind.FUNCTION


def test_a_function_nested_in_a_method_is_a_function_not_a_method() -> None:
    """A ``def`` inside a method belongs to that method, not to the class."""
    source = (
        "class Service:\n"
        "    def login(self):\n"
        "        def inner():\n"
        "            return 1\n"
        "        return inner\n"
    )
    symbols = symbols_of(source)

    assert symbols["Service.login.inner"].kind is SymbolKind.FUNCTION
    assert symbols["Service.login.inner"].parent == "Service.login"


def test_nested_classes_are_qualified() -> None:
    source = "class Outer:\n    class Inner:\n        pass\n"
    assert "Outer.Inner" in symbols_of(source)


def test_async_functions_are_found() -> None:
    assert symbols_of("async def go():\n    return 1\n")["go"].kind is SymbolKind.FUNCTION


def test_decorators_are_recorded() -> None:
    symbol = symbols_of("@property\ndef name():\n    return 'x'\n")["name"]
    assert symbol.decorators == ("property",)


# --------------------------------------------------------------------------- signatures


def test_a_parameter_change_is_a_signature_change() -> None:
    before = signature_of("def f(a, b):\n    return a\n", "f")
    after = signature_of("def f(a, b, c):\n    return a\n", "f")

    assert before != after


def test_a_default_change_is_a_signature_change() -> None:
    before = signature_of("def f(a=1):\n    return a\n", "f")
    after = signature_of("def f(a=2):\n    return a\n", "f")

    assert before != after


def test_an_annotation_change_is_a_signature_change() -> None:
    before = signature_of("def f(a: int) -> int:\n    return a\n", "f")
    after = signature_of("def f(a: str) -> int:\n    return a\n", "f")

    assert before != after


def test_a_return_annotation_change_is_a_signature_change() -> None:
    before = signature_of("def f() -> int:\n    return 1\n", "f")
    after = signature_of("def f() -> str:\n    return 'x'\n", "f")

    assert before != after


def test_a_decorator_change_is_a_signature_change() -> None:
    """A method that becomes ``@staticmethod`` is called differently with identical parameters."""
    before = signature_of("class C:\n    def m(self):\n        return 1\n", "C.m")
    after = signature_of("class C:\n    @staticmethod\n    def m():\n        return 1\n", "C.m")

    assert before != after


def test_keyword_only_marker_is_part_of_the_signature() -> None:
    before = signature_of("def f(a, b):\n    return a\n", "f")
    after = signature_of("def f(a, *, b):\n    return a\n", "f")

    assert before != after


def test_positional_only_marker_is_part_of_the_signature() -> None:
    before = signature_of("def f(a):\n    return a\n", "f")
    after = signature_of("def f(a, /):\n    return a\n", "f")

    assert before != after


def test_varargs_are_part_of_the_signature() -> None:
    before = signature_of("def f(a):\n    return a\n", "f")
    after = signature_of("def f(a, *rest):\n    return a\n", "f")

    assert before != after


def test_a_class_base_change_is_a_signature_change() -> None:
    before = signature_of("class C(Base):\n    pass\n", "C")
    after = signature_of("class C(Other):\n    pass\n", "C")

    assert before != after


def test_the_rendered_signature_is_readable() -> None:
    """A fingerprint alone is not evidence; the declaration has to be legible."""
    symbol = symbols_of("def f(a: int, b: str = 'x', *rest, key=None, **kw) -> bool:\n    pass\n")[
        "f"
    ]

    assert symbol.signature == "a: int, b: str = 'x', *rest, key = None, **kw -> bool"


# --------------------------------------------------------------------------- bodies


def test_a_body_change_is_not_a_signature_change() -> None:
    source_before = "def f(a):\n    return a + 1\n"
    source_after = "def f(a):\n    return a + 2\n"

    assert signature_of(source_before, "f") == signature_of(source_after, "f")
    assert body_of(source_before, "f") != body_of(source_after, "f")


def test_reformatting_changes_neither_fingerprint() -> None:
    """The whole point of fingerprinting the tree rather than the text."""
    compact = "def f(a):\n    return a\n"
    sprawling = "def f(\n    a,\n):\n\n\n        return    a\n"

    assert signature_of(compact, "f") == signature_of(sprawling, "f")
    assert body_of(compact, "f") == body_of(sprawling, "f")


def test_a_comment_changes_neither_fingerprint() -> None:
    without = "def f(a):\n    return a\n"
    with_comment = "def f(a):\n    # explain\n    return a\n"

    assert signature_of(without, "f") == signature_of(with_comment, "f")
    assert body_of(without, "f") == body_of(with_comment, "f")


def test_a_docstring_change_is_a_body_change() -> None:
    """A docstring is content, so a change to it is reported rather than hidden."""
    before = 'def f():\n    """One."""\n    return 1\n'
    after = 'def f():\n    """Two."""\n    return 1\n'

    assert signature_of(before, "f") == signature_of(after, "f")
    assert body_of(before, "f") != body_of(after, "f")


def test_a_change_to_one_function_leaves_its_sibling_alone() -> None:
    before = "def one():\n    return 1\n\n\ndef two():\n    return 2\n"
    after = "def one():\n    return 99\n\n\ndef two():\n    return 2\n"

    assert body_of(before, "one") != body_of(after, "one")
    assert body_of(before, "two") == body_of(after, "two")


def test_a_method_change_does_not_change_the_enclosing_class_body() -> None:
    """Nested definitions are excluded, so a change is reported once, not twice."""
    before = "class C:\n    def m(self):\n        return 1\n"
    after = "class C:\n    def m(self):\n        return 2\n"

    assert body_of(before, "C") == body_of(after, "C")
    assert body_of(before, "C.m") != body_of(after, "C.m")


def test_a_class_level_assignment_changes_the_class_body() -> None:
    before = "class C:\n    LIMIT = 1\n"
    after = "class C:\n    LIMIT = 2\n"

    assert body_of(before, "C") != body_of(after, "C")


def test_adding_a_method_does_not_change_the_class_body_fingerprint() -> None:
    """The class's own body is unchanged; the new method is reported on its own."""
    before = "class C:\n    LIMIT = 1\n"
    after = "class C:\n    LIMIT = 1\n\n    def extra(self):\n        return 1\n"

    assert body_of(before, "C") == body_of(after, "C")


def test_an_empty_body_has_a_stable_fingerprint() -> None:
    assert body_of("def f():\n    pass\n", "f") == body_of("def f():\n    pass\n", "f")


def test_moving_a_function_does_not_change_its_fingerprints() -> None:
    """Line numbers are not part of either fingerprint."""
    before = "def f(a):\n    return a\n"
    after = "x = 1\ny = 2\n\n\ndef f(a):\n    return a\n"

    assert signature_of(before, "f") == signature_of(after, "f")
    assert body_of(before, "f") == body_of(after, "f")


# --------------------------------------------------------------------------- definitions in blocks


def test_a_definition_inside_a_conditional_is_found() -> None:
    """A version shim is a definition, and a change to it still changes how it is called.

    Walking only direct statements made every one of these invisible, so a signature change
    inside an ``if`` was reported as nothing more than "the file changed" and no caller was
    warned.
    """
    cases = {
        "if": "import sys\n\nif sys.platform == 'win32':\n    def f(x):\n        return x\n",
        "try": "try:\n    def f(x):\n        return x\nexcept Exception:\n    pass\n",
        "with": (
            "import contextlib\n\nwith contextlib.suppress(Exception):\n"
            "    def f(x):\n        return x\n"
        ),
        "match": (
            "import sys\n\nmatch sys.platform:\n    case 'win32':\n"
            "        def f(x):\n            return x\n    case _:\n        pass\n"
        ),
        "for": "for _ in (1,):\n    def f(x):\n        return x\n",
    }

    for label, source in cases.items():
        assert "f" in symbols_of(source), label


def test_a_class_inside_a_conditional_keeps_its_methods() -> None:
    source = (
        "import sys\n\nif sys.version_info >= (3, 10):\n    class C:\n"
        "        def m(self):\n            return 1\n"
    )

    found = symbols_of(source)

    assert found["C"].kind is SymbolKind.CLASS
    assert found["C.m"].kind is SymbolKind.METHOD
    assert found["C.m"].parent == "C"


def test_a_definition_inside_a_function_is_still_not_a_method() -> None:
    """``inside_class`` resets when descending into a function body, blocks or not."""
    source = (
        "class C:\n    def m(self):\n        if True:\n"
        "            def inner():\n                return 1\n"
        "        return inner\n"
    )

    assert symbols_of(source)["C.m.inner"].kind is SymbolKind.FUNCTION


def test_a_name_defined_twice_is_one_symbol_covering_both_arms() -> None:
    """A diff is keyed by name, so the two arms of a shim have to fold into one entry.

    Keeping whichever came last would leave a change to the first arm invisible — the same
    silent miss the walk was fixed to avoid.
    """
    source = (
        "import sys\n\nif sys.version_info >= (3, 11):\n    def parse(x):\n        return x\n"
        "else:\n    def parse(x):\n        return x + 1\n"
    )

    found = symbols_of(source)

    assert list(found) == ["parse"]
    assert found["parse"].occurrences == 2


def test_a_change_to_either_arm_of_a_shim_is_detected() -> None:
    before = (
        "import sys\n\nif sys.version_info >= (3, 11):\n    def parse(x):\n        return x\n"
        "else:\n    def parse(x):\n        return x + 1\n"
    )

    for label, after in (
        (
            "first arm",
            before.replace(
                "def parse(x):\n        return x\n", "def parse(x, y):\n        return x\n", 1
            ),
        ),
        ("second arm", before.replace("return x + 1", "return x + 2", 1)),
    ):
        assert signature_of(before, "parse") != signature_of(after, "parse") or body_of(
            before, "parse"
        ) != body_of(after, "parse"), label


def test_a_single_definition_records_one_occurrence() -> None:
    assert symbols_of("def f(x):\n    return x\n")["f"].occurrences == 1
