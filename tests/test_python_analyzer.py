"""Parsing a Python file into an analysis.

The behaviour that matters most is what happens when the source is broken. TraceFlow
watches repositories while they are being edited, so a half-written file is a normal
thing to encounter; it must degrade to "nothing known", never to an exception.
"""

from __future__ import annotations

import pytest

from traceflow.languages.base import ImportRef, render_import
from traceflow.languages.python.analyzer import PythonAnalyzer, analyze_python

ANALYZER = PythonAnalyzer()


def imports_of(source: str) -> dict[str, ImportRef]:
    return {render_import(item): item for item in analyze_python("m.py", source.encode()).imports}


def calls_of(source: str) -> list[str]:
    return [item.name for item in analyze_python("m.py", source.encode()).calls]


# --------------------------------------------------------------------------- basics


def test_a_module_with_nothing_in_it_is_analysed() -> None:
    analysis = analyze_python("empty.py", b"")

    assert analysis.symbols == ()
    assert analysis.imports == ()
    assert analysis.parse_error is None


def test_the_digest_matches_the_content() -> None:
    analysis = analyze_python("m.py", b"x = 1\n")
    assert len(analysis.digest) == 64


def test_symbols_are_extracted() -> None:
    analysis = analyze_python("m.py", b"def f():\n    return 1\n")
    assert [symbol.name for symbol in analysis.symbols] == ["f"]


def test_only_python_files_are_claimed() -> None:
    assert ANALYZER.can_analyze("a.py") is True
    assert ANALYZER.can_analyze("a.pyi") is True
    assert ANALYZER.can_analyze("a.js") is False
    assert ANALYZER.can_analyze("a.pyc") is False


def test_the_cache_kind_carries_the_version() -> None:
    """A change to the analyzer must invalidate what the previous one cached."""
    assert ANALYZER.cache_kind == f"python-{ANALYZER.version}"


# --------------------------------------------------------------------------- imports


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import os\n", "os"),
        ("import a.b.c\n", "a.b.c"),
        ("import numpy as np\n", "numpy"),
        ("from a import b\n", "a.b"),
        ("from a.b import c\n", "a.b.c"),
        ("from . import sibling\n", ".sibling"),
        ("from .mod import thing\n", ".mod.thing"),
        ("from .. import cousin\n", "..cousin"),
        ("from ..pkg import thing\n", "..pkg.thing"),
    ],
)
def test_every_import_form_renders_unambiguously(source: str, expected: str) -> None:
    assert expected in imports_of(source)


def test_a_from_import_of_several_names_produces_several_entries() -> None:
    assert set(imports_of("from a import b, c\n")) == {"a.b", "a.c"}


def test_an_aliased_from_import_records_the_alias() -> None:
    reference = imports_of("from a import b as c\n")["a.b"]
    assert reference.alias == "c"
    assert reference.name == "b"


def test_a_deferred_import_is_still_a_dependency() -> None:
    """An import inside a function is a real dependency of the module."""
    assert "slow" in imports_of("def f():\n    import slow\n    return slow\n")


def test_imports_record_their_line() -> None:
    assert imports_of("\n\nimport os\n")["os"].line == 3


def test_imports_are_sorted_by_position() -> None:
    analysis = analyze_python("m.py", b"import b\nimport a\n")
    assert [item.line for item in analysis.imports] == [1, 2]


# --------------------------------------------------------------------------- calls


def test_a_plain_call_is_recorded() -> None:
    assert calls_of("run()\n") == ["run"]


def test_a_dotted_call_is_recorded_as_written() -> None:
    assert calls_of("service.authenticate(user)\n") == ["service.authenticate"]


def test_a_deeply_dotted_call_is_recorded() -> None:
    assert calls_of("a.b.c.d()\n") == ["a.b.c.d"]


def test_a_call_through_a_subscript_is_not_guessed_at() -> None:
    """There is no static name for ``handlers[0]()``, so none is reported."""
    assert calls_of("handlers[0]()\n") == []


def test_a_call_on_a_literal_is_not_guessed_at() -> None:
    assert calls_of("'text'.upper()\n") == []


def test_calls_inside_functions_are_found() -> None:
    assert calls_of("def f():\n    inner()\n") == ["inner"]


def test_calls_record_their_line() -> None:
    analysis = analyze_python("m.py", b"\n\nrun()\n")
    assert analysis.calls[0].line == 3


# --------------------------------------------------------------------------- malformed source


def test_a_syntax_error_is_recorded_not_raised() -> None:
    analysis = analyze_python("broken.py", b"def f(:\n")

    assert analysis.parse_error is not None
    assert "SyntaxError" in analysis.parse_error
    assert analysis.symbols == ()


def test_null_bytes_are_recorded_not_raised() -> None:
    analysis = analyze_python("binary.py", b"x = 1\x00\n")

    assert analysis.parse_error is not None
    assert analysis.symbols == ()


def test_an_unterminated_string_is_recorded_not_raised() -> None:
    assert analyze_python("broken.py", b"x = 'unterminated\n").parse_error is not None


def test_a_parse_error_is_truncated() -> None:
    """A pathological file must not put a megabyte of error text into an artifact."""
    analysis = analyze_python("broken.py", b"(" * 500 + b"\n")

    assert analysis.parse_error is not None
    assert len(analysis.parse_error) <= 300


def test_the_digest_is_recorded_even_when_parsing_fails() -> None:
    """So a broken file is still identifiable and cacheable."""
    analysis = analyze_python("broken.py", b"def f(:\n")
    assert len(analysis.digest) == 64


def test_a_file_with_an_encoding_declaration_is_read_correctly() -> None:
    source = "# -*- coding: latin-1 -*-\nNAME = 'caf\xe9'\n".encode("latin-1")
    analysis = analyze_python("m.py", source)

    assert analysis.parse_error is None


# --------------------------------------------------------------------------- round trip


def test_an_analysis_survives_a_json_round_trip() -> None:
    from traceflow.languages.base import module_analysis_from_json

    original = analyze_python("m.py", b"import os\n\n\ndef f(a, b=1):\n    return os.getcwd()\n")
    restored = module_analysis_from_json(original.to_json(), path="m.py", module_name="m")

    assert restored.symbols == original.symbols
    assert restored.imports == original.imports
    assert restored.calls == original.calls
    assert restored.digest == original.digest


def test_the_round_trip_overrides_the_stored_path_and_module_name() -> None:
    """Identical content at two paths shares one cache entry, so the caller must win."""
    from traceflow.languages.base import module_analysis_from_json

    original = analyze_python("first.py", b"x = 1\n", module_name="first")
    restored = module_analysis_from_json(original.to_json(), path="second.py", module_name="second")

    assert restored.path == "second.py"
    assert restored.module_name == "second"


def test_a_malformed_payload_is_rejected() -> None:
    from traceflow.languages.base import module_analysis_from_json

    with pytest.raises(ValueError, match="symbol list"):
        module_analysis_from_json({"imports": [], "calls": []}, path="m.py", module_name=None)


def test_an_unknown_symbol_kind_is_rejected() -> None:
    from traceflow.languages.base import module_analysis_from_json

    payload = {
        "symbols": [{"qualified_name": "f", "name": "f", "kind": "wizard"}],
        "imports": [],
        "calls": [],
    }
    with pytest.raises(ValueError):
        module_analysis_from_json(payload, path="m.py", module_name=None)
