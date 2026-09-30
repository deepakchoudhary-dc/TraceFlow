"""Tests for the TypeScript/JavaScript analyzer (plan.md §66, Phase 10).

The fixtures are written as TypeScript, not as what the analyzer hopes TypeScript
looks like: template strings containing quotes, regex literals with unbalanced
brackets, JSX, destructured arrow parameters — the shapes real repositories are
full of and naive scanners are broken by.
"""

from __future__ import annotations

import pytest

from traceflow.languages.typescript.analyzer import TypeScriptAnalyzer
from traceflow.languages.typescript.graph import (
    TypeScriptPaths,
    build_ts_index,
    module_name_for,
    resolve_specifier,
)
from traceflow.languages.typescript.scanner import SourceView

ANALYZER = TypeScriptAnalyzer()


def analyze(source: str, path: str = "mod.ts"):
    return ANALYZER.analyze(path, source.encode("utf-8"), None)


def import_pairs(analysis):
    # The tuples keep the model's None values; only the sort key coerces.
    return sorted(
        ((item.module, item.name, item.alias) for item in analysis.imports),
        key=lambda item: (item[0], item[1] or "", item[2] or ""),
    )


class TestImports:
    def test_named_imports_become_one_record_per_binding(self) -> None:
        analysis = analyze("import { Request, Response } from 'express';\n")
        assert import_pairs(analysis) == [
            ("express", "Request", None),
            ("express", "Response", None),
        ]

    def test_default_import_binds_the_alias(self) -> None:
        analysis = analyze("import prisma from '../config/database';\n")
        assert import_pairs(analysis) == [("../config/database", None, "prisma")]

    def test_default_and_named_in_one_clause(self) -> None:
        analysis = analyze("import React, { useState } from 'react';\n")
        # A default binding has no imported *name* — the alias is what the file
        # calls it, which is what the call resolver binds by.
        assert import_pairs(analysis) == [
            ("react", None, "React"),
            ("react", "useState", None),
        ]

    def test_renamed_binding_keeps_both_names(self) -> None:
        analysis = analyze("import { task as taskHelpers } from '../utils/taskHelpers';\n")
        assert import_pairs(analysis) == [("../utils/taskHelpers", "task", "taskHelpers")]

    def test_type_only_import_is_one_dependency(self) -> None:
        analysis = analyze("import type { Task } from '../types';\n")
        assert import_pairs(analysis) == [("../types", None, None)]

    def test_namespace_import(self) -> None:
        analysis = analyze("import * as path from 'path';\n")
        assert import_pairs(analysis) == [("path", "*", "path")]

    def test_side_effect_import(self) -> None:
        analysis = analyze("import './polyfill';\n")
        assert import_pairs(analysis) == [("./polyfill", None, None)]

    def test_export_from_is_a_dependency(self) -> None:
        analysis = analyze("export { Task } from './task';\nexport * from './legacy';\n")
        assert import_pairs(analysis) == [("./legacy", None, None), ("./task", None, None)]

    def test_dynamic_import(self) -> None:
        analysis = analyze("const m = await import('./heavy');\n")
        assert import_pairs(analysis) == [("./heavy", None, None)]

    def test_require_with_binding(self) -> None:
        analysis = analyze("const express = require('express');\n", "m.js")
        assert import_pairs(analysis) == [("express", None, "express")]

    def test_require_destructured(self) -> None:
        analysis = analyze("const { a, b as c } = require('./m');\n", "m.js")
        assert import_pairs(analysis) == [("./m", "a", None), ("./m", "b", "c")]

    def test_bare_require(self) -> None:
        analysis = analyze("require('./boot');\n", "m.js")
        assert import_pairs(analysis) == [("./boot", None, None)]

    def test_import_inside_a_string_is_not_an_import(self) -> None:
        analysis = analyze("const hint = \"import x from 'nothing';\";\n")
        assert analysis.imports == ()

    def test_import_inside_a_comment_is_not_an_import(self) -> None:
        analysis = analyze("// import fake from './nowhere';\n")
        assert analysis.imports == ()

    def test_removing_one_binding_is_visible(self) -> None:
        """Per-binding granularity: dropping one import is a reportable fact.

        The comparison is on rendered bindings, not module names — both versions
        import 'react', and only a binding-level record can see the difference
        the session actually made.
        """
        before = analyze("import { createElement, useState } from 'react';\n")
        after = analyze("import { useState } from 'react';\n")
        assert import_pairs(before) == [
            ("react", "createElement", None),
            ("react", "useState", None),
        ]
        assert import_pairs(after) == [("react", "useState", None)]


class TestSymbols:
    def test_function_declaration(self) -> None:
        analysis = analyze("export function run(a: string) { return a; }\n")
        assert [(s.qualified_name, s.kind.value) for s in analysis.symbols] == [("run", "function")]
        assert analysis.symbols[0].signature == "(a: string)"

    def test_class_with_members(self) -> None:
        source = (
            "export class TaskService {\n"
            "  constructor(repo: string) { this.repo = repo; }\n"
            "  async create(dto: unknown) { return dto; }\n"
            "  static build() { return new TaskService('x'); }\n"
            "  get value() { return 1; }\n"
            "}\n"
        )
        analysis = analyze(source)
        names = [(s.qualified_name, s.kind.value) for s in analysis.symbols]
        assert ("TaskService", "class") in names
        assert ("TaskService.constructor", "method") in names
        assert ("TaskService.create", "method") in names
        assert ("TaskService.build", "method") in names
        assert ("TaskService.value", "method") in names

    def test_arrow_bound_to_const(self) -> None:
        analysis = analyze("const handler = (req, res) => { res.json({}); };\n")
        assert [(s.qualified_name, s.kind.value) for s in analysis.symbols] == [
            ("handler", "function")
        ]

    def test_arrow_with_single_parameter(self) -> None:
        analysis = analyze("const one = x => { return x; };\n")
        assert [(s.qualified_name, s.kind.value) for s in analysis.symbols] == [("one", "function")]

    def test_method_in_object_literal(self) -> None:
        analysis = analyze("const ctx = { helper() { return 1; } };\n")
        assert ("helper", "function") in [
            (s.qualified_name, s.kind.value) for s in analysis.symbols
        ]

    def test_class_extends_clause_keeps_the_declared_name(self) -> None:
        source = "class Admin extends UserService { ping() { return 1; } }\n"
        analysis = analyze(source)
        names = [s.qualified_name for s in analysis.symbols]
        assert "Admin" in names
        assert "Admin.ping" in names
        # The superclass name must not become a symbol of its own here.
        assert "UserService" not in names

    def test_interface_declaration(self) -> None:
        analysis = analyze("interface Named { name: string; greet(): void; }\n")
        assert ("Named", "class") in [(s.qualified_name, s.kind.value) for s in analysis.symbols]

    def test_type_alias_body_is_not_a_symbol(self) -> None:
        analysis = analyze("type Alias = { a: 1 };\nfunction real() { return 1; }\n")
        assert [s.qualified_name for s in analysis.symbols] == ["real"]

    def test_control_flow_braces_are_not_symbols(self) -> None:
        analysis = analyze("function f() { if (x) { return; } else { return 1; } }\n")
        assert [s.qualified_name for s in analysis.symbols] == ["f"]

    def test_getter_and_setter_fold_to_one_symbol(self) -> None:
        """A get/set pair is one declared name.

        Both records fold into one symbol spanning both bodies, so a change to
        *either* is reported. The setter is finalised after the getter, so the
        surviving line span is the setter's; the fingerprints cover both because
        the fold merges the bodies.
        """
        source = "class C { get v() { return 1; } set v(x) { this._x = x; } }\n"
        analysis = analyze(source)
        # One class, one folded member — no third row for the setter.
        assert [s.qualified_name for s in analysis.symbols] == ["C", "C.v"]
        folded = analysis.symbols[-1]
        # The fold's evidence: both declarations' signatures live on the one symbol.
        assert "|" in folded.signature

    def test_nested_function_is_qualified(self) -> None:
        analysis = analyze("function outer() { function inner() { return 1; } return inner; }\n")
        names = [s.qualified_name for s in analysis.symbols]
        assert "outer" in names
        assert "outer.inner" in names


class TestCalls:
    def test_calls_are_recorded_with_their_line(self) -> None:
        analysis = analyze("function a() {\n  helper();\n  return service.check();\n}\n")
        names = {(item.name, item.line) for item in analysis.calls}
        assert ("helper", 2) in names
        assert ("service.check", 3) in names

    def test_declaration_is_not_a_call(self) -> None:
        analysis = analyze("function process(input: string) { return input; }\n")
        assert analysis.calls == ()

    def test_call_statement_is_a_call(self) -> None:
        """`foo();` is the commonest call form; dropping it loses impact findings."""
        analysis = analyze("function f() { boot(); }\n")
        assert [item.name for item in analysis.calls] == ["boot"]

    def test_nested_calls_are_all_recorded(self) -> None:
        analysis = analyze("const x = outer(inner(1));\n")
        names = {item.name for item in analysis.calls}
        assert names == {"outer", "inner"}

    def test_receiver_chain_is_preserved(self) -> None:
        analysis = analyze("async function h(req: any, res: any) { await res.json(1); }\n")
        assert ("res.json", 1) in {(item.name, item.line) for item in analysis.calls}


class TestRobustness:
    def test_regex_with_unbalanced_bracket_does_not_corrupt(self) -> None:
        source = "const re = /:)/;\nfunction stillFound() { return 1; }\n"
        analysis = analyze(source)
        assert [s.qualified_name for s in analysis.symbols] == ["stillFound"]

    def test_regex_containing_quote_is_masked(self) -> None:
        source = "const re = /['\"]/;\nfunction after() { return 2; }\n"
        analysis = analyze(source)
        assert [s.qualified_name for s in analysis.symbols] == ["after"]

    def test_url_in_string_is_not_a_comment(self) -> None:
        source = "const url = 'http://example.com';\nfunction fine() { return 3; }\n"
        analysis = analyze(source)
        assert [s.qualified_name for s in analysis.symbols] == ["fine"]

    def test_template_with_nested_interpolation(self) -> None:
        source = "const s = `a${ `b${'c'}d` }e`;\nfunction t() { return 1; }\n"
        analysis = analyze(source)
        assert [s.qualified_name for s in analysis.symbols] == ["t"]

    def test_template_containing_braces(self) -> None:
        source = "const s = `x{y}z`;\nfunction u() { return 1; }\n"
        analysis = analyze(source)
        assert [s.qualified_name for s in analysis.symbols] == ["u"]

    def test_jsx_is_expression_syntax(self) -> None:
        source = (
            "import { Card } from './ui';\n"
            "export const View = () => {\n"
            "  return <Card>{'hi'}</Card>;\n"
            "};\n"
        )
        analysis = analyze(source, "v.tsx")
        assert ("View", "function") in [(s.qualified_name, s.kind.value) for s in analysis.symbols]
        assert import_pairs(analysis) == [("./ui", "Card", None)]

    def test_unterminated_template_is_a_parse_error_not_a_crash(self) -> None:
        source = "const s = `never closed\nfunction lost() { return 1; }\n"
        analysis = analyze(source)
        assert analysis.parse_error is not None
        assert "template" in analysis.parse_error

    def test_invalid_utf8_is_a_parse_error_not_a_crash(self) -> None:
        analysis = ANALYZER.analyze("m.ts", b"\xff\xfe\x00garbage", None)
        assert analysis.parse_error is not None
        assert "UTF-8" in analysis.parse_error

    def test_half_written_file_never_raises(self) -> None:
        source = "class Broken {\n  method() {\n     if (x) {\n"
        analysis = analyze(source)
        assert ("Broken", "class") in [(s.qualified_name, s.kind.value) for s in analysis.symbols]


class TestModuleNames:
    def test_extension_is_stripped(self) -> None:
        assert module_name_for("src/utils/taskHelpers.ts") == "src/utils/taskHelpers"
        assert module_name_for("data.json") == "data"


class TestResolution:
    def test_relative_import_resolves_across_directories(self) -> None:
        index = build_ts_index(("src/utils/helper.ts", "src/app.ts"))
        target = resolve_specifier(
            "../utils/helper",
            "src/controllers/task.ts",
            index,
            TypeScriptPaths(),
        )
        assert target == "src/utils/helper.ts"

    def test_directory_index_resolution(self) -> None:
        index = build_ts_index(("src/ui/index.ts",))
        target = resolve_specifier("./ui", "src/app.ts", index, TypeScriptPaths())
        assert target == "src/ui/index.ts"

    def test_explicit_json_suffix_resolves(self) -> None:
        index = build_ts_index(("src/data/fixtures.json",))
        target = resolve_specifier("./data/fixtures.json", "src/a.ts", index, TypeScriptPaths())
        assert target == "src/data/fixtures.json"

    def test_bare_module_is_external(self) -> None:
        target = resolve_specifier("react", "src/a.ts", build_ts_index(()), TypeScriptPaths())
        assert target is None

    def test_wildcard_alias_rewrites(self) -> None:
        paths = TypeScriptPaths.from_payload({"paths": {"@app/*": ["src/app/*"]}, "baseUrl": "."})
        index = build_ts_index(("src/app/core.ts",))
        target = resolve_specifier("@app/core", "src/other.ts", index, paths)
        assert target == "src/app/core.ts"

    def test_exact_alias(self) -> None:
        paths = TypeScriptPaths.from_payload({"paths": {"@config": ["src/config/index.ts"]}})
        index = build_ts_index(("src/config/index.ts",))
        target = resolve_specifier("@config", "src/other.ts", index, paths)
        assert target == "src/config/index.ts"

    def test_longer_alias_wins(self) -> None:
        paths = TypeScriptPaths.from_payload(
            {"paths": {"@app/*": ["src/app/*"], "@app/utils/*": ["src/shared/*"]}}
        )
        index = build_ts_index(("src/shared/strings.ts",))
        target = resolve_specifier("@app/utils/strings", "src/other.ts", index, paths)
        assert target == "src/shared/strings.ts"

    def test_base_url_rooted_import(self) -> None:
        paths = TypeScriptPaths.from_payload({"baseUrl": "src"})
        index = build_ts_index(("src/utils/a.ts",))
        target = resolve_specifier("utils/a", "src/deep/b.ts", index, paths)
        assert target == "src/utils/a.ts"

    def test_unresolvable_relative_import_is_none(self) -> None:
        target = resolve_specifier("./ghost", "src/a.ts", build_ts_index(()), TypeScriptPaths())
        assert target is None


class TestScanner:
    def test_in_code_rejects_strings_and_comments(self) -> None:
        view = SourceView("const a = 'x'; // y\n")
        assert view.in_code(4)
        quote = view.text.index("'")
        assert not view.in_code(quote + 1)
        comment = view.text.index("//")
        assert not view.in_code(comment + 1)

    def test_line_of(self) -> None:
        view = SourceView("one\ntwo\nthree")
        assert view.line_of(1) == 1
        assert view.line_of(5) == 2
        assert view.line_of(10) == 3

    def test_division_is_not_a_regex(self) -> None:
        view = SourceView("const half = total / 2; const next = a / b;\n")
        slash = view.text.index("/ 2")
        assert not view._regex_allowed(slash)

    def test_regex_after_return(self) -> None:
        view = SourceView("function f() { return /x/.test(y); }\n")
        slash = view.text.index("/x")
        assert view._regex_allowed(slash)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import a from 'b';", [("b", None, "a")]),
        ("export const x = 1;", []),
    ],
)
def test_import_shapes(source: str, expected: list) -> None:
    assert import_pairs(analyze(source)) == expected
