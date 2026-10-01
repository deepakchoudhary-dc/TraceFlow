"""C-family engine extraction, per language (plan.md §66).

These tests pin the shapes each language actually writes and the engine must
survive: Go's package and import clauses in front of a file's first declaration,
Java's modifier walls, Rust's ``impl`` scopes, C#'s async generics. The regression
cases here were each a real defect — a first declaration named by the package
clause, an import block swallowing every declaration after it — which is why they
are stated in terms of whole files rather than isolated headers.
"""

from __future__ import annotations

import pytest

from traceflow.analysis.models import is_test_path
from traceflow.languages.cfamily.analyzer import (
    CFamilyAnalyzer,
    csharp_profile,
    go_profile,
    java_profile,
    rust_profile,
)
from traceflow.languages.registry import analyzer_for, default_registry


@pytest.fixture
def go() -> CFamilyAnalyzer:
    return CFamilyAnalyzer(go_profile())


@pytest.fixture
def java() -> CFamilyAnalyzer:
    return CFamilyAnalyzer(java_profile())


@pytest.fixture
def rust() -> CFamilyAnalyzer:
    return CFamilyAnalyzer(rust_profile())


@pytest.fixture
def csharp() -> CFamilyAnalyzer:
    return CFamilyAnalyzer(csharp_profile())


def names(analysis) -> list[tuple[str, str]]:
    return [(symbol.qualified_name, symbol.kind.value) for symbol in analysis.symbols]


def test_go_first_declaration_after_package_clause(go: CFamilyAnalyzer) -> None:
    """A header read from the file start begins with ``package x`` — the declaration
    after it is still the declaration, not part of the clause."""
    source = b"package store\n\nfunc Write(x int) error {\n    return nil\n}\n"
    assert names(go.analyze("store.go", source, None)) == [("Write", "function")]


def test_go_package_and_import_block_do_not_swallow_declarations(go: CFamilyAnalyzer) -> None:
    """``import ( … )`` has no brace of its own, so a header that begins with the
    clauses once began with the whole file. The type and the method after the
    block must both survive."""
    source = (
        b'package p\n\nimport (\n\t"a"\n\t"b"\n)\n\ntype T struct {\n\tn int\n}\n'
        b"\nfunc (t T) M() int { return t.n }\n"
    )
    assert names(go.analyze("t.go", source, None)) == [("T", "class"), ("T::M", "method")]


def test_go_aliased_import_clause_is_dropped(go: CFamilyAnalyzer) -> None:
    source = b'package p\n\nimport sv "github.com/x/y/sv"\n\nfunc B() { sv.Do() }\n'
    analysis = go.analyze("b.go", source, None)
    assert names(analysis) == [("B", "function")]
    # The call inside B is a call; B's own declaration header is not.
    assert [call.name for call in analysis.calls] == ["sv.Do"]


def test_go_receiver_method_is_qualified_by_its_type(go: CFamilyAnalyzer) -> None:
    source = (
        b"package auth\n\ntype Repo struct {\n\tdb int\n}\n"
        b"\nfunc (r *Repo) Save(u int) error {\n\treturn nil\n}\n"
    )
    assert names(go.analyze("repo.go", source, None)) == [
        ("Repo", "class"),
        ("Repo::Save", "method"),
    ]


def test_java_modifier_wall_and_constructor(java: CFamilyAnalyzer) -> None:
    source = (
        b"package com.example.auth;\n\nimport com.example.db.Store;\n\n"
        b"public class UserService {\n    private final Store store;\n\n"
        b"    public UserService(Store store) {\n        this.store = store;\n    }\n\n"
        b"    public int findAll() {\n        return store.query();\n    }\n}\n"
    )
    analysis = java.analyze("UserService.java", source, None)
    assert ("UserService", "class") in names(analysis)
    assert ("UserService::findAll", "method") in names(analysis)
    # The constructor is a method of the class by the same rule.
    assert ("UserService::UserService", "method") in names(analysis)
    assert [item.module for item in analysis.imports] == ["com.example.db.Store"]


def test_rust_impl_methods_are_qualified(rust: CFamilyAnalyzer) -> None:
    source = (
        b"pub struct Wrapper {\n    name: String,\n}\n\n"
        b"impl Wrapper {\n    pub fn new(name: String) -> Self {\n        Self { name }\n    }\n\n"
        b"    pub fn persist(&self) -> bool {\n        true\n    }\n}\n"
    )
    assert names(rust.analyze("wrapper.rs", source, None)) == [
        ("Wrapper", "class"),
        ("Wrapper::new", "method"),
        ("Wrapper::persist", "method"),
    ]


def test_rust_free_function_and_use_paths(rust: CFamilyAnalyzer) -> None:
    source = (
        b'use crate::auth::service::login;\n\npub fn start() -> bool {\n    login("admin")\n}\n'
    )
    analysis = rust.analyze("handlers.rs", source, None)
    assert names(analysis) == [("start", "function")]
    # The use path is stored dot-separated, resolution's own shape.
    assert [item.module for item in analysis.imports] == ["crate.auth.service.login"]
    # ``login(...)`` is the call; ``start`` is a declaration (its ``-> bool``
    # tail is the shape that says so), not a call site of itself.
    assert [call.name for call in analysis.calls] == ["login"]


def test_csharp_async_method_with_generics(csharp: CFamilyAnalyzer) -> None:
    source = (
        b"namespace App.Auth;\n\nusing App.Data;\n\npublic class PermissionService {\n"
        b"    public async Task<bool> CheckAsync(string role) {\n        return true;\n    }\n}\n"
    )
    analysis = csharp.analyze("PermissionService.cs", source, None)
    assert ("PermissionService", "class") in names(analysis)
    assert ("PermissionService::CheckAsync", "method") in names(analysis)
    assert [item.module for item in analysis.imports] == ["App.Data"]


def test_go_call_extraction_excludes_declaration_headers(go: CFamilyAnalyzer) -> None:
    """``func``/``import`` sit before braces too — they are keywords, not calls."""
    source = b'package p\n\nimport "fmt"\n\nfunc A() {\n\tfmt.Println()\n}\n'
    assert [call.name for call in go.analyze("a.go", source, None).calls] == ["fmt.Println"]


def test_registry_dispatches_by_suffix() -> None:
    registry = default_registry()
    claims = {
        "a.go": "go",
        "A.java": "java",
        "a.rs": "rust",
        "A.cs": "csharp",
        "a.py": "python",
        "a.ts": "typescript",
    }
    for path, expected in claims.items():
        engine = analyzer_for(path, registry)
        assert engine is not None and engine.name == expected, path
    assert analyzer_for("a.rb", registry) is None


def test_go_test_files_are_classified_as_tests() -> None:
    assert is_test_path("internal/store/store_test.go")
    assert not is_test_path("internal/store/store.go")
    assert is_test_path("tests/test_auth.py")  # the Python rule is untouched
