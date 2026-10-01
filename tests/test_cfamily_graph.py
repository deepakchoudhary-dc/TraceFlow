"""C-family import resolution (plan.md §66, §69).

The resolver is per-language because the four disagree about what an import path
is relative to. These tests pin each language's answer — including the two shapes
that hid real defects: Java/C# using a full namespace *alone*, and a Rust use path
that names an item rather than a module.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import write_file

from traceflow.config import STATE_DIRNAME, default_config
from traceflow.derived import AnalysisCache
from traceflow.git.repository import Repository
from traceflow.languages.cfamily.graph import CFamilyFiles, cfamily_languages_for
from traceflow.languages.python.ast_graph import (
    build_dependency_graph,
    list_repository_files,
)


@pytest.fixture
def four_language_repo(repo_root: Path, repo: Repository) -> Repository:
    """One repository containing all four languages, each importing within itself."""
    write_file(repo_root, "go.mod", "module github.com/user/repo\n\ngo 1.21\n")
    write_file(
        repo_root,
        "internal/store/store.go",
        "package store\n\nfunc Write(x int) error {\n    return nil\n}\n",
    )
    write_file(
        repo_root,
        "internal/auth/repo.go",
        'package auth\n\nimport (\n\t"github.com/user/repo/internal/store"\n)\n\n'
        "type Repo struct {\n\tdb int\n}\n\n"
        "func (r *Repo) Save(u int) error {\n\treturn store.Write(u)\n}\n",
    )
    write_file(
        repo_root,
        "src/com/example/db/Store.java",
        "package com.example.db;\n\npublic class Store {\n}\n",
    )
    write_file(
        repo_root,
        "src/com/example/auth/UserService.java",
        "package com.example.auth;\n\nimport com.example.db.Store;\n\n"
        "public class UserService {\n}\n",
    )
    write_file(
        repo_root,
        "src/auth/service.rs",
        "pub fn login(user: &str) -> bool {\n    user.len() > 0\n}\n",
    )
    write_file(repo_root, "src/auth/mod.rs", "pub mod service;\n")
    write_file(repo_root, "src/lib.rs", "pub mod auth;\n")
    write_file(
        repo_root,
        "src/handlers.rs",
        'use crate::auth::service::login;\n\npub fn start() -> bool {\n    login("admin")\n}\n',
    )
    write_file(repo_root, "App/Data/Repo.cs", "namespace App.Data;\n\npublic class Repo {\n}\n")
    write_file(
        repo_root,
        "App/Auth/PermissionService.cs",
        "namespace App.Auth;\n\nusing App.Data;\n\npublic class PermissionService {\n}\n",
    )
    from conftest import commit

    commit(repo_root, "four languages")
    return repo


def test_dispatcher_claims_each_suffix() -> None:
    assert cfamily_languages_for("a.go") == "go"
    assert cfamily_languages_for("A.JAVA") == "java"
    assert cfamily_languages_for("a.rs") == "rust"
    assert cfamily_languages_for("A.cs") == "csharp"
    assert cfamily_languages_for("a.ts") is None
    assert cfamily_languages_for("a.py") is None


def test_module_names_are_extensionless_paths() -> None:
    assert CFamilyFiles.module_name_for("src/auth/service.rs") == "src/auth/service"
    assert CFamilyFiles.module_name_for("internal/store/store.go") == "internal/store/store"
    assert CFamilyFiles.module_name_for("App/Data/Repo.cs") == "App/Data/Repo"
    assert (
        CFamilyFiles.module_name_for("src/com/example/db/Store.java") == "src/com/example/db/Store"
    )


def test_go_resolves_under_the_module_prefix_and_externals_stay_external(
    four_language_repo: Repository,
) -> None:
    files = CFamilyFiles.of(list_repository_files(four_language_repo), four_language_repo)
    index = files.indexes["go"]

    assert index.resolve("github.com/user/repo/internal/store") == "internal/store/store.go"
    assert index.resolve("github.com/other/repo/store") is None
    assert index.resolve("fmt") is None


def test_go_directory_resolves_whatever_the_file_is_named(
    repo_root: Path, repo: Repository
) -> None:
    """Go has no ``<package>.go`` rule: ``server.go`` inside ``pkg/`` is a layout in
    practice, and the import of ``pkg`` must still resolve."""
    write_file(repo_root, "go.mod", "module example.com/app\n\ngo 1.21\n")
    write_file(repo_root, "pkg/server.go", "package pkg\n\nfunc Serve() {}\n")
    write_file(
        repo_root,
        "main.go",
        'package main\n\nimport "example.com/app/pkg"\n\nfunc main() {\n\tpkg.Serve()\n}\n',
    )
    from conftest import commit

    commit(repo_root, "go pkg")
    files = CFamilyFiles.of(list_repository_files(repo), repo)

    assert files.resolve("example.com/app/pkg", "main.go") == "pkg/server.go"


def test_java_resolves_by_package_declaration(four_language_repo: Repository) -> None:
    files = CFamilyFiles.of(list_repository_files(four_language_repo), four_language_repo)

    assert (
        files.resolve("com.example.db.Store", "src/com/example/auth/UserService.java")
        == "src/com/example/db/Store.java"
    )
    # An unresolvable import is external, and None is the honest record of it.
    assert files.resolve("java.util.List", "src/com/example/auth/UserService.java") is None


def test_csharp_resolves_a_namespace_used_alone(four_language_repo: Repository) -> None:
    files = CFamilyFiles.of(list_repository_files(four_language_repo), four_language_repo)

    # ``using App.Data;`` names the whole namespace — tried before the last-dot split.
    assert files.resolve("App.Data", "App/Auth/PermissionService.cs") == "App/Data/Repo.cs"


def test_rust_resolves_item_paths_to_their_module(four_language_repo: Repository) -> None:
    files = CFamilyFiles.of(list_repository_files(four_language_repo), four_language_repo)

    # ``use crate::auth::service::login`` names an item; the module prefix is the file.
    assert files.resolve("crate.auth.service.login", "src/handlers.rs") == "src/auth/service.rs"
    assert files.resolve("serde::ser::Serialize", "src/handlers.rs") is None


def test_rust_climbs_but_stops_at_a_declared_module(four_language_repo: Repository) -> None:
    files = CFamilyFiles.of(list_repository_files(four_language_repo), four_language_repo)

    # A module prefix that resolves is the answer.
    assert files.resolve("crate.auth", "src/handlers.rs") == "src/auth/mod.rs"
    # A declared module whose file is gone is a broken import, not a reason to
    # climb to the parent — the dangling-import reconstruction depends on None.
    (four_language_repo.root / "src/auth/service.rs").unlink()
    broken = CFamilyFiles.of(list_repository_files(four_language_repo), four_language_repo)
    assert broken.resolve("crate.auth.service.login", "src/handlers.rs") is None


def test_graph_connects_all_four_languages(four_language_repo: Repository) -> None:
    cache = AnalysisCache(four_language_repo.root / STATE_DIRNAME)
    graph = build_dependency_graph(
        four_language_repo,
        cache,
        default_config(),
        supported_paths=list_repository_files(four_language_repo),
    )

    edges = {(edge.source_path, edge.target_path) for edge in graph.edges}
    assert ("internal/auth/repo.go", "internal/store/store.go") in edges
    assert ("src/com/example/auth/UserService.java", "src/com/example/db/Store.java") in edges
    assert ("src/handlers.rs", "src/auth/service.rs") in edges
    assert ("App/Auth/PermissionService.cs", "App/Data/Repo.cs") in edges
    # Every import in the fixture is internal; nothing is unresolved.
    assert graph.unresolved == ()
