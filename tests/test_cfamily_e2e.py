"""End-to-end C-family change intelligence (plan.md §79's promise, universal proof).

The TypeScript e2e suite proved the pipeline on one added language; this one proves
the promise plan.md makes about the *architecture*: when an agent changes a helper
in Go, Java, Rust or C#, TraceFlow names the caller — through the same walk, with
no per-language pipeline beyond a profile and a resolver.

The sequence in every test is the watcher's own: commit a clean tree, capture the
baseline from it, make the edit, then analyse. A baseline captured after the edit
would describe the edited file as the starting point, and the session would be
empty — which is exactly the difference plan.md §14 exists to protect.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import commit, write_file

from traceflow.analysis.impact import build_impact_report
from traceflow.analysis.symbols import analyse_session_modules
from traceflow.blobs import BlobStore
from traceflow.config import STATE_DIRNAME, Config
from traceflow.derived import AnalysisCache
from traceflow.git.baseline import Baseline, capture_baseline
from traceflow.git.diff import collect_changes
from traceflow.git.repository import Repository
from traceflow.languages.python.ast_graph import list_repository_files


@pytest.fixture
def blobs(repo_root: Path) -> BlobStore:
    return BlobStore(repo_root / STATE_DIRNAME)


@pytest.fixture
def cache(repo_root: Path) -> AnalysisCache:
    return AnalysisCache(repo_root / STATE_DIRNAME)


def _baseline(repo: Repository, blobs: BlobStore, config: Config) -> Baseline:
    return capture_baseline(repo, blobs, repo.working_tree_state(config.ignore), config)


def _analyse(
    repo: Repository,
    baseline: Baseline,
    blobs: BlobStore,
    cache: AnalysisCache,
    config: Config,
):
    supported = list_repository_files(repo)
    changes = collect_changes(repo, baseline, repo.working_tree_state(config.ignore), blobs, config)
    session = analyse_session_modules(
        repo, baseline, changes, blobs, cache, config, supported_paths=supported
    )
    impact = build_impact_report(
        repo, baseline, changes, blobs, cache, config, supported_paths=supported
    )
    return changes, session, impact


_GO_STORE = "package store\n\nfunc Write(x int) error {\n    return nil\n}\n"

_GO_STORE_CHANGED = "package store\n\nfunc Write(x int, attempt int) error {\n    return nil\n}\n"

_GO_AUTH = (
    'package auth\n\nimport (\n\t"github.com/user/repo/internal/store"\n)\n\n'
    "type Repo struct {\n\tdb int\n}\n\n"
    "func (r *Repo) Save(u int) error {\n\treturn store.Write(u)\n}\n"
)

_JAVA_STORE = (
    "package com.example.db;\n\npublic class Store {\n"
    "    public static int query() {\n        return 1;\n    }\n}\n"
)

_JAVA_STORE_CHANGED = (
    "package com.example.db;\n\npublic class Store {\n"
    "    public static int query(int limit) {\n        return 1;\n    }\n}\n"
)

_JAVA_USER = (
    "package com.example.auth;\n\nimport com.example.db.Store;\n\npublic class UserService {\n"
    "    public int findAll() {\n        return Store.query();\n    }\n}\n"
)

_RUST_SERVICE = "pub fn login(user: &str) -> bool {\n    user.len() > 0\n}\n"

_RUST_SERVICE_CHANGED = "pub fn login(user: &str, strict: bool) -> bool {\n    user.len() > 0\n}\n"

_RUST_HANDLERS = (
    'use crate::auth::service::login;\n\npub fn start() -> bool {\n    login("admin")\n}\n'
)

_CS_REPO = (
    "namespace App.Data;\n\npublic class Repo {\n"
    '    public static bool Validate(string role) {\n        return role == "admin";\n    }\n}\n'
)

_CS_REPO_CHANGED = (
    "namespace App.Data;\n\npublic class Repo {\n"
    "    public static bool Validate(\n"
    "        string role, bool strict = false\n"
    '    ) {\n        return role == "admin";\n    }\n}\n'
)

_CS_PERMISSION = (
    "namespace App.Auth;\n\nusing App.Data;\n\npublic class PermissionService {\n"
    "    public bool CheckAsync(string role) {\n        return Repo.Validate(role);\n    }\n}\n"
)


def go_repo(repo_root: Path) -> None:
    write_file(repo_root, "go.mod", "module github.com/user/repo\n\ngo 1.21\n")
    write_file(repo_root, "internal/store/store.go", _GO_STORE)
    write_file(repo_root, "internal/auth/repo.go", _GO_AUTH)
    commit(repo_root, "go sources")


def java_repo(repo_root: Path) -> None:
    write_file(repo_root, "src/com/example/db/Store.java", _JAVA_STORE)
    write_file(repo_root, "src/com/example/auth/UserService.java", _JAVA_USER)
    commit(repo_root, "java sources")


def rust_repo(repo_root: Path) -> None:
    write_file(repo_root, "src/auth/service.rs", _RUST_SERVICE)
    write_file(repo_root, "src/auth/mod.rs", "pub mod service;\n")
    write_file(repo_root, "src/lib.rs", "pub mod auth;\n")
    write_file(repo_root, "src/handlers.rs", _RUST_HANDLERS)
    commit(repo_root, "rust sources")


def csharp_repo(repo_root: Path) -> None:
    write_file(repo_root, "App/Data/Repo.cs", _CS_REPO)
    write_file(repo_root, "App/Auth/PermissionService.cs", _CS_PERMISSION)
    commit(repo_root, "csharp sources")


def test_changed_go_helper_reaches_its_caller(
    repo_root: Path, repo: Repository, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    go_repo(repo_root)
    baseline = _baseline(repo, blobs, config)
    write_file(repo_root, "internal/store/store.go", _GO_STORE_CHANGED)

    _changes, session, impact = _analyse(repo, baseline, blobs, cache, config)

    symbol_changes = {
        (change.qualified_name, change.change.value)
        for change in session.modules[0].changes.changes
    }
    assert ("Write", "signature_changed") in symbol_changes

    callers = [node for node in impact.nodes if node.path == "internal/auth/repo.go"]
    assert callers, "the Go file calling the changed helper must be reported"
    assert any(node.reason == "signature_changed" for node in callers)


def test_changed_java_class_reaches_its_caller(
    repo_root: Path, repo: Repository, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    java_repo(repo_root)
    baseline = _baseline(repo, blobs, config)
    write_file(repo_root, "src/com/example/db/Store.java", _JAVA_STORE_CHANGED)

    _changes, session, impact = _analyse(repo, baseline, blobs, cache, config)

    symbol_changes = {
        (change.qualified_name, change.change.value)
        for change in session.modules[0].changes.changes
    }
    assert ("Store::query", "signature_changed") in symbol_changes

    java_path = "src/com/example/auth/UserService.java"
    callers = [node for node in impact.nodes if node.path == java_path]
    assert callers, "the Java caller of the changed class must be reported"
    assert any(node.reason == "signature_changed" for node in callers)


def test_changed_rust_function_reaches_its_caller(
    repo_root: Path, repo: Repository, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    rust_repo(repo_root)
    baseline = _baseline(repo, blobs, config)
    write_file(repo_root, "src/auth/service.rs", _RUST_SERVICE_CHANGED)

    _changes, session, impact = _analyse(repo, baseline, blobs, cache, config)

    assert [module.path for module in session.modules] == ["src/auth/service.rs"]
    symbol_changes = {
        (change.qualified_name, change.change.value)
        for change in session.modules[0].changes.changes
    }
    assert ("login", "signature_changed") in symbol_changes

    callers = [node for node in impact.nodes if node.path == "src/handlers.rs"]
    assert callers, "the Rust caller of the changed function must be reported"
    assert any(node.reason == "signature_changed" for node in callers)
    assert any(node.symbol == "start" for node in callers)


def test_changed_csharp_method_reaches_its_caller(
    repo_root: Path, repo: Repository, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    csharp_repo(repo_root)
    baseline = _baseline(repo, blobs, config)
    write_file(repo_root, "App/Data/Repo.cs", _CS_REPO_CHANGED)

    _changes, _session, impact = _analyse(repo, baseline, blobs, cache, config)

    callers = [node for node in impact.nodes if node.path == "App/Auth/PermissionService.cs"]
    assert callers, "the C# caller of the changed method must be reported"
    assert any(node.reason == "signature_changed" for node in callers)


def test_deleted_modules_leave_dangling_importers_across_languages(
    repo_root: Path, repo: Repository, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """One session deleting a Rust module and a Go file names both files left importing."""
    go_repo(repo_root)
    rust_repo(repo_root)
    baseline = _baseline(repo, blobs, config)

    (repo_root / "src/auth/service.rs").unlink()
    (repo_root / "internal/store/store.go").unlink()

    _changes, session, impact = _analyse(repo, baseline, blobs, cache, config)

    # The session is one comparison over both languages.
    assert {module.path for module in session.modules} == {
        "internal/store/store.go",
        "src/auth/service.rs",
    }
    kinds = session.analyzer.split("+")
    assert "go-2" in kinds
    assert "rust-2" in kinds

    dangling = {node.path for node in impact.nodes if node.reason == "dangling_import"}
    assert "internal/auth/repo.go" in dangling
    assert "src/handlers.rs" in dangling
    assert all(node.is_obligation for node in impact.nodes if node.reason == "dangling_import")

    # An import explained by a removal is not an external dependency.
    assert impact.unresolved_imports == 0


def test_cfamily_python_and_typescript_in_one_registry(
    repo_root: Path, repo: Repository, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """The universal registry leaves the earlier languages exactly as they were."""
    go_repo(repo_root)
    write_file(repo_root, "legacy.py", "def run():\n    return 1\n")
    write_file(
        repo_root,
        "util.ts",
        "export function bump(n: number): number {\n  return n + 1;\n}\n",
    )
    commit(repo_root, "add python and ts")
    baseline = _baseline(repo, blobs, config)
    write_file(repo_root, "legacy.py", "def run():\n    return 2\n")
    write_file(
        repo_root,
        "util.ts",
        "export function bump(n: number): number {\n  return n + 2;\n}\n",
    )

    _changes, session, _impact = _analyse(repo, baseline, blobs, cache, config)

    assert {module.path for module in session.modules} == {"legacy.py", "util.ts"}
    kinds = session.analyzer.split("+")
    assert "python-1" in kinds
    assert "typescript-1" in kinds
