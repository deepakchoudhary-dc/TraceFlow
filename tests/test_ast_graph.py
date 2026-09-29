"""Module naming, import resolution, and the repository dependency graph."""

from __future__ import annotations

import time
from pathlib import Path

from traceflow.config import STATE_DIRNAME, Config
from traceflow.derived import AnalysisCache
from traceflow.git.repository import Repository
from traceflow.languages.base import ImportRef
from traceflow.languages.python.ast_graph import (
    build_dependency_graph,
    build_module_index,
    import_candidates,
    is_package_init,
    list_python_files,
    module_names_for,
    package_directories,
    resolve_import,
)

PACKAGES = frozenset({"auth", "auth/models"})


def ref(module: str, name: str | None = None, level: int = 0) -> ImportRef:
    return ImportRef(module=module, name=name, alias=None, level=level, line=1)


# --------------------------------------------------------------------------- naming


def test_packages_are_directories_containing_an_init_file() -> None:
    paths = ("auth/__init__.py", "auth/service.py", "tests/test_auth.py", "src/pkg/__init__.py")
    assert package_directories(paths) == frozenset({"auth", "src/pkg"})


def test_a_plain_directory_is_not_a_package() -> None:
    assert package_directories(("scripts/tool.py",)) == frozenset()


def test_module_names_skip_non_package_roots() -> None:
    """``src/traceflow/cli.py`` is importable as ``traceflow.cli``, not ``src.traceflow.cli``."""
    names = module_names_for("src/traceflow/cli.py", frozenset({"src/traceflow"}))
    assert names == ("src.traceflow.cli", "traceflow.cli")


def test_module_names_inside_a_package_carry_the_prefix() -> None:
    names = module_names_for("auth/models/user.py", PACKAGES)
    assert names == ("auth.models.user",)


def test_an_init_file_names_the_package_itself() -> None:
    assert module_names_for("auth/models/__init__.py", PACKAGES) == ("auth.models",)


def test_a_backslash_path_is_normalised() -> None:
    assert module_names_for("auth\\service.py", PACKAGES) == ("auth.service",)


def test_a_file_with_no_directory_uses_its_stem() -> None:
    assert module_names_for("main.py", frozenset()) == ("main",)


def test_the_module_index_maps_every_name() -> None:
    index = build_module_index(("auth/service.py", "src/pkg/mod.py"), frozenset({"src/pkg"}))
    assert index["auth.service"] == "auth/service.py"
    assert index["pkg.mod"] == "src/pkg/mod.py"


def test_a_name_collision_prefers_the_shallower_file() -> None:
    """``import c`` should find ``c.py``, not ``a/b/c.py`` which merely shares a leaf name."""
    index = build_module_index(("a/b/c.py", "c.py"), frozenset({"a", "a/b"}))
    assert index["c"] == "c.py"


# --------------------------------------------------------------------------- resolution


def test_a_member_import_prefers_the_submodule() -> None:
    """``from package import module`` is the idiom; the module is the real dependency."""
    assert import_candidates(ref("a.b", "c"), "m", is_package=False) == ("a.b.c", "a.b")


def test_a_plain_import_has_one_candidate() -> None:
    assert import_candidates(ref("a.b"), "m", is_package=False) == ("a.b",)


def test_a_relative_import_resolves_against_the_importing_package() -> None:
    assert import_candidates(ref("service", "thing", level=1), "auth.routes", is_package=False) == (
        "auth.service.thing",
        "auth.service",
    )


def test_a_parent_relative_import_walks_up_one_level() -> None:
    assert import_candidates(ref("shared", level=2), "auth.models.user", is_package=False) == (
        "auth.shared",
    )


def test_a_bare_relative_import_targets_the_package() -> None:
    assert import_candidates(ref("", "sibling", level=1), "auth.routes", is_package=False) == (
        "auth.sibling",
        "auth",
    )


def test_a_relative_import_without_a_module_name_is_unresolvable() -> None:
    assert import_candidates(ref("x", level=1), None, is_package=False) == ()


def test_resolution_finds_the_file() -> None:
    index = {"auth.service": "auth/service.py"}
    assert (
        resolve_import(ref("auth.service", "VALUE"), "auth.routes", index, is_package=False)
        == "auth/service.py"
    )


def test_resolution_returns_none_for_an_external_module() -> None:
    assert resolve_import(ref("requests"), "m", {}, is_package=False) is None


def test_resolution_falls_back_to_a_submodule() -> None:
    """``from pkg import thing`` where ``thing`` is itself a module."""
    index = {"pkg.thing": "pkg/thing.py"}
    assert resolve_import(ref("pkg", "thing"), "m", index, is_package=False) == "pkg/thing.py"


# --------------------------------------------------------------------------- repository graph


def test_python_files_come_from_git(repo: Repository, repo_root: Path) -> None:
    (repo_root / "extra.py").write_text("x = 1\n", encoding="utf-8", newline="\n")
    (repo_root / "notes.txt").write_text("hello\n", encoding="utf-8", newline="\n")

    files = list_python_files(repo)

    assert "app.py" in files
    assert "extra.py" in files
    assert "notes.txt" not in files


def test_ignored_files_are_excluded(repo: Repository, repo_root: Path) -> None:
    """``--exclude-standard`` means the repository's own .gitignore decides."""
    (repo_root / ".gitignore").write_text("generated/\n", encoding="utf-8", newline="\n")
    generated = repo_root / "generated"
    generated.mkdir()
    (generated / "gen.py").write_text("x = 1\n", encoding="utf-8", newline="\n")

    assert "generated/gen.py" not in list_python_files(repo)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def test_the_graph_resolves_an_absolute_internal_import(
    repo: Repository, repo_root: Path, config: Config
) -> None:
    _write(repo_root / "auth" / "__init__.py", "")
    _write(repo_root / "auth" / "service.py", "VALUE = 1\n")
    _write(repo_root / "auth" / "routes.py", "from auth.service import VALUE\n")

    graph = build_dependency_graph(repo, AnalysisCache(repo_root / STATE_DIRNAME), config)

    edges = {(edge.source_path, edge.target_path) for edge in graph.edges}
    assert ("auth/routes.py", "auth/service.py") in edges


def test_the_graph_resolves_a_relative_import(
    repo: Repository, repo_root: Path, config: Config
) -> None:
    _write(repo_root / "auth" / "__init__.py", "")
    _write(repo_root / "auth" / "service.py", "VALUE = 1\n")
    _write(repo_root / "auth" / "routes.py", "from .service import VALUE\n")

    graph = build_dependency_graph(repo, AnalysisCache(repo_root / STATE_DIRNAME), config)

    edges = {(edge.source_path, edge.target_path) for edge in graph.edges}
    assert ("auth/routes.py", "auth/service.py") in edges


def test_an_external_import_is_recorded_as_unresolved(
    repo: Repository, repo_root: Path, config: Config
) -> None:
    """Silently dropping it is how a graph becomes confidently incomplete."""
    _write(repo_root / "uses.py", "import requests\n")

    graph = build_dependency_graph(repo, AnalysisCache(repo_root / STATE_DIRNAME), config)

    assert "requests" in {item.module for item in graph.unresolved}


def test_an_unresolved_import_keeps_its_whole_reference(
    repo: Repository, repo_root: Path, config: Config
) -> None:
    """The dotted module alone cannot tell ``from a import b`` from ``import a.b``.

    Those two resolve differently against a different index, which is what finding the
    importers of a module a session deleted depends on.
    """
    _write(repo_root / "uses.py", "from auth import service\n")

    graph = build_dependency_graph(repo, AnalysisCache(repo_root / STATE_DIRNAME), config)

    item = next(candidate for candidate in graph.unresolved if candidate.module == "auth")
    assert item.reference.name == "service"
    assert item.line == 1


def test_a_file_deleted_from_the_working_tree_is_not_listed(
    repo: Repository, repo_root: Path
) -> None:
    """``git ls-files --cached`` reports the index, so an unstaged deletion is still there.

    Listing it would let an import resolve to a file that no longer exists, giving the
    graph an edge to a node it does not contain — and hiding the now-broken import.
    """
    assert "app.py" in list_python_files(repo)

    (repo_root / "app.py").unlink()

    assert "app.py" not in list_python_files(repo)


def test_a_module_importing_itself_adds_no_edge(
    repo: Repository, repo_root: Path, config: Config
) -> None:
    _write(repo_root / "selfy.py", "from selfy import nothing\n")

    graph = build_dependency_graph(repo, AnalysisCache(repo_root / STATE_DIRNAME), config)

    assert graph.edges == ()


def test_a_broken_file_is_recorded_and_does_not_stop_the_graph(
    repo: Repository, repo_root: Path, config: Config
) -> None:
    _write(repo_root / "broken.py", "def f(:\n")
    _write(repo_root / "fine.py", "x = 1\n")

    graph = build_dependency_graph(repo, AnalysisCache(repo_root / STATE_DIRNAME), config)

    assert any("broken.py" in error for error in graph.parse_errors)
    assert "fine.py" in {module.path for module in graph.modules}


def test_importers_and_imports_are_reported(
    repo: Repository, repo_root: Path, config: Config
) -> None:
    _write(repo_root / "a.py", "x = 1\n")
    _write(repo_root / "b.py", "from a import x\n")
    _write(repo_root / "c.py", "from a import x\n")

    graph = build_dependency_graph(repo, AnalysisCache(repo_root / STATE_DIRNAME), config)

    assert graph.importers_of("a.py") == ("b.py", "c.py")
    assert graph.imports_of("b.py") == ("a.py",)


def test_the_graph_round_trips_through_json(
    repo: Repository, repo_root: Path, config: Config
) -> None:
    _write(repo_root / "a.py", "x = 1\n")
    _write(repo_root / "b.py", "from a import x\n")

    payload = build_dependency_graph(
        repo, AnalysisCache(repo_root / STATE_DIRNAME), config
    ).to_json()

    assert payload["modules"]
    assert payload["edges"][0]["target_path"] == "a.py"  # type: ignore[index]


def test_a_second_build_reuses_the_cache(repo: Repository, repo_root: Path, config: Config) -> None:
    _write(repo_root / "a.py", "def f():\n    return 1\n")

    cache = AnalysisCache(repo_root / STATE_DIRNAME)
    first = build_dependency_graph(repo, cache, config)
    cached_files = [path for path in cache.root.rglob("*.json") if path.is_file()]
    second = build_dependency_graph(repo, cache, config)

    assert first.modules == second.modules
    assert cached_files, "the first build should have populated the cache"


def test_an_unusual_filename_is_listed_verbatim(repo: Repository, repo_root: Path) -> None:
    """plan.md §68 asks for malicious filenames. This is the property that makes them safe.

    ``git ls-files -z`` does not quote paths, so a name git would otherwise render as
    ``"caf\\303\\251.py"`` arrives as ``café.py``. Without the ``-z``, every name here would
    be a path that does not exist — the listing would silently point at nothing, and the
    analysis would report an empty repository rather than an error.
    """
    names = [
        "café.py",
        "日本語.py",
        "a&b's.py",
        "sp ace.py",
        "hash#tag.py",
        "plus+minus-.py",
        "brack[et].py",
        "dollar$.py",
        "tilde~.py",
    ]
    for name in names:
        _write(repo_root / name, "def f():\n    return 1\n")

    listed = list_python_files(repo)

    for name in names:
        assert name in listed, name


def test_the_module_index_handles_ten_thousand_paths() -> None:
    """plan.md §68 asks for a 10,000-file scale check. This is the O(n) part of it.

    Synthetic paths rather than a real repository: what could go accidentally quadratic is
    the indexing itself, and building 10,000 files and a git repository to measure a pure
    function would spend minutes of filesystem work learning the same thing. The bound is
    a hundred times the work this takes, so it catches a quadratic regression without
    turning a slow machine into a failure.
    """
    paths = tuple(f"pkg_{index // 100:03d}/mod_{index:05d}.py" for index in range(10_000))

    started = time.monotonic()
    packages = package_directories(paths)
    index = build_module_index(paths, packages)
    elapsed = time.monotonic() - started

    assert index["mod_09999"] == "pkg_099/mod_09999.py"
    assert index["pkg_000.mod_00000"] == "pkg_000/mod_00000.py"
    assert len(index) >= 10_000
    assert elapsed < 10.0, f"indexing 10,000 paths took {elapsed:.1f}s"


# --------------------------------------------------------------------------- relative imports


def test_a_package_init_resolves_its_own_relative_imports() -> None:
    """``from . import x`` inside a package ``__init__`` means ``<package>.x``.

    The module name cannot tell ``pkg/__init__.py`` from ``pkg.py`` — ``module_names_for``
    strips the ``__init__`` — so the file path is what decides where the dots start. Getting
    it wrong sent this import nowhere, and resolved the next one to an unrelated file:

        pkg/sub/__init__.py:  from . import deep   ->  pkg/__init__.py   (should be pkg/sub/deep.py)
    """
    paths = ("pkg/__init__.py", "pkg/helper.py", "pkg/sub/__init__.py", "pkg/sub/deep.py")
    packages = package_directories(paths)
    index = build_module_index(paths, packages)

    assert (
        resolve_import(
            ref("", "helper", level=1), "pkg", index, is_package=is_package_init("pkg/__init__.py")
        )
        == "pkg/helper.py"
    )
    assert (
        resolve_import(
            ref("", "deep", level=1),
            "pkg.sub",
            index,
            is_package=is_package_init("pkg/sub/__init__.py"),
        )
        == "pkg/sub/deep.py"
    )


def test_a_package_init_climbs_out_with_two_dots() -> None:
    """``from .. import x`` in ``pkg/sub/__init__.py`` means ``pkg.x``, not ``sub.x``."""
    paths = ("pkg/__init__.py", "pkg/other.py", "pkg/sub/__init__.py")
    index = build_module_index(paths, package_directories(paths))

    assert (
        resolve_import(
            ref("", "other", level=2),
            "pkg.sub",
            index,
            is_package=is_package_init("pkg/sub/__init__.py"),
        )
        == "pkg/other.py"
    )


def test_an_ordinary_module_still_climbs_out_of_its_package() -> None:
    """The flag only changes the answer for a package ``__init__``; everything else is as it was."""
    paths = ("pkg/__init__.py", "pkg/helper.py", "pkg/other.py")
    index = build_module_index(paths, package_directories(paths))

    assert import_candidates(ref("", "other", level=1), "pkg.helper", is_package=False) == (
        "pkg.other",
        "pkg",
    )
    assert (
        resolve_import(ref("", "other", level=1), "pkg.helper", index, is_package=False)
        == "pkg/other.py"
    )


def test_a_toplevel_module_has_no_package_to_be_relative_to() -> None:
    """``from . import x`` in a module that is not in a package cannot resolve."""
    assert import_candidates(ref("", "x", level=1), "toplevel", is_package=False) == ()
    assert import_candidates(ref("", "x", level=2), "toplevel", is_package=False) == ()


def test_is_package_init_recognises_only_init_files() -> None:
    assert is_package_init("pkg/__init__.py") is True
    assert is_package_init("pkg/sub/__init__.pyi") is True
    assert is_package_init("pkg/__init__.pyc") is False
    assert is_package_init("pkg/module.py") is False
    assert is_package_init("pkg/__init___extra.py") is False
