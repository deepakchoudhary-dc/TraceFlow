"""Repository-level dependency graph (plan.md §17, §59).

Given the Python files in a repository, work out which module each one is, what each
imports, and resolve those imports to files in the same repository. The result is a
directed graph of files, which is what the impact engine walks.

Resolution is deliberately honest about its limits. An import that cannot be resolved
to a file in the repository is recorded as unresolved rather than dropped, because
"this module depends on something I could not find" is a fact worth reporting — and
silently discarding it is how a graph becomes confidently incomplete (plan.md §69).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from traceflow.blobs import digest_of
from traceflow.config import Config
from traceflow.derived import AnalysisCache
from traceflow.git.repository import Repository
from traceflow.languages.base import ImportRef, ModuleAnalysis, module_analysis_from_json
from traceflow.languages.python.analyzer import PythonAnalyzer
from traceflow.languages.typescript.analyzer import TypeScriptAnalyzer
from traceflow.languages.typescript.graph import (
    TypeScriptFiles,
    module_name_for,
    resolve_specifier,
)

if TYPE_CHECKING:
    from traceflow.languages.registry import Analyzer

PACKAGE_MARKERS = frozenset({"__init__.py", "__init__.pyi"})
PYTHON_SUFFIXES = (".py", ".pyi")


def _default_registry() -> tuple[Analyzer, ...]:
    """The registry, imported at call time.

    Lazily, because the registry imports this package's analyzer, and an eager
    import here would chase the registry through the package ``__init__`` while
    that module is still being initialised. One indirection buys a clean import
    order in both directions.
    """
    from traceflow.languages.registry import default_registry

    return default_registry()


def _analyzer_for(
    path: str, entries: tuple[Analyzer, ...] | None
) -> PythonAnalyzer | TypeScriptAnalyzer | None:
    """The analyzer that handles *path* — see :func:`_default_registry` for the why."""
    from traceflow.languages.registry import analyzer_for

    return analyzer_for(path, entries)


@dataclass(frozen=True)
class ModuleNode:
    """One Python file in the repository."""

    module_name: str
    path: str
    digest: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImportEdge:
    """One resolved import, from one file to another."""

    source_path: str
    target_path: str
    module: str
    line: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UnresolvedImport:
    """An import that points outside the repository, or at something that is not there.

    The whole reference is kept rather than only the dotted module name, because an
    unresolved import is exactly the case whose target has to be reconstructed later.
    ``"auth.service"`` alone cannot distinguish ``import auth.service`` from
    ``from auth import service``, and those two resolve differently against a different
    index — which is what finding the importers of a *deleted* module requires.
    """

    source_path: str
    reference: ImportRef

    @property
    def module(self) -> str:
        """The dotted module named by the import."""
        return self.reference.module

    @property
    def line(self) -> int:
        return self.reference.line

    def to_json(self) -> dict[str, Any]:
        return {"source_path": self.source_path, **self.reference.to_json()}


@dataclass(frozen=True)
class DependencyGraph:
    """Files as nodes, imports as edges."""

    modules: tuple[ModuleNode, ...]
    edges: tuple[ImportEdge, ...]
    unresolved: tuple[UnresolvedImport, ...]
    parse_errors: tuple[str, ...] = ()

    @property
    def module_count(self) -> int:
        return len(self.modules)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def importers_of(self, path: str) -> tuple[str, ...]:
        """Files that import *path* — the first-order dependents."""
        return tuple(sorted({edge.source_path for edge in self.edges if edge.target_path == path}))

    def imports_of(self, path: str) -> tuple[str, ...]:
        """Files that *path* imports."""
        return tuple(sorted({edge.target_path for edge in self.edges if edge.source_path == path}))

    def to_json(self) -> dict[str, Any]:
        return {
            "modules": [module.to_json() for module in self.modules],
            "edges": [edge.to_json() for edge in self.edges],
            "unresolved": [item.to_json() for item in self.unresolved],
            "parse_errors": list(self.parse_errors),
        }


def normalise(path: str) -> str:
    """Repository-relative paths are compared as POSIX strings throughout."""
    return path.replace("\\", "/").strip("/")


def package_directories(paths: tuple[str, ...]) -> frozenset[str]:
    """Every directory that contains an ``__init__`` file, and is therefore a package."""
    directories: set[str] = set()
    for path in paths:
        normalised = normalise(path)
        name = normalised.rsplit("/", 1)[-1]
        if name in PACKAGE_MARKERS:
            directories.add(normalised.rsplit("/", 1)[0] if "/" in normalised else "")
    return frozenset(directories)


def module_names_for(path: str, packages: frozenset[str]) -> tuple[str, ...]:
    """Every dotted name under which *path* could be imported, longest first.

    A file can be reachable under more than one name: ``src/traceflow/cli.py`` is both
    ``src.traceflow.cli`` and ``traceflow.cli``, depending on which directory the
    interpreter treats as the source root. Rather than guess which layout a project
    uses, both names are registered. A name that is never imported costs nothing;
    a name that is missing silently breaks resolution for a whole project layout.
    """
    normalised = normalise(path)
    for suffix in PYTHON_SUFFIXES:
        if normalised.endswith(suffix):
            normalised = normalised[: -len(suffix)]
            break

    parts = normalised.split("/")
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts or parts == [""]:
        return ()

    names: list[str] = []
    for start in range(len(parts)):
        if start and "/".join(parts[:start]) in packages:
            # Already inside a package, so the name must carry the package prefix.
            continue
        names.append(".".join(parts[start:]))
    return tuple(names)


def build_module_index(paths: tuple[str, ...], packages: frozenset[str]) -> dict[str, str]:
    """Map every importable dotted name to the file that provides it.

    On collision the shorter path wins: ``import c`` should find the top-level
    ``c.py`` rather than ``a/b/c.py``, which merely happens to share a leaf name.
    """
    index: dict[str, str] = {}
    for path in sorted(paths):
        for name in module_names_for(path, packages):
            existing = index.get(name)
            if existing is None or len(normalise(path)) < len(normalise(existing)):
                index[name] = normalise(path)
    return index


@dataclass(frozen=True)
class PythonFiles:
    """The repository's Python files, with the package layout and module index derived.

    One git call and two cheap derivations. They used to be made at three separate call
    sites, so a single ``analyze`` listed the repository three times to get one answer —
    and a git spawn on Windows costs about 120ms, which made it the largest avoidable cost
    in the command. Passing one of these down also closes a subtler gap: three listings
    taken at three instants can disagree if the tree moves underneath, and the analysis
    would then be reasoning from two different pictures of the repository.
    """

    paths: tuple[str, ...]
    packages: frozenset[str]
    index: dict[str, str]

    @classmethod
    def of(cls, repository: Repository) -> PythonFiles:
        paths = list_python_files(repository)
        packages = package_directories(paths)
        return cls(paths=paths, packages=packages, index=build_module_index(paths, packages))


def is_package_init(path: str) -> bool:
    """True when *path* is a package's ``__init__``.

    The file path is the only reliable answer, which is why this exists rather than a
    membership test on :func:`package_directories`. The *module name* cannot distinguish
    ``pkg/__init__.py`` from ``pkg.py`` — :func:`module_names_for` strips the ``__init__`` —
    and the package set holds *directory* paths, which do not match the module names a
    ``src/`` layout produces (``src/pkg/__init__.py`` is the module ``pkg`` but the directory
    ``src/pkg``). Relative-import resolution turns on exactly this distinction.
    """
    return normalise(path).rsplit("/", 1)[-1] in PACKAGE_MARKERS


def import_candidates(
    reference: ImportRef, importing_module: str | None, *, is_package: bool
) -> tuple[str, ...]:
    """Dotted names an import could refer to, most likely first.

    For ``from X import Y`` the submodule ``X.Y`` is tried before ``X`` itself. The common
    idiom is ``from package import module``, and there the dependency that matters is the
    module: resolving it to the package's ``__init__`` instead would point the dependency
    graph at the wrong file, and the impact walk could not follow it. ``from module import
    name`` still works, because ``X.Y`` only resolves when a submodule by that name exists
    — and a module and a package cannot share a name, so the fallback is always reached for
    an ordinary member import.

    *is_package* says the importing file is a package's ``__init__``, and it changes where a
    relative import starts. Level 1 is the importing module's own package: for a plain module
    that is the directory it sits in, but a package's ``__init__`` *is* its package, so the
    dots must not climb out of it. Treating the two the same way sent ``from . import helper``
    in ``pkg/__init__.py`` nowhere, and resolved ``from . import deep`` in
    ``pkg/sub/__init__.py`` to ``pkg/__init__.py`` — a relationship to an unrelated file.

    It is required rather than defaulted because a default would silently reproduce that
    defect at any call site that forgot it.
    """
    prefix: list[str] = []

    if reference.level:
        if importing_module is None:
            return ()
        parts = importing_module.split(".")
        depth = len(parts) if is_package else len(parts) - 1
        keep = depth - (reference.level - 1)
        if keep < 0:
            return ()
        prefix = parts[:keep]

    stem = [part for part in reference.module.split(".") if part]
    base = [*prefix, *stem]

    candidates: list[str] = []
    if reference.name:
        # `from a.b import c` — c is most often a submodule, a.b.c.
        if base:
            candidates.append(".".join([*base, reference.name]))
        elif prefix:
            candidates.append(".".join([*prefix, reference.name]))
    if base:
        # ...or c is a name inside a.b, so the dependency is on a.b itself.
        candidates.append(".".join(base))

    return tuple(dict.fromkeys(candidates))


def resolve_import(
    reference: ImportRef,
    importing_module: str | None,
    index: dict[str, str],
    *,
    is_package: bool,
) -> str | None:
    """Resolve an import to a file in this repository, or ``None`` when it is external.

    *is_package* must be true when the importing file is a package's ``__init__``; see
    :func:`import_candidates`. Getting it wrong resolves a relative import to the wrong file
    or to nothing, and a wrong resolution is worse than a miss — it is a relationship the
    artifacts do not support.
    """
    for candidate in import_candidates(reference, importing_module, is_package=is_package):
        target = index.get(candidate)
        if target is not None:
            return target
    return None


def list_repository_files(repository: Repository) -> tuple[str, ...]:
    """Every file git considers part of the repository, and that still exists.

    ``--exclude-standard`` applies the repository's real ``.gitignore``, so build
    output and virtual environments are excluded by the same rules everything else
    uses rather than by a second list that would drift.

    The existence check is not redundant. ``git ls-files --cached`` reports the *index*,
    so a file deleted from the working tree but not yet staged is still listed — and an
    import resolving to it would then yield an edge to a file that is not there, leaving
    the graph internally inconsistent and hiding the now-broken import. TraceFlow observes
    the working tree, so the file list has to describe the working tree.

    This is the *raw* listing: every language's inputs partition out of it, and the
    configuration a resolver needs — a TypeScript ``tsconfig.json`` among them — is
    still present, which a per-language filter would have removed.
    """
    output = repository.git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    paths = {normalise(item) for item in output.split("\0") if item}
    return tuple(sorted(path for path in paths if (repository.root / path).is_file()))


def list_supported_files(
    repository: Repository, entries: tuple[Analyzer, ...] | None = None
) -> tuple[str, ...]:
    """The raw listing narrowed to files some analyzer claims."""
    return tuple(
        path
        for path in list_repository_files(repository)
        if _analyzer_for(path, entries) is not None
    )


def list_python_files(repository: Repository) -> tuple[str, ...]:
    """Every Python file in the repository, filtered from the supported listing."""
    return tuple(
        path for path in list_supported_files(repository) if path.endswith(PYTHON_SUFFIXES)
    )


def analyze_cached(
    cache: AnalysisCache,
    analyzer: PythonAnalyzer | TypeScriptAnalyzer,
    path: str,
    source: bytes,
    module_name: str | None,
) -> ModuleAnalysis:
    """Analyse *source*, reusing a cached result when the content is unchanged.

    The cache is keyed by content, so the same file renamed keeps its entry, and a
    baseline snapshot is already cached because the blob digest *is* the key.
    """
    digest = digest_of(source)
    cached = cache.get(analyzer.cache_kind, digest)
    if cached is not None:
        try:
            # Path and module name come from the caller, not the cache: identical
            # content at two paths shares one entry, and the stored path would be
            # the wrong one for whichever caller read it second.
            return module_analysis_from_json(cached, path=path, module_name=module_name)
        except (KeyError, TypeError, ValueError):
            # A malformed entry is a cache miss, not a failure. The cache is derived
            # state, so discarding it costs a re-parse and nothing else.
            pass

    analysis = analyzer.analyze(path, source, module_name)
    cache.put(analyzer.cache_kind, analysis.digest, analysis.to_json())
    return analysis


def _module_name_for_path(
    path: str, packages: frozenset[str], registry: tuple[Analyzer, ...]
) -> str | None:
    """The module name a file is imported under, in whichever language it is written."""
    engine = _analyzer_for(path, registry)
    if engine is not None and engine.name == "typescript":
        return module_name_for(path)
    names = module_names_for(path, packages)
    # The shortest name is the one rooted at the shallowest non-package directory,
    # which is how the module is actually imported in a src/ layout.
    return names[-1] if names else None


def build_dependency_graph(
    repository: Repository,
    cache: AnalysisCache,
    config: Config,
    analyzer: PythonAnalyzer | None = None,
    files: PythonFiles | None = None,
    entries: tuple[Analyzer, ...] | None = None,
    supported_paths: tuple[str, ...] | None = None,
) -> DependencyGraph:
    """Parse every supported file in the repository and resolve the imports between them.

    *files* keeps its original meaning — a caller-supplied **Python** listing — and
    produces a Python-only graph, exactly as it did before TypeScript existed.
    *supported_paths* is the multi-language listing a caller made once; with neither,
    the listing is made here. However it arrives, it is partitioned per language and
    each language resolves with its own rules.
    """
    registry = entries or _default_registry()

    if files is not None:
        paths = files.paths
        packages = files.packages
        index = files.index
        ts_files: TypeScriptFiles | None = None
    else:
        if supported_paths is None:
            supported_paths = list_repository_files(repository)
        py_paths = tuple(p for p in supported_paths if p.endswith(PYTHON_SUFFIXES))
        packages = package_directories(py_paths)
        index = build_module_index(py_paths, packages)
        paths = tuple(p for p in supported_paths if _analyzer_for(p, registry) is not None)
        # The tsconfig discovery needs the raw listing — a tsconfig.json is not a
        # file any analyzer claims, and an alias table read from nothing resolves
        # nothing while looking exactly like a repository without aliases.
        ts_files = TypeScriptFiles.of(supported_paths, repository)

    modules: list[ModuleNode] = []
    edges: list[ImportEdge] = []
    unresolved: list[UnresolvedImport] = []
    parse_errors: list[str] = []

    for path in paths:
        engine = (
            _analyzer_for(path, registry)
            if analyzer is None
            else (analyzer if path.endswith(PYTHON_SUFFIXES) else None)
        )
        if engine is None:
            # A caller pinned this graph to one analyzer (the legacy signature);
            # files outside that language take no part in it.
            continue
        absolute = repository.root / Path(path)
        try:
            if absolute.stat().st_size > config.analysis.max_file_size_bytes:
                continue
            source = absolute.read_bytes()
        except OSError:
            continue

        module_name = _module_name_for_path(path, packages, registry)

        analysis = analyze_cached(cache, engine, path, source, module_name)
        if analysis.parse_error:
            parse_errors.append(f"{path}: {analysis.parse_error}")

        modules.append(
            ModuleNode(module_name=module_name or path, path=path, digest=analysis.digest)
        )

        for reference in analysis.imports:
            if engine.name == "typescript" and ts_files is not None:
                target = resolve_specifier(
                    reference.module, path, ts_files.index, ts_files.tsconfig
                )
            else:
                target = resolve_import(
                    reference, module_name, index, is_package=is_package_init(path)
                )
            if target is None:
                unresolved.append(UnresolvedImport(source_path=path, reference=reference))
                continue
            if target == path:
                continue  # a module importing itself adds nothing
            edges.append(
                ImportEdge(
                    source_path=path,
                    target_path=target,
                    module=reference.module,
                    line=reference.line,
                )
            )

    edges.sort(key=lambda edge: (edge.source_path, edge.target_path, edge.line))

    return DependencyGraph(
        modules=tuple(modules),
        edges=tuple(edges),
        unresolved=tuple(unresolved),
        parse_errors=tuple(parse_errors),
    )
