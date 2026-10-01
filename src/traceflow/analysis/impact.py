"""Impact analysis (plan.md §18, §19, §28, §60).

This is the module the rest of the system exists to feed. Everything before it
establishes *what changed*; this module answers the question the product is actually
sold on: **what does that change reach?**

The walk is deliberately a pull traversal, not a whole-repository graph. plan.md §49
requires a focused subgraph, so the engine starts from the changed symbols and expands
outward through the import graph, stopping after ``analysis.impact_max_depth`` rings.
Nothing disconnected from the session is ever examined.

Three decisions carry the weight here.

**A signature change and a body change propagate differently.** This is the whole reason
the analyzer keeps two fingerprints. A symbol whose *declaration* changed forces every
caller to be re-examined: those callers are reported as INDIRECT, which is an obligation.
A symbol whose *body* changed leaves callers untouched — they need no edit, though their
behaviour may differ — so those callers are reported as POTENTIAL, which is not.
Collapsing the two into "affected" would throw away the only distinction static analysis
can make with confidence, and it is what makes signature changes drive the ranking.

**Impact attenuates with distance.** At the first ring the classification reflects the
change itself. Past that, a node is reached through something that was merely affected
rather than changed, so its signature did not change and its callers cannot be *required*
to act. Everything beyond the first ring is therefore INDIRECT propagation: a possibility,
never an obligation. The engine says so rather than inflating the count.

**Uncertainty is reported, not smoothed over.** Call resolution here is textual: it
follows explicit import bindings and falls back to name matching, and it stops where a
name cannot be tied to a file. The constructs that defeat static analysis — ``getattr``,
``eval``, ``importlib``, wildcard imports — are collected into ``limitations`` so the
report states where it is blind instead of quietly under-reporting (plan.md §70).

**A symbol is one thing, and is reported once.** A session can do two things to the same
symbol: change it, *and* reach it because something it calls moved. Both are true, so both
are recorded — but on the one node that represents the symbol, as its own reason plus a
:class:`~traceflow.analysis.models.Reach`. Two rows for one symbol would invite a reader to
count two components and would double-count it in every total. Keeping only the first
finding was the earlier behaviour, and it is how a session with three call sites of a moved
declaration came to report one caller, and how a changed file that also called something
that moved lost the arrow between them.

The report types live in :mod:`traceflow.analysis.models`, so the command line can read an
impact report without importing a traversal.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum

from traceflow.analysis.models import (
    CATEGORY_ORDER,
    CONFIDENCE_ORDER,
    Confidence,
    EdgeChange,
    Evidence,
    GraphDiff,
    ImpactCategory,
    ImpactedNode,
    ImpactReport,
    Reach,
    classify_file,
)
from traceflow.analysis.symbols import (
    SessionAnalysis,
    SessionModule,
    analyse_session_modules,
)
from traceflow.blobs import BlobStore
from traceflow.config import Config
from traceflow.derived import AnalysisCache
from traceflow.git.baseline import Baseline
from traceflow.git.diff import ChangeSet, ChangeStatus, FileChange
from traceflow.git.repository import Repository
from traceflow.languages.base import (
    ImportRef,
    ModuleAnalysis,
    Symbol,
    SymbolChange,
    SymbolChangeKind,
)
from traceflow.languages.cfamily.graph import CFamilyFiles, cfamily_languages_for
from traceflow.languages.python.analyzer import PythonAnalyzer
from traceflow.languages.python.ast_graph import (
    DependencyGraph,
    PythonFiles,
    analyze_cached,
    build_dependency_graph,
    build_module_index,
    is_package_init,
    list_repository_files,
    module_names_for,
    package_directories,
    resolve_import,
)
from traceflow.languages.registry import Analyzer, analyzer_for, default_registry
from traceflow.languages.typescript.graph import (
    TYPESCRIPT_SUFFIXES,
    TypeScriptFiles,
    build_ts_index,
    resolve_specifier,
)
from traceflow.languages.typescript.graph import module_name_for as ts_module_name

#: How many limitations are recorded before the rest are summarised as a count. A report
#: listing three hundred dynamic call sites is not more honest than one listing twenty and
#: saying how many more there were; it is just harder to read.
_MAX_LIMITATIONS = 20


class _Propagation(str, Enum):
    """What a reached node inherits, and what it passes on."""

    SIGNATURE = "signature"
    REMOVED = "removed"
    BODY = "body"
    INDIRECT = "indirect"


#: A symbol's own change seeds the traversal. ADDED is absent on purpose: nothing
#: referenced a symbol that did not exist, so a new symbol reaches nothing — its file is
#: already reported as DIRECT.
_SEED: dict[SymbolChangeKind, _Propagation] = {
    SymbolChangeKind.SIGNATURE_CHANGED: _Propagation.SIGNATURE,
    SymbolChangeKind.REMOVED: _Propagation.REMOVED,
    SymbolChangeKind.BODY_CHANGED: _Propagation.BODY,
}

#: How a reached node is classified, and the reason shown for it.
_CLASSIFICATION: dict[_Propagation, tuple[ImpactCategory, str]] = {
    _Propagation.SIGNATURE: (ImpactCategory.INDIRECT, "signature_changed"),
    _Propagation.REMOVED: (ImpactCategory.INDIRECT, "symbol_removed"),
    _Propagation.BODY: (ImpactCategory.POTENTIAL, "body_changed"),
    _Propagation.INDIRECT: (ImpactCategory.POTENTIAL, "indirect_dependency"),
}

#: Calls whose target is decided at runtime. Listing them is how the report says "the graph
#: is incomplete here" instead of presenting a smaller graph as a complete one.
_DYNAMIC_DISPATCH = frozenset(
    {
        "__import__",
        "delattr",
        "eval",
        "exec",
        "getattr",
        "globals",
        "locals",
        "setattr",
        "vars",
    }
)


# --------------------------------------------------------------------------- call resolution


def bound_name(reference: ImportRef, path: str | None = None) -> str | None:
    """The local name an import statement binds in the importing module.

    ==================================  ==================
    Source                              Binds
    ==================================  ==================
    ``import a.b``                      ``a``
    ``import a.b as c``                 ``c``
    ``from a import b``                 ``b``
    ``from a import b as c``            ``c``
    ``from . import b``                 ``b``
    ``from a import *``                 nothing — reported as a limitation
    ``import "p"`` (Go)                 nothing — the path is not a name
    ``import a.b.C`` (Java)             ``C`` — the class, which calls name directly
    ==================================  ==================

    Go's imports bind nothing: the identifier used at a call site is the *package
    name*, the last segment of the imported path, and the import does not name it
    as a local — Go call resolution falls to package-name matching below. Java is
    the opposite extreme: ``import com.example.db.Store`` exists so the file can
    write ``Store.query(...)``, so the binding is the *last* segment, not Python's
    first.
    """
    if reference.name is None:
        if not reference.module:
            return None
        if path is not None and path.endswith(".go"):
            return None
        if path is not None and path.endswith(".java"):
            return reference.alias or reference.module.rsplit(".", 1)[-1]
        return reference.alias or reference.module.split(".")[0]
    if reference.name == "*":
        return None
    return reference.alias or reference.name


def bound_imports(analysis: ModuleAnalysis, path: str | None = None) -> dict[str, ImportRef]:
    """Map each locally bound name to the import that binds it.

    Last binding wins, because that is what Python does: importing the same name twice
    leaves the second import in effect, so resolving to the first would point at a
    binding the module no longer has. *path*, when given, lets language shapes into
    the rule — a Go import binds nothing (see :func:`bound_name`).
    """
    bindings: dict[str, ImportRef] = {}
    for reference in analysis.imports:
        name = bound_name(reference, path)
        if name is not None:
            bindings[name] = reference
    return bindings


@dataclass(frozen=True)
class CallResolution:
    """Where a call expression appears to point."""

    path: str
    names: tuple[str, ...]
    """Qualified names in *path* the callee could be."""

    confidence: Confidence
    detail: str


def _candidate_names(remaining: tuple[str, ...], symbols: tuple[Symbol, ...]) -> tuple[str, ...]:
    """Qualified names a dotted call tail could denote in the target module.

    ``service.authenticate`` reaches a module-level ``authenticate``. ``User.create``
    reaches ``User.create``. Both spellings have to be offered, because which one is right
    depends on whether the name bound by the import is a module or a class — which is
    precisely the question the resolution is trying to answer.

    The bare tail is offered only when no symbol in the target matched it. Adding it
    unconditionally would let a call to ``User.create`` match a change to a module-level
    ``create`` in the same file, reporting a caller that is not one. When a symbol *does*
    match, its qualified name is already among the candidates, so nothing is lost.
    """
    names = {symbol.qualified_name for symbol in symbols if symbol.name in remaining}
    if not names:
        # No symbol matched, which is exactly the case a session may have *removed* one:
        # the caller still names it, and that call site is the finding.
        names.add(".".join(remaining))
    return tuple(sorted(names))


def resolve_call(
    call_name: str,
    analysis: ModuleAnalysis,
    module_name: str | None,
    index: dict[str, str],
    path: str,
    symbols_for: Callable[[str], tuple[Symbol, ...]],
    ts_files: TypeScriptFiles | None = None,
    registry: tuple[Analyzer, ...] | None = None,
    cfamily_files: CFamilyFiles | None = None,
) -> CallResolution | None:
    """Work out which file a call expression reaches, as far as the text allows.

    Returns ``None`` when the callee is a builtin, a local variable, or anything else with
    no static name — the honest answer, and the reason ``limitations`` exists rather than a
    guess being emitted here.
    """
    # Rust writes its call chains with ``::`` (``service::login(...)``); every other
    # name these resolvers match — imports, symbols — is dot-separated already, so
    # the chain is folded once here rather than at each comparison below.
    if cfamily_files is not None and cfamily_languages_for(path) is not None:
        call_name = call_name.replace("::", ".")
    parts = call_name.split(".")
    root = parts[0]
    reference = bound_imports(analysis, path).get(root)

    if reference is not None:
        engine = analyzer_for(path, registry) if registry else None
        if ts_files is not None and engine is not None and engine.name == "typescript":
            target = resolve_specifier(reference.module, path, ts_files.index, ts_files.tsconfig)
        elif cfamily_files is not None and cfamily_languages_for(path) is not None:
            target = cfamily_files.resolve(reference.module, path)
            if target is not None and not parts[1:]:
                # ``Service(...)`` — the imported name itself is the callee, and its
                # name in the target is the *imported* name rather than the local
                # alias: a call to a symbol this session removed must still be found,
                # and the candidate is only ever matched against changed symbols.
                imported = reference.module.rsplit(".", 1)[-1]
                names = tuple(
                    symbol.qualified_name
                    for symbol in symbols_for(target)
                    if symbol.name == imported
                )
                return CallResolution(
                    path=target,
                    names=names or (imported,),
                    confidence=Confidence.CONFIRMED,
                    detail=f"'{root}' is bound to {imported} by an import statement",
                )
        else:
            target = resolve_import(reference, module_name, index, is_package=is_package_init(path))
        if target is not None:
            remaining = tuple(parts[1:])
            if not remaining:
                # `Service(...)` — the imported name itself is the callee, and its name in
                # the target is the imported name rather than the local alias.
                imported = reference.name or reference.module.rsplit(".", 1)[-1]
                names = tuple(
                    symbol.qualified_name
                    for symbol in symbols_for(target)
                    if symbol.name == imported
                )
                # The bare name is offered even when the target no longer defines it. A
                # call to a symbol this session *removed* is precisely the case that must
                # be found, and the candidate is only ever matched against the set of
                # changed symbols, so offering it cannot invent an impact.
                return CallResolution(
                    path=target,
                    names=names or (imported,),
                    confidence=Confidence.CONFIRMED,
                    detail=f"'{root}' is bound to {imported} by an import statement",
                )

            symbols = symbols_for(target)
            known = any(symbol.name in remaining for symbol in symbols)
            return CallResolution(
                path=target,
                names=_candidate_names(remaining, symbols),
                confidence=Confidence.CONFIRMED if known else Confidence.INFERRED,
                detail=f"'{root}' is imported from {reference.module or '.'}",
            )

    # `self.helper()` reaches a method of the enclosing class. An instance attribute of the
    # same name would shadow it, so this is high confidence rather than confirmed.
    if len(parts) == 2 and root in {"self", "cls"}:
        methods = tuple(
            symbol.qualified_name for symbol in analysis.symbols if symbol.name == parts[1]
        )
        if methods:
            return CallResolution(
                path=path,
                names=methods,
                confidence=Confidence.HIGH_CONFIDENCE,
                detail=f"'{parts[1]}' is defined in this module and reached through '{root}'",
            )
        return None

    # A bare name this module defines at module level.
    if len(parts) == 1:
        local = tuple(
            symbol.qualified_name
            for symbol in analysis.symbols
            if symbol.name == root and symbol.parent is None
        )
        if local:
            return CallResolution(
                path=path,
                names=local,
                confidence=Confidence.HIGH_CONFIDENCE,
                detail=f"'{root}' is defined in this module",
            )

    # The cfamily fallbacks, in the order the languages write their calls.
    if cfamily_files is not None and cfamily_languages_for(path) is not None:
        # A call whose root names a symbol of an imported file: Java's and C#'s
        # static calls — ``Repo.Validate(...)`` where an import (or a shared
        # namespace) makes ``Repo`` reachable. Each import is resolved to its
        # file and the root matched against that file's symbols; the tail rides
        # along as candidate names, and the answer is inferred, because the same
        # name may exist elsewhere.
        for candidate_import in analysis.imports:
            target = cfamily_files.resolve(candidate_import.module, path)
            if target is None:
                continue
            if any(symbol.name == root for symbol in symbols_for(target)):
                return CallResolution(
                    path=target,
                    names=_candidate_names(tuple(parts[1:]), symbols_for(target))
                    if parts[1:]
                    else (root,),
                    confidence=Confidence.INFERRED,
                    detail=f"'{root}' is defined in {target}, imported by this file",
                )

        # Go across packages: imports bind nothing, so the call's root is matched
        # against each import's *package name* — the last path segment, which is
        # the identifier Go programs actually write — and the import resolves to
        # its package's file. The tail is offered to the target's symbols exactly
        # as an imported dotted call is.
        if len(parts) >= 2:
            for candidate_import in analysis.imports:
                package = candidate_import.module.rstrip("/").rsplit("/", 1)[-1]
                if package != root:
                    continue
                target = cfamily_files.resolve(candidate_import.module, path)
                if target is None:
                    continue
                symbols = symbols_for(target)
                known = any(symbol.name in parts[1:] for symbol in symbols)
                return CallResolution(
                    path=target,
                    names=_candidate_names(tuple(parts[1:]), symbols),
                    confidence=Confidence.CONFIRMED if known else Confidence.INFERRED,
                    detail=f"'{root}' is the package imported from {candidate_import.module}",
                )

        # The same-package (Go) and same-namespace (C#) call: no import binds the
        # name because none is needed — the callee lives in a sibling file. The
        # root is matched against each sibling's symbols and the first file that
        # defines it is the answer. Inferred, because two siblings defining the
        # same name is legal; a changed symbol of that name in one of them is
        # still the finding, and the candidate match keeps it from inventing one.
        for candidate in cfamily_files.siblings_of(path):
            symbols = symbols_for(candidate)
            if any(symbol.name == root for symbol in symbols):
                return CallResolution(
                    path=candidate,
                    names=(root,),
                    confidence=Confidence.INFERRED,
                    detail=f"'{root}' is defined in the sibling file {candidate}",
                )

    return None


def enclosing_symbol(analysis: ModuleAnalysis, line: int) -> str | None:
    """The innermost symbol whose body contains *line*.

    Innermost by span, so a call inside a method is attributed to the method rather than to
    the class around it — the same rule the analyzer applies when it excludes nested
    definitions from a parent's body fingerprint.
    """
    best: Symbol | None = None
    best_span: int | None = None
    for symbol in analysis.symbols:
        if not symbol.line_start <= line <= symbol.line_end:
            continue
        span = symbol.line_end - symbol.line_start
        if best_span is None or span < best_span:
            best = symbol
            best_span = span
    return best.qualified_name if best is not None else None


# --------------------------------------------------------------------------- traversal state


@dataclass(frozen=True)
class _Frontier:
    """One file whose change is being propagated outward."""

    path: str
    names: tuple[str, ...]
    """Qualified names that changed in *path*. Empty means the whole module."""

    propagation: _Propagation
    distance: int
    chain: tuple[str, ...]
    evidence: tuple[Evidence, ...]


class _ModuleLoader:
    """Reads and analyses repository files on demand, once each.

    A file with fan-in is reached by several traversal paths, and reading it is the only
    genuinely expensive step in the walk. Memoising makes that one read per file, while the
    content-addressed analysis cache means the parse is usually one the session's symbol
    analysis already paid for.
    """

    def __init__(
        self,
        repository: Repository,
        cache: AnalysisCache,
        config: Config,
        registry: tuple[Analyzer, ...] | None = None,
    ) -> None:
        self._repository = repository
        self._cache = cache
        self._config = config
        self._registry = registry if registry is not None else default_registry()
        self._loaded: dict[str, ModuleAnalysis | None] = {}

    def __call__(self, path: str) -> ModuleAnalysis | None:
        if path not in self._loaded:
            self._loaded[path] = self._read(path)
        return self._loaded[path]

    def _read(self, path: str) -> ModuleAnalysis | None:
        engine = analyzer_for(path, self._registry)
        if engine is None:
            return None
        absolute = self._repository.root / path
        try:
            if absolute.stat().st_size > self._config.analysis.max_file_size_bytes:
                return None
            source = absolute.read_bytes()
        except OSError:
            return None
        # The walk matches calls against import bindings keyed by module name, so
        # the loader reports the file under the name its language resolves by.
        return analyze_cached(self._cache, engine, path, source, CFamilyFiles.module_name_for(path))


def _module_name_for(
    path: str,
    packages: frozenset[str],
    registry: tuple[Analyzer, ...] | None = None,
) -> str | None:
    """The name a file is imported under, in whichever language it is written.

    *registry* of ``None`` asks the Python question only, which is what every caller
    before TypeScript existed meant.
    """
    if registry is not None:
        engine = analyzer_for(path, registry)
        if engine is not None and engine.name == "typescript":
            return ts_module_name(path)
        if cfamily_languages_for(path) is not None:
            return CFamilyFiles.module_name_for(path)
    names = module_names_for(path, packages)
    return names[-1] if names else None


# --------------------------------------------------------------------------- direct impact


def _symbol_change_evidence(module: SessionModule, change: SymbolChange) -> tuple[Evidence, ...]:
    """Evidence for a changed symbol, including what its declaration changed from and to.

    A hash alone is not evidence (plan.md §33). Showing ``(user, password)`` becoming
    ``(user, password, mfa=False)`` is what turns "the signature changed" into something a
    reader can check.
    """
    detail = change.change.value
    if change.change is SymbolChangeKind.SIGNATURE_CHANGED and module.before is not None:
        before = module.before.symbol(change.qualified_name)
        after = module.after.symbol(change.qualified_name) if module.after is not None else None
        if before is not None and after is not None:
            detail = f"signature changed: {before.signature} -> {after.signature}"
        if after is not None and after.occurrences > 1:
            # The name is defined more than once, so the signature shown is every arm's, and
            # which arm moved cannot be told from here. Saying so is the difference between
            # evidence and a claim the reader cannot check.
            detail += f" (defined {after.occurrences} times in this file)"

    return (
        Evidence(kind="symbol_change", path=module.path, line=change.line_start, detail=detail),
    )


def _file_node(change: FileChange, category: ImpactCategory, reason: str) -> ImpactedNode:
    return ImpactedNode(
        path=change.path,
        symbol=None,
        category=category,
        confidence=Confidence.CONFIRMED,
        distance=0,
        reason=reason,
        chain=(change.path,),
        evidence=(
            Evidence(
                kind="file_change",
                path=change.path,
                line=None,
                detail=f"{change.status.value} during this session",
            ),
        ),
    )


def _direct_nodes(
    change_set: ChangeSet, session: SessionAnalysis
) -> tuple[list[ImpactedNode], list[_Frontier]]:
    """The session's own changes, and the seeds for the traversal.

    Only symbol changes seed the walk. A file whose imports changed affects what it depends
    on, not what depends on it, so it is reported but not propagated.
    """
    nodes: list[ImpactedNode] = []
    seeds: list[_Frontier] = []

    for change in change_set.files:
        category = classify_file(change.path) or ImpactCategory.DIRECT
        module = session.module(change.path)

        if module is None:
            # The analyzer does not handle this file, so the file itself is the finding.
            nodes.append(_file_node(change, category, f"file_{change.status.value}"))
            continue

        imports_changed = bool(module.changes.imports_added or module.changes.imports_removed)

        if not module.changes.changes:
            reason = "imports_changed" if imports_changed else "content_changed"
            nodes.append(_file_node(change, category, reason))
            continue

        for symbol_change in module.changes.changes:
            evidence = _symbol_change_evidence(module, symbol_change)
            nodes.append(
                ImpactedNode(
                    path=change.path,
                    symbol=symbol_change.qualified_name,
                    category=category,
                    confidence=Confidence.CONFIRMED,
                    distance=0,
                    reason=(
                        "file_deleted"
                        if change.status is ChangeStatus.DELETED
                        else f"symbol_{symbol_change.change.value}"
                    ),
                    chain=(change.path,),
                    evidence=evidence,
                )
            )

            propagation = _SEED.get(symbol_change.change)
            if propagation is not None:
                seeds.append(
                    _Frontier(
                        path=change.path,
                        names=(symbol_change.qualified_name,),
                        propagation=propagation,
                        distance=0,
                        chain=(change.path,),
                        evidence=evidence,
                    )
                )

        if imports_changed:
            # The file's own category, not DEPENDENCY: DEPENDENCY means "this file
            # declares external dependencies", and conflating the two would put an
            # ordinary source file in the dependency section for editing an import.
            nodes.append(
                ImpactedNode(
                    path=change.path,
                    symbol=None,
                    category=category,
                    confidence=Confidence.CONFIRMED,
                    distance=0,
                    reason="imports_changed",
                    chain=(change.path,),
                    evidence=(
                        Evidence(
                            kind="import_statement",
                            path=change.path,
                            line=None,
                            detail=(
                                f"+{len(module.changes.imports_added)} "
                                f"-{len(module.changes.imports_removed)} import(s)"
                            ),
                        ),
                    ),
                )
            )

    return nodes, seeds


# --------------------------------------------------------------------------- graph helpers


def _importers_map(graph: DependencyGraph) -> dict[str, list[str]]:
    """Map each file to the files that import it — the direction impact travels."""
    importers: dict[str, list[str]] = {}
    for edge in graph.edges:
        importers.setdefault(edge.target_path, []).append(edge.source_path)
    return {target: sorted(set(sources)) for target, sources in importers.items()}


def _departed_paths(session: SessionAnalysis) -> tuple[str, ...]:
    """Paths that existed before the session and do not exist now.

    A rename belongs here as much as a deletion: the old path is gone either way, and
    anything still importing it is broken.
    """
    departed: set[str] = set()
    for module in session.modules:
        if module.is_deleted:
            departed.add(module.path)
        elif module.status is ChangeStatus.RENAMED and module.original_path:
            departed.add(module.original_path)
    return tuple(sorted(departed))


def _dangling_imports(
    graph: DependencyGraph,
    session: SessionAnalysis,
    packages: frozenset[str],
    current_paths: tuple[str, ...],
    ts_files: TypeScriptFiles | None = None,
    registry: tuple[Analyzer, ...] | None = None,
    cfamily_files: CFamilyFiles | None = None,
    repository: Repository | None = None,
) -> list[tuple[str, str, int, str]]:
    """Files left importing a module this session removed.

    An import that no longer resolves is not merely unresolved: it *used* to resolve, and
    the session is why it does not. Re-resolving each unresolved import against the
    pre-session module index is what separates those two cases — and it is only possible
    because the unresolved record keeps the whole import reference rather than just the
    dotted module name.

    The distinction matters beyond the impact list: an import explained by a removal is not
    an external dependency, and counting it as one would report a module the session
    deleted as a third-party package the repository depends on.
    """
    departed = set(_departed_paths(session))
    if not departed:
        return []

    pre_paths = tuple({*current_paths, *departed})
    pre_index = build_module_index(pre_paths, package_directories(pre_paths))

    # The same reconstruction, in TypeScript terms: an index of every script file
    # that existed *before* the session, so an import of a deleted module resolves
    # against the index in which it was still there. The tsconfig aliases did not
    # depend on the session's file changes, so the current ones are reused.
    ts_pre_index: dict[str, str] = {}
    if ts_files is not None:
        ts_pre_index = build_ts_index(
            tuple(p for p in pre_paths if p.endswith(TYPESCRIPT_SUFFIXES))
        )

    found: list[tuple[str, str, int, str]] = []
    for item in graph.unresolved:
        engine = analyzer_for(item.source_path, registry) if registry else None
        if ts_files is not None and engine is not None and engine.name == "typescript":
            target = resolve_specifier(
                item.reference.module, item.source_path, ts_pre_index, ts_files.tsconfig
            )
        elif (
            cfamily_files is not None
            and repository is not None
            and cfamily_languages_for(item.source_path) is not None
        ):
            # The same reconstruction in cfamily terms: every pre-session file is
            # re-indexed so a deleted package's or module's import resolves one
            # last time — the index in which the target was still there.
            pre_cfamily = CFamilyFiles.of(pre_paths, repository, include_missing=True)
            target = pre_cfamily.resolve(item.reference.module, item.source_path)
        else:
            importing = _module_name_for(item.source_path, packages, registry)
            target = resolve_import(
                item.reference, importing, pre_index, is_package=is_package_init(item.source_path)
            )
        if target is not None and target in departed:
            found.append((item.source_path, target, item.line, item.module))
    return sorted(set(found))


def _edges_of(
    analysis: ModuleAnalysis | None,
    module_name: str | None,
    index: dict[str, str],
    path: str,
    ts_files: TypeScriptFiles | None = None,
    registry: tuple[Analyzer, ...] | None = None,
    cfamily_files: CFamilyFiles | None = None,
) -> dict[str, tuple[str, int]]:
    """Outgoing dependencies keyed by target file, because that is what a relationship is.

    Keyed by target rather than by import statement so that re-ordering imports, or
    importing the same module twice, does not read as a structural change.

    *path* is the file these edges leave, which is what says whether a relative import
    starts at its own package — and which language's resolver answers the question.
    """
    if analysis is None:
        return {}
    engine = analyzer_for(path, registry) if registry else None
    use_ts = ts_files is not None and engine is not None and engine.name == "typescript"
    use_cfamily = cfamily_files is not None and cfamily_languages_for(path) is not None
    edges: dict[str, tuple[str, int]] = {}
    for reference in analysis.imports:
        if use_ts and ts_files is not None:
            target = resolve_specifier(reference.module, path, ts_files.index, ts_files.tsconfig)
        elif use_cfamily and cfamily_files is not None:
            target = cfamily_files.resolve(reference.module, path)
        else:
            target = resolve_import(reference, module_name, index, is_package=is_package_init(path))
        if target is not None:
            edges.setdefault(target, (reference.module, reference.line))
    return edges


def _edge_order(edge: EdgeChange) -> tuple[str, str, int]:
    return (edge.source_path, edge.target_path, edge.line)


def _graph_diff(
    change_set: ChangeSet,
    session: SessionAnalysis,
    index: dict[str, str],
    packages: frozenset[str],
    ts_files: TypeScriptFiles | None = None,
    registry: tuple[Analyzer, ...] | None = None,
    cfamily_files: CFamilyFiles | None = None,
) -> GraphDiff:
    """Compare the changed files' dependencies before and after (plan.md §28).

    Only relationships declared by changed files are compared. An import cannot appear or
    vanish without the file declaring it changing, so this is the complete diff for the
    session without a second parse of the whole repository at the baseline.
    """
    modules_added: list[str] = []
    modules_removed: list[str] = []

    for change in change_set.files:
        module = session.module(change.path)
        if module is None:
            continue
        if module.is_deleted:
            modules_removed.append(change.path)
        elif module.is_new:
            modules_added.append(change.path)

    edges_added: set[EdgeChange] = set()
    edges_removed: set[EdgeChange] = set()

    for module in session.modules:
        before_name = (
            _module_name_for(module.original_path, packages, registry)
            if module.original_path
            else module.module_name
        )
        before = _edges_of(
            module.before,
            before_name,
            index,
            module.original_path or module.path,
            ts_files,
            registry,
            cfamily_files,
        )
        after = _edges_of(
            module.after, module.module_name, index, module.path, ts_files, registry, cfamily_files
        )

        for target, (text, line) in after.items():
            if target not in before:
                edges_added.add(EdgeChange(module.path, target, text, line))
        for target, (text, line) in before.items():
            if target not in after:
                edges_removed.add(EdgeChange(module.path, target, text, line))

    return GraphDiff(
        modules_added=tuple(sorted(set(modules_added))),
        modules_removed=tuple(sorted(set(modules_removed))),
        edges_added=tuple(sorted(edges_added, key=_edge_order)),
        edges_removed=tuple(sorted(edges_removed, key=_edge_order)),
    )


# --------------------------------------------------------------------------- limitations


def _dynamic_limitations(analysis: ModuleAnalysis) -> list[str]:
    """Where this file defeats static analysis (plan.md §70)."""
    found: list[str] = []

    for reference in analysis.imports:
        if reference.name == "*":
            found.append(
                f"{analysis.path}:{reference.line} uses a wildcard import — "
                f"the names it introduces cannot be resolved"
            )

    for call in analysis.calls:
        leaf = call.name.rsplit(".", 1)[-1]
        if leaf in _DYNAMIC_DISPATCH:
            found.append(
                f"{analysis.path}:{call.line} calls {leaf} — "
                f"the call target cannot be determined statically"
            )
        elif leaf == "import_module":
            found.append(
                f"{analysis.path}:{call.line} performs a dynamic import — "
                f"the imported module cannot be determined statically"
            )

    return found


def _limitations(
    external_imports: int,
    session: SessionAnalysis,
    impacted_paths: list[str],
    truncated: bool,
    max_depth: int,
) -> tuple[str, ...]:
    """Everything that made the answer less complete than it looks."""
    found: list[str] = []

    if truncated:
        found.append(
            f"traversal stopped at the configured depth of {max_depth} — "
            f"deeper dependents are not listed"
        )

    if external_imports:
        found.append(
            f"{external_imports} import(s) point outside the repository; "
            f"dependencies through them are not represented"
        )

    for error in session.parse_errors:
        found.append(f"{error} — the file's symbols are unknown")

    for path in sorted(set(impacted_paths)):
        module = session.module(path)
        if module is None:
            continue
        # A deleted module's current analysis is empty by construction, so its own
        # limitations can only be read from the version that existed.
        analysis = module.before if module.is_deleted else module.after
        if analysis is not None:
            found.extend(_dynamic_limitations(analysis))

    for entry in session.skipped:
        found.append(f"{entry} — not analysed")

    unique = tuple(dict.fromkeys(found))
    if len(unique) <= _MAX_LIMITATIONS:
        return unique
    return (*unique[:_MAX_LIMITATIONS], f"and {len(unique) - _MAX_LIMITATIONS} more limitation(s)")


# --------------------------------------------------------------------------- entry point


def graph_is_stale(session: SessionAnalysis) -> bool:
    """True when this session changed the dependency graph's shape.

    The graph records which files exist and which import which. A session that altered
    neither leaves it valid, and rebuilding it would re-parse every Python file in the
    repository for an answer that cannot have changed. Everything else — a file added,
    removed or renamed, or an import statement added or dropped — invalidates it.

    Computed from the session analysis rather than from the graph diff, because it has to
    be known *before* the graph is used: a graph rebuilt after a stale walk would have
    answered the wrong question.
    """
    for module in session.modules:
        if module.is_new or module.is_deleted or module.original_path is not None:
            return True
        if module.changes.imports_added or module.changes.imports_removed:
            return True
    return False


def _merge_evidence(
    existing: tuple[Evidence, ...], extra: tuple[Evidence, ...]
) -> tuple[Evidence, ...]:
    """Union of two evidence sets, in a stable order.

    Deduplicated on the whole record rather than on the location: the same line can carry
    two different statements, and collapsing them would lose one of them.
    """
    merged = list(existing)
    seen = {(item.kind, item.path, item.line, item.detail) for item in existing}
    for item in extra:
        key = (item.kind, item.path, item.line, item.detail)
        if key not in seen:
            seen.add(key)
            merged.append(item)
    return tuple(merged)


def _with_reach(node: ImpactedNode, reach: Reach) -> ImpactedNode:
    """Record one more way a symbol was reached, on the node that already represents it.

    A symbol is one thing, so it is reported once. The alternative — a second row for the
    same symbol — would invite a reader to count two components, and would double-count it
    in every total that asks how many tests or how many symbols a change touched.

    Deduplicated on the reason: a symbol reached by the same reason along five different
    paths is one finding, not five. The first path is kept (the walk is breadth-first, so
    it is the nearest) and a later arrival for the same reason contributes its evidence,
    which is what lets one row say "three call sites".
    """
    for position, existing in enumerate(node.reaches):
        if existing.reason != reach.reason:
            continue
        merged = Reach(
            reason=existing.reason,
            category=existing.category,
            distance=existing.distance,
            chain=existing.chain,
            confidence=existing.confidence,
            evidence=_merge_evidence(existing.evidence, reach.evidence),
        )
        return replace(
            node, reaches=(*node.reaches[:position], merged, *node.reaches[position + 1 :])
        )
    return replace(node, reaches=(*node.reaches, reach))


def build_impact_report(
    repository: Repository,
    baseline: Baseline,
    change_set: ChangeSet,
    blobs: BlobStore,
    cache: AnalysisCache,
    config: Config,
    graph: DependencyGraph | None = None,
    session: SessionAnalysis | None = None,
    analyzer: PythonAnalyzer | None = None,
    files: PythonFiles | None = None,
    supported_paths: tuple[str, ...] | None = None,
    entries: tuple[Analyzer, ...] | None = None,
) -> ImpactReport:
    """Walk from the session's changed symbols to everything they reach.

    *graph* and *session* may be supplied by a caller that has already produced them. That
    matters because a watcher closes many sessions in a row: the graph parses every
    supported source file in the repository and the session analysis reads every changed
    one, so deriving both here is right for a one-shot command and wrong inside a watch
    loop. A caller that supplies them is responsible for rebuilding when
    :func:`graph_is_stale` says to.

    *files* keeps its original meaning — a caller-supplied **Python** listing — and pins
    the report to Python, exactly as before TypeScript existed. *supported_paths* is the
    newer form of the same favour: the multi-language listing a caller made once and
    passes to every pass over one tree. Passing both is not allowed; the Python pin wins
    with an error, because silently ignoring one of them would mislead the caller about
    what was analysed.
    """
    if files is not None and supported_paths is not None:
        raise ValueError("pass either files= or supported_paths=, not both")

    registry = entries or default_registry()
    ts_files: TypeScriptFiles | None = None
    cfamily_files: CFamilyFiles | None = None
    listing: PythonFiles

    if files is not None:
        listing = files
    else:
        if supported_paths is None:
            supported_paths = list_repository_files(repository)
        py_paths = tuple(p for p in supported_paths if p.endswith((".py", ".pyi")))
        py_packages = package_directories(py_paths)
        listing = PythonFiles(
            paths=py_paths,
            packages=py_packages,
            index=build_module_index(py_paths, py_packages),
        )
        # Built from the *raw* listing: tsconfig.json is not a file any analyzer
        # claims, and alias resolution without it resolves nothing while looking
        # exactly like a repository that never declared an alias. The cfamily
        # indexes read go.mod and package/namespace heads the same way.
        ts_files = TypeScriptFiles.of(supported_paths, repository)
        cfamily_files = CFamilyFiles.of(supported_paths, repository)

    session_analysis = session or analyse_session_modules(
        repository,
        baseline,
        change_set,
        blobs,
        cache,
        config,
        analyzer,
        files=files,
        entries=registry,
        supported_paths=supported_paths if files is None else None,
    )
    dependency_graph = graph or build_dependency_graph(
        repository,
        cache,
        config,
        analyzer,
        files=listing if files is not None else None,
        entries=registry,
        supported_paths=supported_paths if files is None else None,
    )

    current_paths = listing.paths
    packages = listing.packages
    index = listing.index

    analyses: dict[str, ModuleAnalysis] = {
        module.path: module.after for module in session_analysis.modules if module.after is not None
    }
    loader = _ModuleLoader(repository, cache, config, registry)

    def analysis_of(path: str) -> ModuleAnalysis | None:
        """A file's current analysis, reusing the session's when it already has one."""
        analysis = analyses.get(path)
        if analysis is None:
            analysis = loader(path)
        return analysis

    def symbols_for(path: str) -> tuple[Symbol, ...]:
        analysis = analysis_of(path)
        return analysis.symbols if analysis is not None else ()

    nodes, frontier = _direct_nodes(change_set, session_analysis)
    # Where each symbol is already recorded, so that a second finding about it can be added
    # to the node that represents it instead of being dropped or duplicated.
    position_of: dict[tuple[str, str | None], int] = {
        (node.path, node.symbol): position for position, node in enumerate(nodes)
    }
    importers = _importers_map(dependency_graph)

    max_depth = config.analysis.impact_max_depth
    depth = 0
    truncated = False

    while frontier:
        if depth >= max_depth:
            # Stopping here is a decision, and the report says so rather than presenting a
            # truncated walk as a complete one.
            truncated = True
            break
        depth += 1
        following: list[_Frontier] = []

        for entry in frontier:
            # The module that changed is always its own candidate source. A symbol can be
            # affected by a change in the same file — the most immediate impact there is —
            # and the import graph cannot express that, because a module importing itself
            # is not an edge.
            for source in (entry.path, *importers.get(entry.path, ())):
                analysis = analysis_of(source)
                if analysis is None:
                    continue
                module_name = _module_name_for(source, packages)

                for call in analysis.calls:
                    resolution = resolve_call(
                        call.name,
                        analysis,
                        module_name,
                        index,
                        source,
                        symbols_for,
                        ts_files,
                        registry,
                        cfamily_files,
                    )
                    if resolution is None or resolution.path != entry.path:
                        continue

                    # An empty name set means the whole source module changed, so any call
                    # into it counts as affected.
                    matched = (
                        tuple(name for name in resolution.names if name in entry.names)
                        if entry.names
                        else resolution.names
                    )
                    if not matched:
                        continue

                    caller = enclosing_symbol(analysis, call.line)
                    key = (source, caller)
                    category, reason = _CLASSIFICATION[entry.propagation]
                    category = classify_file(source) or category
                    evidence = (
                        *entry.evidence,
                        Evidence(
                            kind="call_expression",
                            path=source,
                            line=call.line,
                            detail=f"{call.name} reaches {entry.path}#{matched[0]}",
                        ),
                    )
                    distance = entry.distance + 1
                    chain = (*entry.chain, source)

                    recorded = position_of.get(key)
                    if recorded is not None:
                        # Already recorded — the session changed this symbol, or an earlier
                        # path reached it. The finding is still true, so it is attached to
                        # the node that represents the symbol. Dropping it here is how a
                        # session with three call sites of a moved declaration reported one
                        # caller, and how a changed file that also calls something that
                        # moved lost the arrow between them.
                        #
                        # No frontier entry: the symbol already has one — from its own change
                        # if it was direct, from the path that first reached it otherwise —
                        # so the walk continues outward from it either way. Merging here adds
                        # a finding, not a route.
                        nodes[recorded] = _with_reach(
                            nodes[recorded],
                            Reach(
                                reason=reason,
                                category=category,
                                distance=distance,
                                chain=chain,
                                confidence=resolution.confidence,
                                evidence=evidence,
                            ),
                        )
                        continue

                    position_of[key] = len(nodes)
                    nodes.append(
                        ImpactedNode(
                            path=source,
                            symbol=caller,
                            category=category,
                            confidence=resolution.confidence,
                            distance=distance,
                            reason=reason,
                            chain=chain,
                            evidence=evidence,
                        )
                    )
                    following.append(
                        _Frontier(
                            path=source,
                            names=(caller,) if caller is not None else (),
                            propagation=_Propagation.INDIRECT,
                            distance=distance,
                            chain=chain,
                            evidence=evidence,
                        )
                    )

        frontier = following

    dangling = _dangling_imports(
        dependency_graph,
        session_analysis,
        packages,
        current_paths,
        ts_files,
        registry,
        cfamily_files,
        repository,
    )
    for source, target, line, module_text in dangling:
        evidence = (
            Evidence(
                kind="import_statement",
                path=source,
                line=line,
                detail=f"imports {module_text}, which this session removed",
            ),
        )
        recorded = position_of.get((source, None))
        if recorded is not None:
            # Same rule as a call: the file may also have been changed, or already reached
            # another way, and an import left broken by the session is still broken.
            nodes[recorded] = _with_reach(
                nodes[recorded],
                Reach(
                    reason="dangling_import",
                    category=classify_file(source) or ImpactCategory.INDIRECT,
                    distance=1,
                    chain=(target, source),
                    confidence=Confidence.CONFIRMED,
                    evidence=evidence,
                ),
            )
            continue

        position_of[(source, None)] = len(nodes)
        nodes.append(
            ImpactedNode(
                path=source,
                symbol=None,
                category=classify_file(source) or ImpactCategory.INDIRECT,
                confidence=Confidence.CONFIRMED,
                distance=1,
                reason="dangling_import",
                chain=(target, source),
                evidence=evidence,
            )
        )

    # Only imports that were never resolvable count as external. An import explained by a
    # removal is already reported as a dangling import, and listing it here too would name
    # a module this session deleted as a third-party package the repository depends on.
    explained = {(source, line, module) for source, _target, line, module in dangling}
    external = [
        item
        for item in dependency_graph.unresolved
        if (item.source_path, item.line, item.module) not in explained
    ]

    nodes.sort(
        key=lambda node: (
            CATEGORY_ORDER[node.category],
            node.distance,
            CONFIDENCE_ORDER[node.confidence],
            node.path,
            node.symbol or "",
        )
    )

    # Read off the nodes rather than reusing the loop counter. The counter advances once
    # per ring *processed*, and the final ring is always processed to discover that it
    # leads nowhere — so when the walk ends naturally it would report one ring more than
    # it reached. The deepest distance among the nodes is the honest answer in both the
    # natural and the truncated case.
    #
    # A reach counts: a symbol the session changed sits at distance 0, but it can also be
    # reached several rings out, and reporting depth 0 for that walk would understate it.
    deepest = max(
        [node.distance for node in nodes]
        + [item.distance for node in nodes for item in node.reaches],
        default=0,
    )

    return ImpactReport(
        analyzer=session_analysis.analyzer,
        changed_files=tuple(change.path for change in change_set.files),
        nodes=tuple(nodes),
        graph_diff=_graph_diff(
            change_set, session_analysis, index, packages, ts_files, registry, cfamily_files
        ),
        limitations=_limitations(
            len(external),
            session_analysis,
            [node.path for node in nodes],
            truncated,
            max_depth,
        ),
        unresolved_imports=len(external),
        external_modules=tuple(sorted({item.module for item in external if item.module})),
        max_depth=deepest,
        truncated=truncated,
    )
