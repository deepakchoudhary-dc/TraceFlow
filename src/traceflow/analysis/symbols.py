"""Symbol-level change analysis (plan.md §16, §59).

This module bridges the two halves of the system. The git evidence layer establishes
*which files* changed; the language analyzers establish what changed *inside* them.

The case worth understanding is a file that was **already modified when the session
began**. Its pre-session content is not in git — only TraceFlow's own snapshot has it —
so the symbol comparison runs against that snapshot rather than against the last
commit. This is plan.md §14's distinction carried down from files to symbols, and it is
where a naive implementation quietly reports the wrong thing: comparing against the
commit would attribute the earlier edit to this session.

The module's primary output is :func:`analyse_session_modules`, which yields the *before
and after analysis of every changed module* alongside the diff between them. The diff is
what :class:`SymbolReport` records; the analyses are what the impact engine walks. Both
consumers therefore share one answer to "what did this file look like either side of the
session", which is the only way the symbol changes and the impact derived from them can
be guaranteed to agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from traceflow.blobs import BlobStore, digest_of
from traceflow.config import Config
from traceflow.derived import AnalysisCache
from traceflow.git.baseline import Baseline
from traceflow.git.diff import ChangeSet, ChangeStatus, committed_content
from traceflow.git.repository import Repository
from traceflow.languages.base import (
    ModuleAnalysis,
    ModuleSymbolChanges,
    SymbolChange,
    SymbolChangeKind,
    diff_module_analysis,
)
from traceflow.languages.cfamily.graph import CFamilyFiles, cfamily_languages_for
from traceflow.languages.python.ast_graph import (
    PythonFiles,
    analyze_cached,
    list_repository_files,
    module_names_for,
    package_directories,
)
from traceflow.languages.registry import Analyzer, analyzer_for, default_registry
from traceflow.languages.typescript.graph import module_name_for


@dataclass(frozen=True)
class SymbolReport:
    """Symbol-level changes for one session."""

    analyzer: str
    """Which analyzer produced this, including its version.

    Recorded so a session artifact states the basis of its own conclusions. When the
    analyzer changes, an old report is visibly old rather than silently wrong.
    """

    modules: tuple[ModuleSymbolChanges, ...] = ()
    parse_errors: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    """Files that matched the analyzer but could not be read, with the reason."""

    @property
    def module_count(self) -> int:
        return len(self.modules)

    def _all(self, change_kind: SymbolChangeKind) -> tuple[SymbolChange, ...]:
        return tuple(change for module in self.modules for change in module.by_kind(change_kind))

    @property
    def added(self) -> tuple[SymbolChange, ...]:
        return self._all(SymbolChangeKind.ADDED)

    @property
    def removed(self) -> tuple[SymbolChange, ...]:
        return self._all(SymbolChangeKind.REMOVED)

    @property
    def signature_changes(self) -> tuple[SymbolChange, ...]:
        """Symbols whose declaration changed, so every caller must be re-examined."""
        return self._all(SymbolChangeKind.SIGNATURE_CHANGED)

    @property
    def body_changes(self) -> tuple[SymbolChange, ...]:
        """Symbols whose contents changed without their declaration changing."""
        return self._all(SymbolChangeKind.BODY_CHANGED)

    @property
    def imports_added(self) -> tuple[str, ...]:
        return tuple(item for module in self.modules for item in module.imports_added)

    @property
    def imports_removed(self) -> tuple[str, ...]:
        return tuple(item for module in self.modules for item in module.imports_removed)

    @property
    def has_changes(self) -> bool:
        return any(module.has_changes for module in self.modules)

    def to_json(self) -> dict[str, Any]:
        return {
            "analyzer": self.analyzer,
            "totals": {
                "modules": self.module_count,
                "symbols_added": len(self.added),
                "symbols_removed": len(self.removed),
                "signature_changes": len(self.signature_changes),
                "body_changes": len(self.body_changes),
                "imports_added": len(self.imports_added),
                "imports_removed": len(self.imports_removed),
                "parse_errors": len(self.parse_errors),
            },
            "modules": [module.to_json() for module in self.modules],
            "parse_errors": list(self.parse_errors),
            "skipped": list(self.skipped),
        }


def baseline_source(
    repository: Repository, baseline: Baseline, path: str, blobs: BlobStore
) -> bytes | None:
    """The file's content before the session began, from wherever it can be recovered.

    A snapshot is preferred over the commit because a file that was already modified has a
    snapshot but no committed version matching what was on disk.

    A path whose contents were deliberately withheld returns ``None`` rather than falling
    back to git. Reading the committed version would recover exactly what the policy said
    not to read, which would make the guarantee depend on which code path asked.
    """
    digest = baseline.captured_digests().get(path)
    if digest is not None:
        return blobs.get(digest)
    if baseline.was_withheld(path) or baseline.commit is None:
        return None
    return committed_content(repository, baseline.commit, path)


def current_source(repository: Repository, path: str, config: Config) -> tuple[bytes | None, str]:
    """Read the file as it is now, honouring the size policy.

    Returns ``(content, reason)``, where *reason* is empty on success. The reason is
    kept rather than collapsed into ``None`` because "this file was too large to read"
    and "this file could not be read" are different facts, and a report that cannot
    tell them apart cannot explain itself.
    """
    absolute = repository.root / path
    try:
        size = absolute.stat().st_size
    except OSError as exc:
        return None, f"unreadable: {exc.__class__.__name__}"
    if size > config.analysis.max_file_size_bytes:
        return None, "exceeds analysis.max_file_size_mb"
    try:
        return absolute.read_bytes(), ""
    except OSError as exc:
        return None, f"unreadable: {exc.__class__.__name__}"


@dataclass(frozen=True)
class SessionModule:
    """One changed module, either side of the session, and the diff between them.

    Both analyses are kept, not just the diff, because the impact engine needs to walk
    the module's calls and imports — the diff alone says *what* changed but not what
    the file now depends on.
    """

    path: str
    module_name: str | None
    status: ChangeStatus
    before: ModuleAnalysis | None
    after: ModuleAnalysis | None
    changes: ModuleSymbolChanges
    original_path: str | None = None
    """Where the file was before the session, when it was renamed.

    Kept because a rename *removes* the old path from the repository: any file still
    importing it is broken, and finding those files needs the old name.
    """

    @property
    def is_deleted(self) -> bool:
        """True when the file no longer exists, so it has no outgoing edges.

        Derived from the change status rather than from ``after is None``: a deleted file
        is deliberately analysed as an *empty* module so the ordinary comparison reports
        its symbols as removed, which means ``after`` is never ``None`` here.
        """
        return self.status is ChangeStatus.DELETED

    @property
    def is_new(self) -> bool:
        """True when the file did not exist when the session began."""
        return self.status is ChangeStatus.ADDED


@dataclass(frozen=True)
class SessionAnalysis:
    """Every changed module TraceFlow could analyse, plus what it could not."""

    analyzer: str
    modules: tuple[SessionModule, ...] = ()
    parse_errors: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    """Files that matched the analyzer but could not be read, with the reason."""

    def module(self, path: str) -> SessionModule | None:
        for candidate in self.modules:
            if candidate.path == path:
                return candidate
        return None

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(module.path for module in self.modules)


def analyse_session_modules(
    repository: Repository,
    baseline: Baseline,
    change_set: ChangeSet,
    blobs: BlobStore,
    cache: AnalysisCache,
    config: Config,
    analyzer: Analyzer | None = None,
    files: PythonFiles | None = None,
    entries: tuple[Analyzer, ...] | None = None,
    supported_paths: tuple[str, ...] | None = None,
) -> SessionAnalysis:
    """Analyse both versions of every changed file the analyzers can handle.

    Each changed file is dispatched to the language that claims it, so a session that
    edits Python and TypeScript together is one comparison over both.

    *files* keeps its original meaning — a caller-supplied **Python** listing — and pins
    the pass to Python, exactly as before TypeScript existed. *supported_paths* is the
    multi-language listing a caller made once; without either, the repository is listed
    here (a git spawn is expensive enough that one listing per command is worth
    structuring around).
    """
    registry = entries or default_registry()
    if files is not None and supported_paths is not None:
        # The same rule as the impact engine: passing both forms asks for one
        # repository and receives two, and a resolver cannot say which one to
        # answer for. A caller that passes both has made an error worth naming.
        raise ValueError("pass either files= or supported_paths=, not both")
    if files is not None:
        packages = files.packages
    else:
        if supported_paths is None:
            supported_paths = list_repository_files(repository)
        py_paths = tuple(p for p in supported_paths if p.endswith((".py", ".pyi")))
        packages = package_directories(py_paths)

    modules: list[SessionModule] = []
    parse_errors: list[str] = []
    skipped: list[str] = []
    used_kinds: set[str] = set()

    for change in change_set.files:
        engine: Analyzer | None
        if analyzer is not None:
            # A caller pinned the pass to one analyzer: files outside its language
            # take no part, exactly as in the graph builder.
            engine = analyzer if analyzer.can_analyze(change.path) else None
        else:
            engine = analyzer_for(change.path, registry)
        if engine is None:
            continue

        module_name: str | None
        if engine.name == "typescript":
            module_name = module_name_for(change.path)
        elif cfamily_languages_for(change.path) is not None:
            module_name = CFamilyFiles.module_name_for(change.path)
        else:
            names = module_names_for(change.path, packages)
            module_name = names[-1] if names else None

        # A rename is compared against the file's content under its old name, which is
        # where the baseline recorded it.
        baseline_path = change.original_path or change.path
        before_source = baseline_source(repository, baseline, baseline_path, blobs)

        if before_source is None and baseline.was_withheld(baseline_path):
            # The baseline contents were deliberately not stored. Comparing against the
            # commit instead would read exactly what the policy declined to, and comparing
            # against nothing would report every symbol as newly added — so the file is
            # left uncompared and recorded as such.
            skipped.append(f"{change.path}: baseline contents withheld")
            continue

        after_source: bytes | None = None
        if change.status is not ChangeStatus.DELETED:
            after_source, reason = current_source(repository, change.path, config)
            if after_source is None:
                skipped.append(f"{change.path}: {reason}")
                continue

        before = (
            None
            if before_source is None
            else analyze_cached(cache, engine, change.path, before_source, module_name)
        )

        # A deleted file is analysed as an empty module, so the ordinary comparison
        # reports every symbol it had as removed rather than needing a separate path.
        #
        # Its digest is the digest of that empty content, not an empty string. The diff
        # view compares a file's current digest against this one to detect drift, and an
        # empty string can never match any digest — so every deleted file claimed to have
        # been edited after its session.
        after = (
            ModuleAnalysis(path=change.path, digest=digest_of(b""), module_name=module_name)
            if after_source is None
            else analyze_cached(cache, engine, change.path, after_source, module_name)
        )

        changes = diff_module_analysis(before, after)
        if changes.parse_error:
            parse_errors.append(f"{change.path}: {changes.parse_error}")

        used_kinds.add(engine.cache_kind)
        modules.append(
            SessionModule(
                path=change.path,
                module_name=module_name,
                status=change.status,
                before=before,
                after=after,
                changes=changes,
                original_path=change.original_path,
            )
        )

    if used_kinds:
        analyzer_label = "+".join(sorted(used_kinds))
    else:
        analyzer_label = "+".join(sorted(engine.cache_kind for engine in registry))

    return SessionAnalysis(
        analyzer=analyzer_label,
        modules=tuple(modules),
        parse_errors=tuple(parse_errors),
        skipped=tuple(skipped),
    )


def symbol_report_from_session(session: SessionAnalysis) -> SymbolReport:
    """Project a session analysis into the report recorded as an artifact.

    A projection rather than a second pass. The impact engine consumes the analyses the
    changes were derived from, so deriving both from one :func:`analyse_session_modules`
    call is what keeps them from disagreeing about a file — and what stops every changed
    file being read and analysed twice per session.
    """
    return SymbolReport(
        analyzer=session.analyzer,
        modules=tuple(
            module.changes
            for module in session.modules
            if module.changes.has_changes or module.changes.parse_error
        ),
        parse_errors=session.parse_errors,
        skipped=session.skipped,
    )


def collect_symbol_changes(
    repository: Repository,
    baseline: Baseline,
    change_set: ChangeSet,
    blobs: BlobStore,
    cache: AnalysisCache,
    config: Config,
    analyzer: Analyzer | None = None,
) -> SymbolReport:
    """Compare the baseline and current versions of every changed Python file.

    A convenience wrapper for callers that want only the report. A caller that also needs
    the impact analysis should call :func:`analyse_session_modules` once and project from
    that, so the files are read once rather than twice.
    """
    session = analyse_session_modules(
        repository, baseline, change_set, blobs, cache, config, analyzer
    )
    return symbol_report_from_session(session)
