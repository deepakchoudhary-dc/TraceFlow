"""TraceFlow command line (plan.md §8, §44, §72).

Argument parsing uses the standard library. The command surface is small and the
output is plain text, so a CLI framework would add a dependency without removing
any real work.

Every command that needs analysis goes through :func:`_analyse_session` and
:func:`_write_session`. A session must be recorded identically whether it was observed or
requested, or ``traceflow watch`` and ``traceflow analyze`` would slowly come to disagree
about the same repository.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from traceflow import __version__
from traceflow.analysis.impact import build_impact_report, graph_is_stale
from traceflow.analysis.intent import (
    IntentComparison,
    ModuleFacts,
    compare_intent,
)
from traceflow.analysis.models import ImpactReport
from traceflow.analysis.symbols import (
    SymbolReport,
    analyse_session_modules,
    symbol_report_from_session,
)
from traceflow.blobs import BlobStore
from traceflow.config import (
    CONFIG_FILENAME,
    STATE_DIRNAME,
    Config,
    ConfigError,
    load_config,
)
from traceflow.derived import AnalysisCache
from traceflow.git.baseline import Baseline, baseline_from_json, capture_baseline
from traceflow.git.diff import ChangeSet, collect_changes
from traceflow.git.repository import GitError, Repository, WorkingTreeState
from traceflow.languages.python.ast_graph import (
    DependencyGraph,
    PythonFiles,
    build_dependency_graph,
)
from traceflow.scaffold import gitignore_has_entry, initialise
from traceflow.stamps import now_iso
from traceflow.testing import (
    TestRun,
    TestRunStatus,
    run_session_tests,
)
from traceflow.ui.app import DEFAULT_PORT
from traceflow.ui.app import serve as serve_dashboard
from traceflow.ui.delivery import Delivery, load_delivery
from traceflow.visualization import excalidraw, serializer
from traceflow.visualization.graph import GraphView, before_after, build_evidence_graph, change_map
from traceflow.visualization.svg import render_svg
from traceflow.watcher.activity import PollingActivitySource
from traceflow.watcher.quiescence import QuiescenceDetector, WatchState
from traceflow.watcher.session import (
    BASELINE_FILENAME,
    CHANGES_FILENAME,
    CURRENT_BASELINE_FILENAME,
    EVENT_BASELINE_CAPTURED,
    EVENT_SESSION_STABILIZED,
    EVENT_WATCH_STARTED,
    EVENT_WATCH_STOPPED,
    IMPACT_FILENAME,
    INTENT_FILENAME,
    SESSION_FILENAME,
    STATUS_STABILIZED,
    SYMBOLS_FILENAME,
    TESTS_FILENAME,
    Session,
    SessionEvent,
    SessionStore,
    new_session_id,
)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

#: How much of a task's text is echoed on the console. The full text is on the session
#: and on the delivery; the console needs only enough to recognise the session by.
_INTENT_ECHO_LIMIT = 80

GLYPH_ACTIVE = "●"
GLYPH_QUIETING = "◐"
GLYPH_DONE = "✓"
GLYPH_WARN = "⚠"

_RULE_WIDTH = 68


@dataclass
class _PendingSession:
    """State captured when a session begins, consumed when it stabilises.

    The baseline is the one taken when the repository last settled — that is, before
    the change that opened this session. Holding it here rather than re-deriving it
    at stabilisation time is what makes the session's changes attributable.
    """

    monotonic_start: float
    started_iso: str
    baseline: Baseline


def _configure_output() -> None:
    """Force UTF-8 on the output streams.

    Windows consoles still default to a legacy code page, which cannot encode the
    status glyphs below. ``errors="replace"`` guarantees that a terminal which
    genuinely cannot render them degrades to a placeholder instead of raising.

    Streams that are not a real console — a pipe, or a test harness's capture
    object — may refuse reconfiguration. That is not an error, so it is ignored.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError, AttributeError):
            continue


def _rule(character: str = "─") -> str:
    return character * _RULE_WIDTH


def _describe_tree(state: WorkingTreeState) -> str:
    if state.is_clean:
        return "clean"
    parts = []
    if state.tracked_change_count:
        parts.append(f"{state.tracked_change_count} tracked change(s)")
    if state.untracked_count:
        parts.append(f"{state.untracked_count} untracked file(s)")
    return ", ".join(parts) if parts else "clean"


def _discover_or_report(target: Path) -> Repository | None:
    repository = Repository.discover(target)
    if repository is None:
        print(f"error: {target} is not inside a git repository.", file=sys.stderr)
        print(
            "TraceFlow observes repositories through git; see plan.md §7.",
            file=sys.stderr,
        )
    return repository


def _cmd_init(args: argparse.Namespace) -> int:
    target = Path(str(args.path))
    repository = _discover_or_report(target)
    if repository is None:
        return EXIT_ERROR

    report = initialise(repository.root)

    print(f"TraceFlow {__version__} — initialising {repository.name}")
    print(_rule())
    print(f"  Repository   {repository.root}")
    print(f"  State dir    {report.state_dir}")
    if report.config_created:
        print(f"  Config       created {report.config_path.name}")
    else:
        print(f"  Config       kept existing {report.config_path.name}")
    if report.gitignore_updated:
        print(f"  .gitignore   added {STATE_DIRNAME}/ (prevents state being committed)")
    elif report.gitignore_present:
        print(f"  .gitignore   already ignores {STATE_DIRNAME}/")
    else:
        print(f"  {GLYPH_WARN} .gitignore  could not be updated; add {STATE_DIRNAME}/ by hand")

    print()
    print(f"Next: traceflow watch {repository.root}")
    return EXIT_OK


def _describe_baseline(baseline: Baseline) -> str:
    parts = ["dirty" if baseline.dirty else "clean"]
    if baseline.tracked_changes:
        parts.append(f"{baseline.tracked_changes} tracked change(s)")
    if baseline.untracked_files:
        parts.append(f"{baseline.untracked_files} untracked file(s)")
    return ", ".join(parts)


def _describe_changes(changes: ChangeSet) -> str:
    if not changes.files:
        return "no file changes"
    return f"{changes.file_count} file(s), +{changes.insertions}/-{changes.deletions}"


def _describe_symbols(report: SymbolReport) -> str:
    """Summarise symbol-level changes, signature changes first.

    Ordering is deliberate: a signature change is the fact that forces callers to be
    re-examined, so it should never be buried behind a count of body edits.
    """
    parts: list[str] = []
    if report.signature_changes:
        parts.append(f"{len(report.signature_changes)} signature change(s)")
    if report.body_changes:
        parts.append(f"{len(report.body_changes)} body change(s)")
    if report.added:
        parts.append(f"{len(report.added)} symbol(s) added")
    if report.removed:
        parts.append(f"{len(report.removed)} symbol(s) removed")
    if report.imports_added:
        parts.append(f"+{len(report.imports_added)} import(s)")
    if report.imports_removed:
        parts.append(f"-{len(report.imports_removed)} import(s)")
    return ", ".join(parts) if parts else "no symbol changes"


def _describe_impact(impact: ImpactReport) -> str:
    """Summarise impact, review obligations first.

    The direct count is omitted deliberately: those are the session's own changes, which
    the change summary has already reported. Repeating them here would double-count the
    session in the reader's head.
    """
    parts: list[str] = []
    if impact.requiring_review:
        parts.append(f"{len(impact.requiring_review)} needing review")
    for count, label in (
        (len(impact.indirect), "indirect"),
        (len(impact.tests), "test"),
        (len(impact.potential), "potential"),
        (len(impact.configuration), "config"),
        (len(impact.dependencies), "dependency"),
    ):
        if count:
            parts.append(f"{count} {label}")

    if not parts:
        return "impact: nothing beyond the changed files"

    summary = "impact: " + ", ".join(parts)
    if impact.truncated:
        summary += f" (depth limit {impact.max_depth})"
    return summary


def _print_watch_header(
    repository: Repository,
    baseline: Baseline,
    config: Config,
) -> None:
    print(f"TraceFlow {__version__}")
    print(_rule())
    print(f"  Repository    {repository.name}")
    print(f"  Path          {repository.root}")
    print(f"  Status        {GLYPH_ACTIVE} Watching")
    commit = baseline.commit or "(no commits yet)"
    print(f"  Baseline      {commit} — {_describe_baseline(baseline)}")
    print(f"  Quiet period  {config.activity.quiet_period_seconds:g}s")
    print(_rule())
    print("Waiting for activity...")
    print()


def _settled_baseline(
    repository: Repository,
    blobs: BlobStore,
    source: PollingActivitySource,
    state: WorkingTreeState,
    config: Config,
) -> tuple[Baseline, WorkingTreeState]:
    """Capture a baseline, then confirm the repository did not move while doing it.

    Reading the contents of dirty files takes time. A change landing mid-read would
    be baked into the baseline and then reported as pre-existing — a silent
    misattribution. Re-sampling afterwards turns that into a retry instead.
    """
    while True:
        baseline = capture_baseline(repository, blobs, state, config)
        confirmation = source.sample()
        if confirmation.token == state.token:
            return baseline, state
        state = confirmation


def _cmd_watch(args: argparse.Namespace) -> int:
    target = Path(str(args.path))
    repository = _discover_or_report(target)
    if repository is None:
        return EXIT_ERROR

    config = load_config(repository.root)
    store = SessionStore(repository.root)
    store.ensure_state_dir()
    blobs = BlobStore(store.state_dir)

    source = PollingActivitySource(repository, config.ignore)
    detector = QuiescenceDetector(config.activity)

    tree_state = source.sample()
    baseline, tree_state = _settled_baseline(repository, blobs, source, tree_state, config)
    _remember_baseline(store, baseline)

    # Establish the detector's baseline before announcing readiness. The banner
    # printed below claims TraceFlow is waiting for activity, and that claim has to
    # be true: anything that changes after this point belongs to a session, and
    # anything that changed before it is pre-existing (plan.md §14).
    detector.observe(tree_state.token)

    _print_watch_header(repository, baseline, config)

    if not gitignore_has_entry(repository.root, STATE_DIRNAME):
        print(f"  {GLYPH_WARN} {STATE_DIRNAME}/ is not ignored by git.")
        print("     TraceFlow filters it from activity, but its artifacts could be")
        print("     committed. Run `traceflow init` to add the ignore entry.")
        print()

    store.append_event(
        SessionEvent(
            at=now_iso(),
            type=EVENT_WATCH_STARTED,
            detail={
                "repository": repository.name,
                "root": str(repository.root),
                "baseline_commit": baseline.commit,
                "baseline_dirty": baseline.dirty,
            },
        )
    )

    pending: _PendingSession | None = None
    # Built on the first session and reused until a session changes the repository's
    # shape. Building it parses every Python file in the repository, so doing it once per
    # session rather than once per run would make the watcher's cost scale with the
    # repository on every quiet period.
    graph: DependencyGraph | None = None
    last_reported: WatchState | None = None
    session_count = 0

    try:
        while True:
            try:
                state = source.sample()
            except GitError as exc:
                # A transient git failure (an in-flight operation, a locked index)
                # must not kill a long-running watcher. Report it and keep polling;
                # the next successful sample still carries the complete truth.
                print(f"  {GLYPH_WARN} {exc}", file=sys.stderr)
                time.sleep(config.activity.idle_poll_interval_seconds)
                continue

            verdict = detector.observe(state.token)

            started = detector.session_started_at
            if started is not None and (pending is None or pending.monotonic_start != started):
                # `baseline` is the snapshot taken when the repository last settled,
                # which is before the change that opened this session. That is what
                # makes the session's changes attributable and keeps pre-existing
                # modifications out of the answer (plan.md §14).
                pending = _PendingSession(
                    monotonic_start=started,
                    started_iso=now_iso(),
                    baseline=baseline,
                )

            if verdict is not last_reported:
                if verdict is WatchState.ACTIVE:
                    print(f"  {GLYPH_ACTIVE} activity detected — observing")
                elif verdict is WatchState.QUIETING:
                    print(f"  {GLYPH_QUIETING} repository quieting…")
                elif verdict is WatchState.STABLE:
                    assert pending is not None  # a settled state implies a started session

                    # Take the next baseline before closing this session, so that
                    # changes arriving right after the settle belong to the next one.
                    next_baseline, confirmed = _settled_baseline(
                        repository, blobs, source, state, config
                    )
                    if confirmed.token != state.token:
                        # The repository moved while the baseline was being read.
                        # Do not close the session: the change belongs to it.
                        detector.observe(confirmed.token)
                        last_reported = None
                        continue

                    session_count += 1
                    graph = _close_session(
                        store, repository, pending, confirmed, blobs, config, graph
                    )
                    detector.reset(confirmed.token)
                    baseline = next_baseline
                    _remember_baseline(store, next_baseline)
                    pending = None
                    last_reported = WatchState.IDLE
                    continue
                last_reported = verdict

            active = verdict in (WatchState.ACTIVE, WatchState.QUIETING)
            time.sleep(
                config.activity.poll_interval_seconds
                if active
                else config.activity.idle_poll_interval_seconds
            )
    except KeyboardInterrupt:
        print()
        store.append_event(
            SessionEvent(
                at=now_iso(),
                type=EVENT_WATCH_STOPPED,
                detail={"sessions_recorded": session_count},
            )
        )
        print(f"Stopped. {session_count} session(s) recorded in {store.sessions_dir}")
        return EXIT_INTERRUPTED


TRIGGER_WATCHER = "watcher"
TRIGGER_MANUAL = "manual"


def _remember_baseline(store: SessionStore, baseline: Baseline) -> None:
    """Record where the repository last settled.

    ``traceflow analyze`` measures against this, which is what lets a second run report
    "nothing changed" instead of recording the same session again.
    """
    store.write_state(CURRENT_BASELINE_FILENAME, baseline.to_json())


@dataclass
class _SessionFindings:
    """Everything analysis concluded about one session."""

    changes: ChangeSet
    symbols: SymbolReport
    impact: ImpactReport
    graph: DependencyGraph | None = None
    """The dependency graph the walk ran over, carried so the next session can reuse it.

    Not an artifact: it is the structure the analysis used, not a conclusion about the
    session. A caller that closes sessions in a row keeps it; a one-shot command drops it.
    """

    @property
    def is_empty(self) -> bool:
        return not self.changes.files


def _analyse_session(
    repository: Repository,
    baseline: Baseline,
    changes: ChangeSet,
    blobs: BlobStore,
    cache: AnalysisCache,
    config: Config,
    graph: DependencyGraph | None = None,
) -> _SessionFindings:
    """Run the symbol and impact passes over an already-collected change set.

    The changed files are analysed **once**, and both results are projected from that
    single pass: the symbol report records the changes, and the impact walk consumes the
    analyses those changes came from. Running the two passes separately would read and
    analyse every changed file twice, and would let the two disagree about what a file
    contained before the session.

    *graph* may be supplied to reuse a dependency graph an earlier call built. Building it
    parses every Python file in the repository, so a watch loop holds one and rebuilds it
    only when :func:`graph_is_stale` reports that this session changed its shape.

    Separate from writing the result so a caller can decide *not* to record a session.
    ``traceflow analyze`` on an unchanged tree must be able to say "nothing changed"
    rather than adding an empty session to the history.

    The repository is listed **once** and passed to all three passes. Each of them used to
    list it itself, so one ``analyze`` spawned ``git ls-files`` three times for one answer —
    and at ~120ms a spawn that was the largest avoidable cost in the command. It also removes
    a subtler risk: three listings taken at three instants can disagree if the tree moves
    underneath, leaving the analysis reasoning from two pictures of the repository.
    """
    files = PythonFiles.of(repository)
    session = analyse_session_modules(
        repository, baseline, changes, blobs, cache, config, files=files
    )
    symbols = symbol_report_from_session(session)

    if graph is None or graph_is_stale(session):
        graph = build_dependency_graph(repository, cache, config, files=files)

    impact = build_impact_report(
        repository,
        baseline,
        changes,
        blobs,
        cache,
        config,
        graph=graph,
        session=session,
        files=files,
    )
    return _SessionFindings(changes=changes, symbols=symbols, impact=impact, graph=graph)


def _write_session(
    store: SessionStore,
    repository: Repository,
    baseline: Baseline,
    started_iso: str,
    findings: _SessionFindings,
    trigger: str,
    config: Config,
    task: str | None = None,
    tests: TestRun | None = None,
) -> Session:
    """Persist the findings as a session and its artifacts (plan.md §35).

    The impact report is written here rather than recomputed on demand: recomputing later
    would answer for the working tree as it is *then*, not as the session left it — a
    different question, and a misleading one.

    *task* is the intent the user recorded for this session, if any (plan.md §63). The
    comparison is computed here — once, from the same findings every other artifact was
    written from — and stored, so the delivery reads it rather than recomputing it.

    *tests* is the outcome of the configured test run, when the repository enables one
    (plan.md §64). ``None`` means the repository has not enabled test execution, which
    the Tests section states rather than hides.
    """
    changes = findings.changes
    symbols = findings.symbols
    impact = findings.impact

    session = Session(
        session_id=new_session_id(),
        repository=repository.name,
        root=str(repository.root),
        started_at=started_iso,
        stabilized_at=now_iso(),
        baseline_commit=baseline.commit,
        baseline_dirty=baseline.dirty,
        baseline_tracked_changes=baseline.tracked_changes,
        baseline_untracked_files=baseline.untracked_files,
        baseline_id=baseline.baseline_id,
        status=STATUS_STABILIZED.value,
    )

    # Artifacts first, session record last. `sessions` lists a session as soon as
    # `session.json` exists, so writing it first would advertise a session whose change
    # set, symbols and impact might not be on disk yet — and a watcher stopped at that
    # moment would leave exactly that behind. Written last, the presence of the record
    # guarantees everything the record refers to is there.
    store.write_artifact(session.session_id, BASELINE_FILENAME, baseline.to_json())
    store.write_artifact(session.session_id, CHANGES_FILENAME, changes.to_json())
    store.write_artifact(session.session_id, SYMBOLS_FILENAME, symbols.to_json())
    store.write_artifact(session.session_id, IMPACT_FILENAME, impact.to_json())
    if task:
        facts = tuple(
            ModuleFacts(
                path=change.path,
                symbols=tuple(item.qualified_name for item in module.changes),
                imports=module.imports_added + module.imports_removed,
            )
            for change in changes.files
            for module in symbols.modules
            if module.path == change.path
        )
        files_without_symbols = tuple(
            ModuleFacts(path=change.path)
            for change in changes.files
            if change.path not in {facts.path for facts in facts}
        )
        comparison = compare_intent(task, (*facts, *files_without_symbols))
        store.write_artifact(session.session_id, INTENT_FILENAME, comparison.to_json())
    if tests is not None:
        store.write_artifact(session.session_id, TESTS_FILENAME, tests.to_json())
    store.write_session(session)

    detail: dict[str, object] = {
        "trigger": trigger,
        "repository": repository.name,
        "files": changes.file_count,
        "insertions": changes.insertions,
        "deletions": changes.deletions,
        "pre_existing_files": changes.pre_existing_count,
        "contents_withheld": changes.withheld_count,
        "signature_changes": len(symbols.signature_changes),
        "body_changes": len(symbols.body_changes),
        "symbols_added": len(symbols.added),
        "symbols_removed": len(symbols.removed),
        "impact_nodes": len(impact.nodes),
        "requiring_review": len(impact.requiring_review),
        "truncated": impact.truncated,
    }
    if trigger == TRIGGER_WATCHER:
        detail["quiet_period_seconds"] = config.activity.quiet_period_seconds
    if task:
        detail["intent"] = task
    if tests is not None:
        detail["tests"] = tests.status.value

    store.append_event(
        SessionEvent(
            at=now_iso(),
            type=EVENT_SESSION_STABILIZED,
            session_id=session.session_id,
            detail=detail,
        )
    )
    return session


def _close_session(
    store: SessionStore,
    repository: Repository,
    pending: _PendingSession,
    final_state: WorkingTreeState,
    blobs: BlobStore,
    config: Config,
    graph: DependencyGraph | None = None,
) -> DependencyGraph | None:
    """Record the settled session, announce what it found, and hand back the graph.

    The graph is returned rather than discarded because it is the most expensive thing
    this function touches — building it parses every Python file in the repository — and
    it only changes when the session changed the repository's shape.
    """
    baseline = pending.baseline
    cache = AnalysisCache(store.state_dir)
    changes = collect_changes(repository, baseline, final_state, blobs, config)
    findings = _analyse_session(repository, baseline, changes, blobs, cache, config, graph)
    session = _write_session(
        store, repository, baseline, pending.started_iso, findings, TRIGGER_WATCHER, config
    )

    symbols = findings.symbols
    impact = findings.impact

    print(f"  {GLYPH_DONE} session {session.session_id} stabilised — {_describe_changes(changes)}")
    if changes.pre_existing_count:
        print(
            f"      {changes.pre_existing_count} pre-existing change(s) excluded from this session"
        )
    if symbols.has_changes or symbols.parse_errors:
        print(f"      {_describe_symbols(symbols)}")
    if impact.nodes:
        print(f"      {_describe_impact(impact)}")
    print(f"      delivery ready — `traceflow ui {repository.root}`")

    return findings.graph


def _cmd_status(args: argparse.Namespace) -> int:
    target = Path(str(args.path))
    repository = _discover_or_report(target)
    if repository is None:
        return EXIT_ERROR

    config = load_config(repository.root)
    store = SessionStore(repository.root)
    state = repository.working_tree_state(config.ignore)
    sessions = store.list_sessions()

    print(f"TraceFlow {__version__} — status")
    print(_rule())
    print(f"  Repository    {repository.name}")
    print(f"  Path          {repository.root}")
    print(f"  HEAD          {repository.head_commit() or '(no commits yet)'}")
    print(f"  Working tree  {_describe_tree(state)}")
    print(f"  Git busy      {'yes' if repository.is_git_busy() else 'no'}")
    print(f"  Config        {repository.root / CONFIG_FILENAME}")
    print(f"  State dir     {store.state_dir}")
    print(f"  Sessions      {len(sessions)}")
    return EXIT_OK


def _summarise_change_artifact(payload: dict[str, object] | None) -> str:
    """Render the one-line change summary stored alongside a session."""
    if not payload:
        return "no change record"

    totals = payload.get("totals")
    if not isinstance(totals, dict):
        return "no change record"

    summary = (
        f"{totals.get('files', 0)} file(s), "
        f"+{totals.get('insertions', 0)}/-{totals.get('deletions', 0)}"
    )

    pre_existing = totals.get("pre_existing_files", 0)
    if isinstance(pre_existing, int) and pre_existing:
        summary += f", {pre_existing} pre-existing"

    withheld = totals.get("contents_withheld", 0)
    if isinstance(withheld, int) and withheld:
        summary += f", {withheld} withheld"

    return summary


def _summarise_symbol_artifact(payload: dict[str, object] | None) -> str:
    """Render the one-line symbol summary stored alongside a session."""
    if not payload:
        return "no symbol record"

    totals = payload.get("totals")
    if not isinstance(totals, dict):
        return "no symbol record"

    parts: list[str] = []
    for key, label in (
        ("signature_changes", "signature"),
        ("body_changes", "body"),
        ("symbols_added", "added"),
        ("symbols_removed", "removed"),
    ):
        value = totals.get(key, 0)
        if isinstance(value, int) and value:
            parts.append(f"{value} {label}")

    if not parts:
        return "no symbol changes"

    errors = totals.get("parse_errors", 0)
    summary = ", ".join(parts)
    if isinstance(errors, int) and errors:
        summary += f", {errors} unparsable"
    return summary


def _summarise_impact_artifact(payload: dict[str, object] | None) -> str:
    """Render the one-line impact summary stored alongside a session."""
    if not payload:
        return "no impact record"

    totals = payload.get("totals")
    if not isinstance(totals, dict):
        return "no impact record"

    parts: list[str] = []
    for key, label in (
        ("requiring_review", "review"),
        ("indirect", "indirect"),
        ("tests", "test"),
        ("potential", "potential"),
        ("configuration", "config"),
        ("dependencies", "dependency"),
    ):
        value = totals.get(key, 0)
        if isinstance(value, int) and value:
            parts.append(f"{value} {label}")

    if not parts:
        return "nothing beyond the changed files"

    summary = ", ".join(parts)
    if payload.get("truncated") is True:
        summary += f", depth limit {payload.get('max_depth', 0)}"
    return summary


def _cmd_sessions(args: argparse.Namespace) -> int:
    target = Path(str(args.path))
    repository = _discover_or_report(target)
    if repository is None:
        return EXIT_ERROR

    store = SessionStore(repository.root)
    sessions = store.list_sessions()

    print(f"TraceFlow {__version__} — sessions")
    print(_rule())
    if not sessions:
        print("  No sessions recorded yet.")
        print(f"  Run `traceflow watch {repository.root}` and let an agent make changes.")
        return EXIT_OK

    for session in sessions:
        print(f"  {session.session_id}  [{session.status}]")
        print(f"      started     {session.started_at}")
        print(f"      stabilised  {session.stabilized_at or '—'}")
        baseline = "dirty" if session.baseline_dirty else "clean"
        print(
            f"      baseline    {session.baseline_commit or '(no commits yet)'} — {baseline}"
            f" ({session.baseline_tracked_changes} tracked, "
            f"{session.baseline_untracked_files} untracked)"
        )
        changes = store.read_artifact(session.session_id, CHANGES_FILENAME)
        print(f"      changes     {_summarise_change_artifact(changes)}")
        symbols = store.read_artifact(session.session_id, SYMBOLS_FILENAME)
        print(f"      symbols     {_summarise_symbol_artifact(symbols)}")
        impact = store.read_artifact(session.session_id, IMPACT_FILENAME)
        print(f"      impact      {_summarise_impact_artifact(impact)}")
    return EXIT_OK


def _latest_session_id(store: SessionStore) -> str | None:
    """The most recently started session, or ``None`` when there is no history."""
    sessions = store.list_sessions()
    return sessions[-1].session_id if sessions else None


def _resolve_session(store: SessionStore, session_id: str | None) -> str | None:
    """The session a command was asked about, or the most recent one."""
    return session_id if session_id else _latest_session_id(store)


def _print_intent(comparison: IntentComparison) -> None:
    """Render a stored or freshly built comparison (plan.md §63)."""
    if not comparison.is_compared:
        print(f"  Verdict     {comparison.label}")
        for note in comparison.notes:
            print(f"              {note}")
        return

    print(f"  Task        {_echo(comparison.task)}")
    print(f"  Verdict     {comparison.label}")
    print(f"  Files       {comparison.changed_files} changed in this session")

    if comparison.matches:
        print()
        print("  Matched")
        for item in comparison.matches:
            where = item.detail if item.kind == "path" else f"{item.detail} ({item.kind})"
            print(f"      {item.token:<16} {where}")

    if comparison.unmatched:
        print()
        print("  Not matched")
        print(f"      {', '.join(comparison.unmatched)}")

    for note in comparison.notes:
        print()
        print(f"  {note}")

    print()
    print("  This compares wording, not correctness — plan.md §63. A change the task did")
    print("  not name may still be exactly what the task required.")


def _echo(text: str) -> str:
    """A short echo of a task's text for the console."""
    return text if len(text) <= _INTENT_ECHO_LIMIT else f"{text[: _INTENT_ECHO_LIMIT - 3]}..."


def _cmd_intent(args: argparse.Namespace) -> int:
    """Record the task given to the coding agent (plan.md §63).

    Writes ``intent.json`` into a session that has already been recorded, so the
    delivery's Task section exists for a session whose task was only known afterwards.
    A session analysed with ``analyze --intent`` already has the artifact; this command
    is for a session that was watched live.
    """
    target = Path(str(args.path))
    repository = _discover_or_report(target)
    if repository is None:
        return EXIT_ERROR

    store = SessionStore(repository.root)
    session_id = _resolve_session(store, args.session)
    if session_id is None:
        print("error: no sessions recorded yet.", file=sys.stderr)
        return EXIT_ERROR

    if store.read_artifact(session_id, SESSION_FILENAME) is None:
        print(f"error: no session {session_id}.", file=sys.stderr)
        return EXIT_ERROR

    comparison = _build_comparison(store, session_id, str(args.task if args.task else ""))
    store.write_artifact(session_id, INTENT_FILENAME, comparison.to_json())

    print(f"TraceFlow {__version__} — intent")
    print(_rule())
    print(f"  Session     {session_id}")
    _print_intent(comparison)
    return EXIT_OK


def _build_comparison(store: SessionStore, session_id: str, task: str) -> IntentComparison:
    """Build the comparison from the session's own artifacts.

    Reads the artifacts rather than taking analysis results as arguments so the answer is
    the one any later reader of the artifacts would derive — the delivery's Task section
    and this command cannot disagree.

    The change set's file list is deliberately *not* read here: the symbol report
    already names every file it compared, and the change set can additionally hold
    files the analyzer does not handle — which cannot match a task's words by symbol
    or import, only by path. Building facts from both would attach symbols to a file
    twice; building from the change set alone would lose the symbols. The symbol
    report is the richer record, and a file it does not name cannot be matched
    beyond its path, which no artifact disagrees about.
    """
    symbols_payload = store.read_artifact(session_id, SYMBOLS_FILENAME)

    def _strings(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str)]

    def _objects(value: object) -> list[dict[str, object]]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]

    facts = tuple(
        ModuleFacts(
            path=_str_field(module.get("path")),
            symbols=tuple(
                _str_field(change.get("qualified_name"))
                for change in _objects(module.get("changes"))
            ),
            imports=tuple(
                _strings(module.get("imports_added")) + _strings(module.get("imports_removed"))
            ),
        )
        for module in _objects(symbols_payload.get("modules") if symbols_payload else None)
    )

    return compare_intent(task, facts)


_IMPACT_CATEGORIES = ("direct", "indirect", "test", "configuration", "dependency", "potential")

#: Nodes shown per category before the rest are summarised. A report listing two hundred
#: transitive dependents is unreadable; ``--all`` and ``--json`` are there for the cases
#: that need every one.
_NODE_PREVIEW = 15


def _strings(value: object) -> list[str]:
    """Coerce a JSON list into the strings it contains, ignoring anything else."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _objects(value: object) -> list[dict[str, object]]:
    """Coerce a JSON list into the objects it contains, ignoring anything else."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _str_field(value: object, default: str = "?") -> str:
    return value if isinstance(value, str) else default


def _int_field(value: object, default: int = 0) -> int:
    """A whole number, with `bool` excluded — `isinstance(True, int)` is true."""
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


#: One finding: category, subject, reason, distance, confidence, chain.
_Finding = tuple[str, str, str, int, str, list[str]]


def _findings(node: dict[str, object]) -> list[_Finding]:
    """Every finding recorded about one node, its own record first.

    A session can both change a symbol and reach it, and the artifact keeps both — the node's
    own record, and each additional reach. The CLI lists findings rather than nodes so that a
    file the session changed *and* reached appears under both categories; listing only the
    node's own record is what left the "requiring re-examination" count above a shorter list.

    Reading the artifact's own shape is right here: plan.md §35 makes it the interface, and
    this is the one place that prints it rather than rendering it.
    """
    path = _str_field(node.get("path"))
    symbol = node.get("symbol")
    subject = f"{path}::{symbol}" if isinstance(symbol, str) else path

    found: list[_Finding] = [
        (
            _str_field(node.get("category"), "potential"),
            subject,
            _str_field(node.get("reason")),
            _int_field(node.get("distance")),
            _str_field(node.get("confidence")),
            _strings(node.get("chain")),
        )
    ]
    for reach in _objects(node.get("reaches")):
        found.append(
            (
                _str_field(reach.get("category"), "indirect"),
                subject,
                _str_field(reach.get("reason")),
                _int_field(reach.get("distance")),
                _str_field(reach.get("confidence")),
                _strings(reach.get("chain")),
            )
        )
    return found


def _render_finding(finding: _Finding) -> str:
    """One impact line: what, why, how sure, and how it was reached."""
    _category, subject, reason, _distance, confidence, chain = finding
    line = f"      {subject}  {reason} ({confidence})"
    if len(chain) > 1:
        line += f"  via {' -> '.join(chain[:-1])}"
    return line


def _print_impact(payload: dict[str, object], session_id: str, limit: int | None) -> None:
    """Render a stored impact artifact."""
    totals = payload.get("totals")
    totals = totals if isinstance(totals, dict) else {}

    print(f"  Session   {session_id}")
    analyzer = payload.get("analyzer")
    if isinstance(analyzer, str):
        print(f"  Analyzer  {analyzer}")
    print(
        f"  Changed   {totals.get('changed_files', 0)} file(s)"
        f" -> {totals.get('impacted_files', 0)} file(s) impacted"
    )
    print(f"  Review    {totals.get('requiring_review', 0)} node(s) requiring re-examination")

    depth = payload.get("max_depth", 0)
    depth_text = f"{depth}" if not payload.get("truncated") else f"{depth} (truncated)"
    print(f"  Depth     {depth_text}")

    diff = payload.get("graph_diff")
    if isinstance(diff, dict):
        added_modules = _strings(diff.get("modules_added"))
        removed_modules = _strings(diff.get("modules_removed"))
        added_edges = _objects(diff.get("edges_added"))
        removed_edges = _objects(diff.get("edges_removed"))
        if added_modules or removed_modules or added_edges or removed_edges:
            print()
            print("  STRUCTURE")
            print(
                f"      +{len(added_modules)} module(s), -{len(removed_modules)} module(s), "
                f"+{len(added_edges)} relationship(s), -{len(removed_edges)} relationship(s)"
            )
            for path in added_modules:
                print(f"      + {path}")
            for path in removed_modules:
                print(f"      - {path}")
            for edge in added_edges:
                print(f"      + {edge.get('source_path', '?')} -> {edge.get('target_path', '?')}")
            for edge in removed_edges:
                print(f"      - {edge.get('source_path', '?')} -> {edge.get('target_path', '?')}")

    findings = [finding for node in _objects(payload.get("nodes")) for finding in _findings(node)]
    for category in _IMPACT_CATEGORIES:
        group = [finding for finding in findings if finding[0] == category]
        if not group:
            continue
        print()
        print(f"  {category.upper()}  ({len(group)})")
        shown = group if limit is None else group[:limit]
        for finding in shown:
            print(_render_finding(finding))
        if len(shown) < len(group):
            print(f"      ... and {len(group) - len(shown)} more (use --all or --json)")

    limitations = _strings(payload.get("limitations"))
    if limitations:
        print()
        print("  LIMITATIONS")
        for item in limitations:
            print(f"      {item}")


def _cmd_impact(args: argparse.Namespace) -> int:
    """Show what a session's changes reach (plan.md §18, §60).

    Reads the artifact recorded when the session stabilised rather than recomputing it:
    the working tree may have moved on, and the question this command answers is about the
    session, not about the present.
    """
    target = Path(str(args.path))
    repository = _discover_or_report(target)
    if repository is None:
        return EXIT_ERROR

    store = SessionStore(repository.root)
    session_id = str(args.session) if args.session else _latest_session_id(store)

    if args.json:
        # Machine-readable output carries no banner. A caller piping this into a parser
        # should not have to strip a header off the front first.
        payload = store.read_artifact(session_id, IMPACT_FILENAME) if session_id else None
        print(json.dumps(payload or {}, indent=2, sort_keys=True))
        return EXIT_OK

    print(f"TraceFlow {__version__} — impact")
    print(_rule())

    if session_id is None:
        print("  No sessions recorded yet.")
        print(f"  Run `traceflow watch {repository.root}` or `traceflow analyze` first.")
        return EXIT_OK

    payload = store.read_artifact(session_id, IMPACT_FILENAME)
    if payload is None:
        print(f"  Session   {session_id}")
        print("  No impact record for this session.")
        return EXIT_OK

    _print_impact(payload, session_id, None if args.all else _NODE_PREVIEW)
    return EXIT_OK


def _cmd_ui(args: argparse.Namespace) -> int:
    """Serve the delivery dashboard (plan.md §61).

    Renders from the session's artifacts rather than re-analysing the working tree: the
    question the dashboard answers is what the session did, and the working tree may have
    moved on since.
    """
    target = Path(str(args.path))
    repository = _discover_or_report(target)
    if repository is None:
        return EXIT_ERROR

    config = load_config(repository.root)
    store = SessionStore(repository.root)

    return serve_dashboard(
        repository,
        store,
        BlobStore(store.state_dir),
        config,
        session_id=args.session,
        port=args.port,
        should_open=not args.no_open,
    )


def _cmd_analyze(args: argparse.Namespace) -> int:
    """Analyse the working tree against the last recorded baseline (plan.md §44).

    plan.md §44 calls this command out as important for testing, and it is: waiting out
    the quiet period to find out whether analysis works is not a workable loop.

    A session is only recorded when something actually changed. Recording an empty one
    would fill the history with noise and make ``sessions`` useless.
    """
    target = Path(str(args.path))
    repository = _discover_or_report(target)
    if repository is None:
        return EXIT_ERROR

    config = load_config(repository.root)
    store = SessionStore(repository.root)
    store.ensure_state_dir()
    blobs = BlobStore(store.state_dir)

    print(f"TraceFlow {__version__} — analyze")
    print(_rule())
    print(f"  Repository    {repository.name}")

    stored = store.read_state(CURRENT_BASELINE_FILENAME)
    baseline = baseline_from_json(stored) if stored is not None else None
    state = repository.working_tree_state(config.ignore)

    if baseline is None:
        # Nothing has been observed yet, so there is no "before" to measure against.
        # Recording one now is what makes the next run mean something, and saying so is
        # better than reporting an empty change set as though it were a result.
        baseline = capture_baseline(repository, blobs, state, config)
        _remember_baseline(store, baseline)
        store.append_event(
            SessionEvent(
                at=now_iso(),
                type=EVENT_BASELINE_CAPTURED,
                detail={"trigger": TRIGGER_MANUAL, "commit": baseline.commit},
            )
        )
        print("  Baseline      captured — nothing was recorded before this")
        print()
        print("Nothing to analyse yet. Make a change, then run `traceflow analyze` again.")
        return EXIT_OK

    changes = collect_changes(repository, baseline, state, blobs, config)
    if not changes.files:
        print("  Changes       none")
        print()
        print("Nothing changed since the last baseline. No session recorded.")
        return EXIT_OK

    findings = _analyse_session(
        repository, baseline, changes, blobs, AnalysisCache(store.state_dir), config
    )
    impact = findings.impact

    # Tests run after the analysis and before the session is written, so the test
    # process sees the same tree the session analysed, and tests.json lands with the
    # other artifacts (plan.md §64).
    tests: TestRun | None = None
    if config.tests.enabled:
        tests = run_session_tests(
            repository,
            config,
            impacted_paths=impact.impacted_files,
            changed_paths=tuple(change.path for change in changes.files),
        )

    session = _write_session(
        store,
        repository,
        baseline,
        now_iso(),
        findings,
        TRIGGER_MANUAL,
        config,
        task=(str(args.intent).strip() or None) if args.intent else None,
        tests=tests,
    )

    # Advance the baseline to whatever the tree looks like now. A change that lands while
    # analysis is running therefore belongs to the *next* session rather than being folded
    # into this one.
    next_state = repository.working_tree_state(config.ignore)
    _remember_baseline(store, capture_baseline(repository, blobs, next_state, config))

    symbols = findings.symbols
    impact = findings.impact

    print(f"  Session       {session.session_id}")
    print(f"  Changes       {_describe_changes(changes)}")
    if changes.pre_existing_count:
        print(f"  Pre-existing  {changes.pre_existing_count} change(s) excluded")
    if symbols.has_changes or symbols.parse_errors:
        print(f"  Symbols       {_describe_symbols(symbols)}")
    if impact.nodes:
        print(f"  {_describe_impact(impact)}")
    if tests is not None:
        print(f"  Tests         {_describe_test_run(tests)}")
    print()
    print(f"  Delivery ready — `traceflow ui {repository.root}` to view it.")
    return EXIT_OK


def _write_session_tests(store: SessionStore, session_id: str, run: TestRun) -> None:
    """Record a test outcome that actually ran (plan.md §35, §64).

    A NOT_RUN record carries no evidence — it is the absence of a run, which the
    delivery states in words — so only a run that happened gets an artifact.
    """
    if run.status is TestRunStatus.NOT_RUN:
        return
    store.write_artifact(session_id, TESTS_FILENAME, run.to_json())


def _describe_test_run(run: TestRun) -> str:
    """One line about a test attempt, in plan.md §23's careful voice."""
    if run.status is TestRunStatus.PASSED:
        summary = f"passed ({run.passed} passed"
        if run.skipped:
            summary += f", {run.skipped} skipped"
        return summary + f" in {run.duration_seconds:.1f}s)"
    if run.status is TestRunStatus.FAILED:
        return f"failed ({run.failed} failed) — `traceflow ui` shows the output"
    if run.status is TestRunStatus.TIMED_OUT:
        return "timed out — the run was stopped; no result"
    if run.status is TestRunStatus.ERROR:
        return f"not run — {run.note}"
    return f"not run — {run.note}"


def _most_imported(graph: DependencyGraph, limit: int = 10) -> list[tuple[str, int]]:
    """Files ranked by how many other files import them."""
    counts: dict[str, int] = {}
    for edge in graph.edges:
        counts[edge.target_path] = counts.get(edge.target_path, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:limit]


def _cmd_graph(args: argparse.Namespace) -> int:
    """Build and summarise the repository's import graph (plan.md §59).

    Building parses every Python file, so it is done on demand rather than during
    watching. Results are cached by content, so a second run re-parses only what
    changed.
    """
    target = Path(str(args.path))
    repository = _discover_or_report(target)
    if repository is None:
        return EXIT_ERROR

    config = load_config(repository.root)
    store = SessionStore(repository.root)
    graph = build_dependency_graph(repository, AnalysisCache(store.state_dir), config)

    print(f"TraceFlow {__version__} — dependency graph")
    print(_rule())
    print(f"  Repository    {repository.name}")
    print(f"  Modules       {graph.module_count}")
    print(f"  Import edges  {graph.edge_count}")
    print(f"  Unresolved    {len(graph.unresolved)}")

    if graph.parse_errors:
        print(f"  Parse errors  {len(graph.parse_errors)}")
        for error in graph.parse_errors[:5]:
            print(f"      {error}")

    ranked = _most_imported(graph)
    if ranked:
        print()
        print("  Most imported")
        for path, count in ranked:
            print(f"      {count:>4}  {path}")

    return EXIT_OK


#: Where an export goes when no destination is given, and what each format is called on
#: disk. The state directory is the default because `init` puts it in `.gitignore`:
#: writing beside the source would make TraceFlow's own output show up as an untracked
#: file in the next session, which is the product's output masquerading as a change.
EXPORTS_DIRNAME = "exports"

_EXPORT_EXTENSIONS = {"excalidraw": ".excalidraw", "json": ".json", "svg": ".svg"}


def _export_view(delivery: Delivery, view: str) -> GraphView:
    graph = build_evidence_graph(delivery)
    return change_map(graph) if view == "change" else before_after(graph)


def _export_text(view: GraphView, kind: str) -> str:
    if kind == "json":
        return serializer.dumps(serializer.to_dict(view))
    if kind == "excalidraw":
        return excalidraw.dumps(view)
    return render_svg(view)


def _cmd_export(args: argparse.Namespace) -> int:
    """Write a session's map out for another tool (plan.md §32, §44, §62).

    Renders from the recorded artifacts rather than re-analysing, for the same reason the
    dashboard does: what is being exported is what the session concluded, and the working
    tree may have moved on since.
    """
    target = Path(str(args.path))
    repository = _discover_or_report(target)
    if repository is None:
        return EXIT_ERROR

    store = SessionStore(repository.root)
    session_id = args.session or _latest_session_id(store)
    if session_id is None:
        print("error: no sessions recorded yet — nothing to export.", file=sys.stderr)
        return EXIT_ERROR

    delivery = load_delivery(store, session_id)
    if delivery is None:
        print(f"error: no session {session_id}.", file=sys.stderr)
        return EXIT_ERROR

    view = _export_view(delivery, args.view)
    text = _export_text(view, args.format)

    destination = (
        Path(str(args.output))
        if args.output
        else store.state_dir
        / EXPORTS_DIRNAME
        / f"{session_id}-{view.key}{_EXPORT_EXTENSIONS[args.format]}"
    )

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8", newline="\n")
    except OSError as exc:
        print(f"error: cannot write {destination} — {exc}", file=sys.stderr)
        return EXIT_ERROR

    nodes = f"{len(view.nodes)}"
    if view.omitted:
        nodes += f" ({view.omitted} not drawn)"

    print(f"TraceFlow {__version__} — export")
    print(_rule())
    print(f"  Session       {session_id}")
    print(f"  View          {view.title}")
    print(f"  Format        {args.format}")
    print(f"  Nodes         {nodes}")
    print(f"  Written       {destination}")
    return EXIT_OK


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="traceflow",
        description="Local-first, agent-independent change intelligence.",
    )
    parser.add_argument("--version", action="version", version=f"traceflow {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add(
        name: str, help_text: str, handler: Callable[[argparse.Namespace], int]
    ) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument(
            "path",
            nargs="?",
            default=".",
            help="path inside the target repository (default: current directory)",
        )
        sub.set_defaults(handler=handler)
        return sub

    add("init", "prepare a repository for TraceFlow", _cmd_init)
    add("watch", "observe a repository until interrupted", _cmd_watch)
    add("status", "show repository and watcher state", _cmd_status)
    add("sessions", "list recorded sessions", _cmd_sessions)
    add("graph", "build and summarise the dependency graph", _cmd_graph)
    analyze = add(
        "analyze", "analyse the working tree now, without waiting for quiescence", _cmd_analyze
    )
    analyze.add_argument(
        "--intent",
        help="the task given to the coding agent, recorded on the session (plan.md §63)",
    )

    intent = add("intent", "record the task given to the coding agent (plan.md §63)", _cmd_intent)
    intent.add_argument("--task", required=True, help="the task as it was given to the agent")
    intent.add_argument(
        "--session",
        help="session to attach it to (default: the most recent session)",
    )

    impact = add("impact", "show what a session's changes reach", _cmd_impact)
    impact.add_argument("--session", help="session id (default: the most recent session)")
    impact.add_argument("--json", action="store_true", help="print the stored artifact as JSON")
    impact.add_argument("--all", action="store_true", help="show every impacted node")

    ui = add("ui", "serve the delivery dashboard on localhost", _cmd_ui)
    ui.add_argument("--session", help="session to open (default: the most recent session)")
    ui.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"port to listen on (default {DEFAULT_PORT}; 0 picks a free one)",
    )
    ui.add_argument(
        "--no-open", action="store_true", help="print the URL without opening a browser"
    )

    export = add("export", "write a session's change map for another tool", _cmd_export)
    export.add_argument("--session", help="session id (default: the most recent session)")
    export.add_argument(
        "--view",
        choices=("change", "before-after"),
        default="change",
        help="which map to write (default: change)",
    )
    export.add_argument(
        "--format",
        choices=("excalidraw", "json", "svg"),
        default="excalidraw",
        help="output format (default: excalidraw)",
    )
    export.add_argument(
        "--output",
        help=f"file to write (default: {STATE_DIRNAME}/{EXPORTS_DIRNAME}/…)",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _configure_output()
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    try:
        return handler(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except GitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return EXIT_INTERRUPTED


if __name__ == "__main__":  # pragma: no cover - exercised via the console script
    raise SystemExit(main())
