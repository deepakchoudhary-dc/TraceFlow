"""Turning a session's artifacts into something a screen can show (plan.md §30, §35, §61).

A session already wrote everything the delivery view needs, so this module adds no
analysis. Its whole job is to read five JSON documents defensively and hand the renderer
typed values.

That split is deliberate. Artifacts are the source of truth (plan.md §35), and they are
read from disk where a half-written or hand-edited file is possible. Containing every
`isinstance` check in one place means the renderer can be written against types, and a
damaged artifact degrades to a missing section rather than a broken page (plan.md §46).

The **concerns** list is the one thing derived rather than read. It is not new analysis
either — it is the obligations the impact engine already identified, plus the limits it
already declared, gathered into the one place a reviewer looks first. plan.md §74 puts
"POTENTIAL CONCERNS" on the delivery for exactly this reason.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from urllib.parse import quote

from traceflow.analysis.intent import RELATEDNESS_LABELS, IntentMatch, intent_from_json
from traceflow.analysis.models import REVIEW_REASONS
from traceflow.testing import TestRun, test_run_from_json
from traceflow.ui.diff import DiffRequest
from traceflow.watcher.session import (
    BASELINE_FILENAME,
    CHANGES_FILENAME,
    IMPACT_FILENAME,
    INTENT_FILENAME,
    SYMBOLS_FILENAME,
    TESTS_FILENAME,
    SessionStore,
)

REVIEW = "review"
NOTE = "note"

#: Categories in the order the delivery presents them, with what each one means to a
#: reader who has not read plan.md §19.
CATEGORY_LABELS: tuple[tuple[str, str, str], ...] = (
    ("direct", "Direct", "the session's own changes"),
    ("indirect", "Indirect", "callers that must be re-examined"),
    ("test", "Test", "tests this change reaches"),
    ("configuration", "Configuration", "configuration the session touched"),
    ("dependency", "Dependency", "dependency relationships that moved"),
    ("potential", "Potential", "behaviour may differ; no edit is required"),
)

#: A reason in words, never in the artifact's vocabulary (plan.md §71). Kept here rather
#: than in a renderer because two presenters now read it — the HTML delivery and the
#: change map — and a reason named one way on one page and another way on the next is
#: exactly the kind of drift this module exists to prevent.
REASON_LABELS: dict[str, str] = {
    "symbol_added": "symbol added",
    "symbol_signature_changed": "declaration changed",
    "symbol_body_changed": "body changed",
    "content_changed": "content changed",
    "imports_changed": "imports changed",
    "file_added": "file added",
    "file_deleted": "file deleted",
    "file_modified": "file modified",
    "file_renamed": "file renamed",
    "file_copied": "file copied",
    "file_type_changed": "type changed",
    "signature_changed": "calls a changed declaration",
    "body_changed": "calls a changed body",
    "indirect_dependency": "reached through another change",
    "dangling_import": "imports a removed module",
}

#: The categories worth naming on a row, because each names a section elsewhere. Naming
#: "indirect" or "potential" would only repeat the reason beside it.
NAMED_CATEGORIES = frozenset({"test", "configuration", "dependency"})


def reason_label(reason: str, distance: int) -> str:
    """A reason in words.

    ``symbol_removed`` is both the direct reason for a deletion and the transitive reason
    for a reference to one, so distance disambiguates it — the same string means "this was
    deleted" at the root and "this calls something deleted" one ring out.
    """
    if reason == "symbol_removed" and distance == 0:
        return "removed"
    return REASON_LABELS.get(reason, reason.replace("_", " "))


def short_symbol(qualified_name: str) -> str:
    """The last component of a qualified name. The file is usually already on the row."""
    return qualified_name.rsplit(".", 1)[-1]


# --------------------------------------------------------------------------- accessors


def _text(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _opt_text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _int(value: object, default: int = 0) -> int:
    """A whole number, with `bool` excluded — `isinstance(True, int)` is true."""
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _opt_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _flag(value: object, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _objects(value: object) -> tuple[dict[str, object], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _table(payload: dict[str, object] | None, key: str) -> dict[str, object]:
    if payload is None:
        return {}
    nested = payload.get(key)
    return nested if isinstance(nested, dict) else {}


# --------------------------------------------------------------------------- models


@dataclass(frozen=True)
class SymbolLine:
    """One symbol change inside one file."""

    qualified_name: str
    kind: str
    change: str
    line_start: int | None = None
    line_end: int | None = None

    @property
    def is_signature_change(self) -> bool:
        return self.change == "signature_changed"


@dataclass(frozen=True)
class DeliveryFile:
    """One file the session touched."""

    path: str
    status: str
    original_path: str | None = None
    insertions: int | None = None
    deletions: int | None = None
    binary: bool = False
    withheld: bool = False
    note: str | None = None
    symbols: tuple[SymbolLine, ...] = ()
    imports_added: tuple[str, ...] = ()
    imports_removed: tuple[str, ...] = ()
    parse_error: str | None = None
    recorded_digest: str | None = None

    @property
    def changed_lines(self) -> int | None:
        if self.insertions is None or self.deletions is None:
            return None
        return self.insertions + self.deletions

    @property
    def diff_request(self) -> DiffRequest:
        return DiffRequest(
            path=self.path,
            status=self.status,
            original_path=self.original_path,
            recorded_digest=self.recorded_digest,
        )


@dataclass(frozen=True)
class EvidenceLine:
    kind: str
    path: str
    line: int | None
    detail: str

    def render(self) -> str:
        location = f"{self.path}:{self.line}" if self.line else self.path
        return f"{location} {self.detail}"


@dataclass(frozen=True)
class ReachLine:
    """One way a symbol was reached, beyond the node's own record (plan.md §19, §33)."""

    reason: str
    category: str
    distance: int
    chain: tuple[str, ...]
    confidence: str
    evidence: tuple[EvidenceLine, ...] = ()

    @property
    def root(self) -> str:
        """The file whose change reached here — what this finding hangs off."""
        return self.chain[0] if self.chain else ""

    @property
    def is_obligation(self) -> bool:
        return self.reason in REVIEW_REASONS


@dataclass(frozen=True)
class ImpactNode:
    """One component the session reached."""

    path: str
    symbol: str | None
    category: str
    confidence: str
    distance: int
    reason: str
    chain: tuple[str, ...] = ()
    evidence: tuple[EvidenceLine, ...] = ()
    reaches: tuple[ReachLine, ...] = ()
    """Further findings about the same symbol, when the session both changed it and reached
    it. The delivery shows them beside the node's own reason rather than as a second row,
    because a symbol is one component however many ways it was found."""

    @property
    def subject(self) -> str:
        return self.path if self.symbol is None else f"{self.path}::{self.symbol}"

    @property
    def root(self) -> str:
        """The file whose change reached this node — what it hangs off in the delivery."""
        return self.chain[0] if self.chain else self.path

    @property
    def is_obligation(self) -> bool:
        return self.reason in REVIEW_REASONS or any(item.is_obligation for item in self.reaches)

    @property
    def obligation_reasons(self) -> tuple[str, ...]:
        """Every obligation recorded here, the node's own reason first."""
        found = [self.reason] if self.reason in REVIEW_REASONS else []
        found.extend(item.reason for item in self.reaches if item.is_obligation)
        return tuple(found)

    @property
    def reason_distances(self) -> tuple[tuple[str, int, str], ...]:
        """``(reason, distance, confidence)`` for the node's own record and every reach.

        The delivery shows a symbol once and every reason it was recorded for, which is what
        keeps a caller count and the list beneath it in agreement.
        """
        return (
            (self.reason, self.distance, self.confidence),
            *((item.reason, item.distance, item.confidence) for item in self.reaches),
        )

    @property
    def chains_with_evidence(
        self,
    ) -> tuple[tuple[tuple[str, ...], tuple[EvidenceLine, ...]], ...]:
        """Every path the walk took to this symbol, paired with the records for *that* path.

        The pairing matters. A file the session changed *and* reached has two paths, and the
        records supporting one do not support the other — attaching the node's own evidence to
        a reach's arrow would cite something that says nothing about it.
        """
        return (
            (self.chain, self.evidence),
            *((item.chain, item.evidence) for item in self.reaches),
        )


@dataclass(frozen=True)
class EdgeRef:
    source_path: str
    target_path: str
    module: str
    line: int

    def render(self) -> str:
        return f"{self.source_path} → {self.target_path}"


@dataclass(frozen=True)
class Finding:
    """One reason to look at one symbol (plan.md §19, §74).

    A symbol the session both *changed* and *reached* produces two of these. The delivery
    asks "what must be re-examined?" per finding and "what is affected?" per node, and both
    read the same records — so a count of callers and the list beneath it cannot disagree.

    The category is the finding's own, not the node's: a file the session changed is DIRECT,
    while the fact that it also calls a declaration that moved is INDIRECT, and a list
    grouped by category has to put each where it belongs.
    """

    node: ImpactNode
    category: str
    reason: str
    distance: int
    chain: tuple[str, ...]
    confidence: str
    evidence: tuple[EvidenceLine, ...] = ()

    @property
    def subject(self) -> str:
        return self.node.subject

    @property
    def path(self) -> str:
        return self.node.path

    @property
    def root(self) -> str:
        return self.chain[0] if self.chain else self.path

    @property
    def is_obligation(self) -> bool:
        return self.reason in REVIEW_REASONS

    @property
    def is_transitive(self) -> bool:
        return self.distance > 1


@dataclass(frozen=True)
class Concern:
    """Something worth a human's attention, traceable to the artifact it came from."""

    severity: str
    title: str
    detail: str
    links: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class IntentView:
    """The recorded task and what the session actually did, side by side (plan.md §63).

    A read-only projection of ``intent.json``: the comparison was computed once, when the
    task was recorded, and every presenter renders the same stored verdict.
    """

    task: str
    relatedness: str
    label: str
    tokens: tuple[str, ...] = ()
    matches: tuple[IntentMatch, ...] = ()
    unmatched: tuple[str, ...] = ()
    changed_files: int = 0
    notes: tuple[str, ...] = ()

    @property
    def is_compared(self) -> bool:
        return self.relatedness != "not_compared"


@dataclass(frozen=True)
class TestRunView:
    """One stored test run, ready to render (plan.md §23).

    ``None`` on the delivery means the repository has not enabled test execution —
    a fact about configuration, which the Tests section states plainly rather than
    decorating with a fake "no result".
    """

    run: TestRun

    @property
    def label(self) -> str:
        """plan.md §23's exact distinction: a result about the run, not the change."""
        run = self.run
        if run.status.value == "passed":
            return "Tests passed"
        if run.status.value == "failed":
            return "Tests failed"
        if run.status.value == "timed_out":
            return "Test run timed out"
        if run.status.value == "error":
            return "Tests could not be run"
        return "Tests not run"


@dataclass(frozen=True)
class Totals:
    files: int = 0
    insertions: int = 0
    deletions: int = 0
    pre_existing: int = 0
    withheld: int = 0
    symbols_added: int = 0
    symbols_removed: int = 0
    signature_changes: int = 0
    body_changes: int = 0
    parse_errors: int = 0
    impact_nodes: int = 0
    requiring_review: int = 0
    tests: int = 0


@dataclass(frozen=True)
class Delivery:
    """One session, ready to render."""

    session_id: str
    repository: str
    root: str
    started_at: str
    stabilized_at: str | None
    status: str
    baseline_commit: str | None
    baseline_dirty: bool
    baseline_tracked_changes: int
    baseline_untracked_files: int
    baseline_captured: int
    analyzer: str
    totals: Totals
    files: tuple[DeliveryFile, ...] = ()
    pre_existing: tuple[DeliveryFile, ...] = ()
    nodes: tuple[ImpactNode, ...] = ()
    modules_added: tuple[str, ...] = ()
    modules_removed: tuple[str, ...] = ()
    edges_added: tuple[EdgeRef, ...] = ()
    edges_removed: tuple[EdgeRef, ...] = ()
    external_modules: tuple[str, ...] = ()
    unresolved_imports: int = 0
    limitations: tuple[str, ...] = ()
    max_depth: int = 0
    truncated: bool = False
    concerns: tuple[Concern, ...] = ()
    intent: IntentView | None = None
    test_run: TestRunView | None = None
    """The configured test run, when the repository enabled one (plan.md §64).

    Distinct from the :attr:`tests` property, which is the tests the change *reaches* —
    a fact of analysis that needs no execution. This field is a fact of execution.
    """
    """The recorded task and its comparison with the actual change (plan.md §63).

    ``None`` when the session recorded no task — a session recorded before the intent
    phase existed, or one whose task was simply never given.
    """

    missing: tuple[str, ...] = ()
    """Artifacts that should have been there and were not."""

    not_read: tuple[tuple[str, str], ...] = ()
    """``(path, reason)`` for every file whose contents were deliberately not read.

    Gathered from both places it can be recorded: the baseline, for a file that was already
    dirty or untracked when the session began, and the change set, for a file the session
    touched. Reading only one of them made the Session page claim "every file was read"
    while the concerns list said otherwise — a false assurance, which is worse than no
    assurance at all.
    """

    # ------------------------------------------------------------------ convenience

    def by_category(self, category: str) -> tuple[ImpactNode, ...]:
        return tuple(node for node in self.nodes if node.category == category)

    @property
    def tests(self) -> tuple[ImpactNode, ...]:
        return self.by_category("test")

    @property
    def requiring_review(self) -> tuple[ImpactNode, ...]:
        return tuple(node for node in self.nodes if node.is_obligation)

    @property
    def findings(self) -> tuple[Finding, ...]:
        """Every recorded finding: each node's own record, plus each additional reach."""
        found: list[Finding] = []
        for node in self.nodes:
            found.append(
                Finding(
                    node=node,
                    category=node.category,
                    reason=node.reason,
                    distance=node.distance,
                    chain=node.chain,
                    confidence=node.confidence,
                    evidence=node.evidence,
                )
            )
            found.extend(
                Finding(
                    node=node,
                    category=item.category,
                    reason=item.reason,
                    distance=item.distance,
                    chain=item.chain,
                    confidence=item.confidence,
                    evidence=item.evidence,
                )
                for item in node.reaches
            )
        return tuple(found)

    def findings_from(self, path: str) -> tuple[Finding, ...]:
        """Findings that hang off *path* and are not the change itself (plan.md §74)."""
        return tuple(
            item for item in self.findings if item.root == path and item.category != "direct"
        )

    def findings_by_category(self, category: str) -> tuple[Finding, ...]:
        """Findings in one category. Counted here so a heading and its list agree."""
        return tuple(item for item in self.findings if item.category == category)

    @property
    def has_impact(self) -> bool:
        """True when the session reached anything beyond its own changes.

        A node whose own record is the change still counts when it carries a reach: the file
        changed, and something it calls moved.
        """
        return any(item.category != "direct" for item in self.findings)

    @property
    def impact_roots(self) -> tuple[str, ...]:
        """The files that reached something, in the order the delivery should show them."""
        roots: list[str] = []
        for node in self.nodes:
            for path in _finding_roots(node):
                if path not in roots:
                    roots.append(path)
        return tuple(roots)

    def concern_links(self, suffix: str = "") -> tuple[tuple[str, str], ...]:
        return ((f"Session {self.session_id}", session_url(self.session_id, suffix)),)


def session_url(session_id: str, suffix: str = "") -> str:
    """The canonical URL for a session view. Built in one place so links cannot drift."""
    return f"/session/{quote(session_id, safe='')}{suffix}"


def _finding_roots(node: ImpactNode) -> tuple[str, ...]:
    """Every root a non-direct finding about *node* hangs off, in order and deduplicated.

    The node's own record counts only when it is not the change itself — a file the session
    changed hangs off nothing, it *is* the origin. Each reach counts, because a reach is by
    definition something the walk arrived at from somewhere else.
    """
    found: list[str] = [] if node.category == "direct" else [node.root]
    found.extend(item.root for item in node.reaches)

    roots: list[str] = []
    for path in found:
        if path and path not in roots:
            roots.append(path)
    return tuple(roots)


# --------------------------------------------------------------------------- assembly


def _symbol_lines(module: dict[str, object] | None) -> tuple[SymbolLine, ...]:
    if module is None:
        return ()
    return tuple(
        SymbolLine(
            qualified_name=_text(item.get("qualified_name"), "?"),
            kind=_text(item.get("kind"), "symbol"),
            change=_text(item.get("change"), "changed"),
            line_start=_opt_int(item.get("line_start")),
            line_end=_opt_int(item.get("line_end")),
        )
        for item in _objects(module.get("changes"))
    )


def _file_from(entry: dict[str, object], module: dict[str, object] | None) -> DeliveryFile:
    return DeliveryFile(
        path=_text(entry.get("path"), "?"),
        status=_text(entry.get("status"), "unknown"),
        original_path=_opt_text(entry.get("original_path")),
        insertions=_opt_int(entry.get("insertions")),
        deletions=_opt_int(entry.get("deletions")),
        binary=_flag(entry.get("binary")),
        withheld=_flag(entry.get("contents_withheld")),
        note=_opt_text(entry.get("note")),
        symbols=_symbol_lines(module),
        imports_added=_strings(module.get("imports_added")) if module else (),
        imports_removed=_strings(module.get("imports_removed")) if module else (),
        parse_error=_opt_text(module.get("parse_error")) if module else None,
        recorded_digest=_opt_text(module.get("after_digest")) if module else None,
    )


def _evidence_lines(value: object) -> tuple[EvidenceLine, ...]:
    return tuple(
        EvidenceLine(
            kind=_text(record.get("kind"), "evidence"),
            path=_text(record.get("path"), "?"),
            line=_opt_int(record.get("line")),
            detail=_text(record.get("detail")),
        )
        for record in _objects(value)
    )


def _reaches(value: object) -> tuple[ReachLine, ...]:
    return tuple(
        ReachLine(
            reason=_text(item.get("reason"), "unknown"),
            category=_text(item.get("category"), "indirect"),
            distance=_int(item.get("distance")),
            chain=_strings(item.get("chain")),
            confidence=_text(item.get("confidence"), "unknown"),
            evidence=_evidence_lines(item.get("evidence")),
        )
        for item in _objects(value)
    )


def _nodes(payload: dict[str, object] | None) -> tuple[ImpactNode, ...]:
    return tuple(
        ImpactNode(
            path=_text(item.get("path"), "?"),
            symbol=_opt_text(item.get("symbol")),
            category=_text(item.get("category"), "potential"),
            confidence=_text(item.get("confidence"), "unknown"),
            distance=_int(item.get("distance")),
            reason=_text(item.get("reason"), "unknown"),
            chain=_strings(item.get("chain")),
            evidence=_evidence_lines(item.get("evidence")),
            reaches=_reaches(item.get("reaches")),
        )
        for item in _objects(payload.get("nodes") if payload else None)
    )


def _edges(value: object) -> tuple[EdgeRef, ...]:
    return tuple(
        EdgeRef(
            source_path=_text(item.get("source_path"), "?"),
            target_path=_text(item.get("target_path"), "?"),
            module=_text(item.get("module")),
            line=_int(item.get("line")),
        )
        for item in _objects(value)
    )


def _totals_from(payload: dict[str, object] | None) -> Totals:
    table = _table(payload, "totals")
    return Totals(
        files=_int(table.get("files")),
        insertions=_int(table.get("insertions")),
        deletions=_int(table.get("deletions")),
        pre_existing=_int(table.get("pre_existing_files")),
        withheld=_int(table.get("contents_withheld")),
    )


def load_delivery(store: SessionStore, session_id: str) -> Delivery | None:
    """Read a session's artifacts into a :class:`Delivery`, or ``None`` if it is not there.

    A missing artifact is recorded in ``missing`` rather than treated as fatal: a session
    recorded before a phase existed is still worth showing, with the sections it does not
    have left out.
    """
    session_payload = store.read_artifact(session_id, "session.json")
    if session_payload is None:
        return None

    changes_payload = store.read_artifact(session_id, CHANGES_FILENAME)
    symbols_payload = store.read_artifact(session_id, SYMBOLS_FILENAME)
    impact_payload = store.read_artifact(session_id, IMPACT_FILENAME)
    baseline_payload = store.read_artifact(session_id, BASELINE_FILENAME)
    intent_payload = store.read_artifact(session_id, INTENT_FILENAME)
    tests_payload = store.read_artifact(session_id, TESTS_FILENAME)

    missing: list[str] = []
    if changes_payload is None:
        missing.append("changes.json")
    if symbols_payload is None:
        missing.append("symbols.json")
    if impact_payload is None:
        missing.append("impact.json")
    if baseline_payload is None:
        missing.append("baseline.json")

    modules: dict[str, dict[str, object]] = {}
    for module in _objects(symbols_payload.get("modules") if symbols_payload else None):
        modules[_text(module.get("path"))] = module

    files = tuple(
        _file_from(entry, modules.get(_text(entry.get("path"))))
        for entry in _objects(changes_payload.get("files") if changes_payload else None)
    )
    pre_existing = tuple(
        _file_from(entry, None)
        for entry in _objects(changes_payload.get("pre_existing") if changes_payload else None)
    )

    nodes = _nodes(impact_payload)
    graph_diff = _table(impact_payload, "graph_diff")
    totals = _totals_from(changes_payload)
    symbols_total = _table(symbols_payload, "totals")

    withheld: dict[str, str] = {}
    for item in _objects(baseline_payload.get("captured") if baseline_payload else None):
        reason = _opt_text(item.get("withheld_reason"))
        if reason is not None:
            withheld[_text(item.get("path"), "?")] = reason
    for entry in files:
        if entry.withheld:
            withheld.setdefault(entry.path, entry.note or "sensitive path")

    delivery = Delivery(
        session_id=session_id,
        repository=_text(session_payload.get("repository"), "?"),
        root=_text(session_payload.get("root")),
        started_at=_text(session_payload.get("started_at"), "?"),
        stabilized_at=_opt_text(session_payload.get("stabilized_at")),
        status=_text(session_payload.get("status"), "unknown"),
        baseline_commit=_opt_text(session_payload.get("baseline_commit")),
        baseline_dirty=_flag(session_payload.get("baseline_dirty")),
        baseline_tracked_changes=_int(session_payload.get("baseline_tracked_changes")),
        baseline_untracked_files=_int(session_payload.get("baseline_untracked_files")),
        baseline_captured=len(
            _objects(baseline_payload.get("captured") if baseline_payload else None)
        ),
        not_read=tuple(sorted(withheld.items())),
        analyzer=_text((impact_payload or symbols_payload or {}).get("analyzer"), "not recorded"),
        totals=replace(
            totals,
            symbols_added=_int(symbols_total.get("symbols_added")),
            symbols_removed=_int(symbols_total.get("symbols_removed")),
            signature_changes=_int(symbols_total.get("signature_changes")),
            body_changes=_int(symbols_total.get("body_changes")),
            parse_errors=_int(symbols_total.get("parse_errors")),
            impact_nodes=len(nodes),
            requiring_review=sum(1 for node in nodes if node.is_obligation),
            tests=sum(1 for node in nodes if node.category == "test"),
        ),
        files=files,
        pre_existing=pre_existing,
        nodes=nodes,
        modules_added=_strings(graph_diff.get("modules_added")),
        modules_removed=_strings(graph_diff.get("modules_removed")),
        edges_added=_edges(graph_diff.get("edges_added")),
        edges_removed=_edges(graph_diff.get("edges_removed")),
        external_modules=_strings(
            impact_payload.get("external_modules") if impact_payload else None
        ),
        unresolved_imports=_int(
            impact_payload.get("unresolved_imports") if impact_payload else None
        ),
        limitations=_strings(impact_payload.get("limitations") if impact_payload else None),
        max_depth=_int(impact_payload.get("max_depth") if impact_payload else None),
        truncated=_flag(impact_payload.get("truncated") if impact_payload else None),
        missing=tuple(missing),
    )
    intent = _intent_from(intent_payload)
    if intent is not None:
        delivery = replace(delivery, intent=intent)
    stored_run = test_run_from_json(tests_payload)
    if stored_run is not None:
        delivery = replace(delivery, test_run=TestRunView(run=stored_run))
    return replace(delivery, concerns=derive_concerns(delivery))


# --------------------------------------------------------------------------- intent


def _intent_from(payload: dict[str, object] | None) -> IntentView | None:
    """The stored comparison, or ``None`` when the session recorded no task.

    A malformed intent artifact degrades to no Task section rather than a broken page,
    the same rule every other artifact follows (plan.md §46).
    """
    comparison = intent_from_json(payload)
    if comparison is None:
        return None
    return IntentView(
        task=comparison.task,
        relatedness=comparison.relatedness.value,
        label=RELATEDNESS_LABELS[comparison.relatedness],
        tokens=comparison.tokens,
        matches=comparison.matches,
        unmatched=comparison.unmatched,
        changed_files=comparison.changed_files,
        notes=comparison.notes,
    )


# --------------------------------------------------------------------------- concerns


def derive_concerns(delivery: Delivery) -> tuple[Concern, ...]:
    """Gather the obligations and limits already identified into one list (plan.md §74).

    Nothing here is inferred: every entry restates something an artifact already says. A
    concern that could not be traced back to a record would be a verdict, and TraceFlow
    does not issue verdicts (plan.md §69).
    """
    concerns: list[Concern] = []
    impact_link = delivery.concern_links("/impact")
    evidence_link = delivery.concern_links("/evidence")

    for artifact in delivery.missing:
        concerns.append(
            Concern(
                severity=NOTE,
                title=f"{artifact} is missing",
                detail=(
                    "This session was recorded before that artifact existed, or the file "
                    "could not be read. The sections it feeds are omitted rather than guessed."
                ),
            )
        )

    def reaching(reason: str) -> list[Finding]:
        """Findings with this reason that are *not* the change itself.

        Distance 0 is the change, so counting it here would report a declaration as one of
        its own callers — which is how a session with one caller came to claim two. A reach
        always counts: it is by construction something the walk arrived at, even when it sits
        on a node whose own record is the change. Counting only the node's own record is how
        a session with three call sites of a moved declaration came to report one.
        """
        return [item for item in delivery.findings if item.reason == reason and item.distance > 0]

    signature_callers = reaching("signature_changed")
    if signature_callers:
        concerns.append(
            Concern(
                severity=REVIEW,
                title=f"{len(signature_callers)} caller(s) of a changed declaration",
                detail=(
                    "These symbols are called by something whose declaration moved, so every "
                    "call site has to be re-examined — the change may not be source-compatible."
                ),
                links=impact_link,
            )
        )

    removed_hits = reaching("symbol_removed")
    if removed_hits:
        concerns.append(
            Concern(
                severity=REVIEW,
                title=f"{len(removed_hits)} reference(s) to a removed symbol",
                detail=(
                    "A symbol this session removed is still referenced. Those references "
                    "cannot resolve."
                ),
                links=impact_link,
            )
        )

    dangling = reaching("dangling_import")
    if dangling:
        importers = ", ".join(sorted({item.path for item in dangling})[:5])
        targets = ", ".join(sorted({item.root for item in dangling})[:5])
        concerns.append(
            Concern(
                severity=REVIEW,
                title=f"{len(dangling)} file(s) import something this session removed",
                detail=(
                    f"{importers} still import {targets}, which no longer exists. "
                    f"Those imports cannot resolve."
                ),
                links=impact_link,
            )
        )
    elif delivery.modules_removed:
        # Reported separately only when nothing still references them; otherwise the
        # consequence above already says what was removed and why it matters.
        concerns.append(
            Concern(
                severity=REVIEW,
                title=f"{len(delivery.modules_removed)} module(s) removed",
                detail=", ".join(delivery.modules_removed[:5]),
                links=impact_link,
            )
        )

    if delivery.truncated:
        concerns.append(
            Concern(
                severity=REVIEW,
                title="The impact walk was cut short",
                detail=(
                    f"It stopped at depth {delivery.max_depth}, so dependents further out "
                    f"are not listed. Raise analysis.impact_max_depth to see them."
                ),
                links=impact_link,
            )
        )

    if delivery.unresolved_imports:
        concerns.append(
            Concern(
                severity=NOTE,
                title=f"{delivery.unresolved_imports} import(s) point outside the repository",
                detail=(
                    "Their dependencies are not represented, so impact through them is not "
                    "visible. Some are third-party packages; the rest are worth a look."
                ),
                links=evidence_link,
            )
        )

    if delivery.limitations:
        concerns.append(
            Concern(
                severity=NOTE,
                title=f"{len(delivery.limitations)} limit(s) on this analysis",
                detail=(
                    "Dynamic dispatch, wildcard imports or unparsable files stopped the "
                    "analysis reaching further. Listed in full under Evidence."
                ),
                links=evidence_link,
            )
        )

    if delivery.totals.parse_errors:
        concerns.append(
            Concern(
                severity=NOTE,
                title=f"{delivery.totals.parse_errors} file(s) could not be parsed",
                detail="Their symbols are unknown, so nothing about them is reported.",
                links=evidence_link,
            )
        )

    if delivery.not_read:
        concerns.append(
            Concern(
                severity=NOTE,
                title=f"{len(delivery.not_read)} file(s) were not read",
                detail=(
                    "They matched the sensitive-path policy, so their contents were never "
                    "read and their changes cannot be described."
                ),
                links=delivery.concern_links("/session"),
            )
        )

    if delivery.pre_existing:
        concerns.append(
            Concern(
                severity=NOTE,
                title=f"{len(delivery.pre_existing)} change(s) pre-date this session",
                detail=(
                    "They were already present when the session began and are excluded from "
                    "it — reported so they are not mistaken for the session's work."
                ),
                links=delivery.concern_links("/session"),
            )
        )

    if delivery.tests:
        concerns.append(
            Concern(
                severity=NOTE,
                title=f"{len(delivery.tests)} test(s) reach this change",
                detail=(
                    "Test results are not captured yet (plan.md §64), so this says which "
                    "tests are affected, not whether they passed."
                ),
                links=impact_link,
            )
        )

    if delivery.test_run is not None:
        run = delivery.test_run.run
        if run.status.value == "failed":
            concerns.append(
                Concern(
                    severity=REVIEW,
                    title=f"Tests failed ({run.failed} failed)",
                    detail=(
                        "The configured test command exited non-zero for the tests this "
                        "session reaches. A failed run is evidence about the tests, not a "
                        "verdict on the change — the output tail is under Evidence (plan.md §23)."
                    ),
                    links=delivery.concern_links("/evidence"),
                )
            )
        elif run.status.value == "timed_out":
            concerns.append(
                Concern(
                    severity=NOTE,
                    title="The test run was stopped",
                    detail=run.note or "It exceeded the configured timeout, so there is no result.",
                    links=delivery.concern_links("/session"),
                )
            )
        elif run.status.value == "error":
            concerns.append(
                Concern(
                    severity=NOTE,
                    title="Tests could not be run",
                    detail=run.note or "The configured command could not be executed.",
                    links=delivery.concern_links("/session"),
                )
            )

    if delivery.intent is not None and delivery.intent.relatedness == "scope_expansion":
        unmatched = ", ".join(delivery.intent.unmatched[:5])
        concerns.append(
            Concern(
                severity=NOTE,
                title="Potential scope expansion",
                detail=(
                    f"The recorded task names {len(delivery.intent.tokens)} thing(s) and "
                    f"{len(delivery.intent.unmatched)} of them match nothing the session "
                    f"touched ({unmatched}). Static analysis cannot say whether they were "
                    "unnecessary — only that no detected relationship ties them to the task."
                ),
                links=delivery.concern_links("/session"),
            )
        )

    if not concerns:
        concerns.append(
            Concern(
                severity=NOTE,
                title="Nothing flagged",
                detail=(
                    "No caller of a changed declaration, no removed symbol left referenced, "
                    "and no limit on the analysis. The change is local to what changed."
                ),
            )
        )

    return tuple(concerns)
