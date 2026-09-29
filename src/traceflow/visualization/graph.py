"""The evidence graph for one session (plan.md §28, §31, §32, §43, §49, §62).

A session's artifacts already say which files changed, which files the change reached, and
which relationships moved. This module arranges that into one graph — nodes are files,
edges are recorded relationships — and offers the two views plan.md §62 asks for: the
focused change map, and the before/after comparison.

**Nodes are files, not symbols.** plan.md §43's sketch is symbol-level, and a symbol-level
map would be prettier. It is not built, because the recorded relationships are between
files: a chain is a sequence of paths, and an added import is declared by one file and
resolved to another. A symbol-level edge would have to be inferred from which names sit
inside which file, and an edge that is inferred rather than recorded is the assertion this
product exists to replace. Each node therefore carries the symbols that were found at it,
so nothing is lost — only the claim that two *symbols* are connected, which no artifact
makes.

**An edge's direction is the direction the artifact recorded.** The change map is drawn in
the impact direction, so an arrow from A to B reads "a change in A reaches B" — which is
what the traversal computed and what each edge's evidence records. The before/after map is
drawn in the import direction, because that is the only direction an import has. The two
are never mixed in one diagram, and each states its arrow in words, because two arrow
meanings on one picture is a puzzle rather than a diagram (plan.md §71).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from traceflow.ui.delivery import (
    NAMED_CATEGORIES,
    Delivery,
    DeliveryFile,
    EvidenceLine,
    ImpactNode,
    reason_label,
    session_url,
    short_symbol,
)

#: How many nodes one diagram will draw before it stops. The map is focused by
#: construction — plan.md §49's traversal only pulls the change's neighbourhood — so this
#: is a safety valve for a session with an unusually wide blast radius, not the focusing
#: mechanism. What it leaves out is counted and stated, never silently dropped.
MAX_MAP_NODES = 40


class NodeState(str, Enum):
    """What a node is doing on the map (plan.md §31).

    The five states are plan.md §31's five colours. Each is also rendered as a word, since
    colour must never be the only indicator.
    """

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    AFFECTED = "affected"
    CONTEXT = "context"


#: Precedence when a file could be described two ways, most specific first. A file the
#: session created is *added* even though it is also a file that changed; a file it
#: deleted is *removed* even though something still imports it. Ordering the states once
#: here is what lets the builder simply upgrade a node as better information arrives.
STATE_ORDER: dict[NodeState, int] = {
    NodeState.ADDED: 0,
    NodeState.REMOVED: 1,
    NodeState.MODIFIED: 2,
    NodeState.AFFECTED: 3,
    NodeState.CONTEXT: 4,
}

_FILE_STATES: dict[str, NodeState] = {
    "added": NodeState.ADDED,
    "deleted": NodeState.REMOVED,
    "removed": NodeState.REMOVED,
    "renamed": NodeState.MODIFIED,
    "modified": NodeState.MODIFIED,
    "copied": NodeState.MODIFIED,
    "type_changed": NodeState.MODIFIED,
}


class EdgeKind(str, Enum):
    """What a recorded relationship is.

    ``AFFECTS`` is the impact direction: the traversal walked from *source* to *target*
    because something in *target* reaches into *source*. The two import kinds are the
    dependency direction, because that is the direction an import is declared in.
    """

    AFFECTS = "affects"
    IMPORT_ADDED = "import_added"
    IMPORT_REMOVED = "import_removed"


_EDGE_LABELS: dict[EdgeKind, str] = {
    EdgeKind.AFFECTS: "affects",
    EdgeKind.IMPORT_ADDED: "imports",
    EdgeKind.IMPORT_REMOVED: "no longer imports",
}

#: Which tone an edge is drawn in, for the renderers. Kept with the vocabulary so the SVG
#: and the Excalidraw export cannot disagree about what a solid line means.
EDGE_TONES: dict[EdgeKind, str] = {
    EdgeKind.AFFECTS: "context",
    EdgeKind.IMPORT_ADDED: "added",
    EdgeKind.IMPORT_REMOVED: "removed",
}

#: The word each state is shown as. plan.md §31 requires a label beside every colour, and
#: these are the labels. "unchanged" rather than "context" because §31's gray means
#: "unchanged context", and a reader should not have to learn a vocabulary to read a map.
STATE_LABELS: dict[NodeState, str] = {
    NodeState.ADDED: "added",
    NodeState.REMOVED: "removed",
    NodeState.MODIFIED: "modified",
    NodeState.AFFECTED: "affected",
    NodeState.CONTEXT: "unchanged",
}

#: A glyph as well as a word, because §31 asks for labels *and* icons, and a reader who
#: cannot separate the red from the green still needs to know which is which.
STATE_GLYPHS: dict[NodeState, str] = {
    NodeState.ADDED: "+",
    NodeState.REMOVED: "-",
    NodeState.MODIFIED: "~",
    NodeState.AFFECTED: "→",
    NodeState.CONTEXT: "·",
}

#: Which tone each state is drawn in. plan.md §31's legend exactly: green added, yellow
#: modified, red removed, blue affected, gray unchanged. "modified" maps to the "changed"
#: tone because that is what the delivery page already calls the yellow.
STATE_TONES: dict[NodeState, str] = {
    NodeState.ADDED: "added",
    NodeState.REMOVED: "removed",
    NodeState.MODIFIED: "changed",
    NodeState.AFFECTED: "affected",
    NodeState.CONTEXT: "context",
}

#: The palette, as ``(strong, background)`` per tone, light theme first. Held here rather
#: than in either renderer because two renderers use it — the SVG, where these are the
#: fallbacks for the page's CSS variables, and the Excalidraw export, which needs literal
#: colours because a drawing handed to another tool has no stylesheet.
TONES: dict[str, tuple[str, str]] = {
    "added": ("#1a7f37", "#dafbe1"),
    "removed": ("#cf222e", "#ffebe9"),
    "changed": ("#9a6700", "#fff8c5"),
    "affected": ("#0969da", "#ddf4ff"),
    "context": ("#57606a", "#eaeef2"),
}

#: The page's remaining colours, so a standalone export is legible without the dashboard's
#: stylesheet. The variable names are the page's, which is what makes the same SVG theme
#: itself inside the dashboard.
INKS: dict[str, str] = {
    "fg": "#1c1e21",
    "muted": "#646a73",
    "line": "#e3e5e8",
    "panel": "#f7f8fa",
}


@dataclass(frozen=True)
class GraphNode:
    """One file on the map."""

    id: str
    """The path, which is what edges refer to. Files are unique in a repository, so the
    path is the natural key and needs no separate identity to be invented."""

    path: str
    state: NodeState
    detail: str = ""
    """One line of why the node is here, in words a reader who has not read plan.md can
    use. One line because that is what fits in a box and what the eye reads."""

    symbols: tuple[str, ...] = ()
    distance: int | None = None
    """Rings from the change, when the traversal recorded one. ``None`` for a node that
    only appears because a relationship moved."""

    reached: tuple[str, ...] = ()
    """Reasons this file was reached for, beyond its own record, already in words.

    A file the session changed *and* reached has both, and the box has to name both: the
    obligation is the actionable half, and a box reading only "1 body change" hides the thing
    the concerns list is counting. Held as words rather than as the delivery's records
    because the map is a rendering — giving it a second copy of the analysis vocabulary is
    how the two would come to disagree.
    """

    evidence: tuple[EvidenceLine, ...] = ()
    link: str | None = None


@dataclass(frozen=True)
class GraphEdge:
    """One recorded relationship."""

    source: str
    target: str
    kind: EdgeKind
    evidence: tuple[EvidenceLine, ...] = ()

    @property
    def key(self) -> tuple[str, str, EdgeKind]:
        return (self.source, self.target, self.kind)

    @property
    def label(self) -> str:
        return _EDGE_LABELS[self.kind]


@dataclass(frozen=True)
class EvidenceGraph:
    """Everything one session recorded about what is connected to what."""

    session_id: str
    repository: str
    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def node_ids(self) -> frozenset[str]:
        return frozenset(node.id for node in self.nodes)


@dataclass(frozen=True)
class GraphView:
    """One diagram: a focused subset of the graph, plus how to read it.

    Self-contained on purpose. A renderer receives a view and needs nothing else — no
    session, no artifact, no knowledge of what the edges mean — which is what keeps the
    SVG, the JSON and the Excalidraw export from each developing their own idea of what
    the picture is.
    """

    key: str
    session_id: str
    repository: str
    title: str
    caption: str
    arrow: str
    empty: str
    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()
    omitted: int = 0
    notes: tuple[str, ...] = ()
    show_edge_labels: bool = False

    @property
    def node_ids(self) -> frozenset[str]:
        return frozenset(node.id for node in self.nodes)


# --------------------------------------------------------------------------- helpers


def _merge(
    existing: tuple[EvidenceLine, ...], extra: tuple[EvidenceLine, ...]
) -> tuple[EvidenceLine, ...]:
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


def _join(parts: list[str], limit: int = 2) -> str:
    """A box has room for about two clauses. Saying how many were left out beats
    truncating a sentence mid-word, which reads as the end of the thought."""
    if not parts:
        return ""
    if len(parts) <= limit:
        return " · ".join(parts)
    return f"{' · '.join(parts[:limit])} · +{len(parts) - limit} more"


def file_detail(file: DeliveryFile) -> str:
    """What changed in one file, in the words the Changes section uses.

    Public because the Graph page's textual listing and the map's boxes must describe a
    file the same way; two wordings for one fact is how a reader learns to distrust both.
    """
    if file.parse_error:
        return "could not be parsed — its symbols are unknown"
    if file.withheld:
        return "contents withheld by the sensitive-path policy"

    parts: list[str] = []
    declarations = sum(1 for item in file.symbols if item.change == "signature_changed")
    bodies = sum(1 for item in file.symbols if item.change == "body_changed")
    added = sum(1 for item in file.symbols if item.change == "added")
    removed = sum(1 for item in file.symbols if item.change == "removed")
    if declarations:
        parts.append(f"{declarations} declaration change(s)")
    if bodies:
        parts.append(f"{bodies} body change(s)")
    if added:
        parts.append(f"{added} symbol(s) added")
    if removed:
        parts.append(f"{removed} symbol(s) removed")
    if file.imports_added or file.imports_removed:
        parts.append(f"+{len(file.imports_added)}/-{len(file.imports_removed)} import(s)")
    if not parts:
        parts.append(file.note or f"no symbol-level change ({file.status})")
    return _join(parts)


def _impact_detail(node: ImpactNode) -> str:
    """The node's own record, in the words the delivery uses.

    Its reaches are added by :func:`_with_reach_detail` once every pass has run, so this
    describes only what the node itself was recorded for — including both would name the same
    reason twice on a symbol whose own record is a reason the walk also reached it for.
    """
    parts: list[str] = []
    if node.category in NAMED_CATEGORIES:
        parts.append(node.category)
    parts.append(reason_label(node.reason, node.distance))
    if node.distance:
        parts.append(f"depth {node.distance}")
    return _join(parts, limit=3)


def _node_link(session_id: str, node: GraphNode, *, changed: bool) -> str:
    """Where a reader goes to check a node.

    A file the session changed has a diff, and the diff is the strongest evidence there
    is. A file it merely reached has no diff, so the link goes to the evidence for the
    relationship that reached it.
    """
    if changed:
        return session_url(session_id, f"/diff/{node.path}")
    return session_url(session_id, "/evidence")


# --------------------------------------------------------------------------- builder


class _Builder:
    """Accumulates nodes and edges, upgrading rather than overwriting.

    Built as a class rather than a function because the same path arrives from several
    artifacts — a file change, a graph-diff entry, an impact node, an edge endpoint — and
    each arrival adds to what is known without being allowed to erase it.
    """

    def __init__(self, delivery: Delivery) -> None:
        self.delivery = delivery
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[tuple[str, str, EdgeKind], GraphEdge] = {}
        self._self_affected: dict[str, int] = {}

    # ------------------------------------------------------------------ nodes

    def ensure(self, path: str, state: NodeState, detail: str = "") -> None:
        existing = self.nodes.get(path)
        if existing is None:
            self.nodes[path] = GraphNode(id=path, path=path, state=state, detail=detail)
            return
        if STATE_ORDER[state] < STATE_ORDER[existing.state]:
            self.nodes[path] = replace(existing, state=state)
        elif detail and not existing.detail:
            self.nodes[path] = replace(existing, detail=detail)

    def add_evidence(self, path: str, evidence: tuple[EvidenceLine, ...]) -> None:
        node = self.nodes.get(path)
        if node is not None:
            self.nodes[path] = replace(node, evidence=_merge(node.evidence, evidence))

    def add_symbols(self, path: str, names: tuple[str, ...]) -> None:
        node = self.nodes.get(path)
        if node is None or not names:
            return
        merged = list(node.symbols)
        for name in names:
            if name not in merged:
                merged.append(name)
        self.nodes[path] = replace(node, symbols=tuple(merged))

    def set_distance(self, path: str, distance: int) -> None:
        node = self.nodes.get(path)
        if node is None:
            return
        if node.distance is None or distance < node.distance:
            self.nodes[path] = replace(node, distance=distance)

    def add_reached(self, path: str, labels: tuple[str, ...]) -> None:
        """Record the reasons a file was reached for, in words, beside its own record."""
        node = self.nodes.get(path)
        if node is None or not labels:
            return
        merged = list(node.reached)
        for label in labels:
            if label not in merged:
                merged.append(label)
        self.nodes[path] = replace(node, reached=tuple(merged))

    # ------------------------------------------------------------------ edges

    def add_edge(
        self, source: str, target: str, kind: EdgeKind, evidence: tuple[EvidenceLine, ...] = ()
    ) -> None:
        if source == target:
            # A file reaching itself. plan.md §16's traversal treats the changed module as
            # its own candidate source, which is right — a symbol can be affected by
            # another symbol in the same file — but an arrow from a box to that same box
            # says nothing a reader can use. It is counted onto the node instead.
            self._self_affected[target] = self._self_affected.get(target, 0) + 1
            return
        key = (source, target, kind)
        existing = self.edges.get(key)
        if existing is None:
            self.edges[key] = GraphEdge(source=source, target=target, kind=kind, evidence=evidence)
        else:
            self.edges[key] = replace(existing, evidence=_merge(existing.evidence, evidence))

    # ------------------------------------------------------------------ passes

    def files(self) -> None:
        for file in self.delivery.files:
            state = _FILE_STATES.get(file.status, NodeState.MODIFIED)
            self.ensure(file.path, state, file_detail(file))
            self.add_symbols(file.path, tuple(item.qualified_name for item in file.symbols))

    def graph_diff(self) -> None:
        """plan.md §28's added and removed nodes and relationships.

        Folded in from the graph diff rather than derived from the file list, because the
        graph diff is the artifact's own statement about the shape of the change — and it
        is the only place a relationship that *left* the session is recorded at all.
        """
        for path in self.delivery.modules_added:
            self.ensure(path, NodeState.ADDED, "added by this session")
        for path in self.delivery.modules_removed:
            self.ensure(path, NodeState.REMOVED, "removed by this session")

        for edge in self.delivery.edges_added:
            self.add_edge(
                edge.source_path,
                edge.target_path,
                EdgeKind.IMPORT_ADDED,
                (
                    EvidenceLine(
                        kind="import_statement",
                        path=edge.source_path,
                        line=edge.line or None,
                        detail=f"imports {edge.module}",
                    ),
                ),
            )
        for edge in self.delivery.edges_removed:
            self.add_edge(
                edge.source_path,
                edge.target_path,
                EdgeKind.IMPORT_REMOVED,
                (
                    EvidenceLine(
                        kind="import_statement",
                        path=edge.source_path,
                        line=edge.line or None,
                        detail=f"no longer imports {edge.module}",
                    ),
                ),
            )

    def impact(self) -> None:
        """The traversal's chains, as edges.

        A chain is the sequence of files the walk followed, so consecutive pairs in it are
        the relationships the walk used. Each hop is evidenced by the record whose location
        is the hop's *target* — the call that reaches back, or the import that no longer
        resolves — which is what makes every arrow on the map checkable.

        Every chain counts, not only the node's own. A file the session changed *and* reached
        has two: the record of its own change, and the path by which something it calls came
        to affect it. Drawing only the first is how the arrow between them went missing.
        """
        for node in self.delivery.nodes:
            state = NodeState.AFFECTED if node.distance else NodeState.MODIFIED
            self.ensure(node.path, state, _impact_detail(node))
            self.set_distance(node.path, node.distance)
            self.add_evidence(node.path, node.evidence)
            if node.symbol:
                self.add_symbols(node.path, (short_symbol(node.symbol),))

            # A file the session changed *and* reached has both facts, and the box names both.
            # Obligations first, so the half the box clips is the half its badge implies.
            obligations = node.obligation_reasons
            ordered = sorted(node.reaches, key=lambda item: item.reason not in obligations)
            self.add_reached(
                node.path,
                tuple(reason_label(item.reason, item.distance) for item in ordered),
            )
            for reach in node.reaches:
                self.add_evidence(node.path, reach.evidence)

            for chain, evidence in node.chains_with_evidence:
                self._draw_chain(chain, evidence)

    def _draw_chain(self, chain: tuple[str, ...], evidence: tuple[EvidenceLine, ...]) -> None:
        for index in range(len(chain) - 1):
            source, target = chain[index], chain[index + 1]
            self.add_edge(
                source,
                target,
                EdgeKind.AFFECTS,
                tuple(item for item in evidence if item.path == target),
            )

    def removals(self) -> None:
        """A module the session removed is still imported somewhere.

        The traversal records the *importer*; the module that went away is the first step
        of its chain, and it has to be on the map for the arrow to land on something.
        """
        counts: dict[str, int] = {}
        for node in self.delivery.nodes:
            for reach in node.reaches:
                if reach.reason == "dangling_import" and reach.chain:
                    counts[reach.chain[0]] = counts.get(reach.chain[0], 0) + 1
            if node.reason == "dangling_import" and node.chain:
                counts[node.chain[0]] = counts.get(node.chain[0], 0) + 1
        for path, count in sorted(counts.items()):
            self.ensure(
                path,
                NodeState.REMOVED,
                f"removed by this session; still imported by {count} file(s)",
            )

    def endpoints(self) -> None:
        """Every path an edge refers to has to be a node, or the edge has nowhere to land.

        These arrive last and only ever as context: a file on the far side of a new import
        did not itself change, and calling it *modified* because something now imports it
        would be the kind of overstatement plan.md §69 rules out.
        """
        for source, target, _kind in sorted(self.edges):
            self.ensure(source, NodeState.CONTEXT)
            self.ensure(target, NodeState.CONTEXT)

    def finish(self) -> EvidenceGraph:
        self.files()
        self.graph_diff()
        self.impact()
        self.removals()
        self.endpoints()

        for path, count in sorted(self._self_affected.items()):
            node = self.nodes.get(path)
            if node is None or node.reached:
                # A file whose own symbol was reached from inside it. When the reach was
                # recorded, it already names the reason and this note would only repeat it;
                # when it was not — the affected symbol had no other record — the note is the
                # only trace of it, which is why the counter exists at all.
                continue
            note = f"{count} affected inside this file"
            detail = f"{node.detail} · {note}" if node.detail else note
            self.nodes[path] = replace(node, detail=detail)

        for path, node in list(self.nodes.items()):
            if node.reached:
                self.nodes[path] = replace(node, detail=_with_reach_detail(node))

        nodes = tuple(
            replace(
                node, link=_node_link(self.delivery.session_id, node, changed=_is_changed(node))
            )
            for node in sorted(
                self.nodes.values(), key=lambda item: (STATE_ORDER[item.state], item.path)
            )
        )
        edges = tuple(
            self.edges[key]
            for key in sorted(self.edges, key=lambda item: (item[2].value, item[0], item[1]))
        )
        return EvidenceGraph(
            session_id=self.delivery.session_id,
            repository=self.delivery.repository,
            nodes=nodes,
            edges=edges,
            notes=_graph_notes(self.delivery),
        )


def _with_reach_detail(node: GraphNode) -> str:
    """A node's own summary, plus the reasons it was reached for — obligations first.

    A box reading only "1 body change" hides the fact that the symbol also calls a
    declaration that moved, which is the actionable half and the thing the concerns list is
    counting. Obligations lead because the box clips its text, and the half that gets clipped
    should be the half the state badge already implies.

    A reason already named in the node's own summary is not repeated. A symbol whose own
    record *is* the obligation — the walk reached it before anything else did — would
    otherwise read "calls a changed declaration · calls a changed declaration · depth 1".
    """
    parts = [label for label in node.reached if label not in node.detail]
    if node.detail:
        parts.append(node.detail)
    return _join(parts, limit=2)


def _is_changed(node: GraphNode) -> bool:
    return node.state in {NodeState.ADDED, NodeState.REMOVED, NodeState.MODIFIED}


def _graph_notes(delivery: Delivery) -> tuple[str, ...]:
    """What the graph cannot show, said on the graph.

    A picture invites the reader to assume it is the whole picture. These are the limits
    the artifacts already declared, restated where the assumption would be made.
    """
    notes: list[str] = []
    if delivery.truncated:
        notes.append(
            f"The impact walk stopped at depth {delivery.max_depth}, so files further out "
            "are not on this map."
        )
    if delivery.unresolved_imports:
        notes.append(
            f"{delivery.unresolved_imports} import(s) point outside the repository, so "
            "relationships through them are not drawn."
        )
    if delivery.totals.parse_errors:
        notes.append(
            f"{delivery.totals.parse_errors} file(s) could not be parsed and have no place "
            "on this map."
        )
    if delivery.external_modules:
        shown = ", ".join(delivery.external_modules[:3])
        more = len(delivery.external_modules) - 3
        suffix = f" and {more} more" if more > 0 else ""
        notes.append(f"Imported but not in this repository: {shown}{suffix}.")
    return tuple(notes)


def build_evidence_graph(delivery: Delivery) -> EvidenceGraph:
    """Project a session's artifacts into the evidence graph.

    Nothing here reads the working tree or the repository. The graph is a rearrangement of
    what the session already concluded, so it cannot disagree with `traceflow impact`.
    """
    return _Builder(delivery).finish()


# --------------------------------------------------------------------------- views


def _capped(
    graph: EvidenceGraph, edges: tuple[GraphEdge, ...], keep: frozenset[str]
) -> tuple[tuple[GraphNode, ...], tuple[GraphEdge, ...], int]:
    """The nodes and edges of one view, capped and reported.

    Selection is nearest-first: a session's own changes, then the rings outward. A map
    that showed a distant consequence but dropped the change that caused it would be worse
    than a smaller map.
    """
    wanted = [node for node in graph.nodes if node.id in keep]
    wanted.sort(key=lambda item: (STATE_ORDER[item.state], item.distance or 0, item.path))
    omitted = max(0, len(wanted) - MAX_MAP_NODES)
    kept = tuple(wanted[:MAX_MAP_NODES])
    ids = frozenset(node.id for node in kept)
    drawn = tuple(edge for edge in edges if edge.source in ids and edge.target in ids)
    return kept, drawn, omitted


def change_map(graph: EvidenceGraph) -> GraphView:
    """The focused change map (plan.md §31, §49).

    Every file the session changed, and every file the change reaches, with the arrows the
    traversal actually followed. Drawn in the impact direction, because that is the
    direction the engine computed and the direction in which every arrow has a call or an
    import behind it.
    """
    edges = tuple(edge for edge in graph.edges if edge.kind is EdgeKind.AFFECTS)
    touched = frozenset(
        node.id for node in graph.nodes if _is_changed(node) or node.state is NodeState.AFFECTED
    )
    for edge in edges:
        touched = touched | {edge.source, edge.target}
    nodes, drawn, omitted = _capped(graph, edges, touched)
    return GraphView(
        key="change",
        session_id=graph.session_id,
        repository=graph.repository,
        title="Change map",
        caption=(
            "The files this session changed, and the files the change reaches. Nothing "
            "disconnected from the change is drawn."
        ),
        arrow="An arrow runs from a change to something it reaches.",
        empty="This session changed nothing the map can show.",
        nodes=nodes,
        edges=drawn,
        omitted=omitted,
        notes=graph.notes,
    )


def before_after(graph: EvidenceGraph) -> GraphView:
    """The structural comparison (plan.md §28).

    The relationships the session added and removed, in the direction an import has: from
    the file that declares it to the file it names. This is the whole of plan.md §28's
    "new relationships" and "removed relationships"; the nodes it names are the two ends
    of each.
    """
    edges = tuple(
        edge
        for edge in graph.edges
        if edge.kind in {EdgeKind.IMPORT_ADDED, EdgeKind.IMPORT_REMOVED}
    )
    touched: frozenset[str] = frozenset()
    for edge in edges:
        touched = touched | {edge.source, edge.target}
    nodes, drawn, omitted = _capped(graph, edges, touched)
    return GraphView(
        key="before-after",
        session_id=graph.session_id,
        repository=graph.repository,
        title="Before and after",
        caption=(
            "The dependency relationships this session added and removed, and the modules "
            "they connect."
        ),
        arrow="An arrow runs from the file that declares the import to the file it imports.",
        empty="No import relationship changed in this session.",
        nodes=nodes,
        edges=drawn,
        omitted=omitted,
        show_edge_labels=True,
        notes=(
            "Only relationships declared by files this session changed are compared "
            "(plan.md §28). An import that moved between two untouched files is not shown.",
        ),
    )
