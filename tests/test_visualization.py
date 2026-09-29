"""The evidence graph, its layout and its renderers (plan.md §28, §31, §32, §49, §62).

Deliberately free of git. Every test here builds a :class:`Delivery` by hand, because what
is under test is what the map does with a record — not what the engine records. The
end-to-end path, from a real repository to a rendered page and an exported file, is in
`test_ui.py` beside the other tests that need a repository.

Three properties get the most attention, because they are the ones a plausible-looking map
could quietly get wrong:

**Every edge is backed by a record.** A map is persuasive, and a line drawn between two
files because they look related is exactly the assertion this product exists to replace.
The tests check that an edge carries the evidence of the hop it draws, and that a hop with
no record produces no edge.

**Colour is never the only indicator.** plan.md §31 requires a label beside every colour.
The test asserts that each state appears as a word in the output, so a reader who cannot
see the difference between the red and the green still reads the same map.

**The output is deterministic.** Excalidraw's format wants a random seed and a timestamp,
and a layout that wandered would make the map impossible to test or to diff. Both are
derived from the data instead, and the tests assert that two renders are identical.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from itertools import pairwise
from typing import Any

from traceflow.ui.delivery import (
    Delivery,
    DeliveryFile,
    EdgeRef,
    EvidenceLine,
    ImpactNode,
    ReachLine,
    Totals,
)
from traceflow.ui.delivery import SymbolLine as DeliverySymbolLine
from traceflow.visualization import excalidraw, serializer
from traceflow.visualization.graph import (
    MAX_MAP_NODES,
    STATE_GLYPHS,
    STATE_LABELS,
    EdgeKind,
    GraphEdge,
    GraphNode,
    GraphView,
    NodeState,
    before_after,
    build_evidence_graph,
    change_map,
)
from traceflow.visualization.layout import NODE_HEIGHT, layout, rank
from traceflow.visualization.svg import render_svg

SESSION = "2026-01-01T00-00-00-aaaaaa"


# --------------------------------------------------------------------------- builders


def _evidence(
    path: str, line: int | None, detail: str, kind: str = "call_expression"
) -> EvidenceLine:
    return EvidenceLine(kind=kind, path=path, line=line, detail=detail)


def _delivery(
    *,
    files: tuple[DeliveryFile, ...] = (),
    nodes: tuple[ImpactNode, ...] = (),
    modules_added: tuple[str, ...] = (),
    modules_removed: tuple[str, ...] = (),
    edges_added: tuple[EdgeRef, ...] = (),
    edges_removed: tuple[EdgeRef, ...] = (),
    truncated: bool = False,
    max_depth: int = 0,
    external_modules: tuple[str, ...] = (),
    unresolved_imports: int = 0,
) -> Delivery:
    return Delivery(
        session_id=SESSION,
        repository="stub",
        root=".",
        started_at="2026-01-01T00:00:00",
        stabilized_at="2026-01-01T00:05:00",
        status="stabilized",
        baseline_commit=None,
        baseline_dirty=False,
        baseline_tracked_changes=0,
        baseline_untracked_files=0,
        baseline_captured=0,
        analyzer="python-1",
        totals=Totals(),
        files=files,
        nodes=nodes,
        modules_added=modules_added,
        modules_removed=modules_removed,
        edges_added=edges_added,
        edges_removed=edges_removed,
        truncated=truncated,
        max_depth=max_depth,
        external_modules=external_modules,
        unresolved_imports=unresolved_imports,
    )


def _file(
    path: str, *, status: str = "modified", symbols: tuple[tuple[str, str], ...] = ()
) -> DeliveryFile:
    return DeliveryFile(
        path=path,
        status=status,
        symbols=tuple(
            DeliverySymbolLine(qualified_name=name, kind="function", change=change)
            for name, change in symbols
        ),
    )


def _impact(
    path: str,
    *,
    symbol: str | None = None,
    distance: int = 1,
    reason: str = "signature_changed",
    chain: tuple[str, ...] | None = None,
    evidence: tuple[EvidenceLine, ...] = (),
    category: str = "indirect",
    reaches: tuple[ReachLine, ...] = (),
) -> ImpactNode:
    return ImpactNode(
        path=path,
        symbol=symbol,
        category=category,
        confidence="confirmed",
        distance=distance,
        reason=reason,
        chain=chain if chain is not None else (path,),
        evidence=evidence,
        reaches=reaches,
    )


def _reach(
    reason: str,
    chain: tuple[str, ...],
    *,
    category: str = "indirect",
    distance: int = 1,
    evidence: tuple[EvidenceLine, ...] = (),
) -> ReachLine:
    return ReachLine(
        reason=reason,
        category=category,
        distance=distance,
        chain=chain,
        confidence="confirmed",
        evidence=evidence,
    )


def _edge(source: str, target: str, module: str = "pkg.mod", line: int = 1) -> EdgeRef:
    return EdgeRef(source_path=source, target_path=target, module=module, line=line)


def _node(state: NodeState, path: str) -> GraphNode:
    return GraphNode(id=path, path=path, state=state)


def _graph_edge(source: str, target: str, kind: EdgeKind = EdgeKind.IMPORT_ADDED) -> GraphEdge:
    return GraphEdge(source=source, target=target, kind=kind)


# --------------------------------------------------------------------------- the model


def test_a_changed_file_becomes_a_modified_node_with_its_symbols() -> None:
    delivery = _delivery(
        files=(_file("auth/service.py", symbols=(("authenticate", "signature_changed"),)),)
    )

    graph = build_evidence_graph(delivery)

    assert [node.path for node in graph.nodes] == ["auth/service.py"]
    node = graph.nodes[0]
    assert node.state is NodeState.MODIFIED
    assert node.symbols == ("authenticate",)
    assert "declaration change" in node.detail


def test_an_added_and_a_deleted_file_get_their_own_states() -> None:
    delivery = _delivery(
        files=(_file("new.py", status="added"), _file("gone.py", status="deleted"))
    )

    states = {node.path: node.state for node in build_evidence_graph(delivery).nodes}

    assert states == {"new.py": NodeState.ADDED, "gone.py": NodeState.REMOVED}


def test_a_graph_diff_node_is_folded_in_even_without_a_file_record() -> None:
    """plan.md §28's added nodes come from the graph diff, and the map reflects that."""
    graph = build_evidence_graph(_delivery(modules_added=("middleware/rate_limit.py",)))

    assert graph.nodes[0].state is NodeState.ADDED


def test_a_chain_becomes_an_edge_carrying_the_hop_evidence() -> None:
    """The arrow from A to B must be the call that reaches back, not a resemblance."""
    delivery = _delivery(
        files=(_file("a.py", symbols=(("f", "signature_changed"),)),),
        nodes=(
            _impact(
                "b.py",
                symbol="g",
                chain=("a.py", "b.py"),
                evidence=(
                    _evidence("a.py", 1, "signature changed: f(x) -> f(x, y)"),
                    _evidence("b.py", 4, "f reaches a.py#f"),
                ),
            ),
        ),
    )

    graph = build_evidence_graph(delivery)

    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert (edge.source, edge.target, edge.kind) == ("a.py", "b.py", EdgeKind.AFFECTS)
    assert [item.path for item in edge.evidence] == ["b.py"], (
        "the edge must carry the record located in the file the arrow lands on"
    )


def test_a_hop_never_borrows_evidence_from_another_hop() -> None:
    """An arrow's evidence must be located in the file the arrow lands on, or be absent.

    The chain is itself a record — it is how the traversal says it took this step — so the
    edge is drawn. What it may not do is attach a neighbouring hop's record to make itself
    look sourced.
    """
    delivery = _delivery(
        files=(_file("a.py", symbols=(("f", "signature_changed"),)),),
        nodes=(
            _impact(
                "b.py",
                chain=("a.py", "b.py"),
                evidence=(_evidence("elsewhere.py", 3, "a record about another file"),),
            ),
        ),
    )

    graph = build_evidence_graph(delivery)

    assert len(graph.edges) == 1
    assert graph.edges[0].evidence == ()


def test_a_file_reaching_itself_is_counted_not_drawn() -> None:
    """A box pointing at itself says nothing. The fact is kept on the node instead."""
    delivery = _delivery(
        files=(_file("a.py", symbols=(("f", "signature_changed"), ("g", "body_changed"))),),
        nodes=(
            _impact(
                "a.py",
                symbol="g",
                chain=("a.py", "a.py"),
                evidence=(_evidence("a.py", 9, "f reaches a.py#f"),),
            ),
        ),
    )

    graph = build_evidence_graph(delivery)

    assert graph.edges == (), "a self-loop must not be drawn"
    node = next(item for item in graph.nodes if item.path == "a.py")
    assert "1 affected inside this file" in node.detail


def test_a_removed_module_still_imported_is_on_the_map_and_named_as_removed() -> None:
    delivery = _delivery(
        files=(_file("auth/legacy.py", status="deleted"),),
        modules_removed=("auth/legacy.py",),
        nodes=(
            _impact(
                "reports/build.py",
                reason="dangling_import",
                chain=("auth/legacy.py", "reports/build.py"),
                evidence=(
                    _evidence(
                        "reports/build.py",
                        1,
                        "imports auth.legacy, which this session removed",
                        kind="import_statement",
                    ),
                ),
            ),
        ),
    )

    graph = build_evidence_graph(delivery)
    states = {node.path: node.state for node in graph.nodes}

    assert states["auth/legacy.py"] is NodeState.REMOVED
    assert states["reports/build.py"] is NodeState.AFFECTED
    assert any(
        edge.source == "auth/legacy.py" and edge.target == "reports/build.py"
        for edge in graph.edges
    )


def test_an_import_change_becomes_an_edge_in_the_import_direction() -> None:
    delivery = _delivery(
        files=(_file("auth/routes.py", symbols=(("login", "body_changed"),)),),
        modules_added=("middleware/rate_limit.py",),
        edges_added=(
            _edge("auth/routes.py", "middleware/rate_limit.py", "middleware.rate_limit", 2),
        ),
    )

    graph = build_evidence_graph(delivery)
    edge = next(item for item in graph.edges if item.kind is EdgeKind.IMPORT_ADDED)

    assert (edge.source, edge.target) == ("auth/routes.py", "middleware/rate_limit.py")
    assert "middleware.rate_limit" in edge.evidence[0].detail
    assert edge.evidence[0].line == 2


def test_a_removed_import_is_a_distinct_kind() -> None:
    delivery = _delivery(
        files=(_file("a.py", symbols=(("f", "body_changed"),)),),
        edges_removed=(_edge("a.py", "old.py", "old", 3),),
    )

    graph = build_evidence_graph(delivery)
    kinds = {edge.kind for edge in graph.edges}

    assert EdgeKind.IMPORT_REMOVED in kinds
    removed = next(edge for edge in graph.edges if edge.kind is EdgeKind.IMPORT_REMOVED)
    assert removed.label == "no longer imports"


def test_the_far_end_of_a_new_import_is_context_not_modified() -> None:
    """Something now imports a file does not make that file changed."""
    delivery = _delivery(
        files=(_file("a.py", symbols=(("f", "body_changed"),)),),
        edges_added=(_edge("a.py", "untouched.py"),),
    )

    states = {node.path: node.state for node in build_evidence_graph(delivery).nodes}

    assert states["untouched.py"] is NodeState.CONTEXT


def test_a_changed_file_that_was_also_reached_stays_modified() -> None:
    """Precedence: what a file did outranks what was done to it."""
    delivery = _delivery(
        files=(_file("a.py", symbols=(("f", "body_changed"),)),),
        nodes=(_impact("a.py", symbol="f", chain=("b.py", "a.py")),),
    )

    graph = build_evidence_graph(delivery)
    node = next(item for item in graph.nodes if item.path == "a.py")

    assert node.state is NodeState.MODIFIED


def test_a_changed_file_links_to_its_diff_and_a_reached_one_to_the_evidence() -> None:
    delivery = _delivery(
        files=(_file("a.py", symbols=(("f", "signature_changed"),)),),
        nodes=(_impact("b.py", chain=("a.py", "b.py"), evidence=(_evidence("b.py", 1, "x"),)),),
    )

    links = {node.path: node.link for node in build_evidence_graph(delivery).nodes}

    assert links["a.py"] == f"/session/{SESSION}/diff/a.py"
    assert links["b.py"] == f"/session/{SESSION}/evidence"


def test_a_truncated_walk_and_unresolved_imports_are_stated_on_the_graph() -> None:
    """A picture invites the reader to assume it is complete. Say where it is not."""
    graph = build_evidence_graph(
        _delivery(
            files=(_file("a.py", symbols=(("f", "body_changed"),)),),
            truncated=True,
            max_depth=2,
            unresolved_imports=3,
            external_modules=("requests",),
        )
    )

    joined = " ".join(graph.notes)
    assert "stopped at depth 2" in joined
    assert "3 import(s)" in joined
    assert "requests" in joined


def test_a_delivery_with_nothing_in_it_produces_an_empty_graph() -> None:
    graph = build_evidence_graph(_delivery())

    assert graph.nodes == ()
    assert graph.edges == ()


# --------------------------------------------------------------------------- the views


def test_the_change_map_keeps_only_impact_edges() -> None:
    delivery = _delivery(
        files=(_file("a.py", symbols=(("f", "signature_changed"),)),),
        nodes=(_impact("b.py", chain=("a.py", "b.py"), evidence=(_evidence("b.py", 1, "call"),)),),
        edges_added=(_edge("a.py", "c.py"),),
    )

    view = change_map(build_evidence_graph(delivery))

    assert {edge.kind for edge in view.edges} == {EdgeKind.AFFECTS}
    assert {node.path for node in view.nodes} == {"a.py", "b.py"}


def test_the_before_after_view_keeps_only_relationship_changes() -> None:
    delivery = _delivery(
        files=(_file("a.py", symbols=(("f", "signature_changed"),)),),
        nodes=(_impact("b.py", chain=("a.py", "b.py"), evidence=(_evidence("b.py", 1, "call"),)),),
        edges_added=(_edge("a.py", "c.py"),),
    )

    view = before_after(build_evidence_graph(delivery))

    assert {edge.kind for edge in view.edges} == {EdgeKind.IMPORT_ADDED}
    assert {node.path for node in view.nodes} == {"a.py", "c.py"}


def test_the_before_after_view_says_so_when_nothing_structural_changed() -> None:
    delivery = _delivery(files=(_file("a.py", symbols=(("f", "body_changed"),)),))

    view = before_after(build_evidence_graph(delivery))

    assert view.nodes == ()
    assert "No import relationship changed" in view.empty


def test_a_changed_file_with_no_impact_still_appears_on_the_change_map() -> None:
    """A session that changed one local function still has a change to show."""
    delivery = _delivery(files=(_file("a.py", symbols=(("f", "body_changed"),)),))

    view = change_map(build_evidence_graph(delivery))

    assert [node.path for node in view.nodes] == ["a.py"]
    assert view.edges == ()


def test_the_map_stops_at_its_limit_and_counts_what_it_left_out() -> None:
    """plan.md §49: focused, not exhaustive — and §69: the omission is stated."""
    files = tuple(
        _file(f"m{index:03d}.py", symbols=(("f", "body_changed"),))
        for index in range(MAX_MAP_NODES + 5)
    )

    view = change_map(build_evidence_graph(_delivery(files=files)))

    assert len(view.nodes) == MAX_MAP_NODES
    assert view.omitted == 5


def test_the_limit_drops_the_furthest_node_not_the_change_itself() -> None:
    """A map showing a distant consequence but not its cause would be worse than smaller."""
    files = (_file("origin.py", symbols=(("f", "signature_changed"),)),)
    nodes = tuple(
        _impact(
            f"far{index:03d}.py",
            distance=3,
            reason="indirect_dependency",
            chain=("origin.py", f"far{index:03d}.py"),
            evidence=(_evidence(f"far{index:03d}.py", 1, "call"),),
        )
        for index in range(MAX_MAP_NODES)
    )

    view = change_map(build_evidence_graph(_delivery(files=files, nodes=nodes)))

    assert view.omitted == 1
    assert "origin.py" in {node.path for node in view.nodes}


def test_a_capped_view_draws_no_edge_to_a_node_it_dropped() -> None:
    files = (_file("origin.py", symbols=(("f", "signature_changed"),)),)
    nodes = tuple(
        _impact(
            f"far{index:03d}.py",
            distance=2,
            reason="indirect_dependency",
            chain=("origin.py", f"far{index:03d}.py"),
            evidence=(_evidence(f"far{index:03d}.py", 1, "call"),),
        )
        for index in range(MAX_MAP_NODES + 4)
    )

    view = change_map(build_evidence_graph(_delivery(files=files, nodes=nodes)))

    assert view.omitted == 5
    assert all(
        edge.source in view.node_ids and edge.target in view.node_ids for edge in view.edges
    ), "an edge may not point at a node that is not drawn"


def test_every_view_states_what_an_arrow_means() -> None:
    """Two arrow meanings on one picture would be a puzzle, so each view names its own."""
    delivery = _delivery(
        files=(_file("a.py", symbols=(("f", "signature_changed"),)),),
        edges_added=(_edge("a.py", "c.py"),),
    )
    graph = build_evidence_graph(delivery)

    assert "reaches" in change_map(graph).arrow
    assert "imports" in before_after(graph).arrow


# --------------------------------------------------------------------------- the layout


def test_rank_puts_a_chain_in_one_column_per_ring() -> None:
    nodes = tuple(_node(NodeState.MODIFIED, name) for name in ("a.py", "b.py", "c.py"))
    edges = (_graph_edge("a.py", "b.py"), _graph_edge("b.py", "c.py"))

    assert rank(nodes, edges) == {"a.py": 0, "b.py": 1, "c.py": 2}


def test_a_mutual_import_is_ranked_and_drawn_rather_than_dropped() -> None:
    """A cycle is a real shape in Python. It must not crash the layout or vanish."""
    nodes = (_node(NodeState.MODIFIED, "a.py"), _node(NodeState.MODIFIED, "b.py"))
    edges = (_graph_edge("a.py", "b.py"), _graph_edge("b.py", "a.py"))

    drawing = layout(nodes, edges)

    assert len(drawing.nodes) == 2
    assert len(drawing.edges) == 2, "neither relationship may be dropped"
    assert len({rank(nodes, edges)[item.node.id] for item in drawing.nodes}) == 2
    below = [item for item in drawing.edges if item.points[0][0] > item.points[-1][0]]
    assert below, "the backwards edge should have been routed, not drawn through the boxes"
    deepest = max(point[1] for point in below[0].points)
    assert deepest > NODE_HEIGHT + 10, "the backwards edge must travel clear of the boxes"


def test_the_layout_is_a_function_of_its_input() -> None:
    nodes = tuple(_node(NodeState.MODIFIED, name) for name in ("b.py", "a.py", "c.py"))
    edges = (_graph_edge("a.py", "c.py"), _graph_edge("b.py", "c.py"))

    first = layout(nodes, edges)
    second = layout(nodes, edges)

    assert [(item.node.id, item.x, item.y) for item in first.nodes] == [
        (item.node.id, item.x, item.y) for item in second.nodes
    ]


def test_boxes_in_one_column_do_not_overlap() -> None:
    nodes = tuple(_node(NodeState.MODIFIED, f"m{index}.py") for index in range(4))
    drawing = layout(nodes, ())

    ordered = sorted(drawing.nodes, key=lambda item: item.y)
    for earlier, later in pairwise(ordered):
        assert later.y >= earlier.y + earlier.height


def test_an_edge_to_a_node_outside_the_drawing_is_skipped() -> None:
    """Defensive: a view must never hand the layout an edge it cannot place."""
    drawing = layout((_node(NodeState.MODIFIED, "a.py"),), (_graph_edge("a.py", "ghost.py"),))

    assert drawing.edges == ()


# --------------------------------------------------------------------------- the svg


def _drawn() -> GraphView:
    delivery = _delivery(
        files=(
            _file("auth/service.py", symbols=(("authenticate", "signature_changed"),)),
            _file("new.py", status="added"),
            _file("gone.py", status="deleted"),
        ),
        nodes=(
            _impact(
                "auth/routes.py",
                symbol="login",
                chain=("auth/service.py", "auth/routes.py"),
                evidence=(_evidence("auth/routes.py", 5, "authenticate reaches service"),),
            ),
        ),
    )
    return change_map(build_evidence_graph(delivery))


def test_the_svg_is_a_well_formed_document() -> None:
    svg = render_svg(_drawn())

    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert root.get("viewBox")


def test_colour_is_never_the_only_indicator() -> None:
    """plan.md §31. Every state on the map is also a word and a glyph."""
    view = _drawn()
    svg = render_svg(view)

    for state in {node.state for node in view.nodes}:
        assert STATE_LABELS[state].upper() in svg, STATE_LABELS[state]
        assert STATE_GLYPHS[state] in svg


def test_every_node_is_a_link_to_where_it_can_be_checked() -> None:
    view = _drawn()
    svg = render_svg(view)

    assert svg.count("<a href=") == len(view.nodes)
    for node in view.nodes:
        assert node.link
        assert node.link in svg


def test_a_node_carries_its_evidence_as_a_tooltip() -> None:
    svg = render_svg(_drawn())

    assert "authenticate reaches service" in svg


def test_markup_in_a_path_cannot_escape_into_the_page() -> None:
    """The content is the user's own source tree. Trusted in origin, not in form."""
    delivery = _delivery(
        files=(_file('<img src=x onerror="alert(1)">.py', symbols=(("f", "body_changed"),)),)
    )

    svg = render_svg(change_map(build_evidence_graph(delivery)))

    assert "<img" not in svg
    assert "&lt;img" in svg
    assert ET.fromstring(svg).tag.endswith("svg")


def test_the_svg_contains_no_script() -> None:
    assert "<script" not in render_svg(_drawn())


def test_the_svg_names_the_map_for_a_screen_reader() -> None:
    svg = render_svg(_drawn())

    assert 'role="img"' in svg
    assert "Change map" in svg
    assert "reaches" in svg


def test_an_empty_view_still_produces_a_document_that_explains_itself() -> None:
    """An export that writes a file which will not open is worse than no export."""
    view = before_after(build_evidence_graph(_delivery()))
    svg = render_svg(view)

    assert ET.fromstring(svg).tag.endswith("svg")
    assert "No import relationship changed" in svg


def test_the_svg_is_byte_identical_when_rendered_twice() -> None:
    view = _drawn()
    assert render_svg(view) == render_svg(view)


def test_two_maps_on_one_page_do_not_share_element_ids() -> None:
    """Marker ids are document-wide. Two maps with one id would pick each other's arrows."""
    delivery = _delivery(
        files=(_file("a.py", symbols=(("f", "signature_changed"),)),),
        nodes=(_impact("b.py", chain=("a.py", "b.py"), evidence=(_evidence("b.py", 1, "x"),)),),
        edges_added=(_edge("a.py", "c.py"),),
    )
    graph = build_evidence_graph(delivery)

    change = render_svg(change_map(graph))
    delta = render_svg(before_after(graph))

    change_ids = {part.split('"')[0] for part in change.split('id="')[1:]}
    delta_ids = {part.split('"')[0] for part in delta.split('id="')[1:]}
    assert not (change_ids & delta_ids), change_ids & delta_ids


# --------------------------------------------------------------------------- the json


def test_the_json_is_the_internal_representation_and_says_which() -> None:
    payload = serializer.to_dict(_drawn())

    assert payload["format"] == serializer.FORMAT
    assert payload["version"] == serializer.VERSION
    assert payload["view"] == "change"
    assert payload["session"] == SESSION
    assert payload["nodes"] and payload["edges"]


def test_the_json_round_trips_and_keeps_the_evidence() -> None:
    text = serializer.dumps(serializer.to_dict(_drawn()))

    payload = json.loads(text)
    node = next(item for item in payload["nodes"] if item["path"] == "auth/routes.py")

    assert node["state"] == "affected"
    assert node["symbols"] == ["login"]
    assert node["link"] == f"/session/{SESSION}/evidence"
    assert node["evidence"][0]["path"] == "auth/routes.py"
    assert payload["edges"][0]["kind"] == "affects"


def test_the_json_carries_the_limits_as_well_as_the_graph() -> None:
    """A consumer of the JSON needs the caveats as much as the dashboard does."""
    delivery = _delivery(
        files=(_file("a.py", symbols=(("f", "body_changed"),)),),
        truncated=True,
        max_depth=1,
    )

    payload = serializer.to_dict(change_map(build_evidence_graph(delivery)))

    assert payload["omitted"] == 0
    assert any("depth 1" in note for note in payload["notes"])
    assert payload["arrow"]


def test_the_json_is_byte_identical_when_written_twice() -> None:
    view = _drawn()
    assert serializer.dumps(serializer.to_dict(view)) == serializer.dumps(serializer.to_dict(view))


# --------------------------------------------------------------------------- excalidraw

#: Every field Excalidraw's own element type declares. A file that leans on the reader
#: filling in a missing field only opens in the version that happens to tolerate it.
_REQUIRED_ELEMENT_FIELDS = (
    "id",
    "type",
    "x",
    "y",
    "width",
    "height",
    "angle",
    "strokeColor",
    "backgroundColor",
    "fillStyle",
    "strokeWidth",
    "strokeStyle",
    "roughness",
    "opacity",
    "groupIds",
    "frameId",
    "roundness",
    "seed",
    "version",
    "versionNonce",
    "index",
    "isDeleted",
    "boundElements",
    "updated",
    "link",
    "locked",
)


def _excalidraw() -> dict[str, Any]:
    return excalidraw.to_file(_drawn())


def test_the_excalidraw_file_has_the_wrapper_excalidraw_expects() -> None:
    payload = _excalidraw()

    assert payload["type"] == "excalidraw"
    assert payload["version"] == excalidraw.FILE_VERSION
    assert isinstance(payload["elements"], list)
    assert payload["files"] == {}
    assert isinstance(payload["appState"], dict)


def test_every_excalidraw_element_declares_the_fields_the_format_requires() -> None:
    for element in _excalidraw()["elements"]:
        missing = [field for field in _REQUIRED_ELEMENT_FIELDS if field not in element]
        assert not missing, f"{element['type']} {element['id']} is missing {missing}"


def test_every_excalidraw_element_has_a_unique_id() -> None:
    elements = _excalidraw()["elements"]
    ids = [element["id"] for element in elements]

    assert len(ids) == len(set(ids))


def test_the_excalidraw_boxes_and_arrows_are_both_present() -> None:
    elements = _excalidraw()["elements"]
    kinds: dict[str, int] = {}
    for element in elements:
        kinds[element["type"]] = kinds.get(element["type"], 0) + 1

    view = _drawn()
    assert kinds["rectangle"] == len(view.nodes)
    assert kinds["arrow"] == len(view.edges)


def test_an_excalidraw_arrow_starts_at_its_own_origin_and_ends_with_an_arrowhead() -> None:
    """Excalidraw stores points relative to the element, so the first point must be 0,0."""
    arrows = [item for item in _excalidraw()["elements"] if item["type"] == "arrow"]

    assert arrows
    for arrow in arrows:
        assert arrow["points"][0] == [0, 0]
        assert arrow["endArrowhead"] == "arrow"
        assert arrow["startArrowhead"] is None
        assert arrow["elbowed"] is False
        assert arrow["startBinding"] is None
        assert arrow["endBinding"] is None


def test_a_removed_import_is_dashed_in_the_drawing() -> None:
    """Line style is a second signal beside the colour, as in the map."""
    delivery = _delivery(
        files=(_file("a.py", symbols=(("f", "body_changed"),)),),
        edges_removed=(_edge("a.py", "old.py", "old", 3),),
    )

    payload = excalidraw.to_file(before_after(build_evidence_graph(delivery)))
    arrows = [item for item in payload["elements"] if item["type"] == "arrow"]

    assert arrows and arrows[0]["strokeStyle"] == "dashed"


def test_the_excalidraw_text_carries_the_state_word_not_just_the_colour() -> None:
    """The drawing travels without the page around it, so it must explain itself."""
    payload = _excalidraw()
    texts = [item["text"] for item in payload["elements"] if item["type"] == "text"]

    assert any("MODIFIED" in text for text in texts)
    assert any("AFFECTED" in text for text in texts)
    assert any("ADDED" in text for text in texts)
    assert any("REMOVED" in text for text in texts)


def test_the_excalidraw_header_repeats_the_caption_and_the_arrow() -> None:
    header = _excalidraw()["elements"][0]

    assert header["type"] == "text"
    assert "Change map" in header["text"]
    assert "reaches" in header["text"]
    assert "States on this map" in header["text"]


def test_the_excalidraw_export_is_reproducible() -> None:
    """A random seed would make every export of one session a different file."""
    view = _drawn()
    assert excalidraw.dumps(view) == excalidraw.dumps(view)


def test_the_excalidraw_export_is_valid_json() -> None:
    payload = json.loads(excalidraw.dumps(_drawn()))

    assert payload["type"] == "excalidraw"
    assert payload["elements"]


def test_a_reach_draws_the_arrow_it_recorded() -> None:
    """A changed file that also calls a moved declaration has two chains, and both are edges.

    Drawing only the node's own chain is how the arrow between a changed file and the
    declaration it calls went missing from the map.
    """
    delivery = _delivery(
        files=(_file("auth/routes.py", symbols=(("login", "body_changed"),)),),
        nodes=(
            _impact(
                "auth/routes.py",
                symbol="login",
                distance=0,
                reason="symbol_body_changed",
                chain=("auth/routes.py",),
                reaches=(
                    _reach(
                        "signature_changed",
                        ("auth/service.py", "auth/routes.py"),
                        evidence=(_evidence("auth/routes.py", 5, "authenticate reaches service"),),
                    ),
                ),
            ),
        ),
    )

    graph = build_evidence_graph(delivery)
    edges = {(edge.source, edge.target, edge.kind) for edge in graph.edges}

    assert edges == {("auth/service.py", "auth/routes.py", EdgeKind.AFFECTS)}
    assert graph.edges[0].evidence, "the arrow must carry the record of the hop it draws"


def test_a_changed_file_that_was_also_reached_stays_modified_and_names_the_obligation() -> None:
    """Its own change outranks what was done to it, and the obligation is still stated."""
    delivery = _delivery(
        files=(_file("auth/routes.py", symbols=(("login", "body_changed"),)),),
        nodes=(
            _impact(
                "auth/routes.py",
                symbol="login",
                distance=0,
                reason="symbol_body_changed",
                chain=("auth/routes.py",),
                reaches=(_reach("signature_changed", ("auth/service.py", "auth/routes.py")),),
            ),
        ),
    )

    node = next(
        item for item in build_evidence_graph(delivery).nodes if item.path == "auth/routes.py"
    )

    assert node.state is NodeState.MODIFIED
    assert "calls a changed declaration" in node.detail, "the obligation leads"
    assert "body change" in node.detail, "its own change is still named"


def test_a_reach_is_counted_on_the_node_not_as_a_second_node() -> None:
    """One symbol is one box, however many ways it was found."""
    delivery = _delivery(
        files=(_file("auth/routes.py", symbols=(("login", "body_changed"),)),),
        nodes=(
            _impact(
                "auth/routes.py",
                symbol="login",
                distance=0,
                reason="symbol_body_changed",
                chain=("auth/routes.py",),
                reaches=(_reach("signature_changed", ("auth/service.py", "auth/routes.py")),),
            ),
        ),
    )

    view = change_map(build_evidence_graph(delivery))

    assert sorted(node.path for node in view.nodes) == ["auth/routes.py", "auth/service.py"]
    assert len(view.nodes) == 2


def test_a_ten_thousand_node_graph_is_capped_and_says_so() -> None:
    """plan.md §68's 10,000-file scale, at the renderer: it must not draw everything.

    plan.md §49 is explicit that TraceFlow "must not attempt 50,000 files -> giant graph ->
    render everything". The cap is what makes that true on the drawing side, and the count
    it reports is what keeps it honest.
    """
    files = tuple(
        _file(f"mod_{index:05d}.py", symbols=(("f", "body_changed"),)) for index in range(10_000)
    )

    view = change_map(build_evidence_graph(_delivery(files=files)))

    assert len(view.nodes) == MAX_MAP_NODES
    assert view.omitted == 10_000 - MAX_MAP_NODES
    assert len(render_svg(view)) < 200_000, "the drawing must not grow with the repository"


def test_a_reason_already_on_the_box_is_not_named_twice() -> None:
    """A symbol the walk reached before anything else did carries the reason as its own record.

    Its reach says the same thing by the same words, so the box must read
    "calls a changed declaration · depth 1" — not the phrase twice, which is what happens when
    the node's own summary and its reaches are both written out.
    """
    delivery = _delivery(
        nodes=(
            _impact(
                "z.py",
                symbol="helper",
                distance=1,
                reason="signature_changed",
                chain=("x.py", "z.py"),
                reaches=(
                    _reach(
                        "signature_changed",
                        ("y.py", "z.py"),
                        evidence=(_evidence("z.py", 4, "two reaches z.py#two"),),
                    ),
                ),
            ),
        ),
    )

    node = next(item for item in build_evidence_graph(delivery).nodes if item.path == "z.py")

    assert node.detail.count("calls a changed declaration") == 1, node.detail
    assert "depth 1" in node.detail
    assert len(node.reached) == 1
