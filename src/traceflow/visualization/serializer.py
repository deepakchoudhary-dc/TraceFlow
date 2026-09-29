"""The evidence graph as JSON (plan.md §32, §35, §43).

plan.md §32 names JSON as one of the graph's external renderers, and this is it: the same
nodes and edges the dashboard draws, written out so another tool can read them without
scraping a picture. The shape follows plan.md §43's example — a node has an id, a path and
a type, an edge names a source, a target and a confidence — extended with the two things
§43's sketch does not carry and §33 requires: the state a node is in, and the evidence for
every edge.

A view is what is serialized, not the whole graph, because a view is what the reader asked
for. Exporting the change map gives the change map; exporting the before/after comparison
gives that. The union of the two is the session's whole evidence graph.
"""

from __future__ import annotations

import json
from typing import Any

from traceflow.ui.delivery import EvidenceLine
from traceflow.visualization.graph import EvidenceGraph, GraphEdge, GraphNode, GraphView

#: Stamped on every export so a consumer can tell what it is holding and whether the shape
#: has moved under it. plan.md §35 makes the artifact the interface; an interface without a
#: version is a trap.
FORMAT = "traceflow.evidence-graph"
VERSION = 1


def _evidence(items: tuple[EvidenceLine, ...]) -> list[dict[str, Any]]:
    return [
        {"kind": item.kind, "path": item.path, "line": item.line, "detail": item.detail}
        for item in items
    ]


def _node(node: GraphNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "path": node.path,
        "state": node.state.value,
        "distance": node.distance,
        "detail": node.detail,
        "symbols": list(node.symbols),
        "link": node.link,
        "evidence": _evidence(node.evidence),
    }


def _edge(edge: GraphEdge) -> dict[str, Any]:
    return {
        "source": edge.source,
        "target": edge.target,
        "kind": edge.kind.value,
        "label": edge.label,
        "evidence": _evidence(edge.evidence),
    }


def graph_to_dict(graph: EvidenceGraph) -> dict[str, Any]:
    """The whole session's evidence graph, unfiltered and uncapped."""
    return {
        "format": FORMAT,
        "version": VERSION,
        "session": graph.session_id,
        "repository": graph.repository,
        "nodes": [_node(node) for node in graph.nodes],
        "edges": [_edge(edge) for edge in graph.edges],
        "notes": list(graph.notes),
    }


def to_dict(view: GraphView) -> dict[str, Any]:
    """One view: the subgraph the reader asked for, and how to read it."""
    return {
        "format": FORMAT,
        "version": VERSION,
        "view": view.key,
        "title": view.title,
        "session": view.session_id,
        "repository": view.repository,
        "caption": view.caption,
        "arrow": view.arrow,
        "omitted": view.omitted,
        "notes": list(view.notes),
        "nodes": [_node(node) for node in view.nodes],
        "edges": [_edge(edge) for edge in view.edges],
    }


def dumps(payload: dict[str, Any]) -> str:
    """Serialized the one way, so two exports of one session are identical byte for byte.

    Keys keep their declared order rather than being sorted: the order is the reading
    order, and a diff between two sessions should show the change rather than a reshuffle.
    """
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
