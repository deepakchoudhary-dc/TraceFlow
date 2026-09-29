"""Where the boxes go (plan.md §31, §49).

Layered, left to right: one column per ring, so the picture reads in the order the
analysis reasoned — the session's own change first, then the consequences of it. plan.md
§49 asks for a *focused* visualization, and the focus is the traversal's, not the layout's;
this module only decides how to arrange what the traversal already chose.

**No dependencies, and no randomness.** A layout that moved between runs would make the
map impossible to test and impossible to diff, and a diagram that shifts under the reader
is worse than a plain one. Every position is a function of the node's rank and its index
in the column, both of which are computed from sorted keys. Two runs on one session
produce identical coordinates.

**Every edge goes left to right.** Both views are built so that an edge always crosses at
least one column: the impact map's edges each move one ring out, and the before/after map's
edges run from a file the session changed to a module it imports. The one exception is a
mutual import the session touched on both sides — a real, if rare, shape — and those are
routed below the boxes rather than drawn backwards through them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from traceflow.visualization.graph import STATE_ORDER, GraphEdge, GraphNode

#: Box geometry. Wide enough for a repository-relative path and two short clauses, which
#: is what a reader needs before deciding whether to follow the link.
NODE_WIDTH = 236.0
NODE_HEIGHT = 64.0

#: Room between columns and between boxes in a column.
COLUMN_GAP = 72.0
ROW_GAP = 16.0

#: How far a backward edge dips below the boxes, and how far an arrow's curve bows.
_DIP = 30.0
_BOW = 28.0


@dataclass(frozen=True)
class PlacedNode:
    node: GraphNode
    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def middle(self) -> float:
        return self.y + self.height / 2


@dataclass(frozen=True)
class RoutedEdge:
    edge: GraphEdge
    points: tuple[tuple[float, float], ...]
    label_at: tuple[float, float]


@dataclass(frozen=True)
class Layout:
    """Boxes with coordinates, and edges with a path through them."""

    nodes: tuple[PlacedNode, ...]
    edges: tuple[RoutedEdge, ...]
    width: float
    height: float

    def placed(self, node_id: str) -> PlacedNode | None:
        for item in self.nodes:
            if item.node.id == node_id:
                return item
        return None


def rank(nodes: tuple[GraphNode, ...], edges: tuple[GraphEdge, ...]) -> dict[str, int]:
    """A column per node: the longest path from any node that nothing points at.

    Longest path rather than shortest, so that an edge never skips a column and every
    arrow visibly crosses the gap it claims to cross. Ties are broken by sorted order, so
    the answer does not depend on how the edges happened to be stored.
    """
    ids = [node.id for node in nodes]
    known = set(ids)
    predecessors: dict[str, set[str]] = {node_id: set() for node_id in ids}
    successors: dict[str, set[str]] = {node_id: set() for node_id in ids}

    for edge in edges:
        if edge.source == edge.target or edge.source not in known or edge.target not in known:
            continue
        predecessors[edge.target].add(edge.source)
        successors[edge.source].add(edge.target)

    ranks: dict[str, int] = {}
    remaining = {node_id: len(predecessors[node_id]) for node_id in ids}
    ready = deque(sorted(node_id for node_id in ids if remaining[node_id] == 0))

    while ready:
        node_id = ready.popleft()
        ranks[node_id] = max(
            (ranks[item] + 1 for item in predecessors[node_id] if item in ranks), default=0
        )
        for successor in sorted(successors[node_id]):
            remaining[successor] -= 1
            if remaining[successor] == 0:
                ready.append(successor)

    # Anything left is inside a cycle, which a mutual import makes possible. The cycle is
    # not an error and it is not hidden: each remaining node is placed after the deepest
    # predecessor that did get placed, so the column it lands in is still the honest one.
    for node_id in ids:
        if node_id in ranks:
            continue
        placed = [ranks[item] for item in predecessors[node_id] if item in ranks]
        ranks[node_id] = max(placed) + 1 if placed else max(ranks.values(), default=-1) + 1

    return ranks


def _route(
    source: PlacedNode, target: PlacedNode
) -> tuple[tuple[tuple[float, float], ...], tuple[float, float]]:
    """A path from *source* to *target*, and where to put its label.

    Four points either way, so the renderers have one shape to handle. Forward edges bow
    between the columns they cross; an edge that would have to travel backwards goes below
    the boxes instead, which keeps it visible without drawing it through them.
    """
    start = (source.right, source.middle)
    end = (target.x, target.middle)
    if target.x > source.right + 4.0:
        bow = max(_BOW, (target.x - source.right) / 2)
        points = (start, (start[0] + bow, start[1]), (end[0] - bow, end[1]), end)
        return points, ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)

    bottom = max(source.y + source.height, target.y + target.height) + _DIP
    from_bottom = (source.x + source.width / 2, source.y + source.height)
    to_bottom = (target.x + target.width / 2, target.y + target.height)
    points = (from_bottom, (from_bottom[0], bottom), (to_bottom[0], bottom), to_bottom)
    return points, ((from_bottom[0] + to_bottom[0]) / 2, bottom)


def layout(nodes: tuple[GraphNode, ...], edges: tuple[GraphEdge, ...]) -> Layout:
    """Arrange *nodes* and *edges* deterministically."""
    ranks = rank(nodes, edges)
    columns: dict[int, list[GraphNode]] = {}
    for node in nodes:
        columns.setdefault(ranks[node.id], []).append(node)
    for column in columns.values():
        column.sort(key=lambda item: (STATE_ORDER[item.state], item.path))

    tallest = max(
        (len(column) * NODE_HEIGHT + (len(column) - 1) * ROW_GAP for column in columns.values()),
        default=0.0,
    )

    placed: dict[str, PlacedNode] = {}
    for index in sorted(columns):
        column = columns[index]
        stack = len(column) * NODE_HEIGHT + (len(column) - 1) * ROW_GAP
        top = (tallest - stack) / 2
        for offset, node in enumerate(column):
            placed[node.id] = PlacedNode(
                node=node,
                x=index * (NODE_WIDTH + COLUMN_GAP),
                y=top + offset * (NODE_HEIGHT + ROW_GAP),
                width=NODE_WIDTH,
                height=NODE_HEIGHT,
            )

    routed: list[RoutedEdge] = []
    for edge in edges:
        source = placed.get(edge.source)
        target = placed.get(edge.target)
        if source is None or target is None:
            continue
        points, label_at = _route(source, target)
        routed.append(RoutedEdge(edge=edge, points=points, label_at=label_at))

    width = max((item.right for item in placed.values()), default=0.0)
    height = max(
        [tallest, *(point[1] for item in routed for point in item.points)],
        default=0.0,
    )
    return Layout(
        nodes=tuple(placed[node.id] for node in nodes if node.id in placed),
        edges=tuple(routed),
        width=width,
        height=height,
    )
