"""The map as an Excalidraw drawing (plan.md §31, §32, §62).

plan.md §32 is careful about this: Excalidraw is an export target, not the core
representation. The graph is built and rendered by TraceFlow; this module hands it to a
tool the developer can annotate, which is what an exported diagram is for. Nothing here
reads the graph for meaning — it places what the layout already placed.

**The export is reproducible.** Excalidraw's format carries a `seed` and a `versionNonce`
that are random in the app, and a `updated` timestamp. Both are derived here from the
element's own identity instead, so exporting one session twice produces the same bytes.
That is what makes the export testable at all, and it means a drawing committed beside a
session does not churn on every run.

**One arrow per relationship, drawn straight.** The dashboard bows its arrows between the
columns because a picture can afford the curve. A drawing the reader will move around
cannot: the boxes are theirs to drag, and a curve pinned to the old coordinates would be
wrong the moment they do. So an exported arrow runs from the centre of one box to the
centre of the other, and says the same thing the curve did.

**The drawing explains itself.** An Excalidraw file has no page around it, so the caption,
the meaning of an arrow and the limits of the analysis travel with it as a text element.
A diagram that arrives without its caveats is how a limitation becomes a false claim.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from traceflow.visualization.graph import (
    EDGE_TONES,
    INKS,
    STATE_GLYPHS,
    STATE_LABELS,
    STATE_TONES,
    TONES,
    GraphView,
)
from traceflow.visualization.layout import Layout, PlacedNode, layout

#: The file wrapper Excalidraw writes and reads.
FILE_VERSION = 2
SOURCE = "https://excalidraw.com"

#: Element versions start at 1, as a newly created element does.
ELEMENT_VERSION = 1

#: Monospace (Cascadia), because almost everything on this map is a file path.
FONT_MONOSPACE = 3

_FONT_SIZE = 12
_LINE_HEIGHT = 1.25
_HEADER_WRAP = 92


def _element_id(kind: str, key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"tf-{kind}-{digest}"


def _seed(key: str) -> int:
    """A stable stand-in for Excalidraw's random seed.

    Derived from the element's identity rather than drawn from a random source, so the
    rough.js sketch Excalidraw generates for a box is the same sketch every time. A random
    seed would make every export of the same session a different file.
    """
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16)


def _base(
    key: str, kind: str, *, x: float, y: float, width: float, height: float
) -> dict[str, Any]:
    """The fields every Excalidraw element carries.

    Spelled out in full rather than left to defaults: Excalidraw tolerates a missing field
    by filling it in, but a file that only opens in the version that happens to tolerate
    it is not an export, it is a coincidence.
    """
    seed = _seed(key)
    return {
        "id": _element_id(kind, key),
        "type": kind,
        "x": round(x, 2),
        "y": round(y, 2),
        "width": round(width, 2),
        "height": round(height, 2),
        "angle": 0,
        "strokeColor": INKS["fg"],
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 1,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": seed,
        "version": ELEMENT_VERSION,
        "versionNonce": seed,
        "index": None,
        "isDeleted": False,
        "boundElements": None,
        "updated": 0,
        "created": None,
        "link": None,
        "locked": False,
    }


def _wrap(text: str, width: int = _HEADER_WRAP) -> list[str]:
    """Break on spaces, keeping words whole. A path longer than the line is left long
    rather than cut: half a path is worse than a wide line."""
    lines: list[str] = []
    current = ""
    for word in text.split():
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _header(view: GraphView, *, top: float) -> tuple[dict[str, Any], float]:
    lines = [f"{view.repository} — {view.title}", *_wrap(view.caption), view.arrow]
    for note in view.notes:
        lines.extend(_wrap(note))
    if view.omitted:
        lines.append(f"{view.omitted} further node(s) are not drawn.")
    states = ", ".join(sorted({STATE_LABELS[node.state] for node in view.nodes}))
    if states:
        lines.extend(["", f"States on this map: {states}."])

    text = "\n".join(lines)
    element = _base(f"header:{view.key}", "text", x=0, y=top, width=0, height=0)
    element.update(
        {
            "strokeColor": INKS["fg"],
            "fontSize": _FONT_SIZE,
            "fontFamily": FONT_MONOSPACE,
            "text": text,
            "originalText": text,
            "textAlign": "left",
            "verticalAlign": "top",
            "containerId": None,
            "autoResize": True,
            "lineHeight": _LINE_HEIGHT,
        }
    )
    return element, len(lines) * _FONT_SIZE * _LINE_HEIGHT


def _boxes(drawing: Layout, *, top: float) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    for item in drawing.nodes:
        node = item.node
        tone = STATE_TONES[node.state]
        strong, background = TONES[tone]

        box = _base(
            node.id,
            "rectangle",
            x=item.x,
            y=item.y + top,
            width=item.width,
            height=item.height,
        )
        box.update(
            {
                "strokeColor": strong,
                "backgroundColor": background,
                "strokeWidth": 2,
                "roundness": {"type": 3},
            }
        )
        elements.append(box)

        lines = [f"{STATE_GLYPHS[node.state]} {STATE_LABELS[node.state].upper()}", node.path]
        if node.detail:
            lines.append(node.detail)
        text = "\n".join(lines)
        label = _base(
            f"label:{node.id}", "text", x=item.x + 10, y=item.y + top + 8, width=0, height=0
        )
        label.update(
            {
                "strokeColor": strong,
                "fontSize": _FONT_SIZE,
                "fontFamily": FONT_MONOSPACE,
                "text": text,
                "originalText": text,
                "textAlign": "left",
                "verticalAlign": "top",
                "containerId": None,
                "autoResize": True,
                "lineHeight": _LINE_HEIGHT,
            }
        )
        elements.append(label)
    return elements


def _arrows(drawing: Layout) -> list[dict[str, Any]]:
    """One straight arrow per relationship, box centre to box centre."""
    where: dict[str, PlacedNode] = {item.node.id: item for item in drawing.nodes}
    elements: list[dict[str, Any]] = []
    for routed in drawing.edges:
        source = where.get(routed.edge.source)
        target = where.get(routed.edge.target)
        if source is None or target is None:
            continue
        start = (source.right, source.middle)
        end = (target.x, target.middle)
        tone = EDGE_TONES[routed.edge.kind]
        strong, _background = TONES[tone]

        arrow = _base(
            f"arrow:{routed.edge.source}->{routed.edge.target}:{routed.edge.kind.value}",
            "arrow",
            x=start[0],
            y=start[1],
            width=abs(end[0] - start[0]),
            height=abs(end[1] - start[1]),
        )
        arrow.update(
            {
                "strokeColor": strong,
                "strokeWidth": 2,
                "strokeStyle": "dashed" if routed.edge.kind.value == "import_removed" else "solid",
                "roundness": {"type": 2},
                "points": [[0, 0], [round(end[0] - start[0], 2), round(end[1] - start[1], 2)]],
                "startBinding": None,
                "endBinding": None,
                "startArrowhead": None,
                "endArrowhead": "arrow",
                "elbowed": False,
            }
        )
        elements.append(arrow)
    return elements


def to_file(view: GraphView) -> dict[str, Any]:
    """The view as an Excalidraw document."""
    drawing = layout(view.nodes, view.edges)
    header, header_height = _header(view, top=0.0)
    elements: list[dict[str, Any]] = [header]
    elements.extend(_boxes(drawing, top=header_height + 24.0))
    elements.extend(_arrows(drawing))
    return {
        "type": "excalidraw",
        "version": FILE_VERSION,
        "source": SOURCE,
        "elements": elements,
        "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
        "files": {},
    }


def dumps(view: GraphView) -> str:
    """The view as the text of a ``.excalidraw`` file."""
    return json.dumps(to_file(view), indent=2, ensure_ascii=False, sort_keys=False) + "\n"
