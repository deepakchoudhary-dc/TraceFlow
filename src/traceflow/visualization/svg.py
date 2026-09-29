"""The map, drawn as SVG (plan.md §31, §46, §71).

SVG rather than canvas or a JavaScript graph library, for the same reason the rest of the
delivery is server-rendered: there is no build step, nothing executes in the reader's
browser, and the drawing a test asserts on is the drawing the reader sees. It also means
the map works in a text-mode browser as its ``<title>`` and ``<desc>``, which the text
listings below it make readable in full.

**Colour is never the message.** plan.md §31 asks for labels as well as colours, so every
box carries the state as a word and a glyph, and the palette is only ever a second signal.
A reader who sees no colour at all still reads the same map.

**The stylesheet is generated from the palette** rather than written out beside it, so a
tone cannot be changed in one place and left stale in the other. The colours are the
dashboard's CSS variables with the light-theme value as a fallback: inside the dashboard
the variables win and the map follows the reader's light or dark preference, and exported
on its own it is still legible.
"""

from __future__ import annotations

from html import escape

from traceflow.visualization.graph import (
    EDGE_TONES,
    INKS,
    STATE_GLYPHS,
    STATE_LABELS,
    STATE_TONES,
    TONES,
    GraphNode,
    GraphView,
)
from traceflow.visualization.layout import Layout, PlacedNode, layout

#: Room around the drawing, applied through the viewBox so no coordinate has to be
#: shifted and the layout stays a pure function of the graph.
_PAD = 16.0

#: Boxes are fixed size, so a label that does not fit is clipped rather than allowed to
#: spill. The budgets come from the font sizes below and the box width; they are rounded
#: down, because a character too many reads as a character too many.
_PATH_CHARS = 30
_DETAIL_CHARS = 40

_PILL_HEIGHT = 16.0
_PILL_TOP = 9.0
_STATE_BASELINE = 21.0
_PATH_BASELINE = 43.0
_DETAIL_BASELINE = 58.0


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _esc(value: object) -> str:
    return escape("" if value is None else str(value), quote=True)


def _num(value: float) -> str:
    return f"{value:.1f}"


def _styles() -> str:
    """The map's rules, built from the palette so the two cannot drift apart."""
    rules = [
        ".tf-map { display: block; }",
        (
            '.tf-map text { font-family: ui-sans-serif, -apple-system, "Segoe UI", '
            "system-ui, sans-serif; }"
        ),
        ".tf-map a { text-decoration: none; }",
        (
            f".tf-map .tf-box {{ fill: var(--panel, {INKS['panel']}); "
            f"stroke: var(--line, {INKS['line']}); stroke-width: 1.2; }}"
        ),
        (
            f'.tf-map .tf-path {{ font-family: ui-monospace, "Cascadia Mono", Consolas, '
            f"monospace; font-size: 11.5px; fill: var(--fg, {INKS['fg']}); }}"
        ),
        f".tf-map .tf-detail {{ font-size: 10px; fill: var(--muted, {INKS['muted']}); }}",
        (".tf-map .tf-state { font-size: 9.5px; font-weight: 700; letter-spacing: .05em; }"),
        (
            f".tf-map .tf-edge-label {{ font-size: 9.5px; text-anchor: middle; "
            f"fill: var(--muted, {INKS['muted']}); stroke: var(--panel, {INKS['panel']}); "
            f"stroke-width: 3px; paint-order: stroke fill; }}"
        ),
        ".tf-map .tf-edge { fill: none; stroke-width: 1.4; }",
    ]
    for tone, (strong, background) in TONES.items():
        rules.append(
            f".tf-map .tf-box.{tone} {{ fill: var(--{tone}-bg, {background}); "
            f"stroke: var(--{tone}, {strong}); }}"
        )
        rules.append(
            f".tf-map .tf-pill.{tone} {{ fill: var(--{tone}-bg, {background}); "
            f"stroke: var(--{tone}, {strong}); stroke-width: 1; }}"
        )
        rules.append(f".tf-map .tf-state.{tone} {{ fill: var(--{tone}, {strong}); }}")
        rules.append(f".tf-map .tf-edge.{tone} {{ stroke: var(--{tone}, {strong}); }}")
        rules.append(f".tf-map .tf-arrow.{tone} {{ fill: var(--{tone}, {strong}); }}")
    rules.append(
        f".tf-map .tf-edge.removed {{ stroke-dasharray: 5 4; "
        f"stroke: var(--removed, {TONES['removed'][0]}); }}"
    )
    return "\n".join(rules)


def _markers(prefix: str) -> str:
    """One arrowhead per tone.

    Separate markers rather than one that inherits the line's colour: a marker cannot
    reliably read the stroke of the element referencing it, and an arrowhead in the wrong
    colour is exactly the kind of quiet wrongness plan.md §31's rules exist to prevent.
    """
    parts: list[str] = []
    for tone in TONES:
        parts.append(
            f'<marker id="{prefix}-arrow-{tone}" markerWidth="7" markerHeight="7" '
            f'refX="6" refY="3.5" orient="auto" markerUnits="userSpaceOnUse">'
            f'<path class="tf-arrow {tone}" d="M0,0.6 L6.4,3.5 L0,6.4 z"/></marker>'
        )
    return f"<defs>{''.join(parts)}</defs>"


def _pill(node: GraphNode, placed: PlacedNode, tone: str) -> str:
    label = f"{STATE_GLYPHS[node.state]} {STATE_LABELS[node.state].upper()}"
    width = len(label) * 6.0 + 12.0
    return (
        f'<rect class="tf-pill {tone}" x="{_num(placed.x + 10)}" '
        f'y="{_num(placed.y + _PILL_TOP)}" width="{_num(width)}" '
        f'height="{_num(_PILL_HEIGHT)}" rx="8"/>'
        f'<text class="tf-state {tone}" x="{_num(placed.x + 16)}" '
        f'y="{_num(placed.y + _STATE_BASELINE)}">{_esc(label)}</text>'
    )


def _why(node: GraphNode) -> str:
    """The tooltip: why this file is on the map, and where that was recorded."""
    lines = [f"{node.path} — {STATE_LABELS[node.state]}"]
    if node.detail:
        lines.append(node.detail)
    for item in node.evidence[:4]:
        lines.append(item.render())
    if len(node.evidence) > 4:
        lines.append(f"… and {len(node.evidence) - 4} more")
    return "\n".join(lines)


def _box(placed: PlacedNode) -> str:
    node = placed.node
    tone = STATE_TONES[node.state]
    inner = (
        f'<rect class="tf-box {tone}" x="{_num(placed.x)}" y="{_num(placed.y)}" '
        f'width="{_num(placed.width)}" height="{_num(placed.height)}" rx="8"/>'
        + _pill(node, placed, tone)
        + f'<text class="tf-path" x="{_num(placed.x + 10)}" y="{_num(placed.y + _PATH_BASELINE)}">'
        f"{_esc(_clip(node.path, _PATH_CHARS))}</text>"
    )
    if node.detail:
        inner += (
            f'<text class="tf-detail" x="{_num(placed.x + 10)}" '
            f'y="{_num(placed.y + _DETAIL_BASELINE)}">'
            f"{_esc(_clip(node.detail, _DETAIL_CHARS))}</text>"
        )
    if node.link:
        return f'<a href="{_esc(node.link)}"><title>{_esc(_why(node))}</title>{inner}</a>'
    return f"<g><title>{_esc(_why(node))}</title>{inner}</g>"


def _curve(points: tuple[tuple[float, float], ...]) -> str:
    start, first, second, end = points
    return (
        f"M{_num(start[0])},{_num(start[1])} "
        f"C{_num(first[0])},{_num(first[1])} {_num(second[0])},{_num(second[1])} "
        f"{_num(end[0])},{_num(end[1])}"
    )


def _edges(view: GraphView, drawing: Layout, prefix: str) -> str:
    parts: list[str] = []
    for routed in drawing.edges:
        tone = EDGE_TONES[routed.edge.kind]
        parts.append(
            f'<path class="tf-edge {tone}" d="{_curve(routed.points)}" '
            f'marker-end="url(#{prefix}-arrow-{tone})"/>'
        )
        if view.show_edge_labels:
            parts.append(
                f'<text class="tf-edge-label" x="{_num(routed.label_at[0])}" '
                f'y="{_num(routed.label_at[1] - 4)}">{_esc(routed.edge.label)}</text>'
            )
    return "".join(parts)


def _description(view: GraphView) -> str:
    lines = [view.caption, view.arrow]
    lines.extend(view.notes)
    if view.omitted:
        lines.append(f"{view.omitted} further node(s) are not drawn.")
    return " ".join(lines)


def _empty_document(view: GraphView) -> str:
    """A valid drawing that says why it is empty.

    An export that writes a zero-byte file, or a file with no root element, is one that
    fails to open with no explanation. This one opens and answers the question.
    """
    prefix = f"tf-{view.key}"
    return (
        f'<svg class="tf-map" xmlns="http://www.w3.org/2000/svg" width="560" height="80" '
        f'viewBox="0 0 560 80" role="img" aria-labelledby="{prefix}-title">'
        f'<title id="{prefix}-title">{_esc(view.title)} — {_esc(view.repository)}</title>'
        f"<style>{_styles()}</style>"
        f'<rect class="tf-box" x="0" y="0" width="544" height="64" rx="8"/>'
        f'<text class="tf-path" x="12" y="26">{_esc(view.title)}</text>'
        f'<text class="tf-detail" x="12" y="46">{_esc(view.empty)}</text>'
        f"</svg>"
    )


def render_svg(view: GraphView) -> str:
    """One view as an inline SVG document.

    Total on purpose: an empty view produces a document that opens and explains itself,
    rather than an empty string a caller has to remember to handle.
    """
    if not view.nodes:
        return _empty_document(view)

    drawing = layout(view.nodes, view.edges)
    width = drawing.width + 2 * _PAD
    height = drawing.height + 2 * _PAD
    prefix = f"tf-{view.key}"

    return (
        f'<svg class="tf-map" xmlns="http://www.w3.org/2000/svg" width="{_num(width)}" '
        f'height="{_num(height)}" viewBox="{-_PAD:.1f} {-_PAD:.1f} {_num(width)} {_num(height)}" '
        f'role="img" aria-labelledby="{prefix}-title" aria-describedby="{prefix}-desc">'
        f'<title id="{prefix}-title">{_esc(view.title)} — {_esc(view.repository)}</title>'
        f'<desc id="{prefix}-desc">{_esc(_description(view))}</desc>'
        f"<style>{_styles()}</style>"
        f"{_markers(prefix)}"
        f'<g class="tf-edges">{_edges(view, drawing, prefix)}</g>'
        f'<g class="tf-nodes">{"".join(_box(item) for item in drawing.nodes)}</g>'
        f"</svg>"
    )
