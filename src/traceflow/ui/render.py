"""HTML for the delivery view (plan.md §29, §30, §31, §61, §71).

Server-rendered on purpose. There is no build step, no framework and no JavaScript, which
means the page a reader sees is the same document the tests assert on, and there is
nothing between the artifact and the screen that can fail quietly.

plan.md §31 requires that colour never be the only indicator, so every coloured badge
carries its label as text. plan.md §71 requires that the reader never needs to understand
ASTs, graph traversal or Git internals, so nothing here is named after them: a symbol is
"the declaration changed", not "signature_fingerprint differs".

Every value that reaches the page goes through :func:`esc`. The content is the user's own
source code, so it is trusted in origin and not in form — a path or a symbol name can
contain anything at all.
"""

from __future__ import annotations

from html import escape

from traceflow.ui.delivery import (
    CATEGORY_LABELS,
    Delivery,
    EdgeRef,
    ImpactNode,
    reason_label,
    session_url,
    short_symbol,
)
from traceflow.ui.diff import FileDiff
from traceflow.visualization.graph import (
    STATE_GLYPHS,
    STATE_LABELS,
    STATE_TONES,
    GraphView,
    NodeState,
    before_after,
    build_evidence_graph,
    change_map,
    file_detail,
)
from traceflow.visualization.svg import render_svg

#: Colour is a hint, never the message (plan.md §31). Each of these pairs with a word.
_TONE_ADDED = "added"
_TONE_REMOVED = "removed"
_TONE_CHANGED = "changed"
_TONE_AFFECTED = "affected"
_TONE_CONTEXT = "context"

_STATUS_TONES = {
    "added": _TONE_ADDED,
    "removed": _TONE_REMOVED,
    "deleted": _TONE_REMOVED,
    "modified": _TONE_CHANGED,
    "renamed": _TONE_CHANGED,
    "copied": _TONE_CHANGED,
    "type_changed": _TONE_CHANGED,
    "unknown": _TONE_CONTEXT,
}

_CATEGORY_TONES = {
    "direct": _TONE_CHANGED,
    "indirect": _TONE_AFFECTED,
    "test": _TONE_AFFECTED,
    "configuration": _TONE_CHANGED,
    "dependency": _TONE_CHANGED,
    "potential": _TONE_CONTEXT,
}

_CATEGORY_NAMES = {key: label for key, label, _ in CATEGORY_LABELS}

_SYMBOL_TONES = {
    "added": _TONE_ADDED,
    "removed": _TONE_REMOVED,
    "signature_changed": _TONE_CHANGED,
    "body_changed": _TONE_CHANGED,
}

_STYLE = """
:root {
  color-scheme: light dark;
  --bg: #ffffff; --fg: #1c1e21; --muted: #646a73; --line: #e3e5e8;
  --panel: #f7f8fa; --accent: #2f6feb;
  --added: #1a7f37; --added-bg: #dafbe1;
  --removed: #cf222e; --removed-bg: #ffebe9;
  --changed: #9a6700; --changed-bg: #fff8c5;
  --affected: #0969da; --affected-bg: #ddf4ff;
  --context: #57606a; --context-bg: #eaeef2;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117; --fg: #e6edf3; --muted: #8b949e; --line: #21262d;
    --panel: #161b22; --accent: #58a6ff;
    --added: #3fb950; --added-bg: #12261e;
    --removed: #f85149; --removed-bg: #2d1417;
    --changed: #d29922; --changed-bg: #2b2411;
    --affected: #58a6ff; --affected-bg: #0d2136;
    --context: #8b949e; --context-bg: #1c2128;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 14px/1.55 ui-sans-serif, -apple-system, "Segoe UI", system-ui, sans-serif;
}
code, pre, .mono {
  font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; font-size: 13px;
}
a { color: var(--accent); }
header.top {
  border-bottom: 1px solid var(--line); padding: 14px 24px;
  display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap;
}
header.top .brand { font-weight: 600; letter-spacing: .04em; }
header.top .repo { color: var(--muted); }
header.top nav { margin-left: auto; display: flex; gap: 14px; flex-wrap: wrap; }
main { padding: 24px; max-width: 1100px; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 {
  font-size: 12px; letter-spacing: .08em; text-transform: uppercase;
  color: var(--muted); margin: 28px 0 10px; border-bottom: 1px solid var(--line);
  padding-bottom: 6px;
}
h3 { font-size: 14px; margin: 18px 0 6px; }
p.lede { color: var(--muted); margin: 0 0 4px; }
.meta { display: grid; grid-template-columns: max-content 1fr; gap: 4px 18px; }
.meta dt { color: var(--muted); }
.meta dd { margin: 0; }
table { border-collapse: collapse; width: 100%; }
th, td {
  text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line);
  vertical-align: top;
}
th {
  color: var(--muted); font-weight: 500; font-size: 12px;
  text-transform: uppercase; letter-spacing: .06em;
}
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.badge {
  display: inline-block; padding: 1px 7px; border-radius: 999px;
  font-size: 11px; line-height: 18px; white-space: nowrap;
  border: 1px solid transparent;
}
.badge.added { color: var(--added); background: var(--added-bg); border-color: var(--added); }
.badge.removed {
  color: var(--removed); background: var(--removed-bg); border-color: var(--removed);
}
.badge.changed {
  color: var(--changed); background: var(--changed-bg); border-color: var(--changed);
}
.badge.affected {
  color: var(--affected); background: var(--affected-bg); border-color: var(--affected);
}
.badge.context {
  color: var(--context); background: var(--context-bg); border-color: var(--context);
}
.badge.plain { color: var(--muted); background: transparent; border-color: var(--line); }
.card {
  background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 14px 16px;
}
.card + .card { margin-top: 10px; }
.card.review { border-left: 3px solid var(--removed); }
.card.note { border-left: 3px solid var(--context); }
.card h3 { margin: 0 0 4px; }
.card p { margin: 0; color: var(--muted); }
ul.tree { list-style: none; margin: 0; padding: 0; }
ul.tree li { padding: 5px 0; border-bottom: 1px solid var(--line); }
ul.tree li:last-child { border-bottom: 0; }
.chain { color: var(--muted); font-size: 12px; }
.sub { color: var(--muted); font-size: 12px; }
pre.diff {
  background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 12px 0; overflow-x: auto; margin: 0;
}
pre.diff span { display: block; padding: 0 14px; white-space: pre; }
pre.diff span.add { color: var(--added); background: var(--added-bg); }
pre.diff span.del { color: var(--removed); background: var(--removed-bg); }
pre.diff span.hunk { color: var(--affected); }
pre.diff span.head { color: var(--muted); }
pre.raw {
  background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 14px; overflow-x: auto; margin: 0;
}
.notice {
  border: 1px solid var(--line); border-left: 3px solid var(--changed);
  background: var(--panel); border-radius: 6px; padding: 10px 14px; color: var(--muted);
}
.map {
  border: 1px solid var(--line); border-radius: 8px; background: var(--panel);
  padding: 10px; overflow-x: auto;
}
.map svg { max-width: none; }
p.legend { margin: 10px 0 0; display: flex; gap: 8px; flex-wrap: wrap; }
.actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 24px; }
.actions a {
  border: 1px solid var(--line); border-radius: 6px; padding: 7px 14px;
  text-decoration: none; background: var(--panel);
}
.empty { color: var(--muted); font-style: italic; }
blockquote.task {
  margin: 0 0 10px; padding: 10px 14px;
  background: var(--panel); border: 1px solid var(--line); border-left: 3px solid var(--accent);
  border-radius: 6px;
}
blockquote.task code { white-space: pre-wrap; }
"""


# --------------------------------------------------------------------------- helpers


def esc(value: object) -> str:
    """Escape anything for HTML. Paths and symbol names can contain markup characters."""
    return escape("" if value is None else str(value), quote=True)


def _badge(label: str, tone: str) -> str:
    return f'<span class="badge {tone}">{esc(label)}</span>'


def _status_badge(status: str) -> str:
    return _badge(status.replace("_", " "), _STATUS_TONES.get(status, _TONE_CONTEXT))


def _category_badge(category: str) -> str:
    label = _CATEGORY_NAMES.get(category, category)
    return _badge(label, _CATEGORY_TONES.get(category, _TONE_CONTEXT))


def _reason_badge(reason: str, distance: int) -> str:
    tone = _TONE_CONTEXT
    if reason in {"symbol_added", "file_added"}:
        tone = _TONE_ADDED
    elif reason in {"symbol_removed", "file_deleted", "dangling_import"}:
        tone = _TONE_REMOVED
    elif reason.startswith("symbol_") or reason.startswith("file_"):
        tone = _TONE_CHANGED
    elif reason in {"signature_changed", "symbol_removed"}:
        tone = _TONE_AFFECTED
    return _badge(reason_label(reason, distance), tone)


def _confidence_badge(confidence: str) -> str:
    return _badge(confidence.replace("_", " "), "plain")


def _file_link(session_id: str, path: str) -> str:
    return session_url(session_id, f"/diff/{path}")


def _chains_with_depth(node: ImpactNode) -> tuple[tuple[tuple[str, ...], int], ...]:
    """Every path the walk took to *node*, paired with the distance of that finding."""
    return (
        (node.chain, node.distance),
        *((item.chain, item.distance) for item in node.reaches),
    )


def _via_line(chain: tuple[str, ...], distance: int) -> str:
    """How a transitive finding was reached.

    Omitted for a first-ring result, where the direct relationship is the whole story —
    printing "via" on every row would be noise that buries the rows that need it.
    """
    if distance <= 1:
        return ""
    steps = " → ".join(esc(step) for step in chain[:-1])
    return f'<div class="chain">via {steps}</div>'


def _structural_edge(edge: EdgeRef, label: str, tone: str) -> str:
    return (
        f"<li><code>{esc(edge.source_path)}</code> → "
        f"<code>{esc(edge.target_path)}</code> {_badge(label, tone)}</li>"
    )


def _not_read_list(delivery: Delivery) -> str:
    """The files whose contents were deliberately not read, and why (plan.md §26).

    Rendered from the delivery's combined record rather than from the baseline alone: a
    file the *session* touched can also have been withheld, and reporting only the
    baseline's share let the page say "every file was read" while another section said
    otherwise.
    """
    if not delivery.not_read:
        return ""
    items = "".join(
        f'<li><code>{esc(path)}</code> <span class="sub">{esc(reason)}</span></li>'
        for path, reason in delivery.not_read
    )
    return f'<ul class="tree">{items}</ul>'


# --------------------------------------------------------------------------- chrome


def _nav(session_id: str | None) -> str:
    if session_id is None:
        return '<nav><a href="/sessions">Sessions</a></nav>'
    links = [
        ("Delivery", session_url(session_id)),
        ("Impact", session_url(session_id, "/impact")),
        ("Graph", session_url(session_id, "/graph")),
        ("Diff", session_url(session_id, "/diff")),
        ("Evidence", session_url(session_id, "/evidence")),
        ("Session", session_url(session_id, "/session")),
        ("Sessions", "/sessions"),
    ]
    links_html = "".join(f'<a href="{esc(href)}">{esc(label)}</a>' for label, href in links)
    return f"<nav>{links_html}</nav>"


def page(title: str, body: str, *, repository: str = "", session_id: str | None = None) -> str:
    """Wrap *body* in the shared document. Every page goes through here."""
    repo = f'<span class="repo">{esc(repository)}</span>' if repository else ""
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{esc(title)} · TraceFlow</title>"
        f"<style>{_STYLE}</style></head><body>"
        f'<header class="top"><span class="brand">TRACEFLOW</span>{repo}{_nav(session_id)}</header>'
        f"<main>{body}</main></body></html>\n"
    )


def _empty(message: str) -> str:
    return f'<p class="empty">{esc(message)}</p>'


# --------------------------------------------------------------------------- index


def render_index(
    repository: str, sessions: tuple[tuple[str, str, str, int], ...], latest: str | None
) -> str:
    """The session list (plan.md §36).

    *sessions* is ``(session_id, started_at, status, changed_files)``, newest first.
    """
    if not sessions:
        body = (
            "<h1>No sessions yet</h1>"
            '<p class="lede">Run <code>traceflow watch</code> and let an agent make a '
            "change, or <code>traceflow analyze</code> to analyse the working tree now.</p>"
        )
        return page("Sessions", body, repository=repository)

    rows: list[str] = []
    for session_id, started, status, count in sessions:
        marker = " " + _badge("latest", _TONE_AFFECTED) if session_id == latest else ""
        rows.append(
            "<tr>"
            f'<td><a href="{esc(session_url(session_id))}">{esc(session_id)}</a>{marker}</td>'
            f"<td>{esc(started)}</td>"
            f"<td>{_status_badge(status)}</td>"
            f'<td class="num">{count}</td>'
            "</tr>"
        )

    body = (
        "<h1>Sessions</h1>"
        f'<p class="lede">{len(sessions)} recorded episode(s) of change, newest first.</p>'
        "<table><thead><tr><th>Session</th><th>Started</th><th>Status</th>"
        '<th class="num">Files changed</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    return page("Sessions", body, repository=repository)


def render_missing(message: str, repository: str = "") -> str:
    return page(
        "Not found", f'<h1>Not found</h1><p class="lede">{esc(message)}</p>', repository=repository
    )


# --------------------------------------------------------------------------- delivery


def _session_section(delivery: Delivery, *, show_identity: bool = True) -> str:
    """The session's facts.

    *show_identity* is off on the first screen, whose heading already carries the
    repository and session id — printing them twice makes the page look padded.
    """
    baseline = delivery.baseline_commit or "(no commits yet)"
    if delivery.baseline_dirty:
        baseline += (
            f" — already dirty ({delivery.baseline_tracked_changes} tracked, "
            f"{delivery.baseline_untracked_files} untracked)"
        )
    else:
        baseline += " — clean"

    captured = f"{delivery.baseline_captured} file(s) snapshotted"
    if delivery.not_read:
        captured += f", {len(delivery.not_read)} not read"

    identity = ""
    if show_identity:
        identity = (
            f"<dt>Repository</dt><dd>{esc(delivery.repository)}</dd>"
            f'<dt>Session</dt><dd class="mono">{esc(delivery.session_id)}</dd>'
        )

    return (
        "<h2>Session</h2>"
        '<dl class="meta">'
        f"{identity}"
        f"<dt>Status</dt><dd>{_status_badge(delivery.status)}</dd>"
        f"<dt>Started</dt><dd>{esc(delivery.started_at)}</dd>"
        f"<dt>Stabilised</dt><dd>{esc(delivery.stabilized_at or '—')}</dd>"
        f'<dt>Baseline</dt><dd class="mono">{esc(baseline)}</dd>'
        f"<dt>Snapshot</dt><dd>{esc(captured)}</dd>"
        f'<dt>Analyzer</dt><dd class="mono">{esc(delivery.analyzer)}</dd>'
        "</dl>"
    )


def _changes_section(delivery: Delivery) -> str:
    totals = delivery.totals
    if not delivery.files:
        return "<h2>Changes</h2>" + _empty("Nothing changed in this session.")

    summary = (
        f"{totals.files} file(s), +{totals.insertions}/-{totals.deletions}"
        f" · {totals.signature_changes} declaration change(s)"
        f" · {totals.body_changes} body change(s)"
        f" · {totals.symbols_added} symbol(s) added"
        f" · {totals.symbols_removed} removed"
    )

    rows: list[str] = []
    for file in delivery.files:
        detail: list[str] = []
        for symbol in file.symbols:
            tone = _SYMBOL_TONES.get(symbol.change, _TONE_CONTEXT)
            detail.append(
                f"<div><code>{esc(short_symbol(symbol.qualified_name))}</code> "
                f"{_badge(symbol.change.replace('_', ' '), tone)}</div>"
            )
        if file.imports_added:
            detail.append(
                f'<div class="sub">+ {len(file.imports_added)} import(s): '
                f"<code>{esc(', '.join(file.imports_added[:4]))}</code></div>"
            )
        if file.imports_removed:
            detail.append(
                f'<div class="sub">- {len(file.imports_removed)} import(s): '
                f"<code>{esc(', '.join(file.imports_removed[:4]))}</code></div>"
            )
        if file.parse_error:
            detail.append('<div class="sub">could not be parsed — its symbols are unknown</div>')
        if file.withheld:
            detail.append('<div class="sub">contents withheld by the sensitive-path policy</div>')
        elif file.note:
            detail.append(f'<div class="sub">{esc(file.note)}</div>')

        counts = (
            f"+{file.insertions}/-{file.deletions}"
            if file.changed_lines is not None
            else ("binary" if file.binary else "no line counts")
        )
        rename = ""
        if file.original_path:
            rename = f' <span class="sub">from {esc(file.original_path)}</span>'
        rows.append(
            "<tr>"
            f'<td><a href="{esc(_file_link(delivery.session_id, file.path))}">'
            f"<code>{esc(file.path)}</code></a>{rename}"
            + ("".join(detail) if detail else "")
            + "</td>"
            f"<td>{_status_badge(file.status)}</td>"
            f'<td class="num mono">{esc(counts)}</td>'
            "</tr>"
        )

    return (
        f'<h2>Changes</h2><p class="lede">{esc(summary)}</p>'
        '<table><thead><tr><th>File</th><th>Status</th><th class="num">Lines</th></tr>'
        f"</thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _task_section(delivery: Delivery) -> str:
    """The recorded task versus what the session did (plan.md §63, §71).

    The section is omitted entirely when no task was recorded — an empty Task box on
    every session would say the product does something it was not asked to do.
    """
    intent = delivery.intent
    if intent is None:
        return ""

    relatedness = _badge(intent.label, _TONE_AFFECTED if intent.is_compared else _TONE_CONTEXT)

    if not intent.is_compared:
        detail = "".join(f'<p class="sub">{esc(note)}</p>' for note in intent.notes)
        return (
            f"<h2>Task</h2>"
            f'<blockquote class="task"><code>{esc(intent.task)}</code></blockquote>'
            f'<p class="lede">{relatedness}</p>{detail}'
        )

    matched: list[str] = []
    for item in intent.matches:
        where = (
            esc(item.detail) if item.kind == "path" else f"{esc(item.detail)} ({esc(item.kind)})"
        )
        matched.append(f"<li><code>{esc(item.token)}</code> → <code>{where}</code></li>")
    unmatched = (
        f'<p class="sub">Not matched: <code>{esc(", ".join(intent.unmatched))}</code></p>'
        if intent.unmatched
        else ""
    )
    notes = "".join(f'<p class="sub">{esc(note)}</p>' for note in intent.notes)

    return (
        "<h2>Task</h2>"
        f'<blockquote class="task"><code>{esc(intent.task)}</code></blockquote>'
        f'<p class="lede">{relatedness} · '
        f"{intent.changed_files} file(s) changed</p>"
        + (f'<ul class="tree">{"".join(matched)}</ul>' if matched else "")
        + unmatched
        + notes
        + '<p class="sub">This compares wording, not correctness. A change the task did '
        "not name may still be exactly what the task required.</p>"
    )


def _impact_section(delivery: Delivery) -> str:
    if not delivery.has_impact:
        return "<h2>Impact</h2>" + _empty(
            "Nothing outside the changed files was reached by this session."
        )

    blocks: list[str] = []
    for root in delivery.impact_roots:
        items: list[str] = []
        for finding in delivery.findings_from(root):
            badges = (
                f"{_category_badge(finding.category)} "
                f"{_reason_badge(finding.reason, finding.distance)} "
                f"{_confidence_badge(finding.confidence)}"
            )
            items.append(
                "<li>"
                f"<div><code>{esc(finding.subject)}</code> {badges}</div>"
                + _via_line(finding.chain, finding.distance)
                + "</li>"
            )
        if items:
            listing = f'<ul class="tree">{"".join(items)}</ul>'
            blocks.append(f"<h3><code>{esc(root)}</code></h3>{listing}")

    totals = delivery.totals
    counts = " · ".join(
        f"{len(delivery.findings_by_category(key))} {label.lower()}"
        for key, label, _ in CATEGORY_LABELS
        if key != "direct" and delivery.findings_by_category(key)
    )
    depth = ""
    if delivery.max_depth:
        depth = f" · depth {delivery.max_depth}" + (" (cut short)" if delivery.truncated else "")
    lede = esc(counts or "nothing reached") + depth
    lede += f" · {totals.requiring_review} requiring review"
    return f'<h2>Impact</h2><p class="lede">{lede}</p>' + "".join(blocks)


def _tests_section(delivery: Delivery) -> str:
    """What ran, what it said, and what the run does not prove (plan.md §22, §23, §64).

    Three states, in the reader's order of interest: a stored run (its result and its
    output tail), the tests the change *reaches* (which is known without running
    anything), and — when the repository never enabled test execution — that fact,
    said the same way the plan says it.
    """
    heading = "<h2>Tests</h2>"
    run_view = delivery.test_run

    if run_view is not None:
        run = run_view.run
        tone = _TONE_ADDED if run.status.value == "passed" else _TONE_REMOVED
        if run.status.value in ("not_run", "error"):
            tone = _TONE_CONTEXT
        lines = [
            f'<p class="lede">{_badge(run_view.label, tone)} '
            f"{esc(run.status.value.replace('_', ' '))}</p>"
        ]
        detail: list[str] = []
        if run.ran:
            counts = f"{run.passed} passed"
            if run.failed:
                counts += f", {run.failed} failed"
            if run.skipped:
                counts += f", {run.skipped} skipped"
            duration = f" in {run.duration_seconds:.1f}s" if run.duration_seconds else ""
            detail.append(f"{counts}{duration}.")
            if run.command:
                joined = " ".join(run.command)
                detail.append(f"Command: <code>{esc(joined)}</code>")
        if run.note:
            detail.append(esc(run.note))
        lines.append('<div class="notice">' + " ".join(detail) + "</div>")
        reached = _reached_tests(delivery)
        return heading + "".join(lines) + reached

    # No run: either the repository disabled tests (the default) or this session
    # predates the phase. Both are stated, not decorated.
    notice = (
        '<div class="notice">No test run is recorded. Test execution is off by default; '
        "enable it under <code>[tests]</code> in the configuration (plan.md §64). This "
        "section still shows which tests the change reaches — not whether they pass.</div>"
    )
    reached = _reached_tests(delivery)
    if reached:
        return heading + notice + reached
    return heading + notice + _empty("No test file reaches this change.")


def _reached_tests(delivery: Delivery) -> str:
    """The tests the session's own analysis says the change reaches."""
    tests = delivery.by_category("test")
    if not tests:
        return _empty("No test file reaches this change.")

    items: list[str] = []
    for node in tests:
        tone = (
            _badge("changed", _TONE_CHANGED)
            if node.distance == 0
            else _badge("affected", _TONE_AFFECTED)
        )
        reasons = " ".join(
            _reason_badge(reason, distance)
            for reason, distance, _confidence in node.reason_distances
        )
        items.append(
            "<li>"
            f"<div><code>{esc(node.subject)}</code> {tone} {reasons}</div>"
            + "".join(
                _via_line(chain, distance)
                for chain, distance in _chains_with_depth(node)
                if distance > 1
            )
            + "</li>"
        )
    return f'<ul class="tree">{"".join(items)}</ul>'


def _dependencies_section(delivery: Delivery) -> str:
    notice = (
        '<div class="notice">Dependency manifests (plan.md §24) are not parsed yet. This '
        "shows import relationships and the modules this repository references but does "
        "not contain.</div>"
    )
    blocks: list[str] = []
    if delivery.modules_added:
        blocks.append(
            '<h3>Modules added</h3><ul class="tree">'
            + "".join(
                f"<li><code>{esc(path)}</code> {_badge('added', _TONE_ADDED)}</li>"
                for path in delivery.modules_added
            )
            + "</ul>"
        )
    if delivery.modules_removed:
        blocks.append(
            '<h3>Modules removed</h3><ul class="tree">'
            + "".join(
                f"<li><code>{esc(path)}</code> {_badge('removed', _TONE_REMOVED)}</li>"
                for path in delivery.modules_removed
            )
            + "</ul>"
        )
    if delivery.edges_added or delivery.edges_removed:
        rows = "".join(
            f"<li><code>{esc(edge.source_path)}</code> → <code>{esc(edge.target_path)}</code> "
            f"{_badge('added', _TONE_ADDED)}</li>"
            for edge in delivery.edges_added
        ) + "".join(
            f"<li><code>{esc(edge.source_path)}</code> → <code>{esc(edge.target_path)}</code> "
            f"{_badge('removed', _TONE_REMOVED)}</li>"
            for edge in delivery.edges_removed
        )
        blocks.append(f'<h3>Relationships</h3><ul class="tree">{rows}</ul>')

    if delivery.external_modules:
        shown = delivery.external_modules[:24]
        more = len(delivery.external_modules) - len(shown)
        blocks.append(
            "<h3>Referenced but not in this repository</h3>"
            + '<p class="sub">Imports that resolve outside the repository. Third-party '
            "packages, mostly — impact through them is not visible.</p>"
            + '<ul class="tree">'
            + "".join(f"<li><code>{esc(name)}</code></li>" for name in shown)
            + (f'<li class="sub">… and {more} more</li>' if more > 0 else "")
            + "</ul>"
        )

    if not blocks:
        blocks.append(_empty("No dependency relationship changed in this session."))
    return f"<h2>Dependencies</h2>{notice}{''.join(blocks)}"


def _concerns_section(delivery: Delivery) -> str:
    cards = "".join(
        f'<div class="card {esc(concern.severity)}">'
        f"<h3>{esc(concern.title)}</h3><p>{esc(concern.detail)}</p>"
        + (
            '<p class="sub">'
            + " · ".join(f'<a href="{esc(href)}">{esc(label)}</a>' for label, href in concern.links)
            + "</p>"
            if concern.links
            else ""
        )
        + "</div>"
        for concern in delivery.concerns
    )
    return f"<h2>Potential concerns</h2>{cards}"


def _actions(delivery: Delivery) -> str:
    items = (
        ("View Graph", session_url(delivery.session_id, "/graph")),
        ("View Diff", session_url(delivery.session_id, "/diff")),
        ("View Evidence", session_url(delivery.session_id, "/evidence")),
        ("View Session", session_url(delivery.session_id, "/session")),
    )
    return (
        '<div class="actions">'
        + "".join(f'<a href="{esc(href)}">{esc(label)}</a>' for label, href in items)
        + "</div>"
    )


def render_delivery(delivery: Delivery) -> str:
    """The first screen: plan.md §61's six sections, in §74's order."""
    heading = (
        f"<h1>{esc(delivery.repository)}</h1>"
        f'<p class="lede">Session <span class="mono">{esc(delivery.session_id)}</span> · '
        f"{esc(delivery.started_at)}</p>"
    )
    body = (
        heading
        + _session_section(delivery, show_identity=False)
        + _task_section(delivery)
        + _changes_section(delivery)
        + _impact_section(delivery)
        + _tests_section(delivery)
        + _dependencies_section(delivery)
        + _concerns_section(delivery)
        + _actions(delivery)
    )
    return page(
        delivery.repository, body, repository=delivery.repository, session_id=delivery.session_id
    )


# --------------------------------------------------------------------------- drill-downs


def render_impact(delivery: Delivery, *, full: bool) -> str:
    """The impact view, grouped by category (plan.md §19, §61).

    Listed per *finding*, not per node, because that is what the concerns count and what the
    reader is checking: a file the session changed that also calls a declaration that moved
    belongs under Indirect as well as under Direct, and showing it in only one of them is how
    a count of three callers came to sit above a list of one.
    """
    blocks: list[str] = []
    for key, label, meaning in CATEGORY_LABELS:
        found = [item for item in delivery.findings if item.category == key]
        if not found:
            continue
        rows: list[str] = []
        for finding in found:
            depth = (
                f' <span class="sub">depth {finding.distance}</span>'
                if finding.is_transitive
                else ""
            )
            evidence = ""
            if finding.evidence:
                rendered = " · ".join(esc(item.render()) for item in finding.evidence)
                evidence = f'<div class="sub">{rendered}</div>'
            rows.append(
                "<li>"
                f"<div><code>{esc(finding.subject)}</code> "
                f"{_reason_badge(finding.reason, finding.distance)}"
                f" {_confidence_badge(finding.confidence)}{depth}</div>"
                + _via_line(finding.chain, finding.distance)
                + evidence
                + "</li>"
            )
        heading = f'<h3>{esc(label)} <span class="sub">— {esc(meaning)} ({len(found)})</span></h3>'
        blocks.append(f'{heading}<ul class="tree">{"".join(rows)}</ul>')

    if not blocks:
        blocks.append(_empty("This session reached nothing beyond its own changes."))

    if delivery.truncated:
        blocks.append(
            '<div class="notice">The walk stopped at depth '
            f"{delivery.max_depth}. Dependents further out are not listed.</div>"
        )
    elif not full:
        blocks.append(
            '<div class="notice">Every node the walk reached is shown. '
            "Use <code>--json</code> on <code>traceflow impact</code> for the raw artifact.</div>"
        )

    body = (
        "<h1>Impact</h1>"
        f'<p class="lede">{delivery.totals.impact_nodes} component(s) affected, '
        f"{delivery.totals.requiring_review} requiring review · "
        f"{len(delivery.findings)} finding(s) in all</p>" + "".join(blocks)
    )
    return page("Impact", body, repository=delivery.repository, session_id=delivery.session_id)


#: The order plan.md §31 lists the colours in, so the legend reads the way the plan does.
_LEGEND_ORDER = (
    NodeState.ADDED,
    NodeState.MODIFIED,
    NodeState.REMOVED,
    NodeState.AFFECTED,
    NodeState.CONTEXT,
)


def _legend(view: GraphView) -> str:
    """plan.md §31's colour key, showing only the states this map actually uses.

    Every entry carries the word and the glyph, not just the colour, which is what makes
    the legend a key rather than a decoration.
    """
    entries = [
        f'<span class="badge {STATE_TONES[state]}">{esc(STATE_GLYPHS[state])} '
        f"{esc(STATE_LABELS[state])}</span>"
        for state in _LEGEND_ORDER
        if any(node.state is state for node in view.nodes)
    ]
    if not entries:
        return ""
    return f'<p class="legend">{" ".join(entries)}</p>'


def _map_section(view: GraphView) -> str:
    """One diagram, its key, and what it leaves out (plan.md §31, §46, §69)."""
    heading = f"<h2>{esc(view.title)}</h2>"
    if not view.nodes:
        return heading + _empty(view.empty)

    notices = ""
    if view.omitted:
        notices += (
            f'<div class="notice">{view.omitted} further node(s) are not drawn — the map '
            "shows the change and its nearest consequences first. The listings below and "
            "<code>traceflow export --format json</code> have the whole set.</div>"
        )
    for note in view.notes:
        notices += f'<div class="notice">{esc(note)}</div>'

    return (
        heading
        + f'<p class="lede">{esc(view.caption)}</p>'
        + f'<p class="sub">{esc(view.arrow)}</p>'
        + f'<div class="map">{render_svg(view)}</div>'
        + _legend(view)
        + notices
    )


def _export_section(delivery: Delivery) -> str:
    """How to take the map elsewhere (plan.md §32, §44, §77)."""
    commands = (
        ("change", "excalidraw"),
        ("before-after", "excalidraw"),
        ("change", "json"),
    )
    items = "".join(
        f"<li><code>traceflow export --session {esc(delivery.session_id)} "
        f"--view {view} --format {kind}</code></li>"
        for view, kind in commands
    )
    return (
        "<h2>Export</h2>"
        '<p class="sub">plan.md §32 keeps the graph internal and Excalidraw, JSON and SVG '
        "as renderers of it. These write into <code>.traceflow/exports/</code>, which "
        "<code>init</code> keeps out of version control — use <code>--output</code> to put "
        "one anywhere else.</p>"
        f'<ul class="tree">{items}</ul>'
    )


def render_graph(delivery: Delivery) -> str:
    """The focused change map, the structural comparison, and the same in words.

    plan.md §62 asks for a focused graph, a before/after, changed-node highlighting and
    evidence links; plan.md §49 asks that it not try to draw the repository. The two maps
    are that. plan.md §31 requires that colour never be the only indicator, so every box
    names its own state and the text listings below repeat the maps in full — which is
    also what a reader without a graphical browser gets.
    """
    graph = build_evidence_graph(delivery)
    change = change_map(graph)
    delta = before_after(graph)

    reached = [item for item in delivery.findings if item.category != "direct"]
    if reached:
        rows: list[str] = []
        for finding in sorted(
            reached, key=lambda item: (item.distance, item.chain, item.path, item.reason)
        ):
            trail = " → ".join(esc(step) for step in finding.chain) or esc(finding.path)
            rows.append(
                "<li>"
                f'<div class="mono">{trail}</div>'
                f"<div><code>{esc(finding.subject)}</code> {_category_badge(finding.category)} "
                f"{_reason_badge(finding.reason, finding.distance)}</div>"
                "</li>"
            )
        chains = f'<h3>Traversal chains ({len(reached)})</h3><ul class="tree">{"".join(rows)}</ul>'
    else:
        chains = "<h3>Traversal chains</h3>" + _empty("No chain left the changed files.")

    structural: list[str] = []
    for file in delivery.files:
        structural.append(
            f"<li><code>{esc(file.path)}</code> {_status_badge(file.status)} "
            f'<span class="sub">{esc(file_detail(file))}</span></li>'
        )
    for path in delivery.modules_added:
        structural.append(
            f"<li><code>{esc(path)}</code> {_badge('module added', _TONE_ADDED)}</li>"
        )
    for path in delivery.modules_removed:
        structural.append(
            f"<li><code>{esc(path)}</code> {_badge('module removed', _TONE_REMOVED)}</li>"
        )
    for edge in delivery.edges_added:
        structural.append(_structural_edge(edge, "relationship added", _TONE_ADDED))
    for edge in delivery.edges_removed:
        structural.append(_structural_edge(edge, "relationship removed", _TONE_REMOVED))

    if structural:
        listing = f'<ul class="tree">{"".join(structural)}</ul>'
        structure = (
            f"<h2>Structural change ({len(structural)})</h2>"
            '<p class="sub">plan.md §28\'s five comparisons: the files that changed, the '
            "modules added and removed, and the relationships that appeared and "
            f"disappeared.</p>{listing}"
        )
    else:
        structure = "<h2>Structural change</h2>" + _empty("The shape of the graph did not change.")

    body = (
        "<h1>Graph</h1>"
        '<p class="lede">Only the part of the repository this session touched. '
        "Nothing disconnected from the change is drawn.</p>"
        + _map_section(change)
        + _map_section(delta)
        + "<h2>In words</h2>"
        '<p class="sub">The maps above, as text. Same records, same order.</p>'
        + chains
        + structure
        + _export_section(delivery)
    )
    return page("Graph", body, repository=delivery.repository, session_id=delivery.session_id)


def render_diff_list(delivery: Delivery) -> str:
    rows = "".join(
        "<tr>"
        f'<td><a href="{esc(_file_link(delivery.session_id, file.path))}">'
        f"<code>{esc(file.path)}</code></a></td>"
        f"<td>{_status_badge(file.status)}</td>"
        f'<td class="num mono">'
        + (
            esc(f"+{file.insertions}/-{file.deletions}")
            if file.changed_lines is not None
            else ("binary" if file.binary else "—")
        )
        + "</td></tr>"
        for file in delivery.files
    )
    if not rows:
        rows = f'<tr><td colspan="3">{_empty("Nothing changed.")}</td></tr>'
    body = (
        "<h1>Diff</h1>"
        f'<p class="lede">{len(delivery.files)} file(s) changed in this session, '
        "each measured against the session's own baseline.</p>"
        '<table><thead><tr><th>File</th><th>Status</th><th class="num">Lines</th></tr>'
        f"</thead><tbody>{rows}</tbody></table>"
    )
    return page("Diff", body, repository=delivery.repository, session_id=delivery.session_id)


def _diff_lines(lines: tuple[str, ...]) -> str:
    rendered: list[str] = []
    for line in lines:
        if line.startswith("+++") or line.startswith("---"):
            tone = "head"
        elif line.startswith("@@"):
            tone = "hunk"
        elif line.startswith("+"):
            tone = "add"
        elif line.startswith("-"):
            tone = "del"
        else:
            tone = ""
        rendered.append(f'<span class="{tone}">{esc(line) or "&nbsp;"}</span>')
    return f'<pre class="diff">{"".join(rendered)}</pre>'


def render_diff(delivery: Delivery, diff: FileDiff) -> str:
    """One file's diff, or the reason there is not one (plan.md §26, §46)."""
    heading = (
        f"<h1><code>{esc(diff.path)}</code></h1>"
        f'<p class="lede">{esc(diff.status.replace("_", " "))} · '
        f'<a href="{esc(session_url(delivery.session_id, "/diff"))}">all changed files</a></p>'
    )
    notes = f'<div class="notice">{esc(diff.note)}</div>' if diff.note else ""
    if diff.is_available:
        body = heading + notes + _diff_lines(diff.lines)
    else:
        body = heading + (notes or f'<div class="notice">{esc(diff.note)}</div>')
    return page(
        f"Diff · {diff.path}", body, repository=delivery.repository, session_id=delivery.session_id
    )


def render_evidence(delivery: Delivery) -> str:
    """Every claim's supporting record (plan.md §33)."""
    rows: list[str] = []
    for node in delivery.nodes:
        for item in node.evidence:
            rows.append(
                "<tr>"
                f"<td><code>{esc(item.path)}</code>"
                + (f'<span class="sub">:{item.line}</span>' if item.line else "")
                + "</td>"
                f"<td>{_badge(item.kind.replace('_', ' '), 'plain')}</td>"
                f"<td>{esc(item.detail)}</td>"
                f"<td><code>{esc(node.subject)}</code></td>"
                "</tr>"
            )

    evidence = (
        "<table><thead><tr><th>Where</th><th>Kind</th><th>What it says</th>"
        "<th>Supports</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        if rows
        else _empty("No impact relationship was recorded, so there is no evidence to show.")
    )

    limits = (
        '<h3>Limits on this analysis</h3><ul class="tree">'
        + "".join(f"<li>{esc(item)}</li>" for item in delivery.limitations)
        + "</ul>"
        if delivery.limitations
        else "<h3>Limits on this analysis</h3>" + _empty("None recorded.")
    )

    withheld = ""
    if delivery.not_read:
        withheld = "<h3>Not read</h3>" + _not_read_list(delivery)

    test_output = ""
    if delivery.test_run is not None and delivery.test_run.run.output_tail is not None:
        run = delivery.test_run.run
        if run.ran:
            escaped = esc(run.output_tail or "")
            truncation = (
                '<p class="sub">Output was long; the last part is shown.</p>'
                if run.output_truncated
                else ""
            )
            test_output = (
                "<h3>Test run output</h3>" + truncation + f'<pre class="raw">{escaped}</pre>'
            )

    body = (
        "<h1>Evidence</h1>"
        '<p class="lede">Why TraceFlow believes each relationship it reported. '
        "Every row is a location in the source.</p>" + evidence + limits + test_output + withheld
    )
    return page("Evidence", body, repository=delivery.repository, session_id=delivery.session_id)


def render_session(delivery: Delivery) -> str:
    """The session record itself, including what was deliberately not read (plan.md §14, §26)."""
    baseline_rows = _not_read_list(delivery) or _empty("Every file was read.")

    pre_existing = (
        "<h3>Pre-existing changes</h3>"
        '<p class="sub">Already present when the session began, and excluded from it.</p>'
        "<table><thead><tr><th>File</th><th>Status</th>"
        '<th class="num">Lines</th></tr></thead><tbody>'
        + "".join(
            f"<tr><td><code>{esc(file.path)}</code></td><td>{_status_badge(file.status)}</td>"
            f'<td class="num mono">'
            + (
                esc(f"+{file.insertions}/-{file.deletions}")
                if file.changed_lines is not None
                else "—"
            )
            + "</td></tr>"
            for file in delivery.pre_existing
        )
        + "</tbody></table>"
        if delivery.pre_existing
        else ""
    )

    task_section = _task_section(delivery)
    artifacts_note = (
        f'<div class="notice">Missing: {esc(", ".join(delivery.missing))}</div>'
        if delivery.missing
        else '<p class="sub">All core artifacts are present: session, baseline, changes, '
        "symbols and impact.</p>"
    )

    body = (
        "<h1>Session</h1>"
        + _session_section(delivery)
        + task_section
        + "<h2>Not read</h2>"
        + '<p class="sub">Files the sensitive-path policy kept out of memory. Their '
        "contents were never read, so their changes cannot be described.</p>"
        + baseline_rows
        + pre_existing
        + "<h2>Artifacts</h2>"
        + artifacts_note
        + '<p class="sub">A recorded task is stored in intent.json; test results, when '
        "the repository enables them, in tests.json.</p>"
    )
    return page("Session", body, repository=delivery.repository, session_id=delivery.session_id)
