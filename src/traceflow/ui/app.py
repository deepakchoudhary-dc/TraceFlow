"""A local HTTP server for the delivery view (plan.md §29, §46, §61).

Standard library only. A dashboard that needed a framework, a bundler and a package
manager to show five JSON files would be more infrastructure than product, and plan.md
§77 is explicit that the command line comes first and unnecessary infrastructure does not.

Two properties are non-negotiable.

**It listens on the loopback interface and nothing else.** This serves the contents of
someone's source tree; binding to ``0.0.0.0`` would put that on the network by accident.

**The only paths it will read are the ones a session recorded.** Every route is a lookup
into an artifact that is already on disk — no route takes a path from the request and
opens it. A traversal attempt does not have to be blocked, because it never matches a
recorded file and simply 404s. That is a stronger guarantee than sanitising input, and it
is why :class:`Dashboard` resolves routes rather than serving a directory.

Routing is separated from sockets so the whole surface can be exercised by calling
:meth:`Dashboard.get` directly, with the HTTP layer covered by one live request.
"""

from __future__ import annotations

import contextlib
import sys
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlparse

from traceflow.blobs import BlobStore
from traceflow.config import Config
from traceflow.git.baseline import Baseline, baseline_from_json
from traceflow.git.repository import Repository
from traceflow.ui import render
from traceflow.ui.delivery import Delivery, load_delivery, session_url
from traceflow.ui.diff import file_diff
from traceflow.watcher.session import CHANGES_FILENAME, SessionStore

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

#: Requests longer than this are refused rather than parsed. A target is a session id and
#: a file name; anything much longer is not a request this tool serves.
MAX_REQUEST_TARGET = 4096

HTML = "text/html; charset=utf-8"
TEXT = "text/plain; charset=utf-8"

OK = 200
REDIRECT = 302
NOT_FOUND = 404

#: A baseline holding nothing, used only when a session's `baseline.json` is unreadable.
#: Diffing against it degrades to "the baseline contents are unavailable" rather than
#: rendering a whole file as added, which would be a confident lie about what changed.
EMPTY_BASELINE = Baseline(
    baseline_id="unavailable",
    captured_at="",
    commit=None,
    base_revision="",
    dirty=False,
    tracked_changes=0,
    untracked_files=0,
)


@dataclass(frozen=True)
class Response:
    """One resolved response, before any HTTP is involved."""

    status: int = OK
    body: str = ""
    content_type: str = HTML
    location: str | None = None

    @property
    def payload(self) -> bytes:
        return self.body.encode("utf-8")


class Dashboard:
    """Resolves a request target to a :class:`Response`.

    Deliberately free of sockets: everything the UI can do is decided here, so a test can
    exercise the whole surface without binding a port.
    """

    def __init__(
        self,
        repository: Repository,
        store: SessionStore,
        blobs: BlobStore,
        config: Config,
        default_session: str | None = None,
    ) -> None:
        self._repository = repository
        self._store = store
        self._blobs = blobs
        self._config = config
        self._default_session = default_session

    @property
    def repository(self) -> Repository:
        return self._repository

    # ------------------------------------------------------------------ helpers

    def _index_rows(self) -> tuple[tuple[str, str, str, int], ...]:
        """``(session_id, started_at, status, changed_files)``, newest first."""
        rows: list[tuple[str, str, str, int]] = []
        for session in reversed(self._store.list_sessions()):
            payload = self._store.read_artifact(session.session_id, CHANGES_FILENAME)
            totals = payload.get("totals") if isinstance(payload, dict) else None
            count = 0
            if isinstance(totals, dict):
                raw = totals.get("files")
                count = raw if isinstance(raw, int) and not isinstance(raw, bool) else 0
            rows.append((session.session_id, session.started_at, session.status, count))
        return tuple(rows)

    def _latest(self) -> str | None:
        if self._default_session is not None:
            return self._default_session
        rows = self._index_rows()
        return rows[0][0] if rows else None

    def _delivery(self, session_id: str) -> Delivery | None:
        return load_delivery(self._store, session_id)

    # ------------------------------------------------------------------ routing

    def get(self, target: str) -> Response:
        """Resolve *target*. Never raises: every failure is a status."""
        if len(target) > MAX_REQUEST_TARGET:
            return self._not_found("Request target too long.")

        segments = [unquote(part) for part in urlparse(target).path.split("/") if part]

        if not segments:
            latest = self._latest()
            return Response(
                status=REDIRECT, location=session_url(latest) if latest else "/sessions"
            )

        if segments == ["sessions"]:
            return Response(
                body=render.render_index(self._repository.name, self._index_rows(), self._latest())
            )

        if segments[0] != "session" or len(segments) < 2:
            return self._not_found("No such page.")

        session_id = segments[1]
        view = segments[2] if len(segments) > 2 else ""
        remaining = segments[3:]

        views: dict[str, Callable[[Delivery], str]] = {
            "": render.render_delivery,
            "impact": lambda item: render.render_impact(item, full=True),
            "graph": render.render_graph,
            "evidence": render.render_evidence,
            "session": render.render_session,
        }

        if view == "diff":
            if not remaining:
                return self._view(session_id, render.render_diff_list)
            return self._diff(session_id, "/".join(remaining))

        if view in views:
            return self._view(session_id, views[view])

        return self._not_found("No such page.")

    # ------------------------------------------------------------------ pages

    def _view(self, session_id: str, view: Callable[[Delivery], str]) -> Response:
        delivery = self._delivery(session_id)
        if delivery is None:
            return self._not_found(f"No session {session_id}.")
        return Response(body=view(delivery))

    def _diff(self, session_id: str, path: str) -> Response:
        """One file's diff.

        The requested path is matched against the files the session recorded. It is never
        joined to a directory or opened directly, so no request can name a file outside the
        session — including one containing ``..``.
        """
        delivery = self._delivery(session_id)
        if delivery is None:
            return self._not_found(f"No session {session_id}.")

        wanted = path.strip("/")
        recorded = next((item for item in delivery.files if item.path == wanted), None)
        if recorded is None:
            return self._not_found(f"{wanted or 'That path'} is not a file this session changed.")

        payload = self._store.read_artifact(session_id, "baseline.json")
        parsed = baseline_from_json(payload) if payload else None
        baseline = parsed if parsed is not None else EMPTY_BASELINE

        diff = file_diff(
            self._repository, baseline, self._blobs, self._config, recorded.diff_request
        )
        return Response(body=render.render_diff(delivery, diff))

    def _not_found(self, message: str) -> Response:
        return Response(
            status=NOT_FOUND,
            body=render.render_missing(message, self._repository.name),
        )


# --------------------------------------------------------------------------- server


def _handler_for(dashboard: Dashboard) -> type[BaseHTTPRequestHandler]:
    """Build a handler class bound to *dashboard*.

    A closure rather than a class attribute: shared mutable class state would make two
    dashboards in one process interfere, which is exactly what a test starting a second
    server would do.
    """

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "TraceFlow"

        def _respond(self, *, body: bool) -> None:
            response = dashboard.get(self.path)
            self.send_response(response.status)
            if response.location is not None:
                self.send_header("Location", response.location)
            else:
                self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(response.payload)))
            # A local dashboard reads files that change underneath it, so a cached page
            # would show a session that no longer matches the artifacts on disk.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if body:
                self.wfile.write(response.payload)

        def do_GET(self) -> None:
            self._respond(body=True)

        def do_HEAD(self) -> None:
            self._respond(body=False)

        def log_message(self, _format: str, *_args: Any) -> None:
            """Silence the default request log. A local dashboard is not a web server."""

    return Handler


def create_server(
    dashboard: Dashboard, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> ThreadingHTTPServer:
    """Bind a server without serving it. Port ``0`` picks a free one."""
    server = ThreadingHTTPServer((host, port), _handler_for(dashboard))
    server.daemon_threads = True
    return server


def serve(
    repository: Repository,
    store: SessionStore,
    blobs: BlobStore,
    config: Config,
    *,
    session_id: str | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    should_open: bool = True,
) -> int:
    """Serve the dashboard until interrupted. Returns a process exit code."""
    dashboard = Dashboard(repository, store, blobs, config, default_session=session_id)

    try:
        server = create_server(dashboard, host, port)
    except OSError as exc:
        print(f"error: cannot listen on {host}:{port} — {exc}", file=sys.stderr)
        print("Try another port with --port, or --port 0 to pick a free one.", file=sys.stderr)
        return 1

    url = f"http://{host}:{server.server_address[1]}/"
    print(f"TraceFlow delivery — {repository.name}")
    print(f"  {url}")
    print("  Ctrl-C to stop.")
    sys.stdout.flush()

    if should_open:
        # A machine with no browser is not a failure; the URL was already printed.
        with contextlib.suppress(OSError, webbrowser.Error):
            webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
        print("Stopped.")
        return 130
    finally:
        server.server_close()

    return 0
