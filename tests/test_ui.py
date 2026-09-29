"""The delivery UI (plan.md §26, §46, §61).

Three groups matter here.

**The diff policy.** The dashboard is the only place that shows file contents, so these
tests care less about the diff being pretty than about what it refuses to show. A test
that a secret never reaches the page is worth more than a test that a hunk header has the
right shape.

**Route resolution.** The claim that no request can reach outside a session is a security
property, so it is tested by attempting it rather than by reading the code that prevents
it.

**Live HTTP.** Routing is tested directly against :class:`Dashboard`, which covers the
whole surface without a socket. One test then binds a real port and fetches a page, so the
HTTP layer is exercised too — a routing table that passes in isolation but never answers a
request would otherwise look finished.
"""

from __future__ import annotations

import contextlib
import io
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from tests.conftest import changed_callers_repo, move_the_declaration, run_git
from traceflow.blobs import BlobStore
from traceflow.cli import EXIT_ERROR, EXIT_OK, EXIT_USAGE, main
from traceflow.config import STATE_DIRNAME, Config, load_config, parse_config
from traceflow.git.baseline import Baseline, baseline_from_json, capture_baseline
from traceflow.git.repository import Repository
from traceflow.ui.app import (
    DEFAULT_HOST,
    HTML,
    NOT_FOUND,
    OK,
    REDIRECT,
    Dashboard,
    create_server,
)
from traceflow.ui.delivery import (
    Concern,
    Delivery,
    Totals,
    derive_concerns,
    load_delivery,
    session_url,
)
from traceflow.ui.diff import (
    AVAILABLE,
    BINARY,
    TOO_LARGE,
    UNAVAILABLE,
    WITHHELD,
    DiffRequest,
    file_diff,
)
from traceflow.ui.render import (
    render_delivery,
    render_diff,
    render_graph,
    render_impact,
    render_index,
    render_session,
)
from traceflow.watcher.session import (
    CHANGES_FILENAME,
    CURRENT_BASELINE_FILENAME,
    IMPACT_FILENAME,
    SessionStore,
)

# --------------------------------------------------------------------------- helpers


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _commit(root: Path, message: str = "change") -> None:
    run_git(root, "add", "-A")
    run_git(root, "commit", "-q", "-m", message)


@pytest.fixture
def store(repo_root: Path) -> SessionStore:
    return SessionStore(repo_root)


@pytest.fixture
def blobs(repo_root: Path) -> BlobStore:
    return BlobStore(repo_root / STATE_DIRNAME)


@pytest.fixture
def dashboard(
    repo: Repository, repo_root: Path, store: SessionStore, blobs: BlobStore, config: Config
) -> Dashboard:
    return Dashboard(repo, store, blobs, config)


def _record(
    repo_root: Path, capsys: pytest.CaptureFixture[str] | None, mutate: Callable[[], None]
) -> str:
    """Record a session through the real CLI and return its id.

    The baseline is written directly rather than by running `analyze` twice. The session
    itself is still recorded by the real command, so the UI is asserted against the
    artifacts the product actually writes — and the baseline-writing path has its own
    tests in `test_cli.py`.

    *capsys* may be ``None``, in which case the caller has silenced the command itself.
    That is what lets :func:`mapped_session` be module-scoped: recording a session means
    running real analysis over a real repository, and a dozen tests here only read what it
    recorded.
    """
    repository = Repository.discover(repo_root)
    assert repository is not None
    config = load_config(repo_root)
    store = SessionStore(repo_root)
    store.ensure_state_dir()

    baseline = capture_baseline(
        repository,
        BlobStore(store.state_dir),
        repository.working_tree_state(config.ignore),
        config,
    )
    store.write_state(CURRENT_BASELINE_FILENAME, baseline.to_json())

    mutate()
    assert main(["analyze", str(repo_root)]) == EXIT_OK
    if capsys is not None:
        capsys.readouterr()
    sessions = SessionStore(repo_root).list_sessions()
    assert sessions, "no session was recorded"
    return sessions[-1].session_id


def _auth_repo(repo_root: Path) -> None:
    _write(repo_root, "auth/__init__.py", "")
    _write(repo_root, "auth/service.py", "def authenticate(user, password):\n    return user\n")
    _write(
        repo_root,
        "auth/routes.py",
        "from auth.service import authenticate\n"
        "\n"
        "\n"
        "def login(user, password):\n"
        "    return authenticate(user, password)\n",
    )
    _commit(repo_root, "add auth")


# --------------------------------------------------------------------------- diff policy


def _baseline_of(repo: Repository, blobs: BlobStore, config: Config) -> Baseline:
    return capture_baseline(repo, blobs, repo.working_tree_state(config.ignore), config)


def test_a_modified_file_diffs_against_the_session_baseline(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    _write(repo_root, "app.py", "def main():\n    return 1\n")
    _commit(repo_root, "app")
    baseline = _baseline_of(repo, blobs, config)
    _write(repo_root, "app.py", "def main():\n    return 2\n")

    diff = file_diff(repo, baseline, blobs, config, DiffRequest(path="app.py", status="modified"))

    assert diff.status == AVAILABLE
    assert diff.insertions == 1
    assert diff.deletions == 1
    assert any("return 2" in line for line in diff.lines)
    assert diff.drifted is False


def test_a_secret_file_is_refused_rather_than_redacted(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    """A redaction rule can be got wrong; a file never read cannot leak (plan.md §26)."""
    _write(repo_root, ".env", "API_KEY=do-not-show-me\n")
    _commit(repo_root, "add env")
    baseline = _baseline_of(repo, blobs, config)
    _write(repo_root, ".env", "API_KEY=still-do-not-show-me\n")

    diff = file_diff(repo, baseline, blobs, config, DiffRequest(path=".env", status="modified"))

    assert diff.status == WITHHELD
    assert diff.lines == ()
    rendered = render_diff(_stub_delivery(".env"), diff)
    assert "do-not-show-me" not in rendered
    assert "never" in rendered  # the reason is shown instead


def test_an_added_file_shows_as_wholly_added(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    baseline = _baseline_of(repo, blobs, config)
    _write(repo_root, "new.py", "x = 1\n")

    diff = file_diff(repo, baseline, blobs, config, DiffRequest(path="new.py", status="added"))

    assert diff.status == AVAILABLE
    assert diff.insertions == 1
    assert diff.deletions == 0


def test_a_deleted_file_shows_as_wholly_removed(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    _write(repo_root, "gone.py", "x = 1\n")
    _commit(repo_root, "add gone")
    baseline = _baseline_of(repo, blobs, config)
    (repo_root / "gone.py").unlink()

    diff = file_diff(repo, baseline, blobs, config, DiffRequest(path="gone.py", status="deleted"))

    assert diff.status == AVAILABLE
    assert diff.deletions == 1


def test_binary_content_is_not_diffed(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    """A line diff of binary content is not a meaningful thing to show."""
    (repo_root / "logo.bin").write_bytes(b"\x00\x01\x02")
    _commit(repo_root, "add binary")
    baseline = _baseline_of(repo, blobs, config)
    (repo_root / "logo.bin").write_bytes(b"\x00\x01\x03")

    diff = file_diff(repo, baseline, blobs, config, DiffRequest(path="logo.bin", status="modified"))

    assert diff.status == BINARY


def test_a_file_too_large_to_read_says_so(
    repo: Repository, repo_root: Path, blobs: BlobStore
) -> None:
    tiny = parse_config({"analysis": {"max_file_size_mb": 0.0001}})
    _write(repo_root, "big.py", "x = 1\n" * 200)
    _commit(repo_root, "add big")
    baseline = _baseline_of(repo, blobs, tiny)
    _write(repo_root, "big.py", "x = 2\n" * 200)

    diff = file_diff(repo, baseline, blobs, tiny, DiffRequest(path="big.py", status="modified"))

    assert diff.status == TOO_LARGE
    assert diff.note


def test_drift_since_the_session_is_stated_not_hidden(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    """A diff that silently folds a later edit into a session is a quiet misattribution."""
    _write(repo_root, "app.py", "def main():\n    return 1\n")
    _commit(repo_root, "app")
    baseline = _baseline_of(repo, blobs, config)
    _write(repo_root, "app.py", "def main():\n    return 2\n")

    recorded = DiffRequest(path="app.py", status="modified", recorded_digest="a" * 64)
    diff = file_diff(repo, baseline, blobs, config, recorded)

    assert diff.status == AVAILABLE
    assert diff.drifted is True
    assert "changed since the session was recorded" in diff.note


def test_a_long_diff_is_capped_and_says_so(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    _write(repo_root, "app.py", "x = 1\n")
    _commit(repo_root, "app")
    baseline = _baseline_of(repo, blobs, config)
    _write(repo_root, "app.py", "".join(f"y{i} = {i}\n" for i in range(600)))

    diff = file_diff(repo, baseline, blobs, config, DiffRequest(path="app.py", status="modified"))

    assert diff.truncated is True
    assert len(diff.lines) == 400
    assert "Showing the first 400" in diff.note


def test_a_file_that_vanished_is_reported_not_raised(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    baseline = _baseline_of(repo, blobs, config)

    diff = file_diff(
        repo, baseline, blobs, config, DiffRequest(path="never-existed.py", status="modified")
    )

    assert diff.status == UNAVAILABLE
    assert diff.note


# --------------------------------------------------------------------------- concerns


def _stub_delivery(*paths: str) -> Delivery:
    return Delivery(
        session_id="2026-01-01T00-00-00-aaaaaa",
        repository="stub",
        root=".",
        started_at="2026-01-01T00:00:00",
        stabilized_at=None,
        status="stabilized",
        baseline_commit=None,
        baseline_dirty=False,
        baseline_tracked_changes=0,
        baseline_untracked_files=0,
        baseline_captured=0,
        not_read=(),
        analyzer="python-1",
        totals=Totals(),
    )


def test_nothing_flagged_when_there_is_nothing_to_flag() -> None:
    concerns = derive_concerns(_stub_delivery())

    assert len(concerns) == 1
    assert concerns[0].title == "Nothing flagged"
    assert isinstance(concerns[0], Concern)


def test_a_signature_change_becomes_a_review_concern(
    repo: Repository, repo_root: Path, store: SessionStore, capsys: pytest.CaptureFixture[str]
) -> None:
    _auth_repo(repo_root)
    session_id = _record(
        repo_root,
        capsys,
        lambda: _write(
            repo_root,
            "auth/service.py",
            "def authenticate(user, password, mfa=False):\n    return user\n",
        ),
    )

    delivery = load_delivery(store, session_id)
    assert delivery is not None

    review = [item for item in delivery.concerns if item.severity == "review"]
    assert review, [item.title for item in delivery.concerns]
    assert any("changed declaration" in item.title for item in review)
    assert all(item.links for item in review), "a concern must link to its evidence"


def test_the_caller_count_does_not_include_the_change_itself(
    repo: Repository, repo_root: Path, store: SessionStore, capsys: pytest.CaptureFixture[str]
) -> None:
    """One caller is one caller. Counting the declaration made a session claim two."""
    _auth_repo(repo_root)
    session_id = _record(
        repo_root,
        capsys,
        lambda: _write(
            repo_root,
            "auth/service.py",
            "def authenticate(user, password, mfa=False):\n    return user\n",
        ),
    )

    delivery = load_delivery(store, session_id)
    assert delivery is not None

    reached = [node for node in delivery.nodes if node.reason == "signature_changed"]
    assert len(reached) == 1, [node.subject for node in reached]

    titles = [item.title for item in delivery.concerns]
    assert "1 caller(s) of a changed declaration" in titles, titles
    assert not any("2 caller" in title for title in titles)


def test_a_removed_symbol_with_no_references_is_not_a_concern(
    repo: Repository, repo_root: Path, store: SessionStore, capsys: pytest.CaptureFixture[str]
) -> None:
    """Deleting an unreferenced symbol is a change, not an obligation."""
    _write(repo_root, "spare.py", "def unused():\n    return 1\n")
    _commit(repo_root, "add spare")
    session_id = _record(repo_root, capsys, lambda: (repo_root / "spare.py").unlink())

    delivery = load_delivery(store, session_id)
    assert delivery is not None

    assert any(node.reason == "file_deleted" for node in delivery.nodes)
    assert not any("removed symbol" in item.title for item in delivery.concerns)


def test_a_truncated_walk_is_a_concern(
    repo: Repository, repo_root: Path, store: SessionStore, capsys: pytest.CaptureFixture[str]
) -> None:
    """A report that stopped early must say so, and the concern list is where it says it."""
    _write(repo_root, "a.py", "def f(x):\n    return x\n")
    _write(repo_root, "b.py", "from a import f\n\n\ndef g(x):\n    return f(x)\n")
    _write(repo_root, "c.py", "from b import g\n\n\ndef h(x):\n    return g(x)\n")
    (repo_root / ".traceflow.toml").write_text(
        "[analysis]\nimpact_max_depth = 1\n", encoding="utf-8"
    )
    _commit(repo_root, "chain")

    session_id = _record(
        repo_root, capsys, lambda: _write(repo_root, "a.py", "def f(x, y):\n    return x\n")
    )

    delivery = load_delivery(store, session_id)
    assert delivery is not None
    assert delivery.truncated is True, "the depth limit was not applied"
    assert any("cut short" in item.title for item in delivery.concerns)
    assert any(item.severity == "review" for item in delivery.concerns)


def test_a_missing_artifact_is_reported_rather_than_faked(
    repo: Repository, repo_root: Path, store: SessionStore, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(repo_root, "app.py", "def main():\n    return 1\n")
    _commit(repo_root, "app")
    session_id = _record(
        repo_root, capsys, lambda: _write(repo_root, "app.py", "def main():\n    return 2\n")
    )

    (store.session_dir(session_id) / IMPACT_FILENAME).unlink()

    delivery = load_delivery(store, session_id)

    assert delivery is not None
    assert "impact.json" in delivery.missing
    assert any("impact.json" in item.title for item in delivery.concerns)
    assert delivery.nodes == ()


def test_malformed_artifacts_degrade_instead_of_raising(
    repo: Repository, repo_root: Path, store: SessionStore, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(repo_root, "app.py", "def main():\n    return 1\n")
    _commit(repo_root, "app")
    session_id = _record(
        repo_root, capsys, lambda: _write(repo_root, "app.py", "def main():\n    return 2\n")
    )

    path = store.session_dir(session_id) / CHANGES_FILENAME
    path.write_text("{ this is not json", encoding="utf-8")

    delivery = load_delivery(store, session_id)

    assert delivery is not None
    assert "changes.json" in delivery.missing
    assert delivery.files == ()


# --------------------------------------------------------------------------- routing


def test_the_root_redirects_to_the_latest_session(
    repo_root: Path, dashboard: Dashboard, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(repo_root, "app.py", "def main():\n    return 1\n")
    _commit(repo_root, "app")
    session_id = _record(
        repo_root, capsys, lambda: _write(repo_root, "app.py", "def main():\n    return 2\n")
    )

    response = dashboard.get("/")

    assert response.status == REDIRECT
    assert response.location == session_url(session_id)


def test_the_first_screen_has_the_six_sections_plan_md_61_asks_for(
    repo_root: Path, dashboard: Dashboard, capsys: pytest.CaptureFixture[str]
) -> None:
    _auth_repo(repo_root)
    session_id = _record(
        repo_root,
        capsys,
        lambda: _write(
            repo_root,
            "auth/service.py",
            "def authenticate(user, password, mfa=False):\n    return user\n",
        ),
    )

    response = dashboard.get(session_url(session_id))

    assert response.status == OK
    assert response.content_type == HTML
    for heading in ("Session", "Changes", "Impact", "Tests", "Dependencies", "Potential concerns"):
        assert heading in response.body, f"missing section: {heading}"
    for action in ("View Graph", "View Diff", "View Evidence", "View Session"):
        assert action in response.body, f"missing drill-down: {action}"


def test_every_drill_down_answers(
    repo_root: Path, dashboard: Dashboard, capsys: pytest.CaptureFixture[str]
) -> None:
    _auth_repo(repo_root)
    session_id = _record(
        repo_root,
        capsys,
        lambda: _write(
            repo_root,
            "auth/service.py",
            "def authenticate(user, password, mfa=False):\n    return user\n",
        ),
    )

    for suffix in ("/impact", "/graph", "/diff", "/evidence", "/session"):
        response = dashboard.get(session_url(session_id, suffix))
        assert response.status == OK, f"{suffix} -> {response.status}"
        assert response.body.startswith("<!DOCTYPE html>")

    assert dashboard.get("/sessions").status == OK


def test_a_file_diff_page_renders(
    repo_root: Path, dashboard: Dashboard, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(repo_root, "app.py", "def main():\n    return 1\n")
    _commit(repo_root, "app")
    session_id = _record(
        repo_root, capsys, lambda: _write(repo_root, "app.py", "def main():\n    return 2\n")
    )

    response = dashboard.get(session_url(session_id, "/diff/app.py"))

    assert response.status == OK
    assert "return 2" in response.body


def test_a_secret_never_reaches_the_diff_page(
    repo_root: Path, dashboard: Dashboard, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(repo_root, ".env", "API_KEY=do-not-show-me\n")
    run_git(repo_root, "add", "-f", ".env")
    _commit(repo_root, "track env")
    session_id = _record(
        repo_root, capsys, lambda: _write(repo_root, ".env", "API_KEY=still-secret\n")
    )

    response = dashboard.get(session_url(session_id, "/diff/.env"))

    assert response.status == OK
    assert "do-not-show-me" not in response.body
    assert "still-secret" not in response.body


def test_a_withheld_file_is_reported_as_not_read(
    repo_root: Path, store: SessionStore, capsys: pytest.CaptureFixture[str]
) -> None:
    """The page must not claim every file was read while another section says otherwise.

    A file the *session* touched can have been withheld without the baseline ever capturing
    it — that is the normal case for a tracked sensitive file — so reading only the
    baseline's record produced a false assurance on the Session page.
    """
    _write(repo_root, ".env", "API_KEY=do-not-show-me\n")
    run_git(repo_root, "add", "-f", ".env")
    _commit(repo_root, "track env")
    session_id = _record(
        repo_root, capsys, lambda: _write(repo_root, ".env", "API_KEY=still-secret\n")
    )

    delivery = load_delivery(store, session_id)

    assert delivery is not None
    assert [path for path, _reason in delivery.not_read] == [".env"]
    rendered = render_session(delivery)
    assert "Every file was read" not in rendered
    assert ".env" in rendered
    assert any("were not read" in item.title for item in delivery.concerns)


def test_nothing_not_read_says_so(
    repo_root: Path, store: SessionStore, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(repo_root, "app.py", "def main():\n    return 1\n")
    _commit(repo_root, "app")
    session_id = _record(
        repo_root, capsys, lambda: _write(repo_root, "app.py", "def main():\n    return 2\n")
    )

    delivery = load_delivery(store, session_id)

    assert delivery is not None
    assert delivery.not_read == ()
    assert "Every file was read" in render_session(delivery)


def test_a_request_cannot_name_a_file_outside_the_session(
    repo_root: Path, dashboard: Dashboard, capsys: pytest.CaptureFixture[str]
) -> None:
    """The guarantee is a lookup, not a sanitiser: an unrecorded path simply does not match."""
    _write(repo_root, "app.py", "def main():\n    return 1\n")
    _commit(repo_root, "app")
    session_id = _record(
        repo_root, capsys, lambda: _write(repo_root, "app.py", "def main():\n    return 2\n")
    )

    for attempt in (
        "/diff/../../../../etc/passwd",
        "/diff/..%2F..%2Fsecrets.txt",
        "/diff/C:/Windows/win.ini",
        "/diff/.traceflow/events.jsonl",
        "/diff/not-a-file-this-session-touched.py",
    ):
        response = dashboard.get(session_url(session_id, attempt))
        assert response.status == NOT_FOUND, f"{attempt} was served"
        assert "app.py" not in response.body  # and no other file's contents either


def test_an_unknown_session_is_a_404(dashboard: Dashboard) -> None:
    assert dashboard.get("/session/not-a-session").status == NOT_FOUND
    assert dashboard.get("/session/not-a-session/diff/app.py").status == NOT_FOUND


def test_an_unknown_page_is_a_404(dashboard: Dashboard) -> None:
    assert dashboard.get("/nonsense").status == NOT_FOUND
    assert dashboard.get("/session").status == NOT_FOUND


def test_an_absurdly_long_target_is_refused(dashboard: Dashboard) -> None:
    assert dashboard.get("/session/" + "a" * 5000).status == NOT_FOUND


def test_the_index_works_with_no_sessions(dashboard: Dashboard) -> None:
    response = dashboard.get("/sessions")

    assert response.status == OK
    assert "No sessions yet" in response.body


# --------------------------------------------------------------------------- escaping


def test_html_in_a_path_is_escaped() -> None:
    """The content is the user's own source, trusted in origin and not in form."""
    delivery = _stub_delivery()
    nasty = replace(
        delivery,
        files=(),
        concerns=(Concern(severity="note", title="<script>alert(1)</script>", detail="x & y"),),
    )

    rendered = render_delivery(nasty)

    assert "<script>alert(1)</script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "x &amp; y" in rendered


def test_html_in_a_session_id_is_escaped() -> None:
    rendered = render_index("repo", (("<img src=x onerror=1>", "now", "stabilized", 1),), None)

    assert "<img" not in rendered
    assert "&lt;img" in rendered


# --------------------------------------------------------------------------- live HTTP


def test_a_real_request_is_answered(
    repo_root: Path, dashboard: Dashboard, capsys: pytest.CaptureFixture[str]
) -> None:
    """One test binds a socket, so the HTTP layer is exercised and not just the router."""
    _write(repo_root, "app.py", "def main():\n    return 1\n")
    _commit(repo_root, "app")
    session_id = _record(
        repo_root, capsys, lambda: _write(repo_root, "app.py", "def main():\n    return 2\n")
    )

    server = create_server(dashboard, "127.0.0.1", 0)
    assert server.server_address[0] == "127.0.0.1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10) as response:
            body = response.read().decode("utf-8")
            assert response.status == OK
            assert "Potential concerns" in body

        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}{session_url(session_id, '/diff/app.py')}", timeout=10
        ) as response:
            assert "return 2" in response.read().decode("utf-8")

        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/session/nope", timeout=10)
        assert excinfo.value.code == NOT_FOUND
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def test_there_is_no_way_to_bind_the_dashboard_to_the_network() -> None:
    """This serves someone's source tree, so the host is not a configurable option."""
    assert DEFAULT_HOST == "127.0.0.1"

    with pytest.raises(SystemExit) as excinfo:
        main(["ui", "--host", "0.0.0.0"])

    assert excinfo.value.code == EXIT_USAGE


# --------------------------------------------------------------------------- CLI wiring


def test_ui_on_a_plain_directory_fails(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()

    assert main(["ui", str(plain), "--no-open", "--port", "0"]) == EXIT_ERROR


def test_ui_passes_its_options_to_the_server(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`ui` blocks, so the command is tested by intercepting the server it would start."""
    from traceflow import cli

    captured: dict[str, object] = {}

    def fake_serve(*args: object, **kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "serve_dashboard", fake_serve)

    assert main(["ui", str(repo_root), "--port", "9999", "--no-open"]) == EXIT_OK
    capsys.readouterr()

    assert captured["port"] == 9999
    assert captured["should_open"] is False
    assert captured["session_id"] is None


def test_ui_accepts_an_explicit_session(
    repo_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from traceflow import cli

    captured: dict[str, object] = {}

    def fake_serve(*args: object, **kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(cli, "serve_dashboard", fake_serve)

    assert main(["ui", str(repo_root), "--session", "abc123", "--no-open"]) == EXIT_OK
    capsys.readouterr()

    assert captured["session_id"] == "abc123"


def test_impact_artifact_names_the_external_modules(
    repo_root: Path, store: SessionStore, capsys: pytest.CaptureFixture[str]
) -> None:
    """plan.md §74 wants `+ tenacity`, and a count with no names is not that."""
    _write(repo_root, "app.py", "import requests\n\n\ndef main():\n    return requests\n")
    _commit(repo_root, "app")
    session_id = _record(
        repo_root,
        capsys,
        lambda: _write(
            repo_root, "app.py", "import requests\n\n\ndef main():\n    return requests.get\n"
        ),
    )

    payload = store.read_artifact(session_id, IMPACT_FILENAME)

    assert payload is not None
    assert "requests" in payload["external_modules"]
    assert payload["unresolved_imports"] >= 1


def test_the_delivery_reads_the_artifacts_it_was_given(
    repo_root: Path, store: SessionStore, capsys: pytest.CaptureFixture[str]
) -> None:
    """The dashboard must not disagree with the artifact it renders."""
    _auth_repo(repo_root)
    session_id = _record(
        repo_root,
        capsys,
        lambda: _write(
            repo_root,
            "auth/service.py",
            "def authenticate(user, password, mfa=False):\n    return user\n",
        ),
    )

    artifact = store.read_artifact(session_id, IMPACT_FILENAME)
    delivery = load_delivery(store, session_id)

    assert artifact is not None and delivery is not None
    assert len(delivery.nodes) == len(artifact["nodes"])
    assert delivery.totals.impact_nodes == len(artifact["nodes"])
    assert {node.reason for node in delivery.nodes} == {
        node["reason"] for node in artifact["nodes"]
    }


def test_a_deleted_module_is_not_reported_as_an_external_dependency(
    repo_root: Path, store: SessionStore, capsys: pytest.CaptureFixture[str]
) -> None:
    """A removal explains the unresolved import; it is not a third-party package."""
    _write(repo_root, "legacy.py", "def thing():\n    return 1\n")
    _write(repo_root, "app.py", "from legacy import thing\n\n\ndef main():\n    return thing()\n")
    _write(repo_root, "ext.py", "import requests\n\n\ndef fetch():\n    return requests\n")
    _commit(repo_root, "app")
    session_id = _record(
        repo_root,
        capsys,
        lambda: (
            (repo_root / "legacy.py").unlink(),
            _write(
                repo_root, "ext.py", "import requests\n\n\ndef fetch():\n    return requests.get\n"
            ),
        ),
    )

    payload = store.read_artifact(session_id, IMPACT_FILENAME)
    assert payload is not None

    # `requests` is genuinely outside the repository; `legacy` is not, and reporting it as
    # external would name a module this session deleted as a dependency.
    assert payload["external_modules"] == ["requests"]
    assert payload["unresolved_imports"] == 1
    assert any(node["reason"] == "dangling_import" for node in payload["nodes"])
    assert not any("legacy" in item for item in payload["limitations"])


def test_the_json_on_disk_is_still_valid_after_rendering(
    repo_root: Path, store: SessionStore, capsys: pytest.CaptureFixture[str]
) -> None:
    """Rendering must not write anything: the UI is a view, not a second engine."""
    _write(repo_root, "app.py", "def main():\n    return 1\n")
    _commit(repo_root, "app")
    session_id = _record(
        repo_root, capsys, lambda: _write(repo_root, "app.py", "def main():\n    return 2\n")
    )

    before = (store.session_dir(session_id) / IMPACT_FILENAME).read_bytes()
    delivery = load_delivery(store, session_id)
    assert delivery is not None
    render_delivery(delivery)
    after = (store.session_dir(session_id) / IMPACT_FILENAME).read_bytes()

    assert before == after
    assert json.loads(after.decode("utf-8"))["totals"]["nodes"] == len(delivery.nodes)


# --------------------------------------------------------------------------- the change map


def _map_repo(repo_root: Path) -> None:
    """A call chain, a test, and a module only one other file imports."""
    _write(repo_root, "auth/__init__.py", "")
    _write(repo_root, "auth/service.py", "def authenticate(user, password):\n    return user\n")
    _write(
        repo_root,
        "auth/routes.py",
        "from auth.service import authenticate\n\n\ndef login(user, password):\n"
        "    return authenticate(user, password)\n",
    )
    _write(
        repo_root,
        "tests/test_auth.py",
        "from auth.service import authenticate\n\n\ndef test_authenticate():\n"
        "    return authenticate('a', 'b')\n",
    )
    _write(repo_root, "auth/legacy.py", "def old_login():\n    return None\n")
    _write(repo_root, "reports/__init__.py", "")
    _write(
        repo_root,
        "reports/build.py",
        "from auth.legacy import old_login\n\n\ndef build():\n    return old_login()\n",
    )
    _commit(repo_root, "add auth")


def _reshape(repo_root: Path) -> None:
    """A change with every shape the map has to draw.

    A moved declaration, a new module wired into the request path, a module deleted while
    something still imports it, and a body-only change on a file that is also a caller.
    """
    _write(
        repo_root,
        "auth/service.py",
        "def authenticate(user, password, *, strict=False):\n    return user\n",
    )
    _write(repo_root, "middleware/__init__.py", "")
    _write(repo_root, "middleware/rate_limit.py", "def check_rate_limit(user):\n    return True\n")
    _write(
        repo_root,
        "auth/routes.py",
        "from auth.service import authenticate\n"
        "from middleware.rate_limit import check_rate_limit\n"
        "\n"
        "\n"
        "def login(user, password):\n"
        "    check_rate_limit(user)\n"
        "    return authenticate(user, password)\n",
    )
    (repo_root / "auth/legacy.py").unlink()


@pytest.fixture(scope="module")
def mapped_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One repository, one recorded session, reused by every test that only reads it.

    Module-scoped on purpose. Recording a session runs real analysis over a real
    repository, and the page and export tests below all inspect the same recording. The
    fixture is never mutated after it returns, so sharing it costs no isolation.
    """
    root = tmp_path_factory.mktemp("mapped") / "sample-repo"
    root.mkdir()
    run_git(root, "init", "-q")
    _map_repo(root)
    return root


@pytest.fixture(scope="module")
def mapped_session(mapped_repo: Path) -> str:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return _record(mapped_repo, None, lambda: _reshape(mapped_repo))


def _graph_page(repo_root: Path, session_id: str) -> str:
    delivery = load_delivery(SessionStore(repo_root), session_id)
    assert delivery is not None
    return render_graph(delivery)


def test_the_graph_page_draws_the_map_rather_than_only_listing_chains(
    mapped_repo: Path, mapped_session: str
) -> None:
    """plan.md §62. Phase 5 stopped at a text listing; this is the laid-out map."""
    html = _graph_page(mapped_repo, mapped_session)

    assert html.count("<svg") == 2, "the change map and the before/after comparison"
    assert "Change map" in html
    assert "Before and after" in html


def test_every_box_on_the_map_names_its_state_in_words(
    mapped_repo: Path, mapped_session: str
) -> None:
    """plan.md §31: colour must not be the only indicator, so the map carries the words."""
    html = _graph_page(mapped_repo, mapped_session)

    for word in ("MODIFIED", "ADDED", "REMOVED", "AFFECTED"):
        assert word in html, word


def test_the_map_links_a_changed_file_to_its_diff(mapped_repo: Path, mapped_session: str) -> None:
    """plan.md §62's evidence links: a box is a way to check the claim, not a picture."""
    html = _graph_page(mapped_repo, mapped_session)

    assert f'href="{session_url(mapped_session, "/diff/auth/service.py")}"' in html
    assert f'href="{session_url(mapped_session, "/evidence")}"' in html


def test_the_before_after_map_labels_its_relationships(
    mapped_repo: Path, mapped_session: str
) -> None:
    """Two arrow meanings on one page would be a puzzle, so each map labels its own."""
    html = _graph_page(mapped_repo, mapped_session)

    assert ">imports<" in html, "the added import must be labelled on the arrow"
    assert "An arrow runs from the file that declares the import" in html
    assert "An arrow runs from a change to something it reaches" in html


def test_the_two_maps_do_not_share_document_ids(mapped_repo: Path, mapped_session: str) -> None:
    """Marker ids are document-wide; a shared one would make one map draw the other's arrows."""
    html = _graph_page(mapped_repo, mapped_session)

    ids = {part.split('"')[0] for part in html.split('id="')[1:]}
    assert "tf-change-title" in ids
    assert "tf-before-after-title" in ids
    assert "tf-change-arrow-affected" in ids
    assert "tf-before-after-arrow-added" in ids


def test_the_graph_page_keeps_the_same_records_in_words(
    mapped_repo: Path, mapped_session: str
) -> None:
    """A map is not readable to everyone. The text listing is the same records, in full."""
    html = _graph_page(mapped_repo, mapped_session)

    assert "Traversal chains" in html
    assert "Structural change" in html
    assert "auth/service.py → tests/test_auth.py" in html


def test_the_graph_page_states_all_five_of_plan_md_28s_comparisons(
    mapped_repo: Path, mapped_session: str
) -> None:
    html = _graph_page(mapped_repo, mapped_session)

    assert "auth/routes.py" in html, "a modified node"
    assert "module added" in html, "an added node"
    assert "module removed" in html, "a removed node"
    assert "relationship added" in html, "a new relationship"
    assert "plan.md §28" in html


def test_the_graph_page_says_where_the_limits_are(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A picture invites the reader to assume it is the whole picture."""
    _map_repo(repo_root)
    _write(repo_root, "extra.py", "import requests\n\n\ndef f():\n    return requests.get('x')\n")
    _commit(repo_root, "extra")
    session_id = _record(
        repo_root,
        capsys,
        lambda: _write(repo_root, "auth/service.py", "def authenticate(u, p):\n    return u\n"),
    )

    html = _graph_page(repo_root, session_id)

    assert "point outside the repository" in html


def test_the_map_page_contains_no_script(mapped_repo: Path, mapped_session: str) -> None:
    assert "<script" not in _graph_page(mapped_repo, mapped_session)


def test_rendering_the_map_leaves_the_impact_artifact_untouched(
    mapped_repo: Path, mapped_session: str
) -> None:
    """The map is a view of the artifacts, not a second engine (plan.md §35)."""
    path = SessionStore(mapped_repo).session_dir(mapped_session) / IMPACT_FILENAME
    before = path.read_bytes()

    _graph_page(mapped_repo, mapped_session)

    assert path.read_bytes() == before


def test_the_map_is_escaped_like_every_other_page(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A path is the user's own data, trusted in origin and not in form.

    The name is one Windows will accept, which rules out angle brackets — but ``&`` and
    ``'`` are enough to prove the escaping runs, and they are the two a path is most
    likely to actually contain.
    """
    _write(repo_root, "app.py", "def main():\n    return 1\n")
    _commit(repo_root, "app")
    session_id = _record(
        repo_root,
        capsys,
        lambda: _write(repo_root, "weird&name's.py", "def f():\n    return 1\n"),
    )

    html = _graph_page(repo_root, session_id)

    assert "weird&amp;name&#x27;s.py" in html
    assert "weird&name's.py" not in html


# --------------------------------------------------------------------------- export


def _exports(repo_root: Path) -> list[Path]:
    return sorted((repo_root / ".traceflow" / "exports").glob("*"))


def test_export_writes_into_the_state_directory_not_the_source_tree(
    mapped_repo: Path, mapped_session: str
) -> None:
    """TraceFlow's own output must not appear as a change the agent never made."""
    assert main(["export", str(mapped_repo)]) == EXIT_OK

    written = _exports(mapped_repo)
    assert mapped_repo / ".traceflow" / "exports" / f"{mapped_session}-change.excalidraw" in written


def test_the_exported_drawing_is_a_valid_excalidraw_file(
    mapped_repo: Path, mapped_session: str, tmp_path: Path
) -> None:
    destination = tmp_path / "map.excalidraw"
    assert main(["export", str(mapped_repo), "--output", str(destination)]) == EXIT_OK

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["type"] == "excalidraw"
    assert payload["version"] == 2
    assert payload["files"] == {}

    kinds = {element["type"] for element in payload["elements"]}
    assert {"rectangle", "text", "arrow"} <= kinds
    assert any(
        element["type"] == "arrow" and element["endArrowhead"] == "arrow"
        for element in payload["elements"]
    )


def test_the_exported_drawing_names_each_state(
    mapped_repo: Path, mapped_session: str, tmp_path: Path
) -> None:
    """plan.md §31 again: the drawing travels without the page, so it carries the words."""
    destination = tmp_path / "map.excalidraw"
    assert main(["export", str(mapped_repo), "--output", str(destination)]) == EXIT_OK

    payload = json.loads(destination.read_text(encoding="utf-8"))
    text = "\n".join(item["text"] for item in payload["elements"] if item["type"] == "text")

    assert "MODIFIED" in text
    assert "AFFECTED" in text
    assert "REMOVED" in text
    assert "An arrow runs from a change" in text


def test_export_json_is_the_internal_representation(
    mapped_repo: Path, mapped_session: str, tmp_path: Path
) -> None:
    destination = tmp_path / "map.json"
    assert (
        main(["export", str(mapped_repo), "--format", "json", "--output", str(destination)])
        == EXIT_OK
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["format"] == "traceflow.evidence-graph"
    assert payload["version"] == 1
    assert payload["view"] == "change"
    assert payload["session"] == mapped_session
    assert any(edge["kind"] == "affects" for edge in payload["edges"])


def test_export_svg_is_a_well_formed_document(
    mapped_repo: Path, mapped_session: str, tmp_path: Path
) -> None:
    import xml.etree.ElementTree as ET

    destination = tmp_path / "map.svg"
    assert (
        main(["export", str(mapped_repo), "--format", "svg", "--output", str(destination)])
        == EXIT_OK
    )

    assert ET.fromstring(destination.read_text(encoding="utf-8")).tag.endswith("svg")


def test_export_view_before_after_writes_the_structural_comparison(
    mapped_repo: Path, mapped_session: str, tmp_path: Path
) -> None:
    destination = tmp_path / "delta.json"
    assert (
        main(
            [
                "export",
                str(mapped_repo),
                "--view",
                "before-after",
                "--format",
                "json",
                "--output",
                str(destination),
            ]
        )
        == EXIT_OK
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["view"] == "before-after"
    assert all(edge["kind"].startswith("import_") for edge in payload["edges"])
    assert {node["path"] for node in payload["nodes"]} == {
        "auth/routes.py",
        "middleware/rate_limit.py",
    }


def test_the_two_views_export_to_different_files(mapped_repo: Path, mapped_session: str) -> None:
    main(["export", str(mapped_repo), "--view", "change"])
    main(["export", str(mapped_repo), "--view", "before-after"])

    names = {path.name for path in _exports(mapped_repo)}
    assert {
        f"{mapped_session}-before-after.excalidraw",
        f"{mapped_session}-change.excalidraw",
    } <= names


def test_export_does_not_make_the_repository_look_changed(
    mapped_repo: Path, mapped_session: str
) -> None:
    """The guard that matters: the watcher must not see TraceFlow's own output.

    ``init`` creates ``.traceflow.toml`` and edits ``.gitignore``, and those are meant to
    be visible — the config is a file the developer owns. What must not be visible is
    anything *inside* the state directory, which is where an export lands.
    """
    repository = Repository.discover(mapped_repo)
    assert repository is not None
    assert main(["init", str(mapped_repo)]) == EXIT_OK
    assert main(["export", str(mapped_repo)]) == EXIT_OK
    assert _exports(mapped_repo), "nothing was exported, so this test proves nothing"

    state = repository.working_tree_state(load_config(mapped_repo).ignore)
    assert not any(entry.path.startswith(".traceflow/") for entry in state.entries), state.entries

    status = run_git(mapped_repo, "status", "--porcelain")
    assert ".traceflow/" not in status, status


def test_export_with_an_unknown_session_fails(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["export", str(repo_root), "--session", "nope"]) == EXIT_ERROR

    assert "nope" in capsys.readouterr().err
    assert not _exports(repo_root)


def test_export_with_no_sessions_fails(repo_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["export", str(repo_root)]) == EXIT_ERROR

    assert "nothing to export" in capsys.readouterr().err


def test_export_outside_a_repository_fails(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()

    assert main(["export", str(plain)]) == EXIT_ERROR


def test_export_rejects_an_unknown_view(repo_root: Path) -> None:
    """A typo in a flag must be refused, not quietly answered with the default."""
    with pytest.raises(SystemExit):
        main(["export", str(repo_root), "--view", "graph"])


# --------------------------------------------------------------------------- two findings


def test_the_caller_count_matches_the_list_beneath_it(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The number and the rows are the same records, and they have to agree.

    They disagreed: the list said one caller where three call sites existed, because two of
    the callers had also been changed by the session and the walk dropped the second finding
    about them. A count is a claim, and a claim the list beneath it contradicts is worse than
    no count at all.
    """
    changed_callers_repo(repo_root)
    session_id = _record(repo_root, capsys, lambda: move_the_declaration(repo_root))

    delivery = load_delivery(SessionStore(repo_root), session_id)
    assert delivery is not None

    titles = [item.title for item in delivery.concerns]
    assert "3 caller(s) of a changed declaration" in titles, titles

    html = render_impact(delivery, full=True)
    assert html.count("calls a changed declaration") == 3, html


def test_a_changed_file_that_was_also_reached_is_listed_under_both(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Direct because it changed, indirect because it calls something that moved."""
    changed_callers_repo(repo_root)
    session_id = _record(repo_root, capsys, lambda: move_the_declaration(repo_root))

    delivery = load_delivery(SessionStore(repo_root), session_id)
    assert delivery is not None

    login = [item for item in delivery.findings if item.subject == "auth/routes.py::login"]

    assert {item.category for item in login} == {"direct", "indirect"}
    assert {item.reason for item in login} == {"symbol_body_changed", "signature_changed"}

    html = render_impact(delivery, full=True)
    assert "auth/routes.py::login" in html


def test_a_symbol_found_two_ways_is_one_row_on_the_page(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two rows for one symbol would invite a reader to count two components."""
    changed_callers_repo(repo_root)
    session_id = _record(repo_root, capsys, lambda: move_the_declaration(repo_root))

    delivery = load_delivery(SessionStore(repo_root), session_id)
    assert delivery is not None

    assert len([item for item in delivery.nodes if item.path == "auth/routes.py"]) == 1
    assert delivery.totals.impact_nodes == len(delivery.nodes)


def test_the_map_draws_an_arrow_for_a_reach(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The arrow between a changed file and the declaration it calls is a recorded finding."""
    changed_callers_repo(repo_root)
    session_id = _record(repo_root, capsys, lambda: move_the_declaration(repo_root))

    html = _graph_page(repo_root, session_id)

    assert "auth/service.py → auth/routes.py" in html
    assert "auth/service.py → auth/admin.py" in html


def test_a_reach_makes_a_symbol_an_obligation(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A file that changed *and* calls a moved declaration still has to be re-examined.

    The obligation is a property of the symbol, not of the row it happens to sit on, so the
    delivery has to see it through the reach.
    """
    changed_callers_repo(repo_root)
    session_id = _record(repo_root, capsys, lambda: move_the_declaration(repo_root))

    delivery = load_delivery(SessionStore(repo_root), session_id)
    assert delivery is not None

    login = next(item for item in delivery.nodes if item.subject == "auth/routes.py::login")

    assert login.reason == "symbol_body_changed", "its own change is still the primary record"
    assert login.is_obligation is True
    assert login.obligation_reasons == ("signature_changed",)
    assert login in delivery.requiring_review


def test_an_unusual_path_survives_the_whole_pipeline(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """plan.md §68 asks for malicious filenames; a path is the user's own data.

    Non-ASCII, punctuation and an HTML metacharacter, carried from ``git ls-files`` through
    analysis into the page. The name is escaped where it is rendered and nowhere else, so the
    artifact keeps the real path and the page cannot be made to emit markup.
    """
    names = ["café.py", "日本語.py", "a&b's.py"]
    for name in names:
        _write(repo_root, name, "def f():\n    return 1\n")
    _write(repo_root, "keep.py", "def keep():\n    return 1\n")
    _commit(repo_root, "unusual names")

    def mutate() -> None:
        for name in names:
            _write(repo_root, name, "def f(x):\n    return x\n")

    session_id = _record(repo_root, capsys, mutate)
    delivery = load_delivery(SessionStore(repo_root), session_id)
    assert delivery is not None

    changed = {item.path for item in delivery.files}
    assert set(names) <= changed, sorted(changed)

    html = _graph_page(repo_root, session_id)
    for name in ("café.py", "日本語.py"):
        assert name in html, name
    assert "a&amp;b&#x27;s.py" in html
    assert "a&b's.py" not in html
    assert "<script" not in html


def test_a_deleted_file_s_diff_does_not_claim_drift(
    repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A deleted file has not "changed since the session" — there is nothing to change.

    A deleted module is analysed as an empty one, and it used to record its digest as the
    empty string. The drift check compares a file's current digest against that, and no
    digest is ever the empty string, so every deleted file claimed to include edits that
    were not part of its session.
    """
    _write(repo_root, "gone.py", "def one():\n    return 1\n\n\ndef two():\n    return 2\n")
    _commit(repo_root, "add gone")
    session_id = _record(repo_root, capsys, lambda: (repo_root / "gone.py").unlink())

    delivery = load_delivery(SessionStore(repo_root), session_id)
    assert delivery is not None
    recorded = next(item for item in delivery.files if item.path == "gone.py")
    assert recorded.recorded_digest, "the digest of empty content is still a digest"

    payload = SessionStore(repo_root).read_artifact(session_id, "baseline.json")
    parsed = baseline_from_json(payload) if payload else None
    assert parsed is not None

    repository = Repository.discover(repo_root)
    assert repository is not None
    diff = file_diff(
        repository,
        parsed,
        BlobStore(repo_root / STATE_DIRNAME),
        load_config(repo_root),
        recorded.diff_request,
    )

    assert diff.status == AVAILABLE
    assert diff.drifted is False
    assert "changed since the session" not in diff.note
