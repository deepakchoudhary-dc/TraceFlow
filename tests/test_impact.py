"""The impact engine (plan.md §18, §19, §28, §60).

plan.md §68 asks the impact tests to cover direct dependencies, indirect dependencies,
cycles and disconnected modules. The tests that carry the most weight here are the ones
about *propagation*, because that is where this module can be confidently wrong rather
than merely incomplete: a signature change and a body change reach callers in
fundamentally different ways, and a report that conflates them is worse than no report.

Everything runs against a real repository. The engine's job is to read files, resolve
imports and walk a graph, and a mocked graph would only prove the mock is consistent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import changed_callers_repo, move_the_declaration, run_git
from traceflow.analysis.impact import (
    _candidate_names,
    bound_imports,
    bound_name,
    build_impact_report,
    enclosing_symbol,
    graph_is_stale,
)
from traceflow.analysis.models import (
    Confidence,
    ImpactCategory,
    ImpactedNode,
    ImpactReport,
    classify_file,
    is_test_path,
)
from traceflow.analysis.symbols import analyse_session_modules
from traceflow.blobs import BlobStore
from traceflow.config import STATE_DIRNAME, Config, parse_config
from traceflow.derived import AnalysisCache
from traceflow.git.baseline import Baseline, baseline_from_json, capture_baseline
from traceflow.git.diff import collect_changes
from traceflow.git.repository import Repository
from traceflow.languages.base import ImportRef
from traceflow.languages.python.analyzer import analyze_python
from traceflow.languages.python.ast_graph import build_dependency_graph

# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def blobs(repo_root: Path) -> BlobStore:
    return BlobStore(repo_root / STATE_DIRNAME)


@pytest.fixture
def cache(repo_root: Path) -> AnalysisCache:
    return AnalysisCache(repo_root / STATE_DIRNAME)


# --------------------------------------------------------------------------- helpers


def _write(root: Path, relative: str, text: str) -> None:
    """Write a file with explicit LF endings so line numbers are platform-independent."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _commit(root: Path, message: str = "change") -> None:
    run_git(root, "add", "-A")
    run_git(root, "commit", "-q", "-m", message)


def snapshot(repo: Repository, blobs: BlobStore, config: Config) -> Baseline:
    return capture_baseline(repo, blobs, repo.working_tree_state(config.ignore), config)


def impact(
    repo: Repository,
    baseline: Baseline,
    blobs: BlobStore,
    cache: AnalysisCache,
    config: Config,
) -> ImpactReport:
    changes = collect_changes(repo, baseline, repo.working_tree_state(config.ignore), blobs, config)
    return build_impact_report(repo, baseline, changes, blobs, cache, config)


def node(report: ImpactReport, path: str, symbol: str | None) -> ImpactedNode | None:
    for candidate in report.nodes:
        if candidate.path == path and candidate.symbol == symbol:
            return candidate
    return None


def required(report: ImpactReport, path: str, symbol: str | None) -> ImpactedNode:
    found = node(report, path, symbol)
    assert found is not None, (
        f"expected {path}::{symbol} among {[n.render() for n in report.nodes]}"
    )
    return found


def _auth_repo(repo_root: Path) -> None:
    """A package where a route calls a service function."""
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


def _chain_repo(repo_root: Path) -> None:
    """``c.py`` calls ``b.py`` calls ``a.py`` — two rings of dependency."""
    _write(repo_root, "a.py", "def f(x):\n    return x\n")
    _write(repo_root, "b.py", "from a import f\n\n\ndef g(x):\n    return f(x)\n")
    _write(repo_root, "c.py", "from b import g\n\n\ndef h(x):\n    return g(x)\n")
    _commit(repo_root, "add chain")


# --------------------------------------------------------------------------- direct impact


def test_a_changed_symbol_is_reported_as_direct(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    _write(repo_root, "svc.py", "def run():\n    return 1\n")
    _commit(repo_root, "add svc")
    baseline = snapshot(repo, blobs, config)

    _write(repo_root, "svc.py", "def run():\n    return 2\n")
    report = impact(repo, baseline, blobs, cache, config)

    changed = required(report, "svc.py", "run")
    assert changed.category is ImpactCategory.DIRECT
    assert changed.distance == 0
    assert changed.reason == "symbol_body_changed"
    assert changed.confidence is Confidence.CONFIRMED


def test_a_file_the_analyzer_cannot_read_is_still_reported_as_changed(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    _write(repo_root, "notes.md", "# Notes\n")
    _commit(repo_root, "add notes")
    baseline = snapshot(repo, blobs, config)

    _write(repo_root, "notes.md", "# Notes\n\nMore.\n")
    report = impact(repo, baseline, blobs, cache, config)

    changed = required(report, "notes.md", None)
    assert changed.reason == "file_modified"
    assert changed.distance == 0


def test_an_unchanged_repository_impacts_nothing(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    _write(repo_root, "svc.py", "def run():\n    return 1\n")
    _commit(repo_root, "add svc")
    baseline = snapshot(repo, blobs, config)

    report = impact(repo, baseline, blobs, cache, config)

    assert report.nodes == ()
    assert report.truncated is False


# --------------------------------------------------------------------------- propagation


def test_a_signature_change_reaches_the_caller(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """The central case: changing how a symbol is called forces every caller to be read."""
    _auth_repo(repo_root)
    baseline = snapshot(repo, blobs, config)

    _write(
        repo_root,
        "auth/service.py",
        "def authenticate(user, password, mfa=False):\n    return user\n",
    )
    report = impact(repo, baseline, blobs, cache, config)

    caller = required(report, "auth/routes.py", "login")
    assert caller.category is ImpactCategory.INDIRECT
    assert caller.reason == "signature_changed"
    assert caller.confidence is Confidence.CONFIRMED
    assert caller.distance == 1


def test_a_body_change_reaches_the_caller_only_as_a_possibility(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """A body change needs no caller edited, so reporting it as an obligation would lie."""
    _auth_repo(repo_root)
    baseline = snapshot(repo, blobs, config)

    _write(repo_root, "auth/service.py", "def authenticate(user, password):\n    return password\n")
    report = impact(repo, baseline, blobs, cache, config)

    caller = required(report, "auth/routes.py", "login")
    assert caller.category is ImpactCategory.POTENTIAL
    assert caller.reason == "body_changed"


def test_impact_attenuates_beyond_the_first_ring(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """``c`` is reached through ``b``, whose own signature never changed — so no obligation."""
    _chain_repo(repo_root)
    baseline = snapshot(repo, blobs, config)

    _write(repo_root, "a.py", "def f(x, y):\n    return x\n")
    report = impact(repo, baseline, blobs, cache, config)

    first = required(report, "b.py", "g")
    assert first.category is ImpactCategory.INDIRECT
    assert first.distance == 1

    second = required(report, "c.py", "h")
    assert second.category is ImpactCategory.POTENTIAL
    assert second.reason == "indirect_dependency"
    assert second.distance == 2


def test_a_signature_change_ranks_above_a_body_change(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    _write(repo_root, "core.py", "def sig(x):\n    return x\n\n\ndef body(x):\n    return x\n")
    _write(
        repo_root,
        "callers.py",
        "from core import body, sig\n"
        "\n"
        "\n"
        "def uses_sig(x):\n"
        "    return sig(x)\n"
        "\n"
        "\n"
        "def uses_body(x):\n"
        "    return body(x)\n",
    )
    _commit(repo_root, "add core")
    baseline = snapshot(repo, blobs, config)

    _write(
        repo_root, "core.py", "def sig(x, y):\n    return x\n\n\ndef body(x):\n    return x + 1\n"
    )
    report = impact(repo, baseline, blobs, cache, config)

    positions = {item.symbol: index for index, item in enumerate(report.nodes)}
    assert positions["uses_sig"] < positions["uses_body"]
    assert required(report, "callers.py", "uses_sig").category is ImpactCategory.INDIRECT
    assert required(report, "callers.py", "uses_body").category is ImpactCategory.POTENTIAL


def test_a_symbol_can_impact_a_caller_in_its_own_file(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """The import graph cannot express this: a module importing itself is not an edge."""
    _write(
        repo_root,
        "local.py",
        "def helper(x):\n    return x\n\n\ndef entry():\n    return helper(1)\n",
    )
    _commit(repo_root, "add local")
    baseline = snapshot(repo, blobs, config)

    _write(
        repo_root,
        "local.py",
        "def helper(x, y):\n    return x\n\n\ndef entry():\n    return helper(1)\n",
    )
    report = impact(repo, baseline, blobs, cache, config)

    caller = required(report, "local.py", "entry")
    assert caller.category is ImpactCategory.INDIRECT
    assert caller.reason == "signature_changed"
    # Resolved by a rule rather than an import statement, so not CONFIRMED: a local of the
    # same name would shadow it.
    assert caller.confidence is Confidence.HIGH_CONFIDENCE


def test_a_method_call_through_self_is_attributed_to_the_method(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    _write(
        repo_root,
        "svc.py",
        "class Service:\n"
        "    def helper(self, x):\n"
        "        return x\n"
        "\n"
        "    def entry(self):\n"
        "        return self.helper(1)\n",
    )
    _commit(repo_root, "add svc")
    baseline = snapshot(repo, blobs, config)

    _write(
        repo_root,
        "svc.py",
        "class Service:\n"
        "    def helper(self, x, y):\n"
        "        return x\n"
        "\n"
        "    def entry(self):\n"
        "        return self.helper(1)\n",
    )
    report = impact(repo, baseline, blobs, cache, config)

    caller = required(report, "svc.py", "Service.entry")
    assert caller.reason == "signature_changed"
    assert caller.confidence is Confidence.HIGH_CONFIDENCE


def test_a_changed_class_signature_reaches_its_constructor_callers(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """``User(...)`` names the imported class itself, with no dotted tail to resolve."""
    _write(repo_root, "models.py", "class User:\n    pass\n")
    _write(repo_root, "app.py", "from models import User\n\n\ndef build():\n    return User()\n")
    _commit(repo_root, "add models")
    baseline = snapshot(repo, blobs, config)

    _write(repo_root, "models.py", "class User(dict):\n    pass\n")
    report = impact(repo, baseline, blobs, cache, config)

    caller = required(report, "app.py", "build")
    assert caller.reason == "signature_changed"
    assert caller.confidence is Confidence.CONFIRMED


def test_an_aliased_import_still_resolves(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    _auth_repo(repo_root)
    _write(
        repo_root,
        "auth/routes.py",
        "from auth import service as svc\n\n\ndef login(user, password):\n"
        "    return svc.authenticate(user, password)\n",
    )
    _commit(repo_root, "alias the import")
    baseline = snapshot(repo, blobs, config)

    _write(
        repo_root,
        "auth/service.py",
        "def authenticate(user, password, mfa=False):\n    return user\n",
    )
    report = impact(repo, baseline, blobs, cache, config)

    caller = required(report, "auth/routes.py", "login")
    assert caller.category is ImpactCategory.INDIRECT
    assert caller.confidence is Confidence.CONFIRMED


def test_removing_a_symbol_reaches_every_caller(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """A deleted function whose callers were left behind is the strongest signal there is."""
    _auth_repo(repo_root)
    baseline = snapshot(repo, blobs, config)

    _write(repo_root, "auth/service.py", "def other(user):\n    return user\n")
    report = impact(repo, baseline, blobs, cache, config)

    removed = required(report, "auth/service.py", "authenticate")
    assert removed.reason == "symbol_removed"
    assert removed.distance == 0

    caller = required(report, "auth/routes.py", "login")
    assert caller.reason == "symbol_removed"
    assert caller.category is ImpactCategory.INDIRECT
    assert caller.confidence is Confidence.CONFIRMED


def test_a_removed_module_is_reported_as_a_removal(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    _write(repo_root, "gone.py", "def f():\n    return 1\n")
    _commit(repo_root, "add gone")
    baseline = snapshot(repo, blobs, config)

    (repo_root / "gone.py").unlink()
    report = impact(repo, baseline, blobs, cache, config)

    deleted = required(report, "gone.py", "f")
    assert deleted.reason == "file_deleted"
    assert deleted.distance == 0


# --------------------------------------------------------------------------- graph shape


def test_a_cycle_terminates(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    _write(repo_root, "x.py", "from y import ping\n\n\ndef pong():\n    return ping()\n")
    _write(repo_root, "y.py", "from x import pong\n\n\ndef ping():\n    return pong()\n")
    _commit(repo_root, "add cycle")
    baseline = snapshot(repo, blobs, config)

    _write(repo_root, "x.py", "from y import ping\n\n\ndef pong(extra=1):\n    return ping()\n")
    report = impact(repo, baseline, blobs, cache, config)

    # Bounded, and the mutually-recursive partner is reached exactly once.
    assert required(report, "y.py", "ping").reason == "signature_changed"
    assert len([item for item in report.nodes if item.path == "y.py"]) == 1


def test_a_disconnected_module_is_not_reported(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    _write(repo_root, "lonely.py", "def alone():\n    return 1\n")
    _auth_repo(repo_root)
    baseline = snapshot(repo, blobs, config)

    _write(
        repo_root,
        "auth/service.py",
        "def authenticate(user, password, mfa=False):\n    return user\n",
    )
    report = impact(repo, baseline, blobs, cache, config)

    assert "lonely.py" not in report.impacted_files


def test_a_new_module_and_its_import_appear_in_the_graph_diff(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    _write(repo_root, "app.py", "def main():\n    return 1\n")
    _commit(repo_root, "app")
    baseline = snapshot(repo, blobs, config)

    _write(repo_root, "ratelimit.py", "def check():\n    return True\n")
    _write(
        repo_root, "app.py", "from ratelimit import check\n\n\ndef main():\n    return check()\n"
    )
    report = impact(repo, baseline, blobs, cache, config)

    assert "ratelimit.py" in report.graph_diff.modules_added
    added = {(edge.source_path, edge.target_path) for edge in report.graph_diff.edges_added}
    assert ("app.py", "ratelimit.py") in added


def test_deleting_a_module_reports_who_still_imports_it(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """An import that no longer resolves is not merely unresolved — this session broke it."""
    _write(repo_root, "legacy.py", "def thing():\n    return 1\n")
    _write(repo_root, "app.py", "from legacy import thing\n\n\ndef main():\n    return thing()\n")
    _commit(repo_root, "add legacy")
    baseline = snapshot(repo, blobs, config)

    (repo_root / "legacy.py").unlink()
    report = impact(repo, baseline, blobs, cache, config)

    assert "legacy.py" in report.graph_diff.modules_removed
    dangling = required(report, "app.py", None)
    assert dangling.reason == "dangling_import"
    assert dangling.category is ImpactCategory.INDIRECT
    assert "legacy" in dangling.evidence[0].detail


# --------------------------------------------------------------------------- tests


def test_a_changed_test_file_is_categorised_as_a_test(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    _write(repo_root, "tests/test_app.py", "def test_one():\n    assert True\n")
    _commit(repo_root, "add tests")
    baseline = snapshot(repo, blobs, config)

    _write(repo_root, "tests/test_app.py", "def test_one():\n    assert 1 == 1\n")
    report = impact(repo, baseline, blobs, cache, config)

    changed = required(report, "tests/test_app.py", "test_one")
    assert changed.category is ImpactCategory.TEST
    assert changed.distance == 0


def test_a_test_depending_on_a_change_is_categorised_as_a_test(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    _write(repo_root, "svc.py", "def run(x):\n    return x\n")
    _write(
        repo_root,
        "tests/test_svc.py",
        "from svc import run\n\n\ndef test_run():\n    assert run(1)\n",
    )
    _commit(repo_root, "add svc and its test")
    baseline = snapshot(repo, blobs, config)

    _write(repo_root, "svc.py", "def run(x, y=1):\n    return x\n")
    report = impact(repo, baseline, blobs, cache, config)

    affected = required(report, "tests/test_svc.py", "test_run")
    assert affected.category is ImpactCategory.TEST
    assert affected.reason == "signature_changed"
    assert affected.distance == 1


# --------------------------------------------------------------------------- evidence


def test_signature_evidence_shows_what_the_declaration_became(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """A hash alone is not evidence (plan.md §33)."""
    _auth_repo(repo_root)
    baseline = snapshot(repo, blobs, config)

    _write(
        repo_root,
        "auth/service.py",
        "def authenticate(user, password, mfa=False):\n    return user\n",
    )
    report = impact(repo, baseline, blobs, cache, config)

    changed = required(report, "auth/service.py", "authenticate")
    detail = " ".join(item.detail for item in changed.evidence)
    assert "user, password -> user, password, mfa = False" in detail


def test_impact_evidence_points_at_the_call_site(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    _auth_repo(repo_root)
    baseline = snapshot(repo, blobs, config)

    _write(
        repo_root,
        "auth/service.py",
        "def authenticate(user, password, mfa=False):\n    return user\n",
    )
    report = impact(repo, baseline, blobs, cache, config)

    caller = required(report, "auth/routes.py", "login")
    calls = [item for item in caller.evidence if item.kind == "call_expression"]
    assert len(calls) == 1
    assert calls[0].path == "auth/routes.py"
    assert calls[0].line == 5
    assert "authenticate" in calls[0].detail


def test_the_chain_records_how_a_transitive_node_was_reached(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    _chain_repo(repo_root)
    baseline = snapshot(repo, blobs, config)

    _write(repo_root, "a.py", "def f(x, y):\n    return x\n")
    report = impact(repo, baseline, blobs, cache, config)

    assert required(report, "c.py", "h").chain == ("a.py", "b.py", "c.py")


# --------------------------------------------------------------------------- limitations


def test_dynamic_dispatch_is_reported_as_a_limitation(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """plan.md §70: saying where analysis is blind is what makes the rest trustworthy."""
    _write(repo_root, "dyn.py", "def lookup(registry, name):\n    return getattr(registry, name)\n")
    _commit(repo_root, "add dyn")
    baseline = snapshot(repo, blobs, config)

    _write(
        repo_root,
        "dyn.py",
        "def lookup(registry, name):\n    return getattr(registry, name, None)\n",
    )
    report = impact(repo, baseline, blobs, cache, config)

    assert any("getattr" in item for item in report.limitations)


def test_a_wildcard_import_is_reported_as_a_limitation(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    _write(repo_root, "helpers.py", "def h():\n    return 1\n")
    _write(repo_root, "user.py", "from helpers import *\n\n\ndef go():\n    return h()\n")
    _commit(repo_root, "add user")
    baseline = snapshot(repo, blobs, config)

    _write(repo_root, "user.py", "from helpers import *\n\n\ndef go():\n    return h() + 1\n")
    report = impact(repo, baseline, blobs, cache, config)

    assert any("wildcard import" in item for item in report.limitations)


def test_an_unresolvable_file_degrades_to_a_limitation(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """A half-written file is normal while an agent is working (plan.md §46)."""
    _write(repo_root, "broken.py", "def f(:\n")
    _commit(repo_root, "add broken")
    baseline = snapshot(repo, blobs, config)

    _write(repo_root, "broken.py", "def f(:\n    pass\n")
    report = impact(repo, baseline, blobs, cache, config)

    assert any("broken.py" in item and "symbols are unknown" in item for item in report.limitations)
    assert required(report, "broken.py", None).distance == 0


def test_the_depth_limit_is_recorded_rather_than_hidden(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache
) -> None:
    _chain_repo(repo_root)
    shallow = parse_config({"analysis": {"impact_max_depth": 1}})
    baseline = snapshot(repo, blobs, shallow)

    _write(repo_root, "a.py", "def f(x, y):\n    return x\n")
    report = impact(repo, baseline, blobs, cache, shallow)

    assert report.truncated is True
    assert report.max_depth == 1
    assert node(report, "b.py", "g") is not None
    assert node(report, "c.py", "h") is None
    assert any("configured depth" in item for item in report.limitations)


# --------------------------------------------------------------------------- classification


def test_the_reported_depth_is_the_deepest_ring_actually_reached(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """Not one more. The loop processes a final ring to discover it leads nowhere."""
    _chain_repo(repo_root)
    baseline = snapshot(repo, blobs, config)

    _write(repo_root, "a.py", "def f(x, y):\n    return x\n")
    report = impact(repo, baseline, blobs, cache, config)

    assert max(node.distance for node in report.nodes) == 2
    assert report.max_depth == 2


def test_a_dependency_manifest_is_categorised_as_a_dependency(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    _write(repo_root, "pyproject.toml", "[project]\nname = 'sample'\n")
    _commit(repo_root, "add manifest")
    baseline = snapshot(repo, blobs, config)

    _write(repo_root, "pyproject.toml", "[project]\nname = 'sample'\nversion = '1'\n")
    report = impact(repo, baseline, blobs, cache, config)

    assert required(report, "pyproject.toml", None).category is ImpactCategory.DEPENDENCY


def test_a_configuration_file_is_categorised_as_configuration(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    _write(repo_root, ".traceflow.toml", "ignore = ['.traceflow']\n")
    _commit(repo_root, "add config")
    baseline = snapshot(repo, blobs, config)

    _write(repo_root, ".traceflow.toml", "ignore = ['.traceflow', 'build']\n")
    report = impact(repo, baseline, blobs, cache, config)

    assert required(report, ".traceflow.toml", None).category is ImpactCategory.CONFIGURATION


def test_test_paths_are_recognised_conservatively() -> None:
    assert is_test_path("tests/test_auth.py")
    assert is_test_path("test/unit/thing.py")
    assert is_test_path("auth/test_service.py")
    assert is_test_path("conftest.py")
    # A substring match would classify these as tests and turn the section into noise.
    assert not is_test_path("latest/thing.py")
    assert not is_test_path("src/contest.py")


def test_classify_file_returns_nothing_for_ordinary_source() -> None:
    assert classify_file("auth/service.py") is None
    assert classify_file("pyproject.toml") is ImpactCategory.DEPENDENCY
    assert classify_file("tests/test_auth.py") is ImpactCategory.TEST


# --------------------------------------------------------------------------- units


def test_bound_name_reads_every_import_form() -> None:
    def ref(module: str, name: str | None, alias: str | None = None, level: int = 0) -> ImportRef:
        return ImportRef(module=module, name=name, alias=alias, level=level, line=1)

    assert bound_name(ref("a.b", None)) == "a"
    assert bound_name(ref("a.b", None, "c")) == "c"
    assert bound_name(ref("a", "b")) == "b"
    assert bound_name(ref("a", "b", "c")) == "c"
    assert bound_name(ref("", "b", level=1)) == "b"
    assert bound_name(ref("a", "*")) is None


def test_enclosing_symbol_picks_the_innermost_definition() -> None:
    analysis = analyze_python(
        "m.py",
        b"class C:\n    def m(self):\n        return 1\n\n\ndef f():\n    return 2\n",
    )

    assert enclosing_symbol(analysis, 3) == "C.m"
    assert enclosing_symbol(analysis, 7) == "f"
    assert enclosing_symbol(analysis, 100) is None


# --------------------------------------------------------------------------- artifacts


def test_the_report_serialises_to_json(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    _auth_repo(repo_root)
    baseline = snapshot(repo, blobs, config)

    _write(
        repo_root,
        "auth/service.py",
        "def authenticate(user, password, mfa=False):\n    return user\n",
    )
    report = impact(repo, baseline, blobs, cache, config)

    payload = report.to_json()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["totals"]["nodes"] == len(report.nodes)
    assert payload["totals"]["indirect"] == len(report.indirect)
    assert payload["totals"]["requiring_review"] == len(report.requiring_review)
    assert payload["graph_diff"]["modules_added"] == []


def test_a_baseline_survives_a_round_trip_through_json(
    repo: Repository, repo_root: Path, blobs: BlobStore, config: Config
) -> None:
    """``traceflow analyze`` reads the baseline back, so it has to survive the trip."""
    _write(repo_root, "dirty.py", "def f():\n    return 1\n")
    _commit(repo_root, "add dirty")
    _write(repo_root, "dirty.py", "def f():\n    return 2\n")
    _write(repo_root, "scratch.py", "x = 1\n")

    baseline = snapshot(repo, blobs, config)
    restored = baseline_from_json(baseline.to_json())

    assert restored == baseline
    assert restored is not None
    assert restored.dirty is True
    assert len(restored.captured) == 2


def test_a_malformed_baseline_payload_is_rejected() -> None:
    assert baseline_from_json({}) is None
    assert baseline_from_json({"baseline_id": "x"}) is None
    assert baseline_from_json({"baseline_id": 1, "base_revision": "abc"}) is None


def test_a_boolean_is_not_read_as_a_number() -> None:
    """``isinstance(True, int)`` is true, so a stored ``true`` must not become 1."""
    restored = baseline_from_json(
        {"baseline_id": "x", "base_revision": "abc", "tracked_changes": True}
    )

    assert restored is not None
    assert restored.tracked_changes == 0


# --------------------------------------------------------------------------- resolution units


def test_candidate_names_does_not_add_a_bare_tail_when_a_symbol_matched() -> None:
    """Otherwise a call to ``user.create()`` would match a change to a module-level ``create``."""
    symbols = analyze_python(
        "m.py", b"class User:\n    def create(self):\n        return 1\n"
    ).symbols

    assert _candidate_names(("create",), symbols) == ("User.create",)


def test_candidate_names_offers_the_bare_tail_when_nothing_matched() -> None:
    """The removed-symbol case: the caller still names it, and that call site is the finding."""
    assert _candidate_names(("gone",), ()) == ("gone",)


def test_candidate_names_keeps_every_symbol_sharing_the_leaf() -> None:
    symbols = analyze_python(
        "m.py",
        b"def create():\n    return 1\n\n\nclass User:\n    def create(self):\n        return 2\n",
    ).symbols

    assert _candidate_names(("create",), symbols) == ("User.create", "create")


def test_the_last_import_of_a_name_wins() -> None:
    """Python rebinds, so resolving to the first import would point at a stale binding."""
    analysis = analyze_python("m.py", b"from a import thing\nfrom b import thing\n")
    resolved = bound_imports(analysis)["thing"]

    assert resolved.module == "b"


# --------------------------------------------------------------------------- graph reuse


def _session_of(
    repo: Repository, baseline: Baseline, blobs: BlobStore, cache: AnalysisCache, config: Config
):
    """The session analysis the impact walk consumes."""
    changes = collect_changes(repo, baseline, repo.working_tree_state(config.ignore), blobs, config)
    return analyse_session_modules(repo, baseline, changes, blobs, cache, config)


def test_the_graph_survives_a_session_that_only_changed_a_body(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """Rebuilding it would re-parse every Python file for an answer that cannot differ."""
    _write(repo_root, "svc.py", "def run():\n    return 1\n")
    _commit(repo_root, "add svc")
    baseline = snapshot(repo, blobs, config)

    _write(repo_root, "svc.py", "def run():\n    return 2\n")

    assert graph_is_stale(_session_of(repo, baseline, blobs, cache, config)) is False


def test_the_graph_is_stale_when_a_file_appears(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    baseline = snapshot(repo, blobs, config)

    _write(repo_root, "fresh.py", "x = 1\n")

    assert graph_is_stale(_session_of(repo, baseline, blobs, cache, config)) is True


def test_the_graph_is_stale_when_an_import_is_added(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    _write(repo_root, "helper.py", "def h():\n    return 1\n")
    _write(repo_root, "app.py", "def main():\n    return 1\n")
    _commit(repo_root, "add")
    baseline = snapshot(repo, blobs, config)

    _write(repo_root, "app.py", "from helper import h\n\n\ndef main():\n    return h()\n")

    assert graph_is_stale(_session_of(repo, baseline, blobs, cache, config)) is True


def test_a_reused_graph_produces_the_same_report_as_a_fresh_one(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """The point of reuse: identical answers, less work. A stale graph would show here."""
    _auth_repo(repo_root)
    baseline = snapshot(repo, blobs, config)

    _write(
        repo_root,
        "auth/service.py",
        "def authenticate(user, password, mfa=False):\n    return user\n",
    )

    changes = collect_changes(repo, baseline, repo.working_tree_state(config.ignore), blobs, config)
    session = analyse_session_modules(repo, baseline, changes, blobs, cache, config)
    shared = build_dependency_graph(repo, cache, config)

    reused = build_impact_report(
        repo, baseline, changes, blobs, cache, config, graph=shared, session=session
    )
    fresh = build_impact_report(repo, baseline, changes, blobs, cache, config)

    assert reused.nodes == fresh.nodes
    assert reused.graph_diff == fresh.graph_diff
    assert reused.limitations == fresh.limitations


def test_the_changed_files_are_analysed_once_per_session(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """The symbol report and the impact walk must project from one pass, not two.

    Counts reads of the changed file rather than timing anything: a second analysis pass
    shows up as a second read, and the count is deterministic where a duration is not.

    The dependency graph is built outside the measured region on purpose — building it
    reads every Python file in the repository, which is real work rather than duplicated
    work, and including it would make the measurement say nothing.
    """
    _write(repo_root, "svc.py", "def run():\n    return 1\n")
    _commit(repo_root, "add svc")
    baseline = snapshot(repo, blobs, config)
    _write(repo_root, "svc.py", "def run():\n    return 2\n")

    graph = build_dependency_graph(repo, cache, config)

    reads: list[str] = []
    original = Path.read_bytes

    def counting_read_bytes(self: Path) -> bytes:
        if self.name == "svc.py":
            reads.append(str(self))
        return original(self)

    Path.read_bytes = counting_read_bytes  # type: ignore[method-assign]
    try:
        changes = collect_changes(
            repo, baseline, repo.working_tree_state(config.ignore), blobs, config
        )
        session = analyse_session_modules(repo, baseline, changes, blobs, cache, config)
        build_impact_report(
            repo, baseline, changes, blobs, cache, config, graph=graph, session=session
        )
    finally:
        Path.read_bytes = original  # type: ignore[method-assign]

    # Exactly once: the session analysis reads the current file, and both the symbol
    # report and the impact walk project from that one result.
    assert len(reads) == 1, reads


# --------------------------------------------------------------------------- two findings


def test_a_changed_symbol_keeps_the_obligation_that_reached_it(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """A session can change a symbol *and* reach it, and both facts are true.

    Recorded on the one node that represents the symbol. Dropping the second finding is how
    a session with three call sites of a moved declaration reported one caller.
    """
    changed_callers_repo(repo_root)
    baseline = snapshot(repo, blobs, config)
    move_the_declaration(repo_root)

    report = impact(repo, baseline, blobs, cache, config)

    login = required(report, "auth/routes.py", "login")
    assert login.reason == "symbol_body_changed", "its own change is still the primary record"
    assert [item.reason for item in login.reaches] == ["signature_changed"]

    reach = login.reaches[0]
    assert reach.distance == 1
    assert reach.chain == ("auth/service.py", "auth/routes.py")
    assert reach.category is ImpactCategory.INDIRECT
    assert reach.is_obligation is True
    assert reach.evidence, "a finding without a record behind it is an assertion"


def test_a_symbol_found_two_ways_is_still_one_node(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """Two rows for one symbol would invite a reader to count two components."""
    changed_callers_repo(repo_root)
    baseline = snapshot(repo, blobs, config)
    move_the_declaration(repo_root)

    report = impact(repo, baseline, blobs, cache, config)

    assert len([item for item in report.nodes if item.path == "auth/routes.py"]) == 1
    assert len([item for item in report.nodes if item.path == "auth/admin.py"]) == 1
    assert len(report.nodes) == len({(item.path, item.symbol) for item in report.nodes})


def test_the_obligation_counts_every_caller_not_only_the_untouched_one(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """Three call sites, three callers — two of them also edited by the session."""
    changed_callers_repo(repo_root)
    baseline = snapshot(repo, blobs, config)
    move_the_declaration(repo_root)

    report = impact(repo, baseline, blobs, cache, config)

    callers = [
        item
        for item in report.nodes
        for finding in [
            (item.reason, item.distance),
            *((r.reason, r.distance) for r in item.reaches),
        ]
        if finding == ("signature_changed", 1)
    ]
    assert len(callers) == 3, [item.render() for item in report.nodes]
    assert {item.path for item in callers} == {
        "auth/admin.py",
        "auth/routes.py",
        "tests/test_auth.py",
    }


def test_a_reach_deepens_the_depth_the_report_declares(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """A changed symbol sits at distance 0, but the walk can reach it rings further out."""
    _write(repo_root, "a.py", "def f(x):\n    return x\n")
    _write(repo_root, "b.py", "from a import f\n\n\ndef g(x):\n    return f(x)\n")
    _write(repo_root, "c.py", "from b import g\n\n\ndef h(x):\n    return g(x)\n")
    _commit(repo_root, "chain")
    baseline = snapshot(repo, blobs, config)

    _write(repo_root, "a.py", "def f(x, y):\n    return x\n")
    _write(repo_root, "c.py", "from b import g\n\n\ndef h(x):\n    return g(x) or None\n")
    report = impact(repo, baseline, blobs, cache, config)

    h = required(report, "c.py", "h")
    assert h.distance == 0, "its own change"
    assert [item.distance for item in h.reaches] == [2]
    assert h.reaches[0].chain == ("a.py", "b.py", "c.py")
    assert report.max_depth == 2, "the walk went two rings out, and the report must say so"


def test_the_same_obligation_reached_twice_is_recorded_once(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """One symbol calling two moved declarations is one caller, not two — with both records."""
    _write(repo_root, "x.py", "def one():\n    return 1\n")
    _write(repo_root, "y.py", "def two():\n    return 2\n")
    _write(
        repo_root,
        "z.py",
        "from x import one\nfrom y import two\n\n\ndef helper():\n    return one() + two()\n",
    )
    _commit(repo_root, "add")
    baseline = snapshot(repo, blobs, config)

    _write(repo_root, "x.py", "def one(a=1):\n    return a\n")
    _write(repo_root, "y.py", "def two(b=2):\n    return b\n")
    _write(
        repo_root,
        "z.py",
        "from x import one\nfrom y import two\n\n\ndef helper():\n    return one() + two() + 1\n",
    )
    report = impact(repo, baseline, blobs, cache, config)

    helper = required(report, "z.py", "helper")
    assert [item.reason for item in helper.reaches] == ["signature_changed"]
    assert len(helper.reaches[0].evidence) >= 2, (
        "one row may say 'two call sites', so both records have to be kept"
    )


def test_a_broken_import_is_recorded_alongside_the_symbol_that_changed(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """A changed function and a broken import are two facts about two different subjects.

    ``build.py::build`` changed, and the *module* ``build.py`` imports something the session
    deleted. Neither finding may be dropped for the other, and neither is a duplicate of it.
    """
    _write(repo_root, "legacy.py", "def old():\n    return None\n")
    _write(repo_root, "build.py", "from legacy import old\n\n\ndef build():\n    return old()\n")
    _commit(repo_root, "add")
    baseline = snapshot(repo, blobs, config)

    (repo_root / "legacy.py").unlink()
    _write(
        repo_root, "build.py", "from legacy import old\n\n\ndef build():\n    return old() or 1\n"
    )
    report = impact(repo, baseline, blobs, cache, config)

    assert required(report, "build.py", "build").reason == "symbol_body_changed"

    broken = required(report, "build.py", None)
    assert broken.reason == "dangling_import"
    assert broken.is_obligation is True
    assert broken.chain == ("legacy.py", "build.py")


def test_a_broken_import_merges_into_the_modules_own_record(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """The other merge site: the module already has a record, so the import joins it."""
    _write(repo_root, "legacy.py", "def old():\n    return None\n")
    _write(repo_root, "build.py", "from legacy import old\n\n\ndef build():\n    return old()\n")
    _commit(repo_root, "add")
    baseline = snapshot(repo, blobs, config)

    (repo_root / "legacy.py").unlink()
    # Only the imports change, so the module is recorded at module level rather than as a
    # changed symbol — which is the record the broken import has to join.
    _write(
        repo_root,
        "build.py",
        "import os\n\nfrom legacy import old\n\n\ndef build():\n    return old()\n",
    )
    report = impact(repo, baseline, blobs, cache, config)

    node_for_module = required(report, "build.py", None)
    assert node_for_module.reason == "imports_changed"
    assert [item.reason for item in node_for_module.reaches] == ["dangling_import"]
    assert len([item for item in report.nodes if item.path == "build.py"]) == 1


def test_a_reach_survives_the_json_round_trip(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """plan.md §35: the artifact is the interface, so a finding has to be in it."""
    changed_callers_repo(repo_root)
    baseline = snapshot(repo, blobs, config)
    move_the_declaration(repo_root)
    report = impact(repo, baseline, blobs, cache, config)

    payload = json.loads(json.dumps(report.to_json()))
    login = next(item for item in payload["nodes"] if item["path"] == "auth/routes.py")

    assert login["reason"] == "symbol_body_changed"
    assert [item["reason"] for item in login["reaches"]] == ["signature_changed"]
    assert login["reaches"][0]["chain"] == ["auth/service.py", "auth/routes.py"]
    assert login["reaches"][0]["category"] == "indirect"
    assert login["reaches"][0]["evidence"]


def test_the_report_follows_the_change_not_the_repository(
    repo: Repository, repo_root: Path, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """plan.md §49: the walk follows the change, not the repository.

    Measured separately on this machine — 250, 500, 1,000 and 2,000 files, the same change
    each time — the walk reached exactly three nodes at every size, while the graph build
    grew from 1.0s to 7.5s. Re-deriving that here would cost minutes of git and filesystem
    work for the same shape, so this asserts the shape instead: two hundred modules, and the
    report the three-file change implies.
    """
    _write(repo_root, "core.py", "def compute(x):\n    return x\n")
    for name in ("user_a", "user_b"):
        _write(
            repo_root,
            f"{name}.py",
            f"from core import compute\n\n\ndef {name}(x):\n    return compute(x)\n",
        )
    for index in range(200):
        _write(
            repo_root, f"mod_{index:03d}.py", f"def helper_{index}(x):\n    return x + {index}\n"
        )
    _commit(repo_root, "wide repository")
    baseline = snapshot(repo, blobs, config)

    _write(repo_root, "core.py", "def compute(x, *, strict=False):\n    return x\n")
    report = impact(repo, baseline, blobs, cache, config)

    assert sorted((item.path, item.symbol) for item in report.nodes) == [
        ("core.py", "compute"),
        ("user_a.py", "user_a"),
        ("user_b.py", "user_b"),
    ]
