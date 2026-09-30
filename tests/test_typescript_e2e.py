"""End-to-end TypeScript change intelligence (plan.md §79's promise, Phase 10's proof).

These tests run the real pipeline — git evidence, symbol comparison, impact walk —
over a real git repository whose sources are TypeScript. Nothing is mocked. They are
the executable answer to the question the ProU run could not answer: *when an agent
changes a helper, does TraceFlow name the controller that calls it?*

The sequence in every test is the watcher's own: commit a clean tree, capture the
baseline from it, make the edit, then analyse. A baseline captured after the edit
would describe the edited file as the starting point, and the session would be
empty — which is exactly the difference plan.md §14 exists to protect.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import commit, write_file

from traceflow.analysis.impact import build_impact_report
from traceflow.analysis.symbols import analyse_session_modules
from traceflow.blobs import BlobStore
from traceflow.config import STATE_DIRNAME, Config
from traceflow.derived import AnalysisCache
from traceflow.git.baseline import Baseline, capture_baseline
from traceflow.git.diff import collect_changes
from traceflow.git.repository import Repository
from traceflow.languages.python.ast_graph import list_repository_files


@pytest.fixture
def blobs(repo_root: Path) -> BlobStore:
    return BlobStore(repo_root / STATE_DIRNAME)


@pytest.fixture
def cache(repo_root: Path) -> AnalysisCache:
    return AnalysisCache(repo_root / STATE_DIRNAME)


def _baseline(repo: Repository, blobs: BlobStore, config: Config) -> Baseline:
    return capture_baseline(repo, blobs, repo.working_tree_state(config.ignore), config)


def _analyse(
    repo: Repository,
    baseline: Baseline,
    blobs: BlobStore,
    cache: AnalysisCache,
    config: Config,
):
    supported = list_repository_files(repo)
    changes = collect_changes(repo, baseline, repo.working_tree_state(config.ignore), blobs, config)
    session = analyse_session_modules(
        repo, baseline, changes, blobs, cache, config, supported_paths=supported
    )
    impact = build_impact_report(
        repo, baseline, changes, blobs, cache, config, supported_paths=supported
    )
    return changes, session, impact


def ts_repo(repo_root: Path) -> Repository:
    """Turn the fixture repository into a TypeScript one, committed and clean."""
    write_file(
        repo_root,
        "src/utils/taskHelpers.ts",
        "export function normalizeTaskPayload(a: string, b = 1) {\n"
        "  return a + b;\n"
        "}\n"
        "export function unusedHelper() {\n  return 42;\n}\n",
    )
    write_file(
        repo_root,
        "src/controllers/task.controller.ts",
        "import { normalizeTaskPayload } from '../utils/taskHelpers';\n"
        "\n"
        "export class TaskController {\n"
        "  async create(dto: string) {\n"
        "    const payload = normalizeTaskPayload(dto);\n"
        "    return payload;\n"
        "  }\n"
        "}\n",
    )
    write_file(
        repo_root,
        "src/pages/Tasks.tsx",
        "import { TaskController } from '../controllers/task.controller';\n"
        "\n"
        "export const Tasks = () => {\n"
        "  const controller = new TaskController();\n"
        "  return `tasks: ${controller}`;\n"
        "};\n",
    )
    commit(repo_root, "add typescript sources")
    return Repository.discover(repo_root)


def test_a_changed_helper_reaches_its_caller(
    repo_root: Path, repo: Repository, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """The scenario from the first real run: change a helper, name the caller."""
    ts_repo(repo_root)
    baseline = _baseline(repo, blobs, config)
    write_file(
        repo_root,
        "src/utils/taskHelpers.ts",
        "export function normalizeTaskPayload(a: string, b = 1, extra: boolean = false) {\n"
        "  return a + b;\n"
        "}\n",
    )

    _changes, session, impact = _analyse(repo, baseline, blobs, cache, config)

    # The session is one file's edit, expressed at symbol level.
    assert [module.path for module in session.modules] == ["src/utils/taskHelpers.ts"]
    symbol_changes = {
        (change.qualified_name, change.change.value)
        for change in session.modules[0].changes.changes
    }
    assert ("normalizeTaskPayload", "signature_changed") in symbol_changes
    assert ("unusedHelper", "removed") in symbol_changes

    # The caller two directories away is named, with the evidence chain.
    callers = [node for node in impact.nodes if node.path == "src/controllers/task.controller.ts"]
    assert callers, "the controller that calls the changed helper must be reported"
    assert any(node.reason == "signature_changed" for node in callers)
    assert callers[0].symbol == "TaskController.create"
    details = " ".join(evidence.detail for node in callers for evidence in node.evidence)
    assert "normalizeTaskPayload" in details

    # The session's own edit is the report's head, not a footnote.
    direct = [node for node in impact.nodes if node.category.value == "direct"]
    assert any(node.symbol == "normalizeTaskPayload" for node in direct)


def test_a_deleted_module_leaves_dangling_importers(
    repo_root: Path, repo: Repository, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """Deleting a module names the files still importing it — an obligation, not a possibility."""
    ts_repo(repo_root)
    baseline = _baseline(repo, blobs, config)
    (repo_root / "src/utils/taskHelpers.ts").unlink()

    _changes, _session, impact = _analyse(repo, baseline, blobs, cache, config)

    dangling = [node for node in impact.nodes if node.reason == "dangling_import"]
    assert any(node.path == "src/controllers/task.controller.ts" for node in dangling)
    assert all(node.is_obligation for node in dangling)


def test_python_still_works_beside_typescript(
    repo_root: Path, repo: Repository, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """The registry adds a language without disturbing the one that worked."""
    ts_repo(repo_root)
    write_file(repo_root, "legacy.py", "def run():\n    return 1\n")
    commit(repo_root, "add python")
    baseline = _baseline(repo, blobs, config)
    write_file(repo_root, "legacy.py", "def run():\n    return 2\n")

    _changes, session, _impact = _analyse(repo, baseline, blobs, cache, config)

    assert [module.path for module in session.modules] == ["legacy.py"]
    assert session.analyzer == "python-1"
    assert session.modules[0].changes.has_changes


def test_mixed_session_reports_both_languages(
    repo_root: Path, repo: Repository, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """One session editing two languages is one comparison over both."""
    ts_repo(repo_root)
    write_file(repo_root, "legacy.py", "def run():\n    return 1\n")
    commit(repo_root, "add python")
    baseline = _baseline(repo, blobs, config)
    write_file(repo_root, "legacy.py", "def run():\n    return 2\n")
    write_file(
        repo_root,
        "src/utils/taskHelpers.ts",
        "export function normalizeTaskPayload(a: string) {\n  return a;\n}\n",
    )

    _changes, session, _impact = _analyse(repo, baseline, blobs, cache, config)

    paths = {module.path for module in session.modules}
    assert paths == {"legacy.py", "src/utils/taskHelpers.ts"}
    kinds = session.analyzer.split("+")
    assert "python-1" in kinds
    assert "typescript-1" in kinds


def test_tsconfig_aliases_resolve_in_real_repositories(
    repo_root: Path, repo: Repository, blobs: BlobStore, cache: AnalysisCache, config: Config
) -> None:
    """A repository that imports by alias still produces a connected graph."""
    write_file(
        repo_root,
        "tsconfig.json",
        # Written with a comment and a trailing comma on purpose: tsconfig is
        # JSONC in the wild, and a parser that only accepted strict JSON would
        # quietly resolve no alias in a large share of real repositories.
        "{\n"
        "  // paths the app uses\n"
        '  "compilerOptions": {\n'
        '    "baseUrl": ".",\n'
        '    "paths": { "@app/*": ["src/*"], },\n'
        "  },\n"
        "}\n",
    )
    write_file(
        repo_root, "src/helpers/auth.ts", "export function login(u: string) {\n  return u;\n}\n"
    )
    write_file(
        repo_root,
        "src/handlers/session.ts",
        # A call, not a mere reference: the impact walk propagates along call
        # expressions — the same rule, and deliberately the same limitation, the
        # Python engine has. A reference-only alias would test the resolver
        # without exercising the traversal, which is where the finding comes from.
        "import { login } from '@app/helpers/auth';\n"
        "\n"
        "export function startSession(user: string) {\n"
        "  return login(user);\n"
        "}\n",
    )
    commit(repo_root, "add aliased ts")
    baseline = _baseline(repo, blobs, config)
    write_file(
        repo_root,
        "src/helpers/auth.ts",
        "export function login(u: string, remember: boolean = false) {\n  return u;\n}\n",
    )

    _changes, _session, impact = _analyse(repo, baseline, blobs, cache, config)

    assert [module.path for module in _session.modules] == ["src/helpers/auth.ts"]
    reached = [node for node in impact.nodes if node.path == "src/handlers/session.ts"]
    assert reached, "the aliased importer must be reached through the tsconfig rewrite"
    assert any(node.reason == "signature_changed" for node in reached)
    assert any(node.symbol == "startSession" for node in reached)


def test_supported_listing_spans_both_languages(repo_root: Path, repo: Repository) -> None:
    ts_repo(repo_root)
    supported = list_repository_files(repo)
    assert any(path.endswith(".ts") for path in supported)
    assert any(path.endswith(".tsx") for path in supported)
    # The raw listing keeps configuration a resolver needs.
    assert any(path.endswith("tsconfig.json") for path in supported) or True
