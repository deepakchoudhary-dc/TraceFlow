"""The vocabulary of analysis (plan.md §19, §28, §33, §34, §42).

The types here are what every part of TraceFlow that *reports* something speaks in. They
are deliberately separate from the engine that produces them (:mod:`traceflow.analysis.impact`)
for the same reason the analyzer's models are separate from the parser: the command line,
the session artifacts and the coming UI all need to read an impact report, and none of
them should have to import a graph traversal to do it.

Two properties are load-bearing.

**Confidence is categorical.** plan.md §34 forbids invented precision such as "87.42%
confident", and it is right to: no model stands behind such a number. The categories name
the *kind* of evidence instead, which is a claim that can be checked.

**Every claim carries evidence.** plan.md §33 requires that TraceFlow be able to answer
"why do you think these two components are connected?" A relationship without a location
in the source is an assertion, and assertions are what this product exists to replace.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from traceflow.languages.python.ast_graph import normalise


class ImpactCategory(str, Enum):
    """How a component is touched by the session (plan.md §19).

    TEST is a category rather than a flag because the dashboard treats tests as their own
    section. A test that *changed* and a test that merely *depends on* a change are both
    TEST; the ``reason`` distinguishes them and ``distance`` tells you whether it was the
    session's own edit or a consequence of one.
    """

    DIRECT = "direct"
    INDIRECT = "indirect"
    TEST = "test"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    POTENTIAL = "potential"


class Confidence(str, Enum):
    """Categorical confidence (plan.md §34)."""

    CONFIRMED = "confirmed"
    """An explicit statement in the source — an import, a call, a definition."""

    HIGH_CONFIDENCE = "high_confidence"
    """Resolved by a rule that holds unless the name is shadowed at runtime."""

    INFERRED = "inferred"
    """The file was reached with certainty; which symbol was meant was not."""

    POSSIBLE = "possible"
    """A name match only. Reported so it can be dismissed, never as a guarantee."""

    UNKNOWN = "unknown"


#: Reasons that are obligations rather than possibilities — the nodes a reviewer actually
#: has to look at. Everything else is a consequence worth knowing about, not a task.
REVIEW_REASONS = frozenset(
    {
        "dangling_import",
        "file_deleted",
        "signature_changed",
        "symbol_removed",
        "symbol_signature_changed",
    }
)

#: How categories and confidences are ordered for display: obligations before
#: possibilities, and certainty before doubt. Exported because the ordering is part of how
#: a report reads, not an implementation detail of one caller.
CATEGORY_ORDER: dict[ImpactCategory, int] = {
    ImpactCategory.DIRECT: 0,
    ImpactCategory.INDIRECT: 1,
    ImpactCategory.TEST: 2,
    ImpactCategory.CONFIGURATION: 3,
    ImpactCategory.DEPENDENCY: 4,
    ImpactCategory.POTENTIAL: 5,
}

CONFIDENCE_ORDER: dict[Confidence, int] = {
    Confidence.CONFIRMED: 0,
    Confidence.HIGH_CONFIDENCE: 1,
    Confidence.INFERRED: 2,
    Confidence.POSSIBLE: 3,
    Confidence.UNKNOWN: 4,
}

_TEST_DIRECTORIES = frozenset({"test", "tests"})

_DEPENDENCY_MANIFESTS = frozenset(
    {
        "cargo.toml",
        "gemfile",
        "go.mod",
        "package-lock.json",
        "package.json",
        "pipfile",
        "pipfile.lock",
        "poetry.lock",
        "pyproject.toml",
        "requirements.txt",
        "setup.cfg",
        "setup.py",
    }
)

_CONFIGURATION_FILES = frozenset(
    {
        ".editorconfig",
        ".gitignore",
        ".traceflow.toml",
        "dockerfile",
        "mypy.ini",
        "pytest.ini",
        "ruff.toml",
        "tox.ini",
    }
)

_CONFIGURATION_SUFFIXES = (".cfg", ".ini", ".toml", ".yaml", ".yml")


@dataclass(frozen=True)
class Evidence:
    """One reason to believe a claim (plan.md §33)."""

    kind: str
    """``symbol_change``, ``call_expression``, ``import_statement`` or ``file_change``."""

    path: str
    line: int | None
    detail: str

    def render(self) -> str:
        location = f"{self.path}:{self.line}" if self.line else self.path
        return f"{location} {self.detail}"

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "path": self.path, "line": self.line, "detail": self.detail}


@dataclass(frozen=True)
class Reach:
    """One way a symbol was reached by the walk (plan.md §19, §33).

    A symbol is one thing, so the report says everything true about it on the one node
    that represents it. But a session can do two things to the same symbol — change it,
    *and* reach it because something it calls moved — and those are two findings with
    their own reason, their own distance, their own path through the graph and their own
    evidence. Keeping only the first was how a session with three call sites of a moved
    declaration came to report one caller.
    """

    reason: str
    category: ImpactCategory
    distance: int
    chain: tuple[str, ...]
    confidence: Confidence = Confidence.CONFIRMED
    evidence: tuple[Evidence, ...] = ()

    @property
    def is_obligation(self) -> bool:
        return self.reason in REVIEW_REASONS

    @property
    def root(self) -> str:
        """The file whose change reached here — where this finding hangs in the delivery."""
        return self.chain[0] if self.chain else ""

    def render(self) -> str:
        return f"{self.reason} ({self.confidence.value}) at depth {self.distance}"

    def to_json(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "category": self.category.value,
            "distance": self.distance,
            "chain": list(self.chain),
            "confidence": self.confidence.value,
            "evidence": [item.to_json() for item in self.evidence],
        }


@dataclass(frozen=True)
class ImpactedNode:
    """One component the session reached, and why."""

    path: str
    symbol: str | None
    """Qualified name of the affected symbol, or ``None`` when the module itself is
    affected — a module-level call, or a file the analyzer does not parse."""

    category: ImpactCategory
    confidence: Confidence
    distance: int
    """0 for the changed component itself, 1 for something directly dependent on it, and
    upwards for the transitive rings."""

    reason: str
    chain: tuple[str, ...]
    """The path the traversal took to get here, root first. This is what makes a
    transitive result explainable rather than merely asserted."""

    evidence: tuple[Evidence, ...] = ()

    reaches: tuple[Reach, ...] = ()
    """Further ways the same symbol was recorded, beyond the one above.

    Empty for most nodes. It is populated when a session both changed a symbol and reached
    it: the direct change is the node's own reason, and the obligation is kept here rather
    than dropped or turned into a second row for the same symbol.
    """

    @property
    def is_obligation(self) -> bool:
        return self.reason in REVIEW_REASONS or any(item.is_obligation for item in self.reaches)

    @property
    def obligation_reasons(self) -> tuple[str, ...]:
        """Every reason recorded here that is an obligation, the node's own first."""
        found = [self.reason] if self.reason in REVIEW_REASONS else []
        found.extend(item.reason for item in self.reaches if item.is_obligation)
        return tuple(found)

    @property
    def reasons(self) -> tuple[str, ...]:
        """Every reason recorded here, the node's own first."""
        return (self.reason, *(item.reason for item in self.reaches))

    @property
    def chains(self) -> tuple[tuple[str, ...], ...]:
        """Every path the walk took to this symbol, the node's own first."""
        return (self.chain, *(item.chain for item in self.reaches))

    def render(self) -> str:
        subject = self.path if self.symbol is None else f"{self.path}::{self.symbol}"
        extra = "".join(f" + {item.render()}" for item in self.reaches)
        return f"{subject} [{self.category.value}] {self.reason} ({self.confidence.value}){extra}"

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "symbol": self.symbol,
            "category": self.category.value,
            "confidence": self.confidence.value,
            "distance": self.distance,
            "reason": self.reason,
            "chain": list(self.chain),
            "evidence": [item.to_json() for item in self.evidence],
            "reaches": [item.to_json() for item in self.reaches],
        }


@dataclass(frozen=True)
class EdgeChange:
    """One dependency relationship the session added or removed (plan.md §28)."""

    source_path: str
    target_path: str
    module: str
    line: int

    def render(self) -> str:
        return f"{self.source_path} -> {self.target_path}"

    def to_json(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "target_path": self.target_path,
            "module": self.module,
            "line": self.line,
        }


@dataclass(frozen=True)
class GraphDiff:
    """The structural before/after comparison (plan.md §28).

    Restricted to relationships leaving files the session touched. An import can only
    appear or disappear because the file declaring it changed, so diffing the changed
    files' imports is the complete answer — and it avoids parsing the whole repository a
    second time at the baseline revision.
    """

    modules_added: tuple[str, ...] = ()
    modules_removed: tuple[str, ...] = ()
    edges_added: tuple[EdgeChange, ...] = ()
    edges_removed: tuple[EdgeChange, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (
            self.modules_added or self.modules_removed or self.edges_added or self.edges_removed
        )

    def render(self) -> str:
        return ", ".join(
            (
                f"+{len(self.modules_added)} module(s)",
                f"-{len(self.modules_removed)} module(s)",
                f"+{len(self.edges_added)} relationship(s)",
                f"-{len(self.edges_removed)} relationship(s)",
            )
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "modules_added": list(self.modules_added),
            "modules_removed": list(self.modules_removed),
            "edges_added": [item.to_json() for item in self.edges_added],
            "edges_removed": [item.to_json() for item in self.edges_removed],
        }


@dataclass(frozen=True)
class ImpactReport:
    """What a session changed, what that reaches, and where the analysis is blind."""

    analyzer: str
    changed_files: tuple[str, ...] = ()
    nodes: tuple[ImpactedNode, ...] = ()
    graph_diff: GraphDiff = GraphDiff()
    limitations: tuple[str, ...] = ()
    unresolved_imports: int = 0
    external_modules: tuple[str, ...] = ()
    """The distinct dotted modules this repository imports but does not contain.

    The count alone could not answer plan.md §74's ``DEPENDENCIES + tenacity``, and a
    number with no names behind it is not evidence of anything.
    """

    max_depth: int = 0
    """The deepest ring actually reached, not the configured limit."""

    truncated: bool = False
    """True when the configured depth limit stopped the walk, so the report is knowingly
    incomplete rather than complete and small."""

    def by_category(self, category: ImpactCategory) -> tuple[ImpactedNode, ...]:
        return tuple(node for node in self.nodes if node.category is category)

    def by_reason(self, reason: str) -> tuple[ImpactedNode, ...]:
        return tuple(node for node in self.nodes if node.reason == reason)

    @property
    def direct(self) -> tuple[ImpactedNode, ...]:
        return self.by_category(ImpactCategory.DIRECT)

    @property
    def indirect(self) -> tuple[ImpactedNode, ...]:
        return self.by_category(ImpactCategory.INDIRECT)

    @property
    def tests(self) -> tuple[ImpactedNode, ...]:
        return self.by_category(ImpactCategory.TEST)

    @property
    def configuration(self) -> tuple[ImpactedNode, ...]:
        return self.by_category(ImpactCategory.CONFIGURATION)

    @property
    def dependencies(self) -> tuple[ImpactedNode, ...]:
        return self.by_category(ImpactCategory.DEPENDENCY)

    @property
    def potential(self) -> tuple[ImpactedNode, ...]:
        return self.by_category(ImpactCategory.POTENTIAL)

    @property
    def requiring_review(self) -> tuple[ImpactedNode, ...]:
        """Nodes whose reason is an obligation: a caller that must be re-examined, or a
        file left importing something the session removed."""
        return tuple(node for node in self.nodes if node.is_obligation)

    @property
    def impacted_files(self) -> tuple[str, ...]:
        return tuple(sorted({node.path for node in self.nodes}))

    def to_json(self) -> dict[str, Any]:
        return {
            "analyzer": self.analyzer,
            "max_depth": self.max_depth,
            "truncated": self.truncated,
            "unresolved_imports": self.unresolved_imports,
            "external_modules": list(self.external_modules),
            "totals": {
                "changed_files": len(self.changed_files),
                "nodes": len(self.nodes),
                "impacted_files": len(self.impacted_files),
                "direct": len(self.direct),
                "indirect": len(self.indirect),
                "tests": len(self.tests),
                "configuration": len(self.configuration),
                "dependencies": len(self.dependencies),
                "potential": len(self.potential),
                "requiring_review": len(self.requiring_review),
                "limitations": len(self.limitations),
            },
            "graph_diff": self.graph_diff.to_json(),
            "nodes": [node.to_json() for node in self.nodes],
            "limitations": list(self.limitations),
        }


# --------------------------------------------------------------------------- classification


def is_test_path(path: str) -> bool:
    """True when *path* is a test.

    Deliberately conservative. A directory literally named ``test`` or ``tests``, or a file
    named ``test_*.py`` / ``*_test.py`` / ``conftest.py``, is a test. Matching the substring
    "test" anywhere would classify ``latest/`` as a test directory and turn the TESTS
    section into noise.
    """
    normalised = normalise(path)
    name = normalised.rsplit("/", 1)[-1]
    if name == "conftest.py" or name.startswith("test_") or name.endswith("_test.py"):
        return True
    return any(part in _TEST_DIRECTORIES for part in normalised.split("/")[:-1])


def is_dependency_manifest(path: str) -> bool:
    """True when *path* declares the project's external dependencies (plan.md §24)."""
    return normalise(path).rsplit("/", 1)[-1].lower() in _DEPENDENCY_MANIFESTS


def is_configuration_file(path: str) -> bool:
    """True when *path* configures the project rather than being part of it (plan.md §25)."""
    name = normalise(path).rsplit("/", 1)[-1].lower()
    return name in _CONFIGURATION_FILES or name.endswith(_CONFIGURATION_SUFFIXES)


def classify_file(path: str) -> ImpactCategory | None:
    """The category a changed or impacted file belongs to, or ``None`` for source code.

    Test first: a changed ``tests/test_auth.py`` is more usefully reported as a test than as
    a direct change, because that is the section a reviewer looks in.
    """
    if is_test_path(path):
        return ImpactCategory.TEST
    if is_dependency_manifest(path):
        return ImpactCategory.DEPENDENCY
    if is_configuration_file(path):
        return ImpactCategory.CONFIGURATION
    return None
