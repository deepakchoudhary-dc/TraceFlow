"""Which analyzer handles which file (plan.md §6, §66).

The registry is the whole of the language-plugin architecture, and it is small on
purpose. Every part of the pipeline that used to ask "is this a Python file?" asks
here instead, so adding a language means writing the language's own package and
registering it in :func:`default_registry` — and touching nothing else.

The analyzers are stateless, so module-level instances are safe to share across a
watch loop and a command in the same process. ``analyzer_for`` is the single
dispatch point; ``both`` exists so a caller that will ask several times in one run
can materialise the registry once and pass it down, which is what the CLI does.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from traceflow.languages.base import ModuleAnalysis
from traceflow.languages.cfamily.analyzer import (
    CFamilyAnalyzer,
    csharp_profile,
    go_profile,
    java_profile,
    rust_profile,
)
from traceflow.languages.python.analyzer import PythonAnalyzer
from traceflow.languages.typescript.analyzer import TypeScriptAnalyzer

Analyzer = PythonAnalyzer | TypeScriptAnalyzer | CFamilyAnalyzer

#: Exported so a type annotation elsewhere can name the union without importing
#: both analyzers; ``ModuleAnalysis`` re-exported for the same reason.
__all__ = ["Analyzer", "ModuleAnalysis", "analyzer_for", "both", "default_registry"]


@runtime_checkable
class CacheKind(Protocol):
    """The one protocol every analyzer already satisfies beyond ``LanguageAnalyzer``."""

    @property
    def cache_kind(self) -> str: ...


def default_registry() -> tuple[Analyzer, ...]:
    """The languages the registry ships, in first-match order.

    Python and TypeScript are dedicated engines; Go, Java, Rust and C# share the
    profile-driven C-family engine (plan.md §66's post-Phase-10 list). The shipped
    extensions do not overlap, so the order is a formality — but a formality worth
    pinning down, because a future language claiming ``.js`` would otherwise
    silently steal files from an earlier one.
    """
    return (
        PythonAnalyzer(),
        TypeScriptAnalyzer(),
        CFamilyAnalyzer(go_profile()),
        CFamilyAnalyzer(java_profile()),
        CFamilyAnalyzer(rust_profile()),
        CFamilyAnalyzer(csharp_profile()),
    )


def both() -> tuple[Analyzer, ...]:
    """Alias of :func:`default_registry`, named for call-site readability."""
    return default_registry()


def analyzer_for(path: str, entries: tuple[Analyzer, ...] | None = None) -> Analyzer | None:
    """The analyzer that handles *path*, or ``None`` when no language claims it.

    ``None`` is a first-class answer, not a failure: a repository full of files no
    analyzer claims still gets correct file-level change tracking, and the impact
    report says what it could not look inside rather than pretending.
    """
    for entry in entries if entries is not None else default_registry():
        if entry.can_analyze(path):
            return entry
    return None
