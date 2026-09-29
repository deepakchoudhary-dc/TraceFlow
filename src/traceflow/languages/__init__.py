"""Language analyzers (plan.md §6).

plan.md is explicit that Python's AST cannot analyse every language, and that the
architecture must allow further analyzers without pretending otherwise. This package
is that seam: :mod:`traceflow.languages.base` defines what an analyzer must provide,
and each subpackage implements it for one language.
"""

from traceflow.languages.base import (
    CallRef,
    ImportRef,
    LanguageAnalyzer,
    ModuleAnalysis,
    ModuleSymbolChanges,
    Symbol,
    SymbolChange,
    SymbolChangeKind,
    SymbolKind,
    diff_module_analysis,
    render_import,
)

__all__ = [
    "CallRef",
    "ImportRef",
    "LanguageAnalyzer",
    "ModuleAnalysis",
    "ModuleSymbolChanges",
    "Symbol",
    "SymbolChange",
    "SymbolChangeKind",
    "SymbolKind",
    "diff_module_analysis",
    "render_import",
]
