"""Analysis that goes beyond files (plan.md §16, §18).

The git evidence layer establishes *which files* changed. This package establishes
what changed *inside* them, and — in later phases — what that implies for the rest of
the repository.
"""

from traceflow.analysis.symbols import SymbolReport, collect_symbol_changes

__all__ = ["SymbolReport", "collect_symbol_changes"]
