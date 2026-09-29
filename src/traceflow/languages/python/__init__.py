"""Python language support.

``symbols`` turns a syntax tree into symbols and fingerprints, ``analyzer`` turns a
file into an analysis, and ``ast_graph`` turns a repository into a dependency graph.
"""

from traceflow.languages.python.analyzer import (
    ANALYZER_VERSION,
    PythonAnalyzer,
    analyze_python,
)
from traceflow.languages.python.ast_graph import (
    DependencyGraph,
    ImportEdge,
    ModuleNode,
    PythonFiles,
    UnresolvedImport,
    build_dependency_graph,
    build_module_index,
    is_package_init,
    list_python_files,
    module_names_for,
    package_directories,
    resolve_import,
)
from traceflow.languages.python.symbols import extract_symbols

__all__ = [
    "ANALYZER_VERSION",
    "DependencyGraph",
    "ImportEdge",
    "ModuleNode",
    "PythonAnalyzer",
    "PythonFiles",
    "UnresolvedImport",
    "analyze_python",
    "build_dependency_graph",
    "build_module_index",
    "extract_symbols",
    "is_package_init",
    "list_python_files",
    "module_names_for",
    "package_directories",
    "resolve_import",
]
