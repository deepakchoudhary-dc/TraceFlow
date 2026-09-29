"""Python source analysis (plan.md §59).

Parsing is deliberately forgiving. A file that cannot be parsed produces an analysis
recording the error and no symbols, never an exception: TraceFlow watches repositories
while they are being edited, so a half-written file is a normal thing to encounter
rather than a failure (plan.md §46).
"""

from __future__ import annotations

import ast

from traceflow.blobs import digest_of
from traceflow.languages.base import CallRef, ImportRef, ModuleAnalysis
from traceflow.languages.python.symbols import extract_symbols

#: Bumped whenever the analysis output changes shape or meaning. It forms part of the
#: cache key, so an improvement to the parser invalidates what the previous one
#: produced instead of being silently shadowed by it.
ANALYZER_VERSION = "1"

_MAX_PARSE_ERROR_LENGTH = 300


def _imports_from(tree: ast.Module) -> tuple[ImportRef, ...]:
    """Collect every import, including those inside functions.

    A deferred import is still a real dependency, so the graph should see it.
    """
    imports: list[ImportRef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    ImportRef(
                        module=alias.name,
                        name=None,
                        alias=alias.asname,
                        level=0,
                        line=node.lineno,
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(
                    ImportRef(
                        module=module,
                        name=alias.name,
                        alias=alias.asname,
                        level=node.level,
                        line=node.lineno,
                    )
                )
    imports.sort(key=lambda item: (item.line, item.level, item.module, item.name or ""))
    return tuple(imports)


def _call_name(node: ast.Call) -> str | None:
    """Render the callee as written, or ``None`` when it is not a plain name chain.

    ``service.authenticate(user)`` yields ``service.authenticate``. A call through a
    subscript or a lambda has no static name and is reported as nothing rather than
    guessed at.
    """
    parts: list[str] = []
    target: ast.expr = node.func
    while isinstance(target, ast.Attribute):
        parts.append(target.attr)
        target = target.value
    if not isinstance(target, ast.Name):
        return None
    parts.append(target.id)
    return ".".join(reversed(parts))


def _calls_from(tree: ast.Module) -> tuple[CallRef, ...]:
    calls: list[CallRef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name is not None:
                calls.append(CallRef(name=name, line=node.lineno))
    calls.sort(key=lambda item: (item.line, item.name))
    return tuple(calls)


class PythonAnalyzer:
    """Analyses Python source with the standard library's :mod:`ast`."""

    name = "python"
    version = ANALYZER_VERSION
    extensions = (".py", ".pyi")

    @property
    def cache_kind(self) -> str:
        """The cache namespace for this analyzer and version."""
        return f"{self.name}-{self.version}"

    def can_analyze(self, path: str) -> bool:
        return path.endswith(self.extensions)

    def analyze(self, path: str, source: bytes, module_name: str | None) -> ModuleAnalysis:
        digest = digest_of(source)

        try:
            # Bytes are passed straight through so that a PEP 263 encoding
            # declaration is honoured rather than second-guessed.
            tree = ast.parse(source, filename=path)
        except (SyntaxError, ValueError, RecursionError) as exc:
            return ModuleAnalysis(
                path=path,
                digest=digest,
                module_name=module_name,
                parse_error=f"{type(exc).__name__}: {exc}"[:_MAX_PARSE_ERROR_LENGTH],
            )

        return ModuleAnalysis(
            path=path,
            digest=digest,
            module_name=module_name,
            imports=_imports_from(tree),
            symbols=extract_symbols(tree),
            calls=_calls_from(tree),
        )


def analyze_python(path: str, source: bytes, module_name: str | None = None) -> ModuleAnalysis:
    """Convenience wrapper for the common single-file case."""
    return PythonAnalyzer().analyze(path, source, module_name)
