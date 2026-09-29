"""Symbol extraction and fingerprinting for Python.

Two fingerprints are computed per symbol, and the distinction between them is the
reason this module exists:

* the **signature** covers how the symbol is declared — its name, parameters,
  annotations, decorators and base classes. If it changes, every caller has to be
  re-examined, because the way the symbol must be called has changed.
* the **body** covers what the symbol contains. If only that changes, callers are
  unaffected by construction even though behaviour may differ.

Conflating the two throws away the single most useful thing static analysis can say
about a change.

Both fingerprints are computed from the syntax tree rather than the source text, so
reformatting a file, reindenting a block, or editing a comment does not register as a
change. Only a change to the program's structure does.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import replace

from traceflow.languages.base import Symbol, SymbolKind

#: Definition nodes are excluded from a parent's body fingerprint, so a change to a
#: method is reported once — against the method — rather than again against the class
#: that contains it. Containment is already expressed by the symbol's ``parent``.
_NESTED_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

#: Block nodes that are not themselves statements. ``ast.ExceptHandler`` and
#: ``ast.match_case`` both hold a statement list but are not ``ast.stmt`` subclasses, so a
#: plain field walk would stop at the ``except`` and never see what is inside it.
_BLOCK_HOLDERS = (ast.ExceptHandler, ast.match_case)


def _hash(*pieces: str) -> str:
    return hashlib.sha256("\0".join(pieces).encode("utf-8")).hexdigest()


def _render_arg(argument: ast.arg, default: ast.expr | None) -> str:
    text = argument.arg
    if argument.annotation is not None:
        text += f": {ast.unparse(argument.annotation)}"
    if default is not None:
        text += f" = {ast.unparse(default)}"
    return text


def render_arguments(arguments: ast.arguments) -> str:
    """Render a parameter list canonically.

    Positional-only markers, keyword-only markers, ``*args`` and ``**kwargs`` are all
    preserved: each of them changes how the function may legally be called, so a
    change to any of them is a signature change even when the names are identical.
    """
    positional = [*arguments.posonlyargs, *arguments.args]
    padding = len(positional) - len(arguments.defaults)
    defaults: list[ast.expr | None] = [None] * padding + list(arguments.defaults)

    parts: list[str] = []
    for index, (argument, default) in enumerate(zip(positional, defaults, strict=True)):
        parts.append(_render_arg(argument, default))
        if arguments.posonlyargs and index == len(arguments.posonlyargs) - 1:
            parts.append("/")

    if arguments.vararg is not None:
        parts.append(f"*{_render_arg(arguments.vararg, None)}")
    elif arguments.kwonlyargs:
        parts.append("*")

    for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True):
        parts.append(_render_arg(argument, default))

    if arguments.kwarg is not None:
        parts.append(f"**{_render_arg(arguments.kwarg, None)}")

    return ", ".join(parts)


def render_class_bases(node: ast.ClassDef) -> str:
    """Render a class's bases and keyword arguments canonically."""
    parts = [ast.unparse(base) for base in node.bases]
    for keyword in node.keywords:
        rendered = ast.unparse(keyword.value)
        parts.append(f"{keyword.arg} = {rendered}" if keyword.arg else f"**{rendered}")
    return ", ".join(parts)


def render_decorators(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> tuple[str, ...]:
    """Render decorator expressions.

    Decorators belong to the signature because they change how a symbol is used — a
    method that becomes ``@property`` or ``@staticmethod`` is called completely
    differently while its parameter list stays identical.
    """
    return tuple(ast.unparse(decorator) for decorator in node.decorator_list)


def signature_fingerprint(
    kind: SymbolKind,
    name: str,
    *,
    signature: str = "",
    decorators: tuple[str, ...] = (),
    bases: tuple[str, ...] = (),
) -> str:
    """Hash a symbol's declaration."""
    return _hash(kind.value, name, signature, ",".join(decorators), ",".join(bases))


def body_fingerprint(body: list[ast.stmt]) -> str:
    """Hash a symbol's contents, excluding nested definitions.

    ``ast.dump`` is used rather than the source text, so whitespace, comments and line
    numbers are absent by construction and cannot produce a false change.
    """
    kept = [statement for statement in body if not isinstance(statement, _NESTED_DEFINITIONS)]
    return _hash(*(ast.dump(statement) for statement in kept))


def _function_symbol(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    qualified_name: str,
    parent: str | None,
    kind: SymbolKind,
) -> Symbol:
    signature = render_arguments(node.args)
    if node.returns is not None:
        signature = f"{signature} -> {ast.unparse(node.returns)}"
    decorators = render_decorators(node)

    return Symbol(
        qualified_name=qualified_name,
        name=node.name,
        kind=kind,
        line_start=node.lineno,
        line_end=getattr(node, "end_lineno", node.lineno),
        signature=signature,
        signature_fingerprint=signature_fingerprint(
            kind, node.name, signature=signature, decorators=decorators
        ),
        body_fingerprint=body_fingerprint(node.body),
        parent=parent,
        decorators=decorators,
    )


def _class_symbol(node: ast.ClassDef, qualified_name: str, parent: str | None) -> Symbol:
    bases = render_class_bases(node)
    decorators = render_decorators(node)
    rendered = f"({bases})" if bases else ""

    return Symbol(
        qualified_name=qualified_name,
        name=node.name,
        kind=SymbolKind.CLASS,
        line_start=node.lineno,
        line_end=getattr(node, "end_lineno", node.lineno),
        signature=rendered,
        signature_fingerprint=signature_fingerprint(
            SymbolKind.CLASS, node.name, signature=rendered, decorators=decorators, bases=(bases,)
        ),
        body_fingerprint=body_fingerprint(node.body),
        parent=parent,
        decorators=decorators,
        bases=tuple(ast.unparse(base) for base in node.bases),
    )


def _block_children(node: ast.AST) -> list[ast.stmt]:
    """The statements directly inside *node*, whatever kind of block it is.

    Used for every statement that is not itself a definition, so the walk descends into
    conditionals, loops, ``with`` and ``try`` — and into the handlers of the last two, which
    hold their bodies in nodes that are not statements at all.
    """
    found: list[ast.stmt] = []
    for _field, value in ast.iter_fields(node):
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, _BLOCK_HOLDERS):
                found.extend(_block_children(item))
            elif isinstance(item, ast.stmt):
                found.append(item)
    return found


def _walk(
    body: list[ast.stmt],
    parent: str | None,
    inside_class: bool,
    collected: list[Symbol],
) -> None:
    """Collect definitions, recursing into classes, functions and every other block.

    ``inside_class`` is reset when descending into a function body: a ``def`` nested
    inside a method is a plain function, not a method of the class that happens to
    contain the method.

    Descending into control flow matters as much as descending into a body. A definition
    inside an ``if``, a ``try`` or a loop is still a definition, and a change to it still
    changes how it is called; walking only direct statements made every one of them
    invisible, so a version shim yielded no symbols at all and a signature change to it was
    reported as nothing more than "the file changed".
    """
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualified = f"{parent}.{node.name}" if parent else node.name
            kind = SymbolKind.METHOD if inside_class else SymbolKind.FUNCTION
            collected.append(_function_symbol(node, qualified, parent, kind))
            _walk(node.body, qualified, False, collected)
        elif isinstance(node, ast.ClassDef):
            qualified = f"{parent}.{node.name}" if parent else node.name
            collected.append(_class_symbol(node, qualified, parent))
            _walk(node.body, qualified, True, collected)
        else:
            _walk(_block_children(node), parent, inside_class, collected)


def _fold(occurrences: list[Symbol]) -> Symbol:
    """One symbol for every definition sharing a qualified name.

    A name defined twice — the two arms of a version shim — cannot be two rows, because the
    diff is keyed by name and would keep only the last, leaving a change to the first arm
    invisible. So the fingerprints are *combined* rather than picked: a change to either arm
    changes the combined value. The rendered signature is combined too, because the evidence
    line shows it, and showing one arm while claiming the declaration moved would be a
    misleading citation.
    """
    if len(occurrences) == 1:
        return occurrences[0]

    first = occurrences[0]
    return replace(
        first,
        signature=" | ".join(dict.fromkeys(item.signature for item in occurrences)),
        signature_fingerprint=_hash(*(item.signature_fingerprint for item in occurrences)),
        body_fingerprint=_hash(*(item.body_fingerprint for item in occurrences)),
        line_end=max(item.line_end for item in occurrences),
        occurrences=len(occurrences),
    )


def extract_symbols(tree: ast.Module) -> tuple[Symbol, ...]:
    """Return every class, function and method defined in *tree*, in source order.

    One entry per qualified name, in the order the names are first defined.
    """
    collected: list[Symbol] = []
    _walk(tree.body, None, False, collected)

    order: list[str] = []
    grouped: dict[str, list[Symbol]] = {}
    for symbol in collected:
        if symbol.qualified_name not in grouped:
            order.append(symbol.qualified_name)
            grouped[symbol.qualified_name] = []
        grouped[symbol.qualified_name].append(symbol)

    return tuple(_fold(grouped[name]) for name in order)
