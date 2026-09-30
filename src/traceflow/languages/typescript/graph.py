"""Repository-level dependency graph for TypeScript and JavaScript (plan.md §66).

The mirror of :mod:`traceflow.languages.python.ast_graph`: given the TS/JS files in a
repository, work out which file each import specifier reaches, and record an edge for
every hit. The Python graph resolves dotted module names; this one resolves *specifiers*
— ``./helper``, ``@app/utils`` rewritten by tsconfig ``paths``, ``@icons/heart.svg``
resolved against a ``baseUrl`` — which is how imports are actually written in the wild.

One git call lists the repository; everything after that is string work. Like the
Python resolver, this resolver is honest about its limits: a specifier that reaches
nothing in the repository is left unresolved rather than dropped, because ``react`` and
``../deleted-helper`` both fail to resolve and only one of them is a problem (plan.md
§69).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from traceflow.git.repository import Repository
from traceflow.languages.typescript.scanner import SourceView

#: Every suffix a resolved import may legitimately land on. ``.json`` is included:
#: ``import data from './fixtures.json'`` is a real edge, and dropping the target
#: would report a dependency the program makes at runtime.
TYPESCRIPT_SUFFIXES = (
    ".ts",
    ".tsx",
    ".mts",
    ".cts",
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",
    ".json",
)

TS_FILE_SUFFIXES = (".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs")

_RESOLUTION_SUFFIXES = (*TS_FILE_SUFFIXES, ".json")

_TSCONFIG_FILENAME = "tsconfig.json"

_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def module_name_for(path: str) -> str:
    """The file's name as importers see it: the path without its extension.

    Symbol reports and intent facts display this; unlike Python there is no package
    tree to fold, so the path itself is the module identity.
    """
    normalised = path.replace("\\", "/")
    for suffix in _RESOLUTION_SUFFIXES:
        if normalised.endswith(suffix):
            return normalised[: -len(suffix)]
    return normalised


def _extensionless(path: str) -> str:
    return module_name_for(path)


def _index_candidates(base: str) -> tuple[str, ...]:
    """Every concrete path a *directory-style* import could mean."""
    return tuple(f"{base}/index{suffix}" for suffix in _RESOLUTION_SUFFIXES)


def _file_candidates(base: str) -> tuple[str, ...]:
    """Every concrete path a bare extensionless import could mean.

    The exact string is tried first so a specifier with an explicit suffix resolves
    in one step, then the standard suffixes in TypeScript's own preference order.
    """
    return (base, *(f"{base}{suffix}" for suffix in _RESOLUTION_SUFFIXES))


def _resolve_against_index(base: str, index: dict[str, str]) -> str | None:
    """Try a base path as a file, then as a directory with an index."""
    for candidate in _file_candidates(base):
        target = index.get(candidate)
        if target is not None:
            return target
    for candidate in _index_candidates(base):
        target = index.get(candidate)
        if target is not None:
            return target
    return None


def build_ts_index(paths: tuple[str, ...]) -> dict[str, str]:
    """Map every extensionless path to the file that provides it."""
    index: dict[str, str] = {}
    for path in sorted(paths):
        normalised = path.replace("\\", "/")
        index.setdefault(_extensionless(normalised), normalised)
        # The exact path is also a key, so a specifier that spells out '.ts'
        # resolves without the candidate walk.
        index.setdefault(normalised, normalised)
    return index


@dataclass(frozen=True)
class PathAlias:
    """One ``compilerOptions.paths`` entry, prepared for matching."""

    pattern: str
    """The key as written: ``@app/*``, ``@icons`` or ``#utils/*.``"""

    prefix: str
    """The part before ``*`` (the whole key when there is no star)."""

    suffix: str
    """The part after ``*`` (empty when there is no star)."""

    targets: tuple[str, ...]
    """The rewrite targets as written, each containing one ``*`` (or not)."""

    @property
    def has_wildcard(self) -> bool:
        return "*" in self.pattern


def _strip_jsonc(text: str) -> str:
    """Remove comments and trailing commas from a tsconfig's text.

    tsconfig is JSON *with comments*, so plain ``json.loads`` refuses it. The same
    scanner the analyzer uses marks comments and strings; comment bytes are blanked
    and string contents are kept verbatim, which leaves a document the json module
    accepts after the trailing commas go.
    """
    view = SourceView(text)
    out: list[str] = []
    for position, char in enumerate(text):
        if view.in_code(position):
            out.append(char)
        elif view.in_comment(position):
            out.append(" " if char != "\n" else "\n")
        else:
            # Inside a string or template: keep everything, quotes included.
            out.append(char)
    return _TRAILING_COMMA_RE.sub(r"\1", "".join(out))


def _parse_paths_table(table: object) -> tuple[PathAlias, ...]:
    if not isinstance(table, dict):
        return ()
    aliases: list[PathAlias] = []
    for pattern, targets in table.items():
        if not isinstance(pattern, str) or not isinstance(targets, list):
            continue
        written = tuple(item for item in targets if isinstance(item, str))
        if not written:
            continue
        star = pattern.find("*")
        if star == -1:
            aliases.append(PathAlias(pattern=pattern, prefix=pattern, suffix="", targets=written))
        else:
            aliases.append(
                PathAlias(
                    pattern=pattern,
                    prefix=pattern[:star],
                    suffix=pattern[star + 1 :],
                    targets=written,
                )
            )
    # Longest prefix first: ``@app/utils/*`` must match before ``@app/*``.
    aliases.sort(key=lambda item: (len(item.prefix), len(item.pattern)), reverse=True)
    return tuple(aliases)


@dataclass(frozen=True)
class TypeScriptPaths:
    """The ``compilerOptions`` aliases every tsconfig in the repository declares.

    Merged rather than chosen: a monorepo legitimately declares different aliases in
    different tsconfigs (``frontend/tsconfig.json`` and ``server/tsconfig.json`` do
    not share a root), and an alias that is never used costs nothing. On a conflict
    the first declaration wins, which keeps the answer deterministic.
    """

    aliases: tuple[PathAlias, ...] = ()
    base_url: str | None = None

    @classmethod
    def from_payload(cls, options: dict[str, object]) -> TypeScriptPaths:
        """Build from a ``compilerOptions``-shaped mapping.

        The construction :meth:`discover` reads out of a tsconfig file; public so a
        caller (and the tests) can exercise alias resolution without writing one.
        """
        raw_base = options.get("baseUrl")
        return cls(
            aliases=_parse_paths_table(options.get("paths")),
            base_url=raw_base if isinstance(raw_base, str) else None,
        )

    @classmethod
    def discover(cls, paths: tuple[str, ...], repository: Repository) -> TypeScriptPaths:
        """Parse every tsconfig the listing shows, merging their alias tables."""
        aliases: list[PathAlias] = []
        base_url: str | None = None
        for path in sorted(paths):
            name = path.replace("\\", "/").rsplit("/", 1)[-1]
            if name != _TSCONFIG_FILENAME:
                continue
            absolute = repository.root / path
            try:
                text = absolute.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            try:
                payload = json.loads(_strip_jsonc(text))
            except json.JSONDecodeError:
                # A tsconfig mid-edit is a normal thing to meet; other tsconfigs
                # still contribute, and the next session reparses.
                continue
            if not isinstance(payload, dict):
                continue
            options = payload.get("compilerOptions")
            if not isinstance(options, dict):
                continue
            for alias in _parse_paths_table(options.get("paths")):
                if not any(existing.pattern == alias.pattern for existing in aliases):
                    aliases.append(alias)
            if base_url is None:
                raw_base = options.get("baseUrl")
                if isinstance(raw_base, str) and raw_base:
                    base_url = raw_base
        return cls(aliases=tuple(aliases), base_url=base_url)

    def rewritten(self, specifier: str) -> tuple[str, ...]:
        """The concrete base paths an alias-carrying specifier could mean.

        An exact key matches whole; a wildcard key matches by prefix and substitutes
        the tail into each target. A specifier no key claims returns nothing, which
        sends resolution on to the ordinary bare-module path.
        """
        found: list[str] = []
        for alias in self.aliases:
            if alias.has_wildcard:
                if not (specifier.startswith(alias.prefix) and specifier.endswith(alias.suffix)):
                    continue
                tail = specifier[len(alias.prefix) : len(specifier) - len(alias.suffix)]
                for target in alias.targets:
                    found.append(target.replace("*", tail))
            else:
                if specifier != alias.pattern:
                    continue
                found.extend(alias.targets)
        return tuple(found)

    def base_url_candidates(self, specifier: str) -> tuple[str, ...]:
        """The base path a ``baseUrl``-rooted import could mean."""
        if not self.base_url:
            return ()
        base = PurePosixPath(self.base_url.replace("\\", "/")) / specifier
        return (base.as_posix(),)


@dataclass(frozen=True)
class TypeScriptFiles:
    """The repository's TS/JS/JSON files and the index specifier resolution needs."""

    paths: tuple[str, ...]
    index: dict[str, str]
    tsconfig: TypeScriptPaths

    @classmethod
    def of(
        cls, paths: tuple[str, ...], repository: Repository, *, include_missing: bool = False
    ) -> TypeScriptFiles:
        """Build from an already-listed repository: the listing is one git call, and
        a command that runs several passes over one tree should make it once.

        ``include_missing`` keeps files that no longer exist on disk, which is what
        re-resolving a deleted module's importers needs: the whole point of that
        check is that the target is *gone*, so an existence filter would delete the
        evidence before the question was asked.
        """
        normalised = tuple(path.replace("\\", "/") for path in paths)
        supported = tuple(
            path
            for path in normalised
            if path.endswith(TYPESCRIPT_SUFFIXES)
            and (include_missing or (repository.root / path).is_file())
        )
        return cls(
            paths=supported,
            index=build_ts_index(supported),
            tsconfig=TypeScriptPaths.discover(supported, repository),
        )


def resolve_specifier(
    specifier: str,
    importing_path: str,
    index: dict[str, str],
    tsconfig: TypeScriptPaths,
) -> str | None:
    """The repository file an import specifier reaches, or ``None`` when external.

    Relative (``./x``, ``../x``) and aliased (``@app/x``) specifiers resolve against
    the repository. A bare specifier (``react``, ``express``) resolves only through
    tsconfig aliases or ``baseUrl``; otherwise it is external, and unresolved is the
    honest record for it.
    """
    normalised = specifier.replace("\\", "/").strip()
    if not normalised:
        return None

    if normalised.startswith("."):
        directory = PurePosixPath(importing_path.replace("\\", "/")).parent
        base = (directory / normalised).as_posix()
        # ``./a/b/..`` shapes normalise away; ``PurePosixPath`` does not fold them.
        base = _fold_dots(base)
        return _resolve_against_index(base, index)

    for base in (*tsconfig.rewritten(normalised), *tsconfig.base_url_candidates(normalised)):
        target = _resolve_against_index(_fold_dots(base), index)
        if target is not None:
            return target

    if normalised.startswith("#"):
        # A package ``imports`` entry (``#utils/x``): not an npm name, and worth
        # trying against the repository root before giving up.
        return _resolve_against_index(_fold_dots(normalised.lstrip("#")), index)
    return None


def _fold_dots(path: str) -> str:
    """Collapse ``a/./b`` and ``a/c/../b`` to ``a/b`` without touching the drive."""
    parts: list[str] = []
    for part in path.replace("\\", "/").split("/"):
        if part in ("", "."):
            continue
        if part == ".." and parts and parts[-1] not in ("..", ""):
            parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)
