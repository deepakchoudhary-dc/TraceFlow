"""Repository-level import resolution for the C-family languages (plan.md §66).

The mirror of the TypeScript resolver, for import shapes that name *packages and
namespaces* rather than files: Go's ``import "github.com/user/repo/internal/auth"``,
Java's ``import com.example.auth.service.Service``, C#'s ``using App.Auth;``, Rust's
``use crate::auth::service::login;``.

Resolution is per-language because the four disagree about what an import path is
*relative to*:

* **Go** — a path under the module's own name (read from ``go.mod``) maps to the
  directory of the same relative shape; anything else is external.
* **Java / C#** — a dotted path maps to a file whose package/namespace declaration
  (read from the file's own head) matches the import's package and whose name or
  ``*`` matches the imported tail.
* **Rust** — ``crate::`` paths map to module files: ``crate::auth::service`` is
  ``src/auth/service.rs`` or ``src/auth/service/mod.rs``.

Like every resolver here, misses are recorded, not dropped: "this file imports
something I could not find" is a fact (plan.md §69), and the dangling-import
reconstruction needs the unresolved record to say which removal caused it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from traceflow.git.repository import Repository

_GOMOD_FILENAME = "go.mod"

_MODULE_RE = re.compile(r"^\s*module\s+(\S+)\s*$", re.MULTILINE)

_PACKAGE_RE = re.compile(r"^\s*package\s+([A-Za-z_][\w.]*)", re.MULTILINE)
_NAMESPACE_RE = re.compile(r"^\s*namespace\s+([A-Za-z_][\w.]*)")

_MOD_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+([A-Za-z_]\w*)", re.MULTILINE)

_CFACTORY_SUFFIXES = (".go", ".java", ".rs", ".cs")
_CFACTORY_EXTENSIONS = {"go": ".go", "java": ".java", "rust": ".rs", "csharp": ".cs"}


@dataclass(frozen=True)
class CFamilyIndex:
    """What an import path in one of these languages resolves against."""

    language: str
    files: tuple[str, ...]
    go_module: str | None = None
    """Go: the module path from ``go.mod`` — the prefix of internal import paths."""

    packages: dict[str, str] | None = None
    """Java/C#: dotted package/namespace -> the file declaring it."""

    modules: frozenset[str] = frozenset()
    """Rust: every module path (``auth.service``) the repository's ``mod`` tree declares."""

    @classmethod
    def build(
        cls,
        language: str,
        paths: tuple[str, ...],
        repository: Repository,
    ) -> CFamilyIndex:
        """Read the language's project files and index what its imports can reach.

        A file whose head cannot be read contributes nothing rather than raising:
        a half-written file is a normal state (plan.md §46), and the next session
        reparses it.
        """
        packages: dict[str, str] = {}
        modules: set[str] = set()
        go_module: str | None = None

        if language == "go":
            go_module = cls._read_go_module(repository)
        elif language == "rust":
            modules = cls._read_rust_modules(paths, repository)
        elif language in ("java", "csharp"):
            for path in paths:
                head = _head_text(repository, path)
                if language == "java":
                    match = _PACKAGE_RE.search(head)
                else:
                    match = _NAMESPACE_RE.search(head)
                if match:
                    # First declaration wins; a class is one file in both.
                    packages.setdefault(match.group(1), path)

        return cls(
            language=language,
            files=tuple(paths),
            go_module=go_module,
            packages=packages or None,
            modules=frozenset(modules),
        )

    @staticmethod
    def _read_go_module(repository: Repository) -> str | None:
        path = repository.root / _GOMOD_FILENAME
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        match = _MODULE_RE.search(text)
        return match.group(1) if match else None

    @staticmethod
    def _read_rust_modules(paths: tuple[str, ...], repository: Repository) -> set[str]:
        """Every module path the crate declares, derived from ``mod`` items and files.

        ``mod auth;`` in ``src/lib.rs`` makes ``auth`` a module; ``pub mod service;``
        inside ``src/auth.rs`` (or ``src/auth/mod.rs``) makes ``auth.service`` one.
        Files themselves also contribute: ``src/auth/service.rs`` is reachable as
        ``auth::service`` whether or not a ``mod`` item was read successfully.
        """
        module_paths: set[str] = set()

        # Files as module paths: src/auth/service.rs -> auth.service,
        # src/auth/service/mod.rs -> auth.service, src/lib.rs / src/main.rs -> crate root.
        for path in paths:
            normalised = path.replace("\\", "/")
            if normalised.startswith("src/"):
                stripped = normalised[len("src/") :]
                for suffix in (".rs",):
                    if stripped.endswith(suffix):
                        stripped = stripped[: -len(suffix)]
                if stripped == "lib" or stripped == "main":
                    continue
                if stripped.endswith("/mod"):
                    stripped = stripped[: -len("/mod")]
                module_paths.add(stripped.replace("/", "."))

        # `mod` items refine the tree; they are what makes a module *private* vs
        # declared, but for resolution the file mapping above is the evidence.
        for path in paths:
            head = _head_text(repository, path)
            parent = path.replace("\\", "/").rsplit("/", 1)[0] if "/" in path else ""
            if parent.startswith("src"):
                parent = parent[4:] if len(parent) > 3 else ""
            prefix = parent.replace("/", ".")
            for match in _MOD_RE.finditer(head):
                name = match.group(1)
                module_paths.add(f"{prefix}.{name}" if prefix else name)

        return module_paths

    def resolve(self, import_path: str) -> str | None:
        """The repository file *import_path* reaches, or ``None`` when external."""
        if self.language == "go":
            return self._resolve_go(import_path)
        if self.language == "rust":
            return self._resolve_rust(import_path)
        return self._resolve_package(import_path)

    def _resolve_go(self, import_path: str) -> str | None:
        """A path under the module name maps to the directory of the same shape.

        The directory is matched against the file's *directory* rather than its
        whole path, so an import of ``pkg`` still resolves when the directory's
        file is not named ``<pkg>.go`` — Go has no such rule, and ``server.go``
        inside ``pkg/`` is exactly how a package is laid out in practice.
        """
        if not self.go_module or not import_path.startswith(self.go_module + "/"):
            return None
        relative = import_path[len(self.go_module) + 1 :]
        prefix = relative + "/"
        for file_path in self.files:
            normalised = file_path.replace("\\", "/")
            if normalised.endswith(".go") and normalised.rsplit("/", 1)[0] + "/" == prefix:
                return normalised
        return None

    def _resolve_rust(self, import_path: str) -> str | None:
        """``crate::auth::service`` resolves against the module tree.

        ``crate::`` anchors at ``src/``; ``self`` segments collapse away. A use
        path can also name an *item* — ``use crate::auth::service::login;``
        imports a function, not a module — so the full path is tried against the
        module set and then each parent in turn: the first module prefix that
        exists is the file the item lives in. A path none of whose prefixes is a
        declared module resolves to nothing — an external crate is external.
        """
        # The analyzer stores use paths dot-separated (``crate.auth.service``);
        # both separators are accepted here because the path shape is the same
        # fact whichever way it arrived.
        segments = [
            part for part in import_path.replace("::", ".").split(".") if part and part != "self"
        ]
        if segments and segments[0] == "crate":
            segments = segments[1:]
        normalised = ".".join(segments)
        while normalised:
            if normalised in self.modules:
                base = normalised.replace(".", "/")
                for candidate in (f"src/{base}.rs", f"src/{base}/mod.rs"):
                    if candidate in self.files:
                        return candidate
                # A declared module whose file is gone is a broken import, not a
                # reason to climb to the parent: ``use crate::auth::service::x``
                # after ``service.rs`` is deleted reaches nothing, and recording
                # it as reaching ``src/auth/mod.rs`` would hide exactly the fact
                # the dangling-import reconstruction exists to report.
                return None
            normalised = normalised.rpartition(".")[0]
        return None

    def _resolve_package(self, import_path: str) -> str | None:
        """Java/C#: the import names a package plus a member — usually.

        Java's ``import com.example.db.Store`` splits at the last dot. C#'s
        ``using App.Data;`` may name a *namespace alone* (the using-namespace
        form, the common one), so the whole path is tried first and the split
        second — the order that makes both languages resolve without knowing
        which one wrote the import.
        """
        if not self.packages:
            return None
        whole = self.packages.get(import_path)
        if whole is not None:
            return whole
        package, _, _member = import_path.rpartition(".")
        target = self.packages.get(package)
        if target is not None:
            return target
        # `import com.example.service.*` — the star ends one dot earlier.
        if package.endswith(".*"):
            return self.packages.get(package[:-2])
        return None


# The languages the dispatcher knows, in a fixed order so index construction is
# deterministic. Kept beside the dispatcher rather than derived from the profiles
# so that resolution never needs an analyzer instance — the pipeline's resolution
# sites hold a path and nothing else.
_CFACTORY_LANGUAGES = ("go", "java", "rust", "csharp")


def cfamily_languages_for(path: str) -> str | None:
    """The cfamily language that claims *path* by suffix, or ``None``."""
    normalised = path.replace("\\", "/").lower()
    if normalised.endswith(".go"):
        return "go"
    if normalised.endswith(".java"):
        return "java"
    if normalised.endswith(".rs"):
        return "rust"
    if normalised.endswith(".cs"):
        return "csharp"
    return None


class CFamilyFiles:
    """Every cfamily language's index, built once from one repository listing.

    The mirror of :class:`~traceflow.languages.typescript.graph.TypeScriptFiles`:
    the graph, the symbol pass and the impact walk all resolve against this one
    container, so a single command asks each language's project files (``go.mod``,
    package heads, ``mod`` items) for their answer exactly once. An index is built
    per language the listing shows files for.
    """

    def __init__(self, indexes: dict[str, CFamilyIndex]) -> None:
        self.indexes = indexes

    @classmethod
    def of(
        cls, paths: tuple[str, ...], repository: Repository, *, include_missing: bool = False
    ) -> CFamilyFiles:
        """Build from an already-listed repository, one index per language present.

        ``include_missing`` keeps files that no longer exist on disk, which is what
        re-resolving a deleted module's importers needs: the whole point of that
        check is that the target is *gone*, so an existence filter would delete the
        evidence before the question was asked.
        """
        members: dict[str, list[str]] = {name: [] for name in _CFACTORY_LANGUAGES}
        for path in paths:
            language = cfamily_languages_for(path)
            if language is None:
                continue
            if not include_missing and not (repository.root / path).is_file():
                continue
            members[language].append(path.replace("\\", "/"))
        return cls(
            {
                language: CFamilyIndex.build(language, tuple(paths_for), repository)
                for language, paths_for in members.items()
                if paths_for
            }
        )

    def for_path(self, path: str) -> CFamilyIndex | None:
        """The index of the language that claims *path*, or ``None`` when absent."""
        language = cfamily_languages_for(path)
        if language is None:
            return None
        return self.indexes.get(language)

    def resolve(self, import_path: str, importing_path: str) -> str | None:
        """The file *import_path* reaches when imported from *importing_path*.

        Resolution always uses the importing file's own language index — a Go
        path never resolves by Java rules — and returns ``None`` when the language
        has no files in the repository, which makes every import of it external.
        Even where the importing suffix is not needed as a key (Go and Rust keys
        are exact strings, Java and C# keys are dotted declarations), it stays as
        the one honest check: resolution answers for a file, not for a path.
        """
        index = self.for_path(importing_path)
        if index is None:
            return None
        return index.resolve(import_path)

    def siblings_of(self, path: str) -> tuple[str, ...]:
        """The other files in *path*'s language and directory — the same-package
        (Go) or same-namespace (C#) candidates a no-import call may reach.

        Ordered for deterministic answers, because "first file that defines the
        name" only means something when the order cannot drift.
        """
        index = self.for_path(path)
        if index is None:
            return ()
        normalised = path.replace("\\", "/")
        directory = normalised.rsplit("/", 1)[0]
        prefix = f"{directory}/"
        return tuple(sorted(candidate for candidate in index.files if candidate.startswith(prefix)))

    @staticmethod
    def module_name_for(path: str) -> str | None:
        """The name a cfamily file is reported under: its path without the extension.

        There is no package tree to fold into a shorter name: Go imports by
        directory, and the other three resolve through declarations inside the
        file, so — as with TypeScript — the path itself is the module identity.
        """
        normalised = path.replace("\\", "/")
        language = cfamily_languages_for(normalised)
        if language is not None:
            suffix = _CFACTORY_EXTENSIONS[language]
            if normalised.lower().endswith(suffix):
                return normalised[: -len(suffix)]
        return normalised


def _head_text(repository: Repository, path: str, limit: int = 8192) -> str:
    """The file's first *limit* bytes as text — where package/namespace lines live."""
    try:
        with (repository.root / path).open("rb") as handle:
            return handle.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return ""


__all__ = ["CFamilyFiles", "CFamilyIndex", "cfamily_languages_for"]
