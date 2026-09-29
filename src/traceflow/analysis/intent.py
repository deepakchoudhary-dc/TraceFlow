"""Intent versus actual change (plan.md §63, §20, §21).

The user may record the task they gave the coding agent. This module compares that
task against what the session's own artifacts say changed, and reports the comparison
in plan.md §63's words:

    Potentially related · Potential scope expansion · No detected relationship

and never §21's forbidden verdicts. The comparison is **token overlap** — words the
task shares with a path, a changed symbol, or an import — organised per changed
*file*, because that is the unit plan.md §20 reports in ("5 directly related",
"1 apparently unrelated") and the unit a reviewer acts on. A file counts as related
when the task shares a word with its path, with a symbol the session changed in it,
or with a module it began or stopped importing. Every other changed file is named
without a verdict, so the reader can see exactly what the task's words did not reach.

The comparison deliberately stops there. Whether a change was *necessary* needs the
human, the task and the code; static analysis has none of the three, and a verdict it
could not support would be exactly the assertion this product exists to replace
(plan.md §69). The delivery says so beside the verdict.

Three quiet honesty rules:

* A task that parses to nothing but stopwords is ``NOT_COMPARED`` — the session is
  then reported as not compared, never as "unrelated", because a verdict of unrelated
  built on an empty token set would be fabricated precision.
* A session with no recorded changes is likewise ``NOT_COMPARED``: there is nothing
  to relate the task to, which is a state of the record, not a finding about it.
* Every match is reported on the verdict it produced, so a reader can check the
  overlap instead of trusting it (plan.md §33).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

#: Words that cannot point at anything by themselves. A task that parses to only
#: these carries no signal, and claiming "no detected relationship" from it would be
#: inventing a verdict from an empty set.
_STOPWORDS = frozenset(
    {
        "a",
        "add",
        "added",
        "adding",
        "all",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "code",
        "fix",
        "fixed",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "its",
        "make",
        "me",
        "my",
        "new",
        "not",
        "of",
        "on",
        "or",
        "please",
        "refactor",
        "refactoring",
        "should",
        "so",
        "some",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "this",
        "to",
        "update",
        "updated",
        "updating",
        "use",
        "using",
        "we",
        "with",
        "would",
        "you",
        "your",
        "i",
        "thank",
        "thanks",
    }
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")

#: Inflections stripped when matching, so "limiting" finds ``rate_limit``. Only the
#: suffix is removed and only when what remains is still a plausible word — four
#: characters is the floor, so "rating" never yields "rat".
_INFLECTIONS = ("ing", "ed", "es", "s")
_MIN_STEM_LENGTH = 4


def token_forms(token: str) -> tuple[str, ...]:
    """The spellings *token* can match under: itself, then its uninflected stems.

    "limiting" matches ``rate_limit`` through ``limit``; a token with no removable
    suffix only ever matches as itself. The token is tried first, so an exact name is
    never shadowed by a stem.
    """
    forms = [token]
    for suffix in _INFLECTIONS:
        if token.endswith(suffix):
            stem = token[: -len(suffix)]
            if len(stem) >= _MIN_STEM_LENGTH and stem not in forms:
                forms.append(stem)
    return tuple(forms)


class Relatedness(str, Enum):
    """The verdict the comparison can reach (plan.md §63).

    ``NOT_COMPARED`` is the honest answer when the session recorded no task, or one
    that parses to nothing but stopwords. It is a state of the record, not a finding
    about the change.
    """

    NOT_COMPARED = "not_compared"
    RELATED = "related"
    SCOPE_EXPANSION = "scope_expansion"
    NO_RELATIONSHIP = "no_relationship"


#: How the delivery names each verdict, in plan.md §63's own words.
RELATEDNESS_LABELS: dict[Relatedness, str] = {
    Relatedness.NOT_COMPARED: "not compared",
    Relatedness.RELATED: "potentially related",
    Relatedness.SCOPE_EXPANSION: "potential scope expansion",
    Relatedness.NO_RELATIONSHIP: "no detected relationship",
}


@dataclass(frozen=True)
class ModuleFacts:
    """What one changed file is, in the words a task could share."""

    path: str
    symbols: tuple[str, ...] = ()
    """Qualified names of the symbols the session changed in this file."""

    imports: tuple[str, ...] = ()
    """The modules this file began or stopped importing, as rendered by the analyzer."""


@dataclass(frozen=True)
class IntentMatch:
    """One token that matched one thing the session touched."""

    token: str
    kind: str
    """``path``, ``symbol`` or ``import`` — what the token matched."""

    detail: str = ""
    """What was matched exactly, so the reader can check the overlap themselves."""


@dataclass(frozen=True)
class FileCoverage:
    """Whether the recorded task reaches one changed file, and by what words."""

    path: str
    related: bool
    matches: tuple[IntentMatch, ...] = ()


@dataclass(frozen=True)
class IntentComparison:
    """The task compared against the session's own record (plan.md §63)."""

    task: str
    relatedness: Relatedness
    tokens: tuple[str, ...] = ()
    """The task's informative tokens, in the order they appear in the task."""
    files: tuple[FileCoverage, ...] = ()
    """Every changed file, each with the words that tie it to the task — or none."""
    unmatched: tuple[str, ...] = ()
    """Informative tokens nothing in the session's record shares."""
    changed_files: int = 0
    notes: tuple[str, ...] = ()

    @property
    def is_compared(self) -> bool:
        return self.relatedness is not Relatedness.NOT_COMPARED

    @property
    def label(self) -> str:
        return RELATEDNESS_LABELS[self.relatedness]

    @property
    def related_files(self) -> tuple[str, ...]:
        return tuple(item.path for item in self.files if item.related)

    @property
    def unrelated_files(self) -> tuple[str, ...]:
        return tuple(item.path for item in self.files if not item.related)

    @property
    def matches(self) -> tuple[IntentMatch, ...]:
        """Every match behind the verdict, in file order."""
        return tuple(match for item in self.files for match in item.matches)

    def to_json(self) -> dict[str, object]:
        return {
            "task": self.task,
            "relatedness": self.relatedness.value,
            "tokens": list(self.tokens),
            "files": [
                {
                    "path": item.path,
                    "related": item.related,
                    "matches": [
                        {
                            "token": match.token,
                            "kind": match.kind,
                            "detail": match.detail,
                        }
                        for match in item.matches
                    ],
                }
                for item in self.files
            ],
            "unmatched": list(self.unmatched),
            "changed_files": self.changed_files,
            "notes": list(self.notes),
        }


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _flag(value: object, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _int_or(value: object, default: int) -> int:
    """A whole number from stored JSON, or *default*. ``bool`` is excluded."""
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def intent_from_json(payload: dict[str, object] | None) -> IntentComparison | None:
    """Rebuild a stored comparison, or ``None`` when there is none to read.

    A damaged or hand-edited artifact degrades to "no task recorded" rather than a
    broken delivery — the same rule every other artifact follows (plan.md §46).
    """
    if payload is None:
        return None
    raw_relatedness = payload.get("relatedness")
    if not isinstance(raw_relatedness, str):
        return None
    try:
        relatedness = Relatedness(raw_relatedness)
    except ValueError:
        return None

    raw_files = payload.get("files")
    files = tuple(
        FileCoverage(
            path=_optional_text(item.get("path")) or "?",
            related=_flag(item.get("related")),
            matches=tuple(
                IntentMatch(
                    token=_optional_text(match.get("token")) or "?",
                    kind=_optional_text(match.get("kind")) or "path",
                    detail=_optional_text(match.get("detail")) or "",
                )
                for match in raw_matches
                if isinstance(match, dict)
            ),
        )
        for item in (raw_files if isinstance(raw_files, list) else ())
        if isinstance(item, dict)
        for raw_matches in (item.get("matches"),)
        if isinstance(raw_matches, list)
    )

    return IntentComparison(
        task=_optional_text(payload.get("task")) or "",
        relatedness=relatedness,
        tokens=_strings(payload.get("tokens")),
        files=files,
        unmatched=_strings(payload.get("unmatched")),
        changed_files=_int_or(payload.get("changed_files"), 0),
        notes=_strings(payload.get("notes")),
    )


def parse_tokens(task: str) -> tuple[str, ...]:
    """The informative tokens of a task, in order, deduplicated.

    Lower-cased, split on anything that is not a letter, digit or underscore, with the
    stopwords removed. Underscores survive because they are how Python names are
    spelled, and ``rate_limit`` matching ``rate_limit`` is the whole point.
    """
    found: list[str] = []
    for token in _TOKEN_PATTERN.findall(task.lower()):
        if token in _STOPWORDS or token in found:
            continue
        found.append(token)
    return tuple(found)


def _path_matches(path: str, forms: tuple[str, ...]) -> IntentMatch | None:
    """The words of a task can name a file by its path, its stem, or inside the stem."""
    lowered = path.lower()
    stem = lowered.rsplit("/", 1)[-1]
    bare_stem = stem.rsplit(".", 1)[0].replace("_", "")
    segments = lowered.split("/")

    for form in forms:
        if form in segments or form == stem:
            return IntentMatch(token=forms[0], kind="path", detail=path)
        if len(form) >= 4 and form in bare_stem:
            return IntentMatch(token=forms[0], kind="path", detail=f"{path} (in the file name)")
    return None


def _word_parts(names: tuple[str, ...]) -> frozenset[str]:
    """Every word inside dotted/underscored names: ``rate_limit.check`` → all four parts."""
    parts: set[str] = set()
    for name in names:
        for part in name.replace(".", "_").split("_"):
            if part:
                parts.add(part.lower())
    return frozenset(parts)


def _compare_file(facts: ModuleFacts, tokens: tuple[str, ...]) -> FileCoverage:
    """One file, measured against every informative token of the task."""
    symbol_parts = _word_parts(facts.symbols)
    import_parts = _word_parts(facts.imports)

    matches: list[IntentMatch] = []
    for token in tokens:
        forms = token_forms(token)
        hit = _path_matches(facts.path, forms)
        if hit is not None:
            matches.append(hit)
            continue
        for form in forms:
            if form in symbol_parts:
                matches.append(IntentMatch(token=token, kind="symbol", detail=facts.path))
                break
        else:
            for form in forms:
                if form in import_parts:
                    matches.append(IntentMatch(token=token, kind="import", detail=facts.path))
                    break

    return FileCoverage(path=facts.path, related=bool(matches), matches=tuple(matches))


def compare_intent(task: str, modules: tuple[ModuleFacts, ...]) -> IntentComparison:
    """Compare the task's words with what the session's artifacts say changed.

    *modules* carries each changed file with the symbols changed in it and the imports
    it gained or lost — exactly what ``symbols.json`` and ``changes.json`` record, so
    this can never disagree with `traceflow impact`.
    """
    if not modules:
        return IntentComparison(
            task=task,
            relatedness=Relatedness.NOT_COMPARED,
            notes=(
                "The session recorded no file changes, so there is nothing to compare "
                "the task against.",
            ),
        )

    tokens = parse_tokens(task)
    if not tokens:
        return IntentComparison(
            task=task,
            relatedness=Relatedness.NOT_COMPARED,
            changed_files=len(modules),
            notes=(
                "The task parses to no words beyond common filler, so no comparison is "
                "possible. It is recorded verbatim on the session, not judged.",
            ),
        )

    files = tuple(_compare_file(facts, tokens) for facts in modules)
    matched_paths = {item.path for item in files if item.related}
    unmatched = tuple(
        token
        for token in tokens
        if not any(match.token == token for item in files for match in item.matches)
    )

    # The verdict is about *files*, plan.md §20's unit: every changed file covered by
    # some word of the task is related; the rest are the potential scope expansion.
    # Unmatched tokens are recorded but never turn a fully-covered change into an
    # expansion — "endpoint" describing "login" is vocabulary, not a second task.
    if not matched_paths:
        relatedness = Relatedness.NO_RELATIONSHIP
    elif len(matched_paths) < len(files):
        relatedness = Relatedness.SCOPE_EXPANSION
    else:
        relatedness = Relatedness.RELATED

    return IntentComparison(
        task=task,
        relatedness=relatedness,
        tokens=tokens,
        files=files,
        unmatched=unmatched,
        changed_files=len(modules),
    )
