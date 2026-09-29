"""Activity sources (plan.md §8-11).

TraceFlow's watcher does not need to know *what* changed — only *whether* anything
changed. That single observation is what the quiescence detector consumes, so the
interface here is deliberately narrow: return a fingerprint of the repository as it
is right now, and let the caller compare successive fingerprints.

That narrowness is what makes the watcher robust. Git is the source of truth for
what changed; the activity source merely decides *when* to look. If a signal is
ever missed, the next successful sample still returns the complete truth, so a
missed signal costs latency and never correctness.

Phase 1 ships polling only. A filesystem-event source (``watchdog``, plan.md §8-11)
can be added later by computing the same fingerprint from accumulated events,
without changing a single caller. The abstraction is justified by that known second
implementation, not by speculation.
"""

from __future__ import annotations

from typing import Protocol

from traceflow.git.repository import Repository, WorkingTreeState


class ActivitySource(Protocol):
    """Something that can report the repository's current state.

    Implementations must be *stable*: an unchanged repository must always produce
    an equal fingerprint, or the detector will report activity that never happened.
    """

    def sample(self) -> WorkingTreeState:
        """Return the repository's current state."""
        ...


class PollingActivitySource:
    """Samples repository state by asking git.

    Polling is the default rather than a fallback, for three reasons:

    1. ``plan.md`` §11 mandates an eight-second quiet period. A two-second poll is
       therefore a quarter of the debounce window — well below anything a person
       can perceive — which removes the only real advantage a filesystem-event
       watcher would have offered.
    2. ``git status`` applies the repository's real ``.gitignore``. A hand-maintained
       parallel ignore list would drift from it; git's own rules cannot.
    3. The same call supplies both the activity signal and, from Phase 2 onward, the
       change set — one operation instead of two sources that could disagree.
    """

    def __init__(self, repository: Repository, ignore: tuple[str, ...] = ()) -> None:
        self._repository = repository
        self._ignore = ignore

    @property
    def repository(self) -> Repository:
        return self._repository

    def sample(self) -> WorkingTreeState:
        return self._repository.working_tree_state(self._ignore)
