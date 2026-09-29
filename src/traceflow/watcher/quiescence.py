"""Stability detection (plan.md §11-12).

An AI coding agent does not make one edit. It makes many, in bursts, interleaved
with test runs, shell commands, failed attempts, and retries. Analysing after every
change would be both wasteful and wrong — the interesting question is never "what
changed just now" but "what changed during this episode of work".

This module answers only that second question. It is a pure state machine: you feed
it fingerprints and the current time, it tells you whether the repository has
settled. It does not know about git, sessions, or analysis.

Time is injected rather than read directly, so the whole state machine is testable
without sleeping.

Deliberate deviation from plan.md §12: the spec lists five states
(``ACTIVE`` / ``QUIETING`` / ``STABLE`` / ``ANALYZING`` / ``DELIVERED``). Only the first three
are implemented, and the other two are not coming. This was originally written as "the states
that follow arrive with the phase that introduces analysis"; analysis arrived in Phase 3 and
they were never needed, because the watch loop analyses *synchronously*. The loop is blocked
while a session is being recorded, so there is no moment at which the repository is settled
and the analysis is neither running nor finished, which is the only thing ``ANALYZING`` could
describe. ``DELIVERED`` would describe a state that nothing observes.

The same applies to plan.md §12's "activity during analysis invalidates the run". With a
synchronous loop there is no run to invalidate: a change that lands while analysis is running
is seen by the next sample, and the next session is measured from the baseline taken after
this one closed. Shipping unreachable states would be dead code, and a state machine with
unreachable states is a state machine nobody can trust.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import Enum

from traceflow.config import ActivityConfig


class WatchState(str, Enum):
    """What the watcher currently believes about the repository."""

    IDLE = "idle"
    """Nothing has happened since the watcher started, or since the last reset."""

    ACTIVE = "active"
    """A change was just observed."""

    QUIETING = "quieting"
    """A change was observed recently, but not long enough ago to call it settled."""

    STABLE = "stable"
    """The repository has been untouched for the configured quiet period."""


class QuiescenceDetector:
    """Decides when a burst of repository activity has settled.

    The detector is fed fingerprints, not timestamps, so it cannot be confused by a
    clock that jumps: settling is measured from the last observation that actually
    differed.
    """

    def __init__(
        self,
        config: ActivityConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._clock = clock
        self._state = WatchState.IDLE
        self._last_token: str | None = None
        self._last_activity_at: float | None = None
        self._session_started_at: float | None = None

    @property
    def state(self) -> WatchState:
        return self._state

    @property
    def session_started_at(self) -> float | None:
        """Monotonic timestamp when the current session's first change was seen."""
        return self._session_started_at

    @property
    def last_activity_at(self) -> float | None:
        return self._last_activity_at

    @property
    def quiet_for_seconds(self) -> float:
        """How long the repository has been untouched, or ``0.0`` if never touched."""
        if self._last_activity_at is None:
            return 0.0
        return max(0.0, self._clock() - self._last_activity_at)

    @property
    def session_age_seconds(self) -> float:
        """How long the current session has been running, or ``0.0`` if there is none."""
        if self._session_started_at is None:
            return 0.0
        return max(0.0, self._clock() - self._session_started_at)

    def observe(self, token: str) -> WatchState:
        """Feed a fresh fingerprint and return the resulting state.

        The first observation only establishes a baseline. It is not activity: a
        repository that is already dirty when the watcher starts has pre-existing
        changes (plan.md §14), and reporting those as a new session would be wrong.
        """
        now = self._clock()

        if self._last_token is None:
            self._last_token = token
            self._state = WatchState.IDLE
            return self._state

        if token != self._last_token:
            self._last_token = token
            self._last_activity_at = now
            if self._session_started_at is None:
                self._session_started_at = now
            self._state = WatchState.ACTIVE
            return self._state

        if self._session_started_at is None:
            self._state = WatchState.IDLE
            return self._state

        assert self._last_activity_at is not None  # implied by session_started_at
        quiet_for = now - self._last_activity_at
        session_age = now - self._session_started_at

        settled = (
            quiet_for >= self._config.quiet_period_seconds
            and session_age >= self._config.minimum_session_seconds
        )
        self._state = WatchState.STABLE if settled else WatchState.QUIETING
        return self._state

    def reset(self, token: str | None = None) -> None:
        """End the current session and return to idle.

        Passing the token just observed keeps the new baseline current, so the
        settled state that was just reported is not immediately re-detected.
        """
        self._session_started_at = None
        self._last_activity_at = None
        self._state = WatchState.IDLE
        if token is not None:
            self._last_token = token
