"""The stability state machine.

Every test here drives an injected clock rather than sleeping, so the suite is fast
and the timing assertions are exact.
"""

from __future__ import annotations

from tests.conftest import FakeClock
from traceflow.config import ActivityConfig
from traceflow.watcher.quiescence import QuiescenceDetector, WatchState


def build(clock: FakeClock, quiet: float = 8.0, minimum: float = 2.0) -> QuiescenceDetector:
    return QuiescenceDetector(
        ActivityConfig(
            quiet_period_seconds=quiet,
            minimum_session_seconds=minimum,
            poll_interval_seconds=2.0,
            idle_poll_interval_seconds=5.0,
        ),
        clock=clock,
    )


def test_first_observation_only_establishes_a_baseline(clock: FakeClock) -> None:
    """A repository that is already dirty at startup must not look like a new session.

    plan.md §14: pre-existing modifications are not the session's doing.
    """
    detector = build(clock)

    assert detector.observe("token-a") is WatchState.IDLE
    assert detector.session_started_at is None
    assert detector.quiet_for_seconds == 0.0


def test_unchanged_token_before_any_activity_stays_idle(clock: FakeClock) -> None:
    detector = build(clock)
    detector.observe("token-a")

    clock.advance(30)
    assert detector.observe("token-a") is WatchState.IDLE
    assert detector.session_started_at is None


def test_a_changed_token_starts_a_session(clock: FakeClock) -> None:
    detector = build(clock)
    detector.observe("token-a")

    clock.advance(1)
    assert detector.observe("token-b") is WatchState.ACTIVE
    assert detector.session_started_at == 1.0
    assert detector.last_activity_at == 1.0


def test_repeated_activity_keeps_the_original_session_start(clock: FakeClock) -> None:
    """A burst is one session, not one session per edit."""
    detector = build(clock)
    detector.observe("token-a")

    clock.advance(1)
    detector.observe("token-b")
    clock.advance(1)
    detector.observe("token-c")

    assert detector.session_started_at == 1.0
    assert detector.last_activity_at == 2.0


def test_quiet_but_not_yet_settled_is_quieting(clock: FakeClock) -> None:
    detector = build(clock, quiet=8.0)
    detector.observe("token-a")

    clock.advance(1)
    detector.observe("token-b")

    clock.advance(3)
    assert detector.observe("token-b") is WatchState.QUIETING
    assert detector.quiet_for_seconds == 3.0


def test_settles_once_the_quiet_period_elapses(clock: FakeClock) -> None:
    detector = build(clock, quiet=8.0)
    detector.observe("token-a")

    clock.advance(1)
    detector.observe("token-b")

    clock.advance(7)
    assert detector.observe("token-b") is WatchState.QUIETING

    clock.advance(1)
    assert detector.observe("token-b") is WatchState.STABLE


def test_minimum_session_duration_is_enforced(clock: FakeClock) -> None:
    """A very short quiet period must not settle a session that just started."""
    detector = build(clock, quiet=1.0, minimum=10.0)
    detector.observe("token-a")

    clock.advance(1)
    detector.observe("token-b")

    clock.advance(2)
    assert detector.observe("token-b") is WatchState.QUIETING
    assert detector.session_age_seconds == 2.0

    clock.advance(9)
    assert detector.observe("token-b") is WatchState.STABLE


def test_activity_after_quieting_restarts_the_timer(clock: FakeClock) -> None:
    detector = build(clock, quiet=8.0)
    detector.observe("token-a")

    clock.advance(1)
    detector.observe("token-b")

    clock.advance(6)
    assert detector.observe("token-b") is WatchState.QUIETING

    clock.advance(1)
    assert detector.observe("token-c") is WatchState.ACTIVE

    clock.advance(3)
    assert detector.observe("token-c") is WatchState.QUIETING

    clock.advance(5)
    assert detector.observe("token-c") is WatchState.STABLE


def test_reset_returns_to_idle_without_restarting_the_same_session(clock: FakeClock) -> None:
    """After a session is recorded, the settled state must not immediately repeat."""
    detector = build(clock, quiet=8.0)
    detector.observe("token-a")

    clock.advance(1)
    detector.observe("token-b")
    clock.advance(8)
    assert detector.observe("token-b") is WatchState.STABLE

    detector.reset("token-b")
    assert detector.state is WatchState.IDLE
    assert detector.session_started_at is None

    clock.advance(30)
    assert detector.observe("token-b") is WatchState.IDLE

    clock.advance(1)
    assert detector.observe("token-c") is WatchState.ACTIVE


def test_reset_without_a_token_keeps_the_previous_fingerprint(clock: FakeClock) -> None:
    detector = build(clock)
    detector.observe("token-a")
    clock.advance(1)
    detector.observe("token-b")

    detector.reset()

    # token-b is still the known state, so re-observing it is not new activity.
    assert detector.observe("token-b") is WatchState.IDLE


def test_durations_are_never_negative(clock: FakeClock) -> None:
    detector = build(clock)
    assert detector.quiet_for_seconds == 0.0
    assert detector.session_age_seconds == 0.0

    detector.observe("token-a")
    assert detector.quiet_for_seconds == 0.0
    assert detector.session_age_seconds == 0.0
