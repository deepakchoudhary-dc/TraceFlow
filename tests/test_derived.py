"""The analysis cache.

Derived state is disposable, so every failure mode here degrades to recomputation
rather than to a wrong answer.
"""

from __future__ import annotations

from pathlib import Path

from traceflow.derived import AnalysisCache


def test_put_and_get_round_trip(tmp_path: Path) -> None:
    cache = AnalysisCache(tmp_path)
    cache.put("python-1", "a" * 64, {"symbols": []})

    assert cache.get("python-1", "a" * 64) == {"symbols": []}


def test_a_missing_entry_is_a_miss(tmp_path: Path) -> None:
    assert AnalysisCache(tmp_path).get("python-1", "b" * 64) is None


def test_a_corrupt_entry_is_a_miss_not_an_error(tmp_path: Path) -> None:
    """Discarding a damaged entry costs a re-parse; raising would cost the session."""
    cache = AnalysisCache(tmp_path)
    path = cache.path_for("python-1", "c" * 64)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    assert cache.get("python-1", "c" * 64) is None


def test_a_non_object_entry_is_a_miss(tmp_path: Path) -> None:
    cache = AnalysisCache(tmp_path)
    path = cache.path_for("python-1", "d" * 64)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[1, 2, 3]", encoding="utf-8")

    assert cache.get("python-1", "d" * 64) is None


def test_kinds_are_namespaced(tmp_path: Path) -> None:
    """An analyzer change must not be shadowed by entries its predecessor wrote."""
    cache = AnalysisCache(tmp_path)
    cache.put("python-1", "e" * 64, {"version": 1})
    cache.put("python-2", "e" * 64, {"version": 2})

    assert cache.get("python-1", "e" * 64) == {"version": 1}
    assert cache.get("python-2", "e" * 64) == {"version": 2}


def test_entries_are_sharded_by_digest_prefix(tmp_path: Path) -> None:
    cache = AnalysisCache(tmp_path)
    digest = "ab" + "0" * 62

    assert cache.path_for("python-1", digest).parent.name == "ab"


def test_putting_twice_is_harmless(tmp_path: Path) -> None:
    cache = AnalysisCache(tmp_path)
    cache.put("python-1", "f" * 64, {"a": 1})
    cache.put("python-1", "f" * 64, {"a": 2})

    assert cache.get("python-1", "f" * 64) == {"a": 2}


def test_the_cache_lives_under_derived(tmp_path: Path) -> None:
    """The whole point of the split: this directory can be deleted and rebuilt."""
    assert AnalysisCache(tmp_path).root.parts[-2:] == ("derived", "analysis")
