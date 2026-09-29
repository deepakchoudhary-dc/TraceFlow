"""Content-addressed storage and the binary heuristic."""

from __future__ import annotations

from pathlib import Path

from traceflow.blobs import BlobStore, digest_of, looks_binary


def test_digest_is_stable_and_content_addressed() -> None:
    assert digest_of(b"hello") == digest_of(b"hello")
    assert digest_of(b"hello") != digest_of(b"hello\n")


def test_put_returns_the_digest_and_stores_once(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)

    digest = store.put(b"payload")

    assert digest == digest_of(b"payload")
    assert store.has(digest)
    assert store.get(digest) == b"payload"
    assert store.path_for(digest).read_bytes() == b"payload"


def test_identical_content_is_stored_once_across_paths(tmp_path: Path) -> None:
    """Two files with the same bytes cost one object, which is what makes renames free."""
    store = BlobStore(tmp_path)

    first = store.put(b"shared content")
    second = store.put(b"shared content")

    assert first == second
    stored = [path for path in store.root.rglob("*") if path.is_file()]
    assert len(stored) == 1


def test_get_returns_none_for_absent_content(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    assert store.get("0" * 64) is None
    assert store.has("0" * 64) is False


def test_objects_are_sharded_by_digest_prefix(tmp_path: Path) -> None:
    """Keeps a large capture from landing thousands of files in one directory."""
    store = BlobStore(tmp_path)
    digest = store.put(b"anything")

    assert store.path_for(digest).parent.name == digest[:2]


def test_binary_heuristic_matches_gits_rule() -> None:
    assert looks_binary(b"plain text\n") is False
    assert looks_binary(b"text with a \x00 nul byte") is True


def test_binary_heuristic_only_inspects_the_first_blocks() -> None:
    """git only sniffs the first 8000 bytes; agreeing with it avoids surprises."""
    assert looks_binary(b"a" * 9000) is False
    assert looks_binary(b"a" * 9000 + b"\x00") is False
    assert looks_binary(b"\x00" + b"a" * 9000) is True
