"""The sensitive-path policy (plan.md §26).

The policy is deliberately conservative: a false positive costs one file's line
statistics, while a false negative costs a leaked credential. These tests pin down
both directions.
"""

from __future__ import annotations

import pytest

from traceflow.secrets import DEFAULT_SECRET_PATHS, matches_secret_path


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        "config/.env",
        "server.key",
        "certs/server.pem",
        "deploy/id_rsa",
        "deploy/id_ed25519",
        ".ssh/config",
        ".aws/credentials",
        "home/.netrc",
        ".npmrc",
        "credentials",
        "secrets.yaml",
        "app.secrets",
        "keystore.p12",
        "bundle.pfx",
    ],
)
def test_sensitive_paths_are_matched(path: str) -> None:
    assert matches_secret_path(path, DEFAULT_SECRET_PATHS) is True


@pytest.mark.parametrize(
    "path",
    [
        "src/app.py",
        "README.md",
        "docs/environment.md",
        "tests/test_env.py",
        "my.sshfile",
        "src/keyring.py",
        "environment.py",
        "secrets_manager.py",
    ],
)
def test_ordinary_paths_are_not_matched(path: str) -> None:
    assert matches_secret_path(path, DEFAULT_SECRET_PATHS) is False


def test_a_bare_directory_name_covers_everything_beneath_it() -> None:
    assert matches_secret_path(".ssh/known_hosts", (".ssh",)) is True
    assert matches_secret_path("nested/deep/.ssh/id_rsa", (".ssh",)) is True


def test_a_bare_directory_name_does_not_match_a_similar_filename() -> None:
    """``.ssh`` must not match ``my.sshfile`` just because the text appears."""
    assert matches_secret_path("my.sshfile", (".ssh",)) is False
    assert matches_secret_path("src/.sshrc", (".ssh",)) is False


def test_matching_is_case_insensitive_where_the_filesystem_is() -> None:
    """Windows paths are case-insensitive, so the pattern must be too."""
    assert matches_secret_path("SERVER.PEM", ("*.pem",)) is True


def test_backslashes_are_normalised() -> None:
    assert matches_secret_path("certs\\server.pem", DEFAULT_SECRET_PATHS) is True


def test_leading_and_trailing_separators_are_ignored() -> None:
    assert matches_secret_path("/.env", DEFAULT_SECRET_PATHS) is True
    assert matches_secret_path("config/.env/", DEFAULT_SECRET_PATHS) is True


def test_an_empty_pattern_list_matches_nothing() -> None:
    assert matches_secret_path(".env", ()) is False


def test_blank_patterns_are_skipped_rather_than_matching_everything() -> None:
    """A stray blank line in the config must not silently disable the change set."""
    assert matches_secret_path("src/app.py", ("", "   ")) is False


def test_empty_path_matches_nothing() -> None:
    assert matches_secret_path("", DEFAULT_SECRET_PATHS) is False
