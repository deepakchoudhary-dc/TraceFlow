"""Configuration parsing, validation, and defaults."""

from __future__ import annotations

from pathlib import Path

import pytest

from traceflow.compat import tomllib
from traceflow.config import (
    CONFIG_FILENAME,
    DEFAULT_IGNORE,
    ConfigError,
    default_config,
    load_config,
    parse_config,
    render_default_config,
)
from traceflow.secrets import DEFAULT_SECRET_PATHS


def test_defaults_match_the_plan() -> None:
    config = default_config()
    assert config.activity.quiet_period_seconds == 8.0
    assert config.activity.minimum_session_seconds == 2.0
    assert config.repository.path == "."
    assert config.ignore == DEFAULT_IGNORE


def test_state_directory_is_ignored_by_default() -> None:
    """TraceFlow must never treat its own writes as repository activity."""
    assert ".traceflow" in default_config().ignore


def test_parsing_an_empty_table_yields_defaults() -> None:
    assert parse_config({}) == default_config()


def test_parsing_a_full_document_overrides_only_what_it_sets() -> None:
    config = parse_config(
        {
            "repository": {"path": "sub"},
            "activity": {"quiet_period_seconds": 3, "poll_interval_seconds": 0.5},
            "ignore": [".traceflow", "vendor"],
        }
    )
    assert config.repository.path == "sub"
    assert config.activity.quiet_period_seconds == 3.0
    assert config.activity.poll_interval_seconds == 0.5
    assert config.activity.minimum_session_seconds == 2.0  # untouched default
    assert config.ignore == (".traceflow", "vendor")


def test_unknown_top_level_key_is_rejected() -> None:
    """A typo must fail loudly rather than silently behaving like the default."""
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config({"actvity": {}})


def test_unknown_section_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match=r"\[activity\]"):
        parse_config({"activity": {"quiet_period": 4}})


def test_unknown_repository_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match=r"\[repository\]"):
        parse_config({"repository": {"root": "."}})


def test_section_that_is_not_a_table_is_rejected() -> None:
    with pytest.raises(ConfigError, match="must be a table"):
        parse_config({"activity": 5})


def test_non_numeric_activity_value_is_rejected() -> None:
    with pytest.raises(ConfigError, match="must be a number"):
        parse_config({"activity": {"quiet_period_seconds": "eight"}})


def test_boolean_is_not_accepted_as_a_number() -> None:
    """``True`` is an ``int`` in Python; accepting it would be a silent surprise."""
    with pytest.raises(ConfigError, match="must be a number"):
        parse_config({"activity": {"quiet_period_seconds": True}})


def test_negative_activity_value_is_rejected() -> None:
    with pytest.raises(ConfigError, match="negative"):
        parse_config({"activity": {"quiet_period_seconds": -1}})


def test_non_string_repository_path_is_rejected() -> None:
    with pytest.raises(ConfigError, match=r"repository\.path must be a string"):
        parse_config({"repository": {"path": 5}})


def test_ignore_must_be_a_list_of_strings() -> None:
    with pytest.raises(ConfigError, match="array of strings"):
        parse_config({"ignore": "vendor"})

    with pytest.raises(ConfigError, match="array of strings"):
        parse_config({"ignore": [1, 2]})


# --------------------------------------------------------------------------- loading


def test_missing_file_falls_back_to_defaults(tmp_path: Path) -> None:
    """TraceFlow must work on a repository that was never initialised."""
    assert load_config(tmp_path) == default_config()


def test_a_written_file_is_read_back(tmp_path: Path) -> None:
    (tmp_path / CONFIG_FILENAME).write_text(
        "[activity]\nquiet_period_seconds = 4\n", encoding="utf-8"
    )
    assert load_config(tmp_path).activity.quiet_period_seconds == 4.0


def test_malformed_toml_raises_a_config_error(tmp_path: Path) -> None:
    (tmp_path / CONFIG_FILENAME).write_text("[activity\nbroken", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(tmp_path)


def test_rendered_default_config_parses_back_to_the_defaults() -> None:
    """The file ``traceflow init`` writes must be one the loader accepts."""
    assert parse_config(tomllib.loads(render_default_config())) == default_config()


def test_rendered_config_documents_every_supported_key() -> None:
    rendered = render_default_config()
    for key in (
        "quiet_period_seconds",
        "minimum_session_seconds",
        "poll_interval_seconds",
        "idle_poll_interval_seconds",
        "max_file_size_mb",
        "exclude_paths",
        "ignore",
    ):
        assert key in rendered


# --------------------------------------------------------------------------- analysis


def test_max_file_size_defaults_and_converts_to_bytes() -> None:
    analysis = default_config().analysis
    assert analysis.max_file_size_mb == 5.0
    assert analysis.max_file_size_bytes == 5 * 1024 * 1024


def test_max_file_size_is_configurable() -> None:
    config = parse_config({"analysis": {"max_file_size_mb": 0.5}})
    assert config.analysis.max_file_size_bytes == 512 * 1024


def test_unknown_analysis_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match=r"\[analysis\]"):
        parse_config({"analysis": {"max_file_size": 5}})


def test_negative_max_file_size_is_rejected() -> None:
    with pytest.raises(ConfigError, match="negative"):
        parse_config({"analysis": {"max_file_size_mb": -1}})


# --------------------------------------------------------------------------- secrets


def test_secret_paths_default_to_the_built_in_policy() -> None:
    assert default_config().secrets.exclude_paths == DEFAULT_SECRET_PATHS
    assert ".env" in DEFAULT_SECRET_PATHS


def test_secret_paths_are_configurable() -> None:
    config = parse_config({"secrets": {"exclude_paths": ["*.vault", ".env"]}})
    assert config.secrets.exclude_paths == ("*.vault", ".env")


def test_an_empty_secret_list_is_honoured() -> None:
    """A repository may legitimately decide it has nothing to withhold."""
    assert parse_config({"secrets": {"exclude_paths": []}}).secrets.exclude_paths == ()


def test_unknown_secrets_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match=r"\[secrets\]"):
        parse_config({"secrets": {"exclude": [".env"]}})


def test_secret_paths_must_be_a_list_of_strings() -> None:
    with pytest.raises(ConfigError, match=r"\[secrets\]\.exclude_paths"):
        parse_config({"secrets": {"exclude_paths": ".env"}})


def test_rendered_config_round_trips_including_the_new_sections() -> None:
    """The file ``traceflow init`` writes must load back to exactly the defaults."""
    assert parse_config(tomllib.loads(render_default_config())) == default_config()
