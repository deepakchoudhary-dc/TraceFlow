"""Configuration loading, validation, and defaults.

The configuration file is ``.traceflow.toml`` at the repository root, so it can
be committed and shared alongside the code it configures.

Only the sections that the current implementation actually honours are defined
here. ``tests`` and ``visualization`` (plan.md §45) will be added by the phases
that consume them — a config key that nothing reads is a lie told to the user
about what the tool supports.

Two sections are additions to plan.md §45 rather than transcriptions of it:

* ``[analysis]`` carries §45's ``max_file_size_mb``, which Phase 2 needs to bound
  how much of a file it will read.
* ``[secrets]`` has no counterpart in §45, but §26 requires secret handling and
  provides nowhere to configure it. Path exclusion is the primary guarantee, so it
  needs to be configurable per repository.

Note on format: plan.md §45 sketches YAML, but TOML is used instead. TOML is
Python-native, becomes stdlib in 3.11, and needs no third-party parser.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from traceflow.compat import tomllib
from traceflow.secrets import DEFAULT_SECRET_PATHS

CONFIG_FILENAME = ".traceflow.toml"
STATE_DIRNAME = ".traceflow"


class ConfigError(ValueError):
    """Raised when the configuration file is malformed or contains unknown keys."""


@dataclass(frozen=True)
class RepositoryConfig:
    """Where the watched repository lives."""

    path: str = "."


@dataclass(frozen=True)
class ActivityConfig:
    """Tuning for activity detection and stability (plan.md §11)."""

    quiet_period_seconds: float = 8.0
    minimum_session_seconds: float = 2.0
    poll_interval_seconds: float = 2.0
    idle_poll_interval_seconds: float = 5.0


@dataclass(frozen=True)
class AnalysisConfig:
    """Bounds on how much TraceFlow will read, and how far it will look (plan.md §45, §47)."""

    max_file_size_mb: float = 5.0

    impact_max_depth: int = 5
    """How many rings of dependents the impact walk expands.

    An addition to plan.md §45 rather than a transcription of it: §49 requires a focused
    subgraph, which needs a bound, and §45 provides nowhere to configure one. A report
    that stops at the limit records the fact (``truncated``) rather than presenting a
    short walk as a complete one.
    """

    @property
    def max_file_size_bytes(self) -> int:
        return int(self.max_file_size_mb * 1024 * 1024)


@dataclass(frozen=True)
class SecretsConfig:
    """Files whose contents TraceFlow must never read (plan.md §26)."""

    exclude_paths: tuple[str, ...] = DEFAULT_SECRET_PATHS


@dataclass(frozen=True)
class TestsConfig:
    """Configurable test execution (plan.md §22, §64).

    Defaults to **disabled**, and that is not a placeholder: running a repository's
    tests can be expensive or destructive, and plan.md §22 forbids doing it without
    explicit configuration. Even when enabled, TraceFlow runs only the tests a
    session's own impact analysis reaches — never the whole suite.
    """

    enabled: bool = False
    command: str = "pytest"
    """The command line that runs the repository's tests.

    Read from configuration only — never from a request, an artifact, or anywhere a
    repository's contents could put words in. Split with ``shlex`` and executed
    without a shell, so a configured command is a program and arguments, not a
    script an unexpected filename could extend.
    """

    timeout_seconds: float = 300.0


@dataclass(frozen=True)
class Config:
    """The complete, validated configuration."""

    repository: RepositoryConfig
    activity: ActivityConfig
    analysis: AnalysisConfig
    secrets: SecretsConfig
    tests: TestsConfig
    ignore: tuple[str, ...]


# The default ignore list exists for two reasons. The first is a guarantee:
# TraceFlow must never mistake its own writes for repository activity — without
# the state directory here, writing a session record changes the working tree,
# which registers as activity, which starts a new session, forever. The second is
# a lesson from watching real agent sessions: a few file kinds are *moved*, never
# edited, by every tool in the loop — compiled bytecode, office documents a spec
# lives in, image assets a screenshot step rewrites. They appear in every session
# and explain nothing. An entry is either an exact path, a directory prefix, or a
# suffix beginning with ``*`` (``*.pdf``); anything a project genuinely edits
# belongs to a session, so the default list is deliberately short.
DEFAULT_IGNORE: tuple[str, ...] = (
    STATE_DIRNAME,
    "*.doc",
    "*.docx",
    "*.pdf",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.ico",
    "*.svg",
    "*.webp",
    "*.mp4",
    "*.mov",
    "*.zip",
    "*.gz",
    "*.7z",
    "*.exe",
    "*.dll",
    "*.so",
    "*.dylib",
    "*.class",
    "*.jar",
    "*.pyc",
    "*.pyo",
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.otf",
    "*.eot",
)


def default_config() -> Config:
    """Return the built-in defaults, independent of any file on disk."""
    return Config(
        repository=RepositoryConfig(),
        activity=ActivityConfig(),
        analysis=AnalysisConfig(),
        secrets=SecretsConfig(),
        tests=TestsConfig(),
        ignore=DEFAULT_IGNORE,
    )


def _require_table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{key}] must be a table, found {type(value).__name__}")
    return value


def _reject_unknown(data: dict[str, Any], known: set[str], where: str) -> None:
    unknown = sorted(set(data) - known)
    if unknown:
        joined = ", ".join(repr(key) for key in unknown)
        raise ConfigError(f"unknown key(s) in {where}: {joined}")


def _number(table: dict[str, Any], key: str, default: float, where: str) -> float:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{where}.{key} must be a number, found {type(value).__name__}")
    if value < 0:
        raise ConfigError(f"{where}.{key} must not be negative")
    return float(value)


def _string_list(raw: Any, where: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ConfigError(f"{where} must be an array of strings")
    return tuple(raw)


def _count(table: dict[str, Any], key: str, default: int, where: str) -> int:
    """Read a whole-number key. A float or a bool is rejected rather than coerced.

    ``isinstance(True, int)`` is true in Python, so a bare ``int`` check would accept
    ``impact_max_depth = true`` and silently mean 1.
    """
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}.{key} must be a whole number, found {type(value).__name__}")
    if value < 1:
        raise ConfigError(f"{where}.{key} must be at least 1")
    return value


def parse_config(data: dict[str, Any]) -> Config:
    """Validate raw TOML data and build a :class:`Config`.

    Unknown keys are rejected rather than ignored: a typo in a config file
    should fail loudly, not silently take effect as the default.
    """
    _reject_unknown(
        data,
        {"repository", "activity", "analysis", "secrets", "tests", "ignore"},
        "the config file",
    )

    repo_table = _require_table(data, "repository")
    _reject_unknown(repo_table, {"path"}, "[repository]")
    path = repo_table.get("path", ".")
    if not isinstance(path, str):
        raise ConfigError("repository.path must be a string")

    activity_table = _require_table(data, "activity")
    _reject_unknown(
        activity_table,
        {
            "quiet_period_seconds",
            "minimum_session_seconds",
            "poll_interval_seconds",
            "idle_poll_interval_seconds",
        },
        "[activity]",
    )
    activity_defaults = ActivityConfig()
    activity = ActivityConfig(
        quiet_period_seconds=_number(
            activity_table,
            "quiet_period_seconds",
            activity_defaults.quiet_period_seconds,
            "[activity]",
        ),
        minimum_session_seconds=_number(
            activity_table,
            "minimum_session_seconds",
            activity_defaults.minimum_session_seconds,
            "[activity]",
        ),
        poll_interval_seconds=_number(
            activity_table,
            "poll_interval_seconds",
            activity_defaults.poll_interval_seconds,
            "[activity]",
        ),
        idle_poll_interval_seconds=_number(
            activity_table,
            "idle_poll_interval_seconds",
            activity_defaults.idle_poll_interval_seconds,
            "[activity]",
        ),
    )

    analysis_table = _require_table(data, "analysis")
    _reject_unknown(analysis_table, {"max_file_size_mb", "impact_max_depth"}, "[analysis]")
    analysis_defaults = AnalysisConfig()
    analysis = AnalysisConfig(
        max_file_size_mb=_number(
            analysis_table,
            "max_file_size_mb",
            analysis_defaults.max_file_size_mb,
            "[analysis]",
        ),
        impact_max_depth=_count(
            analysis_table,
            "impact_max_depth",
            analysis_defaults.impact_max_depth,
            "[analysis]",
        ),
    )

    secrets_table = _require_table(data, "secrets")
    _reject_unknown(secrets_table, {"exclude_paths"}, "[secrets]")
    secrets = SecretsConfig(
        exclude_paths=_string_list(
            secrets_table.get("exclude_paths", list(DEFAULT_SECRET_PATHS)),
            "[secrets].exclude_paths",
        )
    )

    tests_table = _require_table(data, "tests")
    _reject_unknown(tests_table, {"enabled", "command", "timeout_seconds"}, "[tests]")
    tests_defaults = TestsConfig()

    raw_enabled = tests_table.get("enabled", tests_defaults.enabled)
    if not isinstance(raw_enabled, bool):
        raise ConfigError(
            f"[tests].enabled must be true or false, found {type(raw_enabled).__name__}"
        )
    raw_command = tests_table.get("command", tests_defaults.command)
    if not isinstance(raw_command, str) or not raw_command.strip():
        raise ConfigError("[tests].command must be a non-empty string")
    timeout = _number(tests_table, "timeout_seconds", tests_defaults.timeout_seconds, "[tests]")
    if timeout == 0:
        raise ConfigError("[tests].timeout_seconds must be greater than 0")

    ignore = _string_list(data.get("ignore", list(DEFAULT_IGNORE)), "ignore")

    return Config(
        repository=RepositoryConfig(path=path),
        activity=activity,
        analysis=analysis,
        secrets=secrets,
        tests=TestsConfig(
            enabled=raw_enabled,
            command=raw_command.strip(),
            timeout_seconds=timeout,
        ),
        ignore=ignore,
    )


def load_config(repository_root: Path) -> Config:
    """Load ``.traceflow.toml`` from *repository_root*, falling back to defaults.

    A missing file is not an error — TraceFlow must work on a repository that has
    never been initialised. A malformed file is an error.
    """
    config_path = repository_root / CONFIG_FILENAME
    if not config_path.is_file():
        return default_config()

    try:
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{config_path} is not valid TOML: {exc}") from exc

    return parse_config(data)


def render_default_config() -> str:
    """Render the default configuration as TOML for ``traceflow init``.

    ``ignore`` is emitted before any section header on purpose: in TOML every key
    belongs to the most recent header, so a bare ``ignore`` written after
    ``[activity]`` would be parsed as ``activity.ignore``.
    """
    config = default_config()
    activity = config.activity
    analysis = config.analysis
    tests = config.tests
    ignored = ", ".join(f'"{item}"' for item in config.ignore)
    secrets = "\n".join(f'    "{item}",' for item in config.secrets.exclude_paths)
    return f"""# TraceFlow configuration.
# See plan.md §45 for the full set of planned options. Sections appear here as
# the phases that consume them land.

# Paths TraceFlow must never treat as repository activity. This is not a
# replacement for .gitignore — git already handles ignored paths correctly.
# It exists so TraceFlow's own writes can never trigger a session.
ignore = [{ignored}]

[repository]
path = "{config.repository.path}"

[activity]
# How long the repository must be untouched before a session is considered stable.
quiet_period_seconds = {activity.quiet_period_seconds:g}
# Ignore activity bursts shorter than this; they are usually editor noise.
minimum_session_seconds = {activity.minimum_session_seconds:g}
# Poll cadence while the repository is changing.
poll_interval_seconds = {activity.poll_interval_seconds:g}
# Poll cadence while nothing is happening, to avoid burning CPU while idle.
idle_poll_interval_seconds = {activity.idle_poll_interval_seconds:g}

[analysis]
# Files larger than this are reported but never read into memory.
max_file_size_mb = {analysis.max_file_size_mb:g}
# How many rings of dependents impact analysis expands. Stopping at the limit is
# recorded in the impact report rather than presented as a complete answer.
impact_max_depth = {analysis.impact_max_depth}

[secrets]
# Files whose contents TraceFlow must never read. They still appear in a change
# set with their path, status and line counts; only their contents are withheld.
# Matching is on paths, not content, because a value that was never read cannot
# leak through a redaction bug.
exclude_paths = [
{secrets}
]

[tests]
# Disabled by default on purpose: plan.md §22 forbids running a repository's
# tests without explicit configuration, and a test run can be expensive or
# destructive. When enabled, only the tests a session's own impact analysis
# reaches are run — never the whole suite.
enabled = {str(tests.enabled).lower()}
# The command line that runs the tests. It is read from this file only, split
# into a program and arguments, and executed without a shell.
command = "{tests.command}"
# A run that lasts longer than this is stopped and recorded as timed out.
timeout_seconds = {tests.timeout_seconds:g}
"""
