"""Intent versus actual change (plan.md §63).

The parser and the verdict ladder are pure functions over a session's own records, so
most of these tests need no repository at all. The end-to-end tests drive the real
`analyze` command, because the artifact — not the function — is the product.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from traceflow.analysis.intent import (
    RELATEDNESS_LABELS,
    ModuleFacts,
    Relatedness,
    compare_intent,
    intent_from_json,
    parse_tokens,
)
from traceflow.cli import EXIT_OK, main

TASK = "Add rate limiting to the login endpoint."


def _facts(path: str, symbols: tuple[str, ...] = (), imports: tuple[str, ...] = ()) -> ModuleFacts:
    return ModuleFacts(path=path, symbols=symbols, imports=imports)


def test_tokens_drop_fillers_and_keep_identifiers() -> None:
    tokens = parse_tokens(TASK)
    assert "rate" in tokens
    assert "limiting" in tokens
    assert "login" in tokens
    assert "endpoint" in tokens
    assert "add" not in tokens
    assert "the" not in tokens


def test_underscores_survive_tokenisation() -> None:
    assert parse_tokens("Fix the rate_limit module") == ("rate_limit", "module")


def test_an_empty_task_parses_to_nothing() -> None:
    assert parse_tokens("") == ()
    assert parse_tokens("please and thank you") == ()
    assert parse_tokens("add the new code") == ()


def test_a_task_whose_words_cover_the_change_is_related() -> None:
    comparison = compare_intent(
        TASK,
        (
            _facts("auth/routes.py", symbols=("routes.login",)),
            _facts("middleware/rate_limit.py", imports=("middleware.rate_limit",)),
        ),
    )
    assert comparison.relatedness is Relatedness.RELATED
    assert comparison.related_files == ("auth/routes.py", "middleware/rate_limit.py")
    assert all(item.related for item in comparison.files)


def test_an_inflected_word_finds_the_identifier() -> None:
    """ "limiting" must find ``rate_limit`` — §63's example task uses the -ing form."""
    comparison = compare_intent(TASK, (_facts("middleware/rate_limit.py"),))
    assert any(match.token == "limiting" for item in comparison.files for match in item.matches)


def test_a_word_inside_a_file_name_matches_the_file() -> None:
    comparison = compare_intent("Add login throttling", (_facts("auth/login.py"),))
    assert any(match.kind == "path" for item in comparison.files for match in item.matches)
    assert comparison.relatedness is Relatedness.RELATED


def test_a_changed_symbol_is_matched_through_its_owner() -> None:
    comparison = compare_intent(
        "Add login throttling", (_facts("auth/routes.py", symbols=("routes.login",)),)
    )
    assert comparison.files[0].related is True
    assert any(match.kind == "symbol" for match in comparison.files[0].matches)


def test_a_file_no_word_reaches_is_the_scope_expansion() -> None:
    """§20's "1 apparently unrelated": one covered file, one the task never touched on."""
    comparison = compare_intent(
        "Add rate limiting to the login endpoint",
        (
            _facts("auth/routes.py", symbols=("routes.login",)),
            _facts("billing/invoice.py", symbols=("invoice.total",)),
        ),
    )
    assert comparison.relatedness is Relatedness.SCOPE_EXPANSION
    assert comparison.related_files == ("auth/routes.py",)
    assert comparison.unrelated_files == ("billing/invoice.py",)


def test_no_file_shares_a_word_with_the_task() -> None:
    comparison = compare_intent("Add rate limiting", (_facts("docs/readme.md"),))
    assert comparison.relatedness is Relatedness.NO_RELATIONSHIP
    assert all(not item.related for item in comparison.files)


def test_a_stopword_only_task_is_not_compared() -> None:
    comparison = compare_intent("please refactor the code", (_facts("auth/routes.py"),))
    assert comparison.relatedness is Relatedness.NOT_COMPARED
    assert comparison.notes


def test_a_session_with_no_changes_is_not_compared() -> None:
    comparison = compare_intent(TASK, ())
    assert comparison.relatedness is Relatedness.NOT_COMPARED


def test_the_artifact_round_trips() -> None:
    comparison = compare_intent(
        TASK,
        (
            _facts("auth/routes.py", symbols=("routes.login",)),
            _facts("middleware/rate_limit.py"),
        ),
    )
    payload = comparison.to_json()
    restored = intent_from_json(json.loads(json.dumps(payload)))

    assert restored is not None
    assert restored.task == comparison.task
    assert restored.relatedness is comparison.relatedness
    assert restored.tokens == comparison.tokens
    assert restored.files == comparison.files
    assert restored.unmatched == comparison.unmatched


def test_a_malformed_payload_degrades_to_none() -> None:
    assert intent_from_json(None) is None
    assert intent_from_json({}) is None
    assert intent_from_json({"relatedness": "no such verdict"}) is None
    assert intent_from_json({"relatedness": 7}) is None


def test_labels_use_the_plans_words() -> None:
    assert RELATEDNESS_LABELS[Relatedness.RELATED] == "potentially related"
    assert RELATEDNESS_LABELS[Relatedness.SCOPE_EXPANSION] == "potential scope expansion"
    assert RELATEDNESS_LABELS[Relatedness.NO_RELATIONSHIP] == "no detected relationship"
    assert RELATEDNESS_LABELS[Relatedness.NOT_COMPARED] == "not compared"


# --------------------------------------------------------------------------- end to end


@pytest.fixture
def auth_repo(repo_root: Path) -> Path:
    """A tiny auth tree, so the task's words have something to land on."""
    (repo_root / "auth").mkdir()
    (repo_root / "auth" / "routes.py").write_text(
        "def login():\n    return 1\n", encoding="utf-8", newline="\n"
    )
    from tests.conftest import commit

    commit(repo_root, "add auth")
    return repo_root


def test_analyze_records_the_task_and_the_comparison(
    auth_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The first analyze on a fresh repository captures the baseline; the second one
    measures against it and records the session."""
    from tests.conftest import write_file

    assert main(["analyze", str(auth_repo)]) == EXIT_OK
    capsys.readouterr()

    write_file(auth_repo, "auth/routes.py", "def login():\n    return 2\n")

    assert main(["analyze", str(auth_repo), "--intent", TASK]) == EXIT_OK

    sessions_dir = auth_repo / ".traceflow" / "sessions"
    session_dir = next(sessions_dir.iterdir())
    payload = json.loads((session_dir / "intent.json").read_text(encoding="utf-8"))

    assert payload["task"] == TASK
    assert payload["relatedness"] == "related"
    assert payload["changed_files"] == 1


def test_the_intent_command_attaches_a_task_to_a_recorded_session(
    auth_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tests.conftest import write_file

    assert main(["analyze", str(auth_repo)]) == EXIT_OK  # baseline
    capsys.readouterr()

    write_file(auth_repo, "auth/routes.py", "def login():\n    return 2\n")
    assert main(["analyze", str(auth_repo)]) == EXIT_OK  # the session
    capsys.readouterr()

    assert main(["intent", "--task", TASK, str(auth_repo)]) == EXIT_OK

    output = capsys.readouterr().out
    assert "potentially related" in output

    sessions_dir = auth_repo / ".traceflow" / "sessions"
    session_dir = next(sessions_dir.iterdir())
    payload = json.loads((session_dir / "intent.json").read_text(encoding="utf-8"))
    assert payload["task"] == TASK


def test_a_task_known_only_afterwards_still_reaches_the_delivery(
    auth_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tests.conftest import write_file
    from traceflow.ui.delivery import load_delivery
    from traceflow.watcher.session import SessionStore

    assert main(["analyze", str(auth_repo)]) == EXIT_OK  # baseline
    write_file(auth_repo, "auth/routes.py", "def login():\n    return 2\n")
    assert main(["analyze", str(auth_repo)]) == EXIT_OK  # the session
    assert main(["intent", "--task", TASK, str(auth_repo)]) == EXIT_OK
    capsys.readouterr()

    store = SessionStore(auth_repo)
    session_id = store.list_sessions()[-1].session_id
    delivery = load_delivery(store, session_id)

    assert delivery is not None
    assert delivery.intent is not None
    assert delivery.intent.relatedness == "related"
    assert delivery.intent.label == "potentially related"
