"""Chat-Logik (core/chat.py) und Chat-Seite (app_pages/chat.py). Keine echten API-Aufrufe."""

from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletionChunk
from streamlit.testing.v1 import AppTest

import core.llm
from core.chat import ChatTurn, build_messages, session_usage
from core.llm import Usage

# ------------------------------------------------------------------ Logik


def test_build_messages_puts_system_prompt_first_and_keeps_history():
    history = [
        ChatTurn(role="user", content="Hallo"),
        ChatTurn(role="assistant", content="Hi!", usage=Usage(cost_usd=0.1)),
        ChatTurn(role="user", content="Wie geht's?"),
    ]
    assert build_messages("  Sei knapp.  ", history) == [
        {"role": "system", "content": "Sei knapp."},
        {"role": "user", "content": "Hallo"},
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "Wie geht's?"},
    ]


def test_build_messages_without_system_prompt():
    history = [ChatTurn(role="user", content="Hallo")]
    assert build_messages("   ", history) == [{"role": "user", "content": "Hallo"}]


def test_session_usage_sums_all_answers():
    history = [
        ChatTurn(role="user", content="a"),
        ChatTurn(
            role="assistant", content="b", usage=Usage(total_tokens=10, cost_usd=0.0001)
        ),
        ChatTurn(role="user", content="c"),
        ChatTurn(
            role="assistant", content="d", usage=Usage(total_tokens=30, cost_usd=0.0002)
        ),
    ]
    total = session_usage(history)
    assert total.total_tokens == 40
    assert total.cost_usd == pytest.approx(0.0003)
    assert session_usage([]) == Usage()


# ------------------------------------------------------------------ Seite


def fake_client(calls: list):
    """Client, der jede Anfrage in calls festhält und 'Hallo Welt' streamt."""

    def create(**kwargs):
        calls.append(kwargs)
        base = {"id": "gen-1", "object": "chat.completion.chunk", "created": 0}
        base["model"] = kwargs["model"]
        parts = [
            {"choices": [{"index": 0, "delta": {"content": "Hallo "}}]},
            {"choices": [{"index": 0, "delta": {"content": "Welt"}}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 3,
                    "total_tokens": 15,
                    "cost": 0.00002,
                },
            },
        ]
        return iter(ChatCompletionChunk.model_validate(base | p) for p in parts)

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def open_chat() -> AppTest:
    """Startet die App angemeldet und wechselt auf die Chat-Seite."""
    at = AppTest.from_file("../app.py", default_timeout=10)
    at.session_state["authenticated"] = True
    at.run()
    return at.switch_page("app_pages/chat.py").run()


def metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_chat_page_streams_answer_and_shows_cost(monkeypatch):
    calls = []
    monkeypatch.setattr(core.llm, "get_client", lambda: fake_client(calls))
    at = open_chat()
    at.chat_input[0].set_value("Hallo").run()

    assert not at.exception
    assert any("Hallo Welt" in md.value for md in at.markdown)
    assert any(box.value == "Hallo" for box in at.info)  # Frage als Box
    assert any("0.000020 USD" in c.value for c in at.caption)
    assert metric(at, "Cost (USD)") == "0.000020"
    assert len(at.session_state["chat_history"]) == 2


def test_sidebar_contains_no_module_settings():
    # Design-Regel: Die Sidebar ist nur für die Navigation da (docs/design.md)
    at = open_chat()
    sidebar = at.sidebar
    assert not (sidebar.radio or sidebar.text_area or sidebar.slider or sidebar.button)
    assert at.radio and at.text_area and at.slider  # Einstellungen gibt es trotzdem


def test_system_prompt_change_applies_to_next_message(monkeypatch):
    calls = []
    monkeypatch.setattr(core.llm, "get_client", lambda: fake_client(calls))
    at = open_chat()
    at.chat_input[0].set_value("First question").run()
    at.text_area[0].set_value("Answer only in German.").run()
    at.chat_input[0].set_value("Second question").run()

    first, second = calls
    assert first["messages"][0]["content"] != "Answer only in German."
    assert second["messages"][0] == {
        "role": "system",
        "content": "Answer only in German.",
    }
    # Der komplette Verlauf geht mit: System, Frage, Antwort, neue Frage
    assert len(second["messages"]) == 4


def test_clearing_the_chat_resets_history_and_cost(monkeypatch):
    monkeypatch.setattr(core.llm, "get_client", lambda: fake_client([]))
    at = open_chat()
    at.chat_input[0].set_value("Hallo").run()
    next(b for b in at.button if b.label == "Clear chat").click().run()

    assert at.session_state["chat_history"] == []
    assert metric(at, "Cost (USD)") == "0.000000"


def test_missing_api_key_shows_message_instead_of_traceback(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    at = open_chat()
    at.chat_input[0].set_value("Hallo").run()

    assert not at.exception
    assert any("OPENROUTER_API_KEY is missing" in e.value for e in at.error)
    assert at.session_state["chat_history"] == []


def test_scroll_script_is_sent_and_valid(monkeypatch):
    monkeypatch.setattr(core.llm, "get_client", lambda: fake_client([]))
    at = open_chat()
    at.chat_input[0].set_value("Hallo").run()

    scripts = [
        n.proto.body for n in at.get("html") if n.proto.body.startswith("<script>")
    ]
    assert scripts, "Skript zum Scrollen ans Ende fehlt"
    body = scripts[-1]
    # Ein Klammerfehler hat das Skript schon einmal still lahmgelegt
    assert body.count("{") == body.count("}")
    assert body.count("(") == body.count(")")
    # Eigene Funktion, sonst kollidiert "const box" beim zweiten Aufruf im Browser
    assert "(() => {" in body
