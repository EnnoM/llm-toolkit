"""Prompt-Labor (core/prompt_lab.py, app_pages/prompt_lab.py). Keine echten API-Aufrufe."""

import threading
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion
from streamlit.testing.v1 import AppTest

import core.llm
from core.prompt_lab import (
    STEP_BY_STEP,
    TASKS,
    TECHNIQUES,
    build_messages,
    format_messages,
    run_comparison,
)

ALL_TASKS = list(TASKS)


def as_text(messages) -> str:
    return "\n".join(m["content"] for m in messages)


# ------------------------------------------------------------------ Vorlagen (ohne API)


def test_there_are_at_least_three_example_tasks():
    assert len(TASKS) >= 3
    for task in TASKS.values():
        assert len(task.examples) >= 2, f"{task.title}: zu wenige Few-Shot-Beispiele"


@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_few_shot_prompt_contains_all_examples(task_id):
    task = TASKS[task_id]
    messages = build_messages(task, "few_shot", task.sample_input)
    prompt = as_text(messages)
    for example in task.examples:
        assert example.input in prompt
        assert example.output in prompt
    # Beispiele als frühere Runden: Antworten stehen als assistant-Nachrichten da
    assert [m["content"] for m in messages if m["role"] == "assistant"] == [
        example.output for example in task.examples
    ]
    # Die eigentliche Aufgabe kommt zuletzt
    assert messages[-1]["role"] == "user"
    assert task.sample_input in messages[-1]["content"]


@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_chain_of_thought_prompt_contains_step_by_step_instruction(task_id):
    task = TASKS[task_id]
    messages = build_messages(task, "chain_of_thought", task.sample_input)
    assert STEP_BY_STEP in as_text(messages)
    # Nur Chain-of-thought bekommt die Anweisung
    for other in ("zero_shot", "few_shot", "role"):
        assert STEP_BY_STEP not in as_text(
            build_messages(task, other, task.sample_input)
        )


@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_role_prompt_sends_the_persona_as_system_message(task_id):
    task = TASKS[task_id]
    messages = build_messages(task, "role", task.sample_input)
    assert messages[0] == {"role": "system", "content": task.role}


@pytest.mark.parametrize("task_id", ALL_TASKS)
def test_zero_shot_sends_only_instruction_and_input(task_id):
    task = TASKS[task_id]
    messages = build_messages(task, "zero_shot", "  My own text.  ")
    assert messages == [
        {"role": "user", "content": f"{task.instruction}\n\nMy own text."}
    ]


def test_format_messages_shows_role_and_content():
    text = format_messages(
        [{"role": "system", "content": "A"}, {"role": "user", "content": "B"}]
    )
    assert text == "[system]\nA\n\n[user]\nB"


# ------------------------------------------------------------------ Ablauf (nachgebauter Client)


def fake_client(calls: list):
    """Antwortet auf jede Anfrage mit 'OK' und festem Verbrauch; merkt sich die Aufrufe."""
    lock = threading.Lock()  # die Varianten laufen in eigenen Threads

    def create(**kwargs):
        with lock:
            calls.append(kwargs)
        return ChatCompletion.model_validate(
            {
                "id": "gen-1",
                "object": "chat.completion",
                "created": 0,
                "model": kwargs["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "OK"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "cost": 0.00001,
                },
            }
        )

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def test_comparison_runs_every_technique_with_the_same_model():
    calls = []
    run = run_comparison(
        "word_problem",
        list(TECHNIQUES),
        "2 + 2?",
        "test/model",
        client=fake_client(calls),
    )
    assert [r.technique for r in run.results] == list(TECHNIQUES)
    assert {c["model"] for c in calls} == {"test/model"}
    assert {c["temperature"] for c in calls} == {0.0}
    assert all(r.text == "OK" and r.error is None for r in run.results)
    assert run.usage.cost_usd == pytest.approx(0.00004)
    assert run.usage.total_tokens == 60


def test_failed_variant_reports_its_error_instead_of_crashing():
    def create(**_):
        raise core.llm.openai.APIConnectionError(request=None)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    run = run_comparison("summary", ["zero_shot"], "text", "test/model", client=client)
    assert (
        run.results[0].error
        == "Cannot reach OpenRouter. Check your internet connection."
    )


# ------------------------------------------------------------------ Seite


def open_lab() -> AppTest:
    at = AppTest.from_file("../app.py", default_timeout=10)
    at.session_state["authenticated"] = True
    at.run()
    return at.switch_page("app_pages/prompt_lab.py").run()


def test_compare_shows_one_card_per_technique_and_totals(monkeypatch):
    calls = []
    monkeypatch.setattr(core.llm, "get_client", lambda: fake_client(calls))
    monkeypatch.setattr("core.prompt_lab.get_client", lambda: fake_client(calls))
    at = open_lab()
    assert not at.exception
    next(b for b in at.button if b.label == "Compare").click().run()

    assert not at.exception
    assert len(calls) == 4
    headings = [md.value for md in at.markdown if md.value.startswith("##### ")]
    for label in TECHNIQUES.values():
        assert f"##### {label}" in headings
    assert next(m.value for m in at.metric if m.label == "Cost (USD)") == "0.000040"
    # Gesendete Prompts: ein Reiter pro Technik
    assert [tab.label for tab in at.tabs] == list(TECHNIQUES.values())


def test_sidebar_contains_no_module_settings():
    at = open_lab()
    sidebar = at.sidebar
    assert not (sidebar.radio or sidebar.text_area or sidebar.button)
