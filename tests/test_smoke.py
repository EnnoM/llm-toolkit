"""Smoke-Tests: core-Module importierbar, Kosten stimmen. Keine echten API-Aufrufe."""

import importlib
from types import SimpleNamespace

import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from core.llm import Usage, complete, parse_usage, stream


def fake_client(result):
    """Baut ein Objekt mit chat.completions.create(), das result zurückgibt."""
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: result))
    )


@pytest.mark.parametrize("module", ["core.config", "core.llm", "core.modules"])
def test_core_modules_importable(module):
    importlib.import_module(module)


def test_parse_usage_reads_openrouter_cost():
    usage = CompletionUsage(
        prompt_tokens=1200, completion_tokens=300, total_tokens=1500, cost=0.00042
    )
    assert parse_usage(usage) == Usage(
        prompt_tokens=1200, completion_tokens=300, total_tokens=1500, cost_usd=0.00042
    )


def test_parse_usage_without_data_is_zero():
    assert parse_usage(None) == Usage()
    usage = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert parse_usage(usage).cost_usd == 0.0


def test_usage_sum():
    first = Usage(prompt_tokens=100, completion_tokens=50, cost_usd=0.0001)
    second = Usage(prompt_tokens=200, completion_tokens=25, cost_usd=0.0002)
    total = first + second
    assert total.prompt_tokens == 300
    assert total.completion_tokens == 75
    assert total.cost_usd == pytest.approx(0.0003)


def test_complete_returns_text_and_cost():
    response = ChatCompletion.model_validate(
        {
            "id": "gen-1",
            "object": "chat.completion",
            "created": 0,
            "model": "test/model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hallo"},
                }
            ],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": 2,
                "total_tokens": 10,
                "cost": 0.00001,
            },
        }
    )
    result = complete(
        [{"role": "user", "content": "Hi"}], "test/model", client=fake_client(response)
    )
    assert result.text == "Hallo"
    assert result.usage.total_tokens == 10
    assert result.usage.cost_usd == pytest.approx(0.00001)


def test_stream_collects_text_and_cost():
    base = {
        "id": "gen-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test/model",
    }
    chunks = [
        ChatCompletionChunk.model_validate(
            {**base, "choices": [{"index": 0, "delta": {"content": "Hal"}}]}
        ),
        ChatCompletionChunk.model_validate(
            {**base, "choices": [{"index": 0, "delta": {"content": "lo"}}]}
        ),
        # Letzter Chunk: OpenRouter liefert hier usage inkl. cost, choices ist leer.
        ChatCompletionChunk.model_validate(
            {
                **base,
                "choices": [],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 2,
                    "total_tokens": 10,
                    "cost": 0.00002,
                },
            }
        ),
    ]
    response = stream(
        [{"role": "user", "content": "Hi"}],
        "test/model",
        client=fake_client(iter(chunks)),
    )
    assert list(response) == ["Hal", "lo"]
    assert response.text == "Hallo"
    assert response.usage.cost_usd == pytest.approx(0.00002)
