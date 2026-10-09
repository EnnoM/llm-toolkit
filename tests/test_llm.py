"""Fehlerbehandlung und Stream-Details von core/llm.py. Keine echten API-Aufrufe."""

from types import SimpleNamespace

import httpx2
import openai
import pytest
from openai.types.chat import ChatCompletionChunk

from core.llm import LLMError, complete, get_client, stream

REQUEST = httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


def raising_client(error: Exception):
    """Client, dessen create() sofort den übergebenen Fehler wirft."""

    def create(**_):
        raise error

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def status_error(code: int) -> openai.APIStatusError:
    return openai.APIStatusError(
        "Fehler", response=httpx2.Response(code, request=REQUEST), body=None
    )


def chunk(content=None, finish_reason=None, usage=None) -> ChatCompletionChunk:
    choices = [
        {"index": 0, "delta": {"content": content}, "finish_reason": finish_reason}
    ]
    return ChatCompletionChunk.model_validate(
        {
            "id": "gen-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test/model",
            "choices": choices,
            "usage": usage,
        }
    )


def test_missing_api_key_gives_clear_error(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY is missing"):
        get_client()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (status_error(401), "API key is invalid"),
        (status_error(402), "Not enough OpenRouter credits"),
        (status_error(429), "Too many requests"),
        (status_error(500), r"OpenRouter error \(500\)"),
        (openai.APIConnectionError(request=REQUEST), "Cannot reach OpenRouter"),
        (openai.APITimeoutError(request=REQUEST), "did not respond in time"),
    ],
)
def test_api_errors_become_readable(error, expected):
    messages = [{"role": "user", "content": "Hi"}]
    with pytest.raises(LLMError, match=expected):
        complete(messages, "test/model", client=raising_client(error))
    with pytest.raises(LLMError, match=expected):
        stream(messages, "test/model", client=raising_client(error))


def test_error_in_the_middle_of_a_stream_becomes_readable():
    def chunks():
        yield chunk("Hal")
        raise openai.APIError("Anbieter abgebrochen", REQUEST, body=None)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: chunks()))
    )
    response = stream([{"role": "user", "content": "Hi"}], "test/model", client=client)
    with pytest.raises(LLMError, match="Anbieter abgebrochen"):
        list(response)
    assert response.text == "Hal"


def test_stream_remembers_truncation_at_token_limit():
    chunks = [chunk("Lange Antwort"), chunk(finish_reason="length")]
    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **_: iter(chunks))
        )
    )
    response = stream([{"role": "user", "content": "Hi"}], "test/model", client=client)
    list(response)
    assert response.finish_reason == "length"
