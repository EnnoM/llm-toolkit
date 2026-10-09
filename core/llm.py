"""LLM-Zugang über OpenRouter: normale und gestreamte Antworten inkl. Tokens und Kosten.

OpenRouter ist OpenAI-kompatibel, daher genügt das openai-SDK mit eigener base_url.
Die Kosten berechnet OpenRouter selbst und liefert sie im Feld usage.cost mit.
"""

from collections.abc import Iterable, Iterator

import openai
from openai import OpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel

from core.config import get_secret

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MAX_TOKENS = 1024  # Kostenbremse pro Antwort

Message = dict[str, str]  # {"role": "user", "content": "..."}

# Verständliche Texte zu den HTTP-Statuscodes, die OpenRouter dokumentiert.
# Englisch, weil sie in den (englischen) Modulen angezeigt werden.
STATUS_MESSAGES = {
    401: "The OpenRouter API key is invalid. Check OPENROUTER_API_KEY.",
    402: "Not enough OpenRouter credits. Top up or raise the limit.",
    403: "The input was rejected by the model provider's moderation.",
    408: "OpenRouter did not respond in time. Please try again.",
    429: "Too many requests. Wait a moment and try again.",
    502: "The model is currently unavailable. Pick another model or try again later.",
    503: "No provider is currently available for this model. Pick another model.",
}


class LLMError(RuntimeError):
    """Fehler mit einer Meldung, die man in der Oberfläche direkt anzeigen kann."""


def explain_error(err: openai.OpenAIError) -> str:
    """Übersetzt Fehler des openai-SDK in eine verständliche deutsche Meldung."""
    if isinstance(err, openai.APITimeoutError):  # Unterklasse von APIConnectionError
        return STATUS_MESSAGES[408]
    if isinstance(err, openai.APIConnectionError):
        return "Cannot reach OpenRouter. Check your internet connection."
    status = getattr(err, "status_code", None)
    if status in STATUS_MESSAGES:
        return STATUS_MESSAGES[status]
    return f"OpenRouter error{f' ({status})' if status else ''}: {err}"


class Usage(BaseModel):
    """Token-Verbrauch und Kosten einer oder mehrerer Anfragen."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0  # OpenRouter rechnet in Credits, 1 Credit = 1 USD

    def __add__(self, other: "Usage") -> "Usage":
        """Summiert den Verbrauch, z. B. für die Gesamtkosten einer Sitzung."""
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


class ToolCall(BaseModel):
    """Aufrufwunsch des Modells: welches Werkzeug mit welchen Argumenten.

    arguments ist ein JSON-Text, so wie das Modell ihn geschrieben hat. Er kann ungültig sein
    und wird erst beim Ausführen geprüft (core/tool_use.py).
    """

    id: str
    name: str
    arguments: str


class LLMResponse(BaseModel):
    """Vollständige Antwort einer nicht gestreamten Anfrage."""

    text: str
    model: str
    usage: Usage
    # Nur gefüllt, wenn die Anfrage Werkzeuge (tools=...) angeboten hat und das Modell eins
    # aufrufen möchte. Dann ist text oft leer.
    tool_calls: list[ToolCall] = []


def parse_usage(usage: CompletionUsage | None) -> Usage:
    """Übersetzt das usage-Objekt der API in unser Usage-Modell."""
    if usage is None:
        return Usage()
    # "cost" ist ein OpenRouter-Zusatzfeld; das openai-SDK behält unbekannte Felder bei.
    cost = getattr(usage, "cost", None) or 0.0
    return Usage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        cost_usd=float(cost),
    )


def get_client() -> OpenAI:
    """Erzeugt einen openai-Client, der auf OpenRouter zeigt."""
    api_key = get_secret("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMError(
            "OPENROUTER_API_KEY is missing. Add it to .env or the app secrets."
        )
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


def default_model() -> str:
    """Standardmodell aus OPENROUTER_MODEL."""
    model = get_secret("OPENROUTER_MODEL")
    if not model:
        raise LLMError(
            "OPENROUTER_MODEL is missing. Add it to .env or the app secrets."
        )
    return model


def complete(
    messages: list[Message],
    model: str | None = None,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    client: OpenAI | None = None,
    **kwargs,
) -> LLMResponse:
    """Schickt die Nachrichten ab und wartet auf die komplette Antwort."""
    client = client or get_client()
    try:
        response = client.chat.completions.create(
            model=model or default_model(),
            messages=messages,
            max_tokens=max_tokens,
            **kwargs,
        )
    except openai.OpenAIError as err:
        raise LLMError(explain_error(err)) from err
    message = response.choices[0].message
    return LLMResponse(
        text=message.content or "",
        model=response.model,
        usage=parse_usage(response.usage),
        tool_calls=[
            ToolCall(
                id=call.id, name=call.function.name, arguments=call.function.arguments
            )
            for call in message.tool_calls or []
            if call.type == "function"
        ],
    )


class EmbeddingResponse(BaseModel):
    """Ein Embedding (Zahlenreihe) pro Eingabetext, in derselben Reihenfolge."""

    vectors: list[list[float]]
    model: str
    usage: Usage


# OpenRouter nimmt viele Texte pro Anfrage; kleinere Pakete halten die Anfragen kurz
EMBEDDING_BATCH_SIZE = 100


def embed(
    texts: list[str], model: str, *, client: OpenAI | None = None
) -> EmbeddingResponse:
    """Rechnet Texte in Embeddings um, in Paketen zu EMBEDDING_BATCH_SIZE Texten."""
    client = client or get_client()
    vectors, usage = [], Usage()
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        try:
            response = client.embeddings.create(
                model=model, input=texts[start : start + EMBEDDING_BATCH_SIZE]
            )
        except openai.OpenAIError as err:
            raise LLMError(explain_error(err)) from err
        # Die API liefert einen index pro Text; danach sortieren statt der Reihenfolge trauen
        vectors += [
            item.embedding for item in sorted(response.data, key=lambda d: d.index)
        ]
        # Embeddings haben keine Ausgabetokens; cost ist wieder das OpenRouter-Zusatzfeld
        tokens = response.usage.prompt_tokens if response.usage else 0
        cost = getattr(response.usage, "cost", None) or 0.0
        usage += Usage(prompt_tokens=tokens, total_tokens=tokens, cost_usd=float(cost))
    return EmbeddingResponse(vectors=vectors, model=model, usage=usage)


class StreamResponse:
    """Gestreamte Antwort: liefert beim Iterieren Textstücke.

    Nach dem Durchlauf stehen text, usage und finish_reason bereit. Passt direkt in
    st.write_stream(response); danach z. B. response.usage.cost_usd anzeigen.
    """

    def __init__(self, chunks: Iterable[ChatCompletionChunk], model: str) -> None:
        self._chunks = chunks
        self.model = model
        self.text = ""
        self.usage = Usage()
        self.finish_reason: str | None = None  # "length" = am Token-Limit abgeschnitten

    def __iter__(self) -> Iterator[str]:
        try:
            for chunk in self._chunks:
                # OpenRouter meldet das tatsächlich genutzte Modell in jedem Chunk.
                self.model = chunk.model or self.model
                # usage kommt nur im letzten Chunk.
                if chunk.usage:
                    self.usage = parse_usage(chunk.usage)
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                self.finish_reason = choice.finish_reason or self.finish_reason
                if choice.delta.content:
                    self.text += choice.delta.content
                    yield choice.delta.content
        except openai.OpenAIError as err:
            # Fehler können auch mitten im Stream kommen, nicht nur beim Verbinden.
            raise LLMError(explain_error(err)) from err


def stream(
    messages: list[Message],
    model: str | None = None,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    client: OpenAI | None = None,
    **kwargs,
) -> StreamResponse:
    """Wie complete(), liefert die Antwort aber Stück für Stück."""
    client = client or get_client()
    model = model or default_model()
    try:
        chunks = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
    except openai.OpenAIError as err:
        raise LLMError(explain_error(err)) from err
    return StreamResponse(chunks, model=model)
