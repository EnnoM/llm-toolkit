"""Chat-Logik: Verlauf, Nachrichten für die API, Kosten der Sitzung. Keine Oberfläche."""

from typing import Literal

from pydantic import BaseModel

from core.llm import Message, Usage

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Keep your answers concise."


class ChatTurn(BaseModel):
    """Eine Nachricht im Verlauf. Antworten tragen zusätzlich Modell und Verbrauch."""

    role: Literal["user", "assistant"]
    content: str
    model: str | None = None
    usage: Usage | None = None


def build_messages(system_prompt: str, history: list[ChatTurn]) -> list[Message]:
    """Nachrichten für die API: System-Prompt plus kompletter bisheriger Verlauf.

    Das Modell hat kein Gedächtnis, deshalb geht bei jeder Anfrage alles mit.
    """
    system = system_prompt.strip()
    messages = [{"role": "system", "content": system}] if system else []
    return messages + [{"role": turn.role, "content": turn.content} for turn in history]


def session_usage(history: list[ChatTurn]) -> Usage:
    """Summe von Tokens und Kosten aller Antworten im Verlauf."""
    return sum((turn.usage for turn in history if turn.usage), Usage())
