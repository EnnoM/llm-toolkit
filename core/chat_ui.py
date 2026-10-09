"""Bausteine für Chat-Seiten (Chat, RAG Chatbot), damit beide gleich aussehen.

Anders als der Rest von core/ zeichnet dieses Modul Oberfläche. Das Layout dazu steht in
app_pages/chat.css und greift über die keys der Container (chat_scroll, chat_settings, chat_stats,
chat_input_row), die beide Seiten deshalb gleich benennen.
"""

from pathlib import Path

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from core.llm import Usage
from core.models import format_usage

CHAT_CSS = Path(__file__).parent.parent / "app_pages" / "chat.css"


def load_chat_css() -> None:
    """Nur der Verlauf scrollt, Titel und Seitenspalten bleiben stehen (Details in chat.css)."""
    st.html(CHAT_CSS)


def show_question(text: str) -> None:
    """Frage als gefüllte, rechts eingerückte Box (st.info ohne Icon), wie in Claude Code."""
    _, box = st.columns([1, 5])
    box.info(text)


def show_meta(model_name: str | None, usage: Usage) -> None:
    """Graue Zeile unter einer Antwort: Modell, Tokens, Kosten."""
    st.caption(f"{model_name} · {format_usage(usage)}")


def scroll_chat_to_bottom(trigger: DeltaGenerator) -> None:
    """Scrollt den Verlauf ans Ende, auch wenn man vorher nach oben gescrollt hatte.

    Streamlits autoscroll folgt neuen Nachrichten nur, wenn man schon ganz unten ist. Das
    Skript ist fest im Code und enthält keine Nutzereingaben. trigger ist ein unsichtbarer
    Platzhalter (st.empty), in den das Skript geschrieben wird.
    """
    # Neuer Zähler bei jedem Aufruf, sonst führt Streamlit dasselbe Skript nicht erneut aus
    st.session_state.scroll_runs = st.session_state.get("scroll_runs", 0) + 1
    # In einer Funktion, damit "box" lokal bleibt: Ein zweites globales "const box" im selben
    # Dokument bricht sonst mit "already been declared" ab.
    trigger.html(
        f"<script>/* {st.session_state.scroll_runs} */ (() => {{ "
        "const box = document.querySelector('.st-key-chat_scroll'); "
        "if (box) box.scrollTop = box.scrollHeight; })();</script>",
        unsafe_allow_javascript=True,
    )
