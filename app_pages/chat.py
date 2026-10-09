"""Chat: gestreamte Antworten, System-Prompt, Tokens und Kosten pro Antwort und Sitzung.

Layout nach docs/design.md: Einstellungen | Chat | Kennzahlen im Verhältnis 1:2:1.
"""

import streamlit as st

from core.chat import DEFAULT_SYSTEM_PROMPT, ChatTurn, build_messages, session_usage
from core.chat_ui import load_chat_css, scroll_chat_to_bottom, show_meta, show_question
from core.config import get_secret
from core.llm import DEFAULT_MAX_TOKENS, LLMError, stream
from core.models import (
    format_cost,
    model_name,
    model_options,
    model_price,
)

st.set_page_config(layout="wide")  # drei Spalten brauchen die volle Breite
load_chat_css()
st.session_state.setdefault("chat_history", [])
history: list[ChatTurn] = st.session_state.chat_history

st.title("Chat")
settings, chat, stats = st.columns([1, 2, 1], gap="large")

# Die Container mit key bekommen die CSS-Klasse st-key-<key>, darüber greift chat.css
with settings.container(key="chat_settings"):
    st.markdown("##### Settings")
    default_model = get_secret("OPENROUTER_MODEL")
    options = model_options(default_model)  # nach Preis sortiert, günstigstes zuerst
    model = st.radio(
        "Model",
        options,
        index=options.index(default_model) if default_model in options else 0,
        format_func=model_name,
        # Preis unter jedem Modell; "$" maskieren, sonst liest Streamlit es als Formel
        captions=[(model_price(m) or "").replace("$", "\\$") for m in options],
        help=(
            "Prices differ per model and are listed in USD per million input / output tokens. "
            "OpenRouter, the platform used here, offers hundreds of models from many "
            "providers (openrouter.ai/models)."
        ),
    )
    system_prompt = st.text_area(
        "System prompt",
        DEFAULT_SYSTEM_PROMPT,
        height=114,  # inkl. Beschriftung; drei Zeilen des Standard-Prompts ohne Scrollen
        help=(
            "A hidden instruction sent ahead of the conversation that sets the model's role, "
            "tone and rules. Changes apply from the next message on."
        ),
    )
    temperature = st.slider(
        "Temperature",
        0.0,
        1.5,
        0.7,
        step=0.1,
        help=(
            "Controls how much randomness goes into choosing each next word: 0 gives "
            "focused, nearly identical answers, higher values more varied and creative ones. "
            "Above 1, answers can become erratic."
        ),
    )
with stats.container(key="chat_stats"):
    st.markdown("##### Session")
    summary = st.empty()  # nach einer neuen Antwort erneut gefüllt
    # Setzt Verlauf und alle Zahlen dieser Spalte zurück, deshalb steht der Button hier
    if st.button("Clear chat", icon=":material/delete:"):
        history.clear()
    # Unsichtbarer Platzhalter für das Skript aus scroll_chat_to_bottom (core/chat_ui.py)
    scroll_trigger = st.empty()


def show_summary() -> None:
    total = session_usage(history)
    with summary.container():
        st.metric("Cost (USD)", format_cost(total.cost_usd, unit=False))
        st.metric(
            "Tokens in",
            total.prompt_tokens,
            help=(
                "The model has no memory: each message resends the full history, "
                "so the input tokens grow with every turn."
            ),
        )
        st.metric("Tokens out", total.completion_tokens)
        st.metric("Tokens total", total.total_tokens)
        st.metric("Responses", sum(turn.role == "assistant" for turn in history))


with chat:
    if not get_secret("OPENROUTER_API_KEY"):
        st.warning(
            "OPENROUTER_API_KEY is not set, the chat cannot answer.",
            icon=":material/key_off:",
        )
    # Eigener Scrollbereich; chat.css passt die Höhe an das Browserfenster an
    messages = st.container(
        height=450, border=False, autoscroll=True, key="chat_scroll"
    )

# Eingabefeld fest am unteren Bildschirmrand (st.bottom), ausgerichtet an der mittleren Spalte
with st.bottom, st.container(key="chat_input_row"):
    _, input_column, _ = st.columns([1, 2, 1], gap="large")
    prompt = input_column.chat_input("Message the model", submit_mode="disable")

show_summary()

with messages:
    if not history and not prompt:
        st.caption("No messages yet. Ask a question below.")
    for turn in history:
        if turn.role == "user":
            show_question(turn.content)
        else:
            st.markdown(turn.content)
            show_meta(turn.model, turn.usage)

if prompt:
    history.append(ChatTurn(role="user", content=prompt))
    with messages:
        show_question(prompt)
    # Neue Frage sofort zeigen; danach hält Streamlits autoscroll die Antwort unten sichtbar
    scroll_chat_to_bottom(scroll_trigger)
    with messages:
        try:
            response = stream(
                build_messages(system_prompt, history), model, temperature=temperature
            )
            st.write_stream(response)
            if not response.text:
                raise LLMError("The model returned no text. Please send again.")
        except LLMError as err:
            history.pop()  # Frage ohne Antwort nicht im Verlauf behalten
            st.error(str(err), icon=":material/error:")
        else:
            history.append(
                ChatTurn(
                    role="assistant",
                    content=response.text,
                    model=response.model,
                    usage=response.usage,
                )
            )
            show_meta(response.model, response.usage)
            if response.finish_reason == "length":
                st.caption(
                    f"Answer cut off after {DEFAULT_MAX_TOKENS} tokens (cost limit)."
                )
    # Auch nach fertiger Antwort (oder Fehlermeldung) ans Ende, falls zwischendurch gescrollt
    scroll_chat_to_bottom(scroll_trigger)
    show_summary()
