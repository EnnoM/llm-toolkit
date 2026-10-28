"""Tool Use: Das Modell ruft selbst Werkzeuge auf, jeder Aufruf ist sichtbar.

Layout nach docs/design.md: Modell | Chat | Kennzahlen im Verhältnis 1:2:1, der Chat sieht aus
wie auf der Chat-Seite (gemeinsame Bausteine in core/chat_ui.py, CSS in chat.css). Werkzeuge
und Beispiele stehen in einer umrahmten Leiste über dem Chat, weil sie zur Frage gehören; damit
der Verlauf darunter trotzdem genau bis zum Eingabefeld reicht, regelt chat.css die Höhe
(Container chat_main).
Die Schleife selbst steht in core/tool_use.py, die Werkzeuge in core/tools.py.
"""

import json

import streamlit as st

from core.chat_ui import load_chat_css, scroll_chat_to_bottom, show_meta, show_question
from core.config import get_secret
from core.llm import LLMError, Usage
from core.models import format_cost, model_name, model_options, model_price
from core.page_intro import show_intro
from core.tool_use import MAX_ROUNDS, Step, ToolTurn, run
from core.tools import TOOLS

EXAMPLES = [
    "What's the weather in Hamburg?",
    "What is 17 percent of 2,340 euros?",
    "What is the latest Python release?",
    "How many days are left until Christmas Eve?",
    "Which towns in Swabia are regional centres (Oberzentren)?",
    "Explain photosynthesis in two sentences.",
]

# Symbol pro Werkzeug für die Buttons in der Leiste
TOOL_ICONS = {
    "calculate": ":material/calculate:",
    "get_current_datetime": ":material/schedule:",
    "get_weather": ":material/partly_cloudy_day:",
    "web_search": ":material/travel_explore:",
    "search_documents": ":material/find_in_page:",
}
TOOL_LABELS = {tool.name: tool.label for tool in TOOLS}

st.set_page_config(layout="wide")  # drei Spalten brauchen die volle Breite
load_chat_css()
history: list[ToolTurn] = st.session_state.setdefault("tool_history", [])


def ask_example(question: str) -> None:
    # Wird im nächsten Durchlauf wie eine getippte Frage behandelt
    st.session_state.tool_pending = question


def show_step(number: int, step: Step) -> None:
    """Ein Werkzeugaufruf: Name, Dauer, Argumente und Ergebnis als JSON."""
    status = ":red-badge[error]" if step.failed else ""
    st.markdown(
        f"**{number}. {step.tool}** :gray-badge[round {step.round}] "
        f":gray-badge[{step.seconds:.2f} s] {status}"
    )
    st.caption("Arguments")
    arguments = (
        json.dumps(step.arguments, ensure_ascii=False, indent=2)
        if isinstance(step.arguments, dict)
        else step.arguments
    )
    st.code(arguments, language="json", wrap_lines=True)
    st.caption("Result")
    result = json.dumps(step.result, ensure_ascii=False, indent=2)
    st.code(
        result,
        language="json",
        wrap_lines=True,
        height=min(300, 60 + 21 * result.count("\n")),
    )


def show_steps(steps: list[Step]) -> None:
    """Aufklappbar unter jeder Antwort, die Werkzeuge benutzt hat."""
    if not steps:
        return
    with st.expander(
        f"Tool calls ({len(steps)})", icon=":material/build:", expanded=False
    ):
        for number, step in enumerate(steps, start=1):
            show_step(number, step)


st.title("Tool Use")
show_intro("tool_use")
settings, chat, stats = st.columns([1, 2, 1], gap="large")

# Die Container mit key bekommen die CSS-Klasse st-key-<key>, darüber greift chat.css
with settings.container(key="chat_settings"):
    st.markdown("##### Model")
    default_model = get_secret("OPENROUTER_MODEL")
    options = model_options(default_model)  # nach Preis sortiert, günstigstes zuerst
    model = st.radio(
        "Model",
        options,
        index=options.index(default_model) if default_model in options else 0,
        format_func=model_name,
        # Preis unter jedem Modell; "$" maskieren, sonst liest Streamlit es als Formel
        captions=[(model_price(m) or "").replace("$", "\\$") for m in options],
        label_visibility="collapsed",
        help=(
            "All four models support tool calling. Each tool round is a separate request, "
            "so a question with tools costs two to three times as much as one without."
        ),
    )

with stats.container(key="chat_stats"):
    st.markdown("##### Session")
    summary = st.empty()  # nach einer neuen Antwort erneut gefüllt
    if st.button("Clear chat", icon=":material/delete:"):
        history.clear()
    # Unsichtbarer Platzhalter für das Skript aus scroll_chat_to_bottom
    scroll_trigger = st.empty()


def show_summary() -> None:
    total = sum((turn.usage for turn in history if turn.usage), Usage())
    with summary.container():
        st.metric(
            "Cost (USD)",
            format_cost(total.cost_usd, unit=False),
            help="All model requests of every tool round, plus embeddings of the "
            "document search.",
        )
        st.metric(
            "Tokens in",
            total.prompt_tokens,
            help="Every round resends the conversation, the tool descriptions and all "
            "tool results so far.",
        )
        st.metric("Tokens out", total.completion_tokens)
        st.metric("Tokens total", total.total_tokens)
        st.metric("Answers", sum(turn.role == "assistant" for turn in history))
        st.metric("Tool calls", sum(len(turn.steps) for turn in history))


# Mitte: Leiste mit Werkzeugen und Beispielen, darunter der Verlauf. Der Container chat_main
# ist so hoch wie das Fenster erlaubt, der Verlauf füllt den Rest (chat.css)
main = chat.container(key="chat_main")
with main.container(border=True, key="tool_bar"):
    # Untereinander statt in einer Zeile: In einer Zeile brechen die Buttons nicht um und
    # liefen bei schmalen Fenstern rechts hinaus
    enabled = st.pills(
        "Tools",
        list(TOOL_ICONS),
        selection_mode="multi",
        default=list(TOOL_ICONS),
        format_func=lambda name: f"{TOOL_ICONS[name]} {TOOL_LABELS[name]}",
        key="tool_enabled",
        wrap=True,
        help=(
            "Only the selected tools are offered to the model. Without tools it answers "
            "from its own knowledge."
        ),
    )
    with st.popover("Examples", icon=":material/lightbulb:"):
        for example in EXAMPLES:
            st.button(
                example,
                type="tertiary",
                on_click=ask_example,
                args=(example,),
                key=f"example_{example}",
            )

with main:
    if not get_secret("OPENROUTER_API_KEY"):
        st.warning(
            "OPENROUTER_API_KEY is not set, the assistant cannot answer.",
            icon=":material/key_off:",
        )
    # Eigener Scrollbereich; chat.css passt die Höhe an das Browserfenster an
    messages = st.container(
        height=450, border=False, autoscroll=True, key="chat_scroll"
    )

# Eingabefeld fest am unteren Bildschirmrand (st.bottom), ausgerichtet an der mittleren Spalte
with st.bottom, st.container(key="chat_input_row"):
    _, input_column, _ = st.columns([1, 2, 1], gap="large")
    typed = input_column.chat_input("Ask anything", submit_mode="disable")
prompt = typed or st.session_state.pop("tool_pending", None)

show_summary()

with messages:
    if not history and not prompt:
        st.caption(
            "Ask a question below or pick one of the examples above. The model decides "
            "itself whether it needs a tool; every call appears under the answer."
        )
    for turn in history:
        if turn.role == "user":
            show_question(turn.content)
        else:
            st.markdown(turn.content)
            show_meta(turn.model, turn.usage)
            show_steps(turn.steps)

if prompt:
    with messages:
        show_question(prompt)
    scroll_chat_to_bottom(scroll_trigger)
    with messages:
        # Schritte schon während der Arbeit zeigen, nicht erst am Ende
        # In einem Platzhalter, damit die Box nach der Antwort ganz verschwinden kann
        live = st.empty()
        status = live.status("Thinking …", expanded=True)
        live_steps: list[Step] = []

        def on_step(step: Step) -> None:
            live_steps.append(step)
            status.update(label=f"Calling tools … ({len(live_steps)})", expanded=True)
            with status:
                show_step(len(live_steps), step)

        try:
            result = run(prompt, history, model, enabled, on_step=on_step)
        except LLMError as err:
            status.update(label="Failed", state="error", expanded=False)
            st.error(str(err), icon=":material/error:")
        else:
            # Statusbox durch die gleiche Anzeige ersetzen wie im Verlauf
            live.empty()
            history.append(ToolTurn(role="user", content=prompt))
            if result.stopped:
                st.warning(
                    f"Stopped after {MAX_ROUNDS} tool rounds without a final answer. "
                    "Try a simpler question or another model.",
                    icon=":material/block:",
                )
                text = f"Stopped after {MAX_ROUNDS} tool rounds without a final answer."
            else:
                text = result.text
                st.markdown(text)
            history.append(
                ToolTurn(
                    role="assistant",
                    content=text,
                    model=result.model,
                    usage=result.usage,
                    steps=result.steps,
                )
            )
            show_meta(result.model, result.usage)
            show_steps(result.steps)
    scroll_chat_to_bottom(scroll_trigger)
    show_summary()
