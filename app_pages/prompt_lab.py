"""Prompt Lab: dieselbe Aufgabe mit vier Prompt-Techniken im Vergleich.

Layout nach docs/design.md: Einstellungen | Aufgabe und Ergebnisse | Kennzahlen im Verhältnis 1:2:1.
"""

import streamlit as st

from core.config import get_secret
from core.llm import LLMError
from core.models import (
    format_cost,
    format_usage,
    model_name,
    model_options,
    model_price,
)
from core.prompt_lab import (
    TASKS,
    TECHNIQUES,
    ComparisonRun,
    VariantResult,
    format_messages,
    run_comparison,
)

st.set_page_config(layout="wide")  # drei Spalten brauchen die volle Breite

st.title("Prompt Lab")
settings, main, stats = st.columns([1, 2, 1], gap="large")

with settings:
    st.markdown("##### Settings")
    task_id = st.radio(
        "Task",
        list(TASKS),
        format_func=lambda key: TASKS[key].title,
        captions=[task.description for task in TASKS.values()],
        help=(
            "Three prepared tasks on which the techniques behave differently. "
            "You can replace the input text next to this column."
        ),
    )
    techniques = st.pills(
        "Techniques",
        list(TECHNIQUES),
        selection_mode="multi",
        default=list(TECHNIQUES),
        format_func=TECHNIQUES.get,
        wrap=True,  # direkt in einer Spalte würde Streamlit sonst seitlich scrollen lassen
        help=(
            "Zero-shot sends only the task, few-shot adds solved examples first, role adds a "
            "persona as system prompt. Chain-of-thought asks the model to reason step by step "
            "before answering."
        ),
    )
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
            "All techniques run on the same model so the comparison is fair."
        ),
    )

task = TASKS[task_id]

with main:
    st.markdown("##### Task")
    st.markdown(task.instruction)
    text = st.text_area(
        "Input",
        task.sample_input,
        key=f"lab_input_{task_id}",  # eigener Text pro Aufgabe, bleibt beim Wechseln erhalten
        height=150,
        help="The text the task is applied to. Prefilled with an example you can replace.",
    )
    if task.reference:
        st.caption(f"Reference answer: {task.reference}")
    compare = st.button(
        "Compare",
        type="primary",
        icon=":material/compare_arrows:",
        disabled=not techniques or not text.strip(),
    )
    st.caption(
        "Temperature is fixed at 0, so differences come from the prompts, not chance."
    )

    if compare:
        try:
            with st.spinner("Running all techniques in parallel..."):
                # Im Session State, damit Klicks auf der Seite den Lauf nicht wiederholen
                st.session_state.lab_run = run_comparison(
                    task_id, techniques, text, model
                )
        except LLMError as err:
            st.error(str(err), icon=":material/error:")

run: ComparisonRun | None = st.session_state.get("lab_run")


LONG_ANSWER = 400  # Zeichen; längere Antworten scrollen in einem Kasten fester Höhe


def show_result(result: VariantResult) -> None:
    """Eine Ergebniskarte: Technik, Antwort, Verbrauch und Antwortzeit."""
    with st.container(border=True):
        st.markdown(f"##### {TECHNIQUES[result.technique]}")
        if result.error:
            st.error(result.error, icon=":material/error:")
            return
        # Lange Antworten (meist Chain-of-thought) scrollen, damit das Raster nicht ausufert
        height = 260 if len(result.text) > LONG_ANSWER else "content"
        with st.container(height=height, border=False):
            # Einfache Zeilenumbrüche erhalten (Markdown macht sonst Leerzeichen daraus),
            # z. B. bei einer Zeile pro Ticket in der Email triage
            st.markdown(result.text.replace("\n", "  \n"))
        st.caption(f"{format_usage(result.usage)} · {result.seconds:.1f} s")


with main:
    if run is None:
        st.caption("Pick a task and techniques, then press Compare.")
    else:
        if run.task_id != task_id or run.model != model:
            st.caption(
                f"Showing the last run: {TASKS[run.task_id].title} with {model_name(run.model)}."
            )
        # Zwei Karten pro Zeile, damit die Antworten nebeneinander lesbar bleiben
        for start in range(0, len(run.results), 2):
            for column, result in zip(
                st.columns(2), run.results[start : start + 2], strict=False
            ):
                with column:
                    show_result(result)
        # Gesendete Prompts in voller Breite, ein Reiter pro Technik
        with st.expander("Prompts sent", icon=":material/description:"):
            tabs = st.tabs([TECHNIQUES[result.technique] for result in run.results])
            for tab, result in zip(tabs, run.results, strict=True):
                with tab:
                    st.code(
                        format_messages(result.messages), language=None, wrap_lines=True
                    )

with stats:
    st.markdown("##### Last run")
    usage = run.usage if run else None
    st.metric("Cost (USD)", format_cost(usage.cost_usd, unit=False) if usage else "–")
    st.metric(
        "Tokens in",
        usage.prompt_tokens if usage else "–",
        help="Few-shot sends its examples with every request, so it needs the most input tokens.",
    )
    st.metric(
        "Tokens out",
        usage.completion_tokens if usage else "–",
        help="Chain-of-thought writes out its reasoning, so it usually produces the most output.",
    )
    st.metric("Tokens total", usage.total_tokens if usage else "–")
    st.metric(
        "Duration (s)",
        f"{run.seconds:.1f}" if run else "–",
        help="All techniques run in parallel, so the run takes about as long as the slowest one.",
    )
