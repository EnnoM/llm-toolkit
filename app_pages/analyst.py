"""AI Data Analyst: Frage in normaler Sprache -> SQL -> Tabelle, Diagramm und ein Satz Antwort.

Layout nach docs/design.md: Einstellungen | Frage und Ergebnis | Kennzahlen im Verhältnis 1:2:1.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from core.analyst import (
    MAX_ROWS,
    AnalystResult,
    ask,
    choose_chart,
    connect,
    requested_chart,
    summarize,
)
from core.config import get_secret
from core.llm import LLMError, Usage
from core.models import (
    format_cost,
    format_usage,
    model_name,
    model_options,
    model_price,
)
from core.page_intro import show_intro

st.set_page_config(layout="wide")  # drei Spalten brauchen die volle Breite

EXAMPLES = [
    "Total revenue per year",
    "Revenue per category in 2024",
    "Orders per month in 2025",
    "Orders per sales channel",
    "Top 5 products by units sold",
    # Zeigt den Schutz: Das Modell darf es versuchen, ausgeführt wird es nie
    "Delete all orders from 2023",
]

# Ask-Button so hoch wie das mitwachsende Eingabefeld (Streamlit kann das nicht selbst)
st.html(Path(__file__).with_name("analyst.css"))

st.title("AI Data Analyst")
show_intro("ki_datenanalyst")
settings, main, stats = st.columns([1, 2, 1], gap="large")


def use_example(example: str) -> None:
    """Übernimmt das gewählte Beispiel ins Fragefeld."""
    st.session_state.analyst_question = example


with settings:
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
            "Prices are in USD per million input / output tokens. The model writes the SQL "
            "and the one-sentence answer."
        ),
    )


# Tabellen in der Reihenfolge, in der man sie liest: wer kauft, was, wann, wie viel
TABLE_NOTES = {
    "customers": "One row per customer: country, city and signup date. "
    "Linked to orders by customer_id.",
    "products": "One row per product with its category and current list price in EUR. "
    "Linked to order_items by product_id.",
    "orders": "One row per order: date and sales channel. Linked to customers by "
    "customer_id and to order_items by order_id.",
    "order_items": "One row per product in an order. The revenue of a line is quantity × "
    "unit_price, the price paid at the time.",
}


@st.cache_data(show_spinner=False)
def dataset_tables() -> dict[str, pd.DataFrame]:
    """Alle Zeilen jeder Tabelle, einmal pro Serverprozess (rund 20 000 Zeilen insgesamt)."""
    cursor = connect().cursor()
    return {t: cursor.execute(f"SELECT * FROM {t}").df() for t in TABLE_NOTES}


def show_dataset() -> None:
    """Ein Tab pro Tabelle mit allen Zeilen; sortieren und suchen kann die Tabelle selbst."""
    tables = dataset_tables()
    # Nur der Name im Tab: Mit Zeilenzahl liefen die Tabs in schmalen Spalten rechts hinaus
    tabs = st.tabs(list(tables))
    for tab, (name, table) in zip(tabs, tables.items()):
        with tab:
            st.caption(f"{len(table):,} rows. {TABLE_NOTES[name]}")
            st.dataframe(
                table,
                hide_index=True,
                height=350,
                # DuckDB-Datumswerte kommen als Zeitstempel; die Uhrzeit ist immer 00:00
                column_config={
                    column: st.column_config.DateColumn(format="YYYY-MM-DD")
                    for column in table.columns
                    if pd.api.types.is_datetime64_any_dtype(table[column])
                },
            )


def show_chart(result: AnalystResult) -> None:
    """Gewünschtes Diagramm, wenn es zur Tabelle passt, sonst die feste Regel."""
    table = result.table
    spec = requested_chart(table, result.chart_request)
    if result.chart_request and spec is None:
        st.caption("The requested chart does not fit the result, showing the default.")
    spec = spec or choose_chart(table)
    if spec is None:
        return
    columns = [c for c in (spec.x, spec.y, spec.color) if c]
    data = table[columns].copy()
    args = {"x": spec.x, "y": spec.y, "color": spec.color}
    if spec.kind in ("bar", "stacked_bar"):
        # Zahlen wie 2024 als Text, sonst steht „2,024“ an der Achse; Reihenfolge wie im SQL
        data[spec.x] = data[spec.x].astype(str)
        # Mit Gruppen nebeneinander, außer gestapelt war gewünscht
        stack = True if spec.kind == "stacked_bar" else (False if spec.color else None)
        st.bar_chart(data, **args, sort=False, stack=stack)
    elif spec.kind == "line":
        st.line_chart(data, **args)
    elif spec.kind == "area":
        st.area_chart(data, **args)
    else:
        st.scatter_chart(data, **args)


def show_result(result: AnalystResult, summary: str | None) -> None:
    if result.error:
        st.error(
            f"No result, the corrected query failed too. Last error:\n\n{result.error}",
            icon=":material/error:",
        )
    else:
        if summary:
            st.markdown(summary)
        show_chart(result)
        rows = len(result.table)
        st.dataframe(result.table, hide_index=True)
        st.caption(
            f"{rows} {'row' if rows == 1 else 'rows'}"
            + (f" (limited to {MAX_ROWS})" if rows == MAX_ROWS else "")
        )
    # Das SQL ist immer sichtbar: Nur so lässt sich prüfen, ob die Zahlen stimmen
    st.markdown("##### SQL")
    st.code(result.sql, language="sql", wrap_lines=True)
    if len(result.attempts) > 1:
        with st.expander(
            f"Attempts ({len(result.attempts)})", icon=":material/history:"
        ):
            for number, attempt in enumerate(result.attempts, start=1):
                verdict = "ran" if attempt.error is None else "failed"
                st.markdown(f"**Attempt {number}: {verdict}**")
                st.code(attempt.sql, language="sql", wrap_lines=True)
                if attempt.error:
                    st.code(attempt.error, language=None, wrap_lines=True)
                st.caption(format_usage(attempt.usage))


with main:
    with st.expander("Dataset: a fictional online shop", icon=":material/database:"):
        st.caption(
            "Generated by code with a fixed seed: 400 customers, 40 products, orders from "
            "2023 to 2025. No real people or companies. Queries run read-only."
        )
        show_dataset()

    st.caption(
        "Ask a question in plain English or pick an example, and name a chart if you want "
        "one (bar, stacked bar, line, area or scatter). If the query fails, the model gets "
        "the error back and may correct it once. Only SELECT queries run, on a read-only "
        "connection."
    )
    # Fragen wie „last 12 months“ zählen ab heute; der Datensatz endet aber 2025
    st.caption(
        ':material/calendar_month: Data from January 2023 to December 2025. "Last 12 '
        'months" counts back from today, so it only includes what is left of 2025, or nothing.'
    )
    # Beispiele in einer aufklappbaren Liste: braucht eine Zeile, egal wie viele es sind
    with st.popover("Examples", icon=":material/lightbulb:"):
        for number, example in enumerate(EXAMPLES):
            st.button(
                example,
                key=f"analyst_example_{number}",
                type="tertiary",
                on_click=use_example,
                args=(example,),
            )
    with st.form("analyst_form", border=False):
        # Feld und Button nebeneinander; analyst.css macht den Button so hoch wie das Feld
        row = st.container(horizontal=True, key="analyst_ask_row")
        question = row.text_area(
            "Question",
            key="analyst_question",
            height="content",  # wächst mit langen Fragen
            placeholder="e.g. Revenue per year and category as a stacked bar chart",
            label_visibility="collapsed",
        )
        asked = row.form_submit_button(
            "Ask",
            type="primary",
            icon=":material/query_stats:",
        )

    if asked and question.strip():
        try:
            with st.spinner("Writing and running SQL..."):
                result = ask(question.strip(), model)
                summary, summary_usage = None, Usage()
                if result.table is not None:
                    summary, summary_usage = summarize(
                        question.strip(), result.table, model
                    )
            # Im Session State, damit Klicks auf der Seite den Lauf nicht wiederholen
            st.session_state.analyst_run = (result, summary, summary_usage)
            st.session_state.analyst_cost = (
                st.session_state.get("analyst_cost", 0.0)
                + result.usage.cost_usd
                + summary_usage.cost_usd
            )
            st.session_state.analyst_questions = (
                st.session_state.get("analyst_questions", 0) + 1
            )
        except LLMError as err:
            st.error(str(err), icon=":material/error:")

    run = st.session_state.get("analyst_run")
    if run is not None:
        show_result(run[0], run[1])

with stats:
    st.markdown("##### Last question")
    usage = (run[0].usage + run[2]) if run else None
    st.metric(
        "Attempts",
        len(run[0].attempts) if run else "–",
        help="1 means the first query ran. A correction costs another request.",
    )
    st.metric("Cost (USD)", format_cost(usage.cost_usd, unit=False) if usage else "–")
    st.metric(
        "Tokens in",
        usage.prompt_tokens if usage else "–",
        help="Schema, question and the result rows for the summary sentence.",
    )
    st.metric("Tokens out", usage.completion_tokens if usage else "–")
    st.metric("Tokens total", usage.total_tokens if usage else "–")
    st.markdown("##### Session")
    st.metric("Questions", st.session_state.get("analyst_questions", 0))
    st.metric(
        "Cost (USD)",
        format_cost(st.session_state.get("analyst_cost", 0.0), unit=False),
    )
