"""Data Extraction: Freitext wird zu geprüftem JSON, mit Korrekturversuchen bei Prüffehlern.

Layout nach docs/design.md: Einstellungen | Eingabe und Ergebnis | Kennzahlen im Verhältnis 1:2:1.
"""

import streamlit as st

from core.config import get_secret
from core.extraction import (
    DOC_TYPES,
    MAX_CORRECTIONS,
    ExtractionResult,
    extract,
    pretty_json,
    schema_fields,
)
from core.llm import LLMError
from core.models import (
    format_cost,
    format_usage,
    model_name,
    model_options,
    model_price,
)

st.set_page_config(layout="wide")  # drei Spalten brauchen die volle Breite

st.title("Data Extraction")
settings, main, stats = st.columns([1, 2, 1], gap="large")

with settings:
    st.markdown("##### Settings")
    doc_type_id = st.radio(
        "Document type",
        list(DOC_TYPES),
        format_func=lambda key: DOC_TYPES[key].title,
        captions=[doc_type.description for doc_type in DOC_TYPES.values()],
        help=(
            "Each type has its own schema: the fields, their types and which may stay empty. "
            "You can replace the example text next to this column."
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
            "Prices are in USD per million input / output tokens. Only providers that "
            "support structured outputs are used, so not every model may be available."
        ),
    )

doc_type = DOC_TYPES[doc_type_id]


def value_text(value) -> str:
    """Wert für die Tabelle Feld/Wert: null sichtbar, Listen kurz."""
    if value is None:
        return "null"
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            return f"{len(value)} entries (see JSON)"
        return ", ".join(str(item) for item in value) or "(empty list)"
    return str(value)


def show_result(result: ExtractionResult) -> None:
    """Geprüftes Ergebnis als Tabelle und als JSON, darunter die Versuche.

    Untereinander statt nebeneinander: In der halben mittleren Spalte wurden Werte abgeschnitten.
    """
    if result.error:
        st.error(result.error, icon=":material/error:")
    else:
        # Widersprüche im Dokument: Werte bleiben wie im Text, der Mensch entscheidet
        for warning in result.warnings:
            st.warning(warning, icon=":material/warning:")
        st.markdown("##### Fields")
        st.table(
            {
                "Field": list(result.data),
                "Value": [value_text(value) for value in result.data.values()],
            },
            hide_index=True,
        )
        st.markdown("##### JSON")
        st.json(result.data, expanded=2)
    # Jeder Versuch mit roher Antwort und Prüfergebnis, damit Korrekturen nachvollziehbar sind
    with st.expander(f"Attempts ({len(result.attempts)})", icon=":material/history:"):
        for number, attempt in enumerate(result.attempts, start=1):
            verdict = "passed" if attempt.error is None else "failed validation"
            st.markdown(f"**Attempt {number}: {verdict}**")
            if attempt.error:
                st.code(attempt.error, language=None, wrap_lines=True)
            st.code(pretty_json(attempt.raw), language="json", wrap_lines=True)
            st.caption(format_usage(attempt.usage))


with main:
    with st.expander(f"Schema: {doc_type.title}", icon=":material/data_object:"):
        st.table(schema_fields(doc_type.schema_model), hide_index=True)
    text = st.text_area(
        "Input",
        doc_type.sample_text,
        key=f"extraction_input_{doc_type_id}",  # eigener Text pro Typ, bleibt erhalten
        height=200,
        help="The text to extract from. Prefilled with an example you can replace.",
    )
    extract_clicked = st.button(
        "Extract",
        type="primary",
        icon=":material/output:",
        disabled=not text.strip(),
    )
    st.caption(
        f"If the answer fails validation, the model gets the errors back and may correct "
        f"itself up to {MAX_CORRECTIONS} times."
    )

    if extract_clicked:
        try:
            with st.spinner("Extracting..."):
                # Im Session State, damit Klicks auf der Seite den Lauf nicht wiederholen
                st.session_state.extraction_run = extract(doc_type_id, text, model)
        except LLMError as err:
            st.error(str(err), icon=":material/error:")

    run: ExtractionResult | None = st.session_state.get("extraction_run")
    if run is None:
        st.caption("Pick a document type, then press Extract.")
    else:
        if run.doc_type_id != doc_type_id or run.model != model:
            st.caption(
                f"Showing the last run: {DOC_TYPES[run.doc_type_id].title} "
                f"with {model_name(run.model)}."
            )
        show_result(run)

with stats:
    st.markdown("##### Last run")
    usage = run.usage if run else None
    st.metric(
        "Attempts",
        len(run.attempts) if run else "–",
        help="1 means the first answer passed validation. Each correction costs another request.",
    )
    st.metric("Cost (USD)", format_cost(usage.cost_usd, unit=False) if usage else "–")
    st.metric(
        "Tokens in",
        usage.prompt_tokens if usage else "–",
        help="Instructions and text. Each correction sends the whole exchange again, so it adds up.",
    )
    st.metric("Tokens out", usage.completion_tokens if usage else "–")
    st.metric("Tokens total", usage.total_tokens if usage else "–")
