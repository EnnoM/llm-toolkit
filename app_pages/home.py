"""Home (öffentlich): Überblick über die fertigen und die geplanten Module."""

import streamlit as st

from core.catalog import CATALOG, built_modules, upcoming_modules

st.set_page_config(layout="centered")  # reiner Lesetext, keine Einstellungen

# Nummer = Position im Katalog, damit die geplanten Module weiterzählen
NUMBERS = {entry.id: number for number, entry in enumerate(CATALOG, start=1)}

st.title("LLM Toolkit")
st.markdown(
    "Hands-on demos of LLM techniques, from a plain chat to autonomous agents. "
    "Every module calls real models through OpenRouter and shows the tokens and "
    "cost of each request."
)

st.subheader("Modules")
for entry in built_modules():
    with st.container(border=True):
        st.markdown(f"#### {entry.icon} Module {NUMBERS[entry.id]}: {entry.title}")
        st.markdown(entry.summary)
        st.caption(" · ".join(entry.techniques))
        st.page_link(
            entry.page, label=f"Open {entry.title}", icon=":material/arrow_forward:"
        )

st.subheader("Coming next")
# Ausgegraut über st.caption: geplante Module haben noch keine Seite und keinen Link
for entry in upcoming_modules():
    with st.container(border=True):
        st.caption(
            f"**{entry.icon} Module {NUMBERS[entry.id]}: {entry.title}** "
            ":gray-badge[WIP]"
        )
        st.caption(entry.summary)
