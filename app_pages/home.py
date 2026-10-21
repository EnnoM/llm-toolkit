"""Home (öffentlich): Ziel des Projekts und alle Module als gleich große Kacheln.

Die Seite ist im Lebenslauf verlinkt, deshalb steht oben in zwei Sätzen, was das Projekt
zeigen soll. Darunter alle Module im Raster, die geplanten ausgegraut.
"""

import streamlit as st

from core.catalog import CATALOG

st.set_page_config(layout="wide")  # Raster über die ganze Breite

COLUMNS = 3  # breit genug für die Kurzbeschreibung; auf dem Handy untereinander
TILE_HEIGHT = 190  # feste Höhe, damit alle Kacheln gleich groß sind

st.title("LLM Toolkit")
st.markdown(
    "Hands-on demos of LLM techniques, from a plain chat to autonomous agents. "
    "I built it to learn how LLM applications work under the hood and to show what I can "
    "build with AI: every module calls real models through OpenRouter and shows the tokens "
    "and cost of each request."
)

for start in range(0, len(CATALOG), COLUMNS):
    row = st.columns(COLUMNS)
    for column, entry in zip(row, CATALOG[start : start + COLUMNS]):
        with column.container(border=True, height=TILE_HEIGHT):
            if entry.page:
                # Der Titel ist der Link: kein extra Button, der bei langem Text verrutscht
                st.page_link(
                    entry.page,
                    label=f"**{entry.title}**",
                    icon=entry.icon,
                )
                # Gleiche Schriftgröße wie bei den geplanten, nur nicht grau
                st.markdown(f":small[{entry.summary}]")
            else:
                # Ausgegraut über st.caption: geplante Module haben noch keine Seite
                st.caption(f"**{entry.icon} {entry.title}** :gray-badge[WIP]")
                st.caption(entry.summary)
