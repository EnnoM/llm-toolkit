"""Home (öffentlich): Ziel des Projekts und alle Module als gleich große Kacheln.

Die Seite ist im Lebenslauf verlinkt, deshalb steht oben in zwei Sätzen, was das Projekt
zeigen soll. Darunter die Module nach Stufen wie in der Seitenleiste, die neueste Stufe zuerst,
die geplanten Module ausgegraut.
"""

import streamlit as st

from core.catalog import LEVELS, level_title, levels_newest_first

st.set_page_config(layout="wide")  # Raster über die ganze Breite

COLUMNS = 3  # breit genug für die Kurzbeschreibung; auf dem Handy untereinander
TILE_HEIGHT = (
    280  # feste Höhe, damit alle Kacheln gleich groß sind (mit Technik-Badges)
)
REPOSITORY = "https://github.com/EnnoM/llm-toolkit"

st.title("LLM Toolkit")
st.markdown(
    "Hands-on demos of LLM techniques, from a plain chat to autonomous agents. "
    "I built it to learn how LLM applications work under the hood and to show what I can "
    "build with AI: every module calls real models through OpenRouter and shows the tokens "
    "and cost of each request."
)
st.caption(f"Built by Enno · [Source code on GitHub]({REPOSITORY})")

for level, entries in levels_newest_first():
    st.markdown(f"#### {level_title(level)}")
    st.caption(LEVELS[level][1])
    for start in range(0, len(entries), COLUMNS):
        row = st.columns(COLUMNS)
        for column, entry in zip(row, entries[start : start + COLUMNS]):
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
                    # Die Techniken als Stichworte: zeigen auf einen Blick, was gebaut wurde
                    st.markdown(" ".join(f":gray-badge[{t}]" for t in entry.techniques))
                else:
                    # Ausgegraut über st.caption: geplante Module haben noch keine Seite
                    st.caption(f"**{entry.icon} {entry.title}** :gray-badge[WIP]")
                    st.caption(entry.summary)
