"""Einheitlicher Kopf jeder Modulseite: ein Satz, die Techniken und „How it works“.

Texte aus core/catalog.py, damit Home und Modulseite dasselbe sagen. „How it works“ ist ein
Popover statt eines Expanders: Aufgeklappt schiebt es nichts nach unten, das ist wichtig auf
den Chat-Seiten, deren Spalten genau bis zum Eingabefeld reichen (chat.css).
"""

import streamlit as st

from core.catalog import CATALOG


def show_intro(module_id: str) -> None:
    entry = next(e for e in CATALOG if e.id == module_id)
    # wrap=False: „How it works“ bleibt rechts in derselben Zeile, der Kopf bleibt niedrig
    row = st.container(
        horizontal=True, vertical_alignment="center", gap="medium", wrap=False
    )
    row.markdown(
        f"{entry.summary} "
        + " ".join(f":gray-badge[{technique}]" for technique in entry.techniques),
        width="stretch",
    )
    with row.popover("How it works", icon=":material/info:"):
        st.markdown(
            "\n".join(f"{number}. {step}" for number, step in enumerate(entry.how, 1))
        )
