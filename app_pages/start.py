"""Start (intern, nur lokal): Arbeits-Checkliste aus docs/modules.yaml."""

import streamlit as st

from core.modules import (
    STATUS_LABELS,
    ModulesError,
    count_done,
    filter_modules,
    load_modules,
    status_problem,
)

STATUS_ICONS = {
    "geplant": ":material/schedule:",
    "in_arbeit": ":material/pending:",
    "fertig": ":material/check_circle:",
}


st.set_page_config(layout="wide")  # Tabelle mit fünf Spalten braucht die Breite


@st.cache_data
def cached_modules():
    """Liest die YAML nur einmal; „Neu laden“ leert diesen Cache."""
    return load_modules()


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def reload_button() -> None:
    # on_click läuft vor dem nächsten Durchlauf, der Cache ist also schon leer,
    # egal an welcher Stelle der Seite der Button steht
    st.button(
        "Neu laden",
        icon=":material/refresh:",
        help="docs/modules.yaml neu einlesen",
        on_click=cached_modules.clear,
    )


try:
    modules = cached_modules()
except ModulesError as err:
    st.error(
        "**docs/modules.yaml ist fehlerhaft.** Nach dem Korrigieren „Neu laden“ klicken.\n\n"
        + bullets(str(err).splitlines()),
        icon=":material/error:",
    )
    reload_button()
    st.stop()

# Kopf
st.title("LLM Toolkit")
st.write(
    "Modulare Streamlit-App zum Lernen und Vorzeigen von LLM-Techniken, "
    "alle Modelle laufen über OpenRouter. Diese Seite ist die Arbeits-Checkliste: "
    "Status, Inhalt und Arbeitsstand jedes Moduls, gepflegt in `docs/modules.yaml`."
)
active = [m for m in modules if m.status == "in_arbeit"]
for m in active:
    st.info(
        f"**In Arbeit: {m.name}.** Nächster Schritt: {m.naechster_schritt}",
        icon=":material/pending:",
    )
if not active:
    upcoming = next((m for m in modules if m.status == "geplant"), None)
    st.info(
        f"Kein Modul in Arbeit. Als Nächstes geplant: {upcoming.name}."
        if upcoming
        else "Alle Module sind fertig."
    )
for m in modules:
    if problem := status_problem(m):
        st.warning(f"**{m.name}:** {problem}", icon=":material/warning:")

# Fortschritt
done = count_done(modules)
st.progress(done / len(modules), text=f"{done} von {len(modules)} Modulen fertig")

# Übersicht mit Filterzeile (die Sidebar ist nur für die Navigation da)
st.subheader("Übersicht")
with st.container(horizontal=True, vertical_alignment="bottom", gap="large"):
    all_phases = sorted({m.phase for m in modules})
    phases = st.pills("Phase", all_phases, selection_mode="multi", default=all_phases)
    statuses = st.pills(
        "Status",
        list(STATUS_LABELS),
        selection_mode="multi",
        default=list(STATUS_LABELS),
        format_func=STATUS_LABELS.get,
    )
    reload_button()

shown = filter_modules(modules, phases, statuses)
if not shown:
    st.info("Kein Modul passt zum Filter.")
    st.stop()
st.dataframe(
    [
        {
            "Modul": m.name,
            "Phase": m.phase,
            "Status": STATUS_LABELS[m.status],
            "Kurz": m.kurz,
            "Nächster Schritt": m.naechster_schritt,
        }
        for m in shown
    ],
    hide_index=True,
    height="content",
    column_config={
        "Modul": st.column_config.TextColumn(width=200),
        "Phase": st.column_config.NumberColumn(width=60),
        "Status": st.column_config.TextColumn(width=100),
        "Kurz": st.column_config.TextColumn(width="large"),
        "Nächster Schritt": st.column_config.TextColumn(width="large"),
    },
)

# Details
st.subheader("Module im Detail")
for m in shown:
    with st.expander(
        f"Phase {m.phase} – {m.name} ({STATUS_LABELS[m.status]})",
        expanded=m.status == "in_arbeit",
        icon=STATUS_ICONS[m.status],
    ):
        st.markdown(f"##### Beschreibung\n\n{m.beschreibung}")
        st.markdown(f"##### Oberfläche\n\n{m.oberflaeche}")
        st.markdown(f"##### Lernziele\n\n{bullets(m.lernziele)}")
        st.markdown(f"##### Akzeptanzkriterien\n\n{bullets(m.akzeptanz)}")
        with st.container(border=True):
            st.markdown(f"##### Arbeitsstand\n\n{m.arbeitsstand}")
            st.markdown(f"##### Nächster Schritt\n\n{m.naechster_schritt}")
            st.markdown(f"##### Notizen\n\n{m.notizen or '–'}")
