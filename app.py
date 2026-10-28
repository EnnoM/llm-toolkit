"""Einstiegspunkt: Passwortabfrage, danach Navigation zwischen den Seiten."""

import hmac

import streamlit as st

from core.catalog import LEVELS, built_modules
from core.config import get_secret

st.set_page_config(page_title="LLM Toolkit", page_icon=":material/handyman:")


def login() -> None:
    """Login-Seite (englisch wie die Module). Ohne Anmeldung ist sie die einzige Seite."""
    st.title("LLM Toolkit")
    expected = get_secret("APP_PASSWORD")
    if not expected:
        st.error("APP_PASSWORD is not set (st.secrets or .env).")
        return

    with st.form("login"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        # compare_digest vergleicht zeitkonstant und verhindert so Timing-Angriffe.
        if hmac.compare_digest(password.encode(), expected.encode()):
            st.session_state.authenticated = True
            st.rerun()
        st.error("Wrong password.")


# Nur für die lokale Entwicklung: SKIP_LOGIN=true in .env schaltet die Passwortabfrage ab.
# Auf Streamlit Cloud gibt es keine .env, dort bleibt das Login aktiv.
skip_login = (get_secret("SKIP_LOGIN") or "").lower() == "true"
# Ebenfalls nur lokal: SHOW_INTERNAL_PAGES=true zeigt die Arbeits-Checkliste „Start“.
# Auf Streamlit Cloud fehlt der Schalter, dort gibt es die Seite gar nicht.
show_internal = (get_secret("SHOW_INTERNAL_PAGES") or "").lower() == "true"

if skip_login or st.session_state.get("authenticated"):
    # Home steht allein oben, ohne Überschrift; darunter die Module nach Stufen, die neueste
    # Stufe und das neueste Modul zuerst
    start = [
        st.Page(
            "app_pages/home.py", title="Home", icon=":material/home:", default=True
        ),
    ]
    if show_internal:
        start.append(
            st.Page("app_pages/start.py", title="Start", icon=":material/checklist:")
        )
    pages = {"": start}
    for level in sorted(LEVELS, reverse=True):
        entries = [e for e in built_modules() if e.level == level][::-1]
        if entries:
            # Nur „Level 3“: Längere Überschriften schneidet die Seitenleiste ab
            pages[f"Level {level}"] = [
                st.Page(e.page, title=e.title, icon=e.icon) for e in entries
            ]
else:
    # Ohne Anmeldung kennt die Navigation nur den Login; andere URLs sind nicht erreichbar.
    pages = [st.Page(login, title="Sign in", icon=":material/lock:")]

st.navigation(pages).run()
