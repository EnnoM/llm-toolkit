"""Konfiguration: Secrets aus st.secrets (Streamlit Cloud), lokal Fallback auf .env."""

import os

import streamlit as st
from dotenv import load_dotenv

# Lokal: Werte aus .env in die Umgebungsvariablen laden (bereits gesetzte bleiben).
load_dotenv()


def get_secret(name: str, default: str | None = None) -> str | None:
    """Liest ein Secret: zuerst st.secrets, dann Umgebungsvariable bzw. .env."""
    # load_if_toml_exists() wirft keinen Fehler, wenn es keine secrets.toml gibt.
    if st.secrets.load_if_toml_exists() and name in st.secrets:
        return str(st.secrets[name])
    # Leere Werte (z. B. "KEY=" in .env) zählen als nicht gesetzt.
    return os.getenv(name) or default
