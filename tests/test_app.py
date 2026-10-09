"""app.py: Passwortabfrage, Entwicklungs-Schalter und welche Seiten es gibt."""

import pytest
from streamlit.testing.v1 import AppTest

from core.catalog import built_modules


def run_app(monkeypatch, skip_login: str, show_internal: str = "false") -> AppTest:
    # Alle Schalter setzen: sonst greifen die Werte aus der lokalen .env
    monkeypatch.setenv("SKIP_LOGIN", skip_login)
    monkeypatch.setenv("SHOW_INTERNAL_PAGES", show_internal)
    monkeypatch.setenv("APP_PASSWORD", "test-password")
    return AppTest.from_file("../app.py", default_timeout=10).run()


def sign_in(at: AppTest, password: str) -> AppTest:
    at.text_input[0].set_value(password)
    return next(b for b in at.button if b.label == "Sign in").click().run()


def shows_home(at: AppTest) -> bool:
    return any(m.value.startswith("#### ") for m in at.markdown)


def test_login_is_required_by_default(monkeypatch):
    at = run_app(monkeypatch, "false")
    assert not at.exception
    assert at.text_input[0].label == "Password"
    assert not shows_home(at)  # Home ist ohne Anmeldung nicht erreichbar


def test_wrong_password_is_rejected(monkeypatch):
    at = sign_in(run_app(monkeypatch, "false"), "wrong")
    assert any(e.value == "Wrong password." for e in at.error)
    assert "authenticated" not in at.session_state


def test_correct_password_opens_home(monkeypatch):
    at = sign_in(run_app(monkeypatch, "false"), "test-password")
    assert not at.exception
    assert shows_home(at)


def test_skip_login_opens_the_app_without_password(monkeypatch):
    at = run_app(monkeypatch, "true")
    assert not at.exception
    assert not at.text_input  # keine Passwortabfrage
    assert shows_home(at)


def test_home_lists_every_module_and_marks_upcoming_as_wip(monkeypatch):
    at = run_app(monkeypatch, "true")
    headings = [m.value for m in at.markdown if m.value.startswith("#### ")]
    assert len(headings) == len(built_modules())
    assert "Module 1: Chat" in headings[0]
    wip = [c.value for c in at.caption if ":gray-badge[WIP]" in c.value]
    assert "Module 6: AI Data Analyst" in wip[0]


def test_start_is_hidden_without_the_local_switch(monkeypatch):
    at = run_app(monkeypatch, "true")
    with pytest.raises(ValueError, match="Could not find a navigation page"):
        at.switch_page("app_pages/start.py")


def test_start_shows_the_checklist_with_the_local_switch(monkeypatch):
    at = run_app(monkeypatch, "true", show_internal="true")
    at.switch_page("app_pages/start.py").run()
    assert not at.exception
    assert at.dataframe  # Arbeits-Checkliste mit der Modultabelle
