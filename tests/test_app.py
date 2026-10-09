"""Login in app.py: Passwortabfrage und der Entwicklungs-Schalter SKIP_LOGIN."""

from streamlit.testing.v1 import AppTest


def run_app(monkeypatch, skip_login: str) -> AppTest:
    monkeypatch.setenv("SKIP_LOGIN", skip_login)
    monkeypatch.setenv("APP_PASSWORD", "test-password")
    return AppTest.from_file("../app.py", default_timeout=10).run()


def sign_in(at: AppTest, password: str) -> AppTest:
    at.text_input[0].set_value(password)
    return next(b for b in at.button if b.label == "Sign in").click().run()


def test_login_is_required_by_default(monkeypatch):
    at = run_app(monkeypatch, "false")
    assert not at.exception
    assert at.text_input[0].label == "Password"
    assert not at.dataframe  # Startseite ist ohne Anmeldung nicht erreichbar


def test_wrong_password_is_rejected(monkeypatch):
    at = sign_in(run_app(monkeypatch, "false"), "wrong")
    assert any(e.value == "Wrong password." for e in at.error)
    assert "authenticated" not in at.session_state


def test_correct_password_opens_the_app(monkeypatch):
    at = sign_in(run_app(monkeypatch, "false"), "test-password")
    assert not at.exception
    assert at.dataframe  # Startseite mit der Modultabelle


def test_skip_login_opens_the_app_without_password(monkeypatch):
    at = run_app(monkeypatch, "true")
    assert not at.exception
    assert not at.text_input  # keine Passwortabfrage
    assert at.dataframe
