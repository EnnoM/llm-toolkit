"""Datenextraktion (core/extraction.py, app_pages/extraction.py). Keine echten API-Aufrufe."""

import json
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion
from streamlit.testing.v1 import AppTest

import core.llm
from core.extraction import (
    DOC_TYPES,
    MAX_CORRECTIONS,
    extract,
    strict_json_schema,
    validate,
)

INVOICE = DOC_TYPES["invoice"]

# Gültige Antwort zur Beispielrechnung, so wie Modelle sie liefern (ohne Steuernummer)
VALID_INVOICE = {
    "vendor": "Brightline Web Studio",
    "invoice_number": "BWS-2026-0412",
    "invoice_date": "2026-09-03",
    "due_date": "2026-09-17",
    "currency": "EUR",
    "line_items": [
        {
            "description": "Website redesign",
            "quantity": None,
            "unit_price": None,
            "amount": 2400.0,
        },
        {"description": "Hosting", "quantity": 12, "unit_price": 15.0, "amount": 180.0},
        {
            "description": "Extra support",
            "quantity": 4,
            "unit_price": 85.0,
            "amount": 340.0,
        },
    ],
    "net_amount": 2920.0,
    "tax_amount": 554.8,
    "total_amount": 3474.8,
    "tax_id": None,
}

# Gespeicherte fehlerhafte Antwort: Betrag als Text, deutsches Datum, erfundenes Zusatzfeld
BROKEN_INVOICE = json.dumps(
    {
        **VALID_INVOICE,
        "invoice_date": "03.09.2026",
        "total_amount": "3.474,80 EUR",
        "iban": "DE00 0000",
    }
)


# ------------------------------------------------------------------ Schema und Prüfung (ohne API)


def all_objects(node):
    """Alle Objekt-Definitionen im Schema, auch verschachtelte (z. B. line_items)."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield node
        for value in node.values():
            yield from all_objects(value)
    elif isinstance(node, list):
        for item in node:
            yield from all_objects(item)


@pytest.mark.parametrize("doc_type_id", list(DOC_TYPES))
def test_strict_schema_requires_every_field_and_forbids_extras(doc_type_id):
    schema = strict_json_schema(DOC_TYPES[doc_type_id].schema_model)
    objects = list(all_objects(schema))
    assert objects
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert obj["required"] == list(obj["properties"])


def test_valid_invoice_passes_and_missing_tax_id_stays_null():
    data, error = validate(INVOICE, json.dumps(VALID_INVOICE))
    assert error is None
    assert data.tax_id is None
    dumped = data.model_dump(mode="json")
    assert dumped["due_date"] == "2026-09-17"  # Datum im Format JJJJ-MM-TT
    assert isinstance(dumped["total_amount"], float)  # Betrag als Zahl


def test_saved_broken_json_fails_with_one_line_per_problem():
    data, error = validate(INVOICE, BROKEN_INVOICE)
    assert data is None
    lines = error.splitlines()
    assert any(line.startswith("- invoice_date:") for line in lines)
    assert any(line.startswith("- total_amount:") for line in lines)
    assert any(line.startswith("- iban:") for line in lines)


def test_totals_that_do_not_add_up_pass_with_a_warning():
    # Fehler im Dokument selbst: Wert bleibt wie im Text, nur ein Hinweis für den Menschen
    wrong = {**VALID_INVOICE, "total_amount": 3500.0}
    data, error = validate(INVOICE, json.dumps(wrong))
    assert error is None
    assert data.total_amount == 3500.0
    assert data.warnings() == [
        (
            "Totals don't add up: 2920.00 + 554.80 = 3474.80, but the invoice says "
            "3500.00. Check the document."
        )
    ]


def test_matching_totals_give_no_warning():
    data, _ = validate(INVOICE, json.dumps(VALID_INVOICE))
    assert data.warnings() == []


def test_reversed_salary_range_is_a_warning():
    posting = DOC_TYPES["job_posting"]
    answer = {
        "job_title": "Data Engineer",
        "company": "Greenfield Energy",
        "location": None,
        "remote": None,
        "employment_type": None,
        "salary_min": 78000,
        "salary_max": 65000,
        "currency": "EUR",
        "skills": [],
        "application_deadline": None,
    }
    data, error = validate(posting, json.dumps(answer))
    assert error is None
    assert data.warnings()[0].startswith("Salary range is reversed")


def test_meeting_time_must_be_hh_mm():
    meeting = DOC_TYPES["meeting"]
    base = {
        "subject": "Review",
        "organizer": "Markus",
        "participants": ["Lena"],
        "meeting_date": "2026-10-08",
        "duration_minutes": 90,
        "location": None,
        "online_link": None,
    }
    assert validate(meeting, json.dumps({**base, "start_time": "14:30"}))[1] is None
    # Erfundene Zeitzone wie bei gpt-4.1-nano mit dem Format "time"
    assert validate(meeting, json.dumps({**base, "start_time": "14:30:00-04:00"}))[1]


def test_text_that_is_not_json_fails_cleanly():
    data, error = validate(INVOICE, "Sure! Here is the data: ...")
    assert data is None
    assert "Invalid JSON" in error


# ------------------------------------------------------------------ Ablauf (nachgebauter Client)


def fake_client(answers: list[str], calls: list):
    """Antwortet der Reihe nach mit answers (die letzte wiederholt sich); merkt sich Aufrufe."""

    def create(**kwargs):
        calls.append(kwargs)
        content = answers[min(len(calls), len(answers)) - 1]
        return ChatCompletion.model_validate(
            {
                "id": "gen-1",
                "object": "chat.completion",
                "created": 0,
                "model": kwargs["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "cost": 0.0001,
                },
            }
        )

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def test_request_enforces_the_schema_and_only_uses_supporting_providers():
    calls = []
    extract(
        "invoice",
        "text",
        "test/model",
        client=fake_client([json.dumps(VALID_INVOICE)], calls),
    )
    request = calls[0]
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["extra_body"] == {"provider": {"require_parameters": True}}
    assert request["temperature"] == 0.0


def test_first_valid_answer_needs_one_attempt():
    calls = []
    result = extract(
        "invoice",
        "text",
        "test/model",
        client=fake_client([json.dumps(VALID_INVOICE)], calls),
    )
    assert result.error is None
    assert result.data["tax_id"] is None
    assert len(result.attempts) == 1


def test_broken_answer_is_corrected_with_the_validation_errors():
    calls = []
    client = fake_client([BROKEN_INVOICE, json.dumps(VALID_INVOICE)], calls)
    result = extract("invoice", "text", "test/model", client=client)
    assert result.error is None
    assert len(result.attempts) == 2
    assert result.attempts[0].error and result.attempts[1].error is None
    # Zweite Anfrage enthält die fehlerhafte Antwort und die Fehlermeldung dazu
    second = calls[1]["messages"]
    assert second[-2] == {"role": "assistant", "content": BROKEN_INVOICE}
    assert "total_amount" in second[-1]["content"]
    assert result.usage.cost_usd == pytest.approx(0.0002)


def test_gives_up_after_two_corrections_with_a_clear_message():
    calls = []
    result = extract(
        "invoice", "text", "test/model", client=fake_client([BROKEN_INVOICE], calls)
    )
    assert len(calls) == 1 + MAX_CORRECTIONS
    assert result.data is None
    assert result.error.startswith("No valid result after 2 corrections.")
    assert "total_amount" in result.error


def test_contradicting_document_is_not_sent_back_for_correction():
    # Sonst „korrigiert“ das Modell den Betrag passend (so geschehen am 23.09.2026)
    calls = []
    wrong = json.dumps({**VALID_INVOICE, "total_amount": 3500.0})
    result = extract(
        "invoice", "text", "test/model", client=fake_client([wrong], calls)
    )
    assert len(calls) == 1
    assert result.data["total_amount"] == 3500.0
    assert result.warnings and result.warnings[0].startswith("Totals don't add up")


# ------------------------------------------------------------------ Seite


def open_page() -> AppTest:
    at = AppTest.from_file("../app.py", default_timeout=10)
    at.session_state["authenticated"] = True
    at.run()
    return at.switch_page("app_pages/extraction.py").run()


def click_extract(monkeypatch, answers: list[str]) -> AppTest:
    calls = []
    monkeypatch.setattr(
        "core.extraction.get_client", lambda: fake_client(answers, calls)
    )
    at = open_page()
    assert not at.exception
    return next(b for b in at.button if b.label == "Extract").click().run()


def test_extract_shows_json_table_and_attempts(monkeypatch):
    at = click_extract(monkeypatch, [json.dumps(VALID_INVOICE)])
    assert not at.exception
    assert at.json  # Ergebnis als JSON
    assert len(at.table) == 2  # Schema-Übersicht und Tabelle Feld/Wert
    assert next(m.value for m in at.metric if m.label == "Attempts") == "1"
    assert next(m.value for m in at.metric if m.label == "Cost (USD)") == "0.000100"


def test_contradicting_document_shows_a_warning(monkeypatch):
    wrong = json.dumps({**VALID_INVOICE, "total_amount": 3500.0})
    at = click_extract(monkeypatch, [wrong])
    assert not at.exception
    assert any(w.value.startswith("Totals don't add up") for w in at.warning)
    assert next(m.value for m in at.metric if m.label == "Attempts") == "1"


def test_failed_validation_shows_an_error_instead_of_a_traceback(monkeypatch):
    at = click_extract(monkeypatch, [BROKEN_INVOICE])
    assert not at.exception
    assert any(e.value.startswith("No valid result") for e in at.error)
    assert next(m.value for m in at.metric if m.label == "Attempts") == "3"


def test_missing_api_key_shows_an_error(monkeypatch):
    def no_key():
        raise core.llm.LLMError("OPENROUTER_API_KEY is missing.")

    monkeypatch.setattr("core.extraction.get_client", no_key)
    at = open_page()
    next(b for b in at.button if b.label == "Extract").click().run()
    assert not at.exception
    assert any("OPENROUTER_API_KEY" in e.value for e in at.error)


def test_sidebar_contains_no_module_settings():
    at = open_page()
    assert not (at.sidebar.radio or at.sidebar.text_area or at.sidebar.button)
