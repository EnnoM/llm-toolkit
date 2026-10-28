"""Tests für den Workflow mit Freigabe: Zustände, Audit-Log, Freigaberegeln. Keine API-Aufrufe."""

import json

import pytest
from streamlit.testing.v1 import AppTest

from core import workflow
from core.workflow import (
    CATEGORIES,
    WorkflowError,
    WorkflowStore,
    approve,
    bucket,
    decision_stats,
    discard,
    load_mails,
    mail_by_id,
    placeholders,
    process,
    reset_inbox,
    shop_facts,
    unverified,
    write_draft,
)
from tests.test_extraction import fake_client

DRAFT = "Hello Julia,\n\nsorry about the jug. We will [replacement or refund].\n\nLarkfield Customer Service"


def draft(reply: str, assessment="clear", facts=("Policy damaged goods",)) -> str:
    """Antwort des Modells im Entwurfsschritt (Structured Outputs)."""
    return json.dumps(
        {
            "facts_used": list(facts),
            "assessment": assessment,
            "assessment_reason": "The policy covers it.",
            "reply": reply,
        }
    )


def classification(category="complaint", priority="high") -> str:
    return json.dumps(
        {"reason": "Damaged goods.", "category": category, "priority": priority}
    )


@pytest.fixture
def store(tmp_path) -> WorkflowStore:
    return WorkflowStore(tmp_path / "workflow.sqlite")


def drafted(store, mail_id="m01", calls=None) -> None:
    """Bringt eine Mail mit dem Fake-Client bis zum Entwurf."""
    process(
        mail_by_id(mail_id),
        store,
        "test/model",
        client=fake_client(
            [classification(), draft(DRAFT)], [] if calls is None else calls
        ),
    )


# ------------------------------------------------------------------ Beispielmails


def test_every_mail_has_a_valid_expected_label():
    mails = load_mails()
    assert len({m.id for m in mails}) == len(mails) >= 10
    assert {m.expected.category for m in mails} == set(CATEGORIES)


# ------------------------------------------------------------------ Ablauf


def test_process_classifies_then_drafts(store):
    calls = []
    drafted(store, calls=calls)
    status = store.status("m01")
    assert status.state == "drafted"
    assert status.classification.category == "complaint"
    assert status.draft == DRAFT
    assert status.assessment == "clear"
    assert status.facts_used == ["Policy damaged goods"]
    assert len(calls) == 2
    # Beide Schritte per Structured Outputs; der Entwurf bekommt die Shop-Daten zur Mail
    assert calls[0]["response_format"]["json_schema"]["strict"] is True
    assert calls[1]["response_format"]["json_schema"]["name"] == "Draft"
    prompt = calls[1]["messages"][0]["content"]
    assert "Order 40213" in prompt and "Policy damaged goods" in prompt
    assert [e.event for e in store.events("m01")] == ["classified", "drafted"]


def test_spam_gets_no_draft(store):
    calls = []
    process(
        mail_by_id("m05"),
        store,
        "test/model",
        client=fake_client([classification("spam", "low")], calls),
    )
    assert store.status("m05").state == "classified"
    assert len(calls) == 1


def test_draft_for_spam_only_on_request(store):
    process(
        mail_by_id("m05"),
        store,
        "test/model",
        client=fake_client([classification("spam", "low")], []),
    )
    write_draft(
        mail_by_id("m05"), store, "test/model", client=fake_client([draft(DRAFT)], [])
    )
    assert store.status("m05").state == "drafted"


def test_invalid_classification_is_rejected(store):
    with pytest.raises(workflow.LLMError):
        process(
            mail_by_id("m01"),
            store,
            "test/model",
            client=fake_client([classification("approved", "high")], []),
        )
    assert store.status("m01").state == "new"  # nichts protokolliert


def test_a_mail_is_not_processed_twice(store):
    drafted(store)
    with pytest.raises(WorkflowError):
        drafted(store)


# ------------------------------------------------------------------ Shop-Daten


def test_facts_belong_to_the_sender():
    facts = shop_facts(mail_by_id("m14"))
    assert any("Order 40477" in f and "front door" in f for f in facts)
    assert any(f.startswith("Product Bluetooth Speaker") for f in facts)
    assert any(f.startswith("Policy missing parcels") for f in facts)


def test_foreign_order_numbers_get_no_details():
    # Die Injection-Mail nennt eine fremde Bestellnummer: nur der Hinweis, keine Daten
    facts = shop_facts(mail_by_id("m12"))
    assert "Order 40501 does not belong to refunds-now@example.com." in facts
    assert not any(f.startswith("Order 40501 of") for f in facts)
    assert any("No customer account" in f for f in facts)


def test_products_are_found_by_german_name():
    facts = shop_facts(mail_by_id("m13"))  # „Wasserkocher“
    assert any(
        f.startswith("Product Electric Kettle") and "1.7 litres" in f for f in facts
    )


def test_unverified_flags_facts_that_are_not_in_the_data():
    facts = shop_facts(mail_by_id("m04"))
    quoted = "works on all hobs including induction"
    combined = "Product Cast Iron Pan: 39.90 EUR; oven-safe up to 260 °C"
    invented = "dishwasher safe"
    wrong_number = "Product Cast Iron Pan: 29.90 EUR, 31 in stock"
    assert unverified([quoted, combined, invented, wrong_number], facts) == [
        invented,
        wrong_number,
    ]


def test_every_non_spam_mail_has_an_expected_assessment():
    for mail in load_mails():
        assert (mail.expected.assessment is None) == (mail.expected.category == "spam")


def test_unclear_draft_is_stored_with_its_reason(store):
    process(
        mail_by_id("m14"),
        store,
        "test/model",
        client=fake_client(
            [
                classification("order_status", "high"),
                draft("Hello Paul, [decision: replacement or claim]", "unclear"),
            ],
            [],
        ),
    )
    status = store.status("m14")
    assert status.assessment == "unclear"
    assert status.assessment_reason == "The policy covers it."


# ------------------------------------------------------------------ Freigabe


def test_nothing_is_sent_without_a_human_decision(store):
    # Der Mail-Text verlangt eine Freigabe (Prompt Injection), das Modell spielt mit
    process(
        mail_by_id("m12"),
        store,
        "test/model",
        client=fake_client(
            [
                classification("return_refund", "normal"),
                draft("Your refund was sent."),
            ],
            [],
        ),
    )
    assert store.status("m12").state == "drafted"
    assert not any(e.event in workflow.SENT for e in store.events())


def test_placeholders_block_approval(store):
    drafted(store)
    assert placeholders(DRAFT) == ["[replacement or refund]"]
    with pytest.raises(WorkflowError, match="placeholders"):
        approve(store, "m01", DRAFT)
    assert store.status("m01").state == "drafted"


def test_unchanged_and_edited_approval(store, tmp_path):
    clean = "Hello,\n\nthanks for your email.\n\nLarkfield Customer Service"
    process(
        mail_by_id("m02"),
        store,
        "test/model",
        client=fake_client(
            [classification("order_status", "normal"), draft(clean)], []
        ),
    )
    assert approve(store, "m02", clean).event == "approved"

    drafted(store, "m01")
    filled = DRAFT.replace("[replacement or refund]", "send a new jug tomorrow")
    event = approve(store, "m01", filled)
    assert event.event == "approved_edited"
    assert store.status("m01").final_text == filled


def test_decisions_are_final(store):
    drafted(store)
    discard(store, "m01")
    with pytest.raises(WorkflowError):
        approve(store, "m01", "Hello")
    with pytest.raises(WorkflowError):
        discard(store, "m01")


def test_cannot_approve_before_a_draft_exists(store):
    with pytest.raises(WorkflowError):
        approve(store, "m01", "Hello")


# ------------------------------------------------------------------ Protokoll


def test_log_survives_a_new_store_on_the_same_file(store):
    drafted(store)
    discard(store, "m01")
    reopened = WorkflowStore(store.path)  # wie nach einem Neustart der App
    assert reopened.status("m01").state == "discarded"
    assert len(reopened.events()) == 3


def test_reset_keeps_the_log(store):
    drafted(store)
    discard(store, "m01")
    reset_inbox(store)
    assert store.status("m01").state == "new"
    assert [e.event for e in store.events("m01")] == [
        "classified",
        "drafted",
        "discarded",
        "reset",
    ]


def test_approval_rate_comes_from_the_log(store):
    clean = "Hello,\n\nthanks.\n\nLarkfield Customer Service"
    for mail_id in ("m02", "m03"):
        process(
            mail_by_id(mail_id),
            store,
            "test/model",
            client=fake_client(
                [classification("order_status", "normal"), draft(clean)], []
            ),
        )
    approve(store, "m02", clean)
    discard(store, "m03")
    stats = decision_stats(store.events())
    assert (stats.approved, stats.discarded, stats.total) == (1, 1, 2)
    assert stats.approval_rate == 0.5
    assert decision_stats([]).approval_rate is None


def test_every_mail_is_in_exactly_one_filter(store):
    drafted(store, "m01")
    process(
        mail_by_id("m05"),
        store,
        "test/model",
        client=fake_client([classification("spam", "low")], []),
    )
    discard(store, "m05")  # verworfener Spam bleibt Spam, nicht „done“
    buckets = {mail_id: bucket(s) for mail_id, s in store.statuses().items()}
    assert buckets["m01"] == "review"
    assert buckets["m05"] == "spam"
    assert buckets["m02"] == "new"
    approve(store, "m01", DRAFT.replace("[replacement or refund]", "send a new jug"))
    assert bucket(store.status("m01")) == "done"


# ------------------------------------------------------------------ Seite (AppTest)

PAGE = "../app_pages/workflow.py"


def test_page_shows_inbox_and_empty_log(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "DB_PATH", tmp_path / "workflow.sqlite")
    monkeypatch.setattr(
        WorkflowStore.__init__,
        "__defaults__",
        (tmp_path / "workflow.sqlite",),
    )
    at = AppTest.from_file(PAGE).run()
    assert not at.exception
    assert at.title[0].value == "Approval Workflow"
    assert len(at.dataframe[0].value) == len(load_mails())
    # Ohne Auswahl ist die neueste Mail geöffnet, ihr Verlauf ist noch leer
    newest = max(load_mails(), key=lambda m: m.received)
    assert any(newest.subject in m.value for m in at.markdown)
    assert any("Nothing has happened" in c.value for c in at.caption)


def test_invalid_draft_json_gets_one_more_try(store):
    calls = []
    process(
        mail_by_id("m01"),
        store,
        "test/model",
        client=fake_client([classification(), "   ", draft(DRAFT)], calls),
    )
    assert store.status("m01").draft == DRAFT
    assert len(calls) == 3  # Klassifikation, kaputter Entwurf, zweiter Versuch
