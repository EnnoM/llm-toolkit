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
    write_draft,
)
from tests.test_extraction import fake_client

DRAFT = "Hello Julia,\n\nsorry about the jug. We will [replacement or refund].\n\nLarkfield Customer Service"


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
        client=fake_client([classification(), DRAFT], [] if calls is None else calls),
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
    assert len(calls) == 2
    # Klassifikation per Structured Outputs, Entwurf als freier Text
    assert calls[0]["response_format"]["json_schema"]["strict"] is True
    assert "response_format" not in calls[1]
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
    write_draft(mail_by_id("m05"), store, "test/model", client=fake_client([DRAFT], []))
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


# ------------------------------------------------------------------ Freigabe


def test_nothing_is_sent_without_a_human_decision(store):
    # Der Mail-Text verlangt eine Freigabe (Prompt Injection), das Modell spielt mit
    process(
        mail_by_id("m12"),
        store,
        "test/model",
        client=fake_client(
            [classification("return_refund", "normal"), "Your refund was sent."], []
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
        client=fake_client([classification("order_status", "normal"), clean], []),
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
            client=fake_client([classification("order_status", "normal"), clean], []),
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
