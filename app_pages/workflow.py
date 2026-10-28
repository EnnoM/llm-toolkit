"""Approval Workflow: ein kleines Mailprogramm, in dem das Modell sortiert und entwirft und ein
Mensch jede Antwort freigibt.

Layout als Ausnahme von docs/design.md (dort begründet): Kopfzeile mit Kennzahlen, darunter
einklappbare Einstellungen (Modell, Aktionen), dann das Mailprogramm in einem Rahmen: links die
Mailliste mit Schnellfiltern, rechts die geöffnete Mail mit Entwurf und Verlauf. Ganz unten das
Protokoll aller Mails.
Die Logik steht in core/workflow.py; diese Seite ruft nur deren Schritte auf.
"""

import pandas as pd
import streamlit as st

from core.config import get_secret
from core.llm import LLMError, Usage
from core.models import (
    format_cost,
    format_usage,
    model_name,
    model_options,
    model_price,
)
from core.workflow import (
    BUCKETS,
    PRIORITIES,
    Event,
    Mail,
    MailStatus,
    WorkflowError,
    WorkflowStore,
    approve,
    bucket,
    decision_stats,
    discard,
    load_mails,
    placeholders,
    process,
    reset_inbox,
    write_draft,
)

st.set_page_config(
    layout="wide"
)  # Liste und Mail nebeneinander brauchen die volle Breite

# Kurz, weil die Schreibmaschinenschrift breit ist und die Liste nur 60 % der Seite hat
CATEGORY_LABELS = {
    "complaint": "Complaint",
    "order_status": "Order",
    "return_refund": "Refund",
    "product_question": "Product",
    "spam": "Spam",
}
STATE_LABELS = {
    "new": "New",
    "classified": "To review",
    "drafted": "To review",
    "approved": "Sent",
    "approved_edited": "Sent (edited)",
    "discarded": "Discarded",
}
BUCKET_LABELS = {
    "all": "All",
    "new": "New",
    "review": "To review",
    "done": "Done",
    "spam": "Spam",
}
EVENT_LABELS = {
    "classified": "Classified",
    "drafted": "Draft written",
    "approved": "Sent unchanged",
    "approved_edited": "Sent with edits",
    "discarded": "Discarded",
    "reset": "Reset to new",
}


@st.cache_resource
def get_store() -> WorkflowStore:
    return WorkflowStore()


def remember(usage: Usage) -> None:
    """Kosten des letzten Modellschritts und der Sitzung für die Kopfzeile."""
    st.session_state.workflow_last = usage
    st.session_state.workflow_cost = (
        st.session_state.get("workflow_cost", 0.0) + usage.cost_usd
    )


def run_step(step) -> None:
    """Führt einen Schritt aus und zeigt Fehler als Meldung statt als Absturz."""
    try:
        step()
    except (LLMError, WorkflowError) as err:
        st.session_state.workflow_error = str(err)
    st.rerun()


def as_text(text: str) -> str:
    """Mailtext als Markdown: Zeilenumbrüche behalten, Sonderzeichen nicht als Formatierung."""
    escaped = "".join("\\" + ch if ch in "\\`*_[]#$<>|~" else ch for ch in text.strip())
    return escaped.replace("\n", "  \n")


def actor(event: Event) -> str:
    """Wer den Schritt ausgelöst hat: das Modell oder ein Mensch."""
    if event.model:
        return model_name(event.model)
    return "" if event.event == "reset" else "Human"


def event_details(event: Event) -> str:
    if event.event == "classified":
        return f"{event.category}, {event.priority} priority"
    return event.text or ""


store = get_store()
mails = sorted(load_mails(), key=lambda m: m.received, reverse=True)  # neueste zuerst
statuses = store.statuses()
events = store.events()

# Kopfzeile: Titel links, Kennzahlen als eine Textzeile rechts daneben. Große st.metric-Zahlen
# hätten eine eigene Zeile gebraucht
head = st.container(horizontal=True, vertical_alignment="bottom")
head.title("Approval Workflow", width="content")
head.space("stretch")
decisions = decision_stats(events)
rate = decisions.approval_rate
last = st.session_state.get("workflow_last")
cost = st.session_state.get("workflow_cost", 0.0)
head.markdown(
    f"Approval rate **{'–' if rate is None else f'{rate:.0%}'}** · "
    f"Sent **{decisions.approved + decisions.approved_edited}** · "
    f"Discarded **{decisions.discarded}** · Cost **{format_cost(cost)}**",
    width="content",
    help=(
        "Approval rate: share of decisions where the draft was sent unchanged, from "
        "the audit log. Cost: all model calls in this session."
        + (f" Last step: {format_usage(last)}." if last else "")
    ),
)
st.caption(
    "Fictional emails to a fictional shop. The model sorts them and drafts replies, a human "
    "approves every reply. Nothing is really sent."
)

# ------------------------------------------------------------------ Einstellungen

# Einklappbar und vom Mailprogramm getrennt: Einstellungen braucht man selten, die Mails ständig
settings = st.expander("Settings", icon=":material/tune:", expanded=True)
model_box, inbox_box = settings.columns([3, 2], gap="medium")
with model_box.container(border=True, height="stretch"):
    st.markdown("**Model**")
    default_model = get_secret("OPENROUTER_MODEL")
    options = model_options(default_model)  # nach Preis sortiert, günstigstes zuerst
    model = st.segmented_control(
        "Model",
        options,
        default=default_model if default_model in options else options[0],
        required=True,
        # Ein Button pro Modell, der Preis steht klein im Button
        format_func=lambda m: f"{model_name(m)} :small[{model_price(m) or ''}]".replace(
            "$", "\\$"
        ),
        key="workflow_model",
        label_visibility="collapsed",
    )
    st.caption(
        "USD per million input / output tokens. The model sorts the emails and writes "
        "drafts; it can never send anything."
    )
with inbox_box.container(border=True, height="stretch"):
    st.markdown("**Inbox**")
    actions = st.container(horizontal=True)
    new_mails = [m for m in mails if statuses[m.id].state == "new"]
    if actions.button(
        f"Process all ({len(new_mails)})",
        icon=":material/play_arrow:",
        disabled=not new_mails,
        help="Classifies every new email and writes a draft for all except spam.",
    ):

        def process_all() -> None:
            usage = Usage()
            try:
                with st.spinner("Classifying and drafting..."):
                    for mail in reversed(new_mails):  # älteste zuerst, wie ein Mensch
                        usage += process(mail, store, model)
            finally:
                # Auch nach einem Fehler mitten im Lauf: Bezahlt sind die Mails davor trotzdem
                remember(usage)

        run_step(process_all)
    if actions.button(
        "Reset inbox",
        icon=":material/restart_alt:",
        help="Sets every email back to new. The audit log keeps all entries.",
    ):
        run_step(lambda: reset_inbox(store))
    st.caption("Reset keeps the audit log; it only adds a reset entry.")

if error := st.session_state.pop("workflow_error", None):
    st.error(error, icon=":material/error:")

# ------------------------------------------------------------------ Mailprogramm

# Ein Rahmen um Liste und geöffnete Mail: Das ist das eigentliche Programm, getrennt von den
# Einstellungen darüber
program = st.container(border=True)
inbox, reader = program.columns([3, 2], gap="large")


def mail_row(mail: Mail, status: MailStatus) -> dict:
    c = status.classification
    return {
        "Received": mail.received.strftime("%H:%M"),  # alle vom selben Morgen
        "From": mail.name,
        "Subject": mail.subject,
        # Eine Spalte statt zwei, damit der Betreff Platz behält
        "Category": f"{CATEGORY_LABELS[c.category]} · {c.priority}" if c else "",
        "State": STATE_LABELS[status.state],
    }


def open_selected(key: str) -> None:
    """Klick in eine Zeile öffnet die Mail. Gemerkt wird die Id, nicht die Zeilennummer:
    Nach einem Schritt kann die Mail aus dem Filter fallen, die Zeilen rutschen dann nach."""
    cells = st.session_state[key].selection.cells
    if cells:
        st.session_state.workflow_mail = st.session_state.workflow_visible[cells[0][0]]


with inbox:
    buckets = {b: [m for m in mails if bucket(statuses[m.id]) == b] for b in BUCKETS}
    filters = st.container(horizontal=True, vertical_alignment="center")
    shown_bucket = filters.segmented_control(
        "State",
        ["all", *BUCKETS],
        default="all",
        required=True,
        format_func=lambda b: (
            f"{BUCKET_LABELS[b]} {len(mails) if b == 'all' else len(buckets[b])}"
        ),
        key="workflow_bucket",
        label_visibility="collapsed",
    )
    shown_priorities = filters.pills(
        "Priority",
        PRIORITIES,
        selection_mode="multi",
        format_func=str.capitalize,
        key="workflow_priority",
        label_visibility="collapsed",
        help="Show only these priorities. None selected shows all.",
    )
    visible = [
        m
        for m in (mails if shown_bucket == "all" else buckets[shown_bucket])
        if not shown_priorities
        or (
            statuses[m.id].classification
            and statuses[m.id].classification.priority in shown_priorities
        )
    ]
    st.session_state.workflow_visible = [m.id for m in visible]
    if visible:
        # Ein Key pro Filter: Wechselt der Filter, beginnt die Auswahl in der Tabelle neu
        key = f"inbox_{shown_bucket}_{'-'.join(shown_priorities)}"
        st.dataframe(
            pd.DataFrame([mail_row(m, statuses[m.id]) for m in visible]),
            hide_index=True,
            # Zelle statt Zeile: ein Klick irgendwo in die Zeile genügt, ohne Kästchen davor
            on_select=lambda: open_selected(key),
            selection_mode="single-cell",
            key=key,
            height=(len(visible) + 1) * 35
            + 3,  # alle Mails ohne Scrollen (35 px pro Zeile)
            column_config={
                "Received": st.column_config.TextColumn(width=60),
                # Zusammen gut 550 px, so breit ist die Liste auf einem Laptop
                "From": st.column_config.TextColumn(width=105),
                "Subject": st.column_config.TextColumn(width=165),
                "Category": st.column_config.TextColumn(width=125),
                "State": st.column_config.TextColumn(width=95),
            },
        )
        st.caption("Click an email to open it.")
    else:
        st.caption("No emails match this filter.")


def show_decision(mail_id: str, status: MailStatus) -> None:
    """Entwurf bearbeiten und entscheiden. Nur hier entsteht „Sent“, und nur per Klick."""
    st.markdown("##### Draft")
    # Key mit Entwurfs-Id: Ein neuer Entwurf ersetzt den alten Text im Feld
    text = st.text_area(
        "Reply",
        status.draft,
        height=260,
        key=f"draft_{mail_id}_{status.draft_event_id}",
        label_visibility="collapsed",
    )
    edited = text.strip() != (status.draft or "").strip()
    gaps = placeholders(text)
    if gaps:
        st.caption(
            f"Fill in before sending: {', '.join(gaps)}. "
            "The model cannot see orders or payments, so it leaves these gaps."
        )
    buttons = st.container(horizontal=True)
    if buttons.button(
        "Approve",
        type="primary",
        icon=":material/send:",
        disabled=edited or bool(gaps),
        key="approve",
    ):
        run_step(lambda: approve(store, mail_id, text))
    if buttons.button(
        "Approve edited",
        icon=":material/edit_note:",
        disabled=not edited or bool(gaps),
        key="approve_edited",
    ):
        run_step(lambda: approve(store, mail_id, text))
    if buttons.button("Discard", icon=":material/delete:", key="discard"):
        run_step(lambda: discard(store, mail_id))


def show_history(mail_events: list[Event]) -> None:
    """Verlauf der geöffneten Mail: wer hat wann was getan, neueste zuerst."""
    st.markdown("##### History")
    if not mail_events:
        st.caption("Nothing has happened to this email yet.")
        return
    for e in reversed(mail_events):
        line = (
            f"**{EVENT_LABELS[e.event]}** · {e.time[5:16].replace('T', ' ')}"
            + (f" · {actor(e)}" if actor(e) else "")
            + (f" · {format_cost(e.cost_usd)}" if e.model else "")
        )
        st.markdown(line)
        if e.event == "classified":
            st.caption(f"{e.category}, {e.priority} priority: {e.reason}")


with reader:
    # Ausgewählte Mail: Klick in der Liste, sonst Link mit ?mail=..., sonst die neueste
    ids = [m.id for m in mails]
    selected = st.session_state.get("workflow_mail") or st.query_params.get("mail")
    mail = next((m for m in mails if m.id == selected), mails[0])
    if mail.id != ids[0] or "mail" in st.query_params:
        st.query_params["mail"] = mail.id  # Link auf genau diese Mail
    status = statuses[mail.id]

    st.markdown(f"#### {mail.subject}")
    st.caption(
        f"From {mail.name} <{mail.address}> · "
        f"{mail.received.strftime('%A, %d %B %Y, %H:%M')} · {STATE_LABELS[status.state]}"
    )
    with st.container(border=True):
        st.markdown(as_text(mail.body))

    c = status.classification
    if c:
        st.markdown(
            f":gray-badge[{c.category}] :gray-badge[{c.priority} priority]  \n"
            f":small[{as_text(c.reason)}]"
        )

    if status.state == "new":
        if st.button(
            "Classify and draft", type="primary", icon=":material/play_arrow:"
        ):

            def process_one() -> None:
                with st.spinner("Classifying and drafting..."):
                    remember(process(mail, store, model))

            run_step(process_one)
    elif status.state == "classified":
        if c.category == "spam":
            st.info(
                "Classified as spam, so no draft was written. Discard it, or ask for a "
                "draft if the classification is wrong.",
                icon=":material/report:",
            )
        else:
            # Schritt 2 ist gescheitert (z. B. Rate-Limit), Schritt 1 steht schon im Log
            st.warning(
                "The draft is missing because the second step failed. Write it now.",
                icon=":material/warning:",
            )
        buttons = st.container(horizontal=True)
        if buttons.button("Discard", icon=":material/delete:"):
            run_step(lambda: discard(store, mail.id))
        if buttons.button(
            "Write draft anyway" if c.category == "spam" else "Write draft",
            icon=":material/edit:",
        ):

            def draft_anyway() -> None:
                with st.spinner("Drafting..."):
                    remember(write_draft(mail, store, model))

            run_step(draft_anyway)
    elif status.state == "drafted":
        show_decision(mail.id, status)
    elif status.state == "discarded":
        st.caption("Discarded, no reply was sent.")
    else:
        st.success(
            "Sent (simulated)"
            + (" with your edits." if status.state == "approved_edited" else "."),
            icon=":material/check_circle:",
        )
        with st.container(border=True):
            st.markdown(as_text(status.final_text))

    show_history([e for e in events if e.mail_id == mail.id])

# ------------------------------------------------------------------ Protokoll aller Mails

with st.expander(f"Audit log, all emails ({len(events)})", icon=":material/history:"):
    if events:
        subjects = {m.id: m.subject for m in mails}
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Time": e.time[5:19].replace("T", " "),  # „10-28 09:43:59“
                        "Email": subjects.get(e.mail_id, e.mail_id),
                        "Event": EVENT_LABELS[e.event],
                        "By": actor(e),
                        "Details": event_details(e),
                        "Cost (USD)": format_cost(e.cost_usd, unit=False)
                        if e.model
                        else "",
                    }
                    for e in reversed(events)  # neueste zuerst
                ]
            ),
            hide_index=True,
            height=360,
            column_config={
                "Time": st.column_config.TextColumn(width=140),
                "Email": st.column_config.TextColumn(width=220),
                "Event": st.column_config.TextColumn(width=140),
                "By": st.column_config.TextColumn(width=170),
                "Details": st.column_config.TextColumn(width="large"),
            },
        )
    else:
        st.caption("No entries yet. Every step and decision will appear here.")
