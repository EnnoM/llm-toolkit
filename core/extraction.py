"""Datenextraktion: Freitext wird zu geprüftem JSON. Schemas und Ablauf, keine Oberfläche.

Das Zielformat steht jeweils als Pydantic-Modell hier. Daraus entstehen das JSON-Schema für
Structured Outputs (das Modell muss es einhalten), die Prüfung der Antwort und die
Feldübersicht im Modul. Besteht eine Antwort die Prüfung nicht, bekommt das Modell die
Fehlermeldung zurück und darf korrigieren. Texte sind englisch, weil das Modul englisch ist.
"""

import json
from copy import deepcopy
from datetime import date
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from core.llm import Message, Usage, complete, get_client

MAX_CORRECTIONS = 2  # nach dem ersten Versuch höchstens zwei Korrekturen
TEMPERATURE = 0.0  # Extraktion soll wiederholbar sein, nicht kreativ

SYSTEM_PROMPT = (
    "You extract structured data from text. Use only information that is stated in the "
    "text. If a field is not mentioned, set it to null (or an empty list) instead of "
    "guessing. Write dates as YYYY-MM-DD and times as HH:MM. Amounts are plain numbers "
    "without currency symbols or thousands separators."
)


class Schema(BaseModel):
    """Basis für alle Zielformate: keine zusätzlichen Felder erlaubt.

    Zwei Arten von Prüfung, bewusst getrennt:
    - Formfehler (Typ, Datumsformat, Pflichtfeld, Zusatzfeld) prüft Pydantic. Die macht das
      Modell, deshalb bekommt es die Meldung zurück und darf korrigieren.
    - Widersprüche im Dokument selbst (Summe stimmt nicht) meldet warnings() nur als Hinweis.
      Ein Korrekturversuch würde das Modell drängen, Werte passend zu machen, die so nicht
      im Text stehen; Test mit gemini-2.5-flash-lite am 23.09.2026: Gesamtbetrag einfach „korrigiert“.
    """

    model_config = ConfigDict(extra="forbid")

    def warnings(self) -> list[str]:
        """Hinweise auf Widersprüche im Dokument, für den Menschen, nicht fürs Modell."""
        return []


# ------------------------------------------------------------------ Zielformate

# Toleranz beim Nachrechnen von Beträgen, damit Rundung auf Cent kein Fehler ist
CENT = 0.01


class LineItem(Schema):
    description: str = Field(description="What was bought or delivered.")
    quantity: float | None = Field(description="Number of units, null if not stated.")
    unit_price: float | None = Field(description="Price per unit, null if not stated.")
    amount: float = Field(description="Line total.")


class Invoice(Schema):
    vendor: str = Field(description="Company that issued the invoice.")
    invoice_number: str | None = Field(description="Invoice number as printed.")
    invoice_date: date | None = Field(description="Date of issue.")
    due_date: date | None = Field(description="Payment due date.")
    currency: str | None = Field(description="Three-letter code, e.g. EUR or USD.")
    line_items: list[LineItem] = Field(description="One entry per invoiced item.")
    net_amount: float | None = Field(description="Total before tax.")
    tax_amount: float | None = Field(description="Tax (VAT) amount.")
    total_amount: float = Field(description="Amount to pay, including tax.")
    tax_id: str | None = Field(description="Vendor's VAT or tax ID, null if missing.")

    def warnings(self) -> list[str]:
        # Nachrechnen findet Rechenfehler auf der Rechnung (oder Lesefehler des Modells)
        known = self.net_amount is not None and self.tax_amount is not None
        if known and abs(self.net_amount + self.tax_amount - self.total_amount) > CENT:
            expected = self.net_amount + self.tax_amount
            return [
                (
                    f"Totals don't add up: {self.net_amount:.2f} + {self.tax_amount:.2f} = "
                    f"{expected:.2f}, but the invoice says {self.total_amount:.2f}. "
                    "Check the document."
                )
            ]
        return []


class JobPosting(Schema):
    job_title: str = Field(description="Title of the position.")
    company: str = Field(description="Hiring company.")
    location: str | None = Field(description="City or region of the workplace.")
    remote: Literal["on-site", "hybrid", "remote"] | None = Field(
        description="Where the work happens, null if not stated."
    )
    employment_type: (
        Literal["full-time", "part-time", "contract", "internship"] | None
    ) = Field(description="Type of contract, null if not stated.")
    salary_min: float | None = Field(
        description="Lower end of the yearly salary range."
    )
    salary_max: float | None = Field(
        description="Upper end of the yearly salary range."
    )
    currency: str | None = Field(
        description="Three-letter code of the salary currency."
    )
    skills: list[str] = Field(description="Required skills and tools, short terms.")
    application_deadline: date | None = Field(description="Last day to apply.")

    def warnings(self) -> list[str]:
        known = self.salary_min is not None and self.salary_max is not None
        if known and self.salary_min > self.salary_max:
            return [
                (
                    f"Salary range is reversed: minimum {self.salary_min:.0f} is above "
                    f"maximum {self.salary_max:.0f}. Check the posting."
                )
            ]
        return []


class MeetingRequest(Schema):
    subject: str = Field(description="Topic of the meeting.")
    organizer: str = Field(description="Person who asks for the meeting.")
    participants: list[str] = Field(
        description="Names of everyone invited, organizer included."
    )
    meeting_date: date | None = Field(description="Day of the meeting.")
    # Als Text mit Muster statt als time: beim Format "time" erfinden Modelle sonst eine
    # Zeitzone dazu (gpt-4.1-nano lieferte "14:30:00-04:00")
    start_time: str | None = Field(
        description="Start time as HH:MM, 24-hour clock.",
        pattern=r"^([01]\d|2[0-3]):[0-5]\d$",
    )
    duration_minutes: int | None = Field(
        description="Length of the meeting in minutes."
    )
    location: str | None = Field(description="Room or address, null if online only.")
    online_link: str | None = Field(
        description="Video call link, null if none is given."
    )


class DocType(BaseModel):
    """Ein Dokumenttyp im Modul: Zielformat plus vorbefüllter Beispieltext."""

    title: str
    description: str  # kurze Beschreibung unter dem Titel in der Auswahl
    schema_model: type[Schema]
    sample_text: str


DOC_TYPES: dict[str, DocType] = {
    # Rechnung ohne Steuernummer: tax_id muss null bleiben (Akzeptanzkriterium)
    "invoice": DocType(
        title="Invoice",
        description="Amounts, dates, line items",
        schema_model=Invoice,
        sample_text=(
            "Brightline Web Studio\n"
            "Invoice no. BWS-2026-0412, issued 3 September 2026\n"
            "Bill to: Nordhafen Logistics GmbH\n\n"
            "Website redesign (fixed price) ........ 2,400.00 EUR\n"
            "Hosting, 12 months at 15.00 EUR ........ 180.00 EUR\n"
            "Extra support, 4 hours at 85.00 EUR .... 340.00 EUR\n\n"
            "Net total: 2,920.00 EUR\n"
            "VAT 19%: 554.80 EUR\n"
            "Total due: 3,474.80 EUR\n\n"
            "Please pay within 14 days, by 17 September 2026."
        ),
    ),
    "job_posting": DocType(
        title="Job posting",
        description="Role, salary range, skills",
        schema_model=JobPosting,
        sample_text=(
            "We're hiring! Greenfield Energy is looking for a Data Engineer (full-time) to "
            "join our analytics team in Hamburg. You can work from home up to three days a "
            "week. You'll build data pipelines in Python and SQL, run them on Airflow and "
            "keep our Snowflake warehouse in shape. Experience with dbt is a plus. We offer "
            "EUR 65,000 to 78,000 a year, 30 days of holiday and a bike leasing scheme. "
            "Apply by 31 October 2026 via our careers page."
        ),
    ),
    "meeting": DocType(
        title="Meeting request",
        description="Date, time, participants",
        schema_model=MeetingRequest,
        sample_text=(
            "Hi Lena, hi Tom,\n\n"
            "could we sit down for the Q4 budget review next Thursday, 8 October 2026? I'd "
            "suggest 14:30 for about 90 minutes in meeting room Elbe (2nd floor). Priya from "
            "finance will join us as well.\n\n"
            "Thanks,\nMarkus"
        ),
    ),
}


# ------------------------------------------------------------------ Schema für die API


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON-Schema des Modells in der strengen Form, die Structured Outputs verlangt.

    Strict Mode braucht in jedem Objekt additionalProperties: false und alle Felder unter
    required. Optionale Felder bleiben trotzdem möglich, weil ihr Typ null erlaubt.
    """
    schema = deepcopy(model.model_json_schema())

    def tighten(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            # Titel pro Feld braucht die API nicht, sie kosten nur Tokens
            node.pop("title", None)
            for value in node.values():
                tighten(value)
        elif isinstance(node, list):
            for item in node:
                tighten(item)

    tighten(schema)
    return schema


def response_format(doc_type: DocType) -> dict[str, Any]:
    """response_format für die API: das Modell muss genau dieses Schema einhalten."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": doc_type.schema_model.__name__,
            "strict": True,
            "schema": strict_json_schema(doc_type.schema_model),
        },
    }


# Nur Anbieter nehmen, die response_format wirklich unterstützen. Ohne diese Angabe darf
# OpenRouter an einen Anbieter weiterleiten, der das Schema stillschweigend ignoriert.
PROVIDER_PREFERENCES = {"provider": {"require_parameters": True}}


def schema_fields(model: type[BaseModel]) -> list[dict[str, str]]:
    """Feldübersicht für die Anzeige: Name, Typ und Beschreibung aus dem Pydantic-Modell."""
    properties = model.model_json_schema()["properties"]
    rows = []
    for name, info in model.model_fields.items():
        rows.append(
            {
                "Field": name,
                "Type": describe_type(properties[name]),
                "Description": info.description or "",
            }
        )
    return rows


def describe_type(prop: dict[str, Any]) -> str:
    """Kurzer Typname aus einem Schema-Eintrag, z. B. 'date or null', 'list of object'."""
    options = prop.get("anyOf", [prop])
    names = []
    for option in options:
        if "enum" in option:
            names.append(" | ".join(option["enum"]))
        elif "format" in option:
            names.append(option["format"])
        elif option.get("type") == "array":
            item = option.get("items", {})
            names.append(f"list of {item.get('type', 'object')}")
        elif "$ref" in option:
            names.append("object")
        else:
            names.append(option.get("type", "any"))
    return " or ".join(names)


# ------------------------------------------------------------------ Prüfung und Ablauf


def validate(doc_type: DocType, raw: str) -> tuple[Schema | None, str | None]:
    """Prüft eine Modellantwort. Gibt (Ergebnis, None) oder (None, Fehlermeldung) zurück.

    Reine Funktion, deshalb ohne API mit gespeicherten Antworten testbar.
    """
    try:
        return doc_type.schema_model.model_validate_json(raw), None
    except ValidationError as err:
        return None, format_errors(err)


def format_errors(err: ValidationError) -> str:
    """Kurze, fürs Modell und die Anzeige lesbare Fehlerliste, eine Zeile pro Fehler."""
    lines = []
    for error in err.errors(include_url=False):
        where = ".".join(str(part) for part in error["loc"]) or "(whole object)"
        lines.append(f"- {where}: {error['msg']}")
    return "\n".join(lines)


class Attempt(BaseModel):
    """Ein Versuch: rohe Antwort, Prüfergebnis, Verbrauch."""

    raw: str
    error: str | None = None  # Meldung der Prüfung, None = bestanden
    usage: Usage = Usage()


class ExtractionResult(BaseModel):
    """Ergebnis einer Extraktion mit allen Versuchen."""

    doc_type_id: str
    model: str
    data: dict[str, Any] | None = None  # geprüftes Ergebnis als JSON-kompatibles dict
    attempts: list[Attempt]
    error: str | None = None  # gesetzt, wenn auch die letzte Korrektur scheiterte
    warnings: list[str] = []  # Widersprüche im Dokument, nur als Hinweis angezeigt

    @property
    def usage(self) -> Usage:
        return sum((attempt.usage for attempt in self.attempts), Usage())


def build_messages(doc_type: DocType, text: str) -> list[Message]:
    """Erste Anfrage: Anweisung als System-Prompt, dann der Text."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Extract the {doc_type.title.lower()} data from this text:\n\n"
            f"{text.strip()}",
        },
    ]


def correction_message(error: str) -> Message:
    """Rückmeldung an das Modell, wenn seine Antwort die Prüfung nicht bestanden hat."""
    return {
        "role": "user",
        "content": "Your answer failed validation:\n"
        f"{error}\n\n"
        "Check the text again and return the corrected JSON. Use null for anything the "
        "text does not state.",
    }


def extract(
    doc_type_id: str, text: str, model: str, client: OpenAI | None = None
) -> ExtractionResult:
    """Extrahiert Daten und lässt das Modell bei Prüffehlern bis zu zweimal korrigieren.

    API-Fehler (Key fehlt, Modell nicht verfügbar) kommen als LLMError. Scheitert nur die
    Prüfung, steht die letzte Meldung in result.error, damit die Seite sie zeigen kann.
    """
    client = client or get_client()
    doc_type = DOC_TYPES[doc_type_id]
    messages = build_messages(doc_type, text)
    attempts: list[Attempt] = []
    for _ in range(1 + MAX_CORRECTIONS):
        response = complete(
            messages,
            model,
            temperature=TEMPERATURE,
            response_format=response_format(doc_type),
            extra_body=PROVIDER_PREFERENCES,
            client=client,
        )
        data, error = validate(doc_type, response.text)
        attempts.append(Attempt(raw=response.text, error=error, usage=response.usage))
        if data is not None:
            return ExtractionResult(
                doc_type_id=doc_type_id,
                model=model,
                data=data.model_dump(mode="json"),
                attempts=attempts,
                warnings=data.warnings(),
            )
        # Antwort und Fehlermeldung ins Gespräch, damit das Modell gezielt nachbessert
        messages = [
            *messages,
            {"role": "assistant", "content": response.text},
            correction_message(error),
        ]
    return ExtractionResult(
        doc_type_id=doc_type_id,
        model=model,
        attempts=attempts,
        error=f"No valid result after {MAX_CORRECTIONS} corrections. "
        f"Last validation errors:\n{error}",
    )


def pretty_json(raw: str) -> str:
    """Rohantwort eingerückt für die Anzeige; kein gültiges JSON bleibt, wie es ist."""
    try:
        return json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return raw
