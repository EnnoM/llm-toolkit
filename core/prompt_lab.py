"""Prompt-Labor: dieselbe Aufgabe mit vier Prompt-Techniken. Vorlagen und Ablauf, keine Oberfläche.

Jede Technik baut aus Aufgabe und Eingabetext eine Nachrichtenliste. Weil das reine Funktionen
sind, lässt sich ohne API-Aufruf testen, was ans Modell geht. Die Prompts sind englisch, weil
das Modul englisch ist.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel

from core.llm import LLMError, Message, Usage, complete, get_client

Technique = Literal["zero_shot", "few_shot", "role", "chain_of_thought"]

TECHNIQUES: dict[Technique, str] = {
    "zero_shot": "Zero-shot",
    "few_shot": "Few-shot",
    "role": "Role",
    "chain_of_thought": "Chain-of-thought",
}

# Anweisung für Chain-of-Thought. Der Test prüft, dass sie im Prompt steht.
STEP_BY_STEP = (
    "Let's think step by step. Write out your reasoning first, then give the final result "
    "on the last line, starting with 'Final answer:'."
)

# Gleiche Bedingungen für alle Varianten: Unterschiede sollen vom Prompt kommen, nicht vom Zufall.
TEMPERATURE = 0.0


class Example(BaseModel):
    """Ein gelöstes Beispiel für Few-Shot."""

    input: str
    output: str


class LabTask(BaseModel):
    """Eine Beispielaufgabe mit allem, was die vier Techniken brauchen."""

    title: str
    description: str  # kurze Beschreibung unter dem Titel in der Auswahl
    instruction: str  # was das Modell tun soll
    sample_input: str  # vorbefüllter Eingabetext, im Modul änderbar
    examples: list[Example]  # gelöste Beispiele für Few-Shot
    role: str  # Persona für den Rollen-Prompt
    reference: str | None = None  # erwartetes Ergebnis, wenn es eindeutig ist


class VariantResult(BaseModel):
    """Ergebnis einer Technik: gesendeter Prompt, Antwort, Verbrauch, Dauer."""

    technique: Technique
    messages: list[Message]
    text: str = ""
    usage: Usage = Usage()
    seconds: float = 0.0
    error: str | None = None


class ComparisonRun(BaseModel):
    """Ein Vergleichslauf: alle gewählten Techniken mit derselben Aufgabe und demselben Modell."""

    task_id: str
    model: str
    results: list[VariantResult]
    seconds: float  # Gesamtdauer; die Varianten laufen parallel

    @property
    def usage(self) -> Usage:
        return sum((result.usage for result in self.results), Usage())


# ------------------------------------------------------------------ Beispielaufgaben

TASKS: dict[str, LabTask] = {
    # Rechenaufgabe mit mehreren Schritten und einer Ablenkung ("sieben Freunde"). Die Anweisung
    # verlangt nur den Betrag, also keine Zeit zum Nachdenken; nur Chain-of-thought darf rechnen.
    # Test am 21.09.2026: gemini-2.5-flash-lite antwortet zero-shot 23.10, mit CoT richtig 14.03.
    "word_problem": LabTask(
        title="Word problem",
        description="Multi-step arithmetic",
        instruction="Solve the word problem. Reply with the amount only, for example '€12.34'.",
        sample_input=(
            "A café sells coffee for €3.20 and cake for €4.50. A group of seven friends "
            "orders five coffees and three pieces of cake. Two of the coffees are refilled at "
            "half price. They pay with a €50 note and add a 10% tip on the bill. How much "
            "change do they get back?"
        ),
        examples=[
            Example(
                input=(
                    "A bookshop sells notebooks for €2.50 and pens for €1.20. Anna buys four "
                    "notebooks and three pens and pays with a €20 note. How much change does "
                    "she get back?"
                ),
                output="€6.40",
            ),
            Example(
                input=(
                    "A cinema ticket costs €9.50 and popcorn costs €4.00. Three friends each "
                    "buy a ticket and share two popcorns. They pay with €50. How much change "
                    "do they get back?"
                ),
                output="€13.50",
            ),
        ],
        role="You are a meticulous accountant who double-checks every calculation.",
        reference="€14.03",
    ),
    # Einordnen in feste Kategorien, eine Zeile pro Problem. Die Mail enthält zwei Probleme mit
    # unterschiedlicher Dringlichkeit; ein Few-Shot-Beispiel zeigt genau dieses Muster.
    "email_triage": LabTask(
        title="Email triage",
        description="Category and priority per issue",
        instruction=(
            "Classify the customer email. Write one line per separate issue in the format "
            "'Category: X | Priority: Y'. Categories: Billing, Technical, Cancellation, "
            "Feedback. Priorities: Low, Medium, High. Rate each issue on its own: High only "
            "if that issue blocks the customer's work or has a deadline."
        ),
        sample_input=(
            "Hi, I was charged twice for my subscription this month, and since yesterday the "
            "app keeps logging me out. I need this fixed today because I'm presenting with it "
            "tomorrow morning."
        ),
        examples=[
            Example(
                input=(
                    "Your latest update deleted all my saved projects. I have a client "
                    "deadline on Friday!"
                ),
                output="Category: Technical | Priority: High",
            ),
            Example(
                input="Please cancel my plan at the end of this month. Thanks for everything.",
                output="Category: Cancellation | Priority: Low",
            ),
            Example(
                input="Love the new dark mode, it's much easier on the eyes.",
                output="Category: Feedback | Priority: Low",
            ),
            Example(
                input="I was billed €30 instead of €20 this month. Can you check?",
                output="Category: Billing | Priority: Medium",
            ),
            # Zwei Probleme mit unterschiedlicher Dringlichkeit: macht die Abwägung vor, dass
            # eine Fehlbuchung ärgerlich, aber nicht blockierend ist (Medium)
            Example(
                input=(
                    "You charged my card twice for last month, and the mobile app crashes "
                    "whenever I upload photos. I need the upload working for a client meeting "
                    "this afternoon."
                ),
                output="Category: Technical | Priority: High\nCategory: Billing | Priority: Medium",
            ),
        ],
        role=(
            "You are an experienced customer support team lead who triages hundreds of "
            "emails a day."
        ),
        reference="Category: Billing | Priority: Medium · Category: Technical | Priority: High",
    ),
    # Zusammenfassung für ein Publikum: hier prägen Rolle und Beispiele Ton und Länge
    "summary": LabTask(
        title="Incident summary",
        description="Plain language for managers",
        instruction=(
            "Summarize the incident report for a non-technical manager in at most two "
            "sentences."
        ),
        sample_input=(
            "Incident report, 14 March: At 09:12 the payment service began rejecting about "
            "30% of card payments. The on-call engineer traced the errors to an expired TLS "
            "certificate on one of three load balancers; requests routed to that node failed "
            "the handshake with the payment provider. Monitoring flagged the rising error rate "
            "after four minutes, but the alert went to a chat channel nobody watched during "
            "the team meeting. The certificate was renewed at 10:03 and error rates were back "
            "to normal by 10:07. Roughly 1,900 payments failed; customers could retry, and "
            "support received 140 complaints. Follow-ups: automate certificate renewal, alert "
            "30 days before expiry, and route critical alerts to phone calls instead of chat."
        ),
        examples=[
            Example(
                input=(
                    "Incident report, 2 February: The nightly backup job failed three nights "
                    "in a row because the storage volume was full. Nobody noticed until a "
                    "developer needed to restore a deleted table. The data came from an older "
                    "backup, losing two days of changes to the reports table. Follow-ups: "
                    "alert on failed backups, grow the volume automatically."
                ),
                output=(
                    "Backups silently failed for three nights because the disk was full, "
                    "which cost us two days of report data. We are adding alerts and "
                    "automatic disk growth so this cannot go unnoticed again."
                ),
            ),
            Example(
                input=(
                    "Incident report, 20 February: A configuration change made the website "
                    "search return no results for 45 minutes. It was rolled back after "
                    "customers reported it on social media. About 3,000 searches returned "
                    "empty pages. Follow-ups: test search in the deployment pipeline."
                ),
                output=(
                    "A faulty configuration change broke website search for 45 minutes, and "
                    "about 3,000 searches came back empty. Search is now tested automatically "
                    "before every release."
                ),
            ),
        ],
        role=(
            "You are a communications lead who explains technical problems to executives in "
            "plain language."
        ),
    ),
}


# ------------------------------------------------------------------ Prompt-Vorlagen


def build_messages(task: LabTask, technique: Technique, text: str) -> list[Message]:
    """Nachrichten für eine Technik. Reine Funktion, deshalb ohne API testbar."""
    request = f"{task.instruction}\n\n{text.strip()}"
    if technique == "zero_shot":
        # Nur die Aufgabe, keine Hilfen
        return [{"role": "user", "content": request}]
    if technique == "few_shot":
        # Gelöste Beispiele als frühere Gesprächsrunden; das Modell übernimmt Muster und Format
        messages: list[Message] = []
        for example in task.examples:
            messages.append(
                {"role": "user", "content": f"{task.instruction}\n\n{example.input}"}
            )
            messages.append({"role": "assistant", "content": example.output})
        return [*messages, {"role": "user", "content": request}]
    if technique == "role":
        # Persona als System-Prompt, die eigentliche Aufgabe bleibt gleich
        return [
            {"role": "system", "content": task.role},
            {"role": "user", "content": request},
        ]
    # chain_of_thought: erst Lösungsweg ausschreiben, dann Ergebnis
    return [{"role": "user", "content": f"{request}\n\n{STEP_BY_STEP}"}]


def format_messages(messages: list[Message]) -> str:
    """Prompt lesbar für die Anzeige „Prompt sent“: Rolle in Klammern, dann Inhalt."""
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)


# ------------------------------------------------------------------ Ablauf


def run_variant(
    task: LabTask, technique: Technique, text: str, model: str, client: OpenAI
) -> VariantResult:
    """Schickt eine Variante ab und misst die Antwortzeit. Fehler landen im Ergebnis."""
    messages = build_messages(task, technique, text)
    start = time.perf_counter()
    try:
        response = complete(messages, model, temperature=TEMPERATURE, client=client)
    except LLMError as err:
        return VariantResult(
            technique=technique,
            messages=messages,
            seconds=time.perf_counter() - start,
            error=str(err),
        )
    return VariantResult(
        technique=technique,
        messages=messages,
        text=response.text,
        usage=response.usage,
        seconds=time.perf_counter() - start,
    )


def run_comparison(
    task_id: str,
    techniques: list[Technique],
    text: str,
    model: str,
    client: OpenAI | None = None,
) -> ComparisonRun:
    """Alle gewählten Techniken gleichzeitig, mit demselben Modell und derselben Eingabe.

    Parallel dauert der Lauf so lange wie die langsamste Variante statt wie alle zusammen.
    Fehlt der API-Key, bricht der Lauf vorher mit LLMError ab.
    """
    client = client or get_client()
    task = TASKS[task_id]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(len(techniques), 1)) as pool:
        futures = [
            pool.submit(run_variant, task, technique, text, model, client)
            for technique in techniques
        ]
        results = [future.result() for future in futures]
    return ComparisonRun(
        task_id=task_id,
        model=model,
        results=results,
        seconds=time.perf_counter() - start,
    )
