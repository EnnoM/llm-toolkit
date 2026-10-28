"""Prüft die Klassifikation des Workflow-Moduls gegen die Musterlösung. Keine Oberfläche.

Mails und Musterlösung stehen in eval/workflow_mails.yaml. Geprüft wird nur Schritt 1, die
Klassifikation und die Einschätzung des Entwurfs. Den Text der Entwürfe beurteilt im Modul ein
Mensch (und später das Modul „Evaluation“ mit einem Modell als Prüfer).

Zusätzlich Schritt 2 für jede Mail, die einen Entwurf bekommt: Ist die Einschätzung clear oder
unclear richtig, und stehen alle Fakten, die das Modell als verwendet angibt, wörtlich in den
Shop-Daten? Dafür bekommt der Entwurf die richtige Klassifikation aus der Musterlösung, damit
ein Fehler in Schritt 1 Schritt 2 nicht mitzieht.

Schwere Fehler werden extra gezählt: eine echte Kundenmail als Spam (sie würde verworfen) und
eine dringende Mail als low (sie bliebe liegen).

Aufruf: python -m core.workflow_eval [--model MODEL] [--runs N]
Ergebnis als Tabelle im Terminal und als JSON in eval/results/.
"""

import argparse
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from core.analyst_eval import RESULTS_DIR
from core.llm import STATUS_MESSAGES, LLMError
from core.workflow import (
    Classification,
    Mail,
    classify,
    draft_reply,
    load_mails,
    shop_facts,
    unverified,
)

DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
RETRIES = 3  # Rate-Limits der Anbieter (429) verfälschen sonst die Quote
RETRY_PAUSE = 10  # Sekunden


class WorkflowEvalResult(BaseModel):
    id: str
    expected: str  # „category/priority“
    got: str
    category_ok: bool
    priority_ok: bool
    severe: bool
    reason: str
    cost_usd: float


class DraftEvalResult(BaseModel):
    id: str
    expected: str
    got: str
    assessment_ok: bool
    unverified_facts: list[str]
    reason: str
    cost_usd: float


def evaluate_drafts(mails: tuple[Mail, ...], model: str) -> list[DraftEvalResult]:
    results = []
    for mail in mails:
        exp = mail.expected
        if exp.assessment is None:
            continue  # Spam bekommt keinen Entwurf
        given = Classification(reason="", category=exp.category, priority=exp.priority)
        try:
            draft, usage = with_retry(draft_reply, mail, given, model)
        except LLMError as err:
            results.append(
                DraftEvalResult(
                    id=mail.id,
                    expected=exp.assessment,
                    got="error",
                    assessment_ok=False,
                    unverified_facts=[],
                    reason=str(err),
                    cost_usd=0.0,
                )
            )
            continue
        results.append(
            DraftEvalResult(
                id=mail.id,
                expected=exp.assessment,
                got=draft.assessment,
                assessment_ok=draft.assessment == exp.assessment,
                unverified_facts=unverified(draft.facts_used, shop_facts(mail)),
                reason=draft.assessment_reason,
                cost_usd=usage.cost_usd,
            )
        )
    return results


def is_severe(expected_category, expected_priority, category, priority) -> bool:
    missed_customer = expected_category != "spam" and category == "spam"
    missed_urgency = expected_priority == "high" and priority == "low"
    return missed_customer or missed_urgency


def with_retry(call, *args):
    """Führt einen Modellaufruf aus, wartet bei „Too many requests“ und versucht es erneut."""
    for attempt in range(RETRIES):
        try:
            return call(*args)
        except LLMError as err:
            if STATUS_MESSAGES[429] != str(err) or attempt == RETRIES - 1:
                raise
            time.sleep(RETRY_PAUSE)


def evaluate(mails: tuple[Mail, ...], model: str) -> list[WorkflowEvalResult]:
    results = []
    for mail in mails:
        exp = mail.expected
        try:
            c, usage = with_retry(classify, mail, model)
        except LLMError as err:
            results.append(
                WorkflowEvalResult(
                    id=mail.id,
                    expected=f"{exp.category}/{exp.priority}",
                    got="error",
                    category_ok=False,
                    priority_ok=False,
                    severe=False,
                    reason=str(err),
                    cost_usd=0.0,
                )
            )
            continue
        results.append(
            WorkflowEvalResult(
                id=mail.id,
                expected=f"{exp.category}/{exp.priority}",
                got=f"{c.category}/{c.priority}",
                category_ok=c.category == exp.category,
                priority_ok=c.priority == exp.priority,
                severe=is_severe(exp.category, exp.priority, c.category, c.priority),
                reason=c.reason,
                cost_usd=usage.cost_usd,
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    mails = load_mails()
    today = datetime.now(ZoneInfo("Europe/Berlin")).date()
    RESULTS_DIR.mkdir(exist_ok=True)
    for number in range(1, args.runs + 1):
        results = evaluate(mails, args.model)
        for r in results:
            mark = "ok" if r.category_ok and r.priority_ok else "--"
            print(
                f"{r.id}  {mark}  erwartet {r.expected:28} bekommen {r.got:28}"
                f"{'  SCHWER' if r.severe else ''}"
            )
        drafts = evaluate_drafts(mails, args.model)
        for d in drafts:
            print(
                f"{d.id}  {'ok' if d.assessment_ok else '--'}  Einschätzung erwartet "
                f"{d.expected:8} bekommen {d.got:8}"
                f"{f'  UNBELEGT: {d.unverified_facts}' if d.unverified_facts else ''}"
            )
        summary = {
            "category": f"{sum(r.category_ok for r in results)}/{len(results)}",
            "priority": f"{sum(r.priority_ok for r in results)}/{len(results)}",
            "both": f"{sum(r.category_ok and r.priority_ok for r in results)}/{len(results)}",
            "severe": sum(r.severe for r in results),
            "assessment": f"{sum(d.assessment_ok for d in drafts)}/{len(drafts)}",
            "drafts_with_unverified_facts": sum(
                bool(d.unverified_facts) for d in drafts
            ),
        }
        cost = sum(r.cost_usd for r in results) + sum(d.cost_usd for d in drafts)
        print(f"\n{args.model}, run {number}: {summary}, {cost:.6f} USD\n")
        name = args.model.split("/")[-1]
        path = RESULTS_DIR / f"workflow-{today}-{name}-run{number}.json"
        payload = {
            "date": str(today),
            "model": args.model,
            "run": number,
            "summary": summary,
            "cost_usd": cost,
            "results": [r.model_dump() for r in results],
            "drafts": [d.model_dump() for d in drafts],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
