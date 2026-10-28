"""Prüft die Klassifikation des Workflow-Moduls gegen die Musterlösung. Keine Oberfläche.

Mails und Musterlösung stehen in eval/workflow_mails.yaml. Geprüft wird nur Schritt 1, die
Klassifikation: Für die Entwürfe gibt es keine einzig richtige Antwort, die beurteilt im
Modul ein Mensch (und später das Modul „Evaluation“ mit einem Modell als Prüfer).

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
from core.workflow import Mail, classify, load_mails

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


def is_severe(expected_category, expected_priority, category, priority) -> bool:
    missed_customer = expected_category != "spam" and category == "spam"
    missed_urgency = expected_priority == "high" and priority == "low"
    return missed_customer or missed_urgency


def classify_with_retry(mail: Mail, model: str):
    """Wie classify(), wartet aber bei „Too many requests“ und versucht es erneut."""
    for attempt in range(RETRIES):
        try:
            return classify(mail, model)
        except LLMError as err:
            if STATUS_MESSAGES[429] != str(err) or attempt == RETRIES - 1:
                raise
            time.sleep(RETRY_PAUSE)


def evaluate(mails: tuple[Mail, ...], model: str) -> list[WorkflowEvalResult]:
    results = []
    for mail in mails:
        exp = mail.expected
        try:
            c, usage = classify_with_retry(mail, model)
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
        summary = {
            "category": f"{sum(r.category_ok for r in results)}/{len(results)}",
            "priority": f"{sum(r.priority_ok for r in results)}/{len(results)}",
            "both": f"{sum(r.category_ok and r.priority_ok for r in results)}/{len(results)}",
            "severe": sum(r.severe for r in results),
        }
        cost = sum(r.cost_usd for r in results)
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
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
