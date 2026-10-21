"""Prüft den KI-Datenanalysten gegen von Hand geschriebenes SQL. Keine Oberfläche.

Fragen und Referenz-SQL stehen in eval/analyst_questions.yaml. Verglichen werden die
Ergebnisse, nicht das SQL: Dieselbe Antwort lässt sich auf viele Arten schreiben.

Aufruf: python -m core.analyst_eval [--model MODEL] [--runs N]
Ergebnis als Tabelle im Terminal und als JSON in eval/results/.
"""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml
from pydantic import BaseModel

from core.analyst import ask, run_query
from core.llm import LLMError

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
QUESTIONS_FILE = EVAL_DIR / "analyst_questions.yaml"
RESULTS_DIR = EVAL_DIR / "results"
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"


class AnalystQuestion(BaseModel):
    id: str
    question: str
    reference_sql: str
    # Nur diese Referenzspalten vergleichen; leer = alle. Für Spalten, deren Darstellung frei
    # ist (Monat als 1, „2025-01“ oder Datum), nicht aber deren Inhalt.
    compare: list[str] = []
    why: str = ""


class AnalystEvalResult(BaseModel):
    id: str
    question: str
    sql: str
    passed: bool
    corrections: int
    error: str | None
    cost_usd: float


def load_questions(path: Path = QUESTIONS_FILE) -> list[AnalystQuestion]:
    return [AnalystQuestion.model_validate(q) for q in yaml.safe_load(path.read_text())]


def column_values(series: pd.Series) -> Counter:
    """Werte einer Spalte als Multimenge; Zahlen auf 2 Stellen gerundet, alles als Text."""

    def norm(value) -> str:
        # Zahlen auch als Text erkennen: strftime('%Y') liefert „2023“ statt 2023
        try:
            return f"{round(float(value), 2):.2f}"
        except (TypeError, ValueError):
            return str(value).strip()

    return Counter(norm(v) for v in series.tolist())


def same_result(
    reference: pd.DataFrame, answer: pd.DataFrame, compare: list[str] | None = None
) -> bool:
    """Jede Referenzspalte kommt mit denselben Werten auch in der Antwort vor.

    Reihenfolge der Zeilen, Spaltennamen und zusätzliche Spalten spielen keine Rolle; die
    Zeilenzahl muss stimmen. Bewusst großzügig: geprüft wird, ob die Zahlen richtig sind.
    """
    if len(reference) != len(answer):
        return False
    available = [column_values(answer[c]) for c in answer.columns]
    return all(
        column_values(reference[c]) in available for c in compare or reference.columns
    )


def evaluate(questions: list[AnalystQuestion], model: str) -> list[AnalystEvalResult]:
    results = []
    for question in questions:
        reference = run_query(question.reference_sql)
        try:
            outcome = ask(question.question, model)
        except LLMError as err:
            results.append(
                AnalystEvalResult(
                    id=question.id,
                    question=question.question,
                    sql="",
                    passed=False,
                    corrections=0,
                    error=str(err),
                    cost_usd=0.0,
                )
            )
            continue
        results.append(
            AnalystEvalResult(
                id=question.id,
                question=question.question,
                sql=outcome.sql,
                passed=outcome.table is not None
                and same_result(reference, outcome.table, question.compare),
                corrections=len(outcome.attempts) - 1,
                error=outcome.error,
                cost_usd=outcome.usage.cost_usd,
            )
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    questions = load_questions()
    today = datetime.now(ZoneInfo("Europe/Berlin")).date()
    RESULTS_DIR.mkdir(exist_ok=True)
    for number in range(1, args.runs + 1):
        results = evaluate(questions, args.model)
        for r in results:
            print(
                f"{r.id}  {'ja' if r.passed else 'NEIN':4}  korr. {r.corrections}  "
                f"{' '.join(r.sql.split())[:90]}"
            )
        passed = sum(r.passed for r in results)
        cost = sum(r.cost_usd for r in results)
        print(
            f"\n{args.model}, run {number}: {passed}/{len(results)}, {cost:.6f} USD\n"
        )
        name = args.model.split("/")[-1]
        path = RESULTS_DIR / f"analyst-{today}-{name}-run{number}.json"
        payload = {
            "date": str(today),
            "model": args.model,
            "run": number,
            "passed": f"{passed}/{len(results)}",
            "cost_usd": cost,
            "results": [r.model_dump() for r in results],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
