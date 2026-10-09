"""Misst die Werkzeugwahl mit dem Fragenkatalog in eval/tool_questions.yaml. Keine Oberfläche.

Pro Frage drei Prüfungen:
- tools: Wurden alle erwarteten Werkzeuge aufgerufen? Bei Fragen ohne Werkzeugbedarf: keins?
- arguments: Stehen die erwarteten Texte in den Argumenten (z. B. „hamburg“ bei get_weather)?
- answer: Stehen die erwarteten Begriffe in der Antwort, und fehlen die verbotenen?

Aufruf: python -m core.tool_eval [--model MODEL] [--runs N]
Ergebnis als Tabelle im Terminal und als JSON in eval/results/.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel

from core.llm import LLMError, Usage
from core.tool_use import Step, ToolRun, run
from core.tools import TOOLS

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
QUESTIONS_FILE = EVAL_DIR / "tool_questions.yaml"
RESULTS_DIR = EVAL_DIR / "results"
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"


class ToolQuestion(BaseModel):
    id: str
    question: str
    tools: list[str] | None = None  # None: Werkzeugwahl wird nicht geprüft
    arguments: dict[str, str] = {}
    expect: list[str] = []
    forbid: list[str] = []  # darf nicht in der Antwort stehen, z. B. erfundene Werte
    why: str = ""


class ToolResult(BaseModel):
    id: str
    question: str
    answer: str
    called: list[str]  # Werkzeuge in Aufrufreihenfolge
    tools_ok: bool | None  # None, wenn der Katalog die Werkzeugwahl offen lässt
    arguments_ok: bool | None  # None, wenn der Katalog keine Argumente vorgibt
    answer_ok: bool | None  # None, wenn der Katalog weder expect noch forbid vorgibt
    extra_calls: int  # Aufrufe von Werkzeugen, die nicht erwartet waren
    rounds: int
    cost_usd: float


def load_questions(path: Path = QUESTIONS_FILE) -> list[ToolQuestion]:
    return [ToolQuestion.model_validate(q) for q in yaml.safe_load(path.read_text())]


def contains(text: str, term: str) -> bool:
    """Eine der Alternativen (mit „|“ getrennt) kommt im Text vor, Groß-/Kleinschreibung egal."""
    return any(option.strip().lower() in text.lower() for option in term.split("|"))


def tools_ok(question: ToolQuestion, steps: list[Step]) -> bool | None:
    if question.tools is None:
        return None
    called = {step.tool for step in steps}
    if not question.tools:
        return not called
    return set(question.tools) <= called


def arguments_ok(question: ToolQuestion, steps: list[Step]) -> bool | None:
    if not question.arguments:
        return None
    return all(
        any(
            step.tool == tool
            and contains(json.dumps(step.arguments, ensure_ascii=False), term)
            for step in steps
        )
        for tool, term in question.arguments.items()
    )


def answer_ok(question: ToolQuestion, answer: str) -> bool | None:
    if not question.expect and not question.forbid:
        return None
    return all(contains(answer, term) for term in question.expect) and not any(
        contains(answer, term) for term in question.forbid
    )


def evaluate(
    questions: list[ToolQuestion], model: str
) -> tuple[list[ToolResult], Usage]:
    """Stellt jede Frage einzeln mit allen fünf Werkzeugen, ohne Verlauf."""
    enabled = [tool.name for tool in TOOLS]
    results, total = [], Usage()
    for question in questions:
        try:
            outcome = run(question.question, [], model, enabled)
        except LLMError as err:
            # Zählt als nicht bestanden; die Kosten dieser Frage sind dann unbekannt
            outcome = ToolRun(text="", model=model, usage=Usage(), steps=[], rounds=0)
            answer = f"(error: {err})"
        else:
            answer = outcome.text if not outcome.stopped else "(stopped at round limit)"
        total += outcome.usage
        results.append(
            ToolResult(
                id=question.id,
                question=question.question,
                answer=answer,
                called=[step.tool for step in outcome.steps],
                tools_ok=tools_ok(question, outcome.steps),
                arguments_ok=arguments_ok(question, outcome.steps),
                answer_ok=answer_ok(question, answer),
                extra_calls=sum(
                    s.tool not in (question.tools or []) for s in outcome.steps
                ),
                rounds=outcome.rounds,
                cost_usd=outcome.usage.cost_usd,
            )
        )
    return results, total


def summary(results: list[ToolResult]) -> dict[str, str]:
    counts = {}
    for key in ("tools_ok", "arguments_ok", "answer_ok"):
        values = [getattr(r, key) for r in results if getattr(r, key) is not None]
        counts[key.removesuffix("_ok")] = f"{sum(values)}/{len(values)}"
    # Alle Prüfungen einer Frage bestanden
    passed = sum(
        False not in (r.tools_ok, r.arguments_ok, r.answer_ok) for r in results
    )
    counts["passed"] = f"{passed}/{len(results)}"
    return counts


def mark(value: bool | None) -> str:
    return "-" if value is None else ("ja" if value else "NEIN")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    questions = load_questions()
    today = datetime.now(ZoneInfo("Europe/Berlin")).date()
    RESULTS_DIR.mkdir(exist_ok=True)
    for number in range(1, args.runs + 1):
        results, total = evaluate(questions, args.model)
        for r in results:
            print(
                f"{r.id}  tools {mark(r.tools_ok):4}  args {mark(r.arguments_ok):4}  "
                f"answer {mark(r.answer_ok):4}  {','.join(r.called) or '-':40}  "
                f"{' '.join(r.answer.split())[:70]}"
            )
        counts = summary(results)
        print(f"\n{args.model}, run {number}: {counts}, {total.cost_usd:.6f} USD\n")
        name = args.model.split("/")[-1]
        path = RESULTS_DIR / f"tools-{today}-{name}-run{number}.json"
        payload = {
            "date": str(today),
            "model": args.model,
            "run": number,
            "summary": counts,
            "cost_usd": total.cost_usd,
            "results": [r.model_dump() for r in results],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
