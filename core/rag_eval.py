"""Misst den RAG-Chatbot mit dem Fragenkatalog in eval/rag_questions.yaml. Keine Oberfläche.

Pro Frage drei Prüfungen, alle ohne zweites Modell (das kommt im Modul Evaluation):
- found: Ist ein Abschnitt aus der richtigen Datei und von der richtigen Seite unter den
  Treffern? Prüft nur die Suche.
- correct: Stehen die erwarteten Begriffe in der Antwort? Bei Fragen, die die Dokumente nicht
  beantworten, muss der feste Satz NOT_COVERED kommen.
- cited: Nennt die Antwort die richtige Datei mit einer richtigen Seite?

Aufruf: python -m core.rag_eval [--words] [--model MODEL] [--top-k N]
Ergebnis als Tabelle im Terminal und als JSON in eval/results/.
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel

from core import llm
from core.llm import Usage
from core.rag import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_TOP_K,
    NOT_COVERED,
    Hit,
    SearchIndex,
    build_rag_messages,
    chunk_document,
    read_pdf,
    sample_files,
    settings_label,
)

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
QUESTIONS_FILE = EVAL_DIR / "rag_questions.yaml"
RESULTS_DIR = EVAL_DIR / "results"
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
# Wie im Modul (app_pages/rag.py), damit die Messung zur Oberfläche passt
ANSWER_TEMPERATURE = 0.2


class Question(BaseModel):
    id: str
    question: str
    source: str = ""
    pages: list[int] = []
    expect: list[str] = []
    not_covered: bool = False
    why: str = ""


class Result(BaseModel):
    id: str
    question: str
    answer: str
    found: bool | None  # None bei Fragen ohne Antwort in den Dokumenten
    correct: bool
    cited: bool | None
    hits: list[str]  # „Datei, p. X“ der Treffer, bester zuerst
    cost_usd: float


def load_questions(path: Path = QUESTIONS_FILE) -> list[Question]:
    return [Question.model_validate(item) for item in yaml.safe_load(path.read_text())]


def pages_overlap(start: int, end: int, pages: list[int]) -> bool:
    return any(start <= page <= end for page in pages)


def is_found(question: Question, hits: list[Hit]) -> bool:
    """Ob ein Treffer aus der richtigen Datei die richtige Seite enthält."""
    return any(
        hit.chunk.filename == question.source
        and pages_overlap(hit.chunk.page_start, hit.chunk.page_end, question.pages)
        for hit in hits
    )


def is_correct(question: Question, answer: str) -> bool:
    """Alle erwarteten Begriffe (eine der Alternativen je Begriff) stehen in der Antwort."""
    if question.not_covered:
        return answer.strip() == NOT_COVERED
    text = answer.lower()
    return all(
        any(option.strip().lower() in text for option in term.split("|"))
        for term in question.expect
    )


def is_cited(question: Question, answer: str) -> bool:
    """Ob die Antwort die richtige Datei mit einer passenden Seite als Quelle nennt."""
    pattern = re.escape(question.source) + r", p\. (\d+)(?:–(\d+))?"
    return any(
        pages_overlap(int(start), int(end or start), question.pages)
        for start, end in re.findall(pattern, answer)
    )


def evaluate(
    questions: list[Question],
    model: str = DEFAULT_MODEL,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    top_k: int = DEFAULT_TOP_K,
    sentences: bool = True,
) -> tuple[list[Result], Usage]:
    """Indexiert die Beispiel-PDFs und stellt jede Frage einzeln, ohne Verlauf."""
    index = SearchIndex()
    try:
        documents = [read_pdf(path.read_bytes(), path.name) for path in sample_files()]
        chunks = [
            chunk
            for document in documents
            for chunk in chunk_document(document, size, overlap, sentences)
        ]
        total = index.add(chunks)
        results = []
        for question in questions:
            hits, search_usage = index.search(
                question.question, size, overlap, top_k, sentences
            )
            response = llm.complete(
                build_rag_messages(question.question, hits, []),
                model,
                temperature=ANSWER_TEMPERATURE,
            )
            usage = search_usage + response.usage
            total += usage
            covered = not question.not_covered
            results.append(
                Result(
                    id=question.id,
                    question=question.question,
                    answer=response.text,
                    found=is_found(question, hits) if covered else None,
                    correct=is_correct(question, response.text),
                    cited=is_cited(question, response.text) if covered else None,
                    hits=[f"{h.chunk.filename}, {h.chunk.pages_label}" for h in hits],
                    cost_usd=usage.cost_usd,
                )
            )
    finally:
        index.delete()
    return results, total


def summary(results: list[Result]) -> dict[str, str]:
    """Anteil bestandener Prüfungen, z. B. {'found': '7/8', 'correct': '6/10', …}."""
    counts = {}
    for key in ("found", "correct", "cited"):
        values = [getattr(r, key) for r in results if getattr(r, key) is not None]
        counts[key] = f"{sum(values)}/{len(values)}"
    return counts


def mark(value: bool | None) -> str:
    return "-" if value is None else ("ja" if value else "NEIN")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--words", action="store_true", help="nur an Wortgrenzen schneiden"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    settings = settings_label(DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, not args.words)
    results, total = evaluate(
        load_questions(), args.model, top_k=args.top_k, sentences=not args.words
    )
    for r in results:
        print(
            f"{r.id}  found {mark(r.found):4}  correct {mark(r.correct):4}  "
            f"cited {mark(r.cited):4}  {r.question}"
        )
        print(f"     {' '.join(r.answer.split())[:160]}")
    counts = summary(results)
    print(
        f"\n{settings}, top {args.top_k}, {args.model}: {counts}, "
        f"{total.cost_usd:.6f} USD"
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    today = datetime.now(ZoneInfo("Europe/Berlin")).date()
    model_name = args.model.split("/")[-1]
    path = RESULTS_DIR / f"rag-{today}-{settings}-top{args.top_k}-{model_name}.json"
    payload = {
        "date": str(today),
        "settings": settings,
        "top_k": args.top_k,
        "model": args.model,
        "summary": counts,
        "cost_usd": total.cost_usd,
        "results": [r.model_dump() for r in results],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Gespeichert: {path.relative_to(EVAL_DIR.parent)}")


if __name__ == "__main__":
    main()
