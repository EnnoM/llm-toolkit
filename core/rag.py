"""RAG-Chatbot: PDFs einlesen, in Abschnitte zerlegen und nach Bedeutung durchsuchen.

Keine Oberfläche. Ablauf:
1. Indexieren: Text pro Seite aus dem PDF holen, Störendes entfernen (Kopfzeilen,
   Silbentrennung, Seitenzahlen), in kurze, überlappende Abschnitte (Chunks) zerlegen und jeden
   Abschnitt als Embedding in Chroma ablegen. Jeder Abschnitt merkt sich Datei und Seiten,
   damit Antworten später eine Quelle nennen können.
2. Abfragen: Die Frage ebenfalls in ein Embedding umrechnen, die ähnlichsten Abschnitte
   holen und das Modell nur aus diesen Abschnitten antworten lassen, mit Datei und Seite als
   Quelle.
"""

import hashlib
import io
import re
import uuid
from bisect import bisect_left, bisect_right
from collections import Counter
from functools import cache
from pathlib import Path

import chromadb
from pydantic import BaseModel, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from core import llm
from core.chat import ChatTurn
from core.llm import Message, Usage

SAMPLES_DIR = Path(__file__).parent.parent / "data" / "samples"

# Startwerte für die Regler im Modul, in Zeichen. Üblich sind 500 bis 1000 Zeichen mit 10 bis
# 20 % Überlappung; die Seite macht das bewusst einstellbar, damit man den Effekt sieht.
DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 150

# Zwischen zwei Seiten im zusammengesetzten Text. Absatzgrenze, damit ein Abschnitt dort
# bevorzugt endet.
PAGE_BREAK = "\n\n"

# Embedding-Modell über OpenRouter: 1536 Zahlen pro Text, 0.02 USD pro Million Tokens
# (gemessen am 23.09. und 05.10.2026). Ähnlichkeitswerte sind nur innerhalb eines Modells
# vergleichbar, deshalb fest und nicht wählbar.
EMBEDDING_MODEL = "openai/text-embedding-3-small"
# Treffer pro Frage. Mit dem Fragenkatalog (eval/rag_questions.yaml) am 06.10.2026 gemessen:
# 4 Treffer 8 von 10 richtig, 6 Treffer 10 von 10, bei rund 50 % mehr Eingabetokens.
DEFAULT_TOP_K = 6

# Satz für Fragen, die die Abschnitte nicht beantworten. Fest vorgegeben, damit man ihn in der
# Oberfläche und in Tests wiedererkennt.
NOT_COVERED = "The documents do not cover this."

RAG_SYSTEM_PROMPT = f"""You answer questions about the user's documents.

Rules:
- Use only the passages in the user's message. Do not use your own knowledge, even if you \
know the answer.
- After every statement, name its source in brackets exactly as given in the passage \
heading, for example (LEP_Bayern_2013.pdf, p. 24). Never leave a statement without a source.
- If the passages do not contain the answer, reply only with this exact English sentence, \
whatever the language of the question: {NOT_COVERED}
- If the passages answer only part of the question, answer that part and say what is missing.
- Otherwise answer in the language of the question. Keep it short: plain sentences or a \
short list, no headings."""

# Wie viele frühere Nachrichten mitgehen (Fragen und Antworten ohne ihre Abschnitte), damit
# Rückfragen wie „And in 2018?“ verständlich bleiben, ohne dass die Kosten mitwachsen.
HISTORY_MESSAGES = 6


class RagError(RuntimeError):
    """Fehler mit einer Meldung, die man in der Oberfläche direkt anzeigen kann."""


class Page(BaseModel):
    """Text einer PDF-Seite. number ist die PDF-Seite ab 1, nicht die gedruckte Seitenzahl."""

    number: int
    text: str


class Document(BaseModel):
    """Ein eingelesenes PDF."""

    filename: str
    file_hash: str  # erkennt dieselbe Datei wieder, auch unter anderem Namen
    pages: list[Page]

    @property
    def characters(self) -> int:
        return sum(len(page.text) for page in self.pages)


class Chunk(BaseModel):
    """Ein Abschnitt mit Herkunft. Spannt er über eine Seitengrenze, ist page_end größer."""

    id: str
    filename: str
    page_start: int
    page_end: int
    text: str
    # Überschriften über dem Abschnitt, z. B. „2. Oberzentren › 2.1 Regierungsbezirk
    # Oberbayern“. Leer, wenn das Dokument keine erkennbaren Überschriften hat.
    section: str = ""

    @property
    def pages_label(self) -> str:
        """Seitenangabe für Quellen: 'p. 3' oder 'p. 3–4'."""
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"p. {self.page_start}–{self.page_end}"

    @property
    def embedding_text(self) -> str:
        """Was in das Embedding eingeht: Dokument und Überschriften vor dem Text.

        Ohne diesen Kopf weiß ein Abschnitt mitten in einer Liste nicht, wovon er handelt
        („Rosenheim, Traunstein, …“ ohne „Oberzentren“). Der Dokumenttitel steht in der Regel
        nur noch im Dateinamen, weil clean_pages die Kopfzeilen entfernt.
        """
        header = " › ".join(
            part
            for part in (Path(self.filename).stem.replace("_", " "), self.section)
            if part
        )
        return f"{header}\n\n{self.text}"


# ------------------------------------------------------------------ Einlesen und Säubern


def read_pdf(data: bytes, filename: str) -> Document:
    """Liest den Text jeder Seite. Gescannte PDFs ohne Textebene liefern leere Seiten."""
    try:
        reader = PdfReader(io.BytesIO(data))
        raw = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, ValueError) as err:
        raise RagError(f"{filename} could not be read as a PDF.") from err
    pages = [
        Page(number=number, text=text)
        for number, text in enumerate(clean_pages(raw), start=1)
    ]
    if not any(page.text for page in pages):
        raise RagError(
            f"{filename} contains no text. It is probably a scan, which would need OCR."
        )
    return Document(
        filename=filename, file_hash=hashlib.sha256(data).hexdigest(), pages=pages
    )


def clean_pages(pages: list[str]) -> list[str]:
    """Entfernt, was die Suche stört, und lässt den Inhalt sonst unverändert.

    - Kopf- und Fußzeilen: Zeilen, die auf mindestens der Hälfte der Seiten stehen
      (im LEP steht der Titel auf 83 von 86 Seiten und würde jeden Abschnitt verrauschen)
    - Zeilen, die nur aus einer Zahl bestehen (Seitenzahlen)
    - Silbentrennung am Zeilenende: „Bewäs-\\nserung“ wird zu „Bewässerung“
    - Punktreihen im Inhaltsverzeichnis und mehrfache Leerzeichen
    """
    repeated = repeated_lines(pages)
    cleaned = []
    for text in pages:
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() not in repeated and not line.strip().isdigit()
        ]
        text = "\n".join(line for line in lines if line)
        # Nur vor Kleinbuchstaben zusammenziehen: „Obst- und“ oder „B-Plan“ bleiben erhalten
        text = re.sub(r"(\w)-\n(?=[a-zäöüß])", r"\1", text)
        text = re.sub(r"\.{4,}", " … ", text)
        text = re.sub(r"[ \t]+", " ", text)
        cleaned.append(text.strip())
    return cleaned


def repeated_lines(pages: list[str]) -> set[str]:
    """Zeilen, die auf mindestens der Hälfte der Seiten vorkommen (Kopf- und Fußzeilen).

    Erst ab drei Seiten, sonst wäre bei zwei Seiten jede gemeinsame Zeile verdächtig.
    """
    if len(pages) < 3:
        return set()
    counts = Counter(
        line.strip()
        for text in pages
        for line in set(text.splitlines())
        if line.strip()
    )
    return {line for line, count in counts.items() if count >= len(pages) / 2}


# ------------------------------------------------------------------ Überschriften


class Heading(BaseModel):
    """Eine nummerierte Überschrift im zusammengesetzten Text."""

    position: int  # Beginn der Zeile
    numbers: tuple[int, ...]  # „2.1 Regierungsbezirk Oberbayern“ ergibt (2, 1)
    title: str  # die ganze Zeile
    path: str  # diese Überschrift mit allen darüber, getrennt durch „ › “


# Zeile aus Nummer und Titel mit Großbuchstaben: „2. Oberzentren“, „2.1.3 Vorzug der …“.
# Aufzählungen im Fließtext („1. eine nach Art. 23 …“) beginnen klein und fallen heraus.
HEADING_LINE = re.compile(r"^(\d+(?:\.\d+)*)\.?[ \t]+([A-ZÄÖÜ].*)$", re.MULTILINE)
# Längere Zeilen sind eher ein nummerierter Satz (längste echte Überschrift in den
# Beispielen: 112 Zeichen)
HEADING_MAX_LENGTH = 120
# Rückverweis am Zeilenanfang: „Zu 2.1.5 (B) …“ oder „Zu 2.1 Das Zentrale-Orte-System …“
REFERENCE_LINE = re.compile(r"^Zu (\d+(?:\.\d+)*)\b", re.MULTILINE)
# So viele Nummern darf eine Überschrift überspringen, falls pypdf eine nicht erkannt hat
HEADING_MAX_GAP = 3


def find_headings(text: str) -> list[Heading]:
    """Nummerierte Überschriften in Lesereihenfolge, jede mit ihrem Pfad.

    Eine Zeile zählt nur, wenn ihre Nummer zur vorigen Überschrift passt: nächste auf
    gleicher oder höherer Ebene (2.1 nach 1.7) oder erste darunter (2.1 nach 2). Das sortiert
    Datumszeilen wie „25. Juni 2012“ aus. Eine 1 auf oberster Ebene beginnt immer neu, weil
    Inhaltsverzeichnis, Anhänge und Hauptteil jeweils wieder bei 1 anfangen.
    Inhaltsverzeichnisse fallen außerdem an „…“ heraus (aus den Punktreihen, clean_pages).

    Rückverweise wie „Zu 2.1.5 (B) Die Mittel- und …“ (Begründung im LEP) hängen sich an die
    schon bekannte Überschrift 2.1.5. Ohne sie bekäme die ganze Begründung, die nach den
    Zielen folgt, die letzte Überschrift davor.
    """
    headings: list[Heading] = []
    path: list[Heading] = []
    for match in HEADING_LINE.finditer(text):
        line = match.group(0).strip()
        numbers = tuple(int(n) for n in match.group(1).split("."))
        if "…" in line or len(line) > HEADING_MAX_LENGTH:
            continue
        current = path[-1].numbers if path else ()
        if not follows(current, numbers):
            continue
        path = [h for h in path if len(h.numbers) < len(numbers)]
        heading = Heading(
            position=match.start(),
            numbers=numbers,
            title=line,
            path=" › ".join([h.title for h in path] + [line]),
        )
        path.append(heading)
        headings.append(heading)
    for match in REFERENCE_LINE.finditer(text):
        numbers = tuple(int(n) for n in match.group(1).split("."))
        # Nur Verweise auf eine Überschrift, die vorher im Text steht
        earlier = [h for h in headings if h.position < match.start()]
        target = next((h for h in reversed(earlier) if h.numbers == numbers), None)
        if target:
            title = f"Zu {match.group(1)}"
            headings.append(
                Heading(
                    position=match.start(),
                    numbers=numbers,
                    title=title,
                    path=f"{target.path} › {title}",
                )
            )
    return sorted(headings, key=lambda h: h.position)


def follows(current: tuple[int, ...], numbers: tuple[int, ...]) -> bool:
    """Ob numbers als nächste Überschrift nach current plausibel ist."""
    if numbers == (1,):
        return True
    if not current:
        return len(numbers) == 1 and numbers[0] <= HEADING_MAX_GAP
    if numbers == current + (1,):  # erste Unterüberschrift
        return True
    level = len(numbers) - 1
    return (
        len(numbers) <= len(current)
        and numbers[:level] == current[:level]
        and 0 < numbers[level] - current[level] <= HEADING_MAX_GAP
    )


def section_at(headings: list[Heading], position: int) -> str:
    """Pfad der letzten Überschrift, die an position oder davor beginnt."""
    index = bisect_right([h.position for h in headings], position) - 1
    return headings[index].path if index >= 0 else ""


# ------------------------------------------------------------------ Zerlegen


# Satzanfang: Satzzeichen, Leerraum, dann ein Großbuchstabe, eine Klammer oder ein
# Anführungszeichen. Nicht nach Abkürzungen aus einzelnen Buchstaben wie „z.B.“ oder „u.a.“
# (vor dem Punkt steht ein Buchstabe, davor Leerraum, Punkt oder Klammer).
SENTENCE_START = re.compile(r"(?<![\s.(]\w)[.!?]\s+(?=[A-ZÄÖÜ(„\"])")
# Zeilen, die einen neuen Gedanken beginnen: Aufzählungspunkt oder Kennung wie „(Z)“, „(G)“
ITEM_START = re.compile(r"\n(?=[-–•]\s|\([A-Z]\)\s)")


def chunk_document(
    document: Document,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    sentences: bool = True,
) -> list[Chunk]:
    """Zerlegt das ganze Dokument in Abschnitte von höchstens size Zeichen.

    Die Seiten werden zu einem Text verbunden, damit ein Gedanke über eine Seitengrenze hinweg
    zusammenbleiben kann. Wo jede Seite in diesem Text beginnt, steht in page_offsets; daraus
    ergeben sich Start- und Endseite jedes Abschnitts. Aufeinanderfolgende Abschnitte teilen
    sich etwa overlap Zeichen, damit ein Satz an der Grenze in beiden vollständig vorkommt.
    Jeder Abschnitt bekommt die Überschriften, unter denen er beginnt.

    sentences=False schneidet nur an Wortgrenzen (zum Vergleich in der Oberfläche).
    """
    if not 0 <= overlap < size:
        raise ValueError("overlap must be at least 0 and smaller than size")
    text, page_offsets = join_pages(document.pages)
    headings = find_headings(text)
    chunks = []
    settings = settings_label(size, overlap, sentences)
    spans = split_spans(text, size, overlap, [h.position for h in headings], sentences)
    for start, end in spans:
        chunks.append(
            Chunk(
                # Gleiche Datei und gleiche Einstellungen ergeben dieselbe id (keine Dubletten)
                id=f"{document.file_hash[:16]}-{settings}-{len(chunks)}",
                filename=document.filename,
                page_start=page_at(page_offsets, start, document.pages),
                page_end=page_at(page_offsets, end - 1, document.pages),
                text=text[start:end].strip(),
                section=section_at(headings, start),
            )
        )
    return chunks


def settings_label(size: int, overlap: int, sentences: bool) -> str:
    """Die Einstellungen, die die Abschnitte bestimmen, z. B. '800-150-s' (s = Sätze,
    w = nur Wortgrenzen). Steht in der id jedes Abschnitts und trennt sie bei der Suche."""
    return f"{size}-{overlap}-{'s' if sentences else 'w'}"


def join_pages(pages: list[Page]) -> tuple[str, list[int]]:
    """Alle Seiten als ein Text, dazu die Position, an der jede Seite beginnt."""
    parts, offsets, position = [], [], 0
    for page in pages:
        offsets.append(position)
        parts.append(page.text)
        position += len(page.text) + len(PAGE_BREAK)
    return PAGE_BREAK.join(parts), offsets


def page_at(page_offsets: list[int], position: int, pages: list[Page]) -> int:
    """PDF-Seite, auf der das Zeichen an position steht."""
    return pages[bisect_right(page_offsets, position) - 1].number


def split_spans(
    text: str,
    size: int,
    overlap: int,
    headings: list[int] = (),
    sentences: bool = True,
) -> list[tuple[int, int]]:
    """Start und Ende jedes Abschnitts im Text; headings sind die Anfänge der Überschriften.

    Ein Abschnitt endet möglichst an einer natürlichen Grenze im letzten Viertel des Fensters:
    zuerst vor einer Überschrift, dann Absatz, Satzende, Leerzeichen. Nur wenn es keine gibt,
    wird hart geschnitten.

    Der nächste Abschnitt beginnt an dem Satzanfang, der overlap Zeichen vor dem Ende am
    nächsten liegt, höchstens doppelt so weit zurück. Die Überlappung ist also ungefähr so
    groß wie eingestellt. Gibt es dort keinen Satzanfang (sehr lange Sätze, Listen ohne
    Punkt), beginnt er am Wortanfang overlap Zeichen vor dem Ende.

    sentences=False: Ende an der letzten Wortgrenze, Anfang am Wortanfang overlap Zeichen vor
    dem Ende. Überschriften, Absätze und Sätze spielen dann keine Rolle.
    """
    if not sentences:
        headings = []
    starts = sentence_starts(text, headings) if sentences else []
    spans = []
    start = skip_space(text, 0)
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            end = natural_end(text, start + size * 3 // 4, end, headings, sentences)
        spans.append((start, end))
        if end >= len(text):
            break
        start = skip_space(text, next_start(text, start, end, overlap, starts))
    return spans


def sentence_starts(text: str, headings: list[int]) -> list[int]:
    """Alle Stellen, an denen ein Abschnitt gut beginnen kann, aufsteigend."""
    found = set(headings)
    found.update(match.end() for match in SENTENCE_START.finditer(text))
    found.update(match.end() for match in ITEM_START.finditer(text))
    found.update(match.end() for match in re.finditer(r"\n\n", text))
    return sorted(found)


def next_start(text: str, start: int, end: int, overlap: int, starts: list[int]) -> int:
    """Beginn des Abschnitts nach (start, end), siehe split_spans."""
    target = end - overlap
    window = starts[bisect_left(starts, end - 2 * overlap) : bisect_right(starts, end)]
    candidates = [s for s in window if s > start]
    if candidates:
        return min(candidates, key=lambda s: (abs(s - target), s))
    # Kein Satzanfang in Reichweite: overlap Zeichen vor dem Ende. Liegt das mitten in einem
    # Wort, geht es zurück an dessen Anfang; an einem Wortende bleibt es stehen (sonst
    # stünde bei Überlappung 0 das letzte Wort doppelt in beiden Abschnitten)
    position = max(target, start + 1)
    while (
        position > start + 1
        and not text[position - 1].isspace()
        and not text[position].isspace()
    ):
        position -= 1
    return position


def natural_end(
    text: str,
    earliest: int,
    latest: int,
    headings: list[int] = (),
    sentences: bool = True,
) -> int:
    """Beste Schnittstelle zwischen earliest und latest (Ende exklusiv).

    Stufen von gut nach notdürftig; innerhalb einer Stufe gilt die späteste Stelle, damit der
    Abschnitt möglichst lang wird. Vor einer Überschrift zu schneiden, hat Vorrang: Sonst
    steht sie am Ende eines Abschnitts, zu dem ihr Inhalt nicht mehr gehört.
    """
    before_heading = [h for h in headings if earliest < h <= latest]
    if before_heading:
        return before_heading[-1]
    window = text[earliest:latest]
    levels = [["\n\n"], [". ", ".\n"], ["\n"], [" "]] if sentences else [[" ", "\n"]]
    for separators in levels:
        cut = max(window.rfind(separator) for separator in separators)
        if cut != -1:
            # Der Satzpunkt gehört noch zum Abschnitt, der Leerraum danach nicht
            return earliest + cut + (1 if separators[0].startswith(".") else 0)
    return latest


def skip_space(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position


# ------------------------------------------------------------------ Beispiel-PDFs


def sample_files() -> list[Path]:
    """Die mitgelieferten Beispiel-PDFs, nach Namen sortiert."""
    return sorted(SAMPLES_DIR.glob("*.pdf"))


# ------------------------------------------------------------------ Embeddings und Suche


class Hit(BaseModel):
    """Ein gefundener Abschnitt mit seiner Ähnlichkeit zur Frage.

    similarity ist die Kosinus-Ähnlichkeit: 1 = gleiche Richtung, um 0 = nichts gemeinsam.
    Kein Prozentwert und nur innerhalb eines Embedding-Modells vergleichbar.
    """

    chunk: Chunk
    similarity: float


@cache
def chroma_client() -> chromadb.ClientAPI:
    """Chroma nur im Arbeitsspeicher, ohne Telemetrie.

    Achtung: Alle EphemeralClients eines Prozesses teilen sich denselben Speicher, also auch
    alle Besucher der App. Getrennt wird über den Namen der Sammlung (siehe SearchIndex).
    """
    return chromadb.EphemeralClient(
        settings=chromadb.Settings(anonymized_telemetry=False)
    )


def settings_key(chunk: Chunk) -> str:
    """Einstellungen aus der id, z. B. '800-150-s' (siehe settings_label)."""
    return "-".join(chunk.id.split("-")[1:4])


class SearchIndex:
    """Die Embeddings einer Sitzung in einer eigenen Chroma-Sammlung.

    Abschnitte verschiedener Einstellungen liegen nebeneinander (ein Wechsel zurück kostet
    nichts) und werden bei der Suche über settings_key getrennt.
    """

    def __init__(self, name: str | None = None) -> None:
        # Zufälliger Name: andere Sitzungen kennen ihn nicht und sehen die Uploads nicht
        self.name = name or f"rag-{uuid.uuid4().hex}"
        self.collection = chroma_client().get_or_create_collection(
            self.name,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,  # Embeddings rechnen wir selbst über OpenRouter
        )

    def missing(self, chunks: list[Chunk]) -> list[Chunk]:
        """Abschnitte, die noch kein Embedding haben."""
        if not chunks:
            return []
        stored = set(self.collection.get(ids=[c.id for c in chunks], include=[])["ids"])
        return [c for c in chunks if c.id not in stored]

    def add(self, chunks: list[Chunk]) -> Usage:
        """Legt fehlende Abschnitte ab. Schon vorhandene kosten nichts und doppeln nicht."""
        chunks = self.missing(chunks)
        if not chunks:
            return Usage()
        response = llm.embed([c.embedding_text for c in chunks], EMBEDDING_MODEL)
        self.collection.add(
            ids=[c.id for c in chunks],
            embeddings=response.vectors,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "filename": c.filename,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "section": c.section,
                    "settings": settings_key(c),
                }
                for c in chunks
            ],
        )
        return response.usage

    def search(
        self,
        question: str,
        size: int,
        overlap: int,
        top_k: int = DEFAULT_TOP_K,
        sentences: bool = True,
    ) -> tuple[list[Hit], Usage]:
        """Die top_k Abschnitte, die der Frage am ähnlichsten sind, bester zuerst."""
        response = llm.embed([question], EMBEDDING_MODEL)
        result = self.collection.query(
            query_embeddings=response.vectors,
            n_results=top_k,
            where={"settings": settings_label(size, overlap, sentences)},
        )
        hits = [
            Hit(
                chunk=Chunk(
                    id=chunk_id,
                    text=text,
                    **{
                        key: meta[key]
                        for key in ("filename", "page_start", "page_end", "section")
                    },
                ),
                # Chroma liefert die Kosinus-Distanz, Ähnlichkeit ist 1 minus Distanz
                similarity=1 - distance,
            )
            for chunk_id, text, meta, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]
        return hits, response.usage

    def delete(self) -> None:
        chroma_client().delete_collection(self.name)


# ------------------------------------------------------------------ Antworten


class RagTurn(ChatTurn):
    """Eine Nachricht im Verlauf. Antworten tragen zusätzlich die genutzten Abschnitte."""

    hits: list[Hit] = Field(default_factory=list)


def format_passages(hits: list[Hit]) -> str:
    """Die Abschnitte für das Modell, jeweils mit der Quelle als Überschrift.

    Die Überschriften des Dokuments stehen in einer eigenen Zeile, nicht in der Quelle: Das
    Modell soll sie zum Verstehen nutzen, aber nur „Datei, Seite“ zitieren.
    """
    return "\n\n".join(
        f"[{number}] {hit.chunk.filename}, {hit.chunk.pages_label}\n"
        + (f"Section: {hit.chunk.section}\n" if hit.chunk.section else "")
        + hit.chunk.text
        for number, hit in enumerate(hits, start=1)
    )


def build_rag_messages(
    question: str, hits: list[Hit], history: list[RagTurn]
) -> list[Message]:
    """System-Prompt, die letzten Nachrichten und die neue Frage samt Abschnitten.

    Die Abschnitte stehen nur bei der neuen Frage. Frühere Abschnitte gehen nicht mit: Sie
    würden die Anfrage mit jeder Runde teurer machen und alte Treffer mit neuen vermischen.
    """
    earlier = [
        {"role": turn.role, "content": turn.content}
        for turn in history[-HISTORY_MESSAGES:]
    ]
    user = f"Passages:\n\n{format_passages(hits)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        *earlier,
        {"role": "user", "content": user},
    ]
