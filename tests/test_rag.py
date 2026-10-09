"""RAG-Chatbot (core/rag.py, app_pages/rag.py): Einlesen, Zerlegen, Suche, Antworten. Keine API."""

from itertools import pairwise
from types import SimpleNamespace

import pytest
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletionChunk
from streamlit.testing.v1 import AppTest

import core.llm
from core.llm import EMBEDDING_BATCH_SIZE, LLMError, embed
from core.rag import (
    HISTORY_MESSAGES,
    NOT_COVERED,
    SAMPLES_DIR,
    Chunk,
    Document,
    Hit,
    Page,
    RagError,
    RagTurn,
    SearchIndex,
    build_rag_messages,
    chunk_document,
    clean_pages,
    find_headings,
    join_pages,
    read_pdf,
    sample_files,
    sentence_starts,
    split_spans,
)
from core.rag_eval import Question, is_cited, is_correct, is_found, load_questions


def document(*pages: str, name: str = "test.pdf") -> Document:
    return Document(
        filename=name,
        file_hash="0" * 64,
        pages=[Page(number=n, text=t) for n, t in enumerate(pages, start=1)],
    )


# Fließtext aus nummerierten Wörtern: jede Stelle ist eindeutig, Leerzeichen dazwischen
WORDS = " ".join(f"word{n:04d}" for n in range(1000))


# ------------------------------------------------------------------ Zerlegen


@pytest.mark.parametrize(("size", "overlap"), [(800, 150), (300, 0), (500, 100)])
def test_chunks_respect_size_and_overlap(size, overlap):
    spans = split_spans(WORDS, size, overlap)
    for start, end in spans:
        assert end - start <= size
    for (start, end), (next_start, _) in pairwise(spans):
        # Überlappung mindestens wie eingestellt (minus das Leerzeichen an der Wortgrenze)
        assert end - next_start >= overlap - 1
        # und nie so groß, dass der nächste Abschnitt nicht vorankommt
        assert next_start > start
    # Nichts geht verloren: erster Abschnitt am Anfang, letzter am Ende
    assert spans[0][0] == 0
    assert spans[-1][1] == len(WORDS)


def test_chunks_end_at_word_boundaries():
    for start, end in split_spans(WORDS, 500, 100):
        assert WORDS[start:end].split()[0].startswith("word")
        assert len(WORDS[start:end].split()[-1]) == len("word0000")


def test_without_overlap_no_text_is_repeated():
    text = "apple apple pear.\n\ncat dog dog.\n\napple pear pear."
    chunks = chunk_document(document(text), size=20, overlap=0)
    assert " ".join(" ".join(c.text.split()) for c in chunks) == " ".join(text.split())


def test_chunks_prefer_sentence_ends():
    text = ("This is one sentence. " * 60).strip()
    for start, end in split_spans(text, 300, 50)[:-1]:
        assert text[start:end].endswith(".")


def test_chunk_across_page_break_names_both_pages():
    chunks = chunk_document(document("a " * 300, "b " * 300), size=400, overlap=0)
    labels = [c.pages_label for c in chunks]
    assert labels[0] == "p. 1"
    assert "p. 1–2" in labels
    assert labels[-1] == "p. 2"


def test_page_offsets_point_to_the_right_page():
    text, offsets = join_pages(document("first", "second").pages)
    assert text[offsets[1] :].startswith("second")


def test_same_file_and_settings_give_the_same_ids():
    doc = document(WORDS)
    assert [c.id for c in chunk_document(doc)] == [c.id for c in chunk_document(doc)]
    other = chunk_document(doc, size=500, overlap=50)
    assert chunk_document(doc)[0].id != other[0].id


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_document(document(WORDS), size=200, overlap=200)


def test_next_chunk_starts_at_a_sentence():
    text = " ".join(f"Sentence number {n} ends here." for n in range(60))
    for start, _ in split_spans(text, 300, 60):
        assert text[start:].startswith("Sentence")


def test_abbreviations_do_not_count_as_sentence_starts():
    text = "Schulen, z.B. Gymnasien, u.a. Realschulen. Danach folgt Text.\n- Punkt"
    found = {text[position:].split()[0] for position in sentence_starts(text, [])}
    assert found == {"Danach", "-"}


def test_chunk_ends_before_a_heading():
    text = "Intro sentence here. " * 9 + "\n2.1 Next Section\nBody text follows. " * 3
    spans = split_spans(text, 220, 0, [h.position for h in find_headings(text)])
    assert text[spans[1][0] :].startswith("2.1 Next Section")


def test_without_sentences_chunks_are_cut_at_words_only():
    text = "Short sentence. " * 40 + "\n2.1 Heading\n" + "More text here. " * 40
    headings = [h.position for h in find_headings(text)]
    by_words = split_spans(text, 200, 0, headings, sentences=False)
    # Ende an der letzten Wortgrenze im Fenster: höchstens ein Wort (9 Zeichen) kürzer
    assert all(end - start >= 190 for start, end in by_words[:-1])
    # Mit Sätzen endet jeder Abschnitt nach einem Satzpunkt oder vor der Überschrift
    for start, end in split_spans(text, 200, 0, headings)[:-1]:
        assert text[start:end].endswith(".") or text[end:].startswith("2.1 Heading")


def test_sentence_setting_gives_separate_chunk_ids():
    doc = document(WORDS)
    ids_sentences = {c.id for c in chunk_document(doc, 300, 50)}
    ids_words = {c.id for c in chunk_document(doc, 300, 50, sentences=False)}
    assert not ids_sentences & ids_words


# ------------------------------------------------------------------ Überschriften


def test_headings_form_a_path_and_skip_toc_and_dates():
    lines = [
        "Inhalt",
        "2 Raumstruktur … 20",  # Inhaltsverzeichnis
        "1. September 2013",  # Datum, beginnt aber bei 1: zählt, bis der Hauptteil kommt
        "1 Grundlagen",
        "Text.",
        "25. Juni 2012 (GVBl S. 254)",  # passt nicht nach 1
        "1.1 Gleichwertigkeit",
        "1. eine Aufzählung im Fließtext",  # klein: keine Überschrift
        "2 Raumstruktur",
        "2.1 Zentrale Orte",
    ]
    paths = [h.path for h in find_headings("\n".join(lines))]
    assert paths == [
        "1. September 2013",
        "1 Grundlagen",
        "1 Grundlagen › 1.1 Gleichwertigkeit",
        "2 Raumstruktur",
        "2 Raumstruktur › 2.1 Zentrale Orte",
    ]


def test_reference_lines_attach_to_their_heading():
    text = "2 Raumstruktur\n2.1 Zentrale Orte\n(Z) Ziel.\n2.2 Gebiete\n(Z) Ziel.\n"
    text += "Zu 2.1 (B) Begründung zu den Zentralen Orten.\n"
    reference = find_headings(text)[-1]
    assert reference.path == "2 Raumstruktur › 2.1 Zentrale Orte › Zu 2.1"


def test_chunks_carry_their_section_into_the_embedding():
    doc = document(
        "1. Mittelzentren\nAmberg, Bamberg.\n2. Oberzentren\nRosenheim, Passau."
    )
    chunks = chunk_document(doc, size=40, overlap=0)
    rosenheim = next(c for c in chunks if "Rosenheim" in c.text)
    assert rosenheim.section == "2. Oberzentren"
    assert rosenheim.embedding_text.startswith("test › 2. Oberzentren\n\n")
    assert chunks[0].section == "1. Mittelzentren"


def test_sample_list_of_regional_centres_knows_its_heading():
    path = SAMPLES_DIR / "LEP_Anhang_1_Zentrale_Orte_2018.pdf"
    chunks = chunk_document(read_pdf(path.read_bytes(), path.name))
    # Oberbayern steht zweimal: unter Mittelzentren und unter Oberzentren
    sections = [c.section for c in chunks if "Traunstein" in c.text]
    assert any(s.startswith("2. Oberzentren") for s in sections)


# ------------------------------------------------------------------ Säubern


def test_header_on_most_pages_is_removed():
    pages = [f"Landesentwicklungsprogramm Bayern\nContent {n}" for n in range(4)]
    assert clean_pages(pages) == [f"Content {n}" for n in range(4)]


def test_page_number_lines_are_removed():
    assert clean_pages(["3\nText on page three"]) == ["Text on page three"]


def test_hyphenation_is_joined_but_real_hyphens_stay():
    [text] = clean_pages(
        ["günstigen Bewäs-\nserungsmöglichkeiten, Obst- und\nGartenbau"]
    )
    assert "Bewässerungsmöglichkeiten" in text
    assert "Obst- und" in text


def test_table_of_contents_dots_are_shortened():
    [text] = clean_pages(["Leitbild ..................... 3"])
    assert text == "Leitbild … 3"


# ------------------------------------------------------------------ Einlesen


def test_sample_pdfs_are_readable_with_text_on_every_page():
    files = sample_files()
    assert len(files) == 3
    for path in files:
        doc = read_pdf(path.read_bytes(), path.name)
        assert all(page.text for page in doc.pages), path.name


def test_sample_header_is_gone():
    path = SAMPLES_DIR / "LEP_Anhang_1_Zentrale_Orte_2018.pdf"
    doc = read_pdf(path.read_bytes(), path.name)
    assert len(doc.pages) == 4
    assert all("Anhang 1 - Zentrale Orte" not in page.text for page in doc.pages)


def test_not_a_pdf_gives_a_clear_error():
    with pytest.raises(RagError, match="could not be read as a PDF"):
        read_pdf(b"hello", "notes.pdf")


# ------------------------------------------------------------------ Embeddings und Suche

# Fake-Embeddings: zählt, wie oft jedes dieser Wörter vorkommt. Texte mit denselben Wörtern
# zeigen in dieselbe Richtung, wie bei echten Embeddings mit ähnlicher Bedeutung.
VOCABULARY = ["apple", "pear", "cat", "dog", "word0001"]


def fake_vector(text: str) -> list[float]:
    words = text.lower().replace(".", " ").split()
    return [float(words.count(w)) for w in VOCABULARY] + [0.01]  # nie ein Nullvektor


def fake_client(calls: list, chats: list | None = None):
    """Client, der jede Embedding-Anfrage in calls festhält. 1 Token pro Wort, 1 USD/Token.

    Chat-Anfragen landen in chats und bekommen die gestreamte Antwort ANSWER.
    """

    def create(model, input):
        calls.append(input)
        tokens = sum(len(text.split()) for text in input)
        return CreateEmbeddingResponse.model_validate(
            {
                "object": "list",
                "model": model,
                # Absichtlich verkehrt herum: embed() muss nach index sortieren
                "data": [
                    {"object": "embedding", "index": i, "embedding": fake_vector(t)}
                    for i, t in reversed(list(enumerate(input)))
                ],
                "usage": {
                    "prompt_tokens": tokens,
                    "total_tokens": tokens,
                    "cost": tokens,
                },
            }
        )

    def complete(**kwargs):
        if chats is not None:
            chats.append(kwargs)
        base = {"id": "gen-1", "object": "chat.completion.chunk", "created": 0}
        base["model"] = kwargs["model"]
        usage = {"prompt_tokens": 500, "completion_tokens": 20, "total_tokens": 520}
        parts = [
            {"choices": [{"index": 0, "delta": {"content": ANSWER}}]},
            {"choices": [], "usage": usage | {"cost": 0.0001}},
        ]
        return iter(ChatCompletionChunk.model_validate(base | p) for p in parts)

    return SimpleNamespace(
        embeddings=SimpleNamespace(create=create),
        chat=SimpleNamespace(completions=SimpleNamespace(create=complete)),
    )


ANSWER = "Munich is a regional centre (LEP_Anhang_1_Zentrale_Orte_2018.pdf, p. 1)."


@pytest.fixture
def chats() -> list:
    return []


@pytest.fixture
def calls(monkeypatch, chats) -> list:
    calls: list = []
    monkeypatch.setattr(core.llm, "get_client", lambda: fake_client(calls, chats))
    return calls


def test_embed_batches_keeps_order_and_sums_cost(calls):
    texts = [f"apple {n}" for n in range(EMBEDDING_BATCH_SIZE + 5)]
    response = embed(texts, "test/model")
    assert [len(batch) for batch in calls] == [EMBEDDING_BATCH_SIZE, 5]
    assert response.vectors[0] == fake_vector(texts[0])
    assert response.usage.prompt_tokens == 2 * len(texts)
    assert response.usage.cost_usd == 2 * len(texts)


def test_embed_errors_become_readable(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        embed(["text"], "test/model")


def fruit_and_pets() -> list:
    return chunk_document(
        document("apple apple pear.\n\ncat dog dog.\n\napple pear pear."),
        size=20,
        overlap=0,
    )


def test_search_returns_most_similar_chunk_first(calls):
    index = SearchIndex()
    chunks = fruit_and_pets()
    index.add(chunks)
    hits, usage = index.search("dog", 20, 0, top_k=2)
    assert "dog dog" in hits[0].chunk.text
    assert "dog" not in hits[1].chunk.text
    assert hits[0].similarity > hits[1].similarity
    assert hits[0].chunk.pages_label == "p. 1"
    assert usage.prompt_tokens == 1
    index.delete()


def test_same_chunks_are_embedded_only_once(calls):
    index = SearchIndex()
    chunks = fruit_and_pets()
    first = index.add(chunks)
    second = index.add(chunks)
    assert first.cost_usd > 0
    assert second.cost_usd == 0
    assert len(calls) == 1
    assert index.collection.count() == len(chunks)
    index.delete()


def test_search_only_uses_chunks_of_current_settings(calls):
    index = SearchIndex()
    doc = document(WORDS)
    index.add(chunk_document(doc, size=300, overlap=0))
    index.add(chunk_document(doc, size=500, overlap=100))
    index.add(chunk_document(doc, size=300, overlap=0, sentences=False))
    hits, _ = index.search("word0001", 300, 0, top_k=50)
    assert hits
    assert all(hit.chunk.id.split("-")[1:4] == ["300", "0", "s"] for hit in hits)
    hits, _ = index.search("word0001", 300, 0, top_k=50, sentences=False)
    assert all(hit.chunk.id.split("-")[1:4] == ["300", "0", "w"] for hit in hits)
    index.delete()


def test_sessions_do_not_see_each_others_chunks(calls):
    mine, theirs = SearchIndex(), SearchIndex()
    mine.add(fruit_and_pets())
    assert theirs.collection.count() == 0
    mine.delete()
    theirs.delete()


# ------------------------------------------------------------------ Antworten


def hit(text: str, filename: str = "plan.pdf", page: int = 3) -> Hit:
    chunk = Chunk(id="x", filename=filename, page_start=page, page_end=page, text=text)
    return Hit(chunk=chunk, similarity=0.5)


def test_messages_contain_rules_passages_with_sources_and_question():
    messages = build_rag_messages("Who?", [hit("Alice did it.")], [])
    assert messages[0]["role"] == "system"
    assert NOT_COVERED in messages[0]["content"]
    question = messages[-1]
    assert question["role"] == "user"
    assert "[1] plan.pdf, p. 3\nAlice did it." in question["content"]
    assert question["content"].endswith("Question: Who?")


def test_passages_show_the_section_on_its_own_line():
    passage = hit("Rosenheim, Passau.")
    passage.chunk.section = "2. Oberzentren"
    content = build_rag_messages("Which?", [passage], [])[-1]["content"]
    # Die Quelle bleibt „Datei, Seite“, die Überschrift steht darunter
    assert "[1] plan.pdf, p. 3\nSection: 2. Oberzentren\nRosenheim, Passau." in content


def test_messages_keep_recent_history_without_old_passages():
    history = [
        RagTurn(role=role, content=f"{role} {n}", hits=[hit("old passage")])
        for n in range(5)
        for role in ("user", "assistant")
    ]
    messages = build_rag_messages("Next?", [hit("new passage")], history)
    earlier = messages[1:-1]
    assert len(earlier) == HISTORY_MESSAGES
    assert earlier[-1] == {"role": "assistant", "content": "assistant 4"}
    assert not any("old passage" in m["content"] for m in messages)


# ------------------------------------------------------------------ Seite


def open_page() -> AppTest:
    at = AppTest.from_file("../app.py", default_timeout=30)
    at.session_state["authenticated"] = True
    at.run()
    return at.switch_page("app_pages/rag.py").run()


def load_examples(at: AppTest) -> AppTest:
    return (
        next(b for b in at.button if b.label == "Load example documents").click().run()
    )


def metric(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_example_documents_are_read_and_split():
    at = load_examples(open_page())
    assert not at.exception
    assert len(at.session_state["rag_documents"]) == 3
    assert any("86 pages" in c.value for c in at.caption)
    # Zweites Laden erzeugt keine Dubletten
    at = load_examples(at)
    assert len(at.session_state["rag_documents"]) == 3


def test_sidebar_contains_no_module_settings():
    at = open_page()
    assert not (at.sidebar.button or at.sidebar.slider or at.sidebar.radio)


def test_chat_is_disabled_without_documents():
    at = open_page()
    assert at.chat_input[0].proto.disabled


def test_question_is_answered_from_passages_with_sources(calls, chats):
    at = load_examples(open_page())
    at.chat_input[0].set_value("Which towns are regional centres?").run()
    assert not at.exception
    # Fehlende Embeddings entstehen mit der ersten Frage, danach die Frage selbst
    assert len(calls) == 5  # 369 Abschnitte in 4 Paketen, dazu die Frage
    # Die Zeile links zeigt schon im selben Durchlauf den neuen Stand
    assert any("369 chunks, 369 with embedding" in c.value for c in at.caption)
    assert not any(b.label == "Create embeddings" for b in at.button)
    [chat] = chats
    assert "Passages:" in chat["messages"][-1]["content"]
    assert any(ANSWER in md.value for md in at.markdown)
    # Expander mit Icon führt AppTest als Status
    assert any(e.label == "Retrieved passages" for e in at.status)
    assert any("similarity" in md.value for md in at.markdown)
    assert metric(at, "Answers") == "1"
    assert float(metric(at, "Cost (USD)")) > 0.0001  # Antwort plus Embeddings

    # Zweite Frage: keine neuen Embeddings für die Abschnitte, nur für die Frage
    at.chat_input[0].set_value("And in Upper Bavaria?").run()
    assert len(calls) == 6
    assert (
        len(chats[1]["messages"]) == 4
    )  # System, erste Frage, erste Antwort, neue Frage
    assert len(at.session_state["rag_history"]) == 4


def test_sentence_toggle_changes_the_chunks():
    at = load_examples(open_page())
    assert any("369 chunks" in c.value for c in at.caption)
    at.toggle[0].set_value(False).run()
    assert not at.exception
    assert any("321 chunks, 0 with embedding" in c.value for c in at.caption)


def test_create_embeddings_button_indexes_ahead(calls):
    at = load_examples(open_page())
    next(b for b in at.button if b.label == "Create embeddings").click().run()
    assert not at.exception
    assert not any(b.label == "Create embeddings" for b in at.button)
    assert any("369 chunks, 369 with embedding" in c.value for c in at.caption)


def test_missing_api_key_shows_message_instead_of_traceback(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    at = load_examples(open_page())
    at.chat_input[0].set_value("Hallo").run()
    assert not at.exception
    assert any("OPENROUTER_API_KEY is missing" in e.value for e in at.error)
    assert at.session_state["rag_history"] == []


def test_reruns_do_not_create_new_collections():
    at = open_page()
    first = at.session_state["rag_index"].name
    at.run()
    assert at.session_state["rag_index"].name == first


# ------------------------------------------------------------------ Fragenkatalog


def test_question_catalogue_is_valid_and_points_to_real_pages():
    questions = load_questions()
    assert len(questions) >= 10
    assert len({q.id for q in questions}) == len(questions)
    page_counts = {
        p.name: len(read_pdf(p.read_bytes(), p.name).pages) for p in sample_files()
    }
    for q in questions:
        if q.not_covered:
            assert not q.source and not q.expect, q.id
        else:
            assert q.source in page_counts, q.id
            assert q.expect, q.id
            assert all(1 <= page <= page_counts[q.source] for page in q.pages), q.id


def test_answer_checks():
    question = Question(
        id="x",
        question="?",
        source="plan.pdf",
        pages=[4],
        expect=["Metropole", "zwei|two"],
    )
    assert is_correct(question, "München ist eine Metropole, seit two years.")
    assert not is_correct(question, "München ist eine Metropole.")
    assert is_cited(question, "Metropole (plan.pdf, p. 3–4).")
    assert not is_cited(question, "Metropole (plan.pdf, p. 3).")
    assert not is_cited(question, "Metropole (other.pdf, p. 4).")
    uncovered = Question(id="y", question="?", not_covered=True)
    assert is_correct(uncovered, f" {NOT_COVERED}\n")
    assert not is_correct(uncovered, "Die Hundesteuer beträgt 100 Euro.")


def test_found_needs_right_file_and_page():
    question = Question(id="x", question="?", source="plan.pdf", pages=[4])
    assert is_found(question, [hit("text", "plan.pdf", page=4)])
    assert not is_found(question, [hit("text", "plan.pdf", page=3)])
    assert not is_found(question, [hit("text", "other.pdf", page=4)])
