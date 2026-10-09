"""RAG Chatbot: Fragen zu eigenen PDFs, beantwortet nur aus gefundenen Abschnitten mit Quelle.

Layout nach docs/design.md: Einstellungen | Chat | Kennzahlen im Verhältnis 1:2:1, der Chat
sieht aus wie auf der Chat-Seite (gemeinsame Bausteine in core/chat_ui.py, CSS in chat.css).
Links stehen beide Phasen von RAG untereinander: erst Indexieren (Dokumente, Abschnitte,
Embeddings), dann Abfragen (Modell, Anzahl Treffer).
"""

import streamlit as st

from core.chat_ui import load_chat_css, scroll_chat_to_bottom, show_meta, show_question
from core.config import get_secret
from core.llm import DEFAULT_MAX_TOKENS, LLMError, Usage, stream
from core.models import format_cost, model_name, model_options, model_price
from core.rag import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    Chunk,
    Document,
    Hit,
    RagError,
    RagTurn,
    SearchIndex,
    build_rag_messages,
    chunk_document,
    read_pdf,
    sample_files,
)

# Niedrig, weil die Antwort nah an den Abschnitten bleiben soll, nicht kreativ sein
ANSWER_TEMPERATURE = 0.2

st.set_page_config(layout="wide")  # drei Spalten brauchen die volle Breite
load_chat_css()


@st.cache_data(max_entries=50)
def cached_read(data: bytes, filename: str) -> Document:
    """Liest jedes PDF nur einmal, auch über Neuausführungen der Seite hinweg."""
    return read_pdf(data, filename)


# Eingelesene Dokumente dieser Sitzung, nach Datei-Hash: dieselbe Datei zählt nur einmal.
# Nur im Session State, also für andere Besucher unsichtbar und nach dem Schließen weg.
documents: dict[str, Document] = st.session_state.setdefault("rag_documents", {})
st.session_state.setdefault("rag_upload_key", 0)
# Embeddings dieser Sitzung in einer eigenen Chroma-Sammlung. Nicht mit setdefault: Das würde
# bei jedem Neuaufbau eine neue Sammlung anlegen, die dann ungenutzt im Speicher bleibt.
if "rag_index" not in st.session_state:
    st.session_state.rag_index = SearchIndex()
index: SearchIndex = st.session_state.rag_index
# Kosten der Embeddings (Abschnitte und Fragen); die Antworten tragen ihre Kosten selbst
st.session_state.setdefault("rag_embedding_usage", Usage())
history: list[RagTurn] = st.session_state.setdefault("rag_history", [])


def add_document(data: bytes, filename: str) -> None:
    try:
        document = cached_read(data, filename)
    except RagError as err:
        st.error(str(err), icon=":material/error:")
        return
    documents.setdefault(document.file_hash, document)


def clear_documents() -> None:
    documents.clear()
    st.session_state.rag_upload_key += 1  # neuer key leert auch das Upload-Feld
    # Embeddings mitlöschen, sonst bleiben sie bis zum Neustart im Speicher des Servers
    st.session_state.rag_index.delete()
    st.session_state.rag_index = SearchIndex()


def embed_chunks(chunks: list[Chunk]) -> None:
    """Legt fehlende Embeddings an; Fehler erscheinen als Meldung."""
    with st.spinner(f"Embedding {len(chunks)} chunks …"):
        st.session_state.rag_embedding_usage += index.add(chunks)


@st.dialog("Chunks", width="large")
def show_chunks(chunks_by_file: dict[str, list[Chunk]]) -> None:
    """Abschnitte eines Dokuments als Tabelle, ein Klick zeigt den ganzen Abschnitt."""
    filename = st.selectbox("Document", list(chunks_by_file))
    chunks = chunks_by_file[filename]
    selection = st.dataframe(
        [
            {
                "#": number,
                "Pages": chunk.pages_label,
                "Section": chunk.section,
                "Characters": len(chunk.text),
                "Text": " ".join(chunk.text.split()),
            }
            for number, chunk in enumerate(chunks)
        ],
        hide_index=True,
        height=300,
        on_select="rerun",
        selection_mode="single-row",
        # Neuer key je Dokument, sonst zeigt die Auswahl auf eine Zeile des vorigen
        key=f"rag_chunk_table_{filename}",
    )
    selected = selection.selection.rows[0] if selection.selection.rows else 0
    chunk = chunks[selected]
    st.markdown(f"**Chunk {selected}** · {chunk.filename}, {chunk.pages_label}")
    if chunk.section:
        st.caption(chunk.section)
    st.code(chunk.text, language=None, wrap_lines=True)
    st.caption(
        f"Click a row to see the whole chunk. "
        f"{sum(c.page_end > c.page_start for c in chunks)} of {len(chunks)} chunks run "
        "across a page break; their source reads 'p. 3–4'. The section line comes from "
        "the numbered headings and is embedded together with the text."
    )


def show_passages(hits: list[Hit]) -> None:
    """Aufklappbar unter jeder Antwort: die Abschnitte, die das Modell bekommen hat."""
    with st.expander("Retrieved passages", icon=":material/find_in_page:"):
        st.caption(
            "The chunks most similar to the question, best first. The model saw only "
            "these. Similarity is only meaningful compared with the other hits."
        )
        for number, hit in enumerate(hits, start=1):
            st.markdown(
                f"**[{number}]** {hit.chunk.filename}, {hit.chunk.pages_label} "
                f":gray-badge[similarity {hit.similarity:.3f}]"
            )
            if hit.chunk.section:
                st.caption(hit.chunk.section)
            st.code(hit.chunk.text, language=None, wrap_lines=True)


st.title("RAG Chatbot")
settings, chat, stats = st.columns([1, 2, 1], gap="large")

# Die Container mit key bekommen die CSS-Klasse st-key-<key>, darüber greift chat.css
with settings.container(key="chat_settings"):
    st.markdown("##### Documents")
    if st.button("Load example documents", icon=":material/folder_open:"):
        for path in sample_files():
            add_document(path.read_bytes(), path.name)
    uploads = st.file_uploader(
        "Your PDFs",
        type="pdf",
        accept_multiple_files=True,
        key=f"rag_uploads_{st.session_state.rag_upload_key}",
        help=(
            "Uploaded files stay in this browser session only; other visitors cannot see "
            "them. Scanned PDFs without a text layer cannot be read."
        ),
    )
    for upload in uploads or []:
        add_document(upload.getvalue(), upload.name)
    # Eine Zeile pro Datei statt Tabelle: lange Dateinamen schoben die Zahlen aus dem Bild
    for d in documents.values():
        st.caption(f"**{d.filename}**  \n{len(d.pages)} pages")
    if documents:
        st.button("Clear documents", icon=":material/delete:", on_click=clear_documents)

    st.markdown("##### Chunks")
    chunk_size = st.slider(
        "Chunk size (characters)",
        200,
        2000,
        DEFAULT_CHUNK_SIZE,
        step=50,
        help=(
            "How long each passage is. Small chunks lose context, large ones mix several "
            "topics and make the search less precise."
        ),
    )
    overlap = st.slider(
        "Overlap (characters)",
        0,
        chunk_size // 2,
        min(DEFAULT_OVERLAP, chunk_size // 2),
        step=25,
        help=(
            "Roughly how much text neighbouring chunks share; each chunk starts at the "
            "nearest sentence start, so the exact amount varies. More overlap means more "
            "chunks and higher cost."
        ),
    )
    sentences = st.toggle(
        "Cut at sentences and headings",
        value=True,
        help=(
            "On: chunks start at a sentence and end before a heading or after a sentence. "
            "Off: chunks are cut at the nearest word, which shows how much the boundaries "
            "matter."
        ),
    )
    chunks_by_file = {
        document.filename: chunk_document(document, chunk_size, overlap, sentences)
        for document in documents.values()
    }
    all_chunks = [chunk for chunks in chunks_by_file.values() for chunk in chunks]
    # Stand der Embeddings erst am Ende zeichnen (show_index_status): Die erste Frage legt
    # fehlende Embeddings an, und die Zeile soll danach schon den neuen Stand zeigen
    index_status = st.empty()

    st.markdown("##### Answers")
    default_model = get_secret("OPENROUTER_MODEL")
    options = model_options(default_model)  # nach Preis sortiert, günstigstes zuerst
    model = st.radio(
        "Model",
        options,
        index=options.index(default_model) if default_model in options else 0,
        format_func=model_name,
        # Preis unter jedem Modell; "$" maskieren, sonst liest Streamlit es als Formel
        captions=[(model_price(m) or "").replace("$", "\\$") for m in options],
        help=(
            "Writes the answer from the retrieved passages. Prices are in USD per million "
            "input / output tokens; the passages make the input the larger part."
        ),
    )
    top_k = st.slider(
        "Passages per question",
        1,
        10,
        DEFAULT_TOP_K,
        help=(
            "How many of the most similar chunks the model gets. More passages give more "
            "context, but also more noise and higher cost."
        ),
    )

with stats.container(key="chat_stats"):
    st.markdown("##### Session")
    summary = st.empty()  # nach einer neuen Antwort erneut gefüllt
    # Setzt den Verlauf zurück; Dokumente und Embeddings bleiben
    if st.button("Clear chat", icon=":material/delete:"):
        history.clear()
    # Unsichtbarer Platzhalter für das Skript aus scroll_chat_to_bottom
    scroll_trigger = st.empty()


def show_index_status() -> None:
    missing = index.missing(all_chunks)
    with index_status.container():
        if all_chunks:
            st.caption(
                f"{len(all_chunks)} chunks, {len(all_chunks) - len(missing)} with embedding"
            )
            if st.button("View chunks", icon=":material/view_agenda:"):
                show_chunks(chunks_by_file)
        if not missing:
            return
        # Grob 4 Zeichen pro Token, 0.02 USD pro Million Tokens (siehe EMBEDDING_MODEL)
        estimate = sum(len(c.text) for c in missing) / 4 * 0.02 / 1e6
        if st.button(
            "Create embeddings",
            icon=":material/database:",
            help=(
                f"Turns each chunk into an embedding with {EMBEDDING_MODEL}, about "
                f"{format_cost(estimate)} for these chunks. Otherwise this happens with "
                "the first question."
            ),
        ):
            try:
                embed_chunks(missing)
            except LLMError as err:
                st.error(str(err), icon=":material/error:")
            else:
                st.rerun()  # Zeile und Kennzahlen mit dem neuen Stand zeigen


def show_summary() -> None:
    answers = sum((turn.usage for turn in history if turn.usage), Usage())
    embeddings: Usage = st.session_state.rag_embedding_usage
    total = answers + embeddings
    with summary.container():
        st.metric(
            "Cost (USD)",
            format_cost(total.cost_usd, unit=False),
            help=(
                f"Answers {format_cost(answers.cost_usd)}, embeddings "
                f"{format_cost(embeddings.cost_usd)}."
            ),
        )
        st.metric(
            "Tokens in",
            total.prompt_tokens,
            help=(
                "Mostly the retrieved passages sent with each question, plus the chunks "
                "turned into embeddings."
            ),
        )
        st.metric("Tokens out", total.completion_tokens)
        st.metric("Tokens total", total.total_tokens)
        st.metric("Answers", sum(turn.role == "assistant" for turn in history))


with chat:
    if not get_secret("OPENROUTER_API_KEY"):
        st.warning(
            "OPENROUTER_API_KEY is not set, the chatbot cannot answer.",
            icon=":material/key_off:",
        )
    # Eigener Scrollbereich; chat.css passt die Höhe an das Browserfenster an
    messages = st.container(
        height=450, border=False, autoscroll=True, key="chat_scroll"
    )

# Eingabefeld fest am unteren Bildschirmrand (st.bottom), ausgerichtet an der mittleren Spalte
with st.bottom, st.container(key="chat_input_row"):
    _, input_column, _ = st.columns([1, 2, 1], gap="large")
    prompt = input_column.chat_input(
        "Ask about your documents" if documents else "Load or upload documents first",
        disabled=not documents,
        submit_mode="disable",
    )

show_summary()

with messages:
    if not history and not prompt:
        st.caption(
            "Load the example documents or upload a PDF on the left, then ask a question "
            "below, for example: Which towns are regional centres (Oberzentren)?"
        )
    for turn in history:
        if turn.role == "user":
            show_question(turn.content)
        else:
            st.markdown(turn.content)
            show_meta(turn.model, turn.usage)
            show_passages(turn.hits)

if prompt:
    with messages:
        show_question(prompt)
    # Neue Frage sofort zeigen; danach hält Streamlits autoscroll die Antwort unten sichtbar
    scroll_chat_to_bottom(scroll_trigger)
    with messages:
        try:
            if missing := index.missing(all_chunks):
                embed_chunks(missing)
            with st.spinner("Searching the documents …"):
                hits, search_usage = index.search(
                    prompt, chunk_size, overlap, top_k, sentences
                )
            st.session_state.rag_embedding_usage += search_usage
            response = stream(
                build_rag_messages(prompt, hits, history),
                model,
                temperature=ANSWER_TEMPERATURE,
            )
            st.write_stream(response)
            if not response.text:
                raise LLMError("The model returned no text. Please send again.")
        except LLMError as err:
            st.error(str(err), icon=":material/error:")
        else:
            # Frage und Antwort erst jetzt in den Verlauf: ohne Antwort keine halbe Runde
            history.append(RagTurn(role="user", content=prompt))
            history.append(
                RagTurn(
                    role="assistant",
                    content=response.text,
                    model=response.model,
                    usage=response.usage,
                    hits=hits,
                )
            )
            show_meta(response.model, response.usage)
            if response.finish_reason == "length":
                st.caption(
                    f"Answer cut off after {DEFAULT_MAX_TOKENS} tokens (cost limit)."
                )
            show_passages(hits)
    # Auch nach fertiger Antwort (oder Fehlermeldung) ans Ende, falls zwischendurch gescrollt
    scroll_chat_to_bottom(scroll_trigger)
    show_summary()

show_index_status()
