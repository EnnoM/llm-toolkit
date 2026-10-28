"""Modulkatalog für die öffentliche Startseite „Home“ und die Navigation. Keine Oberfläche.

Texte auf Englisch wie alle öffentlichen Seiten (docs/design.md). Reihenfolge und ids wie in
docs/modules.yaml; tests/test_catalog.py prüft, dass beide zusammenpassen.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogEntry:
    id: str  # wie in docs/modules.yaml
    title: str
    icon: str
    summary: str
    techniques: tuple[str, ...]
    level: int  # Schwierigkeitsstufe, Schlüssel in LEVELS; entspricht Phase minus 1
    page: str | None = None  # None: noch nicht gebaut, erscheint auf Home als WIP
    how: tuple[
        str, ...
    ] = ()  # „How it works“ auf der Modulseite, drei bis vier Schritte


# Stufen für Seitenleiste und Home: vom einzelnen Aufruf bis zu ganzen Anwendungen
LEVELS = {
    1: ("Basics", "Talking to a model: prompts, settings and structured output."),
    2: ("Knowledge and tools", "The model works with documents, tools and a database."),
    3: ("Workflows and agents", "Several steps, human approval and more autonomy."),
    4: ("Quality and safety", "Measuring and protecting LLM applications."),
}

CATALOG = (
    CatalogEntry(
        id="chat",
        title="Chat",
        icon=":material/chat:",
        summary=(
            "A chat with streamed answers, an adjustable system prompt and temperature. "
            "Every answer shows its tokens and cost."
        ),
        techniques=("Streaming", "System prompt", "Token and cost tracking"),
        level=1,
        how=(
            "Your messages and the system prompt go to the model you pick, through OpenRouter. The whole conversation is sent again with every question.",
            "The answer streams in piece by piece. Temperature controls how much it varies.",
            "OpenRouter reports the tokens and cost of every answer; the right column adds them up.",
        ),
        page="app_pages/chat.py",
    ),
    CatalogEntry(
        id="prompt_labor",
        title="Prompt Lab",
        icon=":material/science:",
        summary=(
            "Runs one task with four prompting techniques side by side and compares "
            "the answers, cost and response time."
        ),
        techniques=("Zero-shot", "Few-shot", "Role prompting", "Step by step"),
        level=1,
        how=(
            "Pick a task with a known correct answer and up to four prompting techniques.",
            "The same input goes to the model once per technique, at temperature 0, so only the prompt differs.",
            "The answers are checked against the reference and compared by cost, tokens and time.",
        ),
        page="app_pages/prompt_lab.py",
    ),
    CatalogEntry(
        id="datenextraktion",
        title="Data Extraction",
        icon=":material/data_object:",
        summary=(
            "Turns free text such as invoices or job ads into validated JSON. Invalid "
            "output goes back to the model to correct."
        ),
        techniques=("Structured outputs", "Pydantic validation", "Self-correction"),
        level=1,
        how=(
            "Each document type is a Pydantic schema: fields, types and rules.",
            "The model has to answer in exactly that JSON schema (structured outputs).",
            "Pydantic validates the answer. Errors go back to the model, which may correct itself up to two times.",
        ),
        page="app_pages/extraction.py",
    ),
    CatalogEntry(
        id="rag_chatbot",
        title="RAG Chatbot",
        icon=":material/find_in_page:",
        summary=(
            "Answers questions about PDF documents using only the most relevant "
            "passages, and names file and page as the source."
        ),
        techniques=("Chunking", "Embeddings", "Vector search", "Source citations"),
        level=2,
        how=(
            "The PDFs are split into overlapping chunks, and each chunk becomes an embedding, a list of numbers that captures its meaning.",
            "Your question becomes an embedding too; the most similar chunks are found by comparing the numbers.",
            "Only these passages go to the model. It has to answer from them and cite file and page.",
        ),
        page="app_pages/rag.py",
    ),
    CatalogEntry(
        id="tool_use",
        title="Tool Use",
        icon=":material/build:",
        summary=(
            "The model decides on its own when to use tools: calculator, date, "
            "weather, web search and document search. Every tool call is visible."
        ),
        techniques=("Function calling", "Tool loop", "Step-by-step trace"),
        level=2,
        how=(
            "The model gets your question and a description of every selected tool.",
            "If it needs a tool, it answers with a tool call instead of text; the app runs the tool and sends back the result.",
            "This repeats for up to five rounds until the model answers. Every call is listed under the answer.",
        ),
        page="app_pages/tool_use.py",
    ),
    CatalogEntry(
        id="ki_datenanalyst",
        title="AI Data Analyst",
        icon=":material/query_stats:",
        summary=(
            "Turns questions in plain language into SQL, runs them read-only and "
            "shows the table, a chart and the query."
        ),
        techniques=("Text to SQL", "Read-only queries", "Charts"),
        level=2,
        how=(
            "The model gets the database schema and your question and writes one SQL query.",
            "Only a single SELECT runs, on a read-only connection. If it fails, the model sees the error and may correct the query once.",
            "The page shows a one-sentence answer, a chart, the table and always the SQL, so every number can be checked.",
        ),
        page="app_pages/analyst.py",
    ),
    CatalogEntry(
        id="workflow_freigabe",
        title="Approval Workflow",
        icon=":material/approval:",
        summary=(
            "Sorts incoming emails and drafts replies from the shop's order data. Unclear "
            "cases are flagged, a human approves every reply, every step is logged."
        ),
        techniques=("Classification", "Grounding", "Human in the loop", "Audit log"),
        level=3,
        how=(
            "The model sorts each email into a category and a priority (structured output).",
            "The app looks up the sender's orders, deliveries, payments and returns, the products mentioned and the shop policies in the shop's database.",
            "The model drafts a reply from these facts only, lists the facts it used and marks cases the data cannot settle as unclear, in red.",
            "A human edits, approves or discards every draft. Nothing is sent automatically, and every step is logged.",
        ),
        page="app_pages/workflow.py",
    ),
    CatalogEntry(
        id="mcp_server",
        title="MCP Server",
        icon=":material/hub:",
        summary=(
            "Makes the tools of this project available to Claude Desktop and other "
            "AI apps through the Model Context Protocol."
        ),
        techniques=(
            "Model Context Protocol",
            "JSON-RPC over stdio",
            "Tool descriptions",
            "Read-only tools",
        ),
        level=3,
        page="app_pages/mcp_server.py",
        how=(
            "The MCP server is a small separate program that offers read-only tools from this project: SQL on the shop database, order and customer lookup, document search.",
            "An AI app starts it and talks to it over stdin and stdout in JSON-RPC: initialize, tools/list, tools/call. Here this page plays the app; normally it is Claude Desktop.",
            "The app's model reads the tool descriptions, decides which tool to call and uses the result in its answer.",
        ),
    ),
    CatalogEntry(
        id="agent",
        title="Agent",
        icon=":material/smart_toy:",
        summary=(
            "Works towards a given goal on its own over several steps, with planning, "
            "tools and clear stop rules."
        ),
        techniques=("Planning", "Multi-step tool use", "Stop rules"),
        level=3,
    ),
    CatalogEntry(
        id="evaluation",
        title="Evaluation",
        icon=":material/fact_check:",
        summary=(
            "Measures answer quality with a test set, a model acting as judge and "
            "recorded calls."
        ),
        techniques=("Test sets", "LLM as judge", "Tracing"),
        level=4,
    ),
    CatalogEntry(
        id="security",
        title="Security",
        icon=":material/shield:",
        summary=(
            "Shows prompt injection live with and without defenses, masks personal "
            "data and measures how many attacks are blocked."
        ),
        techniques=("Prompt injection", "PII masking", "Defense rate"),
        level=4,
    ),
)


def built_modules() -> list[CatalogEntry]:
    """Module mit eigener Seite, in der Reihenfolge des Katalogs."""
    return [entry for entry in CATALOG if entry.page]


def level_title(level: int) -> str:
    return f"Level {level} · {LEVELS[level][0]}"


def levels_newest_first() -> list[tuple[int, list[CatalogEntry]]]:
    """Stufen für Home, neueste zuerst: gebaute Module oben, innerhalb der Stufe das neueste
    zuerst, geplante dahinter. Stufen ohne gebautes Modul kommen ans Ende."""
    groups = []
    for level in LEVELS:
        entries = [e for e in CATALOG if e.level == level]
        built = [e for e in entries if e.page][::-1]
        groups.append((level, built + [e for e in entries if not e.page]))
    started = [g for g in groups if g[1][0].page]
    return started[::-1] + [g for g in groups if not g[1][0].page]
