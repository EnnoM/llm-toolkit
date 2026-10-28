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
        page="app_pages/analyst.py",
    ),
    CatalogEntry(
        id="workflow_freigabe",
        title="Approval Workflow",
        icon=":material/approval:",
        summary=(
            "Classifies incoming emails and drafts replies. A human approves every "
            "draft, and every step is logged."
        ),
        techniques=("Classification", "Human in the loop", "Audit log"),
        level=3,
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
        techniques=("Model Context Protocol", "Tool server"),
        level=3,
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
