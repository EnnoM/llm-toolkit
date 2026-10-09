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
    page: str | None = None  # None: noch nicht gebaut, erscheint auf Home als WIP


CATALOG = (
    CatalogEntry(
        id="chat",
        title="Chat",
        icon=":material/chat:",
        summary=(
            "A chat with streamed answers, an adjustable system prompt and temperature. "
            "Every answer shows its tokens and cost, which grow with each turn because "
            "the whole conversation is sent again."
        ),
        techniques=("Streaming", "System prompt", "Token and cost tracking"),
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
        page="app_pages/prompt_lab.py",
    ),
    CatalogEntry(
        id="datenextraktion",
        title="Data Extraction",
        icon=":material/data_object:",
        summary=(
            "Turns free text such as invoices, job ads or appointment requests into "
            "validated JSON. If the output breaks the schema, the model gets the error "
            "and corrects itself."
        ),
        techniques=("Structured outputs", "Pydantic validation", "Self-correction"),
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
    ),
)


def built_modules() -> list[CatalogEntry]:
    """Module mit eigener Seite, in der Reihenfolge des Katalogs."""
    return [entry for entry in CATALOG if entry.page]


def upcoming_modules() -> list[CatalogEntry]:
    """Module, die noch keine Seite haben (auf Home als WIP)."""
    return [entry for entry in CATALOG if not entry.page]
