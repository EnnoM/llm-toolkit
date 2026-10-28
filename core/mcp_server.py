"""MCP-Server: Werkzeuge aus diesem Projekt für Claude Desktop und andere KI-Apps. Keine Oberfläche.

MCP (Model Context Protocol) ist ein offener Standard: Eine KI-App (Host, z. B. Claude Desktop)
startet diesen Server als eigenen Prozess und spricht mit ihm über stdin/stdout in JSON-RPC.
Sie fragt zuerst, welche Werkzeuge es gibt (tools/list), und ruft sie dann auf (tools/call).

Die Werkzeuge kommen aus anderen Modulen und sind alle nur lesend:
- query_shop_database: SELECT auf den Shop-Datensatz des AI Data Analyst (gleicher SQL-Schutz)
- lookup_order, lookup_customer: Kundendienst-Daten des Approval Workflow
- search_documents: Dokumentsuche des RAG-Moduls in den Beispiel-PDFs

Jedes Werkzeug ist eine normale Funktion, damit Tests es ohne MCP-Client aufrufen können;
build_server() meldet sie beim Server an. Start: python core/mcp_server.py (stdio).

Wichtig bei stdio: stdout gehört dem Protokoll. Jede andere Ausgabe dort (print, Warnungen)
würde die Nachrichten zerstören; Logs gehen nach stderr.
"""

import json
import sys
from datetime import date, datetime
from pathlib import Path

if __package__ in (None, ""):
    # Direkt als Datei gestartet (so startet Claude Desktop ihn): Projektordner importierbar
    # machen, damit „from core ...“ unabhängig vom Arbeitsordner funktioniert
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as MCPToolError
from mcp_types import ToolAnnotations

from core import analyst
from core.analyst import UnsafeSQLError, run_query
from core.tools import ToolError
from core.tools import search_documents as tool_search_documents
from core.workflow import load_shop, order_fact

SERVER_NAME = "llm-toolkit"
MAX_SQL_ROWS = 50  # mehr Zeilen braucht ein Modell selten, und die Antwort bleibt klein

# Alle Werkzeuge lesen nur. Die Kennzeichnung sagt der KI-App, dass ein Aufruf nichts verändert;
# erzwungen wird es trotzdem im Code (nur SELECT, schreibgeschützte Verbindung, nur Lesen).
READ_ONLY = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True
)


class ToolFailure(MCPToolError):
    """Fehler, den das aufrufende Modell sehen soll, z. B. „Order not found“.

    Unterklasse des ToolError aus dem SDK: Nur dessen Meldung kommt beim Client an, jede andere
    Ausnahme wird zu „Error executing tool“ und das Modell erfährt nicht, was falsch war.
    """


# ------------------------------------------------------------------ Werkzeuge


def query_shop_database(sql: str) -> dict:
    try:
        table = run_query(sql, max_rows=MAX_SQL_ROWS + 1)
    except UnsafeSQLError as err:
        raise ToolFailure(str(err)) from err
    except duckdb.Error as err:
        raise ToolFailure(f"The query failed: {err}") from err
    truncated = len(table) > MAX_SQL_ROWS
    table = table.head(MAX_SQL_ROWS)
    # Datumswerte als „2023-01-01“; DuckDB liefert hier Python-Datumsobjekte
    table = table.map(lambda v: v.isoformat() if isinstance(v, date | datetime) else v)
    # Über JSON von pandas: Zahlen bleiben Zahlen, Datumswerte werden ISO-Text
    split = json.loads(table.to_json(orient="split", index=False, date_format="iso"))
    return {
        "columns": split["columns"],
        "rows": split["data"],
        "row_count": len(table),
        "truncated": truncated,
    }


def find_customer(email: str) -> dict | None:
    email = email.strip().lower()
    return next((c for c in load_shop()["customers"] if c["email"] == email), None)


def lookup_order(order_id: int) -> dict:
    for customer in load_shop()["customers"]:
        for order in customer["orders"]:
            if order["id"] == order_id:
                return {"customer": customer["name"], "order": order_fact(order)}
    raise ToolFailure(f"No order {order_id} in the customer service records.")


def lookup_customer(email: str) -> dict:
    customer = find_customer(email)
    if customer is None:
        raise ToolFailure(f"No customer account for {email}.")
    return {
        "name": customer["name"],
        "email": customer["email"],
        "customer_since": str(customer["customer_since"]),
        "orders": [order_fact(o) for o in customer["orders"]],
    }


def search_documents(query: str) -> dict:
    try:
        result, _usage = tool_search_documents({"query": query})
    except ToolError as err:
        raise ToolFailure(str(err)) from err
    return result


# ------------------------------------------------------------------ Server

# Die Beschreibungen liest das fremde Modell, um zu entscheiden, wann und wie es ein Werkzeug
# benutzt. Deshalb englisch, konkret und mit den Grenzen des Werkzeugs.
DESCRIPTIONS = {
    "query_shop_database": (
        "Run one read-only SQL SELECT on the sales database of a fictional online shop "
        "(customers, products, orders and order items from January 2023 to December 2025). "
        "Returns at most 50 rows. DuckDB syntax. Only SELECT is allowed.\n\n"
        "Schema:\n{schema}"
    ),
    "lookup_order": (
        "Look up one order in the customer service records of Larkfield, a fictional shop "
        "(orders from autumn 2026, numbers like 40213): items, payment, shipment and return."
    ),
    "lookup_customer": (
        "Look up a customer of Larkfield by email address: name, customer since and all "
        "their orders with payment, shipment and return status."
    ),
    "search_documents": (
        "Search three public planning documents of the German state of Bavaria (state "
        "development plan 2013, its list of central places, notes on agricultural priority "
        "areas 2025; German language) and return the most relevant passages with file name "
        "and page. Use it for questions about central places such as Oberzentren, land use "
        "or planning rules in Bavaria."
    ),
}


def build_server() -> MCPServer:
    server = MCPServer(
        SERVER_NAME,
        log_level="WARNING",  # Logs gehen nach stderr; nur Wichtiges
        instructions=(
            "Read-only tools from the LLM Toolkit demo project. All data is fictional except "
            "the two public planning documents."
        ),
    )
    tools = {
        "query_shop_database": query_shop_database,
        "lookup_order": lookup_order,
        "lookup_customer": lookup_customer,
        "search_documents": search_documents,
    }
    for name, function in tools.items():
        description = DESCRIPTIONS[name]
        if "{schema}" in description:
            description = description.format(schema=analyst.schema_description())
        server.add_tool(
            function, name=name, description=description, annotations=READ_ONLY
        )
    return server


if __name__ == "__main__":
    build_server().run("stdio")
