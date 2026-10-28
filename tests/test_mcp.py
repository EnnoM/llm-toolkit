"""MCP-Server und -Client: Werkzeuge ohne Client, Server-Liste, echte stdio-Verbindung, Seite."""

import asyncio

import pytest
from streamlit.testing.v1 import AppTest

from core import mcp_server
from core.mcp_client import McpConnection
from core.mcp_server import (
    ToolFailure,
    build_server,
    lookup_customer,
    lookup_order,
    query_shop_database,
    search_documents,
)
from core.tools import ToolError

# ------------------------------------------------------------------ Werkzeuge ohne MCP-Client


def test_query_shop_database_returns_numbers_and_dates():
    result = query_shop_database(
        "SELECT channel, count(*) AS orders, min(order_date) AS first "
        "FROM orders GROUP BY 1 ORDER BY 2 DESC"
    )
    assert result["columns"] == ["channel", "orders", "first"]
    assert result["rows"][0] == ["web", 3816, "2023-01-01"]
    assert result["truncated"] is False


def test_query_shop_database_caps_rows():
    result = query_shop_database("SELECT * FROM customers")
    assert result["row_count"] == mcp_server.MAX_SQL_ROWS
    assert result["truncated"] is True


@pytest.mark.parametrize(
    "sql", ["DELETE FROM orders", "DROP TABLE orders", "SELECT 1; SELECT 2"]
)
def test_query_shop_database_refuses_everything_but_one_select(sql):
    with pytest.raises(ToolFailure, match="Only|one"):
        query_shop_database(sql)


def test_lookup_order_finds_the_record_and_reports_unknown_numbers():
    found = lookup_order(40477)
    assert found["customer"] == "Paul Fischer"
    assert "left at the front door" in found["order"]
    with pytest.raises(ToolFailure, match="No order 12345"):
        lookup_order(12345)


def test_lookup_customer_by_email():
    customer = lookup_customer("Daniel.Roth@example.com ")
    assert customer["name"] == "Daniel Roth"
    assert "limescale" in customer["orders"][0]
    with pytest.raises(ToolFailure):
        lookup_customer("nobody@example.com")


def test_search_documents_passes_errors_to_the_model(monkeypatch):
    def broken(args):
        raise ToolError("No query given.")

    monkeypatch.setattr(mcp_server, "tool_search_documents", broken)
    with pytest.raises(ToolFailure, match="No query given"):
        search_documents("")


# ------------------------------------------------------------------ Server


def test_server_offers_four_read_only_tools_with_descriptions():
    tools = asyncio.run(build_server().list_tools())
    assert [t.name for t in tools] == [
        "query_shop_database",
        "lookup_order",
        "lookup_customer",
        "search_documents",
    ]
    for tool in tools:
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.destructive_hint is False
        assert len(tool.description) > 80, (
            tool.name
        )  # genug Kontext für ein fremdes Modell
    # Das Schema der Datenbank steht in der Beschreibung, sonst kann kein Modell SQL schreiben
    assert "Table orders" in tools[0].description


# ------------------------------------------------------------------ echte Verbindung über stdio


@pytest.fixture(scope="module")
def connection() -> McpConnection:
    link = McpConnection()  # startet core/mcp_server.py als eigenen Prozess
    # Wie die Seite: erst die Werkzeuge abfragen. Sonst holt das SDK die Liste beim ersten
    # Aufruf von selbst nach (um das Ergebnis zu prüfen), und das Protokoll sähe anders aus
    link.list_tools()
    return link


def test_handshake_follows_the_protocol(connection):
    assert [m.method for m in connection.handshake] == [
        "initialize",
        "result",
        "notifications/initialized",
    ]
    assert connection.server_name == "llm-toolkit"


def test_tool_call_over_stdio_logs_request_and_response(connection):
    result, messages = connection.call_tool("lookup_order", {"order_id": 40213})
    assert not result.is_error
    assert "Blender" in result.content[0].text
    assert [m.method for m in messages] == ["tools/call", "result"]
    assert messages[0].body["params"] == {
        "name": "lookup_order",
        "arguments": {"order_id": 40213},
    }


def test_refused_query_comes_back_as_tool_error(connection):
    result, _ = connection.call_tool(
        "query_shop_database", {"sql": "DROP TABLE orders"}
    )
    assert result.is_error
    assert "Only SELECT" in result.content[0].text


# ------------------------------------------------------------------ Seite


def test_page_lists_tools_and_the_claude_config():
    at = AppTest.from_file("../app_pages/mcp.py", default_timeout=60).run()
    assert not at.exception
    assert at.radio(key="mcp_tool").options == [
        "query_shop_database",
        "lookup_order",
        "lookup_customer",
        "search_documents",
    ]
    assert any('"mcpServers"' in c.value for c in at.code)
