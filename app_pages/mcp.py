"""MCP Server: Die Seite spielt die KI-App, startet den Server und zeigt jede Nachricht.

Layout wie im Approval Workflow: Kopf, dann ein Rahmen „Server“ mit der Werkzeugliste links und
dem ausgewählten Werkzeug rechts (Formular, Ergebnis, Protokoll), darunter ein Rahmen mit der
Anleitung für Claude Desktop. Server in core/mcp_server.py, Client in core/mcp_client.py.
"""

import json

import pandas as pd
import streamlit as st

from core.mcp_client import McpConnection, Message
from core.mcp_server import SERVER_NAME
from core.page_intro import show_intro

st.set_page_config(
    layout="wide"
)  # Liste und Werkzeug nebeneinander brauchen die Breite

# Startwerte für das Formular, damit jedes Werkzeug mit einem Klick etwas Sinnvolles zeigt
EXAMPLE_ARGUMENTS = {
    "query_shop_database": {
        "sql": "SELECT channel, count(*) AS orders\nFROM orders\nGROUP BY channel\n"
        "ORDER BY orders DESC"
    },
    "lookup_order": {"order_id": 40477},
    "lookup_customer": {"email": "daniel.roth@example.com"},
    "search_documents": {"query": "Which towns in Schwaben are Oberzentren?"},
}

CLAUDE_CONFIG = {
    "mcpServers": {
        SERVER_NAME: {
            "command": "/path/to/llm-toolkit/.venv/bin/python",
            "args": ["/path/to/llm-toolkit/core/mcp_server.py"],
            "env": {"OPENROUTER_API_KEY": "your key, only needed for search_documents"},
        }
    }
}


@st.cache_resource(show_spinner="Starting the MCP server …")
def get_connection() -> McpConnection:
    """Ein Serverprozess für die ganze App; er bleibt offen, Aufrufe laufen nacheinander."""
    return McpConnection()


def show_messages(messages: list[Message], title: str) -> None:
    """Die JSON-RPC-Nachrichten, wie sie über stdin und stdout gehen."""
    with st.expander(
        f"{title} ({len(messages)} messages)", icon=":material/swap_horiz:"
    ):
        for message in messages:
            arrow = (
                ":material/arrow_forward:"
                if message.direction.startswith("client")
                else ":material/arrow_back:"
            )
            st.markdown(f"{arrow} **{message.direction}** · `{message.method}`")
            text = json.dumps(message.body, ensure_ascii=False, indent=2)
            st.code(
                text,
                language="json",
                wrap_lines=True,
                height=min(300, 40 + 20 * text.count("\n")),
            )


def argument_input(tool_name: str, name: str, schema: dict):
    """Ein Eingabefeld pro Parameter, abgeleitet aus dem JSON-Schema des Werkzeugs."""
    default = EXAMPLE_ARGUMENTS.get(tool_name, {}).get(name)
    key = f"mcp_{tool_name}_{name}"
    if schema.get("type") == "integer":
        return st.number_input(name, value=default or 0, step=1, key=key)
    if name == "sql":
        return st.text_area(name, value=default or "", height=130, key=key)
    return st.text_input(name, value=default or "", key=key)


def show_result(tool_name: str, result) -> None:
    text = "\n".join(c.text for c in result.content if getattr(c, "text", None))
    if result.is_error:
        st.error(text, icon=":material/error:")
        return
    st.caption("Result, as the AI app receives it")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        st.code(text, language=None, wrap_lines=True)
        return
    if tool_name == "query_shop_database":
        st.dataframe(
            pd.DataFrame(data["rows"], columns=data["columns"]), hide_index=True
        )
    with st.expander("Raw result (JSON)", icon=":material/data_object:"):
        st.code(json.dumps(data, ensure_ascii=False, indent=2), language="json")


st.title("MCP Server")
show_intro("mcp_server")

try:
    connection = get_connection()
    tools, list_messages = connection.list_tools()
except Exception as err:  # noqa: BLE001  (jeder Startfehler soll als Meldung erscheinen)
    st.error(f"The MCP server could not be started: {err}", icon=":material/error:")
    st.stop()

# ------------------------------------------------------------------ Server

with st.container(border=True):
    head = st.container(horizontal=True, vertical_alignment="center")
    head.markdown(
        f"**Server `{connection.server_name}`** :gray-badge[stdio] "
        f":gray-badge[{len(tools)} tools] :gray-badge[read-only]"
    )
    st.caption(
        "This page acts as the AI app: it started the server as a separate process, "
        "said hello (initialize) and asked for its tools (tools/list). Pick a tool, call it "
        "and see every message that goes back and forth."
    )
    show_messages(connection.handshake, "Connection")
    show_messages(list_messages, "Tool list")

    tool_list, detail = st.columns([2, 3], gap="large")
    with tool_list:
        names = [tool.name for tool in tools]
        selected = st.radio(
            "Tools",
            names,
            key="mcp_tool",
            # Erster Satz der Beschreibung; ohne rstrip stand bei manchen „return..“
            captions=[
                tool.description.split(". ")[0].rstrip(".") + "." for tool in tools
            ],
        )
    tool = next(t for t in tools if t.name == selected)

    with detail:
        st.markdown(f"##### `{tool.name}`")
        if tool.annotations and tool.annotations.read_only_hint:
            st.markdown(
                ":gray-badge[:material/lock: read-only] "
                ":small[The server marks the tool as read-only, and the code enforces it.]"
            )
        with st.expander("Description and input schema, as the model sees them"):
            st.markdown(tool.description.replace("\n", "  \n"))
            st.code(json.dumps(tool.input_schema, indent=2), language="json")
        with st.form(f"mcp_form_{tool.name}", border=False):
            properties = tool.input_schema.get("properties", {})
            arguments = {
                name: argument_input(tool.name, name, schema)
                for name, schema in properties.items()
            }
            called = st.form_submit_button(
                "Call tool", type="primary", icon=":material/play_arrow:"
            )
        if called:
            with st.spinner("Calling the tool over MCP …"):
                st.session_state[f"mcp_result_{tool.name}"] = connection.call_tool(
                    tool.name, arguments
                )
        if stored := st.session_state.get(f"mcp_result_{tool.name}"):
            result, call_messages = stored
            show_result(tool.name, result)
            show_messages(call_messages, "Tool call")

# ------------------------------------------------------------------ Claude Desktop

with st.container(border=True):
    st.markdown("**Connect to Claude Desktop**")
    st.caption(
        "The same server works in Claude Desktop and other MCP apps. It runs on your own "
        "computer; this hosted page cannot offer it to other apps."
    )
    steps, config = st.columns([1, 1], gap="large")
    with steps:
        st.markdown(
            "1. Clone the project and install it: `pip install -r requirements.txt`.\n"
            "2. In Claude Desktop open **Settings → Developer → Edit Config** and add the "
            "configuration on the right to `claude_desktop_config.json`. Replace "
            "`/path/to/llm-toolkit` with the folder of your copy.\n"
            "3. Restart Claude Desktop. The four tools appear in the tools menu. Ask for "
            'example *"Which sales channel had the most orders?"* and allow the tool call.'
        )
    config.code(json.dumps(CLAUDE_CONFIG, indent=2), language="json", wrap_lines=True)
