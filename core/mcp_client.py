"""MCP-Client für die Seite: startet den Server über stdio und protokolliert jede Nachricht.

Die Seite spielt hier die Rolle der KI-App (wie Claude Desktop): Sie startet core/mcp_server.py
als eigenen Prozess, begrüßt ihn (initialize), fragt die Werkzeuge ab (tools/list) und ruft
eins auf (tools/call). Zwischen Client und Prozess hängt ein Mitschreiber, der jede JSON-RPC-
Nachricht in beide Richtungen festhält, damit die Seite das Protokoll zeigen kann.

Streamlit ist synchron, das SDK asynchron. Deshalb läuft die Verbindung in einem eigenen Thread
mit eigener Ereignisschleife und bleibt offen; Aufrufe aus der Seite werden dorthin übergeben.
So startet der Server nur einmal, und der Suchindex der Dokumentsuche wird nur einmal gebaut.
"""

import asyncio
import os
import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path

import anyio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from core.config import get_secret

SERVER_SCRIPT = Path(__file__).with_name("mcp_server.py")
TIMEOUT = 120  # Sekunden; die erste Dokumentsuche baut den Suchindex (Embeddings)


@dataclass
class Message:
    direction: str  # "client → server" oder "server → client"
    body: dict  # die JSON-RPC-Nachricht, wie sie über stdio geht
    time: float = field(default_factory=time.time)

    @property
    def method(self) -> str:
        """Kurzname für die Anzeige: Methode einer Anfrage, sonst Antwort oder Fehler."""
        if "method" in self.body:
            return self.body["method"]
        return "error" if "error" in self.body else "result"


def server_parameters() -> StdioServerParameters:
    """So startet auch Claude Desktop den Server: Python-Interpreter plus Pfad zum Skript.

    Der API-Key für die Dokumentsuche geht als Umgebungsvariable mit; das SDK gibt dem
    Prozess sonst nur wenige, unkritische Variablen weiter.
    """
    env = {"PATH": os.environ.get("PATH", "")}
    if key := get_secret("OPENROUTER_API_KEY"):
        env["OPENROUTER_API_KEY"] = key
    return StdioServerParameters(
        command=sys.executable, args=[str(SERVER_SCRIPT)], env=env
    )


def to_json(message) -> dict:
    return message.message.model_dump(mode="json", by_alias=True, exclude_none=True)


class McpConnection:
    """Eine offene Verbindung zum Server. Methoden blockieren, bis die Antwort da ist."""

    def __init__(self) -> None:
        self.handshake: list[Message] = []  # Nachrichten beim Verbinden
        self._log: list[Message] = self.handshake
        self._lock = threading.Lock()  # immer nur ein Aufruf gleichzeitig
        self._loop = asyncio.new_event_loop()
        self._jobs: asyncio.Queue | None = None
        self.server_name = ""
        ready: Future = Future()
        threading.Thread(target=self._run, args=(ready,), daemon=True).start()
        ready.result(timeout=TIMEOUT)

    # ---------------------------------------------------------------- im eigenen Thread

    def _run(self, ready: Future) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main(ready))

    async def _main(self, ready: Future) -> None:
        """Hält Prozess, Mitschreiber und Sitzung in einer Aufgabe offen und arbeitet Aufträge ab.

        Alles in einer Aufgabe, weil anyio-Kontexte in derselben Aufgabe geschlossen werden
        müssen, in der sie geöffnet wurden.
        """
        self._jobs = asyncio.Queue()
        try:
            async with stdio_client(server_parameters()) as (read, write):
                to_server_send, to_server_receive = anyio.create_memory_object_stream(
                    16
                )
                to_client_send, to_client_receive = anyio.create_memory_object_stream(
                    16
                )

                async def outgoing() -> None:
                    async for message in to_server_receive:
                        self._log.append(Message("client → server", to_json(message)))
                        await write.send(message)

                async def incoming() -> None:
                    async for message in read:
                        if not isinstance(message, Exception):
                            self._log.append(
                                Message("server → client", to_json(message))
                            )
                        await to_client_send.send(message)

                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(outgoing)
                    tasks.start_soon(incoming)
                    async with ClientSession(
                        to_client_receive, to_server_send
                    ) as session:
                        result = await session.initialize()
                        self.server_name = result.server_info.name
                        # Die Bestätigung notifications/initialized geht erst nach der Antwort
                        # raus; kurz warten, damit sie noch zur Begrüßung zählt
                        await anyio.sleep(0.05)
                        ready.set_result(True)
                        while True:
                            job, future = await self._jobs.get()
                            try:
                                future.set_result(await job(session))
                            except Exception as err:  # noqa: BLE001  (gehört der wartenden Seite)
                                future.set_exception(err)
        except Exception as err:  # noqa: BLE001  (Startfehler an den Konstruktor)
            if not ready.done():
                ready.set_exception(err)

    # ---------------------------------------------------------------- aus der Seite

    def _submit(self, job) -> tuple[object, list[Message]]:
        """Führt job(session) im Verbindungs-Thread aus; liefert Ergebnis und Nachrichten."""
        with self._lock:
            self._log = []
            future: Future = Future()
            self._loop.call_soon_threadsafe(self._jobs.put_nowait, (job, future))
            result = future.result(timeout=TIMEOUT)
            return result, self._log

    def list_tools(self):
        async def job(session: ClientSession):
            return (await session.list_tools()).tools

        return self._submit(job)

    def call_tool(self, name: str, arguments: dict):
        async def job(session: ClientSession):
            return await session.call_tool(name, arguments)

        return self._submit(job)
