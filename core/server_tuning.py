"""Längere Keep-Alive-Zeit für den Webserver von Streamlit (uvicorn). Keine Oberfläche.

Seit Streamlit auf uvicorn läuft, schließt der Server ruhende Verbindungen nach 5 Sekunden.
Auf Streamlit Community Cloud steht davor ein Google-Load-Balancer, der Verbindungen zum
Server bis zu 600 Sekunden offen hält und wiederverwendet. Erwischt er eine, die uvicorn gerade
geschlossen hat, hängt die Anfrage und endet nach bis zu 47 Sekunden mit Fehler 500. Bei einer
App mit wenigen Besuchern liegen Verbindungen oft länger als 5 Sekunden still; am 04.11.2026
hing deshalb etwa jede vierte Anfrage, im Browser „error loading dynamically imported module“.

Streamlit bietet dafür keine Einstellung. Die Konfiguration des laufenden Servers lässt sich aber
ändern: Jede neue Verbindung liest timeout_keep_alive beim Aufbau aus ihr.
"""

import gc

import uvicorn

# Länger als die 600 Sekunden des Load Balancers, damit immer er die Verbindung schließt
KEEP_ALIVE_SECONDS = 650


def extend_keep_alive() -> int:
    """Setzt timeout_keep_alive aller uvicorn-Konfigurationen im Prozess; gibt die Anzahl zurück."""
    configs = [obj for obj in gc.get_objects() if isinstance(obj, uvicorn.Config)]
    for server_config in configs:
        server_config.timeout_keep_alive = KEEP_ALIVE_SECONDS
    return len(configs)
