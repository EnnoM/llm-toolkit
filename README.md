# LLM Toolkit

Modulare Streamlit-App, mit der ich LLM-Techniken von Chat über RAG bis zu Agenten lerne und vorzeige.
Alle Modelle laufen über [OpenRouter](https://openrouter.ai), jede Antwort zeigt Tokenverbrauch und Kosten.

**Demo:** _folgt_ (Streamlit Community Cloud, passwortgeschützt)

## Lokal starten

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # Key, Modell und Passwort eintragen
streamlit run app.py
```

Für die Entwicklung schaltet `SKIP_LOGIN=true` in `.env` die Passwortabfrage ab, `SHOW_INTERNAL_PAGES=true` zeigt
die interne Arbeits-Checkliste „Start“. Beide nie in den Cloud-Secrets setzen.

Tests: `pytest` · Secret-Scan und Linting vor jedem Commit: `pip install pre-commit && pre-commit install`

Auf Streamlit Cloud kommen dieselben Werte aus `.env.example` in die App-Secrets.

## Module

<!-- modules:start (automatisch aus docs/modules.yaml, nicht von Hand ändern) -->
| Modul | Phase | Status | Worum es geht |
|---|---|---|---|
| Chat | 2 | in Arbeit | Chatfenster mit gestreamten Antworten, einstellbarem System-Prompt und Anzeige von Tokens und Kosten. |
| Prompt-Labor | 2 | in Arbeit | Dieselbe Aufgabe mit vier Prompt-Techniken nebeneinander, inklusive Kosten und Antwortzeit. |
| Datenextraktion | 2 | in Arbeit | Macht aus Freitext wie Rechnungen oder E-Mails geprüfte, strukturierte Daten im JSON-Format. |
| RAG-Chatbot | 3 | in Arbeit | Beantwortet Fragen zu eigenen PDF-Dokumenten und nennt dabei Datei und Seite als Quelle. |
| Tool Use | 3 | in Arbeit | Das Modell ruft selbst Werkzeuge wie Rechner, Wetterdienst oder Websuche auf, jeder Aufruf ist sichtbar. |
| KI-Datenanalyst | 3 | geplant | Übersetzt Fragen in normaler Sprache in SQL, führt sie nur lesend aus und zeigt Tabelle, Diagramm und Code. |
| Workflow mit Freigabe | 4 | geplant | Eingehende Mails werden klassifiziert und beantwortet, ein Mensch gibt jeden Entwurf frei, alles wird protokolliert. |
| MCP-Server | 4 | geplant | Stellt Werkzeuge dieses Projekts über das Model Context Protocol für Claude Desktop und andere KI-Apps bereit. |
| Agent | 4 | geplant | Verfolgt ein vorgegebenes Ziel selbstständig in mehreren Schritten mit Planung, Werkzeugen und Abbruchregeln. |
| Evaluation | 5 | geplant | Misst die Antwortqualität mit einem Testset, einem Modell als Prüfer und aufgezeichneten Aufrufen. |
| Security | 5 | geplant | Zeigt Prompt Injection live mit und ohne Schutz, maskiert personenbezogene Daten und misst die Abwehrquote. |
<!-- modules:end -->

Die Tabelle entsteht aus `docs/modules.yaml` (pre-commit-Hook oder `python -m core.readme`).
Dort stehen auch Beschreibung, Akzeptanzkriterien und Arbeitsstand jedes Moduls; die interne Seite „Start“ zeigt sie an.
Die öffentliche Startseite „Home“ stellt die Module auf Englisch vor, Texte in `core/catalog.py`.

Aufbau: Logik in `core/`, Oberfläche in `app_pages/`, Tests in `tests/`, Plan in `docs/`, Testsets in `eval/`.

Entscheidungen und Erkenntnisse pro Modul: [`docs/erkenntnisse.md`](docs/erkenntnisse.md).
