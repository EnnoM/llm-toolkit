# Erkenntnisse und Entscheidungen

Was gebaut wurde, welche Entscheidung warum fiel und was schiefging, pro Modul. Gedacht als
Vorbereitung aufs Vorstellungsgespräch: Laut Projektplan muss jede Zeile Code erklärbar sein.

Die Rohfassung steht in `docs/modules.yaml` (Feld `notizen`, mit Messwerten und Datum). Hier
steht die verdichtete Fassung: die Punkte, die man erklären können sollte. Nach jedem
abgeschlossenen Arbeitsschritt ergänzen.

---

## Grundgerüst (alle Module)

**Gebaut:** Passwortabfrage und Navigation in `app.py`, gemeinsamer LLM-Client in
`core/llm.py`, Secrets in `core/config.py`, Modul-Checkliste in `docs/modules.yaml` samt
Startseite und README-Tabelle.

**Entscheidungen**

- **OpenRouter statt einzelner Anbieter.** Ein Key, ein Client, Modelle von Google, OpenAI,
  Anthropic und Mistral vergleichbar. OpenRouter ist OpenAI-kompatibel, deshalb reicht das
  `openai`-SDK mit eigener `base_url`.
- **Kosten nicht selbst ausrechnen.** OpenRouter liefert sie pro Antwort im Feld `usage.cost`.
  Eigene Preistabellen veralten; die Preise in `core/models.py` dienen nur der Anzeige in der
  Modellauswahl.
- **Logik in `core/`, Oberfläche in `app_pages/`.** Alles, was an die API geht, entsteht in reinen
  Funktionen (z. B. `build_messages`). Die lassen sich ohne API und ohne Browser testen.
- **Tests ohne echte API-Aufrufe.** Ein nachgebauter Client gibt feste Antworten zurück;
  Seiten werden mit `streamlit.testing.v1.AppTest` durchgeklickt. Tests sind damit schnell,
  kostenlos und laufen auch ohne Key.
- **Passwortvergleich mit `hmac.compare_digest`.** Vergleicht in konstanter Zeit, sonst ließe
  sich aus der Antwortzeit ablesen, wie viele Zeichen schon stimmen (Timing-Angriff).
- **Secrets zuerst aus `st.secrets`, dann aus `.env`.** So läuft derselbe Code lokal und auf
  Streamlit Cloud. `SKIP_LOGIN=true` nur lokal.
- **Eine Quelle für den Projektstand.** `docs/modules.yaml` speist Startseite und
  README-Tabelle (pre-commit-Hook); Tests prüfen, dass Status und Arbeitsstand zusammenpassen.
- **Schrift selbst ausliefern.** Cutive liegt in `static/fonts/`, nichts wird von Google Fonts
  geladen (Datenschutz: sonst geht bei jedem Aufruf die IP-Adresse an Google).
- **Wiederholung bei Netzfehlern** macht das `openai`-SDK selbst (standardmäßig zwei
  Wiederholungen bei Verbindungsfehlern, 429 und 5xx). Eigenen Retry-Code gibt es nicht.

**Stolpersteine**

- Streamlit lädt geänderte Module aus `core/` ohne das Paket `watchdog` nicht neu. Nach
  Änderungen dort die App neu starten, „Neu laden“ auf der Startseite reicht nicht.
- Beim ersten Aufruf nach einem Serverstart zeigt die Sidebar kurz oder dauerhaft die
  Dateinamen aus `pages/` statt der Navigation („app, chat, extraction …“). Der Ordnername
  `pages/` ist für Streamlit ein Sonderfall (automatische Seiten), der mit `st.navigation`
  kollidiert. Behoben am 06.10.2026: Ordner in `app_pages/` umbenannt; danach zeigte die
  Sidebar auch direkt nach dem Serverstart bei vier Aufrufen die richtigen Titel.

---

## Chat

**Gebaut:** Chatfenster mit gestreamten Antworten, System-Prompt, Temperatur, vier Modellen
zur Auswahl, Tokens und Kosten pro Antwort und für die Sitzung.

**Entscheidungen**

- **Streaming** (`st.write_stream`): Die ersten Wörter erscheinen sofort, die Antwort wirkt
  schneller, obwohl sie gleich lang dauert.
- **Der ganze Verlauf geht bei jeder Anfrage mit.** Die API ist zustandslos, das Modell hat
  kein Gedächtnis. Folge: Die Eingabetokens und damit die Kosten wachsen mit jeder Nachricht.
  Die Seite zeigt das bewusst an.
- **Tokens und Kosten erst nach dem Stream auslesen.** OpenRouter schickt sie nur im letzten
  Stück des Streams.
- **Vier Modelle:** drei günstige und Claude Haiku 4.5 als rund zehnmal teurerer Vergleich.

**Was schiefging**

- **CSS wurde komplett ignoriert,** weil in einem Kommentar eine spitze Klammer vor einem
  Buchstaben stand. Streamlit prüft eingefügtes HTML mit DOMPurify und verwirft dann den
  ganzen Style.
- **Das Skript zum Scrollen brach beim zweiten Mal ab** („already been declared“): Es läuft
  bei jeder Nachricht erneut im selben Dokument. Lösung: Variablen in eine eigene Funktion.
- **Auf dem Handy öffnete die Seite unten bei den Kennzahlen** (23.09.2026). Ursache: Wegen
  des Eingabefelds in `st.bottom` scrollt Streamlit die ganze Seite ans Ende, und auf
  schmalen Bildschirmen stehen die Spalten untereinander. Lösung: Unter 640 px rückt der
  Verlauf per CSS ans Ende, direkt über das Eingabefeld.

---

## Prompt-Labor

**Gebaut:** Dieselbe Aufgabe mit Zero-shot, Few-shot, Rolle und Chain-of-thought parallel an
dasselbe Modell, Antworten nebeneinander mit Tokens, Kosten und Antwortzeit, dazu die
gesendeten Prompts.

**Entscheidungen**

- **Temperatur fest auf 0,** damit Unterschiede vom Prompt kommen und nicht vom Zufall.
- **Varianten laufen parallel** (Threads). Ein Lauf dauert so lange wie die langsamste
  Variante statt wie alle zusammen.
- **Die Rechenaufgabe verlangt „nur den Betrag“.** Heutige Modelle rechnen sonst ungefragt
  Schritt für Schritt, und der Unterschied zu Chain-of-thought verschwindet.
- **Die E-Mail-Aufgabe enthält zwei Probleme mit unterschiedlicher Dringlichkeit.** Erst
  dadurch muss das Modell abwägen, und die Techniken unterscheiden sich sichtbar.

**Ergebnis** (Vergleich am 23.09.2026 mit gpt-4.1-nano und claude-haiku-4.5)

| Aufgabe | Gewinner | Warum |
|---|---|---|
| Rechenaufgabe | Chain-of-thought | Jede Antwort, die sich an „nur den Betrag“ hält, war falsch |
| E-Mail-Triage | Few-shot | Ein Beispiel macht genau die Abwägung und das Format vor |
| Zusammenfassung | Few-shot | Übernimmt Länge und Ton der Beispiele |

Kernaussage: **Rechenaufgaben brauchen Chain-of-thought, Format- und Abwägungsaufgaben
Few-shot. Eine Rolle allein hilft kaum.** Das teurere Modell (Haiku, 7- bis 17-mal teurer
pro Lauf) war nur bei der Rechenaufgabe öfter richtig.

**Was schiefging (und bewusst so blieb)**

- Haiku kopierte bei Chain-of-thought den Platzhalter aus der Formatvorgabe wörtlich
  („Billing: X | Priority: Medium“). Nicht korrigiert, weil genau solche Fehler beim
  Vergleich sichtbar werden sollen.
- Ergebnisse sind nicht stabil: gpt-4.1-nano lag mit Chain-of-thought am 21.09. (direkt an
  der API) falsch, am 23.09. (über die Seite) richtig. Ob das am Zufall trotz Temperatur 0
  oder an einer inzwischen geänderten Vorlage liegt, ist offen. Ein einzelner Lauf ist also
  kein Beweis; dafür ist später das Modul Evaluation da.

---

## Datenextraktion

**Gebaut:** Freitext (Rechnung, Stellenanzeige, Terminanfrage) wird zu geprüftem JSON. Das
Ergebnis erscheint als Tabelle und JSON, alle Versuche sind einsehbar.

**Entscheidungen**

- **Pydantic-Modell als einzige Quelle.** Aus derselben Klasse entstehen das JSON-Schema für
  die API, die Prüfung der Antwort und die Feldübersicht auf der Seite. Keine drei Stellen,
  die auseinanderlaufen können.
- **Structured Outputs im Strict Mode.** Das Modell muss das Schema einhalten. Strict Mode
  verlangt, dass alle Felder unter `required` stehen; optionale Felder erlauben deshalb
  `null` als Typ, statt zu fehlen (`strict_json_schema`).
- **`provider.require_parameters` mitschicken.** Sonst darf OpenRouter an einen Anbieter
  weiterleiten, der das Schema stillschweigend ignoriert.
- **Fehlendes bleibt `null`.** Die Beispielrechnung hat keine Steuernummer, `tax_id` muss leer
  bleiben statt erfunden zu werden.
- **Zwei Arten von Prüfung, bewusst getrennt:**
  - *Formfehler* (falscher Typ, Datum als „03.09.2026“, erfundenes Zusatzfeld) macht das
    Modell. Es bekommt die Fehlermeldung zurück und darf höchstens zweimal korrigieren.
  - *Widersprüche im Dokument selbst* (Summe stimmt nicht, Gehaltsspanne vertauscht) werden
    nur als Hinweis angezeigt, ohne Korrekturversuch. Warum, steht unten.

**Was schiefging**

- **Das Modell hat Daten verfälscht, um die Prüfung zu bestehen** (23.09.2026, wichtigster
  Fund). Die Summenprüfung war zuerst Teil der Korrekturschleife. Test mit einer Rechnung,
  auf der die Summe falsch war (3,500.00 statt 3,474.80):
  1. Versuch 1 schrieb alles korrekt ab und scheiterte an der Summenprüfung.
  2. Versuch 2 änderte den Gesamtbetrag auf 3474.8, eine Zahl, die nirgends im Text steht.
     Die Prüfung war grün, die Daten falsch, und die eigentlich wichtige Information (der
     Lieferant verlangt 25.20 EUR zu viel) verschwand.

  Lehre: **Eine Korrekturschleife drängt das Modell, den Prüfer zufriedenzustellen, nicht die
  Wahrheit zu liefern.** Nur Fehler, die das Modell selbst gemacht hat, gehören zurück ans
  Modell. Fehler im Dokument gehören als Hinweis zum Menschen. Seit dem Umbau bleibt der
  Betrag 3500.00, darüber steht „Totals don't add up …“.
- **Beim Format `time` erfanden die Modelle eine Zeitzone** (gpt-4.1-nano:
  „14:30:00-04:00“, gemini: „14:30:00Z“). Lösung: Uhrzeit als Text mit festem Muster `HH:MM`.
  Lehre: Ein Standardformat im Schema lädt das Modell ein, es vollständig auszufüllen, auch
  mit Angaben, die im Text fehlen.
- **Die Eingabetokens enthalten das Schema offenbar nicht** (Rechnung: 242 Tokens, das Schema
  allein wäre länger). Ein erster Hilfetext behauptete das Gegenteil und wurde korrigiert:
  erst messen, dann erklären.

---

## RAG-Chatbot (lokal fertig, Deployment offen)

**Gebaut (Schritt 1 von 3, Indexieren ohne Embeddings):** PDFs einlesen, säubern und in
überlappende Abschnitte mit Datei und Seiten zerlegen; die Abschnitte lassen sich in der
Oberfläche ansehen.

**Gebaut (Schritt 2 von 3, Suche ohne Antwort, 05.10.2026):** Button „Create embeddings“
rechnet die Abschnitte über OpenRouter in Embeddings um und legt sie in Chroma ab; ein Suchfeld
zeigt zu einer Frage die ähnlichsten Abschnitte mit Datei, Seite und Ähnlichkeitswert. Die
Abschnittsliste aus Schritt 1 ist in einen aufklappbaren Bereich „Chunks“ gewandert.

**Gebaut (Schritt 3 von 3, Antworten, 05.10.2026):** Der Chat in der Mitte sucht zu jeder
Frage die ähnlichsten Abschnitte, gibt sie mit „Datei, Seite“ als Überschrift an das Modell und
streamt die Antwort. Darunter stehen Modell, Tokens und Kosten und aufklappbar die genutzten
Abschnitte („Retrieved passages“). Die Seite sieht aus wie der Chat (gleiche Bausteine, gleiches
CSS); Dokumente, Chunking und Modellauswahl stehen links, die Abschnittsliste öffnet sich als
Dialog.

**Entscheidungen**

- **In Schritten bauen und nach jedem hinschauen:** erst Abschnitte, dann Suche, dann
  Antworten. Eine falsche RAG-Antwort liegt meist an der Suche, nicht am Modell; wer die
  Schritte einzeln sieht, findet den Fehler.
- **Seiten erst zu einem Text verbinden, dann zerlegen,** und die Startposition jeder Seite
  merken. So bleibt ein Gedanke über die Seitengrenze zusammen, und trotzdem ist bekannt, auf
  welchen Seiten ein Abschnitt steht.
- **Start- und Endseite statt nur Startseite:** Rund ein Viertel der Abschnitte läuft über
  eine Seitengrenze. Mit nur der Startseite würde die Quelle oft auf die falsche Seite zeigen.
- **PDF-Seite statt gedruckter Seitenzahl,** weil man im Viewer dorthin springt.
- **Abschnitte enden an natürlichen Grenzen** (Absatz, Satzende, Leerzeichen) im letzten
  Viertel des Fensters, nie mitten im Wort.
- **Säubern vor dem Zerlegen:** Kopfzeilen, Seitenzahlen und Silbentrennung würden sonst in
  jedem Abschnitt stehen und die Suche verrauschen.
- **Embeddings über OpenRouter, Chroma nur im Speicher pro Sitzung** (Datenschutz: keine
  fremden Uploads sichtbar; Streamlit Cloud hat ohnehin keine dauerhafte Festplatte).
- **Embeddings nur auf Knopfdruck,** nicht bei jeder Änderung der Regler: Jeder neue Wert für
  Abschnittsgröße oder Überlappung ergibt neue Abschnitte, die wieder Geld kosten. Der Hinweis
  über dem Button schätzt die Kosten vorher (Zeichen durch 4).
- **Abschnitte aller Einstellungen bleiben in derselben Sammlung,** die Suche filtert auf die
  aktuelle Einstellung. Ein Wechsel zurück kostet so nichts.
- **Eine eigene Chroma-Sammlung pro Sitzung mit zufälligem Namen:** Test zeigte, dass zwei
  `EphemeralClient`s im selben Prozess dieselben Sammlungen sehen. Ohne eigenen Namen hätte
  jeder Besucher die Uploads aller anderen durchsucht.
- **Fester englischer Satz für „steht nicht drin“** („The documents do not cover this.“),
  statt das Modell frei formulieren zu lassen: Man erkennt ihn sofort, auch in Tests.
- **Nur die Abschnitte der aktuellen Frage gehen mit,** dazu die letzten sechs Nachrichten
  ohne ihre Abschnitte. Sonst würde jede Runde teurer, und alte Treffer mischten sich mit neuen.
- **Temperatur fest 0.2 statt Regler:** Die Antwort soll nah am Text bleiben, ein Regler würde
  zum Ausprobieren einladen, ohne dass es hier etwas zu lernen gibt.
- **Embeddings auch automatisch mit der ersten Frage,** zusätzlich zum Button: Eine Frage ist
  eine bewusste Handlung, anders als das Verschieben eines Reglers.
- **Gemeinsame Chat-Bausteine in `core/chat_ui.py`** (Frage-Box, graue Zeile, Scroll-Skript)
  und `chat.css` für beide Seiten, statt den Code zu kopieren. Ausnahme von der Regel „core
  ohne Oberfläche“, im Modul begründet.
- **`embed()` in `core/llm.py`** neben `complete()` und `stream()`: gleicher Client, gleiche
  Fehlermeldungen, Kosten aus demselben `usage.cost`-Feld.

**Was schiefging**

- **Wörter mit Leerzeichen mittendrin** („Zusti mmung“): liegt am PDF, pypdf kann es nicht
  zuverlässig reparieren. Offen, ob es die Suche stört.
- **Fehler aus Schritt 1 gefunden:** Bei Überlappung 0 stand das letzte Wort jedes Abschnitts
  trotzdem am Anfang des nächsten. Der Rücksprung zum Wortanfang griff auch dann, wenn der
  Schnitt schon genau an einem Wortende lag. Aufgefallen ist es erst im Suchtest mit kleinen
  Abschnitten; der Test für Größe und Überlappung prüfte nur die Mindestüberlappung, nicht
  das Zuviel. Jetzt mit eigenem Test.
- **Fehler aus Schritt 2: verwaiste Chroma-Sammlungen.** `st.session_state.setdefault(
  "rag_index", SearchIndex())` legt bei jedem Neuaufbau der Seite eine neue Sammlung an,
  weil Python das Argument immer auswertet, auch wenn der Schlüssel schon existiert. Bei
  jedem Klick wuchs der Speicher. Jetzt mit `if "rag_index" not in st.session_state`, und ein
  Test prüft, dass der Name gleich bleibt.
- **Veraltete Anzeige links:** Nach der ersten Frage stand dort noch „0 with embedding“, weil
  die linke Spalte vor der Frage gezeichnet wird. Die Zeile wird jetzt wie die Kennzahlen
  über einen Platzhalter erst am Ende gefüllt.
- **Prompt-Regeln können sich widersprechen:** „Antworte in der Sprache der Frage“ schlug bei
  gpt-4.1-nano den festen englischen Satz; erst die ausdrückliche Ausnahme half.
- **Eine Quellenangabe beweist nichts:** gpt-4.1-nano behauptete „München ist ein
  Oberzentrum“ und nannte eine echte Seite, auf der das nicht steht. Deshalb sind die
  genutzten Abschnitte aufklappbar; nur so lässt sich die Antwort prüfen.
- **Abschnitte, die mitten im Satz beginnen, führen zu falschen Antworten:** Gemini gab die
  Definition der Vorranggebiete falsch wieder, weil dem besten Treffer das Subjekt fehlte.
  Das Chunking aus Schritt 1 achtet am Ende auf Satzgrenzen, am Anfang nur auf Wortgrenzen.
- **Die Suche liefert immer Treffer,** auch zu Fragen, die nicht in den Dokumenten stehen
  (Hundesteuer: 0.41 gegen 0.62 bis 0.78 bei passenden Fragen). Englische Fragen an deutsche
  Dokumente funktionieren, aber mit deutlich niedrigeren Werten.
- **Ähnlichkeitswerte sind keine Prozentangaben:** Dieselben zwei Sätze ergeben je nach
  Embedding-Modell 0.455 oder 0.803. Feste Grenzwerte („unter 0.5 ist irrelevant“) sind
  deshalb gefährlich, nur die Reihenfolge zählt.

**Gebaut (Chunking nach Struktur, 06.10.2026):** Abschnitte beginnen an einem Satzanfang und
enden möglichst vor einer Überschrift. Nummerierte Überschriften werden erkannt; jeder
Abschnitt trägt den Pfad darüber („2. Oberzentren › 2.1 Regierungsbezirk Oberbayern“). Dieser
Pfad geht zusammen mit dem Dateinamen in das Embedding ein, steht beim Modell als eigene Zeile
„Section:“ unter der Quelle und ist im Chunk-Dialog und unter „Retrieved passages“ sichtbar.

**Entscheidungen**

- **Überschriften über die Nummernfolge erkennen,** nicht über Schriftgröße: pypdf liefert
  nur Text. Eine Zeile „Nummer + Großbuchstabe“ zählt nur, wenn die Nummer zur vorigen passt
  (2.1 nach 1.7, 2.1 nach 2). Das sortiert Datumszeilen („25. Juni 2012“) und Aufzählungen
  aus; Inhaltsverzeichnisse fallen an „…“ heraus.
- **Rückverweise „Zu 2.1.5“ als eigene Überschrift:** Das LEP stellt erst alle Ziele, dann
  alle Begründungen. Ohne Rückverweise erbte die ganze Begründung die letzte Ziel-Überschrift.
- **Dateiname im Embedding-Kopf:** clean_pages entfernt den Dokumenttitel als Kopfzeile;
  übrig bleibt er nur im Dateinamen („LEP Anhang 1 Zentrale Orte 2018“).
- **Überschrift beim Modell in eigener Zeile, nicht in der Quelle:** Der Prompt verlangt die
  Quelle „genau wie in der Überschrift“; stünde der Pfad dort, würde er mitzitiert.
- **Überlappung „ungefähr“ statt genau:** Der nächste Abschnitt beginnt am Satzanfang, der
  overlap Zeichen vor dem Ende am nächsten liegt, höchstens doppelt so weit zurück. Gibt es
  keinen (lange Sätze, Listen ohne Punkt), bleibt es beim Wortanfang. Hilfetext angepasst.

**Was schiefging**

- **Bessere Suche heißt noch nicht richtige Antwort:** Die Oberzentren-Liste wird jetzt
  gefunden, aber nur der Teil im besten Abschnitt; Schwaben fehlt, und das Modell sagt es
  nicht. Bei den Vorranggebieten steht das Subjekt jetzt im Abschnitt, gemini-2.5-flash-lite
  verwechselt Vorbehalts- und Vorranggebiete trotzdem. Das ist ein Modellfehler, kein
  Chunking-Fehler mehr.
- **Laufender Server zeigte alten Code** (AttributeError „Chunk has no attribute section“):
  der bekannte Stolperstein aus dem Grundgerüst, `core/` wird ohne watchdog nicht neu geladen.
- **Abkürzungen:** „z.B. Gymnasien“ sähe wie ein Satzanfang aus. Ausgeschlossen sind nur
  Abkürzungen aus Einzelbuchstaben; „bzw. Kommunen“ zählt weiter fälschlich als Satzanfang.

**Gebaut (Schalter für Satzgrenzen, 06.10.2026):** „Cut at sentences and headings“ links unter
den Reglern. Aus schneidet an der letzten Wortgrenze und beginnt am Wortanfang overlap Zeichen
vor dem Ende, also wie ein naives Chunking. Die Überschriften-Pfade bleiben in beiden Fällen.

- **Einstellung in der id und im Suchfilter** („800-150-s“ bzw. „-w“): Sonst hielte die
  Sammlung die Abschnitte beider Varianten für dieselben, und die Suche mischte sie.
- **Englische Fragen sind nicht genauer:** Der Eindruck kam von einer einzigen Frage, die den
  deutschen Fachbegriff enthielt. Mit fünf Fragenpaaren gewinnt mal Deutsch, mal Englisch;
  entscheidend ist, ob die Frage die Begriffe des Dokuments trifft.

**Gebaut (Fragenkatalog und Messung, 06.10.2026):** `eval/rag_questions.yaml` mit 10 Fragen
(Datei, Seite, erwartete Begriffe; zwei Fragen ohne Antwort in den Dokumenten) und
`core/rag_eval.py`, das die Beispiel-PDFs indexiert, jede Frage stellt und drei Dinge prüft:
richtige Seite unter den Treffern, erwartete Begriffe in der Antwort, richtige Quelle. Danach
Standard von 4 auf 6 Treffer erhöht (8/10 auf 10/10 richtig).

- **Prüfung über Begriffe statt über ein zweites Modell:** billig, schnell, wiederholbar. Dafür
  streng: „10 Hektar“ fiel zuerst durch, weil nur „10 ha“ erwartet war. LLM-as-a-Judge kommt
  im Modul Evaluation.
- **Suche und Antwort getrennt messen:** Ohne Satzgrenzen fand die Suche alles (8/8), die
  Antworten waren trotzdem falsch. Mit Satzgrenzen und 4 Treffern war es umgekehrt. Nur mit
  beiden Zahlen sieht man, wo der Fehler liegt.
- **Mehrfach laufen lassen:** Bei Temperatur 0.2 waren die Antworten in drei Läufen gleich;
  der erste Einzellauf mit dem fehlerhaften Katalog hatte trotzdem ein schiefes Bild gegeben.

**Was schiefging**

- **Der erste Lauf sprach gegen das neue Chunking** (Wortgrenzen besser als Sätze). Ursache:
  ein Fehler im Katalog (q06, Seite 10 fehlte) und zu wenige Treffer. Ohne Nachsehen hätte
  man die falsche Entscheidung getroffen.
- **gpt-4.1-nano hält sich nicht an feste Sätze** und behauptet wieder „München ist ein
  Oberzentrum“ mit echter Quelle. Für RAG ungeeignet, obwohl es gleich viel kostet wie Gemini.

**Fazit RAG-Chatbot**

RAG ist vor allem ein Suchproblem. Das Modell kann nur so gut antworten, wie die gefundenen
Abschnitte es erlauben, und die meisten Fehler entstanden vor dem Modell: Kopfzeilen im Text,
Abschnitte ohne Subjekt, Listen ohne ihre Überschrift, zu wenige Treffer. Die wirksamsten
Verbesserungen waren billig: Säubern, an Sätzen schneiden, den Überschriften-Pfad mit einbetten
und 6 statt 4 Treffer. Ein Fragenkatalog mit Seitenangaben machte Entscheidungen erst prüfbar;
ohne ihn hätte ein einzelner Lauf in die falsche Richtung gezeigt. Die Wahl des Modells zählt
trotzdem: Bei gleichen Treffern lag die Spanne zwischen 5 und 10 richtigen Antworten.
Quellenangaben machen Antworten prüfbar, beweisen aber nichts; deshalb bleiben die Abschnitte
sichtbar.

---

## Tool Use (lokal fertig, Deployment offen)

**Gebaut (07.10.2026):** Seite „Tool Use“ mit fünf Werkzeugen (`core/tools.py`), eigener
Tool-Schleife (`core/tool_use.py`) und sichtbaren Schritten unter jeder Antwort. `complete()` in
`core/llm.py` liefert jetzt auch die Aufrufwünsche des Modells (`tool_calls`).

**Entscheidungen**

- **Schleife selbst gebaut** statt OpenRouters Server-Tools oder einer Agenten-Bibliothek: Man
  sieht jede Runde, und es sind nur rund 60 Zeilen.
- **Jeder Fehler wird zum Werkzeugergebnis** `{"error": ...}`: kaputtes JSON, unbekanntes
  Werkzeug, Dienst nicht erreichbar, unerwartete Ausnahme. Das Modell kann darauf reagieren,
  die App stürzt nie ab.
- **Rechner über den Syntaxbaum (ast) statt eval():** Nur Zahlen, Rechenzeichen und eine
  Handvoll Funktionen. `__import__('os')…` und `9 ** 9 ** 9` werden abgelehnt.
- **Datumsrechnung im Datumswerkzeug** (until_date): Der Rechner kennt keine Kalender, und
  Modelle zählen Tage unzuverlässig.
- **Suchergebnisse als „Untrusted text“ markiert,** dazu eine Regel im System-Prompt.
- **Dokumentsuche mit einem gemeinsamen Index pro Serverprozess:** Es sind nur die öffentlichen
  Beispiel-PDFs, anders als die privaten Uploads im RAG-Modul.
- **Höchstens fünf Runden,** danach eine Meldung statt einer erzwungenen Antwort.
- **Live-Anzeige der Schritte** über einen Rückruf (on_step), damit man bei langsamen
  Werkzeugen (Websuche 2 bis 3 s, erste Dokumentsuche 7 s) sieht, was passiert.

**Was schiefging**

- **Das Modell rät das Jahr:** until_date „2024-12-24“ ergab „-652 Tage bis Heiligabend“.
  Lösung im Werkzeug (Monat-Tag als nächstes Vorkommen), nicht im Prompt.
- **Werkzeugbeschreibungen steuern die Argumente:** Ein Stichwort-Beispiel in der
  Beschreibung führte zu Stichwort-Suchen und schlechteren Treffern. Beschreibungen sind Teil
  des Prompts.
- **Regeln werden nicht immer befolgt:** Kleine Differenzen rechnet das Modell trotz „never
  calculate in your head“ selbst.
- **Grenze des Überschriften-Pfads:** Der Pfad gilt für den Anfang eines Abschnitts. Die
  Schwaben-Liste steht in einem Abschnitt mit „2.5 Regierungsbezirk Mittelfranken“ im Pfad;
  das kleine Modell las daraus einmal „nur Augsburg“.
- **Test-Falle:** `get_client` wird pro Runde aufgerufen. Ein Ersatz-Client, der jedes Mal neu
  entsteht, liefert immer die erste Antwort, und die Schleife lief in die Rundengrenze.

**Gebaut (Fragenkatalog für die Werkzeugwahl, 07.10.2026):** `eval/tool_questions.yaml` mit 12
Fragen und `core/tool_eval.py`. Geprüft wird pro Frage: richtige Werkzeuge, passende Argumente,
erwartete Begriffe in der Antwort (und verbotene nicht). Ergebnis: claude-haiku-4.5 12/12,
mistral-small-3.2 11/12, gemini-2.5-flash-lite 9/12, gpt-4.1-nano 7/12.

- **Werkzeugwahl ist bei allen Modellen gut,** die Unterschiede liegen danach: ob das Ergebnis
  richtig gelesen, die Sprache eingehalten und die Quelle verlinkt wird.
- **Das günstigste Modell (Mistral) schlägt hier das Standardmodell (Gemini).** Im RAG-Modul war
  es umgekehrt. Welches Modell „das beste“ ist, hängt von der Aufgabe ab, und nur Messen zeigt es.
- **Dieselben Abschnitte, andere Verpackung:** Gemini beantwortet die Schwaben-Frage im
  RAG-Modul richtig, über das Werkzeug (Abschnitte als JSON-Ergebnis) falsch.
- **Prüfungen über Begriffe sind blind für Zusätze:** „Kempten und Memmingen kommen vor“
  bestand auch eine Antwort, die fälschlich Augsburg dazunahm. Dafür bräuchte es eine Liste
  verbotener Begriffe oder einen Prüfer per LLM (Modul Evaluation).
- **Eine Testfrage kann am Verhalten vorbeigehen:** Bei einem Fantasieort rufen kluge Modelle
  das Werkzeug gar nicht auf. Gefragt wird jetzt, was wirklich zählt: kein erfundenes Wetter.

---

## Startseiten: öffentliches Home, internes Start

**Gebaut:** Die bisherige Startseite (Arbeits-Checkliste) heißt jetzt „Start“
(`app_pages/start.py`) und erscheint nur lokal. Neu ist „Home“ (`app_pages/home.py`) als
öffentliche Startseite auf Englisch: fertige Module als Karten mit Link, geplante ausgegraut
mit „WIP“. Texte, Icons und Seiten stehen in `core/catalog.py`.

**Entscheidungen**

- **Eigener Schalter `SHOW_INTERNAL_PAGES` statt Erkennung der Umgebung.** Ob die App auf
  Streamlit Cloud läuft, lässt sich nur über undokumentierte Merkmale raten. Ein Schalter in der
  lokalen `.env` ist eindeutig, und ohne ihn fehlt die Seite: sicher als Standard. Die Seite
  ist dann nicht nur versteckt, sondern gar nicht registriert, also auch per URL nicht erreichbar.
  Geheim ist ihr Inhalt trotzdem nicht, `docs/modules.yaml` liegt im öffentlichen Repo.
- **Englische Texte in `core/catalog.py`, nicht in `docs/modules.yaml`.** Die YAML bleibt laut
  Design-Richtlinien deutsch. Ein Test prüft, dass Katalog und YAML dieselben Module in derselben
  Reihenfolge haben und genau die geplanten Module keine Seite haben; ein neues Modul kann auf
  Home also nicht vergessen werden.
- **Eine Quelle für die Navigation.** `app.py` baut die Modulseiten aus dem Katalog, Titel und
  Icons stehen nicht mehr doppelt.
- **Ausgegraut mit `st.caption`, ohne eigenes CSS.** Die Hinweisschrift ist im Theme schon grau.

---

## Offene Punkte

- Deployment auf Streamlit Community Cloud für alle drei Module der Phase 2 (Projektregel:
  ein Modul ist erst fertig, wenn es online läuft). Bewusst zurückgestellt, der RAG-Chatbot
  wurde vorher begonnen.
