# Design-Richtlinien

Gilt für jede Seite der App. Umgesetzt in `.streamlit/config.toml`. Eigenes CSS nur dort, wo
Streamlit etwas nicht kann: dann klein, kommentiert, in einer eigenen Datei neben der Seite und
nur über Container mit `key` (Klasse `st-key-...`). Ausnahmen: `app_pages/chat.css`, das Chat und RAG Chatbot gemeinsam nutzen
(gleiche keys, Bausteine in `core/chat_ui.py`), und `app_pages/analyst.css`, das den Ask-Button
im AI Data Analyst so hoch macht wie das mitwachsende Eingabefeld, `app_pages/tool_use.css`,
das den Chatverlauf unter der Werkzeugleiste bis zum Eingabefeld reichen lässt, und `login.css`
neben `app.py`, das im Passwortfeld das zweite Auge des Browsers und den Enter-Hinweis ausblendet.

## Sprache

- **Öffentliche Seiten auf Englisch:** Die Login-Seite, die Startseite „Home“ und jede Modulseite
  in `app_pages/` sind komplett englisch: Titel, Beschriftungen, Hilfetexte, Platzhalter,
  Beispielfragen, Standard-Prompts und alle Meldungen, auch Fehlermeldungen aus `core/`, die im
  Modul erscheinen.
- Zahlen im Modul im englischen Format mit Dezimalpunkt (`0.000040 USD`).
- **Auf Deutsch bleiben:** die interne Seite „Start“ (Arbeits-Checkliste, nur lokal), `docs/modules.yaml` und
  Code-Kommentare. In `docs/modules.yaml` stehen Beschriftungen der Oberfläche trotzdem so,
  wie sie im Modul erscheinen, also englisch (z. B. Button „Clear chat“).

## Icons

- Keine bunten Emojis, weder in Titeln, Tabellen, Buttons noch als Favicon.
- Nur neutrale Material-Symbols im Format `:material/name:`, Übersicht unter
  [fonts.google.com/icons](https://fonts.google.com/icons?icon.set=Material+Symbols&icon.style=Rounded).
- Jede Seite bekommt in `st.Page(..., icon=...)` ein passendes Icon.

## Layout

- **Sidebar nur für die Navigation** zwischen den Modulen. Keine Einstellungen, Filter oder
  Buttons eines Moduls in der Sidebar.
- **Navigation nach Stufen:** Home allein oben ohne Überschrift, darunter die Module in
  Abschnitten „Level 3“, „Level 2“, „Level 1“, die neueste Stufe und das neueste Modul zuerst.
  Die Stufe steht als `level` in `core/catalog.py`. Home gruppiert die Kacheln genauso, dort mit
  Namen der Stufe („Level 2 · Knowledge and tools“); in der Sidebar wären die Namen abgeschnitten.
- **Modulseiten mit Einstellungen** nutzen die volle Breite (`layout="wide"`) und drei Spalten
  `st.columns([1, 2, 1], gap="large")` ohne sichtbare Trennlinien:
  links Einstellungen, in der Mitte der eigentliche Inhalt, rechts Kennzahlen
  (Kosten, Tokens in/out/gesamt, Anzahl Antworten).
- **Ausnahme Approval Workflow:** Die Seite ist ein kleines Mailprogramm und folgt dessen
  gewohntem Aufbau statt 1:2:1: Kopfzeile mit Titel und Kennzahlen als Textzeile, darunter
  einklappbare Einstellungen in zwei umrahmten Gruppen (Modell als ein Button pro Modell mit
  Preis im Button, Aktionen), dann das Mailprogramm in einem eigenen Rahmen: links die Mailliste
  mit Schnellfiltern, rechts die geöffnete Mail. Unten das Protokoll. Grund: Liste und geöffnete
  Mail sind dort der eigentliche Inhalt, Einstellungen sind Nebensache.
- **Chat-Nachrichten** schlicht wie in Claude Code: Fragen als gefüllte, rechts eingerückte
  Box, Antworten als reiner Text ohne Avatar, darunter eine graue Zeile mit Modell, Tokens und
  Kosten. Eingabefeld fest am unteren Bildschirmrand (`st.bottom`) in der Breite der mittleren
  Spalte. Nur der Verlauf scrollt; Titel und Seitenspalten bleiben stehen. Ein runder Pfeil unten
  im Verlauf zeigt, dass weiter unten noch etwas kommt, und nach jeder Frage und Antwort springt
  der Verlauf ans Ende.
- **Modellauswahl** immer nach Preis sortiert, günstigstes zuerst, mit dem Preis unter jedem
  Modell.
- **Hilfetexte** (das „?“ an Eingabefeldern) höchstens zwei Sätze: was die Einstellung bewirkt
  und worauf man achten sollte.

## Farben

| Element | Farbe |
|---|---|
| Hintergrund | `#FFFFFF` weiß |
| Schrift | `#000000` schwarz |
| Sidebar | `#C8D2DC` graublau, Schrift schwarz |
| Akzent (Buttons, aktive Elemente) | `#44586C` dunkles Graublau |
| Flächen (Eingabefelder, Tabellenkopf) | `#F2F4F7` hellgrau |
| Hinweise (`st.info`) | `#DDE4EB` hell-graublau, Schrift schwarz |
| Status-Badges (`:gray-badge[...]`) | `#E3E8ED`, Schrift schwarz |

- Nur heller Modus, kein Dark Mode.
- Rot, Gelb und Grün nur für Fehler, Warnungen und Erfolg (`st.error`, `st.warning`,
  `st.success`), nicht zur Dekoration.
  Im Approval Workflow markiert Rot zusätzlich Fälle, die ein Mensch entscheiden muss
  („Unclear“ in der Liste, `st.error` über dem Entwurf): Das ist eine Warnung, keine Dekoration.

## Schrift

- **Cutive** auf allen Geräten, damit die App überall gleich aussieht.
- Vorbild ist American Typewriter (ITC, 1974, Joel Kaden und Tony Stan), eine serifenbetonte
  Linear-Antiqua. Die darf aus Lizenzgründen nicht mit der App ausgeliefert werden; Cutive ist
  ein frei lizenzierter Ersatz im gleichen Schreibmaschinen-Stil (SIL Open Font License 1.1).
- Die Schrift liegt in `static/fonts/` samt Lizenz und wird von der App selbst ausgeliefert.
  Keine Schriften von Google Fonts oder anderen externen Servern laden (Datenschutz).
- Code bleibt in einer Monospace-Schrift.
