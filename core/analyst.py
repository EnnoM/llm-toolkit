"""KI-Datenanalyst: Datensatz, nur lesende Datenbank und SQL-Schutz. Keine Oberfläche.

Der Datensatz ist ein erfundener Online-Shop (Kunden, Produkte, Bestellungen, Positionen),
den generate_sales() aus einem festen Startwert erzeugt. Damit gibt es keine Lizenzfragen,
und jeder Lauf liefert exakt dieselben Zahlen, was die Prüfung gegen von Hand geschriebenes
SQL erst möglich macht.

Zwei Schutzschichten, bewusst doppelt:
1. check_sql() lässt nur genau eine SELECT-Anweisung durch (DuckDBs eigener Parser, keine
   Textsuche nach „DELETE“, die man mit Kommentaren oder Groß-/Kleinschreibung umgeht).
2. Die Verbindung ist read_only und darf keine Dateien lesen oder schreiben. Selbst wenn
   Schicht 1 einen Fehler hätte, kann die Abfrage nichts verändern und nichts vom Server holen.
"""

import random
import tempfile
from datetime import date, timedelta
from functools import cache
from pathlib import Path

import duckdb
import pandas as pd

SEED = 42  # fester Startwert: gleiche Daten bei jedem Lauf
FIRST_DAY = date(2023, 1, 1)
LAST_DAY = date(2025, 12, 31)
MAX_ROWS = (
    1000  # mehr Zeilen zeigt die Tabelle nicht sinnvoll, und die Antwort bleibt klein
)

# Die Datei liegt im Temp-Ordner, nicht im Repo: Sie entsteht aus dem Code und wäre nur ein
# binäres Duplikat. Die Version im Namen erzwingt einen Neubau, wenn sich der Datensatz ändert.
DATASET_VERSION = 1
DB_PATH = Path(tempfile.gettempdir()) / f"llm-toolkit-shop-v{DATASET_VERSION}.duckdb"

COUNTRIES = {
    "Germany": ["Berlin", "Hamburg", "Munich", "Cologne", "Frankfurt"],
    "Austria": ["Vienna", "Graz", "Salzburg"],
    "Switzerland": ["Zurich", "Geneva", "Basel"],
    "France": ["Paris", "Lyon", "Marseille"],
    "Netherlands": ["Amsterdam", "Rotterdam", "Utrecht"],
}
# Deutschland ist der Heimatmarkt, deshalb häufiger
COUNTRY_WEIGHTS = [50, 15, 12, 13, 10]

# Kategorie -> (Produktnamen, Preisspanne in Euro)
CATALOG = {
    "Electronics": (
        [
            "Wireless Headphones",
            "Bluetooth Speaker",
            "Smartwatch",
            "E-Reader",
            "Webcam",
            "Power Bank",
            "Mechanical Keyboard",
            "USB-C Hub",
        ],
        (25, 260),
    ),
    "Home & Kitchen": (
        [
            "Coffee Grinder",
            "Chef's Knife",
            "Cast Iron Pan",
            "Electric Kettle",
            "Blender",
            "Cutting Board",
            "Tea Set",
            "Toaster",
        ],
        (15, 140),
    ),
    "Sports": (
        [
            "Yoga Mat",
            "Running Shoes",
            "Dumbbell Set",
            "Cycling Helmet",
            "Water Bottle",
            "Resistance Bands",
            "Tennis Racket",
            "Hiking Backpack",
        ],
        (10, 180),
    ),
    "Books": (
        [
            "Cookbook",
            "Travel Guide",
            "Crime Novel",
            "Science Fiction Novel",
            "Children's Book",
            "Biography",
            "Programming Handbook",
            "Poetry Collection",
        ],
        (8, 45),
    ),
    "Garden": (
        [
            "Garden Hose",
            "Pruning Shears",
            "Plant Pots (Set)",
            "Watering Can",
            "Garden Gloves",
            "Seed Collection",
            "Bird Feeder",
            "Solar Lights",
        ],
        (6, 90),
    ),
}
CHANNELS = ["web", "app", "marketplace"]
CHANNEL_WEIGHTS = [55, 30, 15]

# Mehr Bestellungen im November und Dezember (Weihnachten), weniger im Sommer
MONTH_FACTOR = {
    1: 0.8,
    2: 0.8,
    3: 0.9,
    4: 0.9,
    5: 1.0,
    6: 0.9,
    7: 0.8,
    8: 0.8,
    9: 1.0,
    10: 1.1,
    11: 1.6,
    12: 1.9,
}
YEAR_GROWTH = {2023: 1.0, 2024: 1.25, 2025: 1.45}  # der Shop wächst
ORDERS_PER_DAY = 5.0  # Mittel im Basisjahr bei Faktor 1

# Gartenartikel verkaufen sich im Frühling und Sommer, sonst kaum
GARDEN_MONTHS = {3: 2.0, 4: 3.0, 5: 3.0, 6: 2.5, 7: 2.0, 8: 1.5, 9: 1.0}


def generate_sales(seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Erzeugt die vier Tabellen des Shops, bei gleichem seed immer identisch."""
    rng = random.Random(seed)

    products = []
    for category, (names, (low, high)) in CATALOG.items():
        for name in names:
            products.append(
                {
                    "product_id": len(products) + 1,
                    "name": name,
                    "category": category,
                    "unit_price": round(rng.uniform(low, high), 2),
                }
            )

    countries = list(COUNTRIES)
    customers = []
    for customer_id in range(1, 401):
        country = rng.choices(countries, COUNTRY_WEIGHTS)[0]
        customers.append(
            {
                "customer_id": customer_id,
                "country": country,
                "city": rng.choice(COUNTRIES[country]),
                "signup_date": FIRST_DAY + timedelta(days=rng.randrange(0, 900)),
            }
        )

    orders, items = [], []
    day = FIRST_DAY
    while day <= LAST_DAY:
        expected = ORDERS_PER_DAY * MONTH_FACTOR[day.month] * YEAR_GROWTH[day.year]
        # Nur Kunden, die sich schon registriert haben, können bestellen
        active = [c for c in customers if c["signup_date"] <= day]
        weights = [
            GARDEN_MONTHS.get(day.month, 0.3) if p["category"] == "Garden" else 1.0
            for p in products
        ]
        # Mit Zufallsanteil, damit nicht jeder Tag gleich viele Bestellungen hat
        for _ in range(max(0, round(rng.gauss(expected, expected * 0.3)))):
            if not active:
                break
            order_id = len(orders) + 1
            orders.append(
                {
                    "order_id": order_id,
                    "customer_id": rng.choice(active)["customer_id"],
                    "order_date": day,
                    "channel": rng.choices(CHANNELS, CHANNEL_WEIGHTS)[0],
                }
            )
            # Jede Bestellung hat ein bis drei Positionen, meistens eine
            for product in rng.choices(products, weights, k=rng.choice([1, 1, 2, 3])):
                items.append(
                    {
                        "order_id": order_id,
                        "product_id": product["product_id"],
                        "quantity": rng.choices([1, 2, 3, 4], [70, 20, 7, 3])[0],
                        "unit_price": product["unit_price"],
                    }
                )
        day += timedelta(days=1)

    return {
        "customers": pd.DataFrame(customers),
        "products": pd.DataFrame(products),
        "orders": pd.DataFrame(orders),
        "order_items": pd.DataFrame(items),
    }


# Bedeutung der Spalten für den Prompt; Englisch, weil das Modell es liest
COLUMN_NOTES = {
    "customers.signup_date": "date the customer registered",
    "orders.channel": "sales channel",
    "order_items.unit_price": "price per unit in EUR at the time of the order",
    "products.unit_price": "current list price in EUR",
}


def build_database(path: Path = DB_PATH) -> Path:
    """Schreibt den Datensatz als DuckDB-Datei, falls sie noch nicht existiert."""
    if path.exists():
        return path
    tmp = path.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    with duckdb.connect(str(tmp)) as con:
        for name, frame in generate_sales().items():
            con.register("frame", frame)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM frame")
            con.unregister("frame")
    # Erst fertig bauen, dann umbenennen: Ein zweiter Prozess sieht nie eine halbe Datei
    tmp.replace(path)
    return path


@cache
def connect() -> duckdb.DuckDBPyConnection:
    """Die einzige Verbindung, über die Modell-SQL läuft: nur lesend, ohne Dateizugriff.

    lock_configuration verhindert, dass eine Abfrage per SET die Sperren wieder aufhebt.
    """
    return duckdb.connect(
        str(build_database()),
        read_only=True,
        config={"enable_external_access": False, "lock_configuration": True},
    )


class UnsafeSQLError(ValueError):
    """Die Abfrage ist keine einzelne SELECT-Anweisung und wird gar nicht erst ausgeführt."""


def check_sql(sql: str) -> str:
    """Gibt die Abfrage zurück, wenn sie genau eine SELECT-Anweisung ist, sonst UnsafeSQLError.

    WITH ... SELECT gilt auch als SELECT. Die Meldungen sind englisch, weil sie im Modul und
    beim Korrekturversuch auch beim Modell landen.
    """
    # Ein Syntaxfehler wirft hier duckdb.ParserException: ungefährlich, das Modell darf ihn
    # korrigieren. Gefährlich ist nur, was sich sauber parsen lässt, aber kein SELECT ist.
    statements = duckdb.extract_statements(sql)
    if len(statements) != 1:
        raise UnsafeSQLError(
            f"Expected exactly one SQL statement, got {len(statements)}."
        )
    if statements[0].type != duckdb.StatementType.SELECT:
        raise UnsafeSQLError(
            f"Only SELECT queries are allowed, not {statements[0].type.name}."
        )
    return sql


def run_query(sql: str, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    """Prüft und führt eine Abfrage aus; höchstens max_rows Zeilen."""
    check_sql(sql)
    cursor = (
        connect().cursor()
    )  # eigener Cursor pro Abfrage, sicher bei mehreren Sitzungen
    result = cursor.execute(sql)
    columns = [d[0] for d in result.description]
    return pd.DataFrame(result.fetchmany(max_rows), columns=columns)


def schema_description() -> str:
    """Tabellen und Spalten mit Typ als Text für den Prompt, dazu die Werte kurzer Kategorien.

    Ohne die Werte rät das Modell Schreibweisen („electronics“ statt „Electronics“) und die
    Abfrage findet nichts, ohne dass ein Fehler auftritt.
    """
    con = connect().cursor()
    lines = []
    tables = con.execute(
        "SELECT table_name FROM information_schema.tables ORDER BY table_name"
    ).fetchall()
    for (table,) in tables:
        rows = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        lines.append(f"Table {table} ({rows} rows):")
        columns = con.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = ? ORDER BY ordinal_position",
            [table],
        ).fetchall()
        for column, dtype in columns:
            line = f"  - {column} {dtype}"
            note = COLUMN_NOTES.get(f"{table}.{column}")
            if note:
                line += f", {note}"
            if dtype == "VARCHAR" and column not in ("name", "city"):
                values = [
                    v
                    for (v,) in con.execute(
                        f"SELECT DISTINCT {column} FROM {table} ORDER BY 1"
                    ).fetchall()
                ]
                line += f", values: {', '.join(values)}"
            lines.append(line)
    return "\n".join(lines)
