"""Tests für den KI-Datenanalysten: Datensatz und SQL-Schutz. Keine API-Aufrufe."""

import duckdb
import pytest

from core import analyst
from core.analyst import UnsafeSQLError, check_sql, connect, run_query

TABLES = ["customers", "orders", "order_items", "products"]


def row_counts() -> dict[str, int]:
    cursor = connect().cursor()
    return {
        t: cursor.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in TABLES
    }


def test_dataset_is_reproducible():
    first, second = analyst.generate_sales(), analyst.generate_sales()
    for name in first:
        assert first[name].equals(second[name])


def test_dataset_has_all_tables():
    counts = row_counts()
    assert counts["products"] == 40
    assert counts["customers"] == 400
    assert counts["orders"] > 5000
    assert counts["order_items"] > counts["orders"]


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE orders",
        "DELETE FROM orders",
        "UPDATE products SET unit_price = 0",
        "INSERT INTO products VALUES (99, 'x', 'Books', 1.0)",
        "CREATE TABLE x AS SELECT 1",
        "ATTACH 'other.duckdb' AS other",
        "COPY orders TO 'orders.csv'",
        "SET lock_configuration = false",
        # Zwei Anweisungen: ein harmloses SELECT darf kein DELETE hineinschmuggeln
        "SELECT 1; DELETE FROM orders",
        # Groß-/Kleinschreibung und Kommentare helfen nicht, der Parser sieht den Typ
        "/* SELECT */ dElEtE FROM orders",
    ],
)
def test_check_sql_rejects_everything_but_one_select(sql):
    before = row_counts()
    with pytest.raises(UnsafeSQLError):
        run_query(sql)
    assert row_counts() == before


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) FROM orders",
        "WITH t AS (SELECT * FROM products) SELECT category, count(*) FROM t GROUP BY 1",
        "select 1 -- a comment",
    ],
)
def test_check_sql_allows_selects(sql):
    assert check_sql(sql) == sql


def test_syntax_error_is_not_a_safety_error():
    # Tippfehler darf das Modell korrigieren, deshalb ein anderer Fehlertyp
    with pytest.raises(duckdb.ParserException):
        check_sql("SELEC * FROM orders")


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "DROP TABLE products",
        "UPDATE products SET unit_price = 0",
    ],
)
def test_connection_itself_is_read_only(sql):
    # Zweite Schicht: selbst ohne check_sql kann die Verbindung nichts verändern
    before = row_counts()
    with pytest.raises(duckdb.Error):
        connect().cursor().execute(sql)
    assert row_counts() == before


def test_select_cannot_read_files_from_the_server(tmp_path):
    secret = tmp_path / "secret.csv"
    secret.write_text("password\nhunter2\n")
    with pytest.raises(duckdb.Error):
        run_query(f"SELECT * FROM read_csv('{secret}')")


def test_settings_cannot_be_unlocked():
    with pytest.raises(duckdb.Error):
        connect().cursor().execute("SET enable_external_access = true")


def test_results_are_capped():
    assert len(run_query("SELECT * FROM order_items", max_rows=50)) == 50


def test_schema_lists_tables_and_category_values():
    text = analyst.schema_description()
    for table in TABLES:
        assert f"Table {table}" in text
    assert "Electronics" in text
    assert "marketplace" in text
