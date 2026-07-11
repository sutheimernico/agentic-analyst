"""Isolation tests for run_python, query_sql, read_schema.

No test in this file makes a real network call: the network tests assert
that an *attempt* raises the sandbox's RuntimeError before any packet would
go out.
"""

from pathlib import Path

import pytest

from agentic_analyst.tools import query_sql, read_schema, run_python

TELCO_CSV = Path(__file__).resolve().parent.parent / "data" / "telco-customer-churn.csv"

EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


# --- run_python -------------------------------------------------------------


def test_run_python_happy_path(tmp_path):
    code = "import pandas as pd\nprint('hello', len(pd.DataFrame({'a': [1, 2]})))"
    result = run_python(code, workdir=tmp_path)

    assert result.ok
    assert "hello 2" in result.stdout
    assert result.error is None


def test_run_python_timeout_kills_infinite_loop(tmp_path):
    result = run_python("while True:\n    pass", workdir=tmp_path, timeout_s=2)

    assert not result.ok
    assert "timed out" in result.error
    # generous slack for process spawn/kill overhead, but proves it didn't hang
    assert result.elapsed_s < 2 + 5


def test_run_python_memory_bomb_fails_without_killing_test_process(tmp_path):
    code = "x = bytearray(2 * 1024**3)\nprint('should not get here')"
    result = run_python(code, workdir=tmp_path, timeout_s=10, memory_mb=256)

    assert not result.ok
    assert "should not get here" not in result.stdout
    assert "MemoryError" in result.stderr


def test_run_python_network_socket_create_connection_blocked(tmp_path):
    code = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 80), 1)\n"
        "except Exception as e:\n"
        "    print(type(e).__name__, e)\n"
    )
    result = run_python(code, workdir=tmp_path)

    assert result.ok
    assert "RuntimeError" in result.stdout
    assert "network disabled in sandbox" in result.stdout


def test_run_python_network_urllib_blocked(tmp_path):
    code = (
        "import urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen('http://1.1.1.1', timeout=1)\n"
        "except Exception as e:\n"
        "    print(type(e).__name__, e)\n"
    )
    result = run_python(code, workdir=tmp_path)

    assert result.ok
    assert "RuntimeError" in result.stdout
    assert "network disabled in sandbox" in result.stdout


def test_run_python_stdout_truncated_at_10000_chars(tmp_path):
    code = "print('x' * 50000)"
    result = run_python(code, workdir=tmp_path)

    assert result.ok
    assert len(result.stdout) < 50000
    assert result.stdout.startswith("x" * 10_000)
    assert "truncated" in result.stdout


def test_run_python_exception_in_user_code_returns_traceback(tmp_path):
    code = "raise ValueError('boom')"
    result = run_python(code, workdir=tmp_path)

    assert not result.ok
    assert "ValueError" in result.stderr
    assert "boom" in result.stderr
    assert "Traceback" in result.stderr


def test_run_python_env_is_scrubbed(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SECRET_API_KEY", "super-secret-value")
    code = "import os\nprint(sorted(os.environ.keys()))"
    result = run_python(code, workdir=tmp_path)

    assert result.ok
    assert "FAKE_SECRET_API_KEY" not in result.stdout
    assert "super-secret-value" not in result.stdout


# --- query_sql ---------------------------------------------------------------


def test_query_sql_select_returns_rows():
    result = query_sql("SELECT customerID, Churn FROM data LIMIT 10", TELCO_CSV)

    assert result.ok
    assert "customerID" in result.stdout
    assert result.stdout.count("\n") >= 10  # header + separator + 10 rows


def test_query_sql_row_cap_respected_with_note():
    result = query_sql("SELECT * FROM data", TELCO_CSV, max_rows=5)

    assert result.ok
    data_lines = [
        line for line in result.stdout.splitlines() if line.startswith("|") and "---" not in line
    ]
    # first data line is the header, so 5 cap + 1 header line = 6
    assert len(data_lines) - 1 == 5
    assert "more rows" in result.stdout


def test_query_sql_trailing_semicolon_allowed():
    result = query_sql("SELECT 1 AS one;", TELCO_CSV)

    assert result.ok


@pytest.mark.parametrize(
    "query",
    [
        "INSERT INTO data VALUES (1)",
        "UPDATE data SET Churn = 'No'",
        "DROP TABLE data",
        "ATTACH 'evil.db' AS evil",
        "COPY data TO 'out.csv'",
    ],
)
def test_query_sql_rejects_ddl_dml_without_executing(query):
    result = query_sql(query, TELCO_CSV)

    assert not result.ok
    assert result.error is not None
    # must be rejected by our validator, not by DuckDB after execution
    assert "DuckDB error" not in result.error


def test_query_sql_rejects_multiple_statements():
    result = query_sql("SELECT 1; SELECT 2", TELCO_CSV)

    assert not result.ok
    assert "multiple statements" in result.error


def test_query_sql_rejects_empty_query():
    result = query_sql("   ", TELCO_CSV)

    assert not result.ok


def test_query_sql_rejects_file_read_via_read_csv_auto():
    # A bare SELECT can otherwise call read_csv_auto('/etc/passwd') and exfil
    # arbitrary files -- must be rejected by the validator before execution.
    result = query_sql("SELECT * FROM read_csv_auto('/etc/passwd') LIMIT 1", TELCO_CSV)

    assert not result.ok
    assert "DuckDB error" not in result.error
    assert "root:" not in result.stdout


def test_query_sql_rejects_glob():
    result = query_sql("SELECT * FROM glob('/etc/*')", TELCO_CSV)

    assert not result.ok
    assert "DuckDB error" not in result.error


def test_query_sql_rejects_with_prefix_dml_bypass():
    # `WITH x AS (...) INSERT ...` is a single statement whose leading keyword
    # is the allowed WITH, so the leading-keyword check passes -- the token
    # scan must still catch the hidden INSERT, and the rejection must come from
    # our validator, NOT from DuckDB after execution.
    result = query_sql("WITH x AS (SELECT 1) INSERT INTO data VALUES (1)", TELCO_CSV)

    assert not result.ok
    assert "DuckDB error" not in result.error
    assert "insert" in result.error.lower()


# --- read_schema ---------------------------------------------------------------


def test_read_schema_reports_all_columns_and_row_count():
    result = read_schema(TELCO_CSV)

    assert result.ok
    assert len(EXPECTED_COLUMNS) == 21
    assert "row_count: 7043" in result.stdout
    for col in EXPECTED_COLUMNS:
        assert col in result.stdout


def test_read_schema_total_charges_is_varchar_due_to_blanks():
    # TotalCharges has blank strings for tenure-0 customers, so DuckDB infers
    # VARCHAR rather than a numeric type -- this is the honesty test for the
    # agent later: it must not assume every charges column is numeric.
    result = read_schema(TELCO_CSV)
    schema_section = result.stdout.split("null_counts:")[0]
    total_charges_line = next(
        line for line in schema_section.splitlines() if "TotalCharges" in line
    )

    assert "VARCHAR" in total_charges_line


def test_read_schema_includes_null_counts_section():
    result = read_schema(TELCO_CSV)

    assert "null_counts:" in result.stdout


def test_read_schema_preview_row_count_matches_n_preview():
    result = read_schema(TELCO_CSV, n_preview=3)

    assert "preview (3 rows):" in result.stdout


def test_read_schema_rejects_non_integer_n_preview():
    # n_preview lands in a SQL LIMIT clause; a non-int value (e.g. from a
    # hallucinated tool call) must be rejected, not interpolated into SQL.
    result = read_schema(TELCO_CSV, n_preview="5); DROP TABLE data; --")

    assert not result.ok
    assert "n_preview must be an integer" in result.error


def test_read_schema_adversarial_column_name_no_stacked_execution(tmp_path):
    # read_schema builds null-count SQL by interpolating column names taken
    # from the (untrusted) CSV header. A header cell that breaks out of the
    # quoted identifier and stacks an ATTACH would create a file on disk.
    # Proper identifier quoting must neutralise it: the tool still reports the
    # (weirdly named) column, and no side-effect file appears.
    sentinel = tmp_path / "pwned.db"
    assert not sentinel.exists()

    evil_col = (
        f'''x" IS NULL THEN 0 END) AS z FROM (SELECT 1 AS x); '''
        f'''ATTACH '{sentinel}' AS ev; SELECT count(*) --'''
    )
    csv_path = tmp_path / "adversarial_header.csv"
    # CSV-quote the header cell by doubling embedded double-quotes
    csv_path.write_text('"' + evil_col.replace('"', '""') + '"\n1\n')

    result = read_schema(csv_path)

    assert not sentinel.exists()  # the stacked ATTACH must NOT have run
    assert result.ok
    # the adversarial string is treated as an ordinary column name
    assert "IS NULL THEN 0 END" in result.stdout
