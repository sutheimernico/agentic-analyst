"""Agent tools: sandboxed run_python, query_sql (DuckDB), read_schema.

These tools execute MODEL-GENERATED code and queries, so every input here is
treated as untrusted.

SECURITY HONESTY NOTE: `run_python`'s sandbox is defense-in-depth for
accidental/careless LLM-generated code (infinite loops, memory bombs, stray
network calls, obviously bad SQL) — it is NOT a security boundary against a
determined adversary. We have no user namespaces, no seccomp, no container,
no root here; a subprocess run as the same OS user with `python -I` and
rlimits can still be escaped by someone who wants to (e.g. via os.fork before
rlimits apply to children, reading other files the OS user can read, etc.).
If this ever runs on multi-tenant infra or executes anything other than
"an LLM occasionally being sloppy", wrap it in a real sandbox (container,
gVisor, firecracker, or a hosted code-execution service).
"""

from __future__ import annotations

import re
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb

_MAX_OUTPUT_CHARS = 10_000

# Prepended to every run_python submission. Disables the two ways stdlib code
# normally reaches the network; anything built directly on lower-level socket
# syscalls (or C extensions that bind their own sockets) is NOT covered — see
# module docstring.
#
# _DisabledSocket subclasses the real socket.socket (rather than replacing it
# with a plain function) because stdlib modules like ssl do
# `class SSLSocket(socket):` at import time — swapping in a function there
# breaks the import with an unrelated TypeError. Raising in __new__ still
# blocks every real connection attempt.
_NETWORK_KILL_SWITCH = """
import socket as _socket

_RealSocket = _socket.socket


class _DisabledSocket(_RealSocket):
    def __new__(cls, *args, **kwargs):
        raise RuntimeError("network disabled in sandbox")


def _disabled_create_connection(*args, **kwargs):
    raise RuntimeError("network disabled in sandbox")


_socket.socket = _DisabledSocket
_socket.create_connection = _disabled_create_connection
"""


@dataclass
class ToolResult:
    """Uniform result envelope returned by every tool in this module."""

    ok: bool
    stdout: str
    stderr: str
    elapsed_s: float
    error: str | None = None


def _truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... (truncated at {limit} chars)"


def _set_rlimits(memory_mb: int) -> None:
    """Run as preexec_fn in the child: cap address space (RLIMIT_AS)."""
    mem_bytes = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))


def run_python(
    code: str,
    workdir: Path,
    timeout_s: int = 30,
    memory_mb: int = 1024,
) -> ToolResult:
    """Execute model-generated Python in a subprocess sandbox.

    Isolation layers (all defense-in-depth, see module docstring):
    - `sys.executable -I` (isolated mode: ignores PYTHON* env vars and user
      site-packages) but still the venv interpreter, so pandas/duckdb/etc.
      resolve via that interpreter's own site-packages.
    - Minimal environment (PATH, HOME=workdir only) so no ambient secrets
      (API keys, tokens) leak into the child process.
    - cwd=workdir: confines relative file writes to the working directory
      (not a real filesystem jail — absolute paths still work).
    - RLIMIT_AS memory cap via preexec_fn, wall-clock timeout via
      subprocess.communicate(timeout=...).
    - Network kill-switch prelude: monkeypatches socket.socket and
      socket.create_connection to raise before user code runs.
    """
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(workdir),
    }
    full_code = _NETWORK_KILL_SWITCH + "\n" + code

    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-c", full_code],
            cwd=str(workdir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=lambda: _set_rlimits(memory_mb),
        )
    except OSError as exc:
        elapsed = time.monotonic() - start
        return ToolResult(ok=False, stdout="", stderr="", elapsed_s=elapsed, error=str(exc))

    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        elapsed = time.monotonic() - start
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        elapsed = time.monotonic() - start
        return ToolResult(
            ok=False,
            stdout=_truncate(stdout),
            stderr=_truncate(stderr),
            elapsed_s=elapsed,
            error=f"timed out after {timeout_s}s",
        )

    ok = proc.returncode == 0
    return ToolResult(
        ok=ok,
        stdout=_truncate(stdout),
        stderr=_truncate(stderr),
        elapsed_s=elapsed,
        error=None if ok else f"process exited with code {proc.returncode}",
    )


# A single SELECT/WITH/SUMMARIZE/DESCRIBE statement is allowed; everything
# else (DDL/DML, ATTACH, COPY, multi-statement) is rejected before it ever
# reaches DuckDB.
_ALLOWED_LEADING_KEYWORDS = ("SELECT", "WITH", "SUMMARIZE", "DESCRIBE")


def _validate_single_select(query: str) -> str | None:
    """Return an error message if the query is not a single read statement."""
    stripped = query.strip()
    if not stripped:
        return "empty query"

    body = stripped[:-1] if stripped.endswith(";") else stripped
    if ";" in body:
        return "multiple statements are not allowed (found ';' inside the query)"

    leading_word_match = re.match(r"^\s*([A-Za-z]+)", body)
    leading_word = leading_word_match.group(1).upper() if leading_word_match else ""
    if leading_word not in _ALLOWED_LEADING_KEYWORDS:
        return (
            f"only {'/'.join(_ALLOWED_LEADING_KEYWORDS)} statements are allowed, "
            f"got statement starting with '{leading_word or body[:20]}'"
        )
    return None


def _create_data_view(con: duckdb.DuckDBPyConnection, csv_path: Path) -> None:
    """Create view `data` over csv_path.

    DuckDB's `CREATE VIEW` cannot bind a prepared parameter, so the path is
    inlined as an escaped SQL string literal. csv_path is our own trusted
    filesystem path (not attacker-controlled query text), so this is not a
    SQL-injection surface -- it's only escaped so paths with a literal
    apostrophe don't break the statement.
    """
    escaped_path = str(csv_path).replace("'", "''")
    con.execute(f"CREATE VIEW data AS SELECT * FROM read_csv_auto('{escaped_path}')")


def _render_markdown_table(columns: list[str], rows: list[tuple]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body_lines = [
        "| " + " | ".join("" if v is None else str(v) for v in row) + " |" for row in rows
    ]
    return "\n".join([header, separator, *body_lines])


def query_sql(query: str, csv_path: Path, max_rows: int = 200) -> ToolResult:
    """Run a single read-only SQL query over `csv_path`, exposed as view `data`."""
    start = time.monotonic()

    validation_error = _validate_single_select(query)
    if validation_error is not None:
        return ToolResult(
            ok=False,
            stdout="",
            stderr="",
            elapsed_s=time.monotonic() - start,
            error=validation_error,
        )

    con = duckdb.connect(":memory:")
    try:
        _create_data_view(con, csv_path)
        cursor = con.execute(query)
        columns = [desc[0] for desc in cursor.description]
        all_rows = cursor.fetchall()
    except duckdb.Error as exc:
        return ToolResult(
            ok=False,
            stdout="",
            stderr=str(exc),
            elapsed_s=time.monotonic() - start,
            error=f"DuckDB error: {exc}",
        )
    finally:
        con.close()

    total = len(all_rows)
    shown_rows = all_rows[:max_rows]
    table = _render_markdown_table(columns, shown_rows)
    if total > max_rows:
        table += f"\n\n... ({total - max_rows} more rows)"

    return ToolResult(
        ok=True, stdout=table, stderr="", elapsed_s=time.monotonic() - start, error=None
    )


def read_schema(csv_path: Path, n_preview: int = 5) -> ToolResult:
    """Describe a CSV: columns, inferred dtypes, row count, null counts, preview rows."""
    start = time.monotonic()

    # n_preview lands directly in a SQL LIMIT clause below; coerce and range-check
    # it here so a non-int argument (e.g. a hallucinated tool call) can't become
    # a SQL injection point instead of a clean ToolResult(ok=False).
    try:
        n_preview = int(n_preview)
    except (TypeError, ValueError):
        return ToolResult(
            ok=False,
            stdout="",
            stderr="",
            elapsed_s=time.monotonic() - start,
            error="n_preview must be an integer",
        )
    if n_preview < 0:
        return ToolResult(
            ok=False,
            stdout="",
            stderr="",
            elapsed_s=time.monotonic() - start,
            error="n_preview must be >= 0",
        )

    con = duckdb.connect(":memory:")
    try:
        _create_data_view(con, csv_path)
        describe_cursor = con.execute("DESCRIBE data")
        describe_rows = describe_cursor.fetchall()
        describe_columns = [desc[0] for desc in describe_cursor.description]

        column_names = [row[0] for row in describe_rows]

        row_count = con.execute("SELECT count(*) FROM data").fetchone()[0]

        null_count_exprs = ", ".join(
            f'sum(CASE WHEN "{col}" IS NULL THEN 1 ELSE 0 END) AS "{col}"' for col in column_names
        )
        null_counts_row = con.execute(f"SELECT {null_count_exprs} FROM data").fetchone()

        preview_cursor = con.execute(f"SELECT * FROM data LIMIT {n_preview}")
        preview_columns = [desc[0] for desc in preview_cursor.description]
        preview_rows = preview_cursor.fetchall()
    except duckdb.Error as exc:
        return ToolResult(
            ok=False,
            stdout="",
            stderr=str(exc),
            elapsed_s=time.monotonic() - start,
            error=f"DuckDB error: {exc}",
        )
    finally:
        con.close()

    sections = [
        f"row_count: {row_count}",
        "",
        "schema:",
        _render_markdown_table(describe_columns, describe_rows),
        "",
        "null_counts:",
        _render_markdown_table(column_names, [tuple(null_counts_row)]),
        "",
        f"preview ({len(preview_rows)} rows):",
        _render_markdown_table(preview_columns, preview_rows),
    ]

    return ToolResult(
        ok=True,
        stdout="\n".join(sections),
        stderr="",
        elapsed_s=time.monotonic() - start,
        error=None,
    )
