"""Agent tools: sandboxed run_python, query_sql (DuckDB), read_schema.

These tools execute MODEL-GENERATED code and queries, so every input here is
treated as untrusted.

SECURITY HONESTY NOTE: `run_python`'s sandbox is defense-in-depth for
accidental/careless LLM-generated code (infinite loops, memory bombs, stray
network calls, obviously bad SQL) — it is NOT a security boundary against a
determined adversary. We have no user namespaces, no seccomp, no container,
no root here; a subprocess run as the same OS user with `python -I` and
rlimits can still be escaped by someone who wants to. Known, unpatched gaps:
- The network kill-switch only monkeypatches `socket.socket` and
  `socket.create_connection`. DNS resolution via `socket.getaddrinfo` is NOT
  blocked, and code can reach the raw C-level socket module (`import _socket`)
  to build a connection that never touches the patched names.
- Any file the OS user can read is readable (no filesystem jail).
- `os.fork`/`subprocess`/`multiprocessing` child-spawning inside the sandbox
  is now capped by `RLIMIT_NPROC` (see `_set_rlimits` below) rather than left
  wide open -- the tianpan.co finding that motivated this: "allowing the
  sandbox to spawn arbitrary subprocesses largely defeats capability
  restrictions regardless of the surrounding sandbox technology". This is a
  SOFT mitigation, not a jail: RLIMIT_NPROC counts processes/threads for the
  real OS *user*, system-wide, not a count scoped to this sandbox's own
  process tree (no PID namespace/cgroup here). A fork bomb inside the
  sandbox is bounded and fails with a clean `OSError` once the ceiling is
  hit, but that ceiling is a budget shared with every other process this OS
  user happens to be running at the time -- not an isolated allowance.
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


def _set_rlimits(memory_mb: int, max_procs: int) -> None:
    """Run as preexec_fn in the child: cap address space (RLIMIT_AS) and the
    number of processes/threads the real OS user may hold (RLIMIT_NPROC) --
    see the module docstring for why the latter is a soft, per-user
    mitigation rather than a hard jail. Both soft AND hard limits are set to
    the same value: an unprivileged process can only ever raise its own
    limit back up to its hard limit (CAP_SYS_RESOURCE is required to go
    higher), so pinning both closes off model-generated code calling
    `resource.setrlimit` on itself to undo this.
    """
    mem_bytes = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    resource.setrlimit(resource.RLIMIT_NPROC, (max_procs, max_procs))


def run_python(
    code: str,
    workdir: Path,
    timeout_s: int = 30,
    memory_mb: int = 1024,
    max_procs: int = 512,
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
    - RLIMIT_AS memory cap and RLIMIT_NPROC process/thread-count cap via
      preexec_fn, wall-clock timeout via subprocess.communicate(timeout=...).
    - Network kill-switch prelude: monkeypatches socket.socket and
      socket.create_connection to raise before user code runs.

    OPENBLAS_NUM_THREADS/OMP_NUM_THREADS are pinned to 1 in the child env
    because OpenBLAS (pulled in transitively by numpy/scikit-learn) reserves
    per-thread virtual address space scaled to the host core count when
    first imported. On a many-core box that reservation blows past the
    RLIMIT_AS cap and any numpy/sklearn work dies with "OpenBLAS error:
    Memory allocation still failed after 10 retries" -- long before actual
    working-set memory is a concern. Setting these here protects ALL
    model-generated code at the sandbox level; the model neither knows about
    the quirk nor should have to.

    `max_procs` (default 512) is the RLIMIT_NPROC ceiling: the maximum
    number of processes/threads the sandboxed child (or anything it forks)
    may cause the real OS user to hold at once. It is NOT scoped to this
    subprocess's own descendants -- see module docstring -- so the value has
    to clear two bars: (a) comfortably above this machine's ordinary ambient
    process count for the user running the sandbox (measured at ~20 on this
    dev machine while writing this), so legitimate work (a `subprocess.run`
    call, a joblib worker) doesn't get starved by pre-existing load, and (b)
    small enough that an actual fork bomb inside the sandbox is stopped
    within a bounded, small number of processes rather than being left
    effectively unlimited (the previous state: this OS user's own `ulimit
    -u` here is 63201). 512 clears both with wide margin on this single-user
    box; a genuinely multi-tenant or heavily loaded deployment should
    recompute this from its own ambient process count rather than reusing
    this constant.
    """
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(workdir),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
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
            preexec_fn=lambda: _set_rlimits(memory_mb, max_procs),
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

# Bare tokens that must never appear anywhere in a query_sql submission, even
# inside an otherwise-allowed SELECT/WITH. This blocks (a) file / external
# table functions that read arbitrary paths (`read_csv_auto('/etc/passwd')`,
# `glob('/etc/*')`), (b) catalog / extension manipulation, and (c) DML that can
# hide behind a leading WITH (`WITH x AS (SELECT 1) INSERT ...`), which the
# leading-keyword check alone does not catch. The scan is deliberately
# conservative: it matches these as whole identifier tokens anywhere in the
# query, so a column or string literal that happens to equal one of these words
# (e.g. a column literally named "insert") is rejected too. That false positive
# is acceptable -- query_sql's only contract is to read the `data` view.
_FORBIDDEN_TOKENS = frozenset(
    {
        # arbitrary file / external table access
        "read_csv",
        "read_csv_auto",
        "read_parquet",
        "read_json",
        "read_json_auto",
        "read_ndjson",
        "read_text",
        "glob",
        # catalog / extension manipulation
        "attach",
        "detach",
        "install",
        "load",
        "copy",
        # DML that can hide after a leading WITH
        "insert",
        "update",
        "delete",
        "merge",
    }
)

_IDENTIFIER_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


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

    tokens = {t.lower() for t in _IDENTIFIER_TOKEN.findall(body)}
    forbidden = tokens & _FORBIDDEN_TOKENS
    if forbidden:
        return (
            f"disallowed keyword(s) in query: {', '.join(sorted(forbidden))} "
            "-- file access, catalog changes and DML are not permitted; "
            "only read the `data` view"
        )
    return None


def _quote_ident(name: str) -> str:
    """Quote a SQL identifier, doubling any embedded double-quote.

    Column names come from the untrusted CSV header (via DESCRIBE); without
    this, a header cell like `x" ...; ATTACH '...' --` would break out of the
    quoted identifier and inject stacked SQL. Doubling embedded `"` keeps the
    whole thing a single valid identifier.
    """
    return '"' + name.replace('"', '""') + '"'


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


def parse_markdown_table(markdown: str) -> tuple[list[str], list[list[str]]]:
    """Parse a `| a | b |` table as emitted by query_sql/read_schema into
    (header, data_rows), skipping the `---` separator line.

    Inverse of `_render_markdown_table`. Lives here (not in agent.py or
    judge.py) because tools.py owns the markdown-table format both other
    modules read back -- a single parser keeps their idea of "a 1x1 result"
    from drifting apart.
    """
    lines = [ln for ln in markdown.strip().splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        raise ValueError(f"expected a markdown table, got: {markdown!r}")
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    data_lines = lines[2:] if "---" in lines[1] else lines[1:]
    rows = [[c.strip() for c in ln.strip("|").split("|")] for ln in data_lines]
    return header, rows


def single_value(markdown: str) -> str:
    """Extract the one cell of a single-column, single-row query result."""
    header, rows = parse_markdown_table(markdown)
    if len(header) != 1 or len(rows) != 1:
        raise ValueError(f"expected a 1x1 result table, got: {markdown!r}")
    return rows[0][0]


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
            f"sum(CASE WHEN {_quote_ident(col)} IS NULL THEN 1 ELSE 0 END) "
            f"AS {_quote_ident(col)}"
            for col in column_names
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
