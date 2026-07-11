# AUTOPILOT LOG — agentic-analyst

- 2026-07-10: M1 done — `run_python` (subprocess sandbox: rlimit-mem, wall-clock timeout, network kill-switch, scrubbed env), `query_sql` (DuckDB, single-statement SELECT/WITH/SUMMARIZE/DESCRIBE only), `read_schema` (DuckDB DESCRIBE + null counts + preview). 23 tests green, ruff clean.
- 2026-07-11: M1 security fixes (review NEEDS_FIXES → resolved) — quote CSV column names in `read_schema` (identifier injection), token-blocklist file functions / catalog ops / DML in `query_sql` (arbitrary file read via `read_csv_auto`/`glob`, WITH-prefix DML bypass), honest docstring gaps (`getaddrinfo`/raw `_socket`). 27 tests green, ruff clean.
