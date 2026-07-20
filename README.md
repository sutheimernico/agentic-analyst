# Agentic Analyst

> Give the agent a raw CSV, get back a reproducible analysis report — and a judge that recomputes every claimed value from its own evidence.

## Why this isn't just another GPT wrapper

An LLM agent that profiles a dataset, proposes features, and trains a baseline model is easy
to build and easy to fake: the model can hallucinate a churn rate, round a metric in its own
favor, or simply not check whether the number it just typed is true. Nothing in a typical
"agent writes a report" pipeline catches that.

This project adds a second, independent layer: **the judge**. Every claim (`Finding`) in the
report carries its own `evidence_sql_or_code` — the exact SQL or Python that produced it — plus
the single value it claims. The judge re-executes that evidence against the real data, extracts
the recomputed value, and compares it to the claim with a tight tolerance. It never asks another
LLM whether the text "looks right"; it recomputes the number and checks.

Be precise about what that mechanism actually guarantees, though: it **recomputes every claimed
value from its own evidence** — catching a misreported number (the claim disagrees with what its
own evidence produces) or evidence that never touches the data (a bare literal dressed up as a
query). It **cannot** catch evidence that is consistently fabricated — code that runs
successfully and genuinely queries the real data, just a different population than the claim's
text describes. See ["What the judge does NOT catch"](#what-the-judge-does-not-catch) below —
that section is a feature of this project's honesty, not a confession.

## The demo: honest report, then two kinds of caught lie

Run against the real Telco Customer Churn CSV (7,043 rows, 21 columns), the agent's report comes
back and the judge verifies it 3 for 3:

| verdict | claim | claimed | recomputed |
|---|---|---|---|
| ✓ verified | 26.5% of customers churned (imbalanced target) | `0.2654` | `0.26537...` |
| ✓ verified | 11 rows have a blank `TotalCharges` (tenure-0 customers, not yet billed) | `11` | `11` |
| ✓ verified | `LogisticRegression` baseline, ROC-AUC | `0.8105` | `0.8105` |

That's an honest report — good, but it doesn't *show* the judge doing its job, since every badge
is green. So the app has two demo toggles, each planting a different class of lie into the
churn-rate finding before re-judging.

**Toggle 1 — "🔬 Inject a planted false claim":** overstates the churn-rate claim to `0.75`
("almost everyone leaves!") while leaving its evidence honest. The claim text is clearly labeled
`[DEMO INJECTED LIE]` — it is never presented as a real number. What comes back:

| verdict | claim | claimed | recomputed (real) |
|---|---|---|---|
| ✗ contradicted | [DEMO INJECTED LIE] 75% of customers churned | `0.75` | `0.2654` |

The judge doesn't know the claim is fake, doesn't get told which finding to distrust — it just
re-runs the same `SELECT avg(...) FROM data` query the finding cites, gets the real ~26.5%, and
flags the mismatch.

**Toggle 2 — "🔬 Inject a *consistent* lie":** fabricates the claim, the claimed value, *and* the
evidence together — the evidence becomes `SELECT 0.75 AS churn_rate`, with no `FROM data` at all.
This is the attack the first toggle cannot demonstrate: a lie that carries its own proof, since
that "evidence" recomputes to exactly its own literal and would "match" whatever value is claimed
next to it. The judge catches this a *different* way — not by disagreeing with the value, but by
rejecting the evidence itself, before any number is ever compared:

| verdict | claim | claimed | recomputed |
|---|---|---|---|
| ? unverified | [DEMO INJECTED LIE, EVIDENCE FABRICATED TOO] 75% of customers churned | `0.75` | `n/a` |

Reason: `evidence_does_not_touch_data: SQL evidence never references the data view in a FROM/JOIN
clause -- it cannot have queried the real dataset`. Note the verdict is `unverified`, not
`contradicted` — the judge never claims to have disproven `0.75`, only that this evidence could
not have legitimately produced any number about the real data. That distinction is the honest
claim this project makes; see ["What the judge does NOT
catch"](#what-the-judge-does-not-catch) below for what *neither* toggle can demonstrate.

## Architecture

```
CSV ──► agent loop (Anthropic tool-use) ──► structured Report ──► judge (recompute + compare) ──► JudgedReport
         read_schema / query_sql / run_python                     query_sql / run_python (again, independently)
```

- **Agent loop** (`src/agentic_analyst/agent.py`) — a manual Anthropic tool-use loop against an
  `LLMClient` protocol: `AnthropicClient` (real SDK, imported lazily so the module is importable
  without a key) and `FakeLLM` (a deterministic, scripted stand-in used by every test and by this
  demo — it still calls the *real* tools against the *real* CSV; only "which tool to call next"
  is scripted). Tools: `read_schema` (DuckDB `DESCRIBE` + null counts + preview), `query_sql`
  (single-statement `SELECT`/`WITH`/`SUMMARIZE`/`DESCRIBE` over the CSV via DuckDB), `run_python`
  (sandboxed subprocess: pandas/duckdb/scikit-learn available). The loop ends when the model
  calls `submit_report` with a report that passes schema validation.
- **Report schema** (`src/agentic_analyst/report.py`) — `Report` = `Dataset` + `Finding[]` +
  `data_quality_issues` + `Baseline` + `caveats`. Each `Finding` is `{claim, evidence_sql_or_code,
  value}` — the evidence is what makes independent re-verification possible at all.
- **Judge** (`src/agentic_analyst/judge.py`) — see below; this is the differentiator.
- **App** (`app.py`, Streamlit) — runs the `FakeLLM` demo pipeline live, renders the report with
  inline verdict badges and per-finding provenance captions, and hosts the two demo-lie toggles.

### The judge is a deterministic structured recompute, not an LLM pass — deliberately

`judge.py` never calls an LLM. A free-text "LLM-as-judge" would first have to *parse* the
report's prose back into a checkable claim — an extra, fallible step. Because this project's
report is already structured (every `Finding` carries its own evidence and the single value it
claims), the judge skips extraction entirely: it **re-executes the evidence** through the same
tools the agent used (`query_sql`/`run_python`), pulls out the one recomputed number, and
compares it with `math.isclose` (numeric claims) or a normalized string compare (categorical
claims). Verdicts are three-valued, not two:

- `verified` — recompute succeeded and matches the claim.
- `contradicted` — recompute succeeded and disagrees beyond tolerance.
- `unverified` — the evidence couldn't be executed, didn't produce one comparable value, or
  named a model spec the judge doesn't know how to reproduce. This is never collapsed into
  `verified` or `contradicted` — "couldn't check" is its own outcome, and a `recomputed_value`
  of `None` always means exactly that.

`verify_baseline` goes further than trusting the claimed metric: it **re-trains** the same
`LogisticRegression` spec (features + `random_state`/`max_iter` parsed out of `baseline.model`) in
a fresh process and recomputes the metric from scratch (`rel_tol=1e-6`, since the retrain is fully
deterministic — a fixed seed reproduces byte-identically — with a `1e-9` absolute floor that only
ever matters for a metric near zero, where `rel_tol`'s own bound shrinks toward it). This catches a
*misreported* metric value — a number that disagrees with what the declared recipe actually
produces when run. It is explicitly **not** an independent methodology check: the retraining
template reproduces the *exact same* recipe as the agent's own baseline code — same `fillna`, same
split strategy, same seed, same model — so a methodological bug shared by both (e.g. a leakage
mistake baked into the template itself) reproduces identically on both sides and still comes back
`verified`. See "What the judge does NOT catch" below.

Tolerances are tight on purpose: the dominant claim type is a rate/proportion in `[0, 1]`, and
`math.isclose`'s OR logic (`rel_tol` *or* `abs_tol`) means a loose `abs_tol` (e.g. `0.5`) would
let a moderate lie about a proportion slip through (a claim of `0.75` vs. the real `0.2654` is
only `0.48` apart — comfortably under a loose bound). Both default to `0.01`, so real
disagreement is caught by `rel_tol` and `abs_tol` only rescues near-zero claims.

### Sandbox: defense-in-depth, not a hard boundary — stated honestly

`run_python` executes model-generated code in a subprocess with a memory rlimit (`RLIMIT_AS`), a
process-count rlimit (`RLIMIT_NPROC`), a wall-clock timeout, a scrubbed environment (no ambient
secrets), and a network kill-switch that monkeypatches `socket.socket`/`socket.create_connection`.
This is enough to survive "an LLM being sloppy" — an infinite loop, a memory bomb, a fork bomb, an
accidental network call — but it is **not** a security boundary against a determined adversary,
and the code says so rather than pretending otherwise. Documented, unpatched gaps:

- **No filesystem jail.** `cwd=workdir` only confines *relative* writes; any absolute path the
  OS user can read or write is still reachable — there is no chroot/namespace.
- **DNS and raw sockets bypass the kill-switch.** Only `socket.socket` and
  `socket.create_connection` are patched. `socket.getaddrinfo` still resolves DNS, and code that
  imports the raw `_socket` C extension directly can build a connection that never touches the
  patched names.
- **`RLIMIT_NPROC` is a soft, per-OS-user mitigation, not a jail.** `os.fork`/`subprocess`/
  `multiprocessing` spawning is now capped (closing the finding that unrestricted subprocess
  spawning defeats capability limits regardless of the surrounding sandbox tech), but the limit
  is counted system-wide for the real OS user, not scoped to just this sandbox's own process
  tree — there is no PID namespace/cgroup here, so the budget is shared with whatever else that
  user happens to be running.

If this ever ran on multi-tenant infrastructure or executed anything riskier than "an LLM
occasionally being careless," it would need a real sandbox (container, gVisor, Firecracker, or a
hosted code-execution service) — not what's here.

## What the judge does NOT catch

The judge's guarantee is real but narrow: it proves the report's *evidence* is honest about the
real data, not that the report's *methodology* is sound, and not that the evidence describes what
the claim's text says it describes.

- **A consistently fabricated population.** Evidence that legitimately runs `SELECT avg(x) FROM
  data WHERE Contract='Month-to-month'` while the claim's text says "ALL customers" passes both
  the evidence-plausibility check and the value compare — the recomputed number is genuinely,
  correctly derived from a real (if differently scoped) query against the real table. VeriGraph
  (arXiv [2606.16603](https://arxiv.org/html/2606.16603)) names this failure class precisely:
  **"executability can mask weak semantic transitions"** — code that runs successfully without
  actually supporting the claim it's attached to. This judge is execution-based by design (see
  above), so it inherits this exact limitation; closing it would require comparing the claim
  text's stated population against the query's actual filter predicates, a semantic check
  deliberately out of scope here (see `verify_finding`'s docstring in `judge.py` and the pinned,
  documented-known-gap test in `tests/test_judge.py`).
- **Whether a claim is worth making at all.** Nothing here checks scientific validity — whether a
  "finding" is meaningful, or whether the baseline's feature choices are sound. DiscoveryBench
  (arXiv [2407.01725](https://arxiv.org/abs/2407.01725)) benchmarks exactly that harder problem —
  judging whether an agent's *discovery* is correct, not just whether its evidence executed — and
  the best system on that benchmark solves only **~25%** of its tasks. Verifying claim⇔evidence
  consistency (what this judge does) and verifying that a claim is a *true, worthwhile discovery*
  (what DiscoveryBench measures) are different, both-hard problems; this project only attempts
  the first.
- **A shared methodological bug in the baseline recipe.** `verify_baseline` retrains the *exact
  same* declared recipe the agent used — see the judge design section above. A leakage mistake or
  other bug baked into that shared template reproduces identically on both sides and still comes
  back `verified`.

### The tolerance spec is an adopted recipe, not a guessed number

DABstep (arXiv [2506.23719](https://arxiv.org/html/2506.23719v1)) scores 450+ real
financial-analytics tasks with a hybrid deterministic checker — numeric extraction + tolerance,
normalized string/list comparison, explicitly *no LLM judge* — validated at 100% agreement with
human labels on a 75-case sample. This project adopts that same shape as its tolerance spec:
numeric claims compared with `math.isclose` (`rel_tol=0.01`/`abs_tol=0.01` for findings, where the
recompute has its own floating-point noise; `rel_tol=1e-6` dominant with a `1e-9` near-zero floor
for the baseline's deterministic retrain), categorical claims compared with a normalized
(stripped, casefolded) string match — the same two primitives DABstep's recipe uses, tuned to this
project's own claims rather than copied wholesale.

### How this would harden for multi-tenant use

Everything above defends against "an LLM occasionally being sloppy" or, for the judge, "an LLM
lying about the *number*, not the *evidence*." If this ever ran multi-tenant — analyzing untrusted
users' own CSVs, not just this project's own demo data — the sandbox would need a real isolation
boundary instead of a subprocess with rlimits: a gVisor or Firecracker microVM (full syscall/kernel
isolation, the accepted standard for untrusted-code execution platforms) or a WASM/Pyodide runtime
(`langchain-sandbox`-style capability isolation, at a cold-start and library-compatibility cost).
None of that is built here — this project's threat model is a single trusted user running a demo
against one committed CSV, and the code says so rather than pretending otherwise.

## No API key needed for the demo

The Streamlit app (`app.py`) and both demo scripts run entirely on `FakeLLM` — a deterministic,
scripted tool-call sequence that still executes the real sandboxed tools against the real CSV, so
every number on screen is genuinely computed, not invented. **No `ANTHROPIC_API_KEY` is required**
for any of this.

The real-Claude agent path (`AnthropicClient`, the actual tool-use loop against `claude-sonnet-5`)
is wired and importable but untested here by design — set `ANTHROPIC_API_KEY` in `.env` to try it.
**Needs Nico.** The CSV uploader in the app is disabled for the same reason: `FakeLLM`'s tool-call
script is scripted specifically for the telco schema and can't analyze an arbitrary upload; that
needs the real agent.

## Reproduce it

```bash
uv sync

# Agent demo: FakeLLM run over the real telco CSV -> results/report.json
uv run python scripts/demo_fake_run.py

# Judge demo: independently re-verify results/report.json -> results/judged_report.json
uv run python scripts/demo_judge_run.py

# App: report UI with verification badges + the two tamper toggles
uv run streamlit run app.py

# Gate
uv run pytest -q
uv run ruff check .
```

## Tech

Anthropic SDK tool-use loop · DuckDB (via `query_sql`) · a sandboxed `run_python` subprocess ·
scikit-learn (`LogisticRegression` churn baseline) · Streamlit (report UI) · uv + ruff + pytest.

## Dataset

Telco Customer Churn (`data/telco-customer-churn.csv`) — 7,043 customers, 21 columns, target
`Churn`. `TotalCharges` is stored as text with 11 blank (whitespace-only) values for tenure-0
customers who haven't been billed yet — a real data-quality gotcha the agent is expected to catch
and the judge re-verifies.

## Status

Milestones 1–6 (tools, agent loop, baseline modelling, judge layer, report UI, this write-up) are
done — see `PLAN.md` for the milestone-by-milestone detail and `AUTOPILOT_LOG.md` for the full
build history, including two design fixes the judge's own tests forced (a tolerance bug that let
moderate proportion lies through, and an invariant leak on non-numeric recomputes).
