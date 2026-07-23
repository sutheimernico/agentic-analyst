"""First real-LLM agent run (Task C4): a local Ollama model drives the loop.

Until now every committed report was produced by `FakeLLM` -- a hand-written
5-step state machine. This script answers the question "was a real model ever
at the wheel?" by running the full agent loop live against the committed
telco CSV with a local Ollama model (default qwen2.5:7b, native tool
calling, loopback only, no API key).

Decision tree (fixed by the plan, both ends are documented results):

- The model completes the loop with a valid `Report` within --attempts tries:
  write ``results/report_ollama.json`` and ``results/judged_report_ollama.json``
  (each ``{"meta": ..., "report"/"judged_report": ...}``, labeled with model +
  date + wall-clock; the judge verdict on a real model's claims is the point).
- All attempts fail (model never submits a valid report): write the honest
  negative result to ``results/ollama_failure_transcripts.json`` -- every
  request/response pair of every failed attempt, plus the per-attempt error.
  "A 7B model could not follow the tool protocol" is a real finding.

A transport error (Ollama not running / unreachable) is neither: it raises,
because it says nothing about the model's tool-calling ability.

Usage: uv run python scripts/run_real_llm.py [--model qwen2.5:7b]
       [--attempts 3] [--max-iters 20]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from agentic_analyst.agent import AgentIncompleteError, run_agent
from agentic_analyst.judge import verify_report
from agentic_analyst.ollama_client import DEFAULT_HOST, DEFAULT_MODEL, OllamaClient

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "data" / "telco-customer-churn.csv"
RESULTS_DIR = REPO_ROOT / "results"
REPORT_PATH = RESULTS_DIR / "report_ollama.json"
JUDGED_PATH = RESULTS_DIR / "judged_report_ollama.json"
FAILURE_PATH = RESULTS_DIR / "ollama_failure_transcripts.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-iters", type=int, default=20)
    # Small tool-calling models narrate their next step after an error
    # instead of emitting the tool call (qwen2.5:7b did, three runs in a
    # row); the nudge answers a text-only turn with an explicit "make the
    # tool call" user message. Visible in the transcript, counted in meta.
    parser.add_argument("--nudges", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    failures: list[dict] = []
    for attempt in range(1, args.attempts + 1):
        client = OllamaClient(model=args.model, host=args.host)
        print(f"--- attempt {attempt}/{args.attempts} ({args.model}) ---")
        start = time.monotonic()
        try:
            with TemporaryDirectory(prefix="agentic-analyst-ollama-") as tmp:
                report = run_agent(
                    client, CSV_PATH, Path(tmp), max_iters=args.max_iters, nudges=args.nudges
                )
                elapsed = time.monotonic() - start
                judged = verify_report(report, CSV_PATH, Path(tmp))
        except AgentIncompleteError as exc:
            elapsed = time.monotonic() - start
            print(f"attempt {attempt} failed after {elapsed:.0f}s: {exc}")
            failures.append(
                {
                    "attempt": attempt,
                    "error": str(exc),
                    "wall_clock_s": round(elapsed, 1),
                    "transcript": client.transcript,
                }
            )
            continue

        meta = {
            "model": args.model,
            "runner": "ollama /api/chat native tools, loopback",
            "date": date.today().isoformat(),
            "attempt": attempt,
            "attempts_allowed": args.attempts,
            "max_iters": args.max_iters,
            "nudges_allowed": args.nudges,
            "llm_calls": len(client.transcript),
            "wall_clock_s": round(elapsed, 1),
            "note": (
                "temperature=0, but a local LLM run is not bit-reproducible; "
                "this artifact is a labeled record of one real run, unlike the "
                "deterministic FakeLLM demo in results/report.json."
            ),
        }
        REPORT_PATH.write_text(
            json.dumps({"meta": meta, "report": report.to_dict()}, indent=2) + "\n"
        )
        JUDGED_PATH.write_text(
            json.dumps({"meta": meta, "judged_report": judged.to_dict()}, indent=2) + "\n"
        )
        print(f"Wrote {REPORT_PATH}")
        print(f"Wrote {JUDGED_PATH}")
        print(
            f"attempt {attempt} succeeded in {elapsed:.0f}s "
            f"({len(client.transcript)} LLM calls); judge: {judged.summary}"
        )
        for judged_finding in judged.findings:
            print(f"  - [{judged_finding.verdict}] {judged_finding.finding.claim}")
        print(f"  - [{judged.baseline.verdict}] baseline {judged.baseline.baseline.model}")
        return

    FAILURE_PATH.write_text(
        json.dumps(
            {
                "meta": {
                    "model": args.model,
                    "date": date.today().isoformat(),
                    "attempts": args.attempts,
                    "max_iters": args.max_iters,
                    "nudges_allowed": args.nudges,
                    "conclusion": (
                        f"{args.model} did not complete the tool-use loop with a "
                        f"valid report in {args.attempts} attempts -- honest "
                        "negative result; see per-attempt transcripts."
                    ),
                },
                "failures": failures,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"All {args.attempts} attempts failed. Wrote {FAILURE_PATH}")


if __name__ == "__main__":
    main()
