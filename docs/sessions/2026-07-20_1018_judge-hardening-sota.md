# Session 2026-07-20 10:18 — Judge-Härtung & SOTA-Aufarbeitung (agentic-analyst)

## Kontext & Ziel
Überarbeitung der drei ML-Lab-Portfolio-Projekte auf Bewerbungsqualität. Grundlage:
das Council-Review `~/private/ml-lab/REVIEW.md` (12.07.) plus frische SOTA-Recherche
(Stand 2026-07-19) unter `~/private/ml-lab/research/2026-07-19-agentic-judge-sota.md`.
Ausführungsplan: `~/private/ml-lab/docs/superpowers/plans/2026-07-19-sota-upgrade.md`
(Phase 1 = dieses Repo, Tasks A1–A6). Ausführung per `subagent-driven-development`:
frischer Implementer pro Task, danach Spec-Review + Quality-Review.

Branch: **`fix/review-2026-07`** (aus `autopilot/work`). Working Tree ist **clean**.
Session wurde durch das API-Session-Limit unterbrochen (Reset 01:20 Europe/Berlin);
alle laufenden Subagents starben gleichzeitig.

## Ergebnis
Vier Commits auf `fix/review-2026-07`:

- **A1 — Claim⇔Value-Check** (`aaf5573`, `12caa07`, `c9a507c`): Der Judge prüft jetzt
  den natürlichsprachlichen `claim`-Text gegen den behaupteten `value` (`_claim_numbers`
  regext Zahlen inkl. `%`/„percent"/„per cent" → X und X/100, Tausendertrenner). Tötet
  Attack 3 aus REVIEW.md A-1. Strengthen-only (macht `verified` ggf. zu `contradicted`,
  schwächt nie ab). Guard für kategoriale/bool-Werte. **Spec- UND Quality-Review beide
  bestanden.** Count-vs-Rate ist als bewusste, getestete Limitation gepinnt (kein
  Magnitude-Guard, der würde „7500% churned"-Lügen durchlassen) — gehört in die
  README-„What the judge does NOT catch"-Sektion (A6).
- **A2 — Evidence-Plausibilität** (`627cb9a`): SQL-Evidence muss `FROM/JOIN data`
  referenzieren (word-boundary-Regex, Alias-Dodge `SELECT 0.75 AS data` wird erkannt),
  Python-Evidence muss `str(csv_path)` lesen. Verstoß → `unverified`
  (`evidence_does_not_touch_data`). Tötet Attacks 1+2. Attack 4 (Population-Switch) ist
  als dokumentierter Known-Gap gepinnt (VeriGraph arXiv 2606.16603). 90 Tests grün,
  Demo-Artefakte byte-identisch. **Spec-Review lief noch, als die Session starb** — der
  Reviewer hatte Demo-Byte-Identität bestätigt und prüfte gerade die Flow-Reihenfolge in
  `verify_finding` + 3 angepasste Alt-Fixtures. **Quality-Review steht noch aus.**

## Entscheidungen
- Reason-Strings leben als Präfix im Freitext-`detail` (kein eigenes Schema-Feld), weil
  `JudgedFinding` keins hat — konsistent mit der bestehenden Konvention.
- A2-Check läuft als statische Precondition VOR der Ausführung (keine Subprocess-Kosten
  für illegitime Evidence).
- 3 Alt-Fixtures in `test_judge.py` bekamen legitime Data-Referenzen (nur Evidence-
  Strings, keine Assertions) — ihre synthetische Evidence war strukturell attack-förmig.

## Offene Fragen
- A5 (Population-Provenance): Zeilenzahl pro Evidence sauber und billig ermitteln — der
  Plan skizziert CTE-Wrap für SQL bzw. `ROWS:`-Sentinel für Python; genaue Form beim
  Umsetzen entscheiden.

## To-dos
### Nico
1. Nichts Blockierendes. Optional später: echter LLM-Lauf (A5/WP-A5 im Plan) braucht
   `ANTHROPIC_API_KEY` in `.env` — das ist explizit „Needs Nico" und nicht Teil dieser Runde.

### Nächste Session (Agent)
1. **A2 abschließen**: Spec-Review (`627cb9a`) neu anstoßen/bestätigen, dann Quality-Review.
2. **A3** (Plan): `verify_baseline`-Toleranz `rel_tol=0.05` → `1e-6`; Docstring bei
   `judge.py:29-31` ehrlich umschreiben (Retrain ist kein unabhängiger Methoden-Check).
3. **A4**: `RESULT: <n>`-Sentinel für Python-Value-Extraktion + Fallback; bare `assert`
   durch `raise ValueError` ersetzen.
4. **A5**: Population-Provenance (`evidence_row_count`, `narrow_subset`-Note).
5. **A6**: Sandbox `RLIMIT_NPROC`; zweiter Inject-Toggle in `app.py`; README-Reframing +
   „What the judge does NOT catch"-Sektion (VeriGraph/DiscoveryBench/DABstep zitieren).

## Einstieg für die nächste Session
Branch `fix/review-2026-07`, Working Tree clean. Plan öffnen
(`~/private/ml-lab/docs/superpowers/plans/2026-07-19-sota-upgrade.md`, Phase 1) und
`subagent-driven-development` fortsetzen: zuerst A2-Reviews abschließen, dann A3→A6
sequenziell (alle fassen `src/agentic_analyst/judge.py` an → single-threaded, nicht
parallel). Gate pro Task: `uv run pytest -q && uv run ruff check .`.
