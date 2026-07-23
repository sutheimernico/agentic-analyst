"""Error injector for the quantified judge benchmark (Task C1).

Takes a *verified* base finding (an honest claim whose evidence genuinely
recomputes to its value against the real data) and plants a single, controlled
lie into it -- one attack class, one relative magnitude, one direction -- so
`scripts/judge_benchmark.py` can measure how reliably the judge catches each.

The attack classes mirror the ones enumerated in `tests/test_judge.py`:

- ``value_swap``          -- distort only the claimed ``value``; claim text and
                             evidence stay honest. Caught by the value<->evidence
                             recompute -> ``contradicted``.
- ``claim_mismatch``      -- distort only the number quoted in the claim text;
                             ``value`` and evidence stay honest. Caught by the
                             claim-text check -> ``contradicted``.
- ``fabricated_evidence`` -- replace the evidence with a bare literal (``SELECT
                             <v> AS <label>``, no ``FROM data``) that recomputes
                             to exactly the claimed value: a lie carrying its own
                             proof. Rejected before any value compare ->
                             ``unverified`` (magnitude-invariant by construction).
- ``alias_dodge``         -- the same self-proving lie, but aliasing the output
                             column as ``data`` (``SELECT <v> AS data``) to try to
                             fool a naive "the token ``data`` appears" check.
                             Also ``unverified`` (magnitude-invariant).

The fifth attack class, ``population_switch``, measures the judge's Task-C3
population-switch heuristic (evidence legitimately reads ``FROM data`` but
filters to a subpopulation the claim's text hides; see the flipped canonical
test in ``tests/test_judge.py``). Unlike the four classes above it is NOT
synthesized by rescaling ``base.value`` -- there is no dishonest *number* to
distort, the number is genuinely correct for the narrower population -- and it
has no magnitude dimension, so it is emitted ONCE per eligible base rather than
swept:

- ``BaseFinding`` carries two optional fields, ``subset_evidence_sql: str | None``
  and ``subset_value: float | None`` -- a real filtered query (e.g. ``... FROM
  data WHERE Contract='Month-to-month'``) and the value it genuinely computes.
- ``plant_population_switch(base, magnitude, direction)`` returns a ``Finding``
  whose *claim text* still describes the full population (rendered from
  ``base.claim_template``, e.g. "The overall churn rate is ...") while its
  evidence/value are the honest subset pair -- so the value<->evidence compare
  and the claim-text check both pass, and only the C3 heuristic catches it. It
  ignores ``magnitude``/``direction`` and returns ``None`` for a base without a
  subset query (skipped by ``generate_cases``).
- It is listed in ``SINGLE_CASE_CLASSES`` so ``generate_cases`` emits it once
  per eligible base (magnitude ``0.0``), not across the magnitude sweep. Its
  ``CATCH_VERDICT`` is ``unverified`` (the heuristic's downgrade), so a caught
  case is a real catch, not a re-measurement of the old ``verified`` gap.

Determinism: there is no RNG here. Case generation is an exhaustive factorial
over (base x class x magnitude x direction) for the swept classes plus one case
per eligible base for the single-case classes, enumerated in a fixed order, so
the benchmark's numbers are identical on every run without needing a seed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from agentic_analyst.report import Finding

# The four magnitudes the spec sweeps (Task C1: +-5%, +-20%, +-50%, +-200%).
DEFAULT_MAGNITUDES: tuple[float, ...] = (0.05, 0.20, 0.50, 2.00)


@dataclass(frozen=True)
class BaseFinding:
    """A single honest, judge-verifiable finding to plant lies into.

    ``value`` is the true quantity recomputed from the real data by the
    benchmark script; ``claim_template`` carries exactly one ``{n}`` slot the
    number is rendered into. ``as_int`` picks integer vs. 4-decimal rendering
    (a count vs. a rate/mean).
    """

    label: str
    evidence_sql: str
    claim_template: str
    as_int: bool
    value: float
    # Optional honest subpopulation pair, used ONLY by the population_switch
    # injector (both None -> that base produces no population_switch case): a
    # real filtered query and the value it genuinely computes, e.g.
    # ``... WHERE Contract='Month-to-month'`` and its churn rate. The full-
    # population ``claim_template`` is reused verbatim, so the planted lie is
    # "full-population framing over a subset number".
    subset_evidence_sql: str | None = None
    subset_value: float | None = None


@dataclass(frozen=True)
class PlantedCase:
    """One benchmark case: either a planted lie (``tampered=True``) or an
    untouched honest control (``tampered=False``, ``attack_class="clean"``)."""

    base_label: str
    attack_class: str
    magnitude: float
    direction: int
    tampered: bool
    finding: Finding


def _fmt(value: float, as_int: bool) -> str:
    """Render a number for a claim sentence or a SQL literal.

    Integers (counts) render plainly; everything else renders to 4 decimals
    with trailing zeros trimmed, enough precision that an honest number
    round-trips through the judge's claim-text check within tolerance.
    """
    if as_int:
        return str(round(value))
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def distort(value: float, magnitude: float, direction: int) -> float:
    """Scale ``value`` by a signed relative magnitude: ``value * (1 + dir*mag)``."""
    return value * (1.0 + direction * magnitude)


def _honest_claim(base: BaseFinding) -> str:
    return base.claim_template.format(n=_fmt(base.value, base.as_int))


def honest_finding(base: BaseFinding) -> Finding:
    """The untampered finding for a base -- honest claim, evidence, and value."""
    return Finding(
        claim=_honest_claim(base),
        evidence_sql_or_code=base.evidence_sql,
        value=base.value,
    )


def plant_value_swap(base: BaseFinding, magnitude: float, direction: int) -> Finding:
    """Distort only the claimed ``value``; claim text and evidence stay honest."""
    return Finding(
        claim=_honest_claim(base),
        evidence_sql_or_code=base.evidence_sql,
        value=distort(base.value, magnitude, direction),
    )


def plant_claim_mismatch(base: BaseFinding, magnitude: float, direction: int) -> Finding:
    """Distort only the number quoted in the claim text; ``value`` and evidence
    stay honest."""
    distorted = distort(base.value, magnitude, direction)
    return Finding(
        claim=base.claim_template.format(n=_fmt(distorted, base.as_int)),
        evidence_sql_or_code=base.evidence_sql,
        value=base.value,
    )


def plant_fabricated_evidence(base: BaseFinding, magnitude: float, direction: int) -> Finding:
    """Fabricate claim, value, and evidence together: a bare ``SELECT <v> AS
    <label>`` literal (no ``FROM data``) that recomputes to exactly its own
    claimed value -- a self-proving lie."""
    distorted = distort(base.value, magnitude, direction)
    literal = _fmt(distorted, base.as_int)
    return Finding(
        claim=base.claim_template.format(n=literal),
        evidence_sql_or_code=f"SELECT {literal} AS {base.label}",
        value=distorted,
    )


def plant_alias_dodge(base: BaseFinding, magnitude: float, direction: int) -> Finding:
    """The self-proving lie, but aliasing the output column as ``data`` to try
    to fool a naive "does the token ``data`` appear" check."""
    distorted = distort(base.value, magnitude, direction)
    literal = _fmt(distorted, base.as_int)
    return Finding(
        claim=base.claim_template.format(n=literal),
        evidence_sql_or_code=f"SELECT {literal} AS data",
        value=distorted,
    )


def plant_population_switch(base: BaseFinding, magnitude: float, direction: int) -> Finding | None:
    """Full-population claim text over an honest subpopulation number.

    Reuses ``base.claim_template`` verbatim (its full-population framing) but
    renders it with the honest ``subset_value`` and pairs it with the honest
    ``subset_evidence_sql``: the number is genuinely correct for the narrower
    population, so the value<->evidence compare and the claim-text check both
    pass and only the C3 heuristic catches it (via the hidden WHERE filter).
    Ignores ``magnitude``/``direction`` (there is no number to distort);
    returns ``None`` for a base without a subset pair, so it is skipped."""
    if base.subset_evidence_sql is None or base.subset_value is None:
        return None
    return Finding(
        claim=base.claim_template.format(n=_fmt(base.subset_value, base.as_int)),
        evidence_sql_or_code=base.subset_evidence_sql,
        value=base.subset_value,
    )


# An injector may return None to signal "no case for this base" (only
# plant_population_switch does, for a base without a subset pair); the four
# magnitude-swept injectors always return a Finding.
Injector = Callable[[BaseFinding, float, int], Finding | None]

# Registry: attack-class name -> injector. Order fixes the enumeration order in
# generate_cases (and therefore the artifact).
_INJECTORS: dict[str, Injector] = {
    "value_swap": plant_value_swap,
    "claim_mismatch": plant_claim_mismatch,
    "fabricated_evidence": plant_fabricated_evidence,
    "alias_dodge": plant_alias_dodge,
    "population_switch": plant_population_switch,
}
ATTACK_CLASSES: tuple[str, ...] = tuple(_INJECTORS)

# Classes with no magnitude dimension: generate_cases emits ONE case per
# eligible base (magnitude 0.0) instead of sweeping magnitude x direction.
SINGLE_CASE_CLASSES: frozenset[str] = frozenset({"population_switch"})

# The verdict a caught case of each class is meant to produce -- value/claim
# lies are `contradicted` (a number disagreed); fabricated/aliased evidence is
# `unverified` (the evidence could not have touched the data at all); a
# population switch is `unverified` (the C3 heuristic's population_mismatch
# downgrade). scripts/judge_benchmark.py reads this to report a per-case
# `mechanism_match` (caught AND verdict == the class's expected verdict), a
# diagnostic that a catch used the intended mechanism -- it never enters the
# catch-rate/recall numbers, which stay a bare `verdict != "verified"`.
CATCH_VERDICT: dict[str, str] = {
    "value_swap": "contradicted",
    "claim_mismatch": "contradicted",
    "fabricated_evidence": "unverified",
    "alias_dodge": "unverified",
    "population_switch": "unverified",
}


def _directions(magnitude: float) -> tuple[int, ...]:
    """Both signs for a sub-100% magnitude; only the positive side at >=100%,
    where the negative direction would flip the number's sign (``value*(1-2)``
    == ``-value``) rather than distort its magnitude -- and the judge's
    claim-text regex is sign-blind, so a negated number aliases back to the
    honest magnitude. Excluding it keeps every case a genuine magnitude
    distortion."""
    return (1, -1) if magnitude < 1.0 else (1,)


def generate_cases(
    bases: Sequence[BaseFinding],
    magnitudes: Sequence[float] = DEFAULT_MAGNITUDES,
    classes: Sequence[str] = ATTACK_CLASSES,
) -> list[PlantedCase]:
    """Exhaustive, deterministic factorial of benchmark cases: one honest
    control per base, then -- per base -- one planted case per (class,
    magnitude, direction) for the magnitude-swept classes, and one planted case
    for each ``SINGLE_CASE_CLASSES`` class that yields a Finding for that base
    (magnitude 0.0; a base that doesn't produce one, e.g. no subset query for
    population_switch, is skipped)."""
    cases: list[PlantedCase] = []
    for base in bases:
        if base.value == 0:
            # distort(0, m, d) == 0 for every magnitude/direction, so a zero
            # base silently degrades every "lie" back into the honest value --
            # fail loudly rather than emit undetectable non-lies.
            raise ValueError(
                f"base finding {base.label!r} has value 0; distortions of 0 are "
                "still 0 (not a lie) -- pick a non-zero base quantity"
            )
        cases.append(
            PlantedCase(
                base_label=base.label,
                attack_class="clean",
                magnitude=0.0,
                direction=0,
                tampered=False,
                finding=honest_finding(base),
            )
        )
        for attack_class in classes:
            injector = _INJECTORS[attack_class]
            if attack_class in SINGLE_CASE_CLASSES:
                finding = injector(base, 0.0, 0)
                if finding is not None:
                    cases.append(
                        PlantedCase(
                            base_label=base.label,
                            attack_class=attack_class,
                            magnitude=0.0,
                            direction=0,
                            tampered=True,
                            finding=finding,
                        )
                    )
                continue
            for magnitude in magnitudes:
                for direction in _directions(magnitude):
                    finding = injector(base, magnitude, direction)
                    assert finding is not None  # swept injectors always yield a Finding
                    cases.append(
                        PlantedCase(
                            base_label=base.label,
                            attack_class=attack_class,
                            magnitude=magnitude,
                            direction=direction,
                            tampered=True,
                            finding=finding,
                        )
                    )
    return cases
