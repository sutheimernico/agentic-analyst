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

The one attack class the judge is documented NOT to catch -- population-switch
(evidence legitimately reads ``FROM data`` but filters to a subpopulation the
claim's text hides; see the known-gap test in ``tests/test_judge.py``) -- is
deliberately absent here: a follow-up task (C3) adds a heuristic for it and
extends this benchmark with that class. It is a small, local extension, but NOT
a drop-in ``Injector``: unlike the four classes here, it cannot be synthesized
by rescaling ``base.value`` (there is no dishonest *number* to distort -- the
number is genuinely correct for the narrower population) and it has no magnitude
dimension. The honest shape C3 needs (documented here, deliberately not built
yet -- see ``_INJECTORS`` for the registration side):

- ``BaseFinding`` gains two optional fields, ``subset_evidence_sql: str | None``
  and ``subset_value: float | None`` -- a real filtered query (e.g. ``... FROM
  data WHERE Contract='Month-to-month'``) and the value it genuinely computes.
- ``plant_population_switch(base, magnitude, direction)`` returns a ``Finding``
  whose *claim text* still describes the full population ("... of ALL
  customers") while its evidence/value are the honest subset pair. It ignores
  ``magnitude``/``direction`` (like ``fabricated_evidence``/``alias_dodge`` do),
  and skips bases without a subset query. It registers in ``_INJECTORS`` +
  ``CATCH_VERDICT`` exactly like the others, and ``generate_cases`` then picks
  it up automatically -- its expected verdict is ``verified`` (the documented
  gap: an execution-based judge cannot catch it), so it measures the gap rather
  than a catch.

Determinism: there is no RNG here. Case generation is an exhaustive factorial
over (base x class x magnitude x direction), enumerated in a fixed order, so the
benchmark's numbers are identical on every run without needing a seed.
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


Injector = Callable[[BaseFinding, float, int], Finding]

# Registry: attack-class name -> injector. Order fixes the enumeration order in
# generate_cases (and therefore the artifact). C3 adds "population_switch" here
# and its expected verdict ("verified" -- the documented gap) in CATCH_VERDICT;
# that injector needs the two extra BaseFinding fields sketched in the module
# docstring (it can't be synthesized by rescaling value like these four).
_INJECTORS: dict[str, Injector] = {
    "value_swap": plant_value_swap,
    "claim_mismatch": plant_claim_mismatch,
    "fabricated_evidence": plant_fabricated_evidence,
    "alias_dodge": plant_alias_dodge,
}
ATTACK_CLASSES: tuple[str, ...] = tuple(_INJECTORS)

# The verdict a caught case of each class is meant to produce -- value/claim
# lies are `contradicted` (a number disagreed), fabricated/aliased evidence is
# `unverified` (the evidence could not have touched the data at all).
# scripts/judge_benchmark.py reads this to report a per-case `mechanism_match`
# (caught AND verdict == the class's expected verdict), a diagnostic that a
# catch used the intended mechanism -- it never enters the catch-rate/recall
# numbers, which stay a bare `verdict != "verified"`.
CATCH_VERDICT: dict[str, str] = {
    "value_swap": "contradicted",
    "claim_mismatch": "contradicted",
    "fabricated_evidence": "unverified",
    "alias_dodge": "unverified",
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
    control per base, then one planted case per (base, class, magnitude,
    direction)."""
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
            for magnitude in magnitudes:
                for direction in _directions(magnitude):
                    cases.append(
                        PlantedCase(
                            base_label=base.label,
                            attack_class=attack_class,
                            magnitude=magnitude,
                            direction=direction,
                            tampered=True,
                            finding=injector(base, magnitude, direction),
                        )
                    )
    return cases
