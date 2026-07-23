"""Pure aggregation stats for the judge benchmark (Task C1).

Extracted from `scripts/judge_benchmark.py` so the non-trivial precision /
recall / FPR / mechanism-match math is unit-tested (the script stays thin
wiring: build bases -> generate cases -> judge -> aggregate -> render). No I/O,
no matplotlib, no data access here -- just counting over an already-judged
mapping of cases to verdicts.
"""

from __future__ import annotations

from collections import defaultdict

from agentic_analyst.planting import (
    ATTACK_CLASSES,
    CATCH_VERDICT,
    DEFAULT_MAGNITUDES,
    PlantedCase,
)


def _rate(numer: int, denom: int) -> float:
    """Round ``numer/denom`` to 4 decimals; a zero denominator yields ``0.0``
    (an empty cell reads as "nothing caught", never a division error)."""
    return round(numer / denom, 4) if denom else 0.0


def aggregate(per_case: dict[int, tuple[PlantedCase, str]]) -> dict:
    """Reduce judged cases to the committed benchmark metrics.

    ``per_case`` maps a case index to ``(case, verdict)`` as produced by the
    benchmark run. A case is "caught" iff its verdict is not ``verified``.
    Tampered cases split into true positives (caught) / false negatives
    (missed); clean controls split into false positives (wrongly caught) / true
    negatives. Precision/FPR are global (a false positive can only come from a
    clean control, which carries no attack magnitude/class), while recall is
    also broken out per class, per magnitude, and per (class, magnitude) cell.
    """
    tp = fn = fp = tn = 0
    # Of the caught tampered cases, how many used the mechanism the attack class
    # is meant to trip (CATCH_VERDICT) rather than some incidental verdict. This
    # is a diagnostic only -- it never enters catch-rate/recall, which stay a
    # bare `verdict != "verified"`.
    mechanism_matches = 0
    by_class_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [caught, n]
    by_mag_counts: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    by_cell: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    missed: list[dict] = []

    for case, verdict in per_case.values():
        caught = verdict != "verified"
        if case.tampered:
            if caught:
                tp += 1
                if verdict == CATCH_VERDICT[case.attack_class]:
                    mechanism_matches += 1
            else:
                fn += 1
                missed.append(
                    {
                        "base_label": case.base_label,
                        "attack_class": case.attack_class,
                        "magnitude": case.magnitude,
                        "direction": case.direction,
                    }
                )
            by_class_counts[case.attack_class][0] += int(caught)
            by_class_counts[case.attack_class][1] += 1
            by_mag_counts[case.magnitude][0] += int(caught)
            by_mag_counts[case.magnitude][1] += 1
            cell = by_cell[case.attack_class][f"{case.magnitude:g}"]
            cell[0] += int(caught)
            cell[1] += 1
        else:
            if caught:
                fp += 1
            else:
                tn += 1

    overall = {
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "precision": _rate(tp, tp + fp),
        "recall": _rate(tp, tp + fn),
        "false_positive_rate": _rate(fp, fp + tn),
    }
    mechanism_match = {
        "caught": tp,
        "used_intended_mechanism": mechanism_matches,
        "rate": _rate(mechanism_matches, tp),
    }
    by_class = {
        cls: {"caught": c[0], "n": c[1], "recall": _rate(c[0], c[1])}
        for cls, c in ((k, by_class_counts[k]) for k in ATTACK_CLASSES)
    }
    by_magnitude = {
        f"{mag:g}": {"caught": by_mag_counts[mag][0], "n": by_mag_counts[mag][1],
                     "recall": _rate(*by_mag_counts[mag])}
        for mag in DEFAULT_MAGNITUDES
    }
    by_class_magnitude = {
        cls: {
            f"{mag:g}": {
                "caught": by_cell[cls][f"{mag:g}"][0],
                "n": by_cell[cls][f"{mag:g}"][1],
                "catch_rate": _rate(*by_cell[cls][f"{mag:g}"]),
            }
            for mag in DEFAULT_MAGNITUDES
        }
        for cls in ATTACK_CLASSES
    }
    return {
        "overall": overall,
        "mechanism_match": mechanism_match,
        "by_class": by_class,
        "by_magnitude": by_magnitude,
        "by_class_magnitude": by_class_magnitude,
        "missed_cases": missed,
    }
