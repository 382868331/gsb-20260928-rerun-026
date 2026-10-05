from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any, Iterator

SUPPORT = "support"
REFUTE = "refute"
INSUFFICIENT = "insufficient"


def normalize_observations(
    observations: list[dict[str, Any]],
    candidate_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    normalized = []
    for observation in observations:
        linked = tuple(dict.fromkeys(str(candidate_id) for candidate_id in observation.get("candidate_ids", ())))
        unknown = [candidate_id for candidate_id in linked if candidate_id not in candidate_ids]
        if unknown:
            raise ValueError(f"review observation references unknown candidates {unknown}")
        normalized.append(
            {
                "observation_id": observation.get("observation_id", observation.get("scan_index")),
                "scan_index": observation.get("scan_index"),
                "scan_id": observation.get("scan_id"),
                "source_id": str(observation.get("source_id", "unknown")),
                "relevance_known": bool(observation.get("relevance_known", True)),
                "candidate_ids": linked,
            }
        )
    return normalized


def conflict_pairs(
    candidate_ids: tuple[str, ...],
    explicit_pairs: tuple[tuple[str, str], ...] | None = None,
) -> tuple[tuple[str, str], ...]:
    if explicit_pairs is None:
        return tuple((left, right) for left, right in combinations(candidate_ids, 2))
    pairs = []
    for left, right in explicit_pairs:
        if left not in candidate_ids or right not in candidate_ids:
            raise ValueError("ambiguity pair references a candidate outside the review scope")
        if left == right:
            raise ValueError("a candidate cannot be mutually exclusive with itself")
        ordered = tuple(sorted((left, right)))
        if ordered not in pairs:
            pairs.append(ordered)
    return tuple(pairs)


def observation_discriminates(observation: dict[str, Any], left: str, right: str) -> bool:
    if not observation.get("relevance_known", True):
        return False
    linked = set(observation["candidate_ids"])
    return (left in linked) != (right in linked)


def covered_pairs(
    selected: tuple[dict[str, Any], ...],
    pairs: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        pair
        for pair in pairs
        if any(observation_discriminates(observation, pair[0], pair[1]) for observation in selected)
    )


def uncovered_pairs(
    selected: tuple[dict[str, Any], ...],
    pairs: tuple[tuple[str, str], ...],
    refuted_candidate_ids: set[str] | None = None,
) -> tuple[tuple[str, str], ...]:
    refuted = refuted_candidate_ids or set()
    live_pairs = tuple(pair for pair in pairs if pair[0] not in refuted and pair[1] not in refuted)
    covered = set(covered_pairs(selected, live_pairs))
    return tuple(pair for pair in live_pairs if pair not in covered)


def _source_balance(selected: tuple[dict[str, Any], ...]) -> tuple[int, int]:
    counts = Counter(str(item["source_id"]) for item in selected)
    if not counts:
        return (0, 0)
    return (-max(counts.values()), len(counts))


def _subsets(
    observations: list[dict[str, Any]],
    budget: int,
) -> Iterator[tuple[dict[str, Any], ...]]:
    for size in range(1, min(budget, len(observations)) + 1):
        for indices in combinations(range(len(observations)), size):
            yield tuple(observations[index] for index in indices)


def _exact_subsets(
    observations: list[dict[str, Any]],
    budget: int,
) -> Iterator[tuple[dict[str, Any], ...]]:
    size = min(budget, len(observations))
    if size < 1:
        return
    for indices in combinations(range(len(observations)), size):
        yield tuple(observations[index] for index in indices)


def _option(
    name: str,
    selected: tuple[dict[str, Any], ...],
    pairs: tuple[tuple[str, str], ...],
    refuted_candidate_ids: set[str] | None = None,
) -> dict[str, Any]:
    covered = covered_pairs(selected, pairs)
    uncovered = uncovered_pairs(selected, pairs, refuted_candidate_ids)
    unknown = [item["observation_id"] for item in selected if not item.get("relevance_known", True)]
    return {
        "strategy": name,
        "review_count": len(selected),
        "observation_ids": [item["observation_id"] for item in selected],
        "covered_conflicts": [list(pair) for pair in covered],
        "uncovered_conflicts": [list(pair) for pair in uncovered],
        "source_counts": dict(Counter(str(item["source_id"]) for item in selected)),
        "source_balance": _source_balance(selected),
        "source_imbalance": -_source_balance(selected)[0],
        "unknown_relevance_observations": unknown,
        "limitations": (
            ["source relevance is unknown for selected observations; their discrimination power is not counted"]
            if unknown
            else []
        ),
    }


def compare_review_plans(
    candidate_ids: tuple[str, ...] | list[str],
    observations: list[dict[str, Any]],
    review_budget: int,
    ambiguity_pairs: tuple[tuple[str, str], ...] | None = None,
    feedback: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    if review_budget < 1:
        raise ValueError("review budget must be at least one")
    candidate_ids = tuple(str(candidate_id) for candidate_id in candidate_ids)
    pairs = conflict_pairs(candidate_ids, ambiguity_pairs)
    normalized = normalize_observations(observations, candidate_ids)
    refuted = {
        str(item["candidate_id"])
        for item in feedback
        if item.get("verdict") == REFUTE and str(item.get("candidate_id")) in candidate_ids
    }
    options = [_option("maximum_conflict_coverage", selected, pairs, refuted) for selected in _subsets(normalized, review_budget)]
    balanced_options = [_option("balanced_source_coverage", selected, pairs, refuted) for selected in _exact_subsets(normalized, review_budget)]
    if not options:
        empty = _option("maximum_conflict_coverage", (), pairs, refuted)
        balanced = dict(empty)
        balanced["strategy"] = "balanced_source_coverage"
        options = [empty]
        balanced_options = [balanced]

    def conflict_key(option):
        return (
            -len(option["covered_conflicts"]),
            -option["source_imbalance"],
            option["review_count"],
            option["observation_ids"],
        )

    def balanced_key(option):
        return (
            -option["source_balance"][0],
            option["source_balance"][1],
            -len(option["covered_conflicts"]),
            option["review_count"],
            option["observation_ids"],
        )

    maximum = dict(min(options, key=conflict_key))
    balanced = dict(min(balanced_options, key=balanced_key))
    maximum["selected"] = True
    balanced["selected"] = balanced["observation_ids"] == maximum["observation_ids"]
    return {
        "candidate_ids": list(candidate_ids),
        "review_budget": review_budget,
        "ambiguity_pairs": [list(pair) for pair in pairs],
        "plans": [maximum, balanced],
        "recommended_strategy": "maximum_conflict_coverage",
        "remaining_budget": review_budget,
        "refuted_candidate_ids": sorted(refuted),
        "adoption": "requires explicit user decision",
    }
