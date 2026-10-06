from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

SUPPORT = "support"
REFUTE = "refute"
INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class ReviewCandidate:
    candidate_id: str
    ambiguity_id: str
    scan_id: int
    source_id: str
    label: str = ""
    source_relevance_known: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "ambiguity_id": self.ambiguity_id,
            "scan_id": self.scan_id,
            "source_id": self.source_id,
            "label": self.label,
            "source_relevance_known": self.source_relevance_known,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ReviewCandidate":
        return cls(
            str(data["candidate_id"]),
            str(data["ambiguity_id"]),
            int(data["scan_id"]),
            str(data["source_id"]),
            str(data.get("label", "")),
            bool(data.get("source_relevance_known", True)),
        )


def candidate_conflicts(candidates: list[ReviewCandidate]) -> set[frozenset[str]]:
    groups: dict[str, set[str]] = {}
    for candidate in candidates:
        groups.setdefault(candidate.ambiguity_id, set()).add(candidate.candidate_id)
    return {
        frozenset(pair)
        for group in groups.values()
        for pair in combinations(sorted(group), 2)
    }


def scan_coverage(candidates: list[ReviewCandidate]) -> dict[int, set[str]]:
    conflicts = candidate_conflicts(candidates)
    coverage: dict[int, set[str]] = {}
    for candidate in candidates:
        covered = {"|".join(sorted(pair)) for pair in conflicts if candidate.candidate_id in pair}
        if covered:
            coverage.setdefault(candidate.scan_id, set()).update(covered)
    return coverage


def covered_conflicts(scans: set[int], coverage: dict[int, set[str]]) -> set[str]:
    return set().union(*(coverage.get(scan, set()) for scan in scans)) if scans else set()


def candidate_count(candidates: list[ReviewCandidate], scans: set[int]) -> int:
    return len({candidate.candidate_id for candidate in candidates if candidate.scan_id in scans})


def source_distribution(candidates: list[ReviewCandidate], scans: set[int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for candidate in candidates:
        if candidate.scan_id in scans:
            result[candidate.source_id] = result.get(candidate.source_id, 0) + 1
    return result


def balance_penalty(candidates: list[ReviewCandidate], scans: set[int]) -> float:
    counts = list(source_distribution(candidates, scans).values())
    if len(counts) < 2:
        return 0.0
    mean = sum(counts) / len(counts)
    return sum((count - mean) ** 2 for count in counts) / len(counts)


def select_scans(candidates: list[ReviewCandidate], budget: int, strategy: str) -> list[int]:
    coverage = scan_coverage(candidates)
    scans = sorted(coverage)
    all_conflicts = set().union(*coverage.values()) if coverage else set()

    def score(selected: set[int]) -> tuple:
        covered = covered_conflicts(selected, coverage)
        if strategy == "balanced_sources":
            source_counts = source_distribution(candidates, selected)
            return len(covered), len(source_counts), -balance_penalty(candidates, selected), tuple(-x for x in sorted(selected))
        return len(covered), candidate_count(candidates, selected), tuple(-x for x in sorted(selected))

    def minimum_cover() -> set[int] | None:
        if not all_conflicts or budget <= 0:
            return set() if not all_conflicts else None
        covering: dict[str, list[int]] = {}
        for scan, values in coverage.items():
            for conflict in values:
                covering.setdefault(conflict, []).append(scan)
        best: set[int] | None = None

        def search(selected: set[int], uncovered: set[str]) -> None:
            nonlocal best
            if not uncovered:
                if best is None or score(selected) < score(best):
                    best = set(selected)
                return
            if len(selected) >= budget or (best is not None and len(selected) >= len(best)):
                return
            max_gain = max(len((coverage.get(scan, set()) & uncovered)) for scan in scans if scan not in selected)
            if max_gain <= 0 or len(selected) + (len(uncovered) + max_gain - 1) // max_gain > budget:
                return
            target = min(uncovered, key=lambda conflict: len(covering[conflict]))
            choices = sorted(
                (scan for scan in covering[target] if scan not in selected),
                key=lambda scan: (len(coverage[scan] & uncovered), -scan),
                reverse=True,
            )
            for scan in choices:
                search(selected | {scan}, uncovered - coverage[scan])

        search(set(), set(all_conflicts))
        return best

    complete = minimum_cover()
    if complete is not None:
        return sorted(complete)

    selected: set[int] = set()
    while len(selected) < min(budget, len(scans)):
        choice = max((scan for scan in scans if scan not in selected), key=lambda scan: score(selected | {scan}))
        selected.add(choice)
    return sorted(selected)


def review_coverage_plan(candidates: list[ReviewCandidate], budget: int) -> dict[str, object]:
    conflicts = candidate_conflicts(candidates)
    coverage = scan_coverage(candidates)
    plans = []
    for strategy in ("max_conflicts", "balanced_sources"):
        selected_scans = select_scans(candidates, budget, strategy)
        selected = set(selected_scans)
        covered = covered_conflicts(selected, coverage)
        plans.append(
            {
                "strategy": strategy,
                "scan_ids": selected_scans,
                "reviews_used": len(selected_scans),
                "conflicts_covered": len(covered),
                "conflicts_total": len(conflicts),
                "candidate_reviews_available": candidate_count(candidates, selected),
                "source_distribution": source_distribution(candidates, selected),
                "complete": covered == conflicts and len(selected_scans) <= budget,
            }
        )
    unknown = sorted({candidate.source_id for candidate in candidates if not candidate.source_relevance_known})
    return {
        "budget": budget,
        "conflicts_total": len(conflicts),
        "scan_coverage": {str(scan): sorted(values) for scan, values in sorted(coverage.items())},
        "plans": plans,
        "recommended_strategy": max(plans, key=lambda item: (item["complete"], item["conflicts_covered"], -item["reviews_used"]))["strategy"],
        "limitations": [f"source relevance is unknown for source {source}; source balance cannot establish independence" for source in unknown],
    }


def feedback_coverage(candidates: list[ReviewCandidate], feedback: dict[str, str]) -> dict[str, object]:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    verdicts = {candidate_id: feedback.get(candidate_id, "unreviewed") for candidate_id in by_id}
    distinguished = []
    pending = []
    for pair in sorted(candidate_conflicts(candidates), key=lambda item: tuple(sorted(item))):
        left, right = sorted(pair)
        if verdicts[left] == REFUTE or verdicts[right] == REFUTE:
            distinguished.append([left, right])
        else:
            pending.append([left, right])
    states = []
    for ambiguity_id in sorted({candidate.ambiguity_id for candidate in candidates}):
        ids = sorted(candidate.candidate_id for candidate in candidates if candidate.ambiguity_id == ambiguity_id)
        if ids and all(verdicts[candidate_id] == REFUTE for candidate_id in ids):
            state = "all_candidates_refuted"
        elif any(verdicts[candidate_id] == REFUTE for candidate_id in ids) and sum(verdicts[candidate_id] != REFUTE for candidate_id in ids) == 1:
            state = "single_remaining_requires_user_adoption"
        else:
            state = "pending_evidence"
        states.append({"ambiguity_id": ambiguity_id, "state": state})
    return {
        "candidate_states": verdicts,
        "distinguished_conflicts": distinguished,
        "pending_conflicts": pending,
        "ambiguity_states": states,
        "adoption": "requires_explicit_user_decision",
    }
