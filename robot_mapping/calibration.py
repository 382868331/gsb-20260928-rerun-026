from __future__ import annotations

from .hypotheses import elimination_reason
from .models import CalibrationVersion, Hypothesis

# Evidence records keep their raw observation index; a reinterpreted record
# additionally carries the calibration id and its effective position.
SUPERSEDED = "superseded_by_calibration"
CALIBRATION_WITHDRAWN = "calibration_withdrawn"

CONFIRMED = "confirmed"
PROPOSED = "proposed"


def _counted(record: dict[str, object]) -> bool:
    return record.get("status", "valid") == "valid"


def baseline_evidence(hypothesis: Hypothesis) -> list[dict[str, object]]:
    """Evidence used while the old calibration stays in force."""

    return [dict(item) for item in hypothesis.evidence if _counted(item)]


def calibrated_evidence(
    hypothesis: Hypothesis,
    calibration: CalibrationVersion,
    include_proposed: bool = True,
) -> list[dict[str, object]]:
    """Evidence under one calibration branch.

    Shared (non-reinterpreted) observations are inherited from the old branch;
    each raw observation contributes at most one record, so the old and the new
    interpretation are never counted as two independent supports.
    """

    allowed = set()
    for item in calibration.interpretations:
        status = item.get("status", PROPOSED)
        if status == CONFIRMED or (include_proposed and status == PROPOSED):
            allowed.add(int(item["raw_index"]))
    remapped: dict[int, dict[str, object]] = {}
    for evaluation in calibration.evaluations:
        if evaluation.get("hypothesis_id") != hypothesis.hypothesis_id:
            continue
        if evaluation.get("status", "valid") != "valid":
            continue
        raw_index = int(evaluation["raw_index"])
        if raw_index not in allowed:
            continue
        record = dict(evaluation)
        record["calibration_id"] = calibration.calibration_id
        remapped[raw_index] = record
    combined: dict[int, dict[str, object]] = {}
    for item in hypothesis.evidence:
        if not _counted(item) or item.get("calibration_id"):
            continue
        raw_index = int(item["repeat_index"])
        combined[raw_index] = dict(item)
    combined.update(remapped)
    return [combined[key] for key in sorted(combined)]


def scoped_evidence(
    hypothesis: Hypothesis,
    calibration: CalibrationVersion | None,
    include_proposed: bool = True,
) -> list[dict[str, object]]:
    """Exactly one effective record per raw observation under one map scope."""

    if calibration is None:
        return baseline_evidence(hypothesis)
    confirmed = calibrated_evidence(hypothesis, calibration, include_proposed=False)
    if include_proposed:
        proposed = calibrated_evidence(hypothesis, calibration, include_proposed=True)
        return proposed
    return confirmed


def branch_verdict(evidence: list[dict[str, object]]) -> dict[str, object]:
    reason = elimination_reason(tuple(evidence))
    return {
        "status": "eliminated" if reason else "active",
        "reason": reason,
        "evidence_indices": sorted({int(item["repeat_index"]) for item in evidence}),
    }


def compare_branches(
    hypotheses: list[Hypothesis],
    calibration: CalibrationVersion,
) -> dict[str, object]:
    """Side-by-side old-calibration vs new-calibration closure judgments."""

    per_hypothesis = []
    for hypothesis in hypotheses:
        if not hypothesis.closure_ids:
            continue
        old_branch = baseline_evidence(hypothesis)
        new_branch = calibrated_evidence(hypothesis, calibration)
        old_verdict = branch_verdict(old_branch)
        new_verdict = branch_verdict(new_branch)
        moved = {
            int(item["raw_index"]): int(item["interpreted_index"])
            for item in calibration.interpretations
            if item.get("status", PROPOSED) in (PROPOSED, CONFIRMED)
        }
        old_records = {int(item["repeat_index"]): item for item in old_branch}
        new_records = {int(item["repeat_index"]): item for item in new_branch}
        changed_observations = []
        for raw_index in sorted(moved):
            before = old_records.get(raw_index)
            after = new_records.get(raw_index)
            before_verdict = None if before is None else before.get("verdict")
            after_verdict = None if after is None else after.get("verdict")
            if before_verdict != after_verdict:
                changed_observations.append(
                    {
                        "raw_index": raw_index,
                        "old_verdict": before_verdict,
                        "new_verdict": after_verdict,
                        "interpreted_index": moved[raw_index],
                    }
                )
        flipped = old_verdict["status"] != new_verdict["status"]
        ambiguous = not flipped
        per_hypothesis.append(
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "hypothesis_name": hypothesis.name,
                "old_calibration": old_verdict,
                "new_calibration": new_verdict,
                "closure_judgment_changed": flipped,
                "changed_observations": changed_observations,
                "ambiguity_remains": ambiguous,
            }
        )
    return {
        "calibration_id": calibration.calibration_id,
        "calibration_status": calibration.status,
        "hypotheses": per_hypothesis,
    }
