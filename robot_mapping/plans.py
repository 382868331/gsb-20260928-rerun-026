from __future__ import annotations

import statistics
from dataclasses import dataclass

from .alignment import best_icp, range_signature, signature_distance
from .models import Hypothesis, LoopClosure, Scan2D

# A planned position only counts as discriminative when the view predicted by the
# fold is itself distinctive: a plain stretch of corridor looks the same in both
# explanations, so probing there cannot be promised to exclude anything.
DISTINCTIVENESS_LIMIT = 0.60
INLIER_LIMIT = 0.60
SIGNATURE_LIMIT = 0.65

WOULD_EXCLUDE = "would_exclude_on_conflict"
INDISTINGUISHABLE = "indistinguishable"
INSUFFICIENT_INFORMATION = "insufficient_information"

CONSISTENT = "consistent"
CONTRADICTORY = "contradictory"
INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class ObservationOutcome:
    repeat_index: int
    anchor_index: int | None
    signature_distance: float | None
    rmse: float | None
    inlier_fraction: float | None
    verdict: str
    detail: dict[str, float | int]

    @property
    def supports_fold(self) -> bool:
        return self.verdict == CONSISTENT

    def to_dict(self) -> dict[str, object]:
        return {
            "repeat_index": self.repeat_index,
            "anchor_index": self.anchor_index,
            "signature_distance": self.signature_distance,
            "rmse": self.rmse,
            "inlier_fraction": self.inlier_fraction,
            "verdict": self.verdict,
            "supports_fold": self.supports_fold,
            **self.detail,
        }


def fold_anchor_index(closure: LoopClosure, planned_index: int) -> int:
    return closure.anchor_index + (planned_index - closure.repeat_index)


def signature_variability(scan: Scan2D) -> float:
    signature = range_signature(list(scan.points))
    mean = sum(signature) / len(signature)
    if mean <= 0.0:
        return 0.0
    return statistics.pstdev(signature) / mean


def predict_probe(
    scans: list[Scan2D],
    hypothesis: Hypothesis,
    closures: dict[str, LoopClosure],
    planned_index: int,
) -> dict[str, object]:
    """Freeze what one *planned* (not yet made) observation could achieve.

    The prediction never looks at the future scan; it only reasons from the map
    already observed. Positions whose folded anchor has not been visited yet stay
    ``insufficient_information`` instead of being turned into fake evidence.
    """

    base: dict[str, object] = {
        "hypothesis_id": hypothesis.hypothesis_id,
        "hypothesis_name": hypothesis.name,
        "planned_index": planned_index,
    }
    if not hypothesis.closure_ids:
        base.update(
            prediction=INDISTINGUISHABLE,
            rationale=(
                "the distinct-corridor explanation makes no fold prediction; local shape "
                "similarity alone cannot rule it out, a closure must be confirmed instead"
            ),
        )
        return base
    closure = closures[next(iter(hypothesis.closure_ids))]
    anchor_index = fold_anchor_index(closure, planned_index)
    observed_count = len(scans)
    if anchor_index < 0 or anchor_index >= observed_count or anchor_index == planned_index:
        base.update(
            prediction=INSUFFICIENT_INFORMATION,
            predicted_anchor_index=anchor_index,
            rationale=(
                f"folding index {planned_index} lands on anchor index {anchor_index}, which has "
                "not been observed yet; exclusion power is unknown until that place is visited"
            ),
        )
        return base
    anchor_scan = scans[anchor_index]
    variability = signature_variability(anchor_scan)
    base["predicted_anchor_index"] = anchor_index
    base["anchor_distinctiveness"] = variability
    if variability >= DISTINCTIVENESS_LIMIT:
        base.update(
            prediction=WOULD_EXCLUDE,
            rationale=(
                f"the fold places the robot at distinctive anchor scan {anchor_index} "
                f"(signature variability {variability:.2f}); a conflicting view there would "
                "exclude this hypothesis while an agreeing view leaves both explanations open"
            ),
        )
    else:
        base.update(
            prediction=INDISTINGUISHABLE,
            rationale=(
                f"the folded anchor scan {anchor_index} is a featureless corridor stretch "
                f"(signature variability {variability:.2f}); both explanations look the same there"
            ),
        )
    return base


def classify_observation(
    scans: list[Scan2D],
    closure: LoopClosure,
    repeat_index: int,
    effective_index: int | None = None,
) -> ObservationOutcome | None:
    """Classify one raw scan against the fold.

    ``repeat_index`` is the raw observation (its scan geometry, timestamp and
    plan ownership never change). ``effective_index`` is where a calibration
    revision places that observation; the fold is evaluated from the effective
    position while the compared scan is still the raw one.
    """

    position_index = repeat_index if effective_index is None else effective_index
    anchor_index = fold_anchor_index(closure, position_index)
    if (
        repeat_index < 0
        or repeat_index >= len(scans)
        or anchor_index < 0
        or anchor_index >= len(scans)
        or anchor_index == repeat_index
    ):
        return None
    anchor_scan = scans[anchor_index]
    repeat_scan = scans[repeat_index]
    shape_distance = signature_distance(
        range_signature(list(repeat_scan.points)),
        range_signature(list(anchor_scan.points)),
    )
    alignment = best_icp(list(repeat_scan.points), list(anchor_scan.points))
    if shape_distance <= SIGNATURE_LIMIT and alignment.inlier_fraction >= INLIER_LIMIT:
        verdict = CONSISTENT
    elif shape_distance > SIGNATURE_LIMIT:
        verdict = CONTRADICTORY
    else:
        verdict = INCONCLUSIVE
    return ObservationOutcome(
        repeat_index=repeat_index,
        anchor_index=anchor_index,
        signature_distance=shape_distance,
        rmse=alignment.rmse,
        inlier_fraction=alignment.inlier_fraction,
        verdict=verdict,
        detail={
            "aligned_x": alignment.transform.x,
            "aligned_y": alignment.transform.y,
            "aligned_theta": alignment.transform.theta,
            "median_error": alignment.median_error,
        },
    )


def expected_vs_actual(prediction: str, verdict: str, eliminated: bool) -> str:
    if verdict == INCONCLUSIVE:
        return "observation kept as inconclusive; information insufficient to exclude anything"
    if prediction == WOULD_EXCLUDE:
        if verdict == CONTRADICTORY:
            if eliminated:
                return "matched the plan: a conflicting view at the distinctive position excluded the hypothesis"
            return "the view conflicted as anticipated, but retained evidence was not yet enough to exclude the hypothesis"
        return "the view agreed with the fold, so no hypothesis was excluded, exactly as predicted"
    if prediction == INDISTINGUISHABLE:
        if verdict == CONTRADICTORY:
            return "unlike the plan, the supposedly uninformative corridor view still contradicted the fold and served as exclusion evidence"
        return "as planned, the position could not separate the explanations"
    if verdict == CONTRADICTORY:
        return "the plan flagged this position as information-insufficient, yet the actual view contradicted the fold"
    return "the plan flagged this position as information-insufficient and the actual view stayed uninformative"
