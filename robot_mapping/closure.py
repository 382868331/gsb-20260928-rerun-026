from __future__ import annotations

import math
from dataclasses import dataclass

from .alignment import AlignmentResult, best_icp, range_signature, signature_distance
from .geometry import Pose, invert_transform, wrap_angle
from .models import LoopClosure, Scan2D, new_id


@dataclass(frozen=True)
class ClosureCheck:
    accepted: bool
    score: float
    evidence: dict[str, object]
    reason: str


def scan_window(scans: list[Scan2D], index: int) -> list[Scan2D]:
    return [scan for scan in scans if max(0, index - 1) <= scan.index <= min(len(scans) - 1, index + 1)]


def pose_from_alignment(alignment: AlignmentResult) -> Pose:
    return invert_transform(alignment.transform)


def check_loop_candidate(
    scans: list[Scan2D],
    anchor_index: int,
    repeat_index: int,
    signature_limit: float = 0.65,
    rmse_limit: float = 0.14,
    inlier_limit: float = 0.72,
    translation_consistency_limit: float = 0.28,
    angle_consistency_limit: float = 0.22,
) -> ClosureCheck:
    anchors = scan_window(scans, anchor_index)
    repeats = scan_window(scans, repeat_index)
    pairs = min(len(anchors), len(repeats))
    if pairs < 2:
        return ClosureCheck(False, math.inf, {}, "need at least two observations on both sides")

    all_signatures = []
    for anchor in anchors:
        for repeat in repeats:
            all_signatures.append(signature_distance(range_signature(list(anchor.points)), range_signature(list(repeat.points))))
    offset_limit = min(len(anchors), len(repeats))
    best_offset = 0
    best_signatures: list[float] = []
    for offset in range(-(offset_limit - 1), offset_limit):
        signatures = []
        for local_index, anchor in enumerate(anchors):
            repeat_index = local_index + offset
            if 0 <= repeat_index < len(repeats):
                repeat = repeats[repeat_index]
                signatures.append(signature_distance(range_signature(list(anchor.points)), range_signature(list(repeat.points))))
        if signatures and (not best_signatures or sum(signatures) < sum(best_signatures)):
            best_offset = offset
            best_signatures = signatures
    if any(distance > signature_limit for distance in best_signatures):
        return ClosureCheck(False, max(best_signatures), {"signature_distances": best_signatures, "all_signature_distances": all_signatures}, "single-view range shape is not corroborated")

    relations: list[Pose] = []
    alignments: list[dict[str, float | int]] = []
    for local_index, anchor in enumerate(anchors):
        repeat_position = local_index + best_offset
        if not (0 <= repeat_position < len(repeats)):
            continue
        repeat = repeats[repeat_position]
        result = best_icp(list(repeat.points), list(anchor.points))
        relation = pose_from_alignment(result)
        relations.append(relation)
        alignments.append({
            "anchor_index": anchor.index,
            "repeat_index": repeat.index,
            "rmse": result.rmse,
            "median_error": result.median_error,
            "inlier_fraction": result.inlier_fraction,
            "x": relation.x,
            "y": relation.y,
            "theta": relation.theta,
        })
        if result.rmse > rmse_limit or result.inlier_fraction < inlier_limit:
            return ClosureCheck(False, result.rmse, {"alignments": alignments, "signature_distances": best_signatures}, "an adjacent observation fails geometric fit")

    translations = [(pose.x, pose.y) for pose in relations]
    mean_x = sum(x for x, _ in translations) / len(translations)
    mean_y = sum(y for _, y in translations) / len(translations)
    spread = math.sqrt(sum((x - mean_x) ** 2 + (y - mean_y) ** 2 for x, y in translations) / len(translations))
    angles = [pose.theta for pose in relations]
    angle_mean = math.atan2(sum(math.sin(a) for a in angles), sum(math.cos(a) for a in angles))
    angle_spread = math.sqrt(sum(wrap_angle(a - angle_mean) ** 2 for a in angles) / len(angles))
    mean_rmse = sum(item["rmse"] for item in alignments) / len(alignments)
    evidence = {
        "alignments": alignments,
        "signature_distances": best_signatures,
        "relation_translation_spread": spread,
        "relation_angle_spread": angle_spread,
        "observations_checked": len(relations),
        "relative_measurement": {"x": mean_x, "y": mean_y, "theta": angle_mean},
    }
    if spread > translation_consistency_limit or angle_spread > angle_consistency_limit:
        return ClosureCheck(False, mean_rmse, evidence, "similar views do not agree across adjacent observations")
    return ClosureCheck(True, mean_rmse, evidence, "multiple observations agree")


def make_proposal(run_id: str, anchor_index: int, repeat_index: int, check: ClosureCheck) -> LoopClosure:
    return LoopClosure(
        closure_id=new_id("closure"),
        run_id=run_id,
        anchor_index=anchor_index,
        repeat_index=repeat_index,
        status="proposed",
        score=check.score,
        evidence=check.evidence,
        reason=check.reason,
    )
