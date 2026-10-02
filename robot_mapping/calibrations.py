from __future__ import annotations

import math

from .alignment import best_icp, range_signature, signature_distance
from .models import CalibrationItem, LoopClosure, Scan2D
from .plans import CONSISTENT, CONTRADICTORY, INCONCLUSIVE

INLIER_LIMIT = 0.60
SIGNATURE_LIMIT = 0.65

UNFULFILLED = "unfulfilled"
UNCHANGED = "unchanged"
CHANGED = "changed"


def reinterpret_points(
    points: list[tuple[float, float]],
    item: CalibrationItem,
) -> list[tuple[float, float]]:
    """Apply a revised sensor calibration to raw scan-frame points.

    A multiplicative range correction and a fixed angular offset model a sensor
    calibration revision without producing a new observation: the same raw beam
    returns are merely re-read under another calibration version.
    """

    adjusted: list[tuple[float, float]] = []
    cosine = math.cos(item.angle_offset)
    sine = math.sin(item.angle_offset)
    for x, y in points:
        scaled_x = x * item.range_scale
        scaled_y = y * item.range_scale
        adjusted.append(
            (
                cosine * scaled_x - sine * scaled_y,
                sine * scaled_x + cosine * scaled_y,
            )
        )
    return adjusted


def folded_anchor_index(closure: LoopClosure, interpreted_index: int) -> int:
    return closure.anchor_index + (interpreted_index - closure.repeat_index)


def classify_reinterpretation(
    scans: list[Scan2D],
    closure: LoopClosure,
    item: CalibrationItem,
) -> dict[str, object]:
    """Reinterpret one raw observation along a fold under a calibration item.

    The raw observation index, timestamp and plan ownership stay on the raw
    scan. ``interpreted_index`` may point at a neighbouring corridor; when it
    lands beyond the observed map the reinterpretation is reported as
    ``unfulfilled`` instead of being counted as evidence either way.
    """

    raw_index = item.raw_index
    interpreted_index = item.interpreted_index
    anchor_index = folded_anchor_index(closure, interpreted_index)
    base: dict[str, object] = {
        "raw_index": raw_index,
        "interpreted_index": interpreted_index,
        "anchor_index": anchor_index,
        "calibration_verdict": UNFULFILLED,
    }
    if (
        raw_index < 0
        or raw_index >= len(scans)
        or interpreted_index < 0
        or interpreted_index >= len(scans)
        or anchor_index < 0
        or anchor_index >= len(scans)
        or anchor_index == interpreted_index
    ):
        base["rationale"] = (
            f"raw scan {raw_index} re-read as index {interpreted_index} folds onto anchor "
            f"{anchor_index}, which has not been observed yet; the reinterpretation is unfulfilled"
        )
        return base
    anchor_scan = scans[anchor_index]
    raw_scan = scans[raw_index]
    candidate_points = reinterpret_points(list(raw_scan.points), item)
    shape_distance = signature_distance(
        range_signature(candidate_points),
        range_signature(list(anchor_scan.points)),
    )
    alignment = best_icp(candidate_points, list(anchor_scan.points))
    if shape_distance <= SIGNATURE_LIMIT and alignment.inlier_fraction >= INLIER_LIMIT:
        verdict = CONSISTENT
    elif shape_distance > SIGNATURE_LIMIT:
        verdict = CONTRADICTORY
    else:
        verdict = INCONCLUSIVE
    base.update(
        calibration_verdict=verdict,
        signature_distance=shape_distance,
        rmse=alignment.rmse,
        inlier_fraction=alignment.inlier_fraction,
        rationale=(
            f"raw scan {raw_index} re-read as neighbouring index {interpreted_index} under the "
            f"revised calibration folds onto anchor {anchor_index}: {verdict}"
        ),
    )
    return base


def branch_change(old_verdict: str | None, new_verdict: str) -> str:
    """Whether the loop-closure judgment for one raw observation changes branch."""

    if new_verdict == UNFULFILLED or new_verdict == INCONCLUSIVE or old_verdict is None:
        return CHANGED if old_verdict != new_verdict else UNCHANGED
    if old_verdict == CONSISTENT and new_verdict == CONTRADICTORY:
        return CHANGED
    if old_verdict == CONTRADICTORY and new_verdict == CONSISTENT:
        return CHANGED
    if old_verdict in (UNFULFILLED, INCONCLUSIVE):
        return CHANGED
    return UNCHANGED
