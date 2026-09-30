from __future__ import annotations

from dataclasses import dataclass

from .alignment import best_icp, range_signature, signature_distance
from .models import LoopClosure, Scan2D


@dataclass(frozen=True)
class ObservationEvidence:
    repeat_index: int
    anchor_index: int
    signature_distance: float
    rmse: float
    inlier_fraction: float
    supports_fold: bool
    detail: dict[str, float | int]

    def to_dict(self) -> dict[str, object]:
        return {
            "repeat_index": self.repeat_index,
            "anchor_index": self.anchor_index,
            "signature_distance": self.signature_distance,
            "rmse": self.rmse,
            "inlier_fraction": self.inlier_fraction,
            "supports_fold": self.supports_fold,
            **self.detail,
        }


def closures_conflict(first: LoopClosure, second: LoopClosure) -> bool:
    """Two closures conflict when they explain the same repeat place differently.

    Closures that share both endpoints are the same explanation, not a conflict.
    """

    first_pair = (first.anchor_index, first.repeat_index)
    second_pair = (second.anchor_index, second.repeat_index)
    if first_pair == second_pair:
        return False
    return (
        first.anchor_index == second.anchor_index
        or first.repeat_index == second.repeat_index
        or first.anchor_index == second.repeat_index
        or first.repeat_index == second.anchor_index
    )


def fold_reference(closure: LoopClosure, repeat_index: int) -> int | None:
    """Map a later scan to the anchor-side scan predicted by folding along a closure."""

    mapped = closure.anchor_index + (repeat_index - closure.repeat_index)
    if mapped < 0:
        return None
    return mapped


def evaluate_observation(
    scans: list[Scan2D],
    closure: LoopClosure,
    repeat_index: int,
    inlier_limit: float = 0.60,
    signature_limit: float = 0.65,
) -> ObservationEvidence | None:
    anchor_index = fold_reference(closure, repeat_index)
    if anchor_index is None or anchor_index >= len(scans) or anchor_index == repeat_index:
        return None
    anchor_scan = scans[anchor_index]
    repeat_scan = scans[repeat_index]
    shape_distance = signature_distance(
        range_signature(list(repeat_scan.points)),
        range_signature(list(anchor_scan.points)),
    )
    alignment = best_icp(list(repeat_scan.points), list(anchor_scan.points))
    supports = alignment.inlier_fraction >= inlier_limit and shape_distance <= signature_limit
    return ObservationEvidence(
        repeat_index=repeat_index,
        anchor_index=anchor_index,
        signature_distance=shape_distance,
        rmse=alignment.rmse,
        inlier_fraction=alignment.inlier_fraction,
        supports_fold=supports,
        detail={
            "aligned_x": alignment.transform.x,
            "aligned_y": alignment.transform.y,
            "aligned_theta": alignment.transform.theta,
            "median_error": alignment.median_error,
        },
    )


def elimination_reason(
    evidence: tuple[dict[str, object], ...],
    required_failures: int = 2,
    window: int = 3,
) -> str:
    recent = evidence[-window:]
    failures = [
        item
        for item in recent
        if not bool(item["supports_fold"]) or not math_finite(item["rmse"])
    ]
    if len(failures) >= required_failures:
        positions = ", ".join(str(item["repeat_index"]) for item in failures)
        inliers = ", ".join(f"{float(item['inlier_fraction']):.2f}" for item in failures)
        return (
            f"later observations at repeat indices {positions} contradict the fold "
            f"(ICP inlier fractions {inliers}); corridors look similar but connect differently"
        )
    return ""


def math_finite(value: object) -> bool:
    return isinstance(value, (int, float)) and value == value and value not in (float("inf"), float("-inf"))
