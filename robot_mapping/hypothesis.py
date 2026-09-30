from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from .alignment import align_icp, best_icp, symmetric_point_error
from .geometry import Pose, compose_poses, invert_transform
from .models import Scan2D, PoseRecord, new_id

HypothesisStatus = Literal["open", "excluded", "selected"]


@dataclass
class Hypothesis:
    hypothesis_id: str
    run_id: str
    label: str
    closure_ids: list[str]
    status: HypothesisStatus = "open"
    evidence: list[dict[str, Any]] = field(default_factory=list)
    cursors: dict[str, int] = field(default_factory=dict)
    anchor_windows: dict[str, tuple[int, int]] = field(default_factory=dict)
    excluded_reason: str = ""
    excluded_at: float | None = None
    reopened: list[dict[str, Any]] = field(default_factory=list)
    trial_map_id: str = ""
    selected_map_id: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "run_id": self.run_id,
            "label": self.label,
            "closure_ids": list(self.closure_ids),
            "status": self.status,
            "evidence": list(self.evidence),
            "cursors": dict(self.cursors),
            "anchor_windows": dict(self.anchor_windows),
            "excluded_reason": self.excluded_reason,
            "excluded_at": self.excluded_at,
            "reopened": list(self.reopened),
            "trial_map_id": self.trial_map_id,
            "selected_map_id": self.selected_map_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Hypothesis":
        return cls(
            hypothesis_id=data["hypothesis_id"],
            run_id=data["run_id"],
            label=data["label"],
            closure_ids=list(data["closure_ids"]),
            status=data.get("status", "open"),
            evidence=list(data.get("evidence", [])),
        cursors=dict(data.get("cursors", {})),
        anchor_windows={key: tuple(value) for key, value in data.get("anchor_windows", {}).items()},
            excluded_reason=data.get("excluded_reason", ""),
            excluded_at=None if data.get("excluded_at") is None else float(data["excluded_at"]),
            reopened=list(data.get("reopened", [])),
            trial_map_id=data.get("trial_map_id", ""),
            selected_map_id=data.get("selected_map_id", ""),
            created_at=float(data.get("created_at", time.time())),
        )


def make_hypothesis(run_id: str, closure_ids: list[str], label: str) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=new_id("hypothesis"),
        run_id=run_id,
        label=label,
        closure_ids=list(closure_ids),
    )


def closures_conflict(closure_a: Any, closure_b: Any) -> bool:
    """Two closures compete when they explain the same revisit with different anchors."""
    if closure_a.closure_id == closure_b.closure_id:
        return False
    nodes_a = {closure_a.anchor_index, closure_a.repeat_index}
    nodes_b = {closure_b.anchor_index, closure_b.repeat_index}
    if nodes_a & nodes_b:
        return closure_a.anchor_index != closure_b.anchor_index or closure_a.repeat_index != closure_b.repeat_index
    return False


def evaluate_followup(
    scans: list[Scan2D],
    odometry: list[PoseRecord],
    anchor_index: int,
    repeat_index: int,
    later_index: int,
    rmse_limit: float = 0.35,
    inlier_limit: float = 0.3,
    anchor_window: tuple[int, int] | None = None,
    later_window: tuple[int, int] | None = None,
) -> dict[str, Any] | None:
    """Check whether a later observation still fits the closure's predicted correspondence."""
    offset = later_index - repeat_index
    if offset <= 1 or later_index >= len(scans):
        return None
    if later_window is not None and not (later_window[0] <= later_index <= later_window[1]):
        return None
    target_index = anchor_index + offset
    if not (0 <= target_index < len(scans)):
        return None
    if anchor_window is not None and not (anchor_window[0] <= target_index <= anchor_window[1]):
        return None
    base = best_icp(list(scans[repeat_index].points), list(scans[anchor_index].points))
    predicted = compose_poses(invert_transform(odometry[anchor_index].pose), odometry[target_index].pose)
    guess = compose_poses(base.transform, predicted)
    result = align_icp(
        list(scans[later_index].points),
        list(scans[target_index].points),
        initial=guess,
    )
    chamfer = symmetric_point_error(list(scans[later_index].points), list(scans[target_index].points), result.transform)
    supports = math.isfinite(chamfer) and chamfer <= 0.28
    return {
        "observation_index": later_index,
        "predicted_anchor_index": target_index,
        "rmse": result.rmse,
        "chamfer_error": chamfer,
        "median_error": result.median_error,
        "inlier_fraction": result.inlier_fraction,
        "supports": supports,
        "rmse_limit": rmse_limit,
        "inlier_limit": inlier_limit,
    }
