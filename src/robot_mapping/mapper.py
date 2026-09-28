from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import compose_transforms, pose_delta, wrap_angle
from .grid import build_occupancy_grid
from .models import ClosureCandidate, MapVersion, OdometryReading, Pose2D, Scan2D, Transform2D
from .pose_graph import PoseConstraint, optimize_pose_graph, transform_measurement_for_edge
from .ray_matching import range_match
from .scan_matching import (
    candidate_initial_from_poses,
    multi_observation_check,
    signature_distance,
    trajectory_motion_residual,
)


def integrate_odometry(readings: list[OdometryReading]) -> list[Pose2D]:
    poses = [Pose2D(0.0, 0.0, 0.0)]
    for reading in readings[1:]:
        previous = poses[-1]
        c = math.cos(previous.theta)
        s = math.sin(previous.theta)
        poses.append(
            Pose2D(
                previous.x + c * reading.dx - s * reading.dy,
                previous.y + s * reading.dx + c * reading.dy,
                wrap_angle(previous.theta + reading.dtheta),
            )
        )
    return poses


@dataclass
class ClosurePolicy:
    minimum_gap: int = 40
    signature_limit: float = 0.42
    min_overlap: float = 0.48
    max_transform_spread: float = 0.24
    max_rmse: float = 0.22
    max_motion_residual: float = 2.5


class MappingSession:
    def __init__(self, scans: list[Scan2D], odometry: list[OdometryReading], resolution: float = 0.1) -> None:
        if len(scans) != len(odometry):
            raise ValueError("scan and odometry counts must match")
        self.scans = scans
        self.odometry = odometry
        self.resolution = resolution
        self.raw_poses = integrate_odometry(odometry)
        self.current_poses = list(self.raw_poses)
        self.candidates: list[ClosureCandidate] = []
        self.accepted_closure_id: str | None = None
        self.versions: list[MapVersion] = []
        self._next_version = 0
        self._next_candidate = 0
        self._save_version("initial_map", None)

    def find_candidates(self, policy: ClosurePolicy | None = None) -> list[ClosureCandidate]:
        policy = policy or ClosurePolicy()
        candidates: list[ClosureCandidate] = []
        for new_index in range(policy.minimum_gap, len(self.scans)):
            eligible = [
                (signature_distance(self.scans[new_index], self.scans[old_index]), old_index)
                for old_index in range(0, new_index - policy.minimum_gap + 1)
            ]
            eligible = [(distance, old_index) for distance, old_index in eligible if math.isfinite(distance)]
            if not eligible:
                continue
            distance, old_index = min(eligible, key=lambda item: item[0])
            if distance > policy.signature_limit:
                continue
            candidate = self._create_candidate(old_index, new_index)
            self.candidates.append(candidate)
            candidates.append(candidate)
        return candidates

    def add_forced_candidate(self, old_index: int, new_index: int) -> ClosureCandidate:
        candidate = self._create_candidate(old_index, new_index)
        candidate.accepted = None
        candidate.reason = f"{candidate.reason}; manually proposed for comparison"
        self.candidates.append(candidate)
        return candidate

    def _create_candidate(self, old_index: int, new_index: int) -> ClosureCandidate:
        initial = candidate_initial_from_poses(self.raw_poses, old_index, new_index)
        direct = range_match(self.scans[new_index], self.scans[old_index], initial)
        overlap, spread, observations = multi_observation_check(
            self.scans,
            self.raw_poses,
            old_index,
            new_index,
            (direct.transform.x, direct.transform.y, direct.transform.theta),
        )
        rmse_values = [observation.rmse for observation in observations if math.isfinite(observation.rmse)]
        rmse = sum(rmse_values) / len(rmse_values) if rmse_values else math.inf
        motion_residual = trajectory_motion_residual(self.raw_poses, old_index, new_index, direct.transform)
        candidate = ClosureCandidate(
            f"closure-{self._next_candidate:03d}",
            old_index,
            new_index,
            direct.transform,
            overlap,
            spread,
            motion_residual,
            rmse,
        )
        self._next_candidate += 1
        candidate.reason = self._evaluate_reason(candidate, ClosurePolicy())
        return candidate

    @staticmethod
    def _evaluate_reason(candidate: ClosureCandidate, policy: ClosurePolicy) -> str:
        failures = []
        if candidate.observation_overlap < policy.min_overlap:
            failures.append(f"multi-observation overlap {candidate.observation_overlap:.2f}<{policy.min_overlap:.2f}")
        if candidate.transform_spread > policy.max_transform_spread:
            failures.append(f"transform spread {candidate.transform_spread:.2f}>{policy.max_transform_spread:.2f}")
        if candidate.mean_rmse > policy.max_rmse:
            failures.append(f"match rmse {candidate.mean_rmse:.2f}>{policy.max_rmse:.2f}")
        if candidate.motion_residual > policy.max_motion_residual:
            failures.append(f"motion residual {candidate.motion_residual:.2f}>{policy.max_motion_residual:.2f}")
        return "consistent revisit" if not failures else "; ".join(failures)

    def evaluate_candidate(self, candidate: ClosureCandidate, policy: ClosurePolicy | None = None) -> bool:
        policy = policy or ClosurePolicy()
        accepted = candidate.reason == "consistent revisit" and (
            candidate.observation_overlap >= policy.min_overlap
            and candidate.transform_spread <= policy.max_transform_spread
            and candidate.mean_rmse <= policy.max_rmse
            and candidate.motion_residual <= policy.max_motion_residual
        )
        candidate.accepted = accepted
        if accepted:
            self.accept_closure(candidate)
        return accepted

    def accept_closure(self, candidate: ClosureCandidate, force: bool = False) -> MapVersion:
        if self.accepted_closure_id is not None and not force:
            raise RuntimeError("withdraw the existing closure before accepting another")
        candidate.accepted = True
        if not force and candidate.reason != "consistent revisit":
            raise ValueError("candidate failed multi-observation validation; use force only for demonstration")
        self.accepted_closure_id = candidate.candidate_id
        self.current_poses = self._optimize(with_closure=candidate)
        return self._save_version("closure_accepted", candidate.candidate_id)

    def reject_candidate(self, candidate: ClosureCandidate) -> ClosureCandidate:
        candidate.accepted = False
        return candidate

    def withdraw_accepted_closure(self) -> MapVersion:
        if self.accepted_closure_id is None:
            raise RuntimeError("no accepted closure to withdraw")
        closure_id = self.accepted_closure_id
        for candidate in self.candidates:
            if candidate.candidate_id == closure_id:
                candidate.accepted = False
                candidate.reason = (candidate.reason + "; withdrawn by operator").strip("; ")
        self.accepted_closure_id = None
        self.current_poses = self._optimize(with_closure=None)
        return self._save_version("closure_withdrawn", closure_id)

    def _optimize(self, with_closure: ClosureCandidate | None) -> list[Pose2D]:
        constraints: list[PoseConstraint] = []
        for index in range(1, len(self.raw_poses)):
            relative = transform_measurement_for_edge(self.raw_poses[index - 1], self.raw_poses[index])
            constraints.append(PoseConstraint(index - 1, index, relative, (8.0, 8.0, 10.0)))
        if with_closure is not None:
            constraints.append(
                PoseConstraint(
                    with_closure.from_node,
                    with_closure.to_node,
                    with_closure.relative_transform,
                    (3.0, 3.0, 4.0),
                )
            )
        return optimize_pose_graph(self.raw_poses, constraints)

    def _save_version(self, event: str, closure_id: str | None) -> MapVersion:
        parent_id = self.versions[-1].version_id if self.versions else None
        version = MapVersion(
            f"map-{self._next_version:03d}",
            parent_id,
            event,
            closure_id,
            float(self._next_version),
            tuple(self.current_poses),
            self.resolution,
            build_occupancy_grid(self.scans, self.current_poses, self.resolution),
            {"accepted_closure_id": self.accepted_closure_id},
        )
        self.versions.append(version)
        self._next_version += 1
        return version

    def build_version_with_force(self, candidate: ClosureCandidate) -> MapVersion:
        return self.accept_closure(candidate, force=True)
