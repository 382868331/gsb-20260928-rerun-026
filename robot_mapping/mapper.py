from __future__ import annotations

from dataclasses import dataclass

from .alignment import range_signature, signature_distance
from .closure import check_loop_candidate, make_proposal
from .geometry import Pose
from .graph import PoseEdge, optimize_poses, relative_pose
from .models import LoopClosure, PoseRecord, Scan2D, new_id
from .occupancy import OccupancyGrid
from .storage import ProjectStore


@dataclass
class MappingRun:
    run_id: str
    scans: list[Scan2D]
    odometry: list[PoseRecord]
    closures: list[LoopClosure]


class MappingPlatform:
    def __init__(self, root: str):
        self.store = ProjectStore(root)
        self.runs: dict[str, MappingRun] = {}

    def create_run(self, run_id: str) -> MappingRun:
        self.store.run_dir(run_id)
        run = MappingRun(run_id, [], [], [])
        self.runs[run_id] = run
        self.store.append_event({"type": "run_created", "run_id": run_id})
        return run

    def add_observation(self, run_id: str, pose: Pose, points: list[tuple[float, float]], sensor_id: str = "lidar_2d") -> None:
        run = self.runs.setdefault(run_id, MappingRun(run_id, [], [], []))
        index = len(run.scans)
        timestamp = float(index)
        scan = Scan2D(run_id, index, timestamp, sensor_id, tuple(points))
        pose_record = PoseRecord(run_id, index, timestamp, pose, "odometry")
        run.scans.append(scan)
        run.odometry.append(pose_record)
        self.store.save_scan(scan)
        self.store.save_pose(pose_record)

    def detect_closures(self, run_id: str, minimum_gap: int = 7, coarse_limit: float = 0.75) -> list[LoopClosure]:
        run = self._run(run_id)
        proposals: list[LoopClosure] = []
        existing = {(c.anchor_index, c.repeat_index) for c in run.closures}
        for repeat_index in range(minimum_gap, len(run.scans)):
            repeat_signature = range_signature(list(run.scans[repeat_index].points))
            candidates = []
            for anchor_index in range(0, repeat_index - minimum_gap + 1):
                distance = signature_distance(repeat_signature, range_signature(list(run.scans[anchor_index].points)))
                if distance <= coarse_limit:
                    candidates.append((distance, anchor_index))
            for _, anchor_index in sorted(candidates)[:2]:
                if (anchor_index, repeat_index) in existing:
                    continue
                check = check_loop_candidate(run.scans, anchor_index, repeat_index)
                proposal = make_proposal(run_id, anchor_index, repeat_index, check)
                proposal = proposal.with_status("accepted" if check.accepted else "rejected", check.reason)
                run.closures.append(proposal)
                self.store.record_closure(proposal)
                proposals.append(proposal)
        return proposals

    def propose_candidate(self, run_id: str, anchor_index: int, repeat_index: int) -> LoopClosure:
        run = self._run(run_id)
        check = check_loop_candidate(run.scans, anchor_index, repeat_index)
        proposal = make_proposal(run_id, anchor_index, repeat_index, check)
        run.closures.append(proposal)
        self.store.record_closure(proposal)
        return proposal

    def accept_closure(self, closure_id: str) -> LoopClosure:
        closure = self._closure(closure_id)
        if closure.status not in ("proposed", "withdrawn"):
            raise ValueError(f"closure {closure_id} is {closure.status}, cannot accept")
        updated = closure.with_status("accepted", "manually accepted")
        self._replace_closure(updated)
        return updated

    def reject_closure(self, closure_id: str, reason: str = "manually rejected") -> LoopClosure:
        closure = self._closure(closure_id)
        if closure.status == "rejected":
            return closure
        updated = closure.with_status("rejected", reason)
        self._replace_closure(updated)
        return updated

    def withdraw_closure(self, closure_id: str) -> LoopClosure:
        closure = self._closure(closure_id)
        if closure.status != "accepted":
            raise ValueError("only an accepted correction can be withdrawn")
        updated = closure.with_status("withdrawn", "correction withdrawn; previous map version remains available")
        self._replace_closure(updated)
        return updated

    def corrected_poses(self, run_id: str) -> list[Pose]:
        run = self._run(run_id)
        poses = [record.pose for record in run.odometry]
        accepted = [closure for closure in run.closures if closure.status == "accepted"]
        if not accepted:
            return poses
        edges = [PoseEdge(i, i + 1, relative_pose(poses[i], poses[i + 1]), 1.0) for i in range(len(poses) - 1)]
        for closure in accepted:
            data = closure.evidence.get("relative_measurement", {})
            measurement = Pose(data.get("x", 0.0), data.get("y", 0.0), data.get("theta", 0.0))
            edges.append(PoseEdge(closure.anchor_index, closure.repeat_index, measurement, 10.0))
        return optimize_poses(poses, edges)

    def build_map(self, run_id: str, version_label: str, resolution: float = 0.1) -> dict[str, object]:
        run = self._run(run_id)
        poses = self.corrected_poses(run_id)
        all_points = []
        for pose, scan in zip(poses, run.scans):
            all_points.extend(pose.transform(point) for point in scan.points)
        margin = 1.0
        min_x = min(x for x, _ in all_points) - margin
        max_x = max(x for x, _ in all_points) + margin
        min_y = min(y for _, y in all_points) - margin
        max_y = max(y for _, y in all_points) + margin
        grid = OccupancyGrid.create((min_x, min_y, max_x, max_y), resolution)
        for pose, scan in zip(poses, run.scans):
            grid.insert_scan(pose, list(scan.points))
        version_id = new_id("map")
        accepted = [closure.closure_id for closure in run.closures if closure.status == "accepted"]
        withdrawn = [closure.closure_id for closure in run.closures if closure.status == "withdrawn"]
        payload = {
            "map_version_id": version_id,
            "label": version_label,
            "run_id": run_id,
            "resolution": resolution,
            "accepted_closures": accepted,
            "withdrawn_closures": withdrawn,
            "poses": [pose.to_dict() for pose in poses],
            "raw_scan_count": len(run.scans),
        }
        self.store.save_map_version(version_id, payload, grid.to_pgm())
        self.store.append_event({"type": "map_version", "run_id": run_id, "map_version_id": version_id, "label": version_label})
        return payload

    def _replace_closure(self, updated: LoopClosure) -> None:
        run = self._run(updated.run_id)
        run.closures = [updated if closure.closure_id == updated.closure_id else closure for closure in run.closures]
        self.store.record_closure(updated)

    def _closure(self, closure_id: str) -> LoopClosure:
        for run in self.runs.values():
            for closure in run.closures:
                if closure.closure_id == closure_id:
                    return closure
        raise KeyError(f"unknown closure: {closure_id}")

    def _run(self, run_id: str) -> MappingRun:
        if run_id not in self.runs:
            scans = self.store.load_scans(run_id)
            poses = self.store.load_poses(run_id)
            closures = [c for c in self.store.load_closures() if c.run_id == run_id]
            self.runs[run_id] = MappingRun(run_id, scans, poses, closures)
        return self.runs[run_id]
