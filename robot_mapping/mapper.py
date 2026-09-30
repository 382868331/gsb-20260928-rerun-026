from __future__ import annotations

import math
from dataclasses import dataclass, field

from .alignment import range_signature, signature_distance
from .closure import check_loop_candidate, make_proposal
from .geometry import Pose, wrap_angle
from .graph import PoseEdge, affected_nodes, optimize_poses, optimize_subgraph, relative_pose
from .hypothesis import Hypothesis, evaluate_followup, make_hypothesis
from .models import LoopClosure, PoseRecord, Scan2D, new_id
from .occupancy import OccupancyGrid
from .storage import ProjectStore


@dataclass
class MappingRun:
    run_id: str
    scans: list[Scan2D]
    odometry: list[PoseRecord]
    closures: list[LoopClosure]
    hypotheses: dict[str, Hypothesis] = field(default_factory=dict)


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
        dependent = self._dependent_accepted(updated.run_id, closure_id)
        for other in dependent:
            cascaded = other.with_status("withdrawn", f"dependent correction withdrawn together with {closure_id}")
            self._replace_closure(cascaded)
        return updated

    def _dependent_accepted(self, run_id: str, closure_id: str) -> list[LoopClosure]:
        run = self._run(run_id)
        accepted = [c for c in run.closures if c.status == "accepted" and c.closure_id != closure_id]
        target = self._closure(closure_id)
        dependent: list[LoopClosure] = []
        changed = True
        removed = {closure_id}
        while changed:
            changed = False
            for closure in accepted:
                if closure.closure_id in removed:
                    continue
                shares_anchor = any(closure.anchor_index == self._closure(other_id).anchor_index for other_id in removed)
                in_same_family = closure.repeat_index == target.repeat_index or shares_anchor
                if in_same_family:
                    dependent.append(closure)
                    removed.add(closure.closure_id)
                    changed = True
        return dependent

    def corrected_poses(self, run_id: str) -> list[Pose]:
        run = self._run(run_id)
        accepted = [closure for closure in run.closures if closure.status == "accepted"]
        return self._poses_with(run, accepted, freeze_unaffected=bool(accepted))

    def _poses_with(self, run: MappingRun, trial_closures: list[LoopClosure], freeze_unaffected: bool = False) -> list[Pose]:
        poses = [record.pose for record in run.odometry]
        if not trial_closures:
            return poses
        touched: set[int] = set()
        for closure in trial_closures:
            edges = [PoseEdge(i, i + 1, relative_pose(poses[i], poses[i + 1]), 1.0) for i in range(len(poses) - 1)]
            data = closure.evidence.get("relative_measurement", {})
            measurement = Pose(data.get("x", 0.0), data.get("y", 0.0), data.get("theta", 0.0))
            edges.append(PoseEdge(closure.anchor_index, closure.repeat_index, measurement, 10.0))
            touched.add(closure.anchor_index)
            touched.add(closure.repeat_index)
        if freeze_unaffected:
            influenced = affected_nodes(
                [edge for edge in edges if edge.weight <= 1.5],
                touched,
                hop_limit=12,
            )
            return optimize_subgraph(poses, edges, influenced)
        return optimize_poses(poses, edges)

    def build_map(self, run_id: str, version_label: str, resolution: float = 0.1) -> dict[str, object]:
        run = self._run(run_id)
        accepted = [closure for closure in run.closures if closure.status == "accepted"]
        poses = self._poses_with(run, accepted, freeze_unaffected=bool(accepted))
        version_id = new_id("map")
        withdrawn = [closure.closure_id for closure in run.closures if closure.status == "withdrawn"]
        payload = {
            "map_version_id": version_id,
            "label": version_label,
            "run_id": run_id,
            "resolution": resolution,
            "accepted_closures": [closure.closure_id for closure in accepted],
            "withdrawn_closures": withdrawn,
            "poses": [pose.to_dict() for pose in poses],
            "raw_scan_count": len(run.scans),
        }
        self._write_map_version(run, payload, poses, version_id, resolution)
        self.store.append_event({"type": "map_version", "run_id": run_id, "map_version_id": version_id, "label": version_label})
        return payload

    # ------------------------------------------------------------------
    # Competing hypotheses
    # ------------------------------------------------------------------

    def create_hypothesis(self, run_id: str, closure_ids: list[str], label: str) -> Hypothesis:
        run = self._run(run_id)
        members = [self._closure(closure_id) for closure_id in closure_ids]
        for closure in members:
            if closure.status != "proposed":
                raise ValueError(f"closure {closure.closure_id} is {closure.status}, only proposed closures can open a hypothesis")
        hypothesis = make_hypothesis(run_id, closure_ids, label)
        for closure in members:
            window = self._straight_window(run, closure.anchor_index)
            hypothesis.anchor_windows[closure.closure_id] = window
            repeat_window = self._straight_window(run, closure.repeat_index)
            hypothesis.cursors[closure.closure_id] = max(closure.repeat_index + 2, repeat_window[0])
        run.hypotheses[hypothesis.hypothesis_id] = hypothesis
        self.store.record_hypothesis(hypothesis)
        return hypothesis

    @staticmethod
    def _straight_window(run: MappingRun, index: int) -> tuple[int, int]:
        base = run.odometry[index].pose.theta
        low = index
        while low > 0 and abs(wrap_angle(run.odometry[low - 1].pose.theta - base)) < 0.15:
            low -= 1
        high = index
        while high < len(run.odometry) - 1 and abs(wrap_angle(run.odometry[high + 1].pose.theta - base)) < 0.15:
            high += 1
        return low, high

    def build_trial_map(self, hypothesis_id: str, resolution: float = 0.1) -> dict[str, object]:
        hypothesis, run = self._hypothesis(hypothesis_id)
        if hypothesis.status not in ("open", "excluded"):
            raise ValueError(f"hypothesis {hypothesis_id} is {hypothesis.status}")
        payload = self._build_hypothesis_map(run, hypothesis, "trial", resolution)
        hypothesis.trial_map_id = payload["map_version_id"]
        self.store.record_hypothesis(hypothesis)
        return payload

    def advance_observations(self, run_id: str, contradict_limit: int = 2, required_support: float = 0.3) -> list[Hypothesis]:
        """Feed every later observation to each open hypothesis without touching the common map."""
        run = self._run(run_id)
        updated: list[Hypothesis] = []
        for hypothesis in list(run.hypotheses.values()):
            if hypothesis.status != "open":
                continue
            changed = False
            for closure_id in hypothesis.closure_ids:
                closure = self._closure(closure_id)
                cursor = max(hypothesis.cursors.get(closure_id, closure.repeat_index + 1), closure.repeat_index + 2)
                anchor_window = hypothesis.anchor_windows.get(closure_id)
                later_window = self._straight_window(run, closure.repeat_index)
                last_later = len(run.scans) - 1
                if anchor_window is not None:
                    last_later = min(last_later, closure.repeat_index + (anchor_window[1] - closure.anchor_index))
                last_later = min(last_later, later_window[1])
                for later_index in range(cursor, last_later + 1):
                    outcome = evaluate_followup(
                        run.scans,
                        run.odometry,
                        closure.anchor_index,
                        closure.repeat_index,
                        later_index,
                    )
                    if outcome is None:
                        break
                    hypothesis.cursors[closure_id] = later_index + 1
                    hypothesis.evidence.append({"closure_id": closure_id, **outcome})
                    changed = True
            if changed:
                scores_by_closure: dict[str, tuple[int, int, list[dict[str, object]]]] = {}
                for closure_id in hypothesis.closure_ids:
                    streak: list[dict[str, object]] = []
                    supports = 0
                    total = 0
                    for item in (entry for entry in hypothesis.evidence if entry["closure_id"] == closure_id):
                        total += 1
                        if item["supports"]:
                            streak = []
                            supports += 1
                        else:
                            streak = streak + [item]
                    scores_by_closure[closure_id] = (supports, total, streak)
                failing = [
                    (closure_id, streak)
                    for closure_id, (supports, total, streak) in scores_by_closure.items()
                    if len(streak) >= contradict_limit and (total == 0 or supports / total < required_support)
                ]
                if failing:
                    closure_id, streak = max(failing, key=lambda pair: len(pair[1]))
                    decisive = next((item for item in streak if item.get("chamfer_error", math.inf) < 5.0), streak[0])
                    tail = [item for item in streak[-contradict_limit:] if item is not decisive]
                    recent = [decisive, *tail][:3]
                    hypothesis.status = "excluded"
                    hypothesis.excluded_at = __import__("time").time()
                    hypothesis.excluded_reason = (
                        "后续观测与该闭环预测的对应位置不符："
                        + ", ".join(
                            f"scan {item['observation_index']} chamfer={item.get('chamfer_error', item['rmse']):.3f}"
                            for item in recent
                        )
                    )
                    self.store.append_event({
                        "type": "hypothesis_excluded",
                        "run_id": run_id,
                        "hypothesis_id": hypothesis.hypothesis_id,
                        "reason": hypothesis.excluded_reason,
                        "evidence": hypothesis.evidence,
                    })
                self.store.record_hypothesis(hypothesis)
                updated.append(hypothesis)
        return updated

    def exclusion_basis(self, hypothesis_id: str) -> dict[str, object]:
        hypothesis, _ = self._hypothesis(hypothesis_id)
        return {
            "hypothesis_id": hypothesis.hypothesis_id,
            "status": hypothesis.status,
            "excluded_reason": hypothesis.excluded_reason,
            "evidence": list(hypothesis.evidence),
            "trial_map_id": hypothesis.trial_map_id,
        }

    def reopen_hypothesis(self, hypothesis_id: str, reason: str = "reopened for comparison before exclusion") -> Hypothesis:
        hypothesis, run = self._hypothesis(hypothesis_id)
        if hypothesis.status != "excluded":
            raise ValueError("only an excluded hypothesis can be reopened")
        hypothesis.reopened.append({"reason": reason, "at": __import__("time").time(), "previous_reason": hypothesis.excluded_reason})
        hypothesis.status = "open"
        hypothesis.excluded_reason = ""
        hypothesis.excluded_at = None
        hypothesis.evidence = []
        for closure_id in hypothesis.closure_ids:
            closure = self._closure(closure_id)
            hypothesis.cursors[closure_id] = closure.repeat_index + 1
        self.store.append_event({
            "type": "hypothesis_reopened",
            "run_id": run.run_id,
            "hypothesis_id": hypothesis.hypothesis_id,
            "reason": reason,
        })
        self.store.record_hypothesis(hypothesis)
        return hypothesis

    def select_hypothesis(self, hypothesis_id: str, resolution: float = 0.1) -> dict[str, object]:
        hypothesis, run = self._hypothesis(hypothesis_id)
        if hypothesis.status not in ("open", "excluded"):
            raise ValueError(f"hypothesis {hypothesis_id} is {hypothesis.status}")
        for other in run.hypotheses.values():
            if other.hypothesis_id != hypothesis_id and other.status == "selected":
                raise ValueError("another hypothesis is already selected; withdraw it first")
        for closure_id in hypothesis.closure_ids:
            closure = self._closure(closure_id)
            if closure.status in ("proposed", "withdrawn"):
                self._replace_closure(closure.with_status("accepted", f"accepted as part of {hypothesis.label}"))
        hypothesis.status = "selected"
        payload = self._build_hypothesis_map(run, hypothesis, "selected", resolution)
        hypothesis.selected_map_id = payload["map_version_id"]
        self.store.append_event({
            "type": "hypothesis_selected",
            "run_id": run.run_id,
            "hypothesis_id": hypothesis.hypothesis_id,
            "map_version_id": payload["map_version_id"],
        })
        self.store.record_hypothesis(hypothesis)
        return payload

    def withdraw_hypothesis(self, hypothesis_id: str) -> Hypothesis:
        hypothesis, run = self._hypothesis(hypothesis_id)
        if hypothesis.status != "selected":
            raise ValueError("only a selected hypothesis can be withdrawn")
        first = hypothesis.closure_ids[0] if hypothesis.closure_ids else ""
        for closure_id in hypothesis.closure_ids:
            closure = self._closure(closure_id)
            if closure.status == "accepted":
                self._replace_closure(closure.with_status("withdrawn", f"hypothesis {hypothesis.label} withdrawn"))
        if first:
            closure = self._closure(first)
            for other in self._dependent_accepted(run.run_id, first):
                self._replace_closure(other.with_status("withdrawn", f"dependent correction withdrawn with {hypothesis.label}"))
        hypothesis.status = "open"
        hypothesis.selected_map_id = ""
        self.store.append_event({
            "type": "hypothesis_withdrawn",
            "run_id": run.run_id,
            "hypothesis_id": hypothesis.hypothesis_id,
        })
        self.store.record_hypothesis(hypothesis)
        return hypothesis

    def _build_hypothesis_map(self, run: MappingRun, hypothesis: Hypothesis, kind: str, resolution: float) -> dict[str, object]:
        members = [self._closure(closure_id) for closure_id in hypothesis.closure_ids]
        poses = self._poses_with(run, members, freeze_unaffected=True)
        version_id = new_id("map")
        payload = {
            "map_version_id": version_id,
            "label": f"{kind}_{hypothesis.label}",
            "run_id": run.run_id,
            "resolution": resolution,
            "hypothesis_id": hypothesis.hypothesis_id,
            "trial": kind == "trial",
            "accepted_closures": [closure.closure_id for closure in members],
            "withdrawn_closures": [],
            "poses": [pose.to_dict() for pose in poses],
            "raw_scan_count": len(run.scans),
        }
        self._write_map_version(run, payload, poses, version_id, resolution)
        self.store.append_event({
            "type": "map_version",
            "run_id": run.run_id,
            "map_version_id": version_id,
            "label": payload["label"],
            "hypothesis_id": hypothesis.hypothesis_id,
            "trial": kind == "trial",
        })
        return payload

    def _write_map_version(self, run: MappingRun, payload: dict[str, object], poses: list[Pose], version_id: str, resolution: float) -> None:
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
        self.store.save_map_version(version_id, payload, grid.to_pgm())

    def _hypothesis(self, hypothesis_id: str) -> tuple[Hypothesis, MappingRun]:
        for run in self.runs.values():
            hypothesis = run.hypotheses.get(hypothesis_id)
            if hypothesis is not None:
                return hypothesis, run
        raise KeyError(f"unknown hypothesis: {hypothesis_id}")

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
            hypotheses = {h.hypothesis_id: h for h in self.store.load_hypotheses() if h.run_id == run_id}
            self.runs[run_id] = MappingRun(run_id, scans, poses, closures, hypotheses)
        return self.runs[run_id]
