from __future__ import annotations

from dataclasses import dataclass

from .alignment import range_signature, signature_distance
from .closure import check_loop_candidate, make_proposal
from .geometry import Pose
from .graph import PoseEdge, optimize_poses, relative_pose
from .hypotheses import closures_conflict, elimination_reason, evaluate_observation
from .models import Hypothesis, LoopClosure, PoseRecord, Scan2D, new_id
from .occupancy import OccupancyGrid
from .storage import ProjectStore


@dataclass
class MappingRun:
    run_id: str
    scans: list[Scan2D]
    odometry: list[PoseRecord]
    closures: list[LoopClosure]
    hypotheses: list[Hypothesis]


class MappingPlatform:
    def __init__(self, root: str):
        self.store = ProjectStore(root)
        self.runs: dict[str, MappingRun] = {}
        self._loaded: set[str] = set()

    def create_run(self, run_id: str) -> MappingRun:
        self.store.run_dir(run_id)
        run = MappingRun(run_id, [], [], [], [])
        self.runs[run_id] = run
        self._loaded.add(run_id)
        self.store.append_event({"type": "run_created", "run_id": run_id})
        return run

    def add_observation(self, run_id: str, pose: Pose, points: list[tuple[float, float]], sensor_id: str = "lidar_2d") -> None:
        run = self.runs.setdefault(run_id, MappingRun(run_id, [], [], [], []))
        self._loaded.add(run_id)
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

    def propose_competing_closures(
        self,
        run_id: str,
        candidates: list[tuple[int, int, str]],
    ) -> list[Hypothesis]:
        """Create one independent hypothesis per conflicting closure, plus a no-fold explanation.

        Each candidate is an ``(anchor_index, repeat_index, label)`` tuple. The closures
        are kept as proposals with their evidence attached; no common map is rewritten
        until a hypothesis is confirmed.
        """

        run = self._run(run_id)
        proposed = [self.propose_candidate(run_id, anchor, repeat) for anchor, repeat, _ in candidates]
        for index, closure in enumerate(proposed):
            for other in proposed[index + 1:]:
                if not closures_conflict(closure, other):
                    raise ValueError("competing candidates must explain the same repeat place differently")
        hypotheses: list[Hypothesis] = []
        for closure, (_, _, label) in zip(proposed, candidates):
            hypothesis = Hypothesis(
                hypothesis_id=new_id("hyp"),
                run_id=run_id,
                name=label,
                closure_ids=(closure.closure_id,),
                status="active",
            )
            run.hypotheses.append(hypothesis)
            self.store.record_hypothesis(hypothesis)
            hypotheses.append(hypothesis)
        null = Hypothesis(
            hypothesis_id=new_id("hyp"),
            run_id=run_id,
            name="distinct places: no loop closure",
            closure_ids=(),
            status="active",
        )
        run.hypotheses.append(null)
        self.store.record_hypothesis(null)
        hypotheses.append(null)
        return hypotheses

    def feed_observation_to_hypotheses(self, run_id: str, repeat_index: int) -> list[Hypothesis]:
        """Evaluate one later observation against every active fold explanation.

        Evidence is retained on the hypothesis even after elimination so the decision can
        be reviewed and the hypothesis can be reopened.
        """

        run = self._run(run_id)
        updated: list[Hypothesis] = []
        for hypothesis in list(run.hypotheses):
            if hypothesis.status != "active" or not hypothesis.closure_ids:
                continue
            closure_id = next(iter(hypothesis.closure_ids))
            closure = next(item for item in run.closures if item.closure_id == closure_id)
            observation = evaluate_observation(run.scans, closure, repeat_index)
            if observation is None:
                continue
            evidence = hypothesis.evidence + (observation.to_dict(),)
            enriched = hypothesis.with_evidence(evidence)
            reason = elimination_reason(enriched.evidence)
            if reason:
                enriched = enriched.with_status("eliminated", reason)
            run.hypotheses = [enriched if item.hypothesis_id == enriched.hypothesis_id else item for item in run.hypotheses]
            self.store.record_hypothesis(enriched)
            updated.append(enriched)
        return updated

    def hypothesis(self, hypothesis_id: str) -> Hypothesis:
        for run in self.runs.values():
            for hypothesis in run.hypotheses:
                if hypothesis.hypothesis_id == hypothesis_id:
                    return hypothesis
        raise KeyError(f"unknown hypothesis: {hypothesis_id}")

    def hypotheses_for(self, run_id: str) -> list[Hypothesis]:
        return list(self._run(run_id).hypotheses)

    def _replace_hypothesis(self, updated: Hypothesis, event_type: str = "hypothesis_update") -> None:
        run = self._run(updated.run_id)
        run.hypotheses = [
            updated if hypothesis.hypothesis_id == updated.hypothesis_id else hypothesis
            for hypothesis in run.hypotheses
        ]
        self.store.record_hypothesis(updated)
        self.store.append_event({"type": event_type, "run_id": updated.run_id, "hypothesis_id": updated.hypothesis_id, "status": updated.status, "reason": updated.reason})

    def eliminate_hypothesis(self, hypothesis_id: str, reason: str) -> Hypothesis:
        hypothesis = self.hypothesis(hypothesis_id)
        if hypothesis.status == "eliminated":
            return hypothesis
        updated = hypothesis.with_status("eliminated", reason)
        self._replace_hypothesis(updated, "hypothesis_eliminated")
        return updated

    def reopen_hypothesis(self, hypothesis_id: str, reason: str = "reopened for side-by-side comparison before elimination") -> Hypothesis:
        hypothesis = self.hypothesis(hypothesis_id)
        if hypothesis.status not in ("eliminated", "confirmed"):
            return hypothesis
        updated = hypothesis.with_status("active", reason)
        self._replace_hypothesis(updated, "hypothesis_reopened")
        return updated

    def confirm_hypothesis(self, hypothesis_id: str, reason: str = "manually selected map explanation") -> Hypothesis:
        """Commit one explanation: its closures become the accepted common-map correction.

        Rival fold explanations are eliminated. The no-closure explanation, if chosen,
        simply leaves the common map on raw odometry.
        """

        selected = self.hypothesis(hypothesis_id)
        run = self._run(selected.run_id)
        chosen = {closure_id for closure_id in selected.closure_ids}
        for closure in run.closures:
            if closure.closure_id in chosen and closure.status in ("proposed", "withdrawn"):
                self.accept_closure(closure.closure_id, reason)
            elif closure.closure_id not in chosen and closure.status == "accepted":
                self.withdraw_closure(closure.closure_id, "superseded by selected map hypothesis")
        committed = selected.with_status("confirmed", reason)
        for hypothesis in list(run.hypotheses):
            if hypothesis.hypothesis_id == committed.hypothesis_id:
                continue
            if hypothesis.status == "confirmed":
                self._replace_hypothesis(hypothesis.with_status("active", "superseded by a later selection"), "hypothesis_superseded")
            elif hypothesis.closure_ids and hypothesis.status == "active":
                self._replace_hypothesis(
                    hypothesis.with_status("eliminated", f"rival of confirmed hypothesis {hypothesis_id}"),
                    "hypothesis_eliminated",
                )
        self._replace_hypothesis(committed, "hypothesis_confirmed")
        return committed

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

    def corrected_poses(self, run_id: str, hypothesis_id: str | None = None) -> list[Pose]:
        run = self._run(run_id)
        poses = [record.pose for record in run.odometry]
        if hypothesis_id is not None:
            hypothesis = self.hypothesis(hypothesis_id)
            selected = set(hypothesis.closure_ids)
            accepted = [closure for closure in run.closures if closure.closure_id in selected]
        else:
            accepted = [closure for closure in run.closures if closure.status == "accepted"]
        if not accepted:
            return poses
        result = list(poses)
        spans = sorted(
            (min(closure.anchor_index, closure.repeat_index), max(closure.anchor_index, closure.repeat_index))
            for closure in accepted
        )
        merged: list[list[int]] = []
        for start, end in spans:
            if merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        for start, end in merged:
            local_initial = poses[start:end + 1]
            edges = [
                PoseEdge(local, local + 1, relative_pose(local_initial[local], local_initial[local + 1]), 1.0)
                for local in range(end - start)
            ]
            for closure in accepted:
                closure_start = min(closure.anchor_index, closure.repeat_index)
                closure_end = max(closure.anchor_index, closure.repeat_index)
                if closure_start < start or closure_end > end:
                    continue
                data = closure.evidence.get("relative_measurement", {})
                measurement = Pose(data.get("x", 0.0), data.get("y", 0.0), data.get("theta", 0.0))
                edges.append(PoseEdge(closure_start - start, closure_end - start, measurement, 10.0))
            optimized = optimize_poses(local_initial, edges)
            for local, pose in enumerate(optimized):
                result[start + local] = pose
        return result

    def build_map(self, run_id: str, version_label: str, resolution: float = 0.1, hypothesis_id: str | None = None) -> dict[str, object]:
        run = self._run(run_id)
        hypothesis = self.hypothesis(hypothesis_id) if hypothesis_id is not None else None
        poses = self.corrected_poses(run_id, hypothesis_id)
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
        if hypothesis is not None:
            applied = list(hypothesis.closure_ids)
        else:
            applied = [closure.closure_id for closure in run.closures if closure.status == "accepted"]
        withdrawn = [closure.closure_id for closure in run.closures if closure.status == "withdrawn"]
        payload = {
            "map_version_id": version_id,
            "label": version_label,
            "run_id": run_id,
            "resolution": resolution,
            "accepted_closures": applied,
            "withdrawn_closures": withdrawn,
            "hypothesis_id": hypothesis_id,
            "hypothesis_status": None if hypothesis is None else hypothesis.status,
            "poses": [pose.to_dict() for pose in poses],
            "raw_scan_count": len(run.scans),
        }
        self.store.save_map_version(version_id, payload, grid.to_pgm())
        self.store.append_event({"type": "map_version", "run_id": run_id, "map_version_id": version_id, "label": version_label, "hypothesis_id": hypothesis_id})
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
        if run_id not in self._loaded:
            scans = self.store.load_scans(run_id)
            poses = self.store.load_poses(run_id)
            closures = [c for c in self.store.load_closures() if c.run_id == run_id]
            hypotheses = [h for h in self.store.load_hypotheses() if h.run_id == run_id]
            self.runs[run_id] = MappingRun(run_id, scans, poses, closures, hypotheses)
            self._loaded.add(run_id)
        return self.runs[run_id]
