from __future__ import annotations

import time
from dataclasses import dataclass

from .alignment import range_signature, signature_distance
from .closure import check_loop_candidate, make_proposal
from .geometry import Pose
from .graph import PoseEdge, optimize_poses, relative_pose
from .hypotheses import closures_conflict, elimination_reason, evaluate_observation
from .models import Hypothesis, LoopClosure, PoseRecord, Scan2D, new_id
from .occupancy import OccupancyGrid
from .planning import ProbePlan, ProbeTarget, analyze_probe_target, make_plan
from .storage import ProjectStore


@dataclass
class MappingRun:
    run_id: str
    scans: list[Scan2D]
    odometry: list[PoseRecord]
    closures: list[LoopClosure]
    hypotheses: list[Hypothesis]
    plans: list[ProbePlan]


class MappingPlatform:
    def __init__(self, root: str):
        self.store = ProjectStore(root)
        self.runs: dict[str, MappingRun] = {}
        self._loaded: set[str] = set()

    def create_run(self, run_id: str) -> MappingRun:
        self.store.run_dir(run_id)
        run = MappingRun(run_id, [], [], [], [], [])
        self.runs[run_id] = run
        self._loaded.add(run_id)
        self.store.append_event({"type": "run_created", "run_id": run_id})
        return run

    def add_observation(self, run_id: str, pose: Pose, points: list[tuple[float, float]], sensor_id: str = "lidar_2d") -> None:
        run = self.runs.setdefault(run_id, MappingRun(run_id, [], [], [], [], []))
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
            if hypothesis.status not in ("active", "eliminated") or not hypothesis.closure_ids:
                continue
            closure_id = next(iter(hypothesis.closure_ids))
            closure = next(item for item in run.closures if item.closure_id == closure_id)
            observation = evaluate_observation(run.scans, closure, repeat_index)
            if observation is None:
                continue
            payload = observation.to_dict()
            payload.update({
                "collected_at": run.scans[repeat_index].timestamp,
                "withdrawn": False,
            })
            evidence = hypothesis.evidence + (payload,)
            enriched = hypothesis.with_evidence(evidence)
            reason = elimination_reason(self._live_evidence(enriched.evidence))
            if reason:
                enriched = enriched.with_status("eliminated", reason)
            run.hypotheses = [enriched if item.hypothesis_id == enriched.hypothesis_id else item for item in run.hypotheses]
            self.store.record_hypothesis(enriched)
            updated.append(enriched)
        return updated

    @staticmethod
    def _live_evidence(evidence: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
        return tuple(item for item in evidence if not item.get("withdrawn"))

    def create_probe_plan(
        self,
        run_id: str,
        name: str,
        targets: list[tuple[int, str, float, float]],
    ) -> ProbePlan:
        """Group future observation locations and compare what each would rule out.

        The plan is purely prospective: planned indices are not evaluated as evidence and
        the common map and hypothesis states do not change.
        """

        run = self._run(run_id)
        if not targets:
            raise ValueError("a probe plan needs at least one target location")
        observed = len(run.scans)
        branches = tuple(hypothesis for hypothesis in run.hypotheses if hypothesis.status == "active")
        for target_index, _, _, _ in targets:
            if target_index < observed:
                raise ValueError("probe targets must describe future observations, not collected scans")
        plan = make_plan(
            run_id,
            name,
            tuple(targets),
            list(run.scans),
            branches,
            tuple(run.closures),
        )
        run.plans.append(plan)
        self.store.record_probe_plan(plan)
        self.store.append_event({
            "type": "probe_plan_created",
            "run_id": run_id,
            "plan_id": plan.plan_id,
            "targets": [target.target_index for target in plan.targets],
        })
        return plan

    def plan(self, plan_id: str) -> ProbePlan:
        for run in self.runs.values():
            for plan in run.plans:
                if plan.plan_id == plan_id:
                    return plan
        raise KeyError(f"unknown probe plan: {plan_id}")

    def plans_for(self, run_id: str) -> list[ProbePlan]:
        return list(self._run(run_id).plans)

    def execute_probe(
        self,
        plan_id: str,
        target_index: int,
        repeat_index: int | None = None,
        hypothesis_id: str | None = None,
    ) -> dict[str, object]:
        """Apply an observation that has actually arrived to one planned target.

        Returns the planned prediction next to the actual result so callers can report
        expected-vs-actual differences. Only live observations become evidence; the plan
        itself never added evidence.
        """

        run = self._run(self.plan(plan_id).run_id)
        plan = self.plan(plan_id)
        target = self._target(plan, target_index)
        if target.status != "planned":
            raise ValueError(f"probe target {target_index} is already {target.status}")
        actual_index = repeat_index if repeat_index is not None else target.target_index
        if actual_index >= len(run.scans):
            raise ValueError("cannot execute a probe before its observation has been collected")
        updated_hypotheses = self.feed_observation_to_hypotheses(run.run_id, actual_index)
        outcomes: dict[str, dict[str, object]] = {}
        for hypothesis in updated_hypotheses:
            latest = next(item for item in hypothesis.evidence if item.get("repeat_index") == actual_index and not item.get("withdrawn"))
            outcomes[hypothesis.hypothesis_id] = {
                "name": hypothesis.name,
                "supports_branch": bool(latest["supports_fold"]),
                "status_after": hypothesis.status,
                "inlier_fraction": latest["inlier_fraction"],
                "signature_distance": latest["signature_distance"],
            }
            for entry in hypothesis.evidence:
                if entry.get("repeat_index") == actual_index:
                    entry["plan_id"] = plan.plan_id
                    entry["target_index"] = target.target_index
                    entry["branch_hypothesis_id"] = hypothesis_id or hypothesis.hypothesis_id
                    entry["collected_at"] = run.scans[actual_index].timestamp
            self.store.record_hypothesis(hypothesis)
        expected = {entry["hypothesis_id"]: entry["verdict"] for entry in target.analysis}
        executed_target = target.replaced(status="executed", executed_index=actual_index, executed_at=run.scans[actual_index].timestamp)
        self._replace_target(plan, executed_target, "probe_executed", {
            "actual_index": actual_index,
            "branch_hypothesis_id": hypothesis_id,
            "expected": expected,
            "actual": {key: (value["supports_branch"], value["status_after"]) for key, value in outcomes.items()},
        })
        return {
            "plan_id": plan.plan_id,
            "target_index": target.target_index,
            "actual_index": actual_index,
            "collected_at": run.scans[actual_index].timestamp,
            "expected_verdicts": expected,
            "actual_outcomes": outcomes,
            "differences": self._expected_actual_differences(expected, outcomes),
        }

    @staticmethod
    def _expected_actual_differences(
        expected: dict[str, str],
        outcomes: dict[str, dict[str, object]],
    ) -> list[dict[str, object]]:
        differences = []
        for hypothesis_id, verdict in expected.items():
            outcome = outcomes.get(hypothesis_id)
            if outcome is None:
                if verdict == "unaffected":
                    continue
                differences.append({"hypothesis_id": hypothesis_id, "expected": verdict, "actual": "no_evidence_recorded"})
                continue
            supports = bool(outcome["supports_branch"])
            if verdict == "eliminated":
                differences.append({
                    "hypothesis_id": hypothesis_id,
                    "expected": "eliminated only if the scan contradicted the folded prediction",
                    "actual": "branch eliminated" if outcome["status_after"] == "eliminated" else "branch still active; observation matched the prediction",
                })
            elif verdict in ("indistinguishable", "insufficient") and not supports and outcome["status_after"] == "eliminated":
                differences.append({
                    "hypothesis_id": hypothesis_id,
                    "expected": verdict,
                    "actual": "branch eliminated unexpectedly",
                })
        return differences

    def withdraw_probe(self, plan_id: str, target_index: int, reason: str) -> dict[str, object]:
        """Withdraw an observation later found invalid and restore plans it had excluded.

        Evidence contributed by the probe is marked withdrawn but retained; hypotheses
        return to the state justified by the remaining independent evidence.
        """

        plan = self.plan(plan_id)
        run = self._run(plan.run_id)
        target = self._target(plan, target_index)
        if target.executed_index is None:
            raise ValueError("only an executed probe can be withdrawn")
        restored: list[str] = []
        for hypothesis in list(run.hypotheses):
            changed = False
            remaining: list[dict[str, object]] = []
            for entry in hypothesis.evidence:
                if entry.get("plan_id") == plan.plan_id and int(entry.get("target_index", -1)) == target.target_index and not entry.get("withdrawn"):
                    entry["withdrawn"] = True
                    entry["withdrawn_reason"] = reason
                    entry["withdrawn_at"] = time.time()
                    changed = True
                remaining.append(entry)
            if not changed:
                continue
            live = self._live_evidence(hypothesis.evidence)
            updated = hypothesis.with_evidence(tuple(remaining))
            if hypothesis.status == "eliminated" and not elimination_reason(live):
                updated = updated.with_status("active", f"restored after probe withdrawal: {reason}")
                restored.append(hypothesis.hypothesis_id)
            run.hypotheses = [updated if item.hypothesis_id == updated.hypothesis_id else item for item in run.hypotheses]
            self.store.record_hypothesis(updated)
        withdrawn_target = target.replaced(status="withdrawn")
        self._replace_target(plan, withdrawn_target, "probe_withdrawn", {
            "reason": reason,
            "restored_hypotheses": restored,
        })
        return {"plan_id": plan.plan_id, "target_index": target_index, "restored_hypotheses": restored, "reason": reason}

    def reassign_probe(self, plan_id: str, target_index: int, corridor_label: str, reason: str) -> dict[str, object]:
        """Mark an observation as belonging to another corridor, then re-evaluate it.

        The old probe's evidence is withdrawn (restoring affected branches), the target is
        recorded as reassigned rather than deleted, and a fresh planned target continues
        from the same location with independent history.
        """

        plan = self.plan(plan_id)
        target = self._target(plan, target_index)
        withdrawal = self.withdraw_probe(plan_id, target_index, f"reassigned to {corridor_label}: {reason}")
        plan = self.plan(plan_id)
        target = self._target(plan, target_index)
        reassigned = target.replaced(status="reassigned")
        self._replace_target(plan, reassigned, "probe_reassigned", {
            "corridor_label": corridor_label,
            "reason": reason,
        })
        run = self._run(plan.run_id)
        branches = tuple(hypothesis for hypothesis in run.hypotheses if hypothesis.status == "active")
        follow_up = make_plan(
            plan.run_id,
            f"{plan.name} -> {corridor_label}",
            ((target.target_index, target.location_label, target.expected_x, target.expected_y),),
            list(run.scans),
            branches,
            tuple(run.closures),
        )
        run.plans.append(follow_up)
        self.store.record_probe_plan(follow_up)
        self.store.append_event({
            "type": "probe_plan_created",
            "run_id": plan.run_id,
            "plan_id": follow_up.plan_id,
            "derived_from_plan": plan.plan_id,
            "derived_from_target": target.target_index,
            "corridor_label": corridor_label,
        })
        withdrawal["replacement_plan_id"] = follow_up.plan_id
        withdrawal["reassigned_to"] = corridor_label
        return withdrawal

    @staticmethod
    def _target(plan: ProbePlan, target_index: int) -> ProbeTarget:
        for target in plan.targets:
            if target.target_index == target_index:
                return target
        raise KeyError(f"plan {plan.plan_id} has no target {target_index}")

    def _replace_target(self, plan: ProbePlan, target: ProbeTarget, event_type: str, detail: dict[str, object]) -> None:
        run = self._run(plan.run_id)
        updated_plan = plan.with_target(target)
        run.plans = [updated_plan if item.plan_id == plan.plan_id else item for item in run.plans]
        self.store.record_probe_plan(updated_plan)
        event = {
            "type": event_type,
            "run_id": plan.run_id,
            "plan_id": plan.plan_id,
            "target_index": target.target_index,
        }
        event.update(detail)
        self.store.append_event(event)

    def decision_timeline(self, run_id: str) -> list[dict[str, object]]:
        """Reconstruct plans, observations, exclusions and restorations from the event log.

        Reloading a saved project and calling this yields a complete explanation of the
        final map selection. Planned-only probes are reported as "planned" and carry no
        evidence weight.
        """

        self._run(run_id)
        timeline: list[dict[str, object]] = []
        for item in self.store.read_events():
            kind = item.get("type")
            if kind in ("probe_plan_created", "probe_executed", "probe_withdrawn", "probe_reassigned"):
                timeline.append(item)
            elif kind == "hypothesis_update" and item.get("status") in ("eliminated", "active", "confirmed"):
                timeline.append(item)
            elif kind in ("hypothesis_eliminated", "hypothesis_reopened", "hypothesis_confirmed", "hypothesis_superseded"):
                timeline.append(item)
            elif kind == "map_version" and item.get("run_id") == run_id:
                timeline.append(item)
        for entry in timeline:
            if entry.get("type") == "probe_plan_created" and "targets" in entry:
                entry["evidence_weight"] = "none; plan only"
        return timeline

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

    def accept_closure(self, closure_id: str, reason: str = "manually accepted") -> LoopClosure:
        closure = self._closure(closure_id)
        if closure.status not in ("proposed", "withdrawn"):
            raise ValueError(f"closure {closure_id} is {closure.status}, cannot accept")
        updated = closure.with_status("accepted", reason)
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
            plans = [p for p in self.store.load_probe_plans() if p.run_id == run_id]
            self.runs[run_id] = MappingRun(run_id, scans, poses, closures, hypotheses, plans)
            self._loaded.add(run_id)
        return self.runs[run_id]
