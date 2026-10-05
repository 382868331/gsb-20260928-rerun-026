from __future__ import annotations

import time
from dataclasses import dataclass

from .alignment import range_signature, signature_distance
from .calibration import (
    CALIBRATION_WITHDRAWN,
    CONFIRMED,
    PROPOSED,
    SUPERSEDED,
    baseline_evidence,
    branch_verdict,
    calibrated_evidence,
    compare_branches,
    scoped_evidence,
)
from .closure import check_loop_candidate, make_proposal
from .geometry import Pose
from .graph import PoseEdge, optimize_poses, relative_pose
from .hypotheses import closures_conflict, elimination_reason
from .plans import (
    classify_observation,
    expected_vs_actual,
    predict_probe,
)
from .models import (
    CalibrationVersion,
    Hypothesis,
    LoopClosure,
    PoseRecord,
    ProbePlan,
    ReviewPlan,
    Scan2D,
    new_id,
)
from .occupancy import OccupancyGrid
from .reviews import INSUFFICIENT, REFUTE, SUPPORT, compare_review_plans
from .storage import ProjectStore


@dataclass
class MappingRun:
    run_id: str
    scans: list[Scan2D]
    odometry: list[PoseRecord]
    closures: list[LoopClosure]
    hypotheses: list[Hypothesis]
    probe_plans: list[ProbePlan]
    calibrations: list[CalibrationVersion]
    review_plans: list[ReviewPlan]


class MappingPlatform:
    def __init__(self, root: str):
        self.store = ProjectStore(root)
        self.runs: dict[str, MappingRun] = {}
        self._loaded: set[str] = set()

    def create_run(self, run_id: str) -> MappingRun:
        self.store.run_dir(run_id)
        run = MappingRun(run_id, [], [], [], [], [], [], [])
        self.runs[run_id] = run
        self._loaded.add(run_id)
        self.store.append_event({"type": "run_created", "run_id": run_id})
        return run

    def add_observation(self, run_id: str, pose: Pose, points: list[tuple[float, float]], sensor_id: str = "lidar_2d") -> None:
        run = self.runs.setdefault(run_id, MappingRun(run_id, [], [], [], [], [], [], []))
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

    def create_probe_plan(
        self,
        run_id: str,
        ambiguous_region: str,
        target_indices: list[int],
        reason: str = "",
        branch_hypothesis_id: str | None = None,
    ) -> ProbePlan:
        """Group prospective follow-up probes and freeze what each *could* exclude.

        Only observations already on disk are used; planned scans are never read
        and the common map and hypothesis evidence are left untouched.
        """

        run = self._run(run_id)
        hypotheses = self._branch_hypotheses(run, branch_hypothesis_id)
        closures = {closure.closure_id: closure for closure in run.closures}
        predictions = tuple(
            prediction
            for target_index in target_indices
            for hypothesis in hypotheses
            for prediction in (predict_probe(run.scans, hypothesis, closures, target_index),)
        )
        plan = ProbePlan(
            plan_id=new_id("plan"),
            run_id=run_id,
            ambiguous_region=ambiguous_region,
            target_indices=tuple(target_indices),
            predictions=predictions,
            branch_hypothesis_id=branch_hypothesis_id,
            reason=reason,
        )
        run.probe_plans.append(plan)
        self.store.record_probe_plan(plan)
        self.store.append_event(
            {
                "type": "probe_plan_created",
                "run_id": run_id,
                "plan_id": plan.plan_id,
                "ambiguous_region": ambiguous_region,
                "target_indices": list(plan.target_indices),
                "branch_hypothesis_id": branch_hypothesis_id,
            }
        )
        return plan

    @staticmethod
    def _branch_hypotheses(run: MappingRun, branch_hypothesis_id: str | None) -> list[Hypothesis]:
        if branch_hypothesis_id is None:
            return list(run.hypotheses)
        hypothesis = next(item for item in run.hypotheses if item.hypothesis_id == branch_hypothesis_id)
        null = next((item for item in run.hypotheses if not item.closure_ids), None)
        return [hypothesis] + ([null] if null is not None else [])

    def compare_probe_plan(self, plan_id: str) -> dict[str, object]:
        """Compare, per target position, which hypotheses a probe would rule out.

        ``would_exclude`` means the fold survives an agreeing view but is excluded
        by a conflicting one. Positions stay in the insufficient-information or
        indistinguishable buckets so planned observations are not promised power
        they do not have.
        """

        plan = self.probe_plan(plan_id)
        by_target: dict[int, list[dict[str, object]]] = {}
        for prediction in plan.predictions:
            by_target.setdefault(int(prediction["planned_index"]), []).append(dict(prediction))
        return {
            "plan_id": plan_id,
            "ambiguous_region": plan.ambiguous_region,
            "executed": [item["actual_index"] for item in plan.executions],
            "targets": [
                {
                    "planned_index": target_index,
                    "predictions": by_target.get(target_index, []),
                }
                for target_index in plan.target_indices
            ],
        }

    def create_review_plan(
        self,
        run_id: str,
        ambiguity_scope: str,
        review_budget: int,
        candidate_ids: list[str] | None = None,
        observations: list[dict[str, object]] | None = None,
        ambiguity_pairs: list[tuple[str, str]] | None = None,
        preferred_strategy: str = "maximum_conflict_coverage",
        reason: str = "",
    ) -> ReviewPlan:
        """Plan evidence re-checks for unresolved, mutually exclusive explanations.

        The plan is based only on original observation sharing. Re-checking one
        shared scan is represented once, and a plan never adds support or
        changes a hypothesis; adoption remains a separate user decision.
        """

        run = self._run(run_id)
        unresolved = candidate_ids or [hypothesis.hypothesis_id for hypothesis in run.hypotheses if hypothesis.status == "active"]
        unresolved = tuple(unresolved)
        if len(unresolved) < 2:
            raise ValueError("review coverage requires at least two unresolved candidates")
        known = {hypothesis.hypothesis_id for hypothesis in run.hypotheses}
        unknown = [candidate_id for candidate_id in unresolved if candidate_id not in known]
        if unknown:
            raise ValueError(f"unknown review candidates {unknown}")
        review_observations = observations if observations is not None else self._closure_review_observations(run, unresolved)
        comparison = compare_review_plans(
            unresolved,
            review_observations,
            review_budget,
            tuple(ambiguity_pairs) if ambiguity_pairs is not None else None,
        )
        for plan in comparison["plans"]:
            plan["selected"] = plan["strategy"] == preferred_strategy
        if not any(plan["selected"] for plan in comparison["plans"]):
            raise ValueError("preferred_strategy must be maximum_conflict_coverage or balanced_source_coverage")
        selected = next(plan for plan in comparison["plans"] if plan["selected"])
        comparison["selected_observation_ids"] = selected["observation_ids"]
        comparison["remaining_budget"] = review_budget
        plan = ReviewPlan(
            review_plan_id=new_id("review"),
            run_id=run_id,
            ambiguity_scope=ambiguity_scope,
            candidate_ids=unresolved,
            ambiguity_pairs=tuple(tuple(pair) for pair in comparison["ambiguity_pairs"]),
            observations=tuple(review_observations),
            review_budget=review_budget,
            comparison=comparison,
            reason=reason,
        )
        run.review_plans.append(plan)
        self.store.record_review_plan(plan)
        self.store.append_event(
            {
                "type": "review_plan_created",
                "run_id": run_id,
                "review_plan_id": plan.review_plan_id,
                "candidate_ids": list(unresolved),
                "review_budget": review_budget,
                "selected_observation_ids": comparison["selected_observation_ids"],
            }
        )
        return plan

    @staticmethod
    def _closure_review_observations(run: MappingRun, candidate_ids: tuple[str, ...]) -> list[dict[str, object]]:
        by_scan: dict[int, dict[str, object]] = {}
        for hypothesis in run.hypotheses:
            if hypothesis.hypothesis_id not in candidate_ids or not hypothesis.closure_ids:
                continue
            for closure_id in hypothesis.closure_ids:
                closure = next(item for item in run.closures if item.closure_id == closure_id)
                scan_indices = {closure.anchor_index, closure.repeat_index}
                for alignment in closure.evidence.get("alignments", []):
                    scan_indices.add(int(alignment["anchor_index"]))
                    scan_indices.add(int(alignment["repeat_index"]))
                for scan_index in scan_indices:
                    if scan_index < 0 or scan_index >= len(run.scans):
                        continue
                    scan = run.scans[scan_index]
                    item = by_scan.setdefault(
                        scan_index,
                        {
                            "observation_id": f"scan:{scan_index}",
                            "scan_index": scan_index,
                            "scan_id": scan.scan_id,
                            "source_id": scan.sensor_id,
                            "relevance_known": True,
                            "candidate_ids": [],
                        },
                    )
                    if hypothesis.hypothesis_id not in item["candidate_ids"]:
                        item["candidate_ids"].append(hypothesis.hypothesis_id)
        return [by_scan[index] for index in sorted(by_scan)]

    def record_review_feedback(
        self,
        review_plan_id: str,
        observation_id: object,
        verdict: str,
        candidate_id: str | None = None,
        note: str = "",
    ) -> dict[str, object]:
        """Record one scoped review result without adopting the remaining explanation."""

        if verdict not in (SUPPORT, REFUTE, INSUFFICIENT):
            raise ValueError("review verdict must be support, refute, or insufficient")
        plan = self.review_plan(review_plan_id)
        selected_ids = set(plan.comparison.get("selected_observation_ids", []))
        reviewed_ids = {item["observation_id"] for item in plan.feedback}
        if observation_id not in selected_ids:
            raise ValueError("feedback is outside the selected minimum review set")
        if observation_id in reviewed_ids:
            raise ValueError("a selected scan can only be reviewed once; repeated references do not add support")
        if len(plan.feedback) >= plan.review_budget:
            raise ValueError("review budget exhausted")
        observation = next(item for item in plan.observations if item["observation_id"] == observation_id)
        linked = list(observation.get("candidate_ids", ()))
        target_candidate = candidate_id
        if verdict != INSUFFICIENT:
            if target_candidate is None:
                if len(linked) != 1:
                    raise ValueError("feedback for a shared scan must name the candidate it concerns")
                target_candidate = linked[0]
            if target_candidate not in plan.candidate_ids or target_candidate not in linked:
                raise ValueError("review feedback does not correspond to this candidate/scan sharing relation")
        feedback_item = {
            "observation_id": observation_id,
            "scan_index": observation.get("scan_index"),
            "scan_id": observation.get("scan_id"),
            "source_id": observation.get("source_id"),
            "candidate_id": target_candidate,
            "verdict": verdict,
            "note": note,
        }
        feedback = tuple(list(plan.feedback) + [feedback_item])
        reviewed = {item["observation_id"] for item in feedback}
        refuted = {item["candidate_id"] for item in feedback if item["verdict"] == REFUTE}
        remaining_observations = [item for item in plan.observations if item["observation_id"] not in reviewed]
        remaining_budget = plan.review_budget - len(feedback)
        comparison = compare_review_plans(
            plan.candidate_ids,
            remaining_observations,
            max(1, remaining_budget),
            tuple(tuple(pair) for pair in plan.ambiguity_pairs),
            feedback,
        ) if remaining_budget else {
            "candidate_ids": list(plan.candidate_ids),
            "review_budget": 0,
            "ambiguity_pairs": [list(pair) for pair in plan.ambiguity_pairs],
            "plans": [],
            "recommended_strategy": None,
            "remaining_budget": 0,
            "refuted_candidate_ids": sorted(refuted),
            "adoption": "requires explicit user decision",
        }
        comparison["used_budget"] = len(feedback)
        comparison["remaining_budget"] = remaining_budget
        updated_plan = plan.with_feedback(feedback, comparison)
        run = self._run(plan.run_id)
        run.review_plans = [updated_plan if item.review_plan_id == review_plan_id else item for item in run.review_plans]
        self.store.record_review_plan(updated_plan)
        self.store.append_event(
            {
                "type": "review_feedback_recorded",
                "run_id": plan.run_id,
                "review_plan_id": review_plan_id,
                "observation_id": observation_id,
                "candidate_id": target_candidate,
                "verdict": verdict,
                "remaining_budget": remaining_budget,
            }
        )
        return feedback_item

    def feed_observation_to_hypotheses(
        self,
        run_id: str,
        repeat_index: int,
        plan_id: str | None = None,
        source_position: tuple[float, float] | None = None,
        collected_at: float | None = None,
    ) -> dict[str, object]:
        """Apply one *actually collected* observation using its source and timestamp.

        Inconclusive views are retained as such rather than forced into a vote.
        When the observation matches a plan item, the result is compared against
        the frozen plan prediction and the difference is recorded on the plan.
        """

        run = self._run(run_id)
        timestamp = run.scans[repeat_index].timestamp if repeat_index < len(run.scans) else float(repeat_index)
        if collected_at is not None:
            timestamp = collected_at
        updated: list[Hypothesis] = []
        per_hypothesis: dict[str, dict[str, object]] = {}
        plan = self.probe_plan(plan_id) if plan_id is not None else None
        scoped = {hypothesis.hypothesis_id for hypothesis in self._branch_hypotheses(run, plan.branch_hypothesis_id)} if plan is not None else None
        for hypothesis in list(run.hypotheses):
            if scoped is not None and hypothesis.hypothesis_id not in scoped:
                continue
            if not hypothesis.closure_ids:
                continue
            if scoped is None and hypothesis.status != "active":
                continue
            if hypothesis.status == "confirmed":
                continue
            closure_id = next(iter(hypothesis.closure_ids))
            closure = next(item for item in run.closures if item.closure_id == closure_id)
            outcome = classify_observation(run.scans, closure, repeat_index)
            if outcome is None:
                continue
            record = outcome.to_dict()
            record.update(
                {
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "status": "valid",
                    "source_x": None if source_position is None else float(source_position[0]),
                    "source_y": None if source_position is None else float(source_position[1]),
                    "collected_at": timestamp,
                    "plan_id": plan_id,
                }
            )
            enriched = hypothesis.with_evidence(hypothesis.evidence + (record,))
            reason = elimination_reason(enriched.evidence)
            eliminated = bool(reason)
            if reason:
                enriched = enriched.with_status("eliminated", reason)
            run.hypotheses = [enriched if item.hypothesis_id == enriched.hypothesis_id else item for item in run.hypotheses]
            self.store.record_hypothesis(enriched)
            updated.append(enriched)
            prediction = None
            if plan is not None:
                prediction = next(
                    (
                        item
                        for item in plan.predictions
                        if item["hypothesis_id"] == hypothesis.hypothesis_id
                        and int(item["planned_index"]) == repeat_index
                    ),
                    None,
                )
            prediction_label = None if prediction is None else prediction["prediction"]
            per_hypothesis[hypothesis.hypothesis_id] = {
                "hypothesis_id": hypothesis.hypothesis_id,
                "verdict": outcome.verdict,
                "eliminated": eliminated,
                "plan_prediction": prediction_label,
                "expected_vs_actual": None
                if prediction_label is None
                else expected_vs_actual(prediction_label, outcome.verdict, eliminated),
            }
        if plan is not None:
            self._record_execution(plan, repeat_index, timestamp, source_position, per_hypothesis)
        return {
            "run_id": run_id,
            "actual_index": repeat_index,
            "collected_at": timestamp,
            "plan_id": plan_id,
            "hypothesis_results": list(per_hypothesis.values()),
            "updated_hypotheses": updated,
        }

    def _record_execution(
        self,
        plan: ProbePlan,
        actual_index: int,
        timestamp: float,
        source_position: tuple[float, float] | None,
        per_hypothesis: dict[str, dict[str, object]],
    ) -> None:
        run = self._run(plan.run_id)
        execution = {
            "actual_index": actual_index,
            "collected_at": timestamp,
            "source_x": None if source_position is None else float(source_position[0]),
            "source_y": None if source_position is None else float(source_position[1]),
            "status": "valid",
            "results": list(per_hypothesis.values()),
        }
        prior = [item for item in plan.executions if item["actual_index"] != actual_index]
        updated_plan = plan.with_executions(tuple(prior + [execution]))
        run.probe_plans = [updated_plan if item.plan_id == plan.plan_id else item for item in run.probe_plans]
        self.store.record_probe_plan(updated_plan)
        self.store.append_event(
            {
                "type": "probe_executed",
                "run_id": plan.run_id,
                "plan_id": plan.plan_id,
                "actual_index": actual_index,
                "collected_at": timestamp,
            }
        )

    def withdraw_observation(
        self,
        run_id: str,
        repeat_index: int,
        reason: str = "observation withdrawn",
        reassigned_corridor: bool = False,
    ) -> dict[str, object]:
        """Invalidate one observation's evidence without touching raw scans.

        Independent evidence from other observations stays valid. Hypotheses whose
        exclusion depended on the withdrawn evidence are restored to ``active``;
        plans mark the matching execution withdrawn so the plan history explains
        the reversal.
        """

        run = self._run(run_id)
        restored: list[str] = []
        for hypothesis in list(run.hypotheses):
            if not hypothesis.evidence:
                continue
            changed = False
            evidence = []
            for item in hypothesis.evidence:
                if int(item["repeat_index"]) == repeat_index and item.get("status", "valid") == "valid":
                    updated_item = dict(item)
                    updated_item["status"] = "reassigned" if reassigned_corridor else "withdrawn"
                    updated_item["withdrawal_reason"] = reason
                    evidence.append(updated_item)
                    changed = True
                else:
                    evidence.append(item)
            if not changed:
                continue
            replacement = hypothesis.with_evidence(tuple(evidence))
            if hypothesis.status == "eliminated" and not elimination_reason(replacement.evidence):
                replacement = replacement.with_status(
                    "active",
                    f"restored: observation {repeat_index} withdrawn ({reason}); remaining evidence insufficient for exclusion",
                )
                restored.append(hypothesis.hypothesis_id)
            run.hypotheses = [replacement if item.hypothesis_id == hypothesis.hypothesis_id else item for item in run.hypotheses]
            self.store.record_hypothesis(replacement)
            self.store.append_event(
                {
                    "type": "observation_withdrawn",
                    "run_id": run_id,
                    "repeat_index": repeat_index,
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "reassigned_corridor": reassigned_corridor,
                    "restored": hypothesis.hypothesis_id in restored,
                    "reason": reason,
                }
            )
        for plan in list(run.probe_plans):
            changed_executions = []
            for execution in plan.executions:
                if int(execution["actual_index"]) == repeat_index and execution.get("status", "valid") == "valid":
                    updated_execution = dict(execution)
                    updated_execution["status"] = "reassigned" if reassigned_corridor else "withdrawn"
                    updated_execution["withdrawal_reason"] = reason
                    changed_executions.append(updated_execution)
                else:
                    changed_executions.append(execution)
            if changed_executions != list(plan.executions):
                updated_plan = plan.with_executions(tuple(changed_executions))
                run.probe_plans = [updated_plan if item.plan_id == plan.plan_id else item for item in run.probe_plans]
                self.store.record_probe_plan(updated_plan)
        return {"run_id": run_id, "withdrawn_index": repeat_index, "restored_hypotheses": restored}

    def register_calibration(self, run_id: str, name: str, reason: str = "", reviewer: str = "") -> CalibrationVersion:
        """Register a new sensor calibration revision without touching raw data or the map.

        ``reviewer`` identifies who owns the revision so two reviewers can revise
        the same batch of observations independently, each in their own version.
        """

        run = self._run(run_id)
        calibration = CalibrationVersion(
            calibration_id=new_id("cal"),
            run_id=run_id,
            name=name,
            reviewer=reviewer,
            reason=reason,
        )
        run.calibrations.append(calibration)
        self.store.record_calibration(calibration)
        self.store.append_event(
            {
                "type": "calibration_registered",
                "run_id": run_id,
                "calibration_id": calibration.calibration_id,
                "name": name,
                "reviewer": reviewer,
                "reason": reason,
            }
        )
        return calibration

    def reinterpret_observation(
        self,
        calibration_id: str,
        raw_index: int,
        interpreted_index: int,
        plan_id: str | None = None,
        note: str = "",
    ) -> dict[str, object]:
        """Offer a new interpretation of one existing raw observation.

        No scan is created: the raw index, collection timestamp and probe-plan
        ownership stay attached to the original observation. The new reading is
        evaluated inside every fold hypothesis as a proposed branch record and
        never replaces the old interpretation or the public map by itself.
        """

        calibration = self.calibration(calibration_id)
        if calibration.status == "withdrawn":
            raise ValueError("cannot reinterpret under a withdrawn calibration")
        run = self._run(calibration.run_id)
        if raw_index < 0 or raw_index >= len(run.scans):
            raise ValueError(f"unknown raw observation {raw_index}")
        raw_scan = run.scans[raw_index]
        existing = next(
            (dict(item) for item in calibration.interpretations if int(item["raw_index"]) == raw_index),
            None,
        )
        if existing is not None:
            if int(existing["interpreted_index"]) == int(interpreted_index):
                return {
                    "run_id": run.run_id,
                    "calibration_id": calibration_id,
                    "raw_index": raw_index,
                    "interpreted_index": int(interpreted_index),
                    "collected_at": raw_scan.timestamp,
                    "plan_id": existing.get("plan_id"),
                    "hypothesis_results": [],
                    "unchanged": True,
                }
            raise ValueError(
                f"raw observation {raw_index} already has a conflicting interpretation in revision "
                f"{calibration_id}; register the dissenting reading as a separate reviewer revision "
                "so both branches stay comparable side by side"
            )
        interpretations = [dict(item) for item in calibration.interpretations]
        interpretations.append(
            {
                "raw_index": raw_index,
                "interpreted_index": int(interpreted_index),
                "status": PROPOSED,
                "plan_id": plan_id,
                "collected_at": raw_scan.timestamp,
                "note": note,
            }
        )
        evaluations = [dict(item) for item in calibration.evaluations if int(item["raw_index"]) != raw_index]
        closures = {closure.closure_id: closure for closure in run.closures}
        per_hypothesis: list[dict[str, object]] = []
        for hypothesis in run.hypotheses:
            if not hypothesis.closure_ids:
                continue
            closure = closures[next(iter(hypothesis.closure_ids))]
            outcome = classify_observation(run.scans, closure, raw_index, int(interpreted_index))
            if outcome is None:
                continue
            record = outcome.to_dict()
            record.update(
                {
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "status": "valid",
                    "calibration_id": calibration_id,
                    "raw_index": raw_index,
                    "interpreted_index": int(interpreted_index),
                    "plan_id": plan_id,
                    "collected_at": raw_scan.timestamp,
                    "adoption": PROPOSED,
                }
            )
            evaluations.append(record)
            per_hypothesis.append(
                {
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "raw_index": raw_index,
                    "interpreted_index": int(interpreted_index),
                    "verdict": outcome.verdict,
                }
            )
        updated = calibration.with_update(interpretations=tuple(interpretations), evaluations=tuple(evaluations))
        self._replace_calibration(updated)
        self.store.append_event(
            {
                "type": "observation_reinterpreted",
                "run_id": run.run_id,
                "calibration_id": calibration_id,
                "raw_index": raw_index,
                "interpreted_index": int(interpreted_index),
                "plan_id": plan_id,
                "note": note,
            }
        )
        return {
            "run_id": run.run_id,
            "calibration_id": calibration_id,
            "raw_index": raw_index,
            "interpreted_index": int(interpreted_index),
            "collected_at": raw_scan.timestamp,
            "plan_id": plan_id,
            "hypothesis_results": per_hypothesis,
        }

    def compare_calibration_branches(self, calibration_id: str) -> dict[str, object]:
        """Compare keeping the old calibration against adopting the new one."""

        calibration = self.calibration(calibration_id)
        run = self._run(calibration.run_id)
        comparison = compare_branches(run.hypotheses, calibration)
        comparison["raw_scan_count"] = len(run.scans)
        comparison["adopted_scope"] = {
            "confirmed_raw_indices": sorted(
                int(item["raw_index"])
                for item in calibration.interpretations
                if item.get("status") == CONFIRMED
            ),
            "proposed_raw_indices": sorted(
                int(item["raw_index"])
                for item in calibration.interpretations
                if item.get("status", PROPOSED) == PROPOSED
            ),
        }
        return comparison

    def confirm_calibration_interpretation(
        self,
        calibration_id: str,
        raw_indices: list[int],
        reason: str = "manually accepted reinterpretation",
    ) -> dict[str, object]:
        """Confirm only part of a calibration's new interpretations.

        The adoption scope is recorded with every affected hypothesis and map
        constraint, and the old interpretation of the same raw scan is marked
        superseded rather than counted twice. Unconfirmed reinterpretations
        stay on the comparison branch.
        """

        calibration = self.calibration(calibration_id)
        run = self._run(calibration.run_id)
        if calibration.status == "withdrawn":
            raise ValueError("cannot confirm interpretations of a withdrawn calibration")
        requested = {int(index) for index in raw_indices}
        wanted = set(requested)
        available = {int(item["raw_index"]) for item in calibration.interpretations}
        unknown = wanted - available
        if unknown:
            raise ValueError(f"no reinterpretation for raw observations {sorted(unknown)}")
        already = {
            int(item["raw_index"])
            for item in calibration.interpretations
            if item.get("status") == CONFIRMED
        }
        wanted -= already
        if not wanted:
            return {
                "run_id": run.run_id,
                "calibration_id": calibration_id,
                "confirmed_raw_indices": sorted(requested & already),
                "affected_hypotheses": [],
                "unchanged": True,
            }
        interpretations = []
        for item in calibration.interpretations:
            updated_item = dict(item)
            if int(updated_item["raw_index"]) in wanted:
                updated_item["status"] = CONFIRMED
                updated_item["confirmation_reason"] = reason
            interpretations.append(updated_item)
        evaluations = []
        for item in calibration.evaluations:
            updated_item = dict(item)
            if int(updated_item["raw_index"]) in wanted:
                updated_item["adoption"] = CONFIRMED
            evaluations.append(updated_item)
        updated = calibration.with_update(interpretations=tuple(interpretations), evaluations=tuple(evaluations))
        self._replace_calibration(updated)

        for other in list(run.calibrations):
            if other.calibration_id == calibration_id or other.status == "withdrawn":
                continue
            other_interpretations = []
            other_changed = False
            for item in other.interpretations:
                updated_item = dict(item)
                if int(updated_item["raw_index"]) in wanted and updated_item.get("status") == CONFIRMED:
                    updated_item["status"] = SUPERSEDED
                    updated_item["superseded_by"] = calibration_id
                    other_changed = True
                other_interpretations.append(updated_item)
            if other_changed:
                self._replace_calibration(other.with_update(interpretations=tuple(other_interpretations)))

        affected: list[str] = []
        for hypothesis in list(run.hypotheses):
            if not hypothesis.closure_ids:
                continue
            evidence = []
            changed = False
            for item in hypothesis.evidence:
                updated_item = dict(item)
                if updated_item.get("calibration_id") == calibration_id and int(updated_item["repeat_index"]) in wanted:
                    continue
                if (
                    int(updated_item["repeat_index"]) in wanted
                    and updated_item.get("status", "valid") == "valid"
                    and updated_item.get("calibration_id") != calibration_id
                ):
                    updated_item["status"] = "superseded_by_calibration"
                    updated_item["superseded_by"] = calibration_id
                    changed = True
                evidence.append(updated_item)
            branch_records = {
                int(item["raw_index"]): dict(item)
                for item in updated.evaluations
                if item.get("hypothesis_id") == hypothesis.hypothesis_id
                and int(item["raw_index"]) in wanted
            }
            existing = {int(item["repeat_index"]) for item in evidence if item.get("calibration_id") == calibration_id}
            for raw_index in sorted(branch_records):
                if raw_index not in existing:
                    evidence.append(branch_records[raw_index])
                    changed = True
            if not changed:
                continue
            replacement = hypothesis.with_evidence(tuple(evidence))
            combined = scoped_evidence(replacement, updated, include_proposed=False)
            verdict = branch_verdict(combined)
            if verdict["status"] == "eliminated":
                replacement = replacement.with_status("eliminated", verdict["reason"])
            elif replacement.status == "eliminated":
                replacement = replacement.with_status(
                    "active",
                    "partially confirmed calibration scope insufficient for exclusion",
                )
            run.hypotheses = [
                replacement if item.hypothesis_id == replacement.hypothesis_id else item
                for item in run.hypotheses
            ]
            self.store.record_hypothesis(replacement)
            affected.append(hypothesis.hypothesis_id)
        self.store.append_event(
            {
                "type": "calibration_interpretation_confirmed",
                "run_id": run.run_id,
                "calibration_id": calibration_id,
                "raw_indices": sorted(wanted),
                "scope": sorted(wanted),
                "affected_hypotheses": affected,
                "reason": reason,
            }
        )
        return {
            "run_id": run.run_id,
            "calibration_id": calibration_id,
            "confirmed_raw_indices": sorted(wanted),
            "affected_hypotheses": affected,
        }

    def compare_reviewer_conflicts(self, run_id: str) -> dict[str, object]:
        """Compare, side by side, conflicting reviewer revisions of the same raw observations.

        Each revision keeps its own interpretation; nothing is overwritten or
        merged. For every raw observation read differently by two or more
        revisions, each branch shows the per-hypothesis closure judgment under
        that reading, and the entry stays ``pending`` until the user explicitly
        confirms one revision's scope.
        """

        run = self._run(run_id)
        by_raw: dict[int, list[tuple[CalibrationVersion, dict[str, object]]]] = {}
        for calibration in run.calibrations:
            for item in calibration.interpretations:
                by_raw.setdefault(int(item["raw_index"]), []).append((calibration, item))
        conflicts = []
        for raw_index in sorted(by_raw):
            entries = by_raw[raw_index]
            distinct = {int(item["interpreted_index"]) for _, item in entries}
            if len(entries) < 2 or len(distinct) < 2:
                continue
            branches = []
            for calibration, item in entries:
                comparison = compare_branches(run.hypotheses, calibration)
                branches.append(
                    {
                        "calibration_id": calibration.calibration_id,
                        "reviewer": calibration.reviewer,
                        "calibration_status": calibration.status,
                        "interpreted_index": int(item["interpreted_index"]),
                        "interpretation_status": item.get("status", PROPOSED),
                        "hypotheses": comparison["hypotheses"],
                    }
                )
            adopted = [branch for branch in branches if branch["interpretation_status"] == CONFIRMED]
            live = [
                branch
                for branch in branches
                if branch["interpretation_status"] in (PROPOSED, CONFIRMED)
                and branch["calibration_status"] == "active"
            ]
            # Resolution is recomputed from the original observation, the
            # candidate readings still in force and explicit confirmations.
            # A single surviving revision after the other one was withdrawn is
            # merely a still-pending candidate: it must not be auto-adopted
            # just because it is the last revision left.
            if adopted:
                resolution = f"adopted:{adopted[0]['calibration_id']}"
            elif live:
                resolution = "pending"
            else:
                resolution = "resolved"
            conflicts.append(
                {
                    "raw_index": raw_index,
                    "branches": branches,
                    "resolution": resolution,
                    "ambiguity_remains": resolution == "pending",
                }
            )
        return {"run_id": run_id, "conflicts": conflicts}

    def withdraw_calibration(self, calibration_id: str, reason: str = "calibration revision withdrawn") -> dict[str, object]:
        """Withdraw a revision: restore judgments that depended on it only.

        Independent evidence from other observations keeps counting, and map
        versions already built remain on disk as reviewable history.
        """

        calibration = self.calibration(calibration_id)
        if calibration.status == "withdrawn":
            return {"run_id": calibration.run_id, "calibration_id": calibration_id, "restored_hypotheses": []}
        run = self._run(calibration.run_id)
        interpretations = tuple(
            {**item, "status": "withdrawn", "withdrawal_reason": reason}
            for item in calibration.interpretations
        )
        evaluations = tuple(
            {**item, "status": CALIBRATION_WITHDRAWN, "withdrawal_reason": reason}
            for item in calibration.evaluations
        )
        updated = calibration.with_update(
            interpretations=interpretations,
            evaluations=evaluations,
            status="withdrawn",
            withdrawn_at=time.time(),
            reason=reason,
        )
        self._replace_calibration(updated)
        restored: list[str] = []
        for hypothesis in list(run.hypotheses):
            evidence = []
            touched = False
            for item in hypothesis.evidence:
                updated_item = dict(item)
                if updated_item.get("calibration_id") == calibration_id:
                    updated_item["status"] = CALIBRATION_WITHDRAWN
                    updated_item["withdrawal_reason"] = reason
                    touched = True
                elif updated_item.get("superseded_by") == calibration_id:
                    updated_item["status"] = "valid"
                    updated_item.pop("superseded_by", None)
                    touched = True
                evidence.append(updated_item)
            if not touched:
                continue
            replacement = hypothesis.with_evidence(tuple(evidence))
            baseline = baseline_evidence(replacement)
            baseline_reason = elimination_reason(tuple(baseline))
            if baseline_reason and replacement.status in ("active", "eliminated"):
                replacement = replacement.with_status("eliminated", baseline_reason)
            elif replacement.status == "eliminated" and not baseline_reason:
                replacement = replacement.with_status(
                    "active",
                    f"restored: calibration {calibration_id} withdrawn ({reason}); remaining independent evidence insufficient for exclusion",
                )
                restored.append(hypothesis.hypothesis_id)
            run.hypotheses = [
                replacement if item.hypothesis_id == replacement.hypothesis_id else item
                for item in run.hypotheses
            ]
            self.store.record_hypothesis(replacement)
        self.store.append_event(
            {
                "type": "calibration_withdrawn",
                "run_id": run.run_id,
                "calibration_id": calibration_id,
                "restored_hypotheses": restored,
                "reason": reason,
            }
        )
        return {
            "run_id": run.run_id,
            "calibration_id": calibration_id,
            "restored_hypotheses": restored,
        }

    def calibration(self, calibration_id: str) -> CalibrationVersion:
        for run in self.runs.values():
            for calibration in run.calibrations:
                if calibration.calibration_id == calibration_id:
                    return calibration
        for calibration in self.store.load_calibrations():
            if calibration.calibration_id == calibration_id:
                return calibration
        raise KeyError(f"unknown calibration: {calibration_id}")

    def calibrations_for(self, run_id: str) -> list[CalibrationVersion]:
        return list(self._run(run_id).calibrations)

    def _replace_calibration(self, calibration: CalibrationVersion) -> None:
        run = self._run(calibration.run_id)
        run.calibrations = [
            calibration if item.calibration_id == calibration.calibration_id else item
            for item in run.calibrations
        ]
        self.store.record_calibration(calibration)

    def probe_plan(self, plan_id: str) -> ProbePlan:
        for run in self.runs.values():
            for plan in run.probe_plans:
                if plan.plan_id == plan_id:
                    return plan
        raise KeyError(f"unknown probe plan: {plan_id}")

    def probe_plans_for(self, run_id: str) -> list[ProbePlan]:
        return list(self._run(run_id).probe_plans)

    def review_plan(self, review_plan_id: str) -> ReviewPlan:
        for run in self.runs.values():
            for plan in run.review_plans:
                if plan.review_plan_id == review_plan_id:
                    return plan
        for plan in self.store.load_review_plans():
            if plan.review_plan_id == review_plan_id:
                return plan
        raise KeyError(f"unknown review plan: {review_plan_id}")

    def review_plans_for(self, run_id: str) -> list[ReviewPlan]:
        return list(self._run(run_id).review_plans)

    def decision_trace(self, run_id: str) -> dict[str, object]:
        """Replay plans, actual probes, exclusions, withdrawals and the final pick."""

        run = self._run(run_id)
        return {
            "run_id": run_id,
            "plans": [
                {
                    "plan_id": plan.plan_id,
                    "ambiguous_region": plan.ambiguous_region,
                    "branch_hypothesis_id": plan.branch_hypothesis_id,
                    "target_indices": list(plan.target_indices),
                    "predictions": list(plan.predictions),
                    "executions": list(plan.executions),
                }
                for plan in run.probe_plans
            ],
            "review_plans": [
                {
                    "review_plan_id": plan.review_plan_id,
                    "ambiguity_scope": plan.ambiguity_scope,
                    "candidate_ids": list(plan.candidate_ids),
                    "ambiguity_pairs": [list(pair) for pair in plan.ambiguity_pairs],
                    "observations": list(plan.observations),
                    "review_budget": plan.review_budget,
                    "comparison": plan.comparison,
                    "feedback": list(plan.feedback),
                    "reason": plan.reason,
                }
                for plan in run.review_plans
            ],
            "hypotheses": [
                {
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "name": hypothesis.name,
                    "status": hypothesis.status,
                    "reason": hypothesis.reason,
                    "evidence": list(hypothesis.evidence),
                }
                for hypothesis in run.hypotheses
            ],
            "closures": [
                {
                    "closure_id": closure.closure_id,
                    "anchor_index": closure.anchor_index,
                    "repeat_index": closure.repeat_index,
                    "status": closure.status,
                    "reason": closure.reason,
                }
                for closure in run.closures
            ],
            "calibrations": [
                {
                    "calibration_id": calibration.calibration_id,
                    "name": calibration.name,
                    "reviewer": calibration.reviewer,
                    "status": calibration.status,
                    "reason": calibration.reason,
                    "interpretations": list(calibration.interpretations),
                    "evaluations": list(calibration.evaluations),
                }
                for calibration in run.calibrations
            ],
            "observation_provenance": self._observation_provenance(run),
        }

    @staticmethod
    def _observation_provenance(run: MappingRun) -> list[dict[str, object]]:
        """Per raw observation: which revision read it where and what was adopted.

        Every adopted, still-pending and withdrawn reading keeps its source
        (revision and reviewer) so an export explains each map decision without
        collapsing conflicts or double-counting supports.
        """

        by_raw: dict[int, list[dict[str, object]]] = {}
        for calibration in run.calibrations:
            for item in calibration.interpretations:
                by_raw.setdefault(int(item["raw_index"]), []).append(
                    {
                        "calibration_id": calibration.calibration_id,
                        "reviewer": calibration.reviewer,
                        "calibration_status": calibration.status,
                        "interpreted_index": int(item["interpreted_index"]),
                        "status": item.get("status", PROPOSED),
                        "plan_id": item.get("plan_id"),
                        "collected_at": item.get("collected_at"),
                    }
                )
        provenance = []
        for raw_index in sorted(by_raw):
            entries = by_raw[raw_index]
            statuses = {entry["status"] for entry in entries}
            live = [
                entry
                for entry in entries
                if entry["status"] in (PROPOSED, CONFIRMED) and entry["calibration_status"] == "active"
            ]
            live_positions = {entry["interpreted_index"] for entry in live}
            if CONFIRMED in statuses:
                state = "adopted"
            elif not live:
                state = "withdrawn"
            elif len(live_positions) >= 2:
                state = "pending_conflict"
            else:
                state = "proposed"
            provenance.append({"raw_index": raw_index, "state": state, "entries": entries})
        return provenance

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

    def build_map(
        self,
        run_id: str,
        version_label: str,
        resolution: float = 0.1,
        hypothesis_id: str | None = None,
        calibration_id: str | None = None,
        include_proposed_interpretations: bool = True,
    ) -> dict[str, object]:
        run = self._run(run_id)
        hypothesis = self.hypothesis(hypothesis_id) if hypothesis_id is not None else None
        poses = self.corrected_poses(run_id, hypothesis_id)
        calibration = self.calibration(calibration_id) if calibration_id is not None else None
        placement: dict[int, int] = {}
        if calibration is not None:
            for item in calibration.interpretations:
                status = item.get("status", PROPOSED)
                if status == CONFIRMED or (include_proposed_interpretations and status == PROPOSED):
                    placement[int(item["raw_index"])] = int(item["interpreted_index"])
        all_points = []
        for pose, scan in zip(poses, run.scans):
            placed_pose = poses[placement.get(scan.index, scan.index)]
            all_points.extend(placed_pose.transform(point) for point in scan.points)
        margin = 1.0
        min_x = min(x for x, _ in all_points) - margin
        max_x = max(x for x, _ in all_points) + margin
        min_y = min(y for _, y in all_points) - margin
        max_y = max(y for _, y in all_points) + margin
        grid = OccupancyGrid.create((min_x, min_y, max_x, max_y), resolution)
        for pose_index, scan in enumerate(run.scans):
            grid.insert_scan(poses[placement.get(scan.index, pose_index)], list(scan.points))
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
            "calibration_id": calibration_id,
            "calibration_scope": None
            if calibration is None
            else {
                "calibration_id": calibration_id,
                "calibration_status": calibration.status,
                "interpreted_raw_indices": sorted(placement),
                "confirmed_raw_indices": sorted(
                    int(item["raw_index"])
                    for item in calibration.interpretations
                    if item.get("status") == CONFIRMED
                ),
                "proposed_only_raw_indices": sorted(
                    int(item["raw_index"])
                    for item in calibration.interpretations
                    if item.get("status", PROPOSED) == PROPOSED and include_proposed_interpretations
                ),
            },
        }
        self.store.save_map_version(version_id, payload, grid.to_pgm())
        self.store.append_event({"type": "map_version", "run_id": run_id, "map_version_id": version_id, "label": version_label, "hypothesis_id": hypothesis_id, "calibration_id": calibration_id, "calibration_scope": payload["calibration_scope"]})
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
            review_plans = [p for p in self.store.load_review_plans() if p.run_id == run_id]
            calibrations = self.store.load_calibrations()
            run_calibrations = [c for c in calibrations if c.run_id == run_id]
            self.runs[run_id] = MappingRun(run_id, scans, poses, closures, hypotheses, plans, run_calibrations, review_plans)
            self._loaded.add(run_id)
        return self.runs[run_id]
