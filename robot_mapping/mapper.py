from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

from .alignment import range_signature, signature_distance
from .calibration import (
    CALIBRATION_WITHDRAWN,
    CONFIRMED,
    PROPOSED,
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
    ReviewBatch,
    Scan2D,
    new_id,
)
from .occupancy import OccupancyGrid
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
    review_batches: list[ReviewBatch]


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

    def register_calibration(
        self,
        run_id: str,
        name: str,
        reason: str = "",
        batch_id: str | None = None,
        reviewer_id: str | None = None,
    ) -> CalibrationVersion:
        """Register a new sensor calibration revision without touching raw data or the map."""

        run = self._run(run_id)
        calibration = CalibrationVersion(
            calibration_id=new_id("cal"),
            run_id=run_id,
            name=name,
            reason=reason,
            batch_id=batch_id,
            reviewer_id=reviewer_id,
        )
        run.calibrations.append(calibration)
        if batch_id is not None:
            batch = self.review_batch(batch_id)
            if batch.run_id != run_id:
                raise ValueError("review batch belongs to a different run")
            self._replace_review_batch(
                batch.with_revision_ids(tuple(sorted(set(batch.revision_ids + (calibration.calibration_id,)))))
            )
        self.store.record_calibration(calibration)
        self.store.append_event(
            {
                "type": "calibration_registered",
                "run_id": run_id,
                "calibration_id": calibration.calibration_id,
                "name": name,
                "reason": reason,
                "batch_id": batch_id,
                "reviewer_id": reviewer_id,
            }
        )
        return calibration

    def create_review_batch(
        self,
        run_id: str,
        raw_indices: list[int],
        name: str = "independent reviewer batch",
    ) -> ReviewBatch:
        """Freeze exactly the user-selected raw observation scope for review."""

        run = self._run(run_id)
        wanted = sorted({int(index) for index in raw_indices})
        if not wanted:
            raise ValueError("review batch must contain at least one raw observation")
        unknown = [index for index in wanted if index < 0 or index >= len(run.scans)]
        if unknown:
            raise ValueError(f"unknown raw observations {unknown}")
        scan_path = self.store.run_dir(run_id) / "scans.jsonl"
        digest = hashlib.sha256()
        selected_bytes = 0
        with scan_path.open("rb") as stream:
            for line in stream:
                item = json.loads(line.decode("utf-8"))
                if int(item["index"]) in wanted:
                    digest.update(line)
                    selected_bytes += len(line)
        batch = ReviewBatch(
            batch_id=new_id("review"),
            run_id=run_id,
            name=name,
            raw_indices=tuple(wanted),
            raw_scan_sha256=digest.hexdigest(),
            raw_scan_bytes=selected_bytes,
        )
        run.review_batches.append(batch)
        self.store.record_review_batch(batch)
        self.store.append_event(
            {
                "type": "review_batch_created",
                "run_id": run_id,
                "batch_id": batch.batch_id,
                "raw_indices": wanted,
                "raw_scan_sha256": batch.raw_scan_sha256,
            }
        )
        return batch

    def register_reviewer_revision(
        self,
        batch_id: str,
        reviewer_id: str,
        name: str,
        reason: str = "",
    ) -> CalibrationVersion:
        batch = self.review_batch(batch_id)
        existing = [
            calibration
            for calibration in self._run(batch.run_id).calibrations
            if calibration.batch_id == batch_id and calibration.reviewer_id == reviewer_id
        ]
        if existing:
            raise ValueError(f"reviewer {reviewer_id} already has revision {existing[0].calibration_id}")
        return self.register_calibration(
            batch.run_id,
            name,
            reason=reason,
            batch_id=batch_id,
            reviewer_id=reviewer_id,
        )

    def compare_review_batch(self, batch_id: str) -> dict[str, object]:
        """Place competing reviewer explanations side by side without selecting one."""

        batch = self.review_batch(batch_id)
        run = self._run(batch.run_id)
        revisions = self._batch_revisions(batch)
        active_revisions = [revision for revision in revisions if revision.status != "withdrawn"]
        interpretation_maps = [
            {
                int(item["raw_index"]): item
                for item in revision.interpretations
                if int(item["raw_index"]) in set(batch.raw_indices)
            }
            for revision in active_revisions
        ]
        conflicts = []
        for raw_index in batch.raw_indices:
            present = [item.get(raw_index) for item in interpretation_maps]
            present = [item for item in present if item is not None and item.get("status") != "withdrawn"]
            if any(item.get("status") == CONFIRMED for item in present):
                continue
            by_revision = {}
            original_hypotheses = {hypothesis.hypothesis_id: hypothesis for hypothesis in run.hypotheses if hypothesis.closure_ids}
            original_status = "valid" if any(
                item.get("status", "valid") == "valid"
                and not item.get("calibration_id")
                and int(item["repeat_index"]) == raw_index
                for hypothesis in original_hypotheses.values()
                for item in hypothesis.evidence
            ) else "not_observed"
            original_records = [
                item
                for hypothesis in original_hypotheses.values()
                for item in hypothesis.evidence
                if int(item["repeat_index"]) == raw_index and not item.get("calibration_id")
            ]
            if original_records:
                original_status = str(original_records[0].get("status", "valid"))
            source_positions = {raw_index} if original_status == "valid" else set()
            by_revision["original"] = {
                "revision_id": "original",
                "reviewer_id": "original_survey",
                "interpreted_index": raw_index,
                "status": original_status,
            }
            for revision, item in zip(active_revisions, interpretation_maps):
                if raw_index not in item:
                    continue
                interpretation = item[raw_index]
                source_positions.add(int(interpretation["interpreted_index"]))
                by_revision[revision.calibration_id] = {
                    "revision_id": revision.calibration_id,
                    "reviewer_id": revision.reviewer_id,
                    "interpreted_index": int(interpretation["interpreted_index"]),
                    "status": interpretation.get("status", PROPOSED),
                }
            if len(source_positions) <= 1:
                continue
            per_hypothesis = []
            for hypothesis in run.hypotheses:
                if not hypothesis.closure_ids:
                    continue
                branches = {
                    "original": self._raw_review_verdict(hypothesis, raw_index),
                }
                for revision, item in zip(active_revisions, interpretation_maps):
                    interpretation = item.get(raw_index)
                    if interpretation is None:
                        continue
                    branches[revision.calibration_id] = {
                        "reviewer_id": revision.reviewer_id,
                        "interpreted_index": int(interpretation["interpreted_index"]),
                        "verdict": self._evaluation_verdict(revision, hypothesis.hypothesis_id, raw_index),
                        "branch_status": branch_verdict(calibrated_evidence(hypothesis, revision))["status"],
                    }
                per_hypothesis.append(
                    {
                        "hypothesis_id": hypothesis.hypothesis_id,
                        "hypothesis_name": hypothesis.name,
                        "branches": branches,
                    }
                )
            conflicts.append(
                {
                    "raw_index": raw_index,
                    "status": "unresolved_conflict",
                    "sources": list(by_revision.values()),
                    "hypothesis_impacts": per_hypothesis,
                }
            )
        return {
            "batch_id": batch_id,
            "run_id": batch.run_id,
            "selected_raw_indices": list(batch.raw_indices),
            "raw_scan_sha256": batch.raw_scan_sha256,
            "raw_scan_bytes": batch.raw_scan_bytes,
            "raw_scan_unchanged": self._batch_raw_scan_digest(batch) == batch.raw_scan_sha256,
            "revisions": [self._revision_summary(revision) for revision in revisions],
            "conflicts": conflicts,
        }

    def confirm_review_scope(
        self,
        batch_id: str,
        revision_id: str,
        raw_indices: list[int],
        reason: str = "user selected reviewer observation scope",
    ) -> dict[str, object]:
        """Adopt exactly the chosen observations from one reviewer revision."""

        batch = self.review_batch(batch_id)
        run = self._run(batch.run_id)
        revision = self._batch_revision(batch, revision_id)
        wanted = sorted({int(index) for index in raw_indices})
        outside = [index for index in wanted if index not in set(batch.raw_indices)]
        if outside:
            raise ValueError(f"observations {outside} are outside the selected review batch")
        available = {
            int(item["raw_index"]): item
            for item in revision.interpretations
            if item.get("status") != "withdrawn"
        }
        unknown = [index for index in wanted if index not in available]
        if unknown:
            raise ValueError(f"revision has no active interpretation for observations {unknown}")
        other_decision = []
        for other in self._batch_revisions(batch):
            if other.calibration_id == revision_id or other.status == "withdrawn":
                continue
            for item in other.interpretations:
                if int(item["raw_index"]) in wanted and item.get("status") == CONFIRMED:
                    other_decision.append((int(item["raw_index"]), other.calibration_id))
        if other_decision:
            raise ValueError(f"existing competing decisions must be withdrawn first: {other_decision}")
        already = sorted(
            int(item["raw_index"])
            for item in revision.interpretations
            if int(item["raw_index"]) in set(wanted) and item.get("status") == CONFIRMED
        )
        newly_confirmed = sorted(set(wanted) - set(already))
        if newly_confirmed:
            self.confirm_calibration_interpretation(revision_id, newly_confirmed, reason)
        event = {
            "type": "review_scope_confirmed",
            "run_id": run.run_id,
            "batch_id": batch_id,
            "revision_id": revision_id,
            "reviewer_id": revision.reviewer_id,
            "raw_indices": wanted,
            "already_confirmed": already,
            "newly_confirmed": newly_confirmed,
            "reason": reason,
        }
        if newly_confirmed:
            self.store.append_event(event)
        return {
            "run_id": run.run_id,
            "batch_id": batch_id,
            "revision_id": revision_id,
            "reviewer_id": revision.reviewer_id,
            "confirmed_raw_indices": wanted,
            "already_confirmed": already,
            "newly_confirmed": newly_confirmed,
        }

    def withdraw_reviewer_revision(
        self,
        batch_id: str,
        revision_id: str,
        raw_indices: list[int] | None = None,
        reason: str = "reviewer revision withdrawn",
    ) -> dict[str, object]:
        """Withdraw all or an explicitly scoped part of one reviewer revision."""

        batch = self.review_batch(batch_id)
        run = self._run(batch.run_id)
        revision = self._batch_revision(batch, revision_id)
        scope = set(batch.raw_indices) if raw_indices is None else {int(index) for index in raw_indices}
        unknown = sorted(scope - set(batch.raw_indices))
        if unknown:
            raise ValueError(f"observations {unknown} are outside the selected review batch")
        available = {int(item["raw_index"]) for item in revision.interpretations if int(item["raw_index"]) in scope}
        missing = sorted(scope - available)
        if missing:
            raise ValueError(f"revision has no interpretation for observations {missing}")
        interpretations = []
        for item in revision.interpretations:
            updated = dict(item)
            if int(updated["raw_index"]) in scope and updated.get("status") != "withdrawn":
                updated["status"] = "withdrawn"
                updated["withdrawal_reason"] = reason
            interpretations.append(updated)
        evaluations = []
        for item in revision.evaluations:
            updated = dict(item)
            if int(updated["raw_index"]) in scope and updated.get("status", "valid") == "valid":
                updated["status"] = CALIBRATION_WITHDRAWN
                updated["withdrawal_reason"] = reason
            evaluations.append(updated)
        remaining = [item for item in interpretations if item.get("status") != "withdrawn"]
        status = "withdrawn" if not remaining else "partially_withdrawn"
        withdrawn_at = time.time() if status == "withdrawn" else revision.withdrawn_at
        updated_revision = revision.with_update(
            interpretations=tuple(interpretations),
            evaluations=tuple(evaluations),
            status=status,
            withdrawn_at=withdrawn_at,
            reason=reason if status == "withdrawn" else revision.reason,
        )
        self._replace_calibration(updated_revision)
        restored = self._restore_withdrawn_interpretation_effects(run, revision_id, scope, reason)
        self.store.append_event(
            {
                "type": "reviewer_revision_withdrawn",
                "run_id": run.run_id,
                "batch_id": batch_id,
                "revision_id": revision_id,
                "reviewer_id": revision.reviewer_id,
                "raw_indices": sorted(scope),
                "remaining_interpretation_indices": [int(item["raw_index"]) for item in remaining],
                "restored_hypotheses": restored,
                "reason": reason,
            }
        )
        return {
            "run_id": run.run_id,
            "batch_id": batch_id,
            "revision_id": revision_id,
            "reviewer_id": revision.reviewer_id,
            "withdrawn_raw_indices": sorted(scope),
            "revision_status": status,
            "restored_hypotheses": restored,
        }

    def export_review_batch(self, batch_id: str, write_file: bool = True) -> dict[str, object]:
        comparison = self.compare_review_batch(batch_id)
        batch = self.review_batch(batch_id)
        run = self._run(batch.run_id)
        revisions = self._batch_revisions(batch)
        adopted, withdrawn, revision_sources = [], [], []
        for revision in revisions:
            revision_sources.append(self._revision_summary(revision))
            for item in revision.interpretations:
                raw_index = int(item["raw_index"])
                if raw_index not in set(batch.raw_indices):
                    continue
                scan = run.scans[raw_index]
                source = {
                    "raw_index": raw_index,
                    "scan_id": scan.scan_id,
                    "sensor_id": scan.sensor_id,
                    "collected_at": scan.timestamp,
                    "revision_id": revision.calibration_id,
                    "reviewer_id": revision.reviewer_id,
                    "interpreted_index": int(item["interpreted_index"]),
                    "status": item.get("status", PROPOSED),
                    "plan_id": item.get("plan_id"),
                    "note": item.get("note", ""),
                    "reason": item.get("confirmation_reason") or item.get("withdrawal_reason") or revision.reason,
                }
                if item.get("status") == CONFIRMED:
                    adopted.append(source)
                elif item.get("status") == "withdrawn":
                    withdrawn.append(source)
        payload = {
            **comparison,
            "adopted": sorted(adopted, key=lambda item: (item["raw_index"], item["revision_id"])),
            "withdrawn": sorted(withdrawn, key=lambda item: (item["raw_index"], item["revision_id"])),
            "remaining_conflicts": comparison["conflicts"],
            "existing_decisions": [item for item in adopted],
            "revision_sources": revision_sources,
        }
        if write_file:
            path = self.store.save_review_export(batch_id, payload)
            payload["export_path"] = str(path)
        return payload

    def review_batch(self, batch_id: str) -> ReviewBatch:
        for run in self.runs.values():
            for batch in run.review_batches:
                if batch.batch_id == batch_id:
                    return batch
        for batch in self.store.load_review_batches():
            if batch.batch_id == batch_id:
                return batch
        raise KeyError(f"unknown review batch: {batch_id}")

    def review_batches_for(self, run_id: str) -> list[ReviewBatch]:
        return list(self._run(run_id).review_batches)

    def _replace_review_batch(self, batch: ReviewBatch) -> None:
        run = self._run(batch.run_id)
        run.review_batches = [
            batch if item.batch_id == batch.batch_id else item
            for item in run.review_batches
        ]
        self.store.record_review_batch(batch)

    def _batch_revisions(self, batch: ReviewBatch) -> list[CalibrationVersion]:
        return [
            revision
            for revision in self._run(batch.run_id).calibrations
            if revision.batch_id == batch.batch_id
        ]

    def _batch_revision(self, batch: ReviewBatch, revision_id: str) -> CalibrationVersion:
        revisions = [item for item in self._batch_revisions(batch) if item.calibration_id == revision_id]
        if not revisions:
            raise ValueError(f"revision {revision_id} does not belong to batch {batch.batch_id}")
        return revisions[0]

    def _batch_raw_scan_digest(self, batch: ReviewBatch) -> str:
        wanted = set(batch.raw_indices)
        digest = hashlib.sha256()
        scan_path = self.store.run_dir(batch.run_id) / "scans.jsonl"
        with scan_path.open("rb") as stream:
            for line in stream:
                item = json.loads(line.decode("utf-8"))
                if int(item["index"]) in wanted:
                    digest.update(line)
        return digest.hexdigest()

    @staticmethod
    def _revision_summary(revision: CalibrationVersion) -> dict[str, object]:
        return {
            "revision_id": revision.calibration_id,
            "reviewer_id": revision.reviewer_id,
            "name": revision.name,
            "status": revision.status,
            "reason": revision.reason,
            "raw_indices": sorted(int(item["raw_index"]) for item in revision.interpretations),
        }

    @staticmethod
    def _evaluation_verdict(
        revision: CalibrationVersion,
        hypothesis_id: str,
        raw_index: int,
    ) -> str | None:
        for item in revision.evaluations:
            if item.get("hypothesis_id") == hypothesis_id and int(item["raw_index"]) == raw_index:
                if item.get("status", "valid") != "valid":
                    return None
                return item.get("verdict")
        return None

    @staticmethod
    def _raw_review_verdict(hypothesis: Hypothesis, raw_index: int) -> dict[str, object]:
        for item in hypothesis.evidence:
            if int(item["repeat_index"]) != raw_index or item.get("calibration_id"):
                continue
            return {
                "status": item.get("status", "valid"),
                "verdict": item.get("verdict"),
                "branch_status": branch_verdict(baseline_evidence(hypothesis))["status"],
            }
        return {"status": "not_observed", "verdict": None, "branch_status": branch_verdict(baseline_evidence(hypothesis))["status"]}

    def _restore_withdrawn_interpretation_effects(
        self,
        run: MappingRun,
        revision_id: str,
        raw_scope: set[int],
        reason: str,
    ) -> list[str]:
        restored: list[str] = []
        for hypothesis in list(run.hypotheses):
            evidence = []
            touched = False
            for item in hypothesis.evidence:
                updated_item = dict(item)
                if updated_item.get("calibration_id") == revision_id and int(updated_item["repeat_index"]) in raw_scope:
                    updated_item["status"] = CALIBRATION_WITHDRAWN
                    updated_item["withdrawal_reason"] = reason
                    touched = True
                elif updated_item.get("superseded_by") == revision_id and int(updated_item["repeat_index"]) in raw_scope:
                    updated_item["status"] = "valid"
                    updated_item.pop("superseded_by", None)
                    updated_item.pop("supersession_reason", None)
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
                    f"restored: reviewer revision {revision_id} withdrawn ({reason})",
                )
                restored.append(hypothesis.hypothesis_id)
            run.hypotheses = [
                replacement if item.hypothesis_id == replacement.hypothesis_id else item
                for item in run.hypotheses
            ]
            self.store.record_hypothesis(replacement)
        return restored

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
        if calibration.batch_id is not None:
            review_batch = self.review_batch(calibration.batch_id)
            if raw_index not in set(review_batch.raw_indices):
                raise ValueError(f"raw observation {raw_index} is outside review batch {calibration.batch_id}")
        raw_scan = run.scans[raw_index]
        interpretations = [dict(item) for item in calibration.interpretations if int(item["raw_index"]) != raw_index]
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
        wanted = {int(index) for index in raw_indices}
        available = {int(item["raw_index"]) for item in calibration.interpretations}
        unknown = wanted - available
        if unknown:
            raise ValueError(f"no reinterpretation for raw observations {sorted(unknown)}")
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

    def withdraw_calibration(self, calibration_id: str, reason: str = "calibration revision withdrawn") -> dict[str, object]:
        """Withdraw a revision: restore judgments that depended on it only.

        Independent evidence from other observations keeps counting, and map
        versions already built remain on disk as reviewable history.
        """

        calibration = self.calibration(calibration_id)
        if calibration.status == "withdrawn":
            return {"run_id": calibration.run_id, "calibration_id": calibration_id, "restored_hypotheses": []}
        run = self._run(calibration.run_id)
        interpretations = []
        evaluations = []
        for item in calibration.interpretations:
            updated_item = dict(item)
            if updated_item.get("status") != "withdrawn":
                updated_item["status"] = "withdrawn"
                updated_item["withdrawal_reason"] = reason
            interpretations.append(updated_item)
        for item in calibration.evaluations:
            updated_item = dict(item)
            if updated_item.get("status", "valid") == "valid":
                updated_item["status"] = CALIBRATION_WITHDRAWN
                updated_item["withdrawal_reason"] = reason
            evaluations.append(updated_item)
        updated = calibration.with_update(
            interpretations=tuple(interpretations),
            evaluations=tuple(evaluations),
            status="withdrawn",
            withdrawn_at=time.time(),
            reason=reason,
        )
        self._replace_calibration(updated)
        scope = {int(item["raw_index"]) for item in calibration.interpretations}
        restored = self._restore_withdrawn_interpretation_effects(run, calibration_id, scope, reason)
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
                    "batch_id": calibration.batch_id,
                    "reviewer_id": calibration.reviewer_id,
                    "name": calibration.name,
                    "status": calibration.status,
                    "reason": calibration.reason,
                    "interpretations": list(calibration.interpretations),
                    "evaluations": list(calibration.evaluations),
                }
                for calibration in run.calibrations
            ],
            "review_batches": [batch.to_dict() for batch in run.review_batches],
        }

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
        if calibration is not None and calibration.status == "withdrawn":
            raise ValueError("cannot build from a fully withdrawn revision")
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
            calibrations = [c for c in self.store.load_calibrations() if c.run_id == run_id]
            batches = [b for b in self.store.load_review_batches() if b.run_id == run_id]
            self.runs[run_id] = MappingRun(run_id, scans, poses, closures, hypotheses, plans, calibrations, batches)
            self._loaded.add(run_id)
        return self.runs[run_id]
