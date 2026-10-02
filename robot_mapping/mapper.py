from __future__ import annotations

from dataclasses import dataclass

from .alignment import range_signature, signature_distance
from .calibrations import (
    CHANGED,
    UNFULFILLED,
    branch_change,
    classify_reinterpretation,
    reinterpret_points,
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
from .models import CalibrationItem, CalibrationRevision, Hypothesis, LoopClosure, PoseRecord, ProbePlan, Scan2D, new_id
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
    calibrations: list[CalibrationRevision]


class MappingPlatform:
    def __init__(self, root: str):
        self.store = ProjectStore(root)
        self.runs: dict[str, MappingRun] = {}
        self._loaded: set[str] = set()

    def create_run(self, run_id: str) -> MappingRun:
        self.store.run_dir(run_id)
        run = MappingRun(run_id, [], [], [], [], [], [])
        self.runs[run_id] = run
        self._loaded.add(run_id)
        self.store.append_event({"type": "run_created", "run_id": run_id})
        return run

    def add_observation(self, run_id: str, pose: Pose, points: list[tuple[float, float]], sensor_id: str = "lidar_2d") -> None:
        run = self.runs.setdefault(run_id, MappingRun(run_id, [], [], [], [], [], []))
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
                    "raw_index": repeat_index,
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
                raw_identity = int(item.get("raw_index", item["repeat_index"]))
                if raw_identity == repeat_index and item.get("status", "valid") == "valid":
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

    def probe_plan(self, plan_id: str) -> ProbePlan:
        for run in self.runs.values():
            for plan in run.probe_plans:
                if plan.plan_id == plan_id:
                    return plan
        raise KeyError(f"unknown probe plan: {plan_id}")

    # ------------------------------------------------------------------
    # Sensor calibration revisions and reinterpretation of probe evidence
    # ------------------------------------------------------------------

    def propose_calibration_revision(
        self,
        run_id: str,
        reinterpretations: list[dict[str, object]],
        sensor_id: str = "lidar_2d",
        reason: str = "",
    ) -> CalibrationRevision:
        """Register a later calibration version that re-reads existing raw scans.

        ``reinterpretations`` are ``{raw_index, interpreted_index, range_scale,
        angle_offset}`` dicts. No raw scan, timestamp or plan ownership changes,
        no new observation is appended and the common map is not touched: every
        item stays tentative until a human confirms part or all of the scope.
        """

        run = self._run(run_id)
        raw_indices = [int(item["raw_index"]) for item in reinterpretations]
        if len(raw_indices) != len(set(raw_indices)):
            raise ValueError("a calibration revision reinterprets each raw observation at most once")
        for raw_index in raw_indices:
            if raw_index < 0 or raw_index >= len(run.scans):
                raise ValueError(f"raw observation {raw_index} does not exist")
        previous = [c for c in run.calibrations if c.status == "active"]
        base = previous[-1].calibration_id if previous else None
        items = tuple(
            CalibrationItem(
                raw_index=int(item["raw_index"]),
                interpreted_index=int(item["interpreted_index"]),
                range_scale=float(item.get("range_scale", 1.0)),
                angle_offset=float(item.get("angle_offset", 0.0)),
            )
            for item in reinterpretations
        )
        revision = CalibrationRevision(
            calibration_id=new_id("calib"),
            run_id=run_id,
            sensor_id=sensor_id,
            base_calibration_id=base,
            status="active",
            items=items,
            reason=reason,
        )
        run.calibrations.append(revision)
        self.store.record_calibration(revision)
        self.store.append_event(
            {
                "type": "calibration_proposed",
                "run_id": run_id,
                "calibration_id": revision.calibration_id,
                "base_calibration_id": base,
                "raw_indices": raw_indices,
                "reason": reason,
            }
        )
        return revision

    def calibration(self, calibration_id: str) -> CalibrationRevision:
        for run in self.runs.values():
            for revision in run.calibrations:
                if revision.calibration_id == calibration_id:
                    return revision
        raise KeyError(f"unknown calibration revision: {calibration_id}")

    def calibrations_for(self, run_id: str) -> list[CalibrationRevision]:
        return list(self._run(run_id).calibrations)

    @staticmethod
    def _closure_for(hypothesis: Hypothesis, run: MappingRun) -> LoopClosure | None:
        if not hypothesis.closure_ids:
            return None
        closure_id = next(iter(hypothesis.closure_ids))
        return next(item for item in run.closures if item.closure_id == closure_id)

    @staticmethod
    def _raw_verdict(hypothesis: Hypothesis, raw_index: int) -> dict[str, object] | None:
        """The baseline (old-calibration) evidence record for one raw observation."""

        for record in reversed(hypothesis.evidence):
            if int(record.get("raw_index", record["repeat_index"])) == raw_index:
                return record
        return None

    def _branch_eval(
        self,
        hypothesis: Hypothesis,
        run: MappingRun,
        revision: CalibrationRevision | None,
        include_tentative: bool,
    ) -> list[dict[str, object]]:
        """Re-evaluate every raw observation under the old- or new-calibration branch.

        The new-calibration branch only adopts items explicitly confirmed unless
        ``include_tentative`` asks for the full proposal for comparison. Each raw
        observation contributes exactly once, so old and new readings are never
        double counted as two independent supports.
        """

        closure = self._closure_for(hypothesis, run)
        if closure is None:
            return []
        adopted: dict[int, CalibrationItem] = {}
        if revision is not None and revision.status == "active":
            for item in revision.items:
                if include_tentative or item.status == "confirmed":
                    adopted[item.raw_index] = item
        results: list[dict[str, object]] = []
        for raw_index in sorted(
            {int(record.get("raw_index", record["repeat_index"])) for record in hypothesis.evidence}
            | set(adopted)
        ):
            baseline = self._raw_verdict(hypothesis, raw_index)
            if raw_index in adopted:
                evaluation = classify_reinterpretation(run.scans, closure, adopted[raw_index])
                evaluation["calibration_id"] = revision.calibration_id
                evaluation["calibration_item_status"] = adopted[raw_index].status
                evaluation["collected_at"] = run.scans[raw_index].timestamp
                evaluation["plan_id"] = None if baseline is None else baseline.get("plan_id")
                results.append(evaluation)
            elif baseline is not None:
                results.append(
                    {
                        "raw_index": raw_index,
                        "interpreted_index": int(baseline.get("raw_index", baseline["repeat_index"])),
                        "anchor_index": baseline.get("anchor_index"),
                        "calibration_verdict": baseline.get("verdict"),
                        "calibration_id": None,
                    }
                )
        return results

    @staticmethod
    def _branch_elimination(evaluations: list[dict[str, object]]) -> tuple[bool, str]:
        recent = evaluations[-3:]
        failures = [
            item
            for item in recent
            if item.get("calibration_verdict") == "contradictory"
        ]
        if len(failures) >= 2:
            positions = ", ".join(str(item["raw_index"]) for item in failures)
            return True, f"under this calibration branch raw observations {positions} contradict the fold"
        return False, ""

    def compare_calibration_branches(self, run_id: str, calibration_id: str) -> dict[str, object]:
        """Compare keeping the old calibration versus adopting the new one.

        Both branches are evaluated without writing anything: tentative items
        are adopted for the comparison only. The report shows which loop-closure
        judgments flip and which ambiguities remain in place.
        """

        run = self._run(run_id)
        revision = self.calibration(calibration_id)
        per_hypothesis: list[dict[str, object]] = []
        for hypothesis in run.hypotheses:
            if not hypothesis.closure_ids:
                continue
            old_eval = self._branch_eval(hypothesis, run, None, False)
            new_eval = self._branch_eval(hypothesis, run, revision, True)
            old_map = {int(item["raw_index"]): item for item in old_eval}
            new_map = {int(item["raw_index"]): item for item in new_eval}
            changed: list[dict[str, object]] = []
            for raw_index in sorted(set(old_map) | set(new_map)):
                old_item = old_map.get(raw_index)
                new_item = new_map.get(raw_index)
                old_verdict = None if old_item is None else old_item.get("calibration_verdict")
                new_verdict = None if new_item is None else new_item.get("calibration_verdict")
                change = branch_change(old_verdict, new_verdict)
                if change == CHANGED:
                    changed.append(
                        {
                            "raw_index": raw_index,
                            "old_verdict": old_verdict,
                            "new_verdict": new_verdict,
                            "old_anchor": None if old_item is None else old_item.get("anchor_index"),
                            "new_anchor": None if new_item is None else new_item.get("anchor_index"),
                        }
                    )
            old_eliminated, _ = self._branch_elimination(old_eval)
            new_eliminated, new_reason = self._branch_elimination(new_eval)
            unresolved = [
                {
                    "raw_index": item["raw_index"],
                    "verdict": item.get("calibration_verdict"),
                }
                for item in new_eval
                if item.get("calibration_verdict") in (UNFULFILLED, "inconclusive")
            ]
            per_hypothesis.append(
                {
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "hypothesis_name": hypothesis.name,
                    "old_calibration_eliminated": old_eliminated,
                    "new_calibration_eliminated": new_eliminated,
                    "loop_judgment_changed": old_eliminated != new_eliminated,
                    "changed_observations": changed,
                    "remaining_ambiguities": unresolved,
                    "new_elimination_reason": new_reason,
                }
            )
        confirmed = sorted(item.raw_index for item in revision.items if item.status == "confirmed")
        return {
            "run_id": run_id,
            "calibration_id": calibration_id,
            "status": revision.status,
            "confirmed_scope": confirmed,
            "tentative_scope": sorted(item.raw_index for item in revision.items if item.status == "tentative"),
            "common_map_modified": False,
            "hypotheses": per_hypothesis,
        }

    def confirm_calibration_items(
        self,
        calibration_id: str,
        raw_indices: list[int],
        reason: str = "human-confirmed reinterpretation scope",
    ) -> dict[str, object]:
        """Confirm new-calibration readings for a subset of raw observations.

        Each confirmed raw observation replaces its own old-calibration record on
        the hypothesis evidence; the old and new reading can never both count as
        independent support. Unconfirmed items stay tentative and do not change
        any hypothesis or the common map. Loop-closure judgments are re-derived
        from the adopted scope, so a hypothesis eliminated only by readings that
        are now superseded is restored while independent evidence stays valid.
        """

        revision = self.calibration(calibration_id)
        if revision.status == "withdrawn":
            raise ValueError("cannot confirm items of a withdrawn calibration revision")
        run = self._run(revision.run_id)
        wanted = set(raw_indices)
        unknown = wanted - {item.raw_index for item in revision.items}
        if unknown:
            raise ValueError(f"raw observations {sorted(unknown)} are not reinterpreted by {calibration_id}")
        updated_items = tuple(
            item.confirmed(reason) if item.raw_index in wanted and item.status == "tentative" else item
            for item in revision.items
        )
        updated_revision = revision.with_items(updated_items)
        run.calibrations = [
            updated_revision if value.calibration_id == revision.calibration_id else value
            for value in run.calibrations
        ]
        self.store.record_calibration(updated_revision)

        restored: list[str] = []
        re_eliminated: list[str] = []
        for hypothesis in list(run.hypotheses):
            closure = self._closure_for(hypothesis, run)
            if closure is None or hypothesis.status == "confirmed":
                continue
            merged = self._merge_calibration_evidence(run, hypothesis, updated_revision)
            replacement = hypothesis.with_evidence(tuple(merged))
            reason_text = elimination_reason(replacement.evidence)
            if reason_text:
                if replacement.status != "eliminated":
                    re_eliminated.append(hypothesis.hypothesis_id)
                replacement = replacement.with_status(
                    "eliminated",
                    f"{reason_text} (adopted calibration {calibration_id} scope {sorted(wanted)})",
                )
            elif hypothesis.status == "eliminated":
                replacement = replacement.with_status(
                    "active",
                    (
                        f"restored: calibration {calibration_id} superseded observations {sorted(wanted)}; "
                        "remaining adopted evidence is insufficient to exclude the hypothesis"
                    ),
                )
                restored.append(hypothesis.hypothesis_id)
            run.hypotheses = [
                replacement if value.hypothesis_id == hypothesis.hypothesis_id else value
                for value in run.hypotheses
            ]
            self.store.record_hypothesis(replacement)
        self.store.append_event(
            {
                "type": "calibration_scope_confirmed",
                "run_id": run.run_id,
                "calibration_id": calibration_id,
                "confirmed_raw_indices": sorted(wanted),
                "restored_hypotheses": restored,
                "re_eliminated_hypotheses": re_eliminated,
                "reason": reason,
            }
        )
        return {
            "run_id": run.run_id,
            "calibration_id": calibration_id,
            "confirmed_scope": sorted(wanted),
            "restored_hypotheses": restored,
            "re_eliminated_hypotheses": re_eliminated,
            "common_map_modified": False,
        }

    def _merge_calibration_evidence(
        self,
        run: MappingRun,
        hypothesis: Hypothesis,
        revision: CalibrationRevision,
    ) -> list[dict[str, object]]:
        """Produce the adopted evidence list with calibration readings applied once.

        Active confirmed calibrations replace the baseline record for the raw
        observations in their scope; withdrawn calibrations fall back to the old
        reading. Raw indices and timestamps are preserved and each raw scan
        appears at most once.
        """

        closure = self._closure_for(hypothesis, run)
        adopted = {
            item.raw_index: item
            for item in revision.items
            if item.status == "confirmed" and revision.status == "active"
        }
        merged: list[dict[str, object]] = []
        for record in hypothesis.evidence:
            raw_index = int(record.get("raw_index", record["repeat_index"]))
            current = next(
                (item for item in merged if int(item.get("raw_index", item["repeat_index"])) == raw_index),
                None,
            )
            if current is not None:
                continue
            if raw_index in adopted:
                evaluation = classify_reinterpretation(run.scans, closure, adopted[raw_index])
                existing_baseline = record.get("baseline_record")
                baseline_record = (
                    dict(existing_baseline)
                    if isinstance(existing_baseline, dict) and existing_baseline
                    else dict(record)
                )
                record = {
                    **baseline_record,
                    "raw_index": raw_index,
                    "repeat_index": evaluation["interpreted_index"],
                    "anchor_index": evaluation["anchor_index"],
                    "verdict": evaluation["calibration_verdict"],
                    "supports_fold": evaluation["calibration_verdict"] == "consistent",
                    "calibration_id": revision.calibration_id,
                    "calibration_status": "confirmed",
                    "status": "valid" if evaluation["calibration_verdict"] != UNFULFILLED else "unfulfilled",
                    "signature_distance": evaluation.get("signature_distance"),
                    "inlier_fraction": evaluation.get("inlier_fraction"),
                    "rmse": evaluation.get("rmse"),
                    "interpretation": evaluation.get("rationale"),
                    "baseline_record": baseline_record,
                }
            merged.append(record)
        for raw_index, item in adopted.items():
            if any(int(value.get("raw_index", value["repeat_index"])) == raw_index for value in merged):
                continue
            evaluation = classify_reinterpretation(run.scans, closure, item)
            baseline = self._raw_verdict(hypothesis, raw_index)
            merged.append(
                {
                    "raw_index": raw_index,
                    "repeat_index": evaluation["interpreted_index"],
                    "anchor_index": evaluation["anchor_index"],
                    "verdict": evaluation["calibration_verdict"],
                    "supports_fold": evaluation["calibration_verdict"] == "consistent",
                    "status": "valid" if evaluation["calibration_verdict"] != UNFULFILLED else "unfulfilled",
                    "calibration_id": revision.calibration_id,
                    "calibration_status": "confirmed",
                    "collected_at": run.scans[raw_index].timestamp,
                    "plan_id": None if baseline is None else baseline.get("plan_id"),
                    "signature_distance": evaluation.get("signature_distance"),
                    "inlier_fraction": evaluation.get("inlier_fraction"),
                    "rmse": evaluation.get("rmse"),
                    "interpretation": evaluation.get("rationale"),
                    "baseline_record": {} if baseline is None else dict(baseline),
                }
            )
        return merged

    def withdraw_calibration(
        self,
        calibration_id: str,
        reason: str = "calibration revision later withdrawn",
    ) -> dict[str, object]:
        """Withdraw a later calibration and restore the judgments relying on it.

        Evidence reinterpreted only under this calibration falls back to the
        original raw reading; hypotheses excluded by that reinterpretation alone
        reopen. Independent evidence from other observations and from other
        calibrations is not undone.
        """

        revision = self.calibration(calibration_id)
        if revision.status == "withdrawn":
            return {"run_id": revision.run_id, "calibration_id": calibration_id, "restored_hypotheses": []}
        run = self._run(revision.run_id)
        withdrawn_revision = revision.withdrawn(reason)
        run.calibrations = [
            withdrawn_revision if value.calibration_id == calibration_id else value
            for value in run.calibrations
        ]
        self.store.record_calibration(withdrawn_revision)
        restored: list[str] = []
        for hypothesis in list(run.hypotheses):
            closure = self._closure_for(hypothesis, run)
            if closure is None or hypothesis.status == "confirmed":
                continue
            rolled_back = self._rollback_calibration_evidence(hypothesis, calibration_id)
            replacement = hypothesis.with_evidence(tuple(rolled_back))
            reason_text = elimination_reason(replacement.evidence)
            if reason_text:
                replacement = replacement.with_status(
                    "eliminated",
                    f"{reason_text} (calibration {calibration_id} withdrawn; old-calibration evidence restored)",
                )
            elif hypothesis.status == "eliminated":
                replacement = replacement.with_status(
                    "active",
                    f"restored: calibration {calibration_id} withdrawn ({reason}); independent evidence retained",
                )
                restored.append(hypothesis.hypothesis_id)
            run.hypotheses = [
                replacement if value.hypothesis_id == hypothesis.hypothesis_id else value
                for value in run.hypotheses
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

    def _rollback_calibration_evidence(
        self,
        hypothesis: Hypothesis,
        calibration_id: str,
    ) -> list[dict[str, object]]:
        rolled_back: list[dict[str, object]] = []
        for record in hypothesis.evidence:
            raw_index = int(record.get("raw_index", record["repeat_index"]))
            if record.get("calibration_id") == calibration_id:
                baseline = record.get("baseline_record")
                if isinstance(baseline, dict) and baseline:
                    restored_record = {
                        key: value
                        for key, value in baseline.items()
                        if not key.startswith("calibration_")
                        and key not in ("interpretation", "baseline_record")
                    }
                    rolled_back.append(restored_record)
                continue
            cleaned = dict(record)
            cleaned.pop("baseline_record", None)
            rolled_back.append(cleaned)
        return rolled_back

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
                    "closure_ids": list(hypothesis.closure_ids),
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
                    "calibration_id": revision.calibration_id,
                    "sensor_id": revision.sensor_id,
                    "base_calibration_id": revision.base_calibration_id,
                    "status": revision.status,
                    "reason": revision.reason,
                    "created_at": revision.created_at,
                    "withdrawn_at": revision.withdrawn_at,
                    "scope": [
                        {
                            "raw_index": item.raw_index,
                            "interpreted_index": item.interpreted_index,
                            "status": item.status,
                            "range_scale": item.range_scale,
                            "angle_offset": item.angle_offset,
                            "decided_at": item.decided_at,
                            "decision_reason": item.decision_reason,
                        }
                        for item in revision.items
                    ],
                }
                for revision in run.calibrations
            ],
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
        include_tentative: bool = False,
    ) -> dict[str, object]:
        run = self._run(run_id)
        hypothesis = self.hypothesis(hypothesis_id) if hypothesis_id is not None else None
        revision = None
        if calibration_id is not None:
            revision = self.calibration(calibration_id)
            if revision.run_id != run_id:
                raise ValueError("calibration revision belongs to a different run")
        poses = self.corrected_poses(run_id, hypothesis_id)
        reinterpreted: dict[int, CalibrationItem] = {}
        if revision is not None and revision.status == "active":
            for item in revision.items:
                if include_tentative or item.status == "confirmed":
                    reinterpreted[item.raw_index] = item

        def scan_points(index: int, scan: Scan2D) -> list[tuple[float, float]]:
            item = reinterpreted.get(index)
            if item is None:
                return list(scan.points)
            return reinterpret_points(list(scan.points), item)

        all_points = []
        for index, (pose, scan) in enumerate(zip(poses, run.scans)):
            all_points.extend(pose.transform(point) for point in scan_points(index, scan))
        margin = 1.0
        min_x = min(x for x, _ in all_points) - margin
        max_x = max(x for x, _ in all_points) + margin
        min_y = min(y for _, y in all_points) - margin
        max_y = max(y for _, y in all_points) + margin
        grid = OccupancyGrid.create((min_x, min_y, max_x, max_y), resolution)
        for index, (pose, scan) in enumerate(zip(poses, run.scans)):
            grid.insert_scan(pose, scan_points(index, scan))
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
            "calibration_id": None if revision is None else revision.calibration_id,
            "calibration_status": None if revision is None else revision.status,
            "calibration_scope": "preview_tentative" if include_tentative else ("confirmed" if revision is not None else "none"),
            "reinterpreted_raw_indices": sorted(reinterpreted),
            "poses": [pose.to_dict() for pose in poses],
            "raw_scan_count": len(run.scans),
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
            self.runs[run_id] = MappingRun(run_id, scans, poses, closures, hypotheses, plans, calibrations)
            self._loaded.add(run_id)
        return self.runs[run_id]
