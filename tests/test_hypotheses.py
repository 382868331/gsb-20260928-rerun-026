from __future__ import annotations

import tempfile
import unittest

from robot_mapping.demo import load_scenario_data, run_competing_corridor_demo
from robot_mapping.demo import _load_one
from robot_mapping.hypotheses import closures_conflict
from robot_mapping.mapper import MappingPlatform
from robot_mapping.simulator import similar_corridor_scenario, true_loop_scenario


class CompetingHypothesisTest(unittest.TestCase):
    def test_conflicting_closures_share_repeat_but_differ_on_anchor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            scenario = similar_corridor_scenario()
            load_scenario_data(platform, "similar", scenario)
            first = platform.propose_candidate("similar", 0, 33)
            second = platform.propose_candidate("similar", 14, 33)
            self.assertTrue(closures_conflict(first, second))

    def test_similar_corridors_keep_both_explanations_until_later_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            scenario = similar_corridor_scenario()
            summary = run_competing_corridor_demo(platform, "similar", scenario)
            hypotheses = {h.name: h for h in platform.hypotheses_for("similar")}
            fold = next(h for h in hypotheses.values() if h.closure_ids)
            distinct = next(h for h in hypotheses.values() if not h.closure_ids)
            self.assertEqual(fold.status, "eliminated")
            self.assertEqual(distinct.status, "confirmed")
            self.assertGreaterEqual(summary["eliminating_observation"], 34)
            contradictions = [item for item in fold.evidence if not item["supports_fold"]]
            self.assertGreaterEqual(len(contradictions), 2)
            self.assertIn("repeat indices", summary["elimination_reason"])
            # the trial preview must never have rewritten the common map
            common = platform.build_map("similar", "common_check")
            self.assertEqual(common["accepted_closures"], [])

    def test_eliminated_hypothesis_can_be_reopened_and_compared(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            scenario = similar_corridor_scenario()
            summary = run_competing_corridor_demo(platform, "similar", scenario)
            fold = platform.hypothesis(summary["fold_hypothesis"])
            reopened = platform.reopen_hypothesis(fold.hypothesis_id)
            self.assertEqual(reopened.status, "active")
            retained = [item for item in reopened.evidence if not item["supports_fold"]]
            self.assertTrue(retained)
            preview = platform.build_map("similar", "compare_reopened", hypothesis_id=reopened.hypothesis_id)
            self.assertEqual(preview["hypothesis_id"], reopened.hypothesis_id)

    def test_local_correction_leaves_unaffected_poses_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            scenario = true_loop_scenario()
            load_scenario_data(platform, "loop", scenario)
            proposal = platform.propose_candidate("loop", scenario.expected_anchor, scenario.expected_repeat)
            platform.accept_closure(proposal.closure_id)
            raw = [record.pose for record in platform.runs["loop"].odometry]
            corrected = platform.corrected_poses("loop")
            span = range(min(proposal.anchor_index, proposal.repeat_index), max(proposal.anchor_index, proposal.repeat_index) + 1)
            outside = [i for i in range(len(raw)) if i not in span]
            self.assertTrue(outside)
            for index in outside:
                self.assertEqual(raw[index].as_tuple(), corrected[index].as_tuple())
            self.assertNotEqual(raw[proposal.repeat_index].as_tuple(), corrected[proposal.repeat_index].as_tuple())
            platform.withdraw_closure(proposal.closure_id)
            withdrawn = platform.corrected_poses("loop")
            self.assertEqual([p.as_tuple() for p in raw], [p.as_tuple() for p in withdrawn])

    def test_probe_plan_predicts_exclusion_power_without_becoming_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            scenario = similar_corridor_scenario()
            arrival = scenario.expected_repeat - 10
            platform.create_run("similar")
            from robot_mapping.demo import _load_range
            _load_range(platform, "similar", scenario, 0, arrival + 1)
            hypotheses = platform.propose_competing_closures(
                "similar", [(0, arrival, "same corridor: fold onto first corridor")]
            )
            fold = hypotheses[0]
            # featureless stretch (anchor 3), distinctive notch (anchor 14), beyond the map
            plan = platform.create_probe_plan("similar", "junction", [arrival + 3, arrival + 14, arrival + 40])
            comparison = platform.compare_probe_plan(plan.plan_id)
            labels = {
                (target["planned_index"], item["hypothesis_name"]): item["prediction"]
                for target in comparison["targets"]
                for item in target["predictions"]
            }
            self.assertEqual(labels[(arrival + 3, fold.name)], "indistinguishable")
            self.assertEqual(labels[(arrival + 14, fold.name)], "would_exclude_on_conflict")
            self.assertEqual(labels[(arrival + 40, fold.name)], "insufficient_information")
            self.assertTrue(any(value == "indistinguishable" for value in labels.values()))
            # nothing executed => no evidence, no status change, no common-map closure
            self.assertTrue(all(len(h.evidence) == 0 for h in platform.hypotheses_for("similar")))
            common = platform.build_map("similar", "common_unchanged")
            self.assertEqual(common["accepted_closures"], [])

    def test_withdrawing_an_observation_restores_affected_plan_but_keeps_independent_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            scenario = similar_corridor_scenario()
            summary = run_competing_corridor_demo(platform, "similar", scenario)
            fold = platform.hypothesis(summary["fold_hypothesis"])
            self.assertEqual(fold.status, "eliminated")
            # withdraw every decisive contradictory observation except the last one
            contradictory = sorted(
                item["repeat_index"]
                for item in fold.evidence
                if not item["supports_fold"] and item.get("status", "valid") == "valid"
            )
            self.assertGreaterEqual(len(contradictory), 2)
            first = platform.withdraw_observation(
                "similar", contradictory[0], "belongs to another corridor", reassigned_corridor=True
            )
            fold = platform.hypothesis(fold.hypothesis_id)
            # independent evidence from the other observation is still valid
            self.assertTrue(any(item.get("status", "valid") == "valid" and not item["supports_fold"] for item in fold.evidence))
            result = platform.withdraw_observation("similar", contradictory[1], "sensor glitch")
            restored_ids = set(first["restored_hypotheses"]) | set(result["restored_hypotheses"])
            self.assertIn(fold.hypothesis_id, restored_ids)
            restored = platform.hypothesis(fold.hypothesis_id)
            self.assertEqual(restored.status, "active")
            valid_contradictions = [
                item for item in restored.evidence if not item["supports_fold"] and item.get("status", "valid") == "valid"
            ]
            self.assertEqual(valid_contradictions, [])
            # unexecuted plan item still has no evidence attached
            self.assertTrue(all(int(item["repeat_index"]) != 73 for item in restored.evidence))

    def test_branch_plan_advances_only_inside_its_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            scenario = similar_corridor_scenario()
            arrival = scenario.expected_repeat - 10
            platform.create_run("similar")
            from robot_mapping.demo import _load_range
            _load_range(platform, "similar", scenario, 0, arrival + 1)
            hypotheses = platform.propose_competing_closures(
                "similar", [(0, arrival, "same corridor: fold onto first corridor")]
            )
            fold = hypotheses[0]
            distinct = hypotheses[1]
            branch_plan = platform.create_probe_plan(
                "similar",
                "fold branch notch",
                [arrival + 11],
                branch_hypothesis_id=fold.hypothesis_id,
            )
            # the plan is scoped: only the fold branch + its null comparator are predicted
            predicted_hypotheses = {item["hypothesis_id"] for item in branch_plan.predictions}
            self.assertEqual(predicted_hypotheses, {fold.hypothesis_id, distinct.hypothesis_id})
            _load_range(platform, "similar", scenario, arrival + 1, arrival + 12)
            result = platform.feed_observation_to_hypotheses(
                "similar", arrival + 11, plan_id=branch_plan.plan_id
            )
            evaluated = {item["hypothesis_id"] for item in result["hypothesis_results"]}
            self.assertEqual(evaluated, {fold.hypothesis_id})

    def test_decision_trace_survives_save_and_reopen_and_explains_final_choice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            scenario = similar_corridor_scenario()
            summary = run_competing_corridor_demo(platform, "similar", scenario)
            plan_id = summary["probe_plan"]

            reopened = MappingPlatform(directory)
            plans = reopened.probe_plans_for("similar")
            self.assertTrue(any(plan.plan_id == plan_id for plan in plans))
            stored_plan = next(plan for plan in plans if plan.plan_id == plan_id)
            executed = {item["actual_index"] for item in stored_plan.executions}
            self.assertIn(35, executed)
            self.assertNotIn(73, executed)
            notes = [
                result["expected_vs_actual"]
                for item in stored_plan.executions
                for result in item["results"]
                if result.get("expected_vs_actual")
            ]
            self.assertTrue(any("unlike the plan" in note for note in notes))
            trace = reopened.decision_trace("similar")
            selected = next(h for h in trace["hypotheses"] if h["status"] == "confirmed")
            self.assertEqual(selected["hypothesis_id"], summary["selected_hypothesis"])
            common = reopened.build_map("similar", "reloaded_common")
            self.assertEqual(common["accepted_closures"], [])


if __name__ == "__main__":
    unittest.main()
