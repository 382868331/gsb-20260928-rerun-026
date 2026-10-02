from __future__ import annotations

import tempfile
import unittest

from robot_mapping.demo import _load_one, _load_range
from robot_mapping.mapper import MappingPlatform
from robot_mapping.simulator import similar_corridor_scenario


def _eliminated_fold(platform: MappingPlatform, run_id: str = "similar"):
    scenario = similar_corridor_scenario()
    arrival = scenario.expected_repeat - 10
    platform.create_run(run_id)
    _load_range(platform, run_id, scenario, 0, arrival + 1)
    hypotheses = platform.propose_competing_closures(
        run_id, [(0, arrival, "same corridor: fold onto first corridor")]
    )
    fold, distinct = hypotheses[0], hypotheses[1]
    plan = platform.create_probe_plan(
        run_id,
        "junction after the second similar corridor",
        [arrival + 13, arrival + 14],
        branch_hypothesis_id=fold.hypothesis_id,
    )
    for index in range(arrival + 1, len(scenario.truth)):
        _load_one(platform, run_id, scenario, index)
        linked = plan.plan_id if index in plan.target_indices else None
        platform.feed_observation_to_hypotheses(run_id, index, plan_id=linked, collected_at=float(index))
    return scenario, fold, distinct, plan


class CalibrationReinterpretationTest(unittest.TestCase):
    def test_reinterpretation_is_not_a_new_observation_and_keeps_raw_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            scenario, fold, _distinct, _plan = _eliminated_fold(platform)
            raw_count = len(platform.runs["similar"].scans)
            calibration = platform.register_calibration("similar", "adjacent-corridor revision")
            result = platform.reinterpret_observation(
                calibration.calibration_id,
                raw_index=46,
                interpreted_index=34,
                plan_id=None,
                note="sensor bias moved the reading to the neighboring corridor",
            )
            self.assertEqual(result["raw_index"], 46)
            self.assertEqual(result["collected_at"], 46.0)
            self.assertEqual(len(platform.runs["similar"].scans), raw_count)
            stored = next(item for item in platform.calibration(calibration.calibration_id).interpretations if item["raw_index"] == 46)
            self.assertEqual(stored["interpreted_index"], 34)
            self.assertEqual(stored["collected_at"], 46.0)
            self.assertEqual(stored["status"], "proposed")
            # proposing a reinterpretation alone does not change the old judgment or the map
            self.assertEqual(platform.hypothesis(fold.hypothesis_id).status, "eliminated")
            common = platform.build_map("similar", "common_after_proposal")
            self.assertEqual(common["accepted_closures"], [])
            self.assertIsNone(common["calibration_scope"])

    def test_branches_show_changed_closure_judgment_and_remaining_ambiguity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            _scenario, fold, _distinct, _plan = _eliminated_fold(platform)
            calibration = platform.register_calibration("similar", "adjacent-corridor revision")
            platform.reinterpret_observation(calibration.calibration_id, 46, 34)
            platform.reinterpret_observation(calibration.calibration_id, 47, 35)
            comparison = platform.compare_calibration_branches(calibration.calibration_id)
            branch = next(item for item in comparison["hypotheses"] if item["hypothesis_id"] == fold.hypothesis_id)
            self.assertTrue(branch["closure_judgment_changed"])
            self.assertEqual(branch["old_calibration"]["status"], "eliminated")
            self.assertEqual(branch["new_calibration"]["status"], "active")
            flipped = {item["raw_index"]: item["new_verdict"] for item in branch["changed_observations"]}
            self.assertEqual(flipped.get(46), "consistent")
            self.assertEqual(flipped.get(47), "consistent")
            # an ambiguous reinterpretation leaves both explanations unresolved
            ambiguous_calibration = platform.register_calibration("similar", "partial biased revision")
            platform.reinterpret_observation(ambiguous_calibration.calibration_id, 46, 34)
            ambiguous = platform.compare_calibration_branches(ambiguous_calibration.calibration_id)
            ambiguous_branch = next(item for item in ambiguous["hypotheses"] if item["hypothesis_id"] == fold.hypothesis_id)
            self.assertTrue(ambiguous_branch["ambiguity_remains"])
            self.assertEqual(ambiguous_branch["new_calibration"]["status"], "eliminated")
            preview = platform.build_map(
                "similar",
                "new_calibration_preview",
                hypothesis_id=fold.hypothesis_id,
                calibration_id=calibration.calibration_id,
            )
            self.assertEqual(preview["calibration_scope"]["calibration_id"], calibration.calibration_id)
            self.assertEqual(preview["calibration_scope"]["proposed_only_raw_indices"], [46, 47])
            self.assertEqual(preview["calibration_scope"]["confirmed_raw_indices"], [])
            # the old branch and the common map are not rewritten
            old_preview = platform.build_map("similar", "old_calibration_preview", hypothesis_id=fold.hypothesis_id)
            from pathlib import Path

            new_pgm = (platform.store.maps_dir / preview["map_version_id"] / "map.pgm").read_text()
            old_pgm = (platform.store.maps_dir / old_preview["map_version_id"] / "map.pgm").read_text()
            self.assertNotEqual(new_pgm, old_pgm)
            self.assertEqual(platform.build_map("similar", "common_still_common")["accepted_closures"], [])

    def test_partial_confirmation_scopes_map_constraints_without_double_counting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            _scenario, fold, _distinct, _plan = _eliminated_fold(platform)
            calibration = platform.register_calibration("similar", "adjacent-corridor revision")
            platform.reinterpret_observation(calibration.calibration_id, 46, 34)
            platform.reinterpret_observation(calibration.calibration_id, 47, 35)
            confirmation = platform.confirm_calibration_interpretation(
                calibration.calibration_id, [46], "only the first repositioning is checked"
            )
            self.assertEqual(confirmation["confirmed_raw_indices"], [46])
            updated_fold = platform.hypothesis(fold.hypothesis_id)
            # raw 46 has one confirmed new record and its old record is superseded, never two supports
            records_46 = [item for item in updated_fold.evidence if int(item["repeat_index"]) == 46]
            statuses = {item.get("status", "valid") for item in records_46}
            self.assertEqual(statuses, {"valid", "superseded_by_calibration"})
            self.assertEqual(len([item for item in records_46 if item.get("status", "valid") == "valid"]), 1)
            self.assertEqual(len([item for item in records_46 if item.get("calibration_id") == calibration.calibration_id]), 1)
            scoped_map = platform.build_map("similar", "partially_confirmed_scope", calibration_id=calibration.calibration_id)
            scope = scoped_map["calibration_scope"]
            self.assertEqual(scope["confirmed_raw_indices"], [46])
            self.assertEqual(scope["proposed_only_raw_indices"], [47])
            # reopening later must see the old branch and unaccepted map untouched
            reopened = MappingPlatform(directory)
            reloaded = reopened.calibration(calibration.calibration_id)
            self.assertEqual([item["status"] for item in reloaded.interpretations if item["raw_index"] == 46], ["confirmed"])
            self.assertEqual([item["status"] for item in reloaded.interpretations if item["raw_index"] == 47], ["proposed"])
            self.assertEqual(reopened.build_map("similar", "reloaded_common")["accepted_closures"], [])

    def test_withdrawing_calibration_restores_only_dependent_judgments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            _scenario, fold, _distinct, _plan = _eliminated_fold(platform)
            calibration = platform.register_calibration("similar", "adjacent-corridor revision")
            platform.reinterpret_observation(calibration.calibration_id, 46, 34)
            platform.reinterpret_observation(calibration.calibration_id, 47, 35)
            platform.confirm_calibration_interpretation(calibration.calibration_id, [46, 47])
            confirmed_fold = platform.hypothesis(fold.hypothesis_id)
            self.assertEqual(confirmed_fold.status, "active")
            withdrawal = platform.withdraw_calibration(calibration.calibration_id, "revision later recalled")
            restored_fold = platform.hypothesis(fold.hypothesis_id)
            # the judgment that depended on the revision is restored: old calibration eliminated the fold
            self.assertEqual(restored_fold.status, "eliminated")
            valid_records = [item for item in restored_fold.evidence if item.get("status", "valid") == "valid"]
            self.assertFalse(any(item.get("calibration_id") == calibration.calibration_id for item in valid_records))
            self.assertTrue(any(int(item["repeat_index"]) in (34, 35) for item in valid_records))
            recalibrated = platform.calibration(calibration.calibration_id)
            self.assertEqual(recalibrated.status, "withdrawn")

    def test_trace_links_plan_raw_observation_calibration_and_manual_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            _scenario, fold, _distinct, plan = _eliminated_fold(platform)
            calibration = platform.register_calibration("similar", "adjacent-corridor revision")
            platform.reinterpret_observation(
                calibration.calibration_id, 47, 35, plan_id=plan.plan_id, note="shared probe reinterpreted"
            )
            platform.confirm_calibration_interpretation(calibration.calibration_id, [47])
            trace = platform.decision_trace("similar")
            stored_calibration = trace["calibrations"][0]
            interpretation = next(item for item in stored_calibration["interpretations"] if item["raw_index"] == 47)
            self.assertEqual(interpretation["plan_id"], plan.plan_id)
            self.assertEqual(interpretation["interpreted_index"], 35)
            self.assertEqual(interpretation["status"], "confirmed")
            evaluation = next(
                item
                for item in stored_calibration["evaluations"]
                if item["raw_index"] == 47 and item["hypothesis_id"] == fold.hypothesis_id
            )
            self.assertEqual(evaluation["verdict"], "consistent")
            reloaded = MappingPlatform(directory).decision_trace("similar")
            self.assertEqual(len(reloaded["calibrations"]), 1)


if __name__ == "__main__":
    unittest.main()
