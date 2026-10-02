from __future__ import annotations

import tempfile
import unittest

from robot_mapping.calibrations import reinterpret_points
from robot_mapping.demo import _load_one, _load_range, run_calibration_demo
from robot_mapping.mapper import MappingPlatform
from robot_mapping.models import CalibrationItem
from robot_mapping.simulator import similar_corridor_scenario


class CalibrationRevisionTest(unittest.TestCase):
    def _arrived(self, platform: MappingPlatform) -> tuple[object, int, object, object, object]:
        scenario = similar_corridor_scenario()
        arrival = scenario.expected_repeat - 10
        platform.create_run("calib")
        _load_range(platform, "calib", scenario, 0, arrival + 1)
        hypotheses = platform.propose_competing_closures(
            "calib",
            [
                (0, arrival, "fold a"),
                (2, arrival, "fold b"),
            ],
        )
        fold_a, fold_b, distinct = hypotheses
        plan = platform.create_probe_plan(
            "calib", "junction", [arrival + 1, arrival + 2, arrival + 3]
        )
        for index in range(arrival + 1, arrival + 4):
            _load_one(platform, "calib", scenario, index)
            platform.feed_observation_to_hypotheses(
                "calib", index, plan_id=plan.plan_id, collected_at=float(index)
            )
        _load_range(platform, "calib", scenario, arrival + 4, len(scenario.truth))
        return scenario, arrival, fold_a, fold_b, plan

    def test_tentative_calibration_compares_branches_without_rewriting_anything(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            scenario, arrival, fold_a, fold_b, _ = self._arrived(platform)
            raw_scan_count = len(platform.runs["calib"].scans)
            revision = platform.propose_calibration_revision(
                "calib",
                [
                    {"raw_index": arrival + 1, "interpreted_index": arrival + 8},
                    {"raw_index": arrival + 2, "interpreted_index": arrival + 7},
                    {"raw_index": arrival + 3, "interpreted_index": arrival + 27},
                ],
            )
            comparison = platform.compare_calibration_branches("calib", revision.calibration_id)
            self.assertEqual(comparison["confirmed_scope"], [])
            self.assertFalse(comparison["common_map_modified"])
            for item in comparison["hypotheses"]:
                self.assertTrue(item["old_calibration_eliminated"])
                self.assertFalse(item["new_calibration_eliminated"])
                self.assertTrue(item["loop_judgment_changed"])
                new_verdicts = {change["raw_index"]: change["new_verdict"] for change in item["changed_observations"]}
                self.assertEqual(new_verdicts[arrival + 1], "consistent")
                self.assertEqual(new_verdicts[arrival + 2], "consistent")
                self.assertEqual(new_verdicts[arrival + 3], "unfulfilled")
            # tentative proposal: same raw scans, hypotheses and common map untouched
            self.assertEqual(len(platform.runs["calib"].scans), raw_scan_count)
            self.assertEqual(platform.hypothesis(fold_a.hypothesis_id).status, "eliminated")
            common = platform.build_map("calib", "common_unchanged")
            self.assertEqual(common["accepted_closures"], [])
            preview = platform.build_map(
                "calib",
                "new_calibration_preview",
                hypothesis_id=fold_a.hypothesis_id,
                calibration_id=revision.calibration_id,
                include_tentative=True,
            )
            self.assertEqual(preview["calibration_scope"], "preview_tentative")
            self.assertEqual(preview["reinterpreted_raw_indices"], [arrival + 1, arrival + 2, arrival + 3])
            # raw files keep exactly the original observations
            with open(f"{directory}/runs/calib/scans.jsonl", encoding="utf-8") as stream:
                self.assertEqual(len(stream.readlines()), raw_scan_count)

    def test_partial_confirmation_scopes_map_and_never_double_counts_a_raw_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            _, arrival, fold_a, fold_b, plan = self._arrived(platform)
            revision = platform.propose_calibration_revision(
                "calib",
                [
                    {"raw_index": arrival + 1, "interpreted_index": arrival + 8},
                    {"raw_index": arrival + 2, "interpreted_index": arrival + 7},
                    {"raw_index": arrival + 3, "interpreted_index": arrival + 27},
                ],
            )
            partial = platform.confirm_calibration_items(revision.calibration_id, [arrival + 1])
            self.assertEqual(partial["restored_hypotheses"], [])
            fold = platform.hypothesis(fold_a.hypothesis_id)
            self.assertEqual(fold.status, "eliminated")
            occurrences = [item for item in fold.evidence if int(item["raw_index"]) == arrival + 1]
            self.assertEqual(len(occurrences), 1)
            adopted = occurrences[0]
            self.assertEqual(adopted["calibration_id"], revision.calibration_id)
            self.assertEqual(adopted["verdict"], "consistent")
            self.assertEqual(adopted["repeat_index"], arrival + 8)
            self.assertEqual(adopted["collected_at"], float(arrival + 1))
            self.assertEqual(adopted["plan_id"], plan.plan_id)
            # the still-tentative readings are not applied to the confirmed-scope map
            scoped_map = platform.build_map("calib", "partial_scope", calibration_id=revision.calibration_id)
            self.assertEqual(scoped_map["calibration_scope"], "confirmed")
            self.assertEqual(scoped_map["reinterpreted_raw_indices"], [arrival + 1])
            # confirming the second shared observation restores BOTH fold hypotheses
            full = platform.confirm_calibration_items(revision.calibration_id, [arrival + 2])
            self.assertEqual(set(full["restored_hypotheses"]), {fold_a.hypothesis_id, fold_b.hypothesis_id})
            for fold_id in (fold_a.hypothesis_id, fold_b.hypothesis_id):
                hypothesis = platform.hypothesis(fold_id)
                self.assertEqual(hypothesis.status, "active")
                raw_indices = [int(item["raw_index"]) for item in hypothesis.evidence]
                self.assertEqual(sorted(raw_indices), [arrival + 1, arrival + 2, arrival + 3])

    def test_withdrawing_calibration_restores_old_judgments_but_keeps_independent_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            _, arrival, fold_a, _, _ = self._arrived(platform)
            revision = platform.propose_calibration_revision(
                "calib",
                [
                    {"raw_index": arrival + 1, "interpreted_index": arrival + 8},
                    {"raw_index": arrival + 2, "interpreted_index": arrival + 7},
                ],
            )
            platform.confirm_calibration_items(revision.calibration_id, [arrival + 1, arrival + 2])
            self.assertEqual(platform.hypothesis(fold_a.hypothesis_id).status, "active")
            # an independent later raw observation keeps contradicting under every calibration
            result = platform.feed_observation_to_hypotheses("calib", arrival + 11)
            self.assertTrue(result["hypothesis_results"])
            withdrawal = platform.withdraw_calibration(revision.calibration_id)
            self.assertEqual(withdrawal["restored_hypotheses"], [])
            fold = platform.hypothesis(fold_a.hypothesis_id)
            self.assertEqual(fold.status, "eliminated")
            self.assertTrue(all(item.get("calibration_id") != revision.calibration_id for item in fold.evidence))
            verdicts = {
                int(item["raw_index"]): item["verdict"]
                for item in fold.evidence
            }
            self.assertEqual(verdicts[arrival + 1], "contradictory")
            self.assertEqual(verdicts[arrival + 2], "contradictory")

    def test_reinterpretation_moves_points_without_creating_an_observation(self) -> None:
        item = CalibrationItem(raw_index=1, interpreted_index=5, range_scale=2.0)
        points = [(1.0, 0.0), (0.0, 1.0)]
        adjusted = reinterpret_points(points, item)
        self.assertEqual(adjusted, [(2.0, 0.0), (0.0, 2.0)])
        self.assertEqual(points, [(1.0, 0.0), (0.0, 1.0)])

    def test_full_demo_reopens_from_partial_confirmation_and_traces_every_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            summary = run_calibration_demo(platform, "calibration_corridors", similar_corridor_scenario())
            reopened = MappingPlatform(directory)
            trace = reopened.decision_trace("calibration_corridors")
            calibration = trace["calibrations"][0]
            self.assertEqual(calibration["calibration_id"], summary["calibration_id"])
            self.assertEqual(calibration["status"], "withdrawn")
            scope = {item["raw_index"]: item["status"] for item in calibration["scope"]}
            self.assertEqual(scope, {34: "confirmed", 35: "confirmed", 36: "tentative"})
            plans = {plan["plan_id"] for plan in trace["plans"]}
            self.assertIn(summary["resumed_plan"], plans)
            folds = [h for h in trace["hypotheses"] if h["closure_ids"]]
            self.assertTrue(all(h["status"] == "eliminated" for h in folds))
            # every map decision names its plan lineage through raw indices, calibration and status
            for map_version in (summary["old_preview"], summary["new_preview"], summary["partial_map"]):
                self.assertTrue(map_version.startswith("map_"))


if __name__ == "__main__":
    unittest.main()
