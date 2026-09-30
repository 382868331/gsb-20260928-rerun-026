from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from robot_mapping.demo import load_scenario_data
from robot_mapping.mapper import MappingPlatform
from robot_mapping.simulator import competing_corridors_scenario


class CompetingHypothesisTest(unittest.TestCase):
    def _platform(self, directory: str) -> tuple[MappingPlatform, tuple[int, int], tuple[int, int], int]:
        platform = MappingPlatform(directory)
        scenario = competing_corridors_scenario()
        load_scenario_data(platform, "competing", scenario)
        (true_pair, false_pair) = scenario.competing_closures
        return platform, true_pair, false_pair, scenario.discriminating_index

    def test_conflicting_closures_live_as_independent_hypotheses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform, true_pair, false_pair, _ = self._platform(directory)
            true_closure = platform.propose_candidate("competing", *true_pair)
            false_closure = platform.propose_candidate("competing", *false_pair)
            true_hypothesis = platform.create_hypothesis("competing", [true_closure.closure_id], "same_corridor")
            false_hypothesis = platform.create_hypothesis("competing", [false_closure.closure_id], "twin_corridor")
            true_trial = platform.build_trial_map(true_hypothesis.hypothesis_id)
            false_trial = platform.build_trial_map(false_hypothesis.hypothesis_id)
            self.assertTrue(true_trial["trial"])
            self.assertTrue(false_trial["trial"])
            self.assertNotEqual(true_trial["map_version_id"], false_trial["map_version_id"])
            statuses = [closure.status for closure in platform.runs["competing"].closures]
            self.assertEqual(statuses, ["proposed", "proposed"])

    def test_followup_observations_exclude_only_the_wrong_corridor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform, true_pair, false_pair, _ = self._platform(directory)
            true_closure = platform.propose_candidate("competing", *true_pair)
            false_closure = platform.propose_candidate("competing", *false_pair)
            true_hypothesis = platform.create_hypothesis("competing", [true_closure.closure_id], "same_corridor")
            false_hypothesis = platform.create_hypothesis("competing", [false_closure.closure_id], "twin_corridor")
            platform.build_trial_map(true_hypothesis.hypothesis_id)
            platform.build_trial_map(false_hypothesis.hypothesis_id)
            platform.advance_observations("competing")
            self.assertEqual(true_hypothesis.status, "open")
            self.assertEqual(false_hypothesis.status, "excluded")
            basis = platform.exclusion_basis(false_hypothesis.hypothesis_id)
            self.assertIn("scan", basis["excluded_reason"])
            self.assertGreaterEqual(len(basis["evidence"]), 2)
            self.assertTrue(all(item["supports"] is False for item in basis["evidence"] if not item["supports"]))

    def test_excluded_hypothesis_can_be_reopened_compared_and_excluded_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform, true_pair, false_pair, _ = self._platform(directory)
            true_closure = platform.propose_candidate("competing", *true_pair)
            false_closure = platform.propose_candidate("competing", *false_pair)
            true_hypothesis = platform.create_hypothesis("competing", [true_closure.closure_id], "same_corridor")
            false_hypothesis = platform.create_hypothesis("competing", [false_closure.closure_id], "twin_corridor")
            platform.build_trial_map(true_hypothesis.hypothesis_id)
            platform.build_trial_map(false_hypothesis.hypothesis_id)
            platform.advance_observations("competing")
            previous_basis = platform.exclusion_basis(false_hypothesis.hypothesis_id)
            reopened = platform.reopen_hypothesis(false_hypothesis.hypothesis_id)
            self.assertEqual(reopened.status, "open")
            self.assertTrue(reopened.reopened)
            platform.advance_observations("competing")
            self.assertEqual(false_hypothesis.status, "excluded")
            events = Path(directory, "events.jsonl").read_text(encoding="utf-8")
            self.assertIn("hypothesis_excluded", events)
            self.assertIn("hypothesis_reopened", events)
            self.assertTrue(previous_basis["trial_map_id"])

    def test_selecting_a_hypothesis_only_corrects_the_local_region_and_withdraws_consistently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform, true_pair, false_pair, _ = self._platform(directory)
            true_closure = platform.propose_candidate("competing", *true_pair)
            false_closure = platform.propose_candidate("competing", *false_pair)
            true_hypothesis = platform.create_hypothesis("competing", [true_closure.closure_id], "same_corridor")
            false_hypothesis = platform.create_hypothesis("competing", [false_closure.closure_id], "twin_corridor")
            platform.advance_observations("competing")
            self.assertEqual(false_hypothesis.status, "excluded")
            raw_poses = [record.pose.as_tuple() for record in platform.runs["competing"].odometry]
            platform.select_hypothesis(true_hypothesis.hypothesis_id)
            selected_poses = [pose.as_tuple() for pose in platform.corrected_poses("competing")]
            changed = [index for index, (before, after) in enumerate(zip(raw_poses, selected_poses)) if before != after]
            self.assertLess(len(changed), len(raw_poses))
            self.assertTrue(all(not (30 <= index <= 150) for index in changed))
            platform.withdraw_hypothesis(true_hypothesis.hypothesis_id)
            withdrawn_poses = [pose.as_tuple() for pose in platform.corrected_poses("competing")]
            self.assertEqual(raw_poses, withdrawn_poses)

    def test_hypothesis_state_persists_between_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform, true_pair, false_pair, _ = self._platform(directory)
            true_closure = platform.propose_candidate("competing", *true_pair)
            false_closure = platform.propose_candidate("competing", *false_pair)
            true_hypothesis = platform.create_hypothesis("competing", [true_closure.closure_id], "same_corridor")
            false_hypothesis = platform.create_hypothesis("competing", [false_closure.closure_id], "twin_corridor")
            platform.advance_observations("competing")
            self.assertEqual(false_hypothesis.status, "excluded")
            reloaded = MappingPlatform(directory)
            run = reloaded._run("competing")
            restored = {hypothesis.label: hypothesis for hypothesis in run.hypotheses.values()}
            self.assertEqual(restored["same_corridor"].status, "open")
            self.assertEqual(restored["twin_corridor"].status, "excluded")
            self.assertIn("scan", restored["twin_corridor"].excluded_reason)


if __name__ == "__main__":
    unittest.main()
