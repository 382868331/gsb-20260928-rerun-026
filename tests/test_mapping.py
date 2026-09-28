from __future__ import annotations

import tempfile
import unittest

from robot_mapping.closure import check_loop_candidate
from robot_mapping.demo import load_scenario_data
from robot_mapping.mapper import MappingPlatform
from robot_mapping.simulator import similar_corridor_scenario, true_loop_scenario


class MappingPlatformTest(unittest.TestCase):
    def test_true_loop_is_supported_by_multiple_observations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            scenario = true_loop_scenario()
            load_scenario_data(platform, "loop", scenario)
            check = check_loop_candidate(platform.runs["loop"].scans, scenario.expected_anchor, scenario.expected_repeat)
            self.assertTrue(check.accepted)
            self.assertEqual(check.evidence["observations_checked"], 3)
            self.assertLess(check.score, 0.15)

    def test_similar_corridors_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            scenario = similar_corridor_scenario()
            load_scenario_data(platform, "similar", scenario)
            proposal = platform.propose_candidate("similar", scenario.expected_anchor, scenario.expected_repeat)
            self.assertEqual(proposal.status, "proposed")
            rejected = platform.reject_closure(proposal.closure_id)
            self.assertEqual(rejected.status, "rejected")
            before = platform.corrected_poses("similar")
            after_map = platform.build_map("similar", "still uncorrected")
            self.assertEqual(after_map["accepted_closures"], [])
            self.assertEqual([p.as_tuple() for p in before], [p.as_tuple() for p in platform.corrected_poses("similar")])

    def test_accepted_correction_can_be_withdrawn_and_restored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            scenario = true_loop_scenario()
            load_scenario_data(platform, "loop", scenario)
            proposal = platform.propose_candidate("loop", scenario.expected_anchor, scenario.expected_repeat)
            raw = platform.corrected_poses("loop")
            platform.accept_closure(proposal.closure_id)
            accepted = platform.corrected_poses("loop")
            self.assertNotEqual([p.as_tuple() for p in raw], [p.as_tuple() for p in accepted])
            platform.withdraw_closure(proposal.closure_id)
            withdrawn = platform.corrected_poses("loop")
            self.assertEqual([p.as_tuple() for p in raw], [p.as_tuple() for p in withdrawn])
            platform.accept_closure(proposal.closure_id)
            restored = platform.corrected_poses("loop")
            self.assertEqual([p.as_tuple() for p in accepted], [p.as_tuple() for p in restored])


if __name__ == "__main__":
    unittest.main()
