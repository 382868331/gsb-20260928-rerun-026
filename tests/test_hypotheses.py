from __future__ import annotations

import tempfile
import unittest

from robot_mapping.demo import load_scenario_data, run_competing_corridor_demo
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


if __name__ == "__main__":
    unittest.main()
