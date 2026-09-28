from __future__ import annotations

import math
import unittest

from robot_mapping.mapper import MappingSession
from robot_mapping.simulation import build_drifted_odometry, similar_corridors_scenario, simulate_scan, square_loop_scenario


def pose_rmse(estimated, truth) -> float:
    values = []
    for estimate, actual in zip(estimated, truth):
        values.append((estimate.x - actual.x) ** 2 + (estimate.y - actual.y) ** 2)
    return math.sqrt(sum(values) / len(values))


class MappingPlatformTests(unittest.TestCase):
    def test_square_loop_accepts_and_withdraws_correct_closure(self):
        scenario = square_loop_scenario()
        truth = scenario.path()
        scans = [simulate_scan(pose, i, scenario) for i, pose in enumerate(truth)]
        session = MappingSession(scans, build_drifted_odometry(truth))
        candidates = session.find_candidates()

        accepted = [candidate for candidate in candidates if session.evaluate_candidate(candidate)]
        self.assertTrue(accepted, "synthetic revisit should pass multi-observation validation")
        accepted_error = pose_rmse(session.current_poses, truth)
        raw_error = pose_rmse(session.raw_poses, truth)
        self.assertLess(accepted_error, raw_error)

        accepted_poses = list(session.current_poses)
        accepted_version = session.versions[-1]
        session.withdraw_accepted_closure()
        withdrawn_version = session.versions[-1]
        self.assertNotEqual(accepted_version.version_id, withdrawn_version.version_id)
        self.assertEqual(withdrawn_version.parent_id, accepted_version.version_id)
        self.assertFalse(any(candidate.accepted is True for candidate in session.candidates))
        self.assertGreater(pose_rmse(session.current_poses, truth), pose_rmse(accepted_poses, truth))

    def test_similar_corridor_is_rejected(self):
        scenario = similar_corridors_scenario()
        truth = scenario.path()
        scans = [simulate_scan(pose, i, scenario) for i, pose in enumerate(truth)]
        session = MappingSession(scans, build_drifted_odometry(truth))
        candidate = session.add_forced_candidate(2, 26)

        self.assertFalse(session.evaluate_candidate(candidate))
        self.assertIsNone(session.accepted_closure_id)
        self.assertLess(candidate.observation_overlap, 0.48)
        self.assertEqual(len(session.versions), 1)


if __name__ == "__main__":
    unittest.main()
