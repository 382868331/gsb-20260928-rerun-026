from __future__ import annotations

import tempfile
import unittest

from robot_mapping.demo import _load_range
from robot_mapping.mapper import MappingPlatform
from robot_mapping.models import Hypothesis
from robot_mapping.reviews import compare_review_plans
from robot_mapping.simulator import similar_corridor_scenario


class ReviewCoveragePlanTest(unittest.TestCase):
    def test_shared_scan_is_counted_once_and_does_not_support_by_planning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            scenario = similar_corridor_scenario()
            arrival = scenario.expected_repeat - 10
            platform.create_run("similar")
            _load_range(platform, "similar", scenario, 0, arrival + 2)
            hypotheses = platform.propose_competing_closures(
                "similar",
                [
                    (0, arrival, "fold onto first corridor"),
                    (14, arrival, "fold onto notch corridor"),
                ],
            )
            first, second, distinct = hypotheses
            plan = platform.create_review_plan(
                "similar",
                "junction identity",
                review_budget=1,
                candidate_ids=[first.hypothesis_id, second.hypothesis_id, distinct.hypothesis_id],
            )
            observations = {item["scan_index"]: item for item in plan.observations}
            self.assertIn(arrival, observations)
            self.assertEqual(
                set(observations[arrival]["candidate_ids"]),
                {first.hypothesis_id, second.hypothesis_id},
            )
            selected = plan.comparison["selected_observation_ids"]
            self.assertEqual(len(selected), 1)
            selected_scan = int(str(selected[0]).replace("scan:", ""))
            selected_observation = observations[selected_scan]
            platform.record_review_feedback(
                plan.review_plan_id,
                selected_observation["observation_id"],
                "refute",
                selected_observation["candidate_ids"][0],
            )
            updated = platform.review_plan(plan.review_plan_id)
            self.assertEqual(updated.comparison["refuted_candidate_ids"], [selected_observation["candidate_ids"][0]])
            self.assertTrue(all(hypothesis.status == "active" for hypothesis in hypotheses))
            self.assertEqual(platform.build_map("similar", "common")["accepted_closures"], [])

    def test_compares_max_conflicts_with_balanced_sources_and_keeps_unknown_limits(self) -> None:
        a, b, c, d = "A", "B", "C", "D"
        observations = [
            {"observation_id": "same-a", "scan_index": 1, "source_id": "sensor-1", "candidate_ids": [a]},
            {"observation_id": "same-b", "scan_index": 2, "source_id": "sensor-1", "candidate_ids": [b]},
            {"observation_id": "same-d", "scan_index": 5, "source_id": "sensor-1", "candidate_ids": [d]},
            {"observation_id": "other-c", "scan_index": 3, "source_id": "sensor-2", "candidate_ids": [c, d]},
            {"observation_id": "unknown", "scan_index": 4, "source_id": "unknown", "relevance_known": False, "candidate_ids": [a]},
        ]
        comparison = compare_review_plans((a, b, c, d), observations, 2)
        by_strategy = {item["strategy"]: item for item in comparison["plans"]}
        self.assertEqual(
            by_strategy["maximum_conflict_coverage"]["observation_ids"], ["same-a", "same-b"]
        )
        self.assertEqual(len(by_strategy["maximum_conflict_coverage"]["covered_conflicts"]), 5)
        self.assertEqual(
            by_strategy["balanced_source_coverage"]["observation_ids"], ["same-a", "other-c"]
        )
        self.assertEqual(by_strategy["balanced_source_coverage"]["source_counts"], {"sensor-1": 1, "sensor-2": 1})
        unknown = compare_review_plans((a, b), [observations[-1]], 1)
        selected = next(item for item in unknown["plans"] if item["selected"])
        selected_id = selected["observation_ids"][0]
        self.assertEqual(selected_id, "unknown")
        self.assertEqual(selected["covered_conflicts"], [])
        self.assertTrue(selected["limitations"])

    def test_partial_feedback_is_scoped_does_not_auto_adopt_and_survives_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            platform.create_run("run")
            a, b, c = "A", "B", "C"
            run = platform.runs["run"]
            for candidate_id, name in ((a, "A"), (b, "B"), (c, "C")):
                hypothesis = Hypothesis(candidate_id, "run", name, (), "active")
                run.hypotheses.append(hypothesis)
                platform.store.record_hypothesis(hypothesis)
            observations = [
                {"observation_id": "scan-a", "scan_index": 1, "source_id": "sensor-1", "candidate_ids": [a]},
                {"observation_id": "scan-b", "scan_index": 2, "source_id": "sensor-2", "candidate_ids": [b]},
                {"observation_id": "shared", "scan_index": 3, "source_id": "sensor-3", "candidate_ids": [a, b]},
            ]
            plan = platform.create_review_plan(
                "run", "candidate identity", 2, [a, b, c], observations
            )
            observation_id = plan.comparison["selected_observation_ids"][0]
            target = next(item for item in observations if item["observation_id"] == observation_id)["candidate_ids"][0]
            platform.record_review_feedback(plan.review_plan_id, observation_id, "refute", target)
            with self.assertRaises(ValueError):
                platform.record_review_feedback(plan.review_plan_id, observation_id, "support", target)
            updated = platform.review_plan(plan.review_plan_id)
            self.assertEqual(updated.feedback[0]["candidate_id"], target)
            self.assertIn(target, updated.comparison["refuted_candidate_ids"])
            remaining_pairs = updated.comparison["plans"][0]["uncovered_conflicts"]
            self.assertTrue(all(target not in pair for pair in remaining_pairs))
            self.assertEqual(updated.comparison["adoption"], "requires explicit user decision")

            reloaded = MappingPlatform(directory)
            restored = reloaded.review_plan(plan.review_plan_id)
            self.assertEqual(restored.comparison["used_budget"], 1)
            self.assertEqual(restored.comparison["remaining_budget"], 1)
            self.assertEqual(restored.feedback[0]["verdict"], "refute")


if __name__ == "__main__":
    unittest.main()
