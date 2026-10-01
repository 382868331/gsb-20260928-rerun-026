from __future__ import annotations

import tempfile
import unittest

from robot_mapping.demo import _load_range, load_scenario_data
from robot_mapping.mapper import MappingPlatform
from robot_mapping.simulator import similar_corridor_scenario


class ProbePlanTest(unittest.TestCase):
    def _arrival_platform(self, directory: str) -> tuple[MappingPlatform, int]:
        platform = MappingPlatform(directory)
        scenario = similar_corridor_scenario()
        platform.create_run("s")
        arrival = scenario.expected_repeat - 10
        _load_range(platform, "s", scenario, 0, arrival + 1)
        platform.propose_competing_closures(
            "s",
            [(0, arrival, "fold onto first corridor")],
        )
        return platform, arrival

    def test_planned_probes_add_no_evidence_and_keep_insufficient_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform, arrival = self._arrival_platform(directory)
            before = [h.evidence for h in platform.hypotheses_for("s")]
            plan = platform.create_probe_plan(
                "s",
                "follow-up",
                [
                    (arrival + 1, "near", 0.0, -4.0),
                    (arrival + 40, "beyond mapped data", 0.0, -4.0),
                ],
            )
            verdicts = {
                target.target_index: {entry["hypothesis_name"]: entry["verdict"] for entry in target.analysis}
                for target in plan.targets
            }
            fold_verdict = verdicts[arrival + 1]
            self.assertIn("eliminated", fold_verdict.values())
            self.assertIn("unaffected", verdicts[arrival + 1].values())
            self.assertIn("insufficient", verdicts[arrival + 40].values())
            self.assertTrue(all(target.status == "planned" for target in plan.targets))
            # a plan is not an observation: no hypothesis evidence and no common-map change
            self.assertEqual([h.evidence for h in platform.hypotheses_for("s")], before)
            common = platform.build_map("s", "common")
            self.assertEqual(common["accepted_closures"], [])
            with self.assertRaises(ValueError):
                platform.create_probe_plan("s", "past", [(0, "already seen", 0.0, 0.0)])

    def test_actual_observation_reports_expected_vs_actual_and_records_source_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform, arrival = self._arrival_platform(directory)
            plan = platform.create_probe_plan("s", "follow-up", [(arrival + 1, "near", 0.0, -4.0)])
            scenario = similar_corridor_scenario()
            from robot_mapping.demo import _load_one

            _load_one(platform, "s", scenario, arrival + 1)
            result = platform.execute_probe(plan.plan_id, arrival + 1)
            self.assertEqual(result["target_index"], arrival + 1)
            self.assertEqual(result["actual_index"], arrival + 1)
            self.assertEqual(result["collected_at"], float(arrival + 1))
            self.assertIn("eliminated", result["expected_verdicts"].values())
            fold = next(h for h in platform.hypotheses_for("s") if h.closure_ids)
            latest = fold.evidence[-1]
            self.assertEqual(latest["plan_id"], plan.plan_id)
            self.assertEqual(latest["collected_at"], float(arrival + 1))
            self.assertFalse(latest["supports_fold"])
            self.assertTrue(result["differences"])

    def test_withdrawal_restores_branch_but_keeps_independent_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform, arrival = self._arrival_platform(directory)
            plan = platform.create_probe_plan(
                "s",
                "follow-up",
                [(arrival + 1, "a", 0.0, -4.0), (arrival + 2, "b", 0.0, -4.0)],
            )
            scenario = similar_corridor_scenario()
            from robot_mapping.demo import _load_one

            for index in (arrival + 1, arrival + 2):
                _load_one(platform, "s", scenario, index)
                platform.execute_probe(plan.plan_id, index)
            _load_one(platform, "s", scenario, arrival + 3)
            platform.feed_observation_to_hypotheses("s", arrival + 3)
            fold = next(h for h in platform.hypotheses_for("s") if h.closure_ids)
            self.assertEqual(fold.status, "eliminated")

            platform.withdraw_probe(plan.plan_id, arrival + 1, "wrong corridor")
            still_excluded = next(h for h in platform.hypotheses_for("s") if h.closure_ids)
            self.assertEqual(still_excluded.status, "eliminated")
            platform.withdraw_probe(plan.plan_id, arrival + 2, "wrong corridor")
            restored = next(h for h in platform.hypotheses_for("s") if h.closure_ids)
            self.assertEqual(restored.status, "active")
            live = [item for item in restored.evidence if not item.get("withdrawn")]
            withdrawn = [item for item in restored.evidence if item.get("withdrawn")]
            self.assertEqual(len(live), 1)
            self.assertEqual(len(withdrawn), 2)

    def test_reassign_creates_a_new_plan_and_keeps_other_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform, arrival = self._arrival_platform(directory)
            plan = platform.create_probe_plan("s", "follow-up", [(arrival + 1, "a", 0.0, -4.0)])
            scenario = similar_corridor_scenario()
            from robot_mapping.demo import _load_one

            _load_one(platform, "s", scenario, arrival + 1)
            platform.execute_probe(plan.plan_id, arrival + 1)
            _load_one(platform, "s", scenario, arrival + 2)
            platform.feed_observation_to_hypotheses("s", arrival + 2)
            outcome = platform.reassign_probe(plan.plan_id, arrival + 1, "third corridor", "scan came from a side branch")
            target = next(t for t in platform.plan(plan.plan_id).targets if t.target_index == arrival + 1)
            self.assertEqual(target.status, "reassigned")
            replacement = platform.plan(outcome["replacement_plan_id"])
            self.assertEqual(len(replacement.targets), 1)
            fold = next(h for h in platform.hypotheses_for("s") if h.closure_ids)
            self.assertTrue(any(item.get("withdrawn") for item in fold.evidence))
            self.assertTrue(any(not item.get("withdrawn") for item in fold.evidence))

    def test_plans_and_timeline_survive_reopen_without_counting_unexecuted_probes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform, arrival = self._arrival_platform(directory)
            plan = platform.create_probe_plan("s", "follow-up", [(arrival + 1, "a", 0.0, -4.0)])
            reloaded = MappingPlatform(directory)
            restored_plan = reloaded.plans_for("s")[0]
            self.assertEqual(restored_plan.plan_id, plan.plan_id)
            self.assertEqual(restored_plan.targets[0].status, "planned")
            self.assertEqual(reloaded.hypotheses_for("s")[0].evidence, ())
            scenario = similar_corridor_scenario()
            from robot_mapping.demo import _load_one

            _load_one(reloaded, "s", scenario, arrival + 1)
            reloaded.execute_probe(restored_plan.plan_id, arrival + 1)
            distinct = next(h for h in reloaded.hypotheses_for("s") if not h.closure_ids)
            reloaded.confirm_hypothesis(distinct.hypothesis_id, "chosen from real observations only")
            timeline = reloaded.decision_timeline("s")
            kinds = [event["type"] for event in timeline]
            self.assertIn("probe_plan_created", kinds)
            self.assertIn("probe_executed", kinds)
            self.assertIn("hypothesis_confirmed", kinds)


if __name__ == "__main__":
    unittest.main()
