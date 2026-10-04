from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from robot_mapping.mapper import MappingPlatform

from tests.test_calibration import _eliminated_fold


class IndependentReviewerBatchTest(unittest.TestCase):
    def _two_reviewers(self, platform: MappingPlatform, batch_id: str):
        reviewer_a = platform.register_reviewer_revision(batch_id, "reviewer_a", "reviewer A revision")
        reviewer_b = platform.register_reviewer_revision(batch_id, "reviewer_b", "reviewer B revision")
        platform.reinterpret_observation(reviewer_a.calibration_id, 46, 34)
        platform.reinterpret_observation(reviewer_a.calibration_id, 47, 35)
        platform.reinterpret_observation(reviewer_b.calibration_id, 46, 32)
        platform.reinterpret_observation(reviewer_b.calibration_id, 47, 35)
        return reviewer_a, reviewer_b

    def test_conflicting_revisions_remain_side_by_side_without_double_support(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            _scenario, fold, _distinct, _plan = _eliminated_fold(platform)
            batch = platform.create_review_batch("similar", [46, 47], "two reviewer pass")
            reviewer_a, reviewer_b = self._two_reviewers(platform, batch.batch_id)
            comparison = platform.compare_review_batch(batch.batch_id)
            self.assertEqual(comparison["selected_raw_indices"], [46, 47])
            self.assertTrue(comparison["raw_scan_unchanged"])
            conflicts = {item["raw_index"]: item for item in comparison["conflicts"]}
            self.assertIn(46, conflicts)
            self.assertIn(47, conflicts)
            sources = {(item["reviewer_id"], item["interpreted_index"]) for item in conflicts[46]["sources"]}
            self.assertEqual(
                sources,
                {("original_survey", 46), ("reviewer_a", 34), ("reviewer_b", 32)},
            )
            impact = next(item for item in conflicts[46]["hypothesis_impacts"] if item["hypothesis_id"] == fold.hypothesis_id)
            self.assertEqual(impact["branches"]["original"]["verdict"], "contradictory")
            self.assertEqual(impact["branches"][reviewer_a.calibration_id]["verdict"], "consistent")
            self.assertIsNone(impact["branches"][reviewer_b.calibration_id]["verdict"])
            self.assertEqual(impact["branches"][reviewer_b.calibration_id]["branch_status"], "eliminated")
            fold_records = [item for item in platform.hypothesis(fold.hypothesis_id).evidence if int(item["repeat_index"]) == 46]
            self.assertEqual({item.get("status", "valid") for item in fold_records}, {"valid"})
            self.assertFalse(any(item.get("calibration_id") for item in fold_records))
            self.assertEqual(platform.hypothesis(fold.hypothesis_id).status, "eliminated")

    def test_explicit_half_scope_confirms_once_and_withdraws_only_original_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            _scenario, fold, _distinct, _plan = _eliminated_fold(platform)
            batch = platform.create_review_batch("similar", [46, 47], "half-first review")
            reviewer_a, reviewer_b = self._two_reviewers(platform, batch.batch_id)
            first = platform.confirm_review_scope(batch.batch_id, reviewer_a.calibration_id, [46])
            second = platform.confirm_review_scope(batch.batch_id, reviewer_a.calibration_id, [46])
            self.assertEqual(first["newly_confirmed"], [46])
            self.assertEqual(second["newly_confirmed"], [])
            self.assertEqual(second["already_confirmed"], [46])
            comparison = platform.compare_review_batch(batch.batch_id)
            self.assertEqual([item["raw_index"] for item in comparison["conflicts"]], [47])
            records_46 = [item for item in platform.hypothesis(fold.hypothesis_id).evidence if int(item["repeat_index"]) == 46]
            self.assertEqual(len([item for item in records_46 if item.get("status", "valid") == "valid"]), 1)
            self.assertEqual([item.get("calibration_id") for item in records_46 if item.get("status", "valid") == "valid"], [reviewer_a.calibration_id])
            with self.assertRaises(ValueError):
                platform.confirm_review_scope(batch.batch_id, reviewer_b.calibration_id, [46])
            platform.withdraw_reviewer_revision(batch.batch_id, reviewer_a.calibration_id, [46])
            withdrawn_comparison = platform.compare_review_batch(batch.batch_id)
            self.assertEqual({item["raw_index"] for item in withdrawn_comparison["conflicts"]}, {46, 47})
            self.assertEqual(platform.hypothesis(fold.hypothesis_id).status, "eliminated")
            self.assertEqual(platform.calibration(reviewer_a.calibration_id).status, "partially_withdrawn")

    def test_reopen_preserves_decisions_conflicts_and_byte_identical_scans(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            _scenario, _fold, _distinct, _plan = _eliminated_fold(platform)
            batch = platform.create_review_batch("similar", [46, 47], "persistable review")
            reviewer_a, reviewer_b = self._two_reviewers(platform, batch.batch_id)
            platform.confirm_review_scope(batch.batch_id, reviewer_a.calibration_id, [46])
            platform.withdraw_reviewer_revision(batch.batch_id, reviewer_b.calibration_id, [46])
            scan_path = Path(directory) / "runs" / "similar" / "scans.jsonl"
            before = scan_path.read_bytes()
            reopened = MappingPlatform(directory)
            after = scan_path.read_bytes()
            self.assertEqual(before, after)
            restored_batch = reopened.review_batch(batch.batch_id)
            self.assertEqual(restored_batch.raw_indices, (46, 47))
            self.assertEqual(set(restored_batch.revision_ids), {reviewer_a.calibration_id, reviewer_b.calibration_id})
            comparison = reopened.compare_review_batch(batch.batch_id)
            self.assertTrue(comparison["raw_scan_unchanged"])
            self.assertEqual([item["raw_index"] for item in comparison["conflicts"]], [47])
            export = reopened.export_review_batch(batch.batch_id)
            self.assertEqual([item["raw_index"] for item in export["adopted"]], [46])
            self.assertEqual([item["raw_index"] for item in export["withdrawn"]], [46])
            self.assertEqual([item["raw_index"] for item in export["remaining_conflicts"]], [47])
            source_revisions = {item["revision_id"] for item in export["revision_sources"]}
            self.assertEqual(source_revisions, {reviewer_a.calibration_id, reviewer_b.calibration_id})
            exported = json.loads(Path(export["export_path"]).read_text(encoding="utf-8"))
            self.assertEqual(exported["batch_id"], batch.batch_id)
            trace = reopened.decision_trace("similar")
            self.assertEqual(trace["review_batches"][0]["raw_scan_sha256"], batch.raw_scan_sha256)


if __name__ == "__main__":
    unittest.main()
