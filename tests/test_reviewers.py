from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from robot_mapping.mapper import MappingPlatform

from test_calibration import _eliminated_fold


def _scan_bytes(platform: MappingPlatform, run_id: str = "similar") -> bytes:
    return (platform.store.runs_dir / run_id / "scans.jsonl").read_bytes()


class IndependentReviewerRevisionTest(unittest.TestCase):
    def test_conflicting_revisions_compared_side_by_side_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            _scenario, fold, _distinct, _plan = _eliminated_fold(platform)
            revision_a = platform.register_calibration(
                "similar", "reviewer A revision", reviewer="reviewer-a"
            )
            platform.reinterpret_observation(revision_a.calibration_id, 46, 34)
            platform.reinterpret_observation(revision_a.calibration_id, 47, 35)
            revision_b = platform.register_calibration(
                "similar", "reviewer B revision", reviewer="reviewer-b"
            )
            platform.reinterpret_observation(revision_b.calibration_id, 46, 40)
            platform.reinterpret_observation(revision_b.calibration_id, 47, 41)
            # neither revision overwrote the other: both keep their own readings
            stored_a = platform.calibration(revision_a.calibration_id)
            stored_b = platform.calibration(revision_b.calibration_id)
            self.assertEqual(
                {item["raw_index"]: item["interpreted_index"] for item in stored_a.interpretations},
                {46: 34, 47: 35},
            )
            self.assertEqual(
                {item["raw_index"]: item["interpreted_index"] for item in stored_b.interpretations},
                {46: 40, 47: 41},
            )
            # a conflicting rewrite inside one revision is rejected, not silently overwritten
            with self.assertRaises(ValueError):
                platform.reinterpret_observation(revision_a.calibration_id, 46, 40)
            comparison = platform.compare_reviewer_conflicts("similar")
            self.assertEqual([entry["raw_index"] for entry in comparison["conflicts"]], [46, 47])
            conflict_46 = comparison["conflicts"][0]
            self.assertEqual(conflict_46["resolution"], "pending")
            self.assertTrue(conflict_46["ambiguity_remains"])
            branches = {
                branch["reviewer"]: branch for branch in conflict_46["branches"]
            }
            self.assertEqual(set(branches), {"reviewer-a", "reviewer-b"})
            self.assertEqual(branches["reviewer-a"]["interpreted_index"], 34)
            self.assertEqual(branches["reviewer-b"]["interpreted_index"], 40)
            # each branch shows the per-hypothesis impact of its own position assumption
            for branch in branches.values():
                fold_branch = next(
                    item for item in branch["hypotheses"] if item["hypothesis_id"] == fold.hypothesis_id
                )
                self.assertIn("old_calibration", fold_branch)
                self.assertIn("new_calibration", fold_branch)
            # proposing alone never changed the common map
            self.assertEqual(platform.build_map("similar", "common_unchanged")["accepted_closures"], [])

    def test_partial_confirm_then_conflicting_revision_and_withdrawal_survive_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            _scenario, fold, _distinct, _plan = _eliminated_fold(platform)
            raw_before = _scan_bytes(platform)
            revision_a = platform.register_calibration(
                "similar", "reviewer A revision", reviewer="reviewer-a"
            )
            platform.reinterpret_observation(revision_a.calibration_id, 46, 34)
            platform.reinterpret_observation(revision_a.calibration_id, 47, 35)
            # confirm only half of the batch: exactly the user-chosen scope is adopted
            first = platform.confirm_calibration_interpretation(
                revision_a.calibration_id, [46], "reviewer A: only the first reading checked"
            )
            self.assertEqual(first["confirmed_raw_indices"], [46])
            # repeated confirmation of the same scope adds no extra support
            fold_after_first = platform.hypothesis(fold.hypothesis_id)
            valid_46 = [
                item
                for item in fold_after_first.evidence
                if int(item["repeat_index"]) == 46 and item.get("status", "valid") == "valid"
            ]
            self.assertEqual(len(valid_46), 1)
            again = platform.confirm_calibration_interpretation(
                revision_a.calibration_id, [46], "duplicate confirmation"
            )
            self.assertTrue(again["unchanged"])
            fold_after_repeat = platform.hypothesis(fold.hypothesis_id)
            self.assertEqual(len(fold_after_repeat.evidence), len(fold_after_first.evidence))
            # a second reviewer registers a conflicting revision of the same batch
            revision_b = platform.register_calibration(
                "similar", "reviewer B revision", reviewer="reviewer-b"
            )
            platform.reinterpret_observation(revision_b.calibration_id, 46, 40)
            platform.reinterpret_observation(revision_b.calibration_id, 47, 41)
            pending = platform.compare_reviewer_conflicts("similar")
            self.assertEqual(
                {entry["raw_index"] for entry in pending["conflicts"]}, {46, 47},
            )
            # the original revision is withdrawn; dependent judgments are restored
            platform.withdraw_calibration(revision_a.calibration_id, "reviewer A recalled the revision")
            restored_fold = platform.hypothesis(fold.hypothesis_id)
            self.assertEqual(restored_fold.status, "eliminated")
            withdrawn = platform.calibration(revision_a.calibration_id)
            self.assertEqual(withdrawn.status, "withdrawn")
            self.assertEqual(
                {item["status"] for item in withdrawn.interpretations}, {"withdrawn"},
            )
            # confirming a withdrawn revision is impossible
            with self.assertRaises(ValueError):
                platform.confirm_calibration_interpretation(revision_a.calibration_id, [47])
            # raw scans are byte-identical through the whole review workflow
            self.assertEqual(
                hashlib.sha256(raw_before).hexdigest(),
                hashlib.sha256(_scan_bytes(platform)).hexdigest(),
            )
            # save & reopen: pending observations and earlier decisions are intact
            reopened = MappingPlatform(directory)
            reloaded_b = reopened.calibration(revision_b.calibration_id)
            self.assertEqual(reloaded_b.reviewer, "reviewer-b")
            self.assertEqual(
                {item["raw_index"]: item["status"] for item in reloaded_b.interpretations},
                {46: "proposed", 47: "proposed"},
            )
            reloaded_a = reopened.calibration(revision_a.calibration_id)
            self.assertEqual(reloaded_a.status, "withdrawn")
            reloaded_fold = next(
                item for item in reopened.hypotheses_for("similar") if item.hypothesis_id == fold.hypothesis_id
            )
            self.assertEqual(reloaded_fold.status, "eliminated")
            self.assertEqual(reopened.build_map("similar", "reopened_common")["accepted_closures"], [])
            self.assertEqual(
                hashlib.sha256(raw_before).hexdigest(),
                hashlib.sha256(_scan_bytes(reopened)).hexdigest(),
            )

    def test_export_traces_adopted_withdrawn_and_remaining_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            _scenario, fold, _distinct, _plan = _eliminated_fold(platform)
            revision_a = platform.register_calibration(
                "similar", "reviewer A revision", reviewer="reviewer-a"
            )
            platform.reinterpret_observation(revision_a.calibration_id, 46, 34)
            revision_b = platform.register_calibration(
                "similar", "reviewer B revision", reviewer="reviewer-b"
            )
            platform.reinterpret_observation(revision_b.calibration_id, 46, 40)
            platform.reinterpret_observation(revision_b.calibration_id, 47, 41)
            platform.confirm_calibration_interpretation(
                revision_b.calibration_id, [46], "user explicitly adopts reviewer B for raw 46"
            )
            platform.withdraw_calibration(revision_a.calibration_id, "reviewer A recalled")
            trace = platform.decision_trace("similar")
            provenance = {entry["raw_index"]: entry for entry in trace["observation_provenance"]}
            self.assertEqual(provenance[46]["state"], "adopted")
            sources_46 = {
                (item["reviewer"], item["status"]) for item in provenance[46]["entries"]
            }
            self.assertEqual(sources_46, {("reviewer-a", "withdrawn"), ("reviewer-b", "confirmed")})
            self.assertEqual(provenance[47]["state"], "proposed")
            # the adopted scope counts raw 46 exactly once for the fold hypothesis
            fold_now = platform.hypothesis(fold.hypothesis_id)
            valid_46 = [
                item
                for item in fold_now.evidence
                if int(item["repeat_index"]) == 46 and item.get("status", "valid") == "valid"
            ]
            self.assertEqual(len(valid_46), 1)
            self.assertEqual(valid_46[0]["calibration_id"], revision_b.calibration_id)
            # export survives a save & reopen cycle
            reloaded_trace = MappingPlatform(directory).decision_trace("similar")
            reloaded_provenance = {
                entry["raw_index"]: entry for entry in reloaded_trace["observation_provenance"]
            }
            self.assertEqual(reloaded_provenance[46]["state"], "adopted")
            reviewers = {item["reviewer"] for item in reloaded_trace["calibrations"]}
            self.assertEqual(reviewers, {"reviewer-a", "reviewer-b"})

    def test_withdrawing_adopted_revision_reopens_conflict_instead_of_auto_adopting_rival(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            platform = MappingPlatform(directory)
            _scenario, fold, _distinct, _plan = _eliminated_fold(platform)
            raw_before = _scan_bytes(platform)
            revision_a = platform.register_calibration(
                "similar", "reviewer A revision", reviewer="reviewer-a"
            )
            platform.reinterpret_observation(revision_a.calibration_id, 46, 34)
            platform.reinterpret_observation(revision_a.calibration_id, 47, 35)
            # explicit confirmation covers only half of the batch
            platform.confirm_calibration_interpretation(
                revision_a.calibration_id, [46], "reviewer A: only the first reading checked"
            )
            # the other reviewer registers a conflicting revision of both observations
            revision_b = platform.register_calibration(
                "similar", "reviewer B revision", reviewer="reviewer-b"
            )
            platform.reinterpret_observation(revision_b.calibration_id, 46, 40)
            platform.reinterpret_observation(revision_b.calibration_id, 47, 41)
            # the original decision is withdrawn; reviewer B's readings were never adopted
            platform.withdraw_calibration(revision_a.calibration_id, "reviewer A recalled the revision")
            comparison = platform.compare_reviewer_conflicts("similar")
            resolutions = {entry["raw_index"]: entry["resolution"] for entry in comparison["conflicts"]}
            # a lone surviving revision must not be auto-adopted: both observations
            # return to pending because no explicit confirmation remains in force
            self.assertEqual(resolutions, {46: "pending", 47: "pending"})
            for entry in comparison["conflicts"]:
                self.assertTrue(entry["ambiguity_remains"])
                statuses = {branch["interpretation_status"] for branch in entry["branches"]}
                self.assertEqual(statuses, {"withdrawn", "proposed"})
            trace = platform.decision_trace("similar")
            provenance = {entry["raw_index"]: entry for entry in trace["observation_provenance"]}
            for raw_index in (46, 47):
                self.assertNotEqual(provenance[raw_index]["state"], "adopted")
            # confirming one observation alone leaves the other conflict pending
            platform.confirm_calibration_interpretation(
                revision_b.calibration_id, [46], "adopt reviewer B for raw 46 only"
            )
            partial = platform.compare_reviewer_conflicts("similar")
            partial_resolutions = {entry["raw_index"]: entry["resolution"] for entry in partial["conflicts"]}
            self.assertTrue(partial_resolutions[46].startswith("adopted:"))
            self.assertEqual(partial_resolutions[47], "pending")
            # repeated confirmation adds no support: raw 46 counts exactly once
            fold_now = platform.hypothesis(fold.hypothesis_id)
            valid_46 = [
                item
                for item in fold_now.evidence
                if int(item["repeat_index"]) == 46 and item.get("status", "valid") == "valid"
            ]
            self.assertEqual(len(valid_46), 1)
            repeat = platform.confirm_calibration_interpretation(
                revision_b.calibration_id, [46], "duplicate confirmation"
            )
            self.assertTrue(repeat["unchanged"])
            fold_after_repeat = platform.hypothesis(fold.hypothesis_id)
            valid_46_after = [
                item
                for item in fold_after_repeat.evidence
                if int(item["repeat_index"]) == 46 and item.get("status", "valid") == "valid"
            ]
            self.assertEqual(len(valid_46_after), 1)
            # save & reopen keeps adoption, withdrawal and the pending conflict consistent
            reopened = MappingPlatform(directory)
            reloaded = reopened.compare_reviewer_conflicts("similar")
            reloaded_resolutions = {entry["raw_index"]: entry["resolution"] for entry in reloaded["conflicts"]}
            self.assertTrue(reloaded_resolutions[46].startswith("adopted:"))
            self.assertEqual(reloaded_resolutions[47], "pending")
            reloaded_provenance = {
                entry["raw_index"]: entry
                for entry in reopened.decision_trace("similar")["observation_provenance"]
            }
            self.assertEqual(reloaded_provenance[46]["state"], "adopted")
            self.assertNotEqual(reloaded_provenance[47]["state"], "adopted")
            # the raw scan log is byte-identical through the entire workflow
            self.assertEqual(
                hashlib.sha256(raw_before).hexdigest(),
                hashlib.sha256(_scan_bytes(reopened)).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
