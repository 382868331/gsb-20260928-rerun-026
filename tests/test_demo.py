from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from robot_mapping.demo import run_demo


class DemoTest(unittest.TestCase):
    def test_built_in_scenarios_and_map_versions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_demo(directory)
            root = Path(directory)
            self.assertTrue(report.exists())
            self.assertEqual(len(list((root / "runs" / "true_loop").glob("scans.jsonl"))), 1)
            self.assertEqual(len(list((root / "runs" / "similar_corridors").glob("scans.jsonl"))), 1)
            self.assertGreaterEqual(len(list((root / "map_versions").iterdir())), 5)
            raw_true = (root / "runs" / "true_loop" / "scans.jsonl").read_text(encoding="utf-8")
            events = (root / "events.jsonl").read_text(encoding="utf-8")
            self.assertIn("loop_closure", events)
            self.assertIn("withdrawn", events)
            self.assertIn("accepted", events)
            self.assertIn("rejected", events)
            self.assertIn("map_version", events)
            self.assertNotIn("corrected", raw_true)


if __name__ == "__main__":
    unittest.main()
