from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .models import CalibrationVersion, Hypothesis, LoopClosure, PoseRecord, ProbePlan, ReviewPlan, Scan2D


class ProjectStore:
    """Append-only storage for immutable scans, odometry and review records."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.maps_dir = self.root / "map_versions"
        self.events_path = self.root / "events.jsonl"
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)
        self.maps_dir.mkdir(exist_ok=True)
        self.events_path.touch(exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        path = self.runs_dir / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_scan(self, scan: Scan2D) -> None:
        self._append_jsonl(self.run_dir(scan.run_id) / "scans.jsonl", scan.to_dict())

    def save_pose(self, pose: PoseRecord) -> None:
        self._append_jsonl(self.run_dir(pose.run_id) / "poses.jsonl", pose.to_dict())

    def load_scans(self, run_id: str) -> list[Scan2D]:
        return [Scan2D.from_dict(x) for x in self._read_jsonl(self.runs_dir / run_id / "scans.jsonl")]

    def load_poses(self, run_id: str) -> list[PoseRecord]:
        return [PoseRecord.from_dict(x) for x in self._read_jsonl(self.runs_dir / run_id / "poses.jsonl")]

    def load_closures(self) -> list[LoopClosure]:
        latest: dict[str, LoopClosure] = {}
        for item in self._read_jsonl(self.events_path):
            if "closure" in item:
                closure = LoopClosure.from_dict(item["closure"])
                latest[closure.closure_id] = closure
        return [latest[key] for key in sorted(latest)]

    def record_closure(self, closure: LoopClosure) -> None:
        self.append_event({"type": "loop_closure", "closure": closure.to_dict()})

    def load_hypotheses(self) -> list[Hypothesis]:
        latest: dict[str, Hypothesis] = {}
        for item in self._read_jsonl(self.events_path):
            if "hypothesis" in item:
                hypothesis = Hypothesis.from_dict(item["hypothesis"])
                latest[hypothesis.hypothesis_id] = hypothesis
        return [latest[key] for key in sorted(latest)]

    def record_hypothesis(self, hypothesis: Hypothesis) -> None:
        self.append_event({"type": "hypothesis", "hypothesis": hypothesis.to_dict()})

    def load_probe_plans(self) -> list[ProbePlan]:
        latest: dict[str, ProbePlan] = {}
        for item in self._read_jsonl(self.events_path):
            if "probe_plan" in item:
                plan = ProbePlan.from_dict(item["probe_plan"])
                latest[plan.plan_id] = plan
        return [latest[key] for key in sorted(latest)]

    def record_probe_plan(self, plan: ProbePlan) -> None:
        self.append_event({"type": "probe_plan", "probe_plan": plan.to_dict()})

    def load_review_plans(self) -> list[ReviewPlan]:
        latest: dict[str, ReviewPlan] = {}
        for item in self._read_jsonl(self.events_path):
            if "review_plan" in item:
                plan = ReviewPlan.from_dict(item["review_plan"])
                latest[plan.review_plan_id] = plan
        return [latest[key] for key in sorted(latest)]

    def record_review_plan(self, plan: ReviewPlan) -> None:
        self.append_event({"type": "review_plan", "review_plan": plan.to_dict()})

    def load_calibrations(self) -> list[CalibrationVersion]:
        latest: dict[str, CalibrationVersion] = {}
        for item in self._read_jsonl(self.events_path):
            if "calibration" in item:
                calibration = CalibrationVersion.from_dict(item["calibration"])
                latest[calibration.calibration_id] = calibration
        return [latest[key] for key in sorted(latest)]

    def record_calibration(self, calibration: CalibrationVersion) -> None:
        self.append_event({"type": "calibration", "calibration": calibration.to_dict()})

    def append_event(self, event: dict[str, Any]) -> None:
        self._append_jsonl(self.events_path, event)

    def save_map_version(self, version_id: str, payload: dict[str, Any], pgm: str | None = None) -> Path:
        target = self.maps_dir / version_id
        target.mkdir(parents=True, exist_ok=True)
        self._write_json(target / "map.json", payload)
        if pgm is not None:
            (target / "map.pgm").write_text(pgm, encoding="ascii")
        return target

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(temporary, path)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as stream:
            return [json.loads(line) for line in stream if line.strip()]
