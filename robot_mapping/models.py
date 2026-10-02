from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from .geometry import Pose

ClosureStatus = Literal["proposed", "accepted", "rejected", "withdrawn"]
HypothesisStatus = Literal["active", "eliminated", "confirmed"]
CalibrationStatus = Literal["active", "withdrawn"]
CalibrationItemStatus = Literal["tentative", "confirmed"]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class Scan2D:
    run_id: str
    index: int
    timestamp: float
    sensor_id: str
    points: tuple[tuple[float, float], ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    scan_id: str = field(default_factory=lambda: new_id("scan"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "run_id": self.run_id,
            "index": self.index,
            "timestamp": self.timestamp,
            "sensor_id": self.sensor_id,
            "sensor_type": "scan2d",
            "points": [[x, y] for x, y in self.points],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scan2D":
        return cls(
            scan_id=data["scan_id"],
            run_id=data["run_id"],
            index=int(data["index"]),
            timestamp=float(data["timestamp"]),
            sensor_id=data["sensor_id"],
            points=tuple((float(x), float(y)) for x, y in data["points"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class PoseRecord:
    run_id: str
    index: int
    timestamp: float
    pose: Pose
    source: str = "odometry"
    pose_id: str = field(default_factory=lambda: new_id("pose"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pose_id": self.pose_id,
            "run_id": self.run_id,
            "index": self.index,
            "timestamp": self.timestamp,
            "source": self.source,
            **self.pose.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PoseRecord":
        return cls(
            pose_id=data["pose_id"],
            run_id=data["run_id"],
            index=int(data["index"]),
            timestamp=float(data["timestamp"]),
            source=data.get("source", "odometry"),
            pose=Pose.from_dict(data),
        )


@dataclass(frozen=True)
class LoopClosure:
    closure_id: str
    run_id: str
    anchor_index: int
    repeat_index: int
    status: ClosureStatus
    score: float
    evidence: dict[str, Any]
    proposed_at: float = field(default_factory=time.time)
    decided_at: float | None = None
    reason: str = ""

    def with_status(self, status: ClosureStatus, reason: str = "") -> "LoopClosure":
        return LoopClosure(
            self.closure_id,
            self.run_id,
            self.anchor_index,
            self.repeat_index,
            status,
            self.score,
            dict(self.evidence),
            self.proposed_at,
            time.time(),
            reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "closure_id": self.closure_id,
            "run_id": self.run_id,
            "anchor_index": self.anchor_index,
            "repeat_index": self.repeat_index,
            "status": self.status,
            "score": self.score,
            "evidence": self.evidence,
            "proposed_at": self.proposed_at,
            "decided_at": self.decided_at,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoopClosure":
        return cls(
            closure_id=data["closure_id"],
            run_id=data["run_id"],
            anchor_index=int(data["anchor_index"]),
            repeat_index=int(data["repeat_index"]),
            status=data["status"],
            score=float(data["score"]),
            evidence=dict(data.get("evidence", {})),
            proposed_at=float(data["proposed_at"]),
            decided_at=None if data.get("decided_at") is None else float(data["decided_at"]),
            reason=data.get("reason", ""),
        )


@dataclass(frozen=True)
class Hypothesis:
    """An independent map explanation built from a mutually compatible set of loop closures.

    The closure set is empty for the "distinct places" explanation that assumes none of
    the competing closures is real. A hypothesis keeps accumulating evidence without
    rewriting the common map until it is confirmed.
    """

    hypothesis_id: str
    run_id: str
    name: str
    closure_ids: tuple[str, ...]
    status: HypothesisStatus
    evidence: tuple[dict[str, Any], ...] = ()
    created_at: float = field(default_factory=time.time)
    decided_at: float | None = None
    reason: str = ""

    def with_status(self, status: HypothesisStatus, reason: str = "") -> "Hypothesis":
        return Hypothesis(
            self.hypothesis_id,
            self.run_id,
            self.name,
            self.closure_ids,
            status,
            self.evidence,
            self.created_at,
            time.time(),
            reason,
        )

    def with_evidence(self, evidence: tuple[dict[str, Any], ...], reason: str = "") -> "Hypothesis":
        return Hypothesis(
            self.hypothesis_id,
            self.run_id,
            self.name,
            self.closure_ids,
            self.status,
            evidence,
            self.created_at,
            self.decided_at,
            reason or self.reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "run_id": self.run_id,
            "name": self.name,
            "closure_ids": list(self.closure_ids),
            "status": self.status,
            "evidence": list(self.evidence),
            "created_at": self.created_at,
            "decided_at": self.decided_at,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Hypothesis":
        return cls(
            hypothesis_id=data["hypothesis_id"],
            run_id=data["run_id"],
            name=data["name"],
            closure_ids=tuple(data.get("closure_ids", [])),
            status=data["status"],
            evidence=tuple(data.get("evidence", [])),
            created_at=float(data["created_at"]),
            decided_at=None if data.get("decided_at") is None else float(data["decided_at"]),
            reason=data.get("reason", ""),
        )


@dataclass(frozen=True)
class ProbePlan:
    """A group of follow-up observations the user *plans* to collect.

    Plans are prospective: their predicted exclusion power is frozen when they
    are created and they never add evidence to a hypothesis until an actual
    observation arrives and is linked back to a plan item.
    """

    plan_id: str
    run_id: str
    ambiguous_region: str
    target_indices: tuple[int, ...]
    predictions: tuple[dict[str, Any], ...]
    executions: tuple[dict[str, Any], ...] = ()
    branch_hypothesis_id: str | None = None
    created_at: float = field(default_factory=time.time)
    reason: str = ""

    def with_executions(self, executions: tuple[dict[str, Any], ...]) -> "ProbePlan":
        return ProbePlan(
            self.plan_id,
            self.run_id,
            self.ambiguous_region,
            self.target_indices,
            self.predictions,
            executions,
            self.branch_hypothesis_id,
            self.created_at,
            self.reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "run_id": self.run_id,
            "ambiguous_region": self.ambiguous_region,
            "target_indices": list(self.target_indices),
            "predictions": list(self.predictions),
            "executions": list(self.executions),
            "branch_hypothesis_id": self.branch_hypothesis_id,
            "created_at": self.created_at,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProbePlan":
        return cls(
            plan_id=data["plan_id"],
            run_id=data["run_id"],
            ambiguous_region=data["ambiguous_region"],
            target_indices=tuple(data.get("target_indices", [])),
            predictions=tuple(data.get("predictions", [])),
            executions=tuple(data.get("executions", [])),
            branch_hypothesis_id=data.get("branch_hypothesis_id"),
            created_at=float(data["created_at"]),
            reason=data.get("reason", ""),
        )


@dataclass(frozen=True)
class CalibrationItem:
    """A reinterpretation of one raw observation under a later sensor calibration.

    ``raw_index`` is the index of the original, append-only scan; the raw
    observation, its collection time and its plan ownership never change.
    ``interpreted_index`` is where the same raw observation lands under the new
    calibration, which may be a neighbouring corridor or an index that has not
    been observed yet. The item is only a tentative explanation until a human
    confirms it; it is never a new observation collection.
    """

    raw_index: int
    interpreted_index: int
    range_scale: float = 1.0
    angle_offset: float = 0.0
    status: CalibrationItemStatus = "tentative"
    detail: dict[str, Any] = field(default_factory=dict)
    decided_at: float | None = None
    decision_reason: str = ""

    def confirmed(self, reason: str = "") -> "CalibrationItem":
        return CalibrationItem(
            self.raw_index,
            self.interpreted_index,
            self.range_scale,
            self.angle_offset,
            "confirmed",
            dict(self.detail),
            time.time(),
            reason or self.decision_reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_index": self.raw_index,
            "interpreted_index": self.interpreted_index,
            "range_scale": self.range_scale,
            "angle_offset": self.angle_offset,
            "status": self.status,
            "detail": self.detail,
            "decided_at": self.decided_at,
            "decision_reason": self.decision_reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationItem":
        return cls(
            raw_index=int(data["raw_index"]),
            interpreted_index=int(data["interpreted_index"]),
            range_scale=float(data.get("range_scale", 1.0)),
            angle_offset=float(data.get("angle_offset", 0.0)),
            status=data.get("status", "tentative"),
            detail=dict(data.get("detail", {})),
            decided_at=None if data.get("decided_at") is None else float(data["decided_at"]),
            decision_reason=data.get("decision_reason", ""),
        )


@dataclass(frozen=True)
class CalibrationRevision:
    """A sensor calibration version that reinterprets already collected raw scans.

    A revision never appends scans: every item keeps its ``raw_index`` plus the
    original collection time and plan ownership. Its items start tentative so the
    old- and new-calibration branches can be compared side by side. Withdrawing a
    revision restores judgments that depended solely on it while independent
    evidence remains in force.
    """

    calibration_id: str
    run_id: str
    sensor_id: str
    base_calibration_id: str | None
    status: CalibrationStatus
    items: tuple[CalibrationItem, ...]
    created_at: float = field(default_factory=time.time)
    withdrawn_at: float | None = None
    reason: str = ""

    def with_items(self, items: tuple[CalibrationItem, ...], reason: str | None = None) -> "CalibrationRevision":
        return CalibrationRevision(
            self.calibration_id,
            self.run_id,
            self.sensor_id,
            self.base_calibration_id,
            self.status,
            items,
            self.created_at,
            self.withdrawn_at,
            self.reason if reason is None else reason,
        )

    def withdrawn(self, reason: str) -> "CalibrationRevision":
        return CalibrationRevision(
            self.calibration_id,
            self.run_id,
            self.sensor_id,
            self.base_calibration_id,
            "withdrawn",
            tuple(
                CalibrationItem(
                    item.raw_index,
                    item.interpreted_index,
                    item.range_scale,
                    item.angle_offset,
                    item.status,
                    dict(item.detail),
                    item.decided_at,
                    item.decision_reason,
                )
                for item in self.items
            ),
            self.created_at,
            time.time(),
            reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "run_id": self.run_id,
            "sensor_id": self.sensor_id,
            "base_calibration_id": self.base_calibration_id,
            "status": self.status,
            "items": [item.to_dict() for item in self.items],
            "created_at": self.created_at,
            "withdrawn_at": self.withdrawn_at,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationRevision":
        return cls(
            calibration_id=data["calibration_id"],
            run_id=data["run_id"],
            sensor_id=data.get("sensor_id", "lidar_2d"),
            base_calibration_id=data.get("base_calibration_id"),
            status=data.get("status", "active"),
            items=tuple(CalibrationItem.from_dict(item) for item in data.get("items", [])),
            created_at=float(data["created_at"]),
            withdrawn_at=None if data.get("withdrawn_at") is None else float(data["withdrawn_at"]),
            reason=data.get("reason", ""),
        )
