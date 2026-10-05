from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from .geometry import Pose

ClosureStatus = Literal["proposed", "accepted", "rejected", "withdrawn"]
HypothesisStatus = Literal["active", "eliminated", "confirmed"]
CalibrationStatus = Literal["active", "withdrawn"]
InterpretationStatus = Literal["proposed", "confirmed", "withdrawn"]
BASELINE_CALIBRATION_ID = "cal_baseline"


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
class ReviewPlan:
    review_plan_id: str
    run_id: str
    ambiguity_scope: str
    candidate_ids: tuple[str, ...]
    ambiguity_pairs: tuple[tuple[str, str], ...]
    observations: tuple[dict[str, Any], ...]
    review_budget: int
    comparison: dict[str, Any]
    feedback: tuple[dict[str, Any], ...] = ()
    created_at: float = field(default_factory=time.time)
    reason: str = ""

    def with_feedback(self, feedback: tuple[dict[str, Any], ...], comparison: dict[str, Any]) -> "ReviewPlan":
        return ReviewPlan(
            self.review_plan_id,
            self.run_id,
            self.ambiguity_scope,
            self.candidate_ids,
            self.ambiguity_pairs,
            self.observations,
            self.review_budget,
            comparison,
            feedback,
            self.created_at,
            self.reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_plan_id": self.review_plan_id,
            "run_id": self.run_id,
            "ambiguity_scope": self.ambiguity_scope,
            "candidate_ids": list(self.candidate_ids),
            "ambiguity_pairs": [list(pair) for pair in self.ambiguity_pairs],
            "observations": list(self.observations),
            "review_budget": self.review_budget,
            "comparison": self.comparison,
            "feedback": list(self.feedback),
            "created_at": self.created_at,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewPlan":
        return cls(
            review_plan_id=data["review_plan_id"],
            run_id=data["run_id"],
            ambiguity_scope=data.get("ambiguity_scope", ""),
            candidate_ids=tuple(data.get("candidate_ids", [])),
            ambiguity_pairs=tuple(tuple(pair) for pair in data.get("ambiguity_pairs", [])),
            observations=tuple(data.get("observations", [])),
            review_budget=int(data["review_budget"]),
            comparison=dict(data.get("comparison", {})),
            feedback=tuple(data.get("feedback", [])),
            created_at=float(data["created_at"]),
            reason=data.get("reason", ""),
        )


@dataclass(frozen=True)
class CalibrationVersion:
    """A sensor calibration revision.

    A revision never creates a new observation. It only offers a different
    interpretation of the same raw scan: the same ``raw_index`` keeps its
    collection timestamp and probe-plan ownership, but can be read at a
    different effective position (``interpreted_index``) when the calibration
    is applied. Interpretations are proposed, confirmed partially or withdrawn
    without rewriting the raw scan log or the public map.
    """

    calibration_id: str
    run_id: str
    name: str
    reviewer: str = ""
    interpretations: tuple[dict[str, Any], ...] = ()
    evaluations: tuple[dict[str, Any], ...] = ()
    status: CalibrationStatus = "active"
    created_at: float = field(default_factory=time.time)
    withdrawn_at: float | None = None
    reason: str = ""

    def with_update(
        self,
        interpretations: tuple[dict[str, Any], ...] | None = None,
        evaluations: tuple[dict[str, Any], ...] | None = None,
        status: CalibrationStatus | None = None,
        withdrawn_at: float | None | str = "unchanged",
        reason: str | None = None,
    ) -> "CalibrationVersion":
        return CalibrationVersion(
            self.calibration_id,
            self.run_id,
            self.name,
            self.reviewer,
            self.interpretations if interpretations is None else interpretations,
            self.evaluations if evaluations is None else evaluations,
            self.status if status is None else status,
            self.created_at,
            self.withdrawn_at if withdrawn_at == "unchanged" else withdrawn_at,
            self.reason if reason is None else reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "run_id": self.run_id,
            "name": self.name,
            "reviewer": self.reviewer,
            "interpretations": list(self.interpretations),
            "evaluations": list(self.evaluations),
            "status": self.status,
            "created_at": self.created_at,
            "withdrawn_at": self.withdrawn_at,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationVersion":
        return cls(
            calibration_id=data["calibration_id"],
            run_id=data["run_id"],
            name=data.get("name", data["calibration_id"]),
            reviewer=data.get("reviewer", ""),
            interpretations=tuple(data.get("interpretations", [])),
            evaluations=tuple(data.get("evaluations", [])),
            status=data.get("status", "active"),
            created_at=float(data["created_at"]) if "created_at" in data else time.time(),
            withdrawn_at=None if data.get("withdrawn_at") is None else float(data["withdrawn_at"]),
            reason=data.get("reason", ""),
        )
