from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    theta: float

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "theta": self.theta}


@dataclass(frozen=True)
class Transform2D:
    x: float
    y: float
    theta: float

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "theta": self.theta}


@dataclass(frozen=True)
class Scan2D:
    """One raw sensor-frame lidar sweep.

    Rays use the robot frame: x forward, y left. Zero/negative ranges are
    treated as no return and are preserved rather than filled in.
    """

    scan_id: str
    timestamp: float
    angle_min: float
    angle_max: float
    ranges: tuple[float, ...]
    max_range: float

    @property
    def angles(self) -> list[float]:
        if len(self.ranges) == 1:
            return [self.angle_min]
        step = (self.angle_max - self.angle_min) / (len(self.ranges) - 1)
        return [self.angle_min + step * i for i in range(len(self.ranges))]

    def points(self) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for angle, distance in zip(self.angles, self.ranges):
            if 0.0 < distance < self.max_range:
                points.append((distance * math.cos(angle), distance * math.sin(angle)))
        return points

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "timestamp": self.timestamp,
            "angle_min": self.angle_min,
            "angle_max": self.angle_max,
            "ranges": list(self.ranges),
            "max_range": self.max_range,
        }


@dataclass(frozen=True)
class OdometryReading:
    timestamp: float
    dx: float
    dy: float
    dtheta: float

    def to_dict(self) -> dict[str, float]:
        return {"timestamp": self.timestamp, "dx": self.dx, "dy": self.dy, "dtheta": self.dtheta}


@dataclass(frozen=True)
class SensorRecord:
    record_id: str
    sensor_id: str
    sensor_type: str
    timestamp: float
    payload: Any

    def to_dict(self) -> dict[str, Any]:
        payload = self.payload.to_dict() if hasattr(self.payload, "to_dict") else self.payload
        return {
            "record_id": self.record_id,
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type,
            "timestamp": self.timestamp,
            "payload": payload,
        }


@dataclass
class ClosureCandidate:
    candidate_id: str
    from_node: int
    to_node: int
    relative_transform: Transform2D
    observation_overlap: float
    transform_spread: float
    motion_residual: float
    mean_rmse: float
    accepted: bool | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "relative_transform": self.relative_transform.to_dict(),
            "observation_overlap": self.observation_overlap,
            "transform_spread": self.transform_spread,
            "motion_residual": self.motion_residual,
            "mean_rmse": None if math.isinf(self.mean_rmse) else self.mean_rmse,
            "accepted": self.accepted,
            "reason": self.reason,
        }


@dataclass
class MapVersion:
    version_id: str
    parent_id: str | None
    event: str
    closure_id: str | None
    timestamp: float
    poses: tuple[Pose2D, ...]
    resolution: float
    cells: dict[tuple[int, int], int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "parent_id": self.parent_id,
            "event": self.event,
            "closure_id": self.closure_id,
            "timestamp": self.timestamp,
            "poses": [pose.to_dict() for pose in self.poses],
            "resolution": self.resolution,
            "cells": [[x, y, value] for (x, y), value in sorted(self.cells.items())],
            "metadata": self.metadata,
        }
