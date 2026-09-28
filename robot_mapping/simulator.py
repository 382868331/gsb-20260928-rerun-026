from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import Pose


@dataclass(frozen=True)
class Segment:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def dx(self) -> float:
        return self.x2 - self.x1

    @property
    def dy(self) -> float:
        return self.y2 - self.y1


@dataclass(frozen=True)
class Scenario:
    name: str
    walls: tuple[Segment, ...]
    truth: tuple[Pose, ...]
    odometry: tuple[Pose, ...]
    expected_anchor: int
    expected_repeat: int
    expected_loop: bool


def _add_rectangle(walls: list[Segment], x: float, y: float, width: float, height: float) -> None:
    corners = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
    for i, start in enumerate(corners):
        end = corners[(i + 1) % 4]
        walls.append(Segment(start[0], start[1], end[0], end[1]))


def _path_along(points: list[tuple[float, float]], per_edge: int) -> list[Pose]:
    result: list[Pose] = []
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        for step in range(per_edge):
            ratio = step / per_edge
            x = start[0] + (end[0] - start[0]) * ratio
            y = start[1] + (end[1] - start[1]) * ratio
            result.append(Pose(x, y, angle))
    return result


def _drift_pose(pose: Pose, index: int, scale: float) -> Pose:
    return Pose(
        pose.x + scale * (0.012 * index * index),
        pose.y - scale * (0.004 * index * index),
        pose.theta + scale * 0.0008 * index,
    )


def true_loop_scenario() -> Scenario:
    walls: list[Segment] = []
    _add_rectangle(walls, -1.25, -7.25, 10.5, 8.5)
    _add_rectangle(walls, 1.5, -2.5, 1.5, 1.5)
    _add_rectangle(walls, 5.5, -5.0, 1.3, 1.3)
    points = [(0, 0), (8, 0), (8, -6), (0, -6), (0, 0)]
    closed_path = tuple(_path_along(points[:-1], 8))
    truth = closed_path + (Pose(0.0, 0.0, 0.0), Pose(1.0, 0.0, 0.0), Pose(2.0, 0.0, 0.0))
    odometry = tuple(_drift_pose(pose, i, 0.35) for i, pose in enumerate(truth))
    return Scenario("true_loop", tuple(walls), truth, odometry, 1, 33, True)


def similar_corridor_scenario() -> Scenario:
    walls: list[Segment] = []
    _add_rectangle(walls, -2.0, -1.5, 18.0, 3.0)
    _add_rectangle(walls, 2.0, -5.5, 18.0, 3.0)
    walls.append(Segment(7.0, -1.5, 7.0, -0.55))
    walls.append(Segment(7.0, -0.55, 8.0, -0.55))
    walls.append(Segment(8.0, -0.55, 8.0, -1.5))
    walls.append(Segment(12.0, -5.5, 12.0, -4.45))
    walls.append(Segment(12.0, -4.45, 13.2, -4.45))
    walls.append(Segment(13.2, -4.45, 13.2, -5.5))
    truth = tuple(
        Pose(x, y, 0.0)
        for x, y in [(x * 0.5, 0.0) for x in range(25)] + [(12.0, y * 0.5 - 4.0) for y in range(5)] + [(12.0 - x * 0.5, -4.0) for x in range(25)]
    )
    odometry = tuple(_drift_pose(pose, i, 0.25) for i, pose in enumerate(truth))
    return Scenario("similar_corridors", tuple(walls), truth, odometry, 14, 43, False)


def ray_intersection(pose: Pose, angle: float, walls: tuple[Segment, ...], max_range: float = 8.0) -> float | None:
    ray_x = pose.x
    ray_y = pose.y
    ray_dx = math.cos(pose.theta + angle)
    ray_dy = math.sin(pose.theta + angle)
    best = math.inf
    for wall in walls:
        denominator = ray_dx * wall.dy - ray_dy * wall.dx
        if abs(denominator) < 1e-9:
            continue
        tx = wall.x1 - ray_x
        ty = wall.y1 - ray_y
        distance = (tx * wall.dy - ty * wall.dx) / denominator
        along = (tx * ray_dy - ty * ray_dx) / denominator
        if distance > 1e-4 and 0.0 <= along <= 1.0 and distance < best:
            best = distance
    return None if best > max_range else best


def simulate_scan(pose: Pose, walls: tuple[Segment, ...], beams: int = 72, max_range: float = 8.0) -> list[tuple[float, float]]:
    points = []
    for beam in range(beams):
        angle = -math.pi + beam * 2.0 * math.pi / beams
        distance = ray_intersection(pose, angle, walls, max_range)
        if distance is None:
            continue
        points.append((distance * math.cos(angle), distance * math.sin(angle)))
    return points
