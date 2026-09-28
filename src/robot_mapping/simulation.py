from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import Point, wrap_angle
from .models import OdometryReading, Pose2D, Scan2D


Segment = tuple[Point, Point]


def rect_segments(x0: float, y0: float, x1: float, y1: float) -> list[Segment]:
    return [
        ((x0, y0), (x1, y0)),
        ((x1, y0), (x1, y1)),
        ((x1, y1), (x0, y1)),
        ((x0, y1), (x0, y0)),
    ]


def obstacle_segments(cx: float, cy: float, width: float, height: float) -> list[Segment]:
    return rect_segments(cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)


@dataclass(frozen=True)
class Scenario:
    name: str
    waypoints: tuple[Point, ...]
    walls: tuple[Segment, ...]
    spacing: float = 0.5
    range_max: float = 4.0
    beam_count: int = 121

    def path(self) -> list[Pose2D]:
        poses: list[Pose2D] = []
        for segment_index, (start, end) in enumerate(zip(self.waypoints, self.waypoints[1:])):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = math.hypot(dx, dy)
            theta = math.atan2(dy, dx)
            count = max(1, int(round(length / self.spacing)))
            if not poses or math.hypot(start[0] - poses[-1].x, start[1] - poses[-1].y) > 1e-6:
                poses.append(Pose2D(start[0], start[1], theta))
            for i in range(1, count + 1):
                ratio = i / count
                point = (start[0] + dx * ratio, start[1] + dy * ratio)
                poses.append(Pose2D(point[0], point[1], theta))
            if segment_index + 1 < len(self.waypoints) - 1:
                next_end = self.waypoints[segment_index + 2]
                next_theta = math.atan2(next_end[1] - end[1], next_end[0] - end[0])
                if abs(wrap_angle(next_theta - theta)) > 1e-9:
                    poses.append(Pose2D(end[0], end[1], next_theta))
        if len(self.waypoints) > 2 and math.hypot(self.waypoints[-1][0] - self.waypoints[0][0], self.waypoints[-1][1] - self.waypoints[0][1]) < 1e-9:
            first = self.waypoints[0]
            second = self.waypoints[1]
            final_theta = math.atan2(second[1] - first[1], second[0] - first[0])
            poses.append(Pose2D(first[0], first[1], final_theta))
        return poses


def ray_segment_distance(origin: Point, angle: float, max_range: float, segment: Segment) -> float | None:
    ox, oy = origin
    dx = math.cos(angle)
    dy = math.sin(angle)
    (ax, ay), (bx, by) = segment
    sx = bx - ax
    sy = by - ay
    denominator = dx * sy - dy * sx
    if abs(denominator) < 1e-12:
        return None
    t = ((ax - ox) * sy - (ay - oy) * sx) / denominator
    u = ((ax - ox) * dy - (ay - oy) * dx) / denominator
    if t >= 0.0 and t <= max_range and 0.0 <= u <= 1.0:
        return t
    return None


def simulate_scan(pose: Pose2D, index: int, scenario: Scenario) -> Scan2D:
    angle_min = -math.pi * 0.82
    angle_max = math.pi * 0.82
    beam_step = (angle_max - angle_min) / (scenario.beam_count - 1)
    ranges: list[float] = []
    for beam in range(scenario.beam_count):
        sensor_angle = angle_min + beam * beam_step
        world_angle = pose.theta + sensor_angle
        distance = scenario.range_max
        for segment in scenario.walls:
            hit = ray_segment_distance((pose.x, pose.y), world_angle, scenario.range_max, segment)
            if hit is not None:
                distance = min(distance, hit)
        ranges.append(distance)
    return Scan2D(f"scan-{index:03d}", float(index), angle_min, angle_max, tuple(ranges), scenario.range_max)


def build_drifted_odometry(
    truth: list[Pose2D], linear_drift: float = 0.006, angular_drift: float = 0.004
) -> list[OdometryReading]:
    readings = [OdometryReading(0.0, 0.0, 0.0, 0.0)]
    accumulated_x = 0.0
    accumulated_theta = 0.0
    for i in range(1, len(truth)):
        previous = truth[i - 1]
        current = truth[i]
        c = math.cos(-previous.theta)
        s = math.sin(-previous.theta)
        world_dx = current.x - previous.x
        world_dy = current.y - previous.y
        local_x = c * world_dx - s * world_dy
        local_y = s * world_dx + c * world_dy
        step = math.hypot(local_x, local_y)
        accumulated_x = 0.88 * accumulated_x + linear_drift * step * math.sin(i * 0.31)
        accumulated_theta = 0.88 * accumulated_theta + angular_drift * math.sin(i * 0.23 + 0.7)
        readings.append(
            OdometryReading(
                float(i),
                local_x + accumulated_x,
                local_y,
                wrap_angle(current.theta - previous.theta + accumulated_theta),
            )
        )
    return readings


def square_loop_scenario() -> Scenario:
    outer = rect_segments(-1.0, -1.0, 9.0, 9.0)
    inner = rect_segments(1.0, 1.0, 7.0, 7.0)
    return Scenario(
        "loop",
        ((0.0, 0.0), (8.0, 0.0), (8.0, 8.0), (0.0, 8.0), (0.0, 0.0)),
        tuple(outer + inner),
    )


def similar_corridors_scenario() -> Scenario:
    walls: list[Segment] = []
    walls.extend(rect_segments(-0.8, -1.0, 8.8, 1.0))
    walls.extend(rect_segments(-0.8, 3.0, 8.8, 5.0))
    walls.extend([((7.0, 1.0), (9.0, 1.0)), ((7.0, 3.0), (9.0, 3.0)), ((9.0, 1.0), (9.0, 3.0))])
    walls.extend(obstacle_segments(6.0, 0.72, 0.45, 0.56))
    walls.extend(obstacle_segments(2.0, 3.28, 0.45, 0.56))
    return Scenario(
        "similar_region",
        ((0.0, 0.0), (8.0, 0.0), (8.0, 4.0), (0.0, 4.0)),
        tuple(walls),
    )
