from __future__ import annotations

import math


Point = tuple[float, float]


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def transform_local(pose_x: float, pose_y: float, pose_theta: float, point: Point) -> Point:
    x, y = point
    c = math.cos(pose_theta)
    s = math.sin(pose_theta)
    return (pose_x + c * x - s * y, pose_y + s * x + c * y)


def invert_transform(x: float, y: float, theta: float) -> tuple[float, float, float]:
    c = math.cos(theta)
    s = math.sin(theta)
    return (-c * x - s * y, s * x - c * y, -theta)


def compose_transforms(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    ax, ay, at = a
    bx, by, bt = b
    c = math.cos(at)
    s = math.sin(at)
    return (ax + c * bx - s * by, ay + s * bx + c * by, wrap_angle(at + bt))


def pose_delta(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return compose_transforms(invert_transform(*a), b)


def squared_distance(a: Point, b: Point) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def rms(values: list[float]) -> float:
    if not values:
        return math.inf
    return math.sqrt(sum(value * value for value in values) / len(values))
