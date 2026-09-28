from __future__ import annotations

import math

Point = tuple[float, float]


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rotate_point(point: Point, angle: float) -> Point:
    x, y = point
    c = math.cos(angle)
    s = math.sin(angle)
    return c * x - s * y, s * x + c * y


def transform_point(point: Point, pose: "Pose") -> Point:
    x, y = rotate_point(point, pose.theta)
    return x + pose.x, y + pose.y


def invert_transform(pose: "Pose") -> "Pose":
    c = math.cos(pose.theta)
    s = math.sin(pose.theta)
    x = -c * pose.x - s * pose.y
    y = s * pose.x - c * pose.y
    return Pose(x, y, -pose.theta)


def compose_poses(a: "Pose", b: "Pose") -> "Pose":
    x, y = transform_point((b.x, b.y), a)
    return Pose(x, y, wrap_angle(a.theta + b.theta))


def interpolate_transform(transform: "Pose", alpha: float) -> "Pose":
    return Pose(transform.x * alpha, transform.y * alpha, wrap_angle(transform.theta * alpha))


def squared_distance(a: Point, b: Point) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.inf
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def fit_rigid(src: list[Point], dst: list[Point]) -> "Pose":
    if len(src) != len(dst) or not src:
        raise ValueError("corresponding point sets must be non-empty and equal in length")
    src_mean_x = sum(p[0] for p in src) / len(src)
    src_mean_y = sum(p[1] for p in src) / len(src)
    dst_mean_x = sum(p[0] for p in dst) / len(dst)
    dst_mean_y = sum(p[1] for p in dst) / len(dst)
    numerator = 0.0
    denominator = 0.0
    for (sx, sy), (dx, dy) in zip(src, dst):
        ax = sx - src_mean_x
        ay = sy - src_mean_y
        bx = dx - dst_mean_x
        by = dy - dst_mean_y
        numerator += ax * by - ay * bx
        denominator += ax * bx + ay * by
    theta = math.atan2(numerator, denominator)
    x, y = dst_mean_x, dst_mean_y
    rotated_src_mean = rotate_point((src_mean_x, src_mean_y), theta)
    x -= rotated_src_mean[0]
    y -= rotated_src_mean[1]
    return Pose(x, y, theta)


class Pose:
    __slots__ = ("x", "y", "theta")

    def __init__(self, x: float, y: float, theta: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.theta = wrap_angle(theta)

    def as_tuple(self) -> tuple[float, float, float]:
        return self.x, self.y, self.theta

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "theta": self.theta}

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "Pose":
        return cls(data["x"], data["y"], data.get("theta", 0.0))

    def transform(self, point: Point) -> Point:
        return transform_point(point, self)

    def inverse(self) -> "Pose":
        return invert_transform(self)

    def compose(self, other: "Pose") -> "Pose":
        return compose_poses(self, other)

    def __repr__(self) -> str:
        return f"Pose(x={self.x:.3f}, y={self.y:.3f}, theta={self.theta:.3f})"
