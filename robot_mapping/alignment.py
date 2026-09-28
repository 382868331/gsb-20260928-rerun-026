from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import Pose, fit_rigid, median, squared_distance, transform_point


@dataclass(frozen=True)
class AlignmentResult:
    transform: Pose
    rmse: float
    inlier_fraction: float
    median_error: float


def nearest_error(point: tuple[float, float], targets: list[tuple[float, float]]) -> float:
    return math.sqrt(min(squared_distance(point, target) for target in targets))


def align_icp(
    source: list[tuple[float, float]],
    target: list[tuple[float, float]],
    initial: Pose | None = None,
    iterations: int = 35,
    inlier_threshold: float = 0.22,
) -> AlignmentResult:
    transform = initial or Pose(0, 0, 0)
    result = transform
    inliers: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for _ in range(iterations):
        transformed = [transform_point(point, transform) for point in source]
        pairs: list[tuple[tuple[float, float], tuple[float, float], float]] = []
        for source_point, mapped in zip(source, transformed):
            errors = [(math.sqrt(squared_distance(mapped, target_point)), target_point) for target_point in target]
            error, target_point = min(errors, key=lambda item: item[0])
            if error <= inlier_threshold:
                pairs.append((source_point, target_point, error))
        if len(pairs) < max(6, len(source) // 3):
            return AlignmentResult(transform, math.inf, 0.0, math.inf)
        src = [pair[0] for pair in pairs]
        dst = [pair[1] for pair in pairs]
        step = fit_rigid(src, dst)
        delta = math.hypot(step.x - transform.x, step.y - transform.y)
        angle_delta = abs((step.theta - transform.theta + math.pi) % (2 * math.pi) - math.pi)
        transform = step
        inliers = [(pair[0], pair[1]) for pair in pairs]
        result = transform
        if delta < 1e-5 and angle_delta < 1e-6:
            break
    if not inliers:
        return AlignmentResult(result, math.inf, 0.0, math.inf)
    errors = []
    for source_point, target_point in inliers:
        mapped = transform_point(source_point, result)
        errors.append(math.sqrt(squared_distance(mapped, target_point)))
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    return AlignmentResult(result, rmse, len(inliers) / len(source), median(errors))


def range_signature(points: list[tuple[float, float]], bins: int = 24) -> tuple[float, ...]:
    values = [math.inf] * bins
    for x, y in points:
        angle = (math.atan2(y, x) + math.pi) % (2 * math.pi)
        index = min(bins - 1, int(angle / (2 * math.pi) * bins))
        distance = math.hypot(x, y)
        values[index] = min(values[index], distance)
    finite = [value for value in values if math.isfinite(value)]
    fill = max(finite) if finite else 8.0
    return tuple(value if math.isfinite(value) else fill for value in values)


def signature_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    shifts = []
    for shift in range(0, len(a), len(a) // 4):
        shifted = a[shift:] + a[:shift]
        shifts.append(math.sqrt(sum((x - y) ** 2 for x, y in zip(shifted, b)) / len(b)))
    return min(shifts)


def rotation_guesses() -> list[Pose]:
    return [Pose(0, 0, angle) for angle in (0.0, math.pi / 2, math.pi, -math.pi / 2)]


def best_icp(source: list[tuple[float, float]], target: list[tuple[float, float]]) -> AlignmentResult:
    candidates = [align_icp(source, target, guess) for guess in rotation_guesses()]
    return min(candidates, key=lambda result: (result.rmse, -result.inlier_fraction))
