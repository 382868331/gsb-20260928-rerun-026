from __future__ import annotations

import math
from dataclasses import dataclass

from .models import Scan2D, Transform2D


@dataclass(frozen=True)
class AlignmentResult:
    transform: Transform2D
    rmse: float
    overlap: float


def range_match(
    source: Scan2D,
    target: Scan2D,
    initial: tuple[float, float, float],
    x_span: float = 0.50,
    y_span: float = 0.50,
    theta_span: float = 0.18,
    steps: int = 4,
) -> AlignmentResult:
    usable = [(angle, distance) for angle, distance in zip(source.angles, source.ranges) if 0 < distance < source.max_range]
    target_step = (target.angle_max - target.angle_min) / max(1, len(target.ranges) - 1)
    best: tuple[float, float, tuple[float, float, float]] | None = None

    for xi in range(-steps, steps + 1):
        for yi in range(-steps, steps + 1):
            for theta_index in (-1, 0, 1):
                candidate = (
                    initial[0] + xi * x_span / steps,
                    initial[1] + yi * y_span / steps,
                    initial[2] + theta_index * theta_span,
                )
                c = math.cos(candidate[2])
                s = math.sin(candidate[2])
                errors: list[float] = []
                for angle, source_range in usable:
                    local_x = source_range * math.cos(angle)
                    local_y = source_range * math.sin(angle)
                    observed_x = c * local_x - s * local_y + candidate[0]
                    observed_y = s * local_x + c * local_y + candidate[1]
                    observed_angle = math.atan2(observed_y, observed_x)
                    observed_range = math.hypot(observed_x, observed_y)
                    beam = int(round((observed_angle - target.angle_min) / target_step))
                    best_range: float | None = None
                    for nearby in range(beam - 1, beam + 2):
                        if 0 <= nearby < len(target.ranges) and 0 < target.ranges[nearby] < target.max_range:
                            error = abs(target.ranges[nearby] - observed_range)
                            if error <= 0.28 and (best_range is None or error < abs(best_range - observed_range)):
                                best_range = target.ranges[nearby]
                    if best_range is None:
                        continue
                    errors.append((best_range - observed_range) ** 2)
                overlap = len(errors) / max(1, len(usable))
                if len(errors) < 12:
                    continue
                rmse = math.sqrt(sum(errors) / len(errors))
                score = rmse - 0.30 * overlap
                if best is None or score < best[0] - 0.30 * best[1]:
                    best = (rmse, overlap, candidate)

    if best is None:
        return AlignmentResult(Transform2D(*initial), math.inf, 0.0)
    return AlignmentResult(Transform2D(*best[2]), best[0], best[1])
