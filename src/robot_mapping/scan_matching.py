from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import Point, compose_transforms, invert_transform, pose_delta, rms, transform_local, wrap_angle
from .models import Pose2D, Scan2D, Transform2D


@dataclass(frozen=True)
class AlignmentResult:
    transform: Transform2D
    rmse: float
    overlap: float


def transformed_points(points: list[Point], transform: tuple[float, float, float]) -> list[Point]:
    x, y, theta = transform
    return [transform_local(x, y, theta, point) for point in points]


def nearest_distance(point: Point, targets: list[Point], max_correspondence: float) -> float | None:
    best: float | None = None
    px, py = point
    limit_sq = max_correspondence * max_correspondence
    for tx, ty in targets:
        dx = px - tx
        dy = py - ty
        distance_sq = dx * dx + dy * dy
        if distance_sq <= limit_sq and (best is None or distance_sq < best):
            best = distance_sq
    return None if best is None else math.sqrt(best)


def range_signature(scan: Scan2D) -> list[float]:
    step = max(1, len(scan.ranges) // 33)
    return [distance for distance in scan.ranges[::step] if distance < scan.max_range]


def signature_distance(a: Scan2D, b: Scan2D) -> float:
    av = range_signature(a)
    bv = range_signature(b)
    count = min(len(av), len(bv))
    if count < 12:
        return math.inf
    return sum(abs(av[i] - bv[i]) for i in range(count)) / count


def align_scan(
    source: Scan2D,
    target: Scan2D,
    initial: tuple[float, float, float] = (0.0, 0.0, 0.0),
    iterations: int = 14,
    max_correspondence: float = 0.45,
) -> AlignmentResult:
    source_points = source.points()
    target_points = target.points()
    transform = initial
    last_rmse = math.inf
    last_overlap = 0.0

    for _ in range(iterations):
        moved = transformed_points(source_points, transform)
        pairs: list[tuple[Point, Point]] = []
        residuals: list[float] = []
        for source_point, moved_point in zip(source_points, moved):
            distance = nearest_distance(moved_point, target_points, max_correspondence)
            if distance is None:
                continue
            best_target = min(target_points, key=lambda target_point: (target_point[0] - moved_point[0]) ** 2 + (target_point[1] - moved_point[1]) ** 2)
            pairs.append((source_point, best_target))
            residuals.append(distance)
        if len(pairs) < 12:
            break
        source_mean_x = sum(p[0][0] for p in pairs) / len(pairs)
        source_mean_y = sum(p[0][1] for p in pairs) / len(pairs)
        target_mean_x = sum(p[1][0] for p in pairs) / len(pairs)
        target_mean_y = sum(p[1][1] for p in pairs) / len(pairs)
        hxx = hxy = hyy = bxt = byt = 0.0
        for (sx, sy), (tx, ty) in pairs:
            sx -= source_mean_x
            sy -= source_mean_y
            tx -= target_mean_x
            ty -= target_mean_y
            hxx += sx * sx
            hyy += sy * sy
            hxy += sx * sy
            bxt += sx * tx
            byt += sy * ty
        determinant = hxx * hyy - hxy * hxy
        c = 1.0
        s = 0.0
        if abs(determinant) > 1e-12:
            c = (bxt * hyy - byt * hxy) / determinant
            s = (hxx * byt - hxy * bxt) / determinant
            norm = math.hypot(c, s)
            c /= norm
            s /= norm
        delta_theta = math.atan2(s, c)
        old_x, old_y, old_theta = transform
        global_source_mean = transform_local(old_x, old_y, old_theta, (source_mean_x, source_mean_y))
        delta_x = target_mean_x - global_source_mean[0]
        delta_y = target_mean_y - global_source_mean[1]
        transform = (old_x + delta_x * 0.7, old_y + delta_y * 0.7, wrap_angle(old_theta + delta_theta * 0.7))
        last_rmse = rms(residuals)
        last_overlap = len(pairs) / max(1, len(source_points))

    return AlignmentResult(Transform2D(*transform), last_rmse, last_overlap)


def grid_align_scan(
    source: Scan2D,
    target: Scan2D,
    initial: tuple[float, float, float],
    max_correspondence: float = 0.45,
) -> AlignmentResult:
    source_points = source.points()
    target_points = target.points()
    best: AlignmentResult | None = None
    search_levels = [(0.35, 5, 0.14), (0.10, 4, 0.05)]
    center = initial
    for xy_span, xy_steps, theta_span in search_levels:
        candidates = [center]
        for xi in range(-xy_steps, xy_steps + 1):
            for yi in range(-xy_steps, xy_steps + 1):
                for theta_index in (-1, 0, 1):
                    candidates.append(
                        (
                            center[0] + xi * xy_span / xy_steps,
                            center[1] + yi * xy_span / xy_steps,
                            wrap_angle(center[2] + theta_index * theta_span / 2),
                        )
                    )
        for candidate in candidates:
            moved = transformed_points(source_points, candidate)
            errors: list[float] = []
            for point in moved:
                distance = nearest_distance(point, target_points, max_correspondence)
                if distance is not None:
                    errors.append(distance)
            overlap = len(errors) / max(1, len(source_points))
            if len(errors) < 12:
                continue
            rmse = rms(errors)
            score = rmse - 0.12 * overlap
            if best is None or score < best.rmse - 0.12 * best.overlap:
                best = AlignmentResult(Transform2D(*candidate), rmse, overlap)
        if best is None:
            return AlignmentResult(Transform2D(*initial), math.inf, 0.0)
        center = (best.transform.x, best.transform.y, best.transform.theta)
    refined = align_scan(source, target, center, iterations=5, max_correspondence=max_correspondence)
    if math.isfinite(refined.rmse) and refined.overlap >= best.overlap - 0.05 and refined.rmse <= best.rmse + 0.02:
        return refined
    return best


def multi_observation_check(
    scans: list[Scan2D],
    poses: list[Pose2D],
    old_index: int,
    new_index: int,
    initial: tuple[float, float, float],
    window: int = 2,
) -> tuple[float, float, list[AlignmentResult]]:
    results: list[AlignmentResult] = []
    for offset in range(-window, window + 1):
        a = old_index + offset
        b = new_index - offset
        if 0 <= a < len(scans) and 0 <= b < len(scans) and a != b:
            old_motion = pose_delta(
                (poses[old_index].x, poses[old_index].y, poses[old_index].theta),
                (poses[a].x, poses[a].y, poses[a].theta),
            )
            new_motion = pose_delta(
                (poses[new_index].x, poses[new_index].y, poses[new_index].theta),
                (poses[b].x, poses[b].y, poses[b].theta),
            )
            adjusted_initial = compose_transforms(compose_transforms(invert_transform(*old_motion), initial), new_motion)
            aligned = range_match(scans[b], scans[a], adjusted_initial)
            normalized_transform = compose_transforms(
                compose_transforms(old_motion, (aligned.transform.x, aligned.transform.y, aligned.transform.theta)),
                invert_transform(*new_motion),
            )
            results.append(
                type(aligned)(Transform2D(*normalized_transform), aligned.rmse, aligned.overlap)
            )
    if len(results) < 3:
        return 0.0, math.inf, results
    overlaps = [result.overlap for result in results]
    mean_x = sum(result.transform.x for result in results) / len(results)
    mean_y = sum(result.transform.y for result in results) / len(results)
    mean_theta = math.atan2(
        sum(math.sin(result.transform.theta) for result in results),
        sum(math.cos(result.transform.theta) for result in results),
    )
    spread_values = []
    for result in results:
        spread_values.append((result.transform.x - mean_x) ** 2 + (result.transform.y - mean_y) ** 2)
        spread_values.append(wrap_angle(result.transform.theta - mean_theta) ** 2)
    return min(overlaps), math.sqrt(sum(spread_values) / len(spread_values)), results


def trajectory_motion_residual(
    poses: list[Pose2D], old_index: int, new_index: int, transform: Transform2D
) -> float:
    old_pose = (poses[old_index].x, poses[old_index].y, poses[old_index].theta)
    new_pose = (poses[new_index].x, poses[new_index].y, poses[new_index].theta)
    predicted = pose_delta(old_pose, new_pose)
    candidate = (transform.x, transform.y, transform.theta)
    delta = pose_delta(predicted, candidate)
    return math.sqrt(delta[0] * delta[0] + delta[1] * delta[1])


def candidate_initial_from_poses(poses: list[Pose2D], old_index: int, new_index: int) -> tuple[float, float, float]:
    new_pose = (poses[new_index].x, poses[new_index].y, poses[new_index].theta)
    old_pose = (poses[old_index].x, poses[old_index].y, poses[old_index].theta)
    return pose_delta(new_pose, old_pose)


from .ray_matching import range_match
