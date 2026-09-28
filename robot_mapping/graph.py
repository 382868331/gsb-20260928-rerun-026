from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import Pose, compose_poses, invert_transform, wrap_angle


@dataclass(frozen=True)
class PoseEdge:
    i: int
    j: int
    measurement: Pose
    weight: float


def relative_pose(a: Pose, b: Pose) -> Pose:
    return compose_poses(invert_transform(a), b)


def optimize_poses(initial: list[Pose], edges: list[PoseEdge], iterations: int = 12) -> list[Pose]:
    values: list[float] = []
    for pose in initial:
        values.extend(pose.as_tuple())

    def pose_at(vector: list[float], index: int) -> Pose:
        base = index * 3
        return Pose(vector[base], vector[base + 1], vector[base + 2])

    def residual(vector: list[float]) -> list[float]:
        result = [vector[0] * 1000.0, vector[1] * 1000.0, vector[2] * 1000.0]
        for edge in edges:
            pi = pose_at(vector, edge.i)
            pj = pose_at(vector, edge.j)
            actual = relative_pose(pi, pj)
            predicted = compose_poses(invert_transform(edge.measurement), actual)
            scale = edge.weight
            result.extend([predicted.x * scale, predicted.y * scale, predicted.theta * scale])
        return result

    dimension = len(values)
    epsilon = 1e-5
    for _ in range(iterations):
        r0 = residual(values)
        jacobian = [[0.0] * dimension for _ in r0]
        for column in range(dimension):
            perturbed = values.copy()
            perturbed[column] += epsilon
            r1 = residual(perturbed)
            for row, (after, before) in enumerate(zip(r1, r0)):
                jacobian[row][column] = (after - before) / epsilon
        normal = [[0.0] * dimension for _ in range(dimension)]
        gradient = [0.0] * dimension
        for row, value in enumerate(r0):
            jrow = jacobian[row]
            for col, jac in enumerate(jrow):
                gradient[col] -= jrow[col] * value
                for inner in range(dimension):
                    normal[col][inner] += jrow[col] * jacobian[row][inner]
        step = solve_symmetric(normal, gradient)
        max_step = max(abs(x) for x in step)
        values = [values[i] + step[i] for i in range(dimension)]
        if max_step < 1e-6:
            break
    return [pose_at(values, i) for i in range(len(initial))]


def solve_symmetric(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    a = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for column in range(n):
        pivot = max(range(column, n), key=lambda row: abs(a[row][column]))
        if abs(a[pivot][column]) < 1e-12:
            return [0.0] * n
        a[column], a[pivot] = a[pivot], a[column]
        pivot_value = a[column][column]
        for row in range(column + 1, n):
            factor = a[row][column] / pivot_value
            if factor == 0.0:
                continue
            for col in range(column, n + 1):
                a[row][col] -= factor * a[column][col]
    answer = [0.0] * n
    for row in range(n - 1, -1, -1):
        tail = sum(a[row][col] * answer[col] for col in range(row + 1, n))
        answer[row] = (a[row][n] - tail) / a[row][row]
    return answer
