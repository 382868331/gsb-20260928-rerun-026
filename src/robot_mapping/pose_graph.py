from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import pose_delta, wrap_angle
from .models import Pose2D, Transform2D


@dataclass(frozen=True)
class PoseConstraint:
    source: int
    target: int
    relative: Transform2D
    information: tuple[float, float, float]

    def key(self) -> tuple[int, int]:
        return (self.source, self.target)


def state_to_poses(state: list[float]) -> list[Pose2D]:
    return [Pose2D(state[i], state[i + 1], state[i + 2]) for i in range(0, len(state), 3)]


def poses_to_state(poses: list[Pose2D]) -> list[float]:
    state: list[float] = []
    for pose in poses:
        state.extend((pose.x, pose.y, pose.theta))
    return state


def constraint_residual(state: list[float], constraint: PoseConstraint) -> tuple[float, float, float]:
    i = constraint.source * 3
    j = constraint.target * 3
    source = (state[i], state[i + 1], state[i + 2])
    target = (state[j], state[j + 1], state[j + 2])
    predicted = pose_delta(source, target)
    measured = (constraint.relative.x, constraint.relative.y, constraint.relative.theta)
    error = pose_delta(measured, predicted)
    info = constraint.information
    return (error[0] * info[0], error[1] * info[1], wrap_angle(error[2]) * info[2])


def optimize_pose_graph(
    initial: list[Pose2D], constraints: list[PoseConstraint], iterations: int = 18, fixed_first: bool = True
) -> list[Pose2D]:
    state = poses_to_state(initial)
    for _ in range(iterations):
        hessian = [[0.0] * len(state) for _ in range(len(state))]
        gradient = [0.0] * len(state)
        for constraint in constraints:
            residual = constraint_residual(state, constraint)
            jacobian_i, jacobian_j = _analytic_jacobians(state, constraint)
            entries = ((constraint.source, jacobian_i), (constraint.target, jacobian_j))
            for node, jacobian in entries:
                base = node * 3
                for parameter in range(3):
                    for row in range(3):
                        gradient[base + parameter] -= jacobian[row][parameter] * residual[row]
                        for other_node, other_jacobian in entries:
                            other_base = other_node * 3
                            for other_parameter in range(3):
                                hessian[base + parameter][other_base + other_parameter] += (
                                    jacobian[row][parameter] * other_jacobian[row][other_parameter]
                                )
        if fixed_first:
            for index in range(3):
                hessian[index][index] += 1e8
        try:
            delta = _solve_symmetric(hessian, gradient)
        except ValueError:
            break
        for index, value in enumerate(delta):
            state[index] += value
            if (index + 1) % 3 == 0:
                state[index] = wrap_angle(state[index])
    return state_to_poses(state)


def _analytic_jacobians(state: list[float], constraint: PoseConstraint):
    i = constraint.source * 3
    j = constraint.target * 3
    theta_i = state[i + 2]
    dx = state[j] - state[i]
    dy = state[j + 1] - state[i + 1]
    c = math.cos(theta_i)
    s = math.sin(theta_i)
    info = constraint.information
    base_i = ((-c, -s, -s * dx + c * dy), (s, -c, -c * dx - s * dy), (0.0, 0.0, -1.0))
    base_j = ((c, s, 0.0), (-s, c, 0.0), (0.0, 0.0, 1.0))
    jacobian_i = tuple(tuple(value * info[row] for value in values) for row, values in enumerate(base_i))
    jacobian_j = tuple(tuple(value * info[row] for value in values) for row, values in enumerate(base_j))
    return jacobian_i, jacobian_j


def _solve_symmetric(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n = len(vector)
    a = [row[:] for row in matrix]
    b = vector[:]
    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(a[r][i]))
        if abs(a[pivot][i]) < 1e-12:
            raise ValueError("singular system")
        if pivot != i:
            a[i], a[pivot] = a[pivot], a[i]
            b[i], b[pivot] = b[pivot], b[i]
        pivot_value = a[i][i]
        for row in range(i + 1, n):
            factor = a[row][i] / pivot_value
            for col in range(i, n):
                a[row][col] -= factor * a[i][col]
            b[row] -= factor * b[i]
    result = [0.0] * n
    for i in range(n - 1, -1, -1):
        result[i] = (b[i] - sum(a[i][j] * result[j] for j in range(i + 1, n))) / a[i][i]
    return result


def transform_measurement_for_edge(pose_i: Pose2D, pose_j: Pose2D) -> Transform2D:
    z = pose_delta((pose_i.x, pose_i.y, pose_i.theta), (pose_j.x, pose_j.y, pose_j.theta))
    return Transform2D(*z)
