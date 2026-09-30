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


def affected_nodes(edges: list[PoseEdge], seeds: set[int], hop_limit: int | None = None) -> set[int]:
    """Breadth-first propagation: nodes connected through odometry are locally influenced."""
    adjacency: dict[int, set[int]] = {}
    for edge in edges:
        adjacency.setdefault(edge.i, set()).add(edge.j)
        adjacency.setdefault(edge.j, set()).add(edge.i)
    reached = set(seeds)
    frontier = [(seed, 0) for seed in seeds]
    while frontier:
        current, depth = frontier.pop()
        if hop_limit is not None and depth >= hop_limit:
            continue
        for neighbour in adjacency.get(current, ()):
            if neighbour not in reached:
                reached.add(neighbour)
                frontier.append((neighbour, depth + 1))
    return reached


def optimize_subgraph(initial: list[Pose], edges: list[PoseEdge], active: set[int]) -> list[Pose]:
    """Optimize only the locally influenced nodes; boundary nodes are anchored."""
    ordered = sorted(active)
    index_map = {node: position for position, node in enumerate(ordered)}
    sub_edges: list[PoseEdge] = []
    for edge in edges:
        if edge.i == edge.j:
            continue
        if edge.i in active and edge.j in active:
            sub_edges.append(PoseEdge(index_map[edge.i], index_map[edge.j], edge.measurement, edge.weight))
    sub_initial = [initial[node] for node in ordered]
    # Strong anchoring priors keep the local window tied to the untouched map.
    anchors = {index_map[node] for node in ordered if node == min(ordered) or node == max(ordered)}
    optimized = optimize_poses(sub_initial, sub_edges, fixed=anchors)
    result = list(initial)
    for position, node in enumerate(ordered):
        if position in anchors:
            continue
        result[node] = optimized[position]
    return result


def relative_pose(a: Pose, b: Pose) -> Pose:
    return compose_poses(invert_transform(a), b)


def optimize_poses(initial: list[Pose], edges: list[PoseEdge], iterations: int = 12, fixed: set[int] | None = None) -> list[Pose]:
    values: list[float] = []
    for pose in initial:
        values.extend(pose.as_tuple())

    def pose_at(vector: list[float], index: int) -> Pose:
        base = index * 3
        return Pose(vector[base], vector[base + 1], vector[base + 2])

    fixed_nodes = fixed or {0}

    def residual_and_jacobian(vector: list[float]) -> tuple[list[float], list[list[float]]]:
        result: list[float] = []
        jacobian: list[list[float]] = []
        for index in range(len(initial)):
            if index in fixed_nodes:
                base = index * 3
                result.extend([
                    (vector[base] - initial[index].x) * 10000.0,
                    (vector[base + 1] - initial[index].y) * 10000.0,
                    (wrap_angle(vector[base + 2] - initial[index].theta)) * 10000.0,
                ])
                row_x = [0.0] * dimension
                row_y = [0.0] * dimension
                row_t = [0.0] * dimension
                row_x[base] = 10000.0
                row_y[base + 1] = 10000.0
                row_t[base + 2] = 10000.0
                jacobian.extend([row_x, row_y, row_t])
        for edge in edges:
            base_i = edge.i * 3
            base_j = edge.j * 3
            xi, yi, ti = vector[base_i], vector[base_i + 1], vector[base_i + 2]
            xj, yj, tj = vector[base_j], vector[base_j + 1], vector[base_j + 2]
            mx, my, mt = edge.measurement.as_tuple()
            dx = xj - xi
            dy = yj - yi
            dt = wrap_angle(tj - ti)
            ci = math.cos(ti)
            si = math.sin(ti)
            rx = ci * dx + si * dy - mx
            ry = -si * dx + ci * dy - my
            rt = wrap_angle(dt - mt)
            scale = edge.weight
            result.extend([rx * scale, ry * scale, rt * scale])
            row_x = [0.0] * dimension
            row_y = [0.0] * dimension
            row_t = [0.0] * dimension
            row_x[base_i] = -ci
            row_x[base_i + 1] = -si
            row_x[base_i + 2] = -si * dx + ci * dy
            row_x[base_j] = ci
            row_x[base_j + 1] = si
            row_y[base_i] = si
            row_y[base_i + 1] = -ci
            row_y[base_i + 2] = -ci * dx - si * dy
            row_y[base_j] = -si
            row_y[base_j + 1] = ci
            row_t[base_i + 2] = -1.0
            row_t[base_j + 2] = 1.0
            jacobian.extend([[value * scale for value in row] for row in (row_x, row_y, row_t)])
        return result, jacobian

    dimension = len(values)
    for _ in range(iterations):
        r0, jacobian = residual_and_jacobian(values)
        sparse_normal: list[dict[int, float]] = [dict() for _ in range(dimension)]
        gradient = [0.0] * dimension
        for row, value in enumerate(r0):
            nonzero = [(col, jac) for col, jac in enumerate(jacobian[row]) if jac != 0.0]
            for col, jac in nonzero:
                gradient[col] -= jac * value
                nrow = sparse_normal[col]
                for inner, inner_jac in nonzero:
                    nrow[inner] = nrow.get(inner, 0.0) + jac * inner_jac
        step = solve_sparse(sparse_normal, gradient)
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


def solve_sparse(matrix: list[dict[int, float]], vector: list[float]) -> list[float]:
    n = len(vector)
    a = [dict(row) for row in matrix]
    for column in range(n):
        candidates = [row for row in range(column, n) if column in a[row] and abs(a[row][column]) > 1e-12]
        if not candidates:
            continue
        pivot = max(candidates, key=lambda row: abs(a[row][column]))
        a[column], a[pivot] = a[pivot], a[column]
        vector[column], vector[pivot] = vector[pivot], vector[column]
        pivot_value = a[column][column]
        for row in range(column + 1, n):
            if column not in a[row]:
                continue
            factor = a[row][column] / pivot_value
            if factor == 0.0:
                continue
            for col, value in a[column].items():
                if col < column:
                    continue
                updated = a[row].get(col, 0.0) - factor * value
                if abs(updated) > 1e-14:
                    a[row][col] = updated
                else:
                    a[row].pop(col, None)
            vector[row] -= factor * vector[column]
    answer = [0.0] * n
    for row in range(n - 1, -1, -1):
        tail = sum(value * answer[col] for col, value in a[row].items() if col > row)
        pivot = a[row].get(row)
        answer[row] = (vector[row] - tail) / pivot if pivot else 0.0
    return answer
