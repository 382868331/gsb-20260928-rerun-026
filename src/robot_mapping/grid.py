from __future__ import annotations

import math

from .geometry import Point
from .models import Pose2D, Scan2D


def bresenham_cells(start: Point, end: Point, resolution: float) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    sx = int(math.floor(start[0] / resolution))
    sy = int(math.floor(start[1] / resolution))
    ex = int(math.floor(end[0] / resolution))
    ey = int(math.floor(end[1] / resolution))
    x0, y0, x1, y1 = sx, sy, ex, ey
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    error = dx + dy
    free: list[tuple[int, int]] = []
    while True:
        if (x0, y0) != (x1, y1):
            free.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        error2 = 2 * error
        if error2 >= dy:
            error += dy
            x0 += step_x
        if error2 <= dx:
            error += dx
            y0 += step_y
    return free, (x1, y1)


def build_occupancy_grid(scans: list[Scan2D], poses: list[Pose2D], resolution: float = 0.1) -> dict[tuple[int, int], int]:
    if len(scans) != len(poses):
        raise ValueError("scans and poses must have equal length")
    cells: dict[tuple[int, int], int] = {}
    for scan, pose in zip(scans, poses):
        origin = (pose.x, pose.y)
        for angle, distance in zip(scan.angles, scan.ranges):
            if not (0.0 < distance < scan.max_range):
                continue
            local_end = (distance * math.cos(angle), distance * math.sin(angle))
            end = (
                pose.x + math.cos(pose.theta) * local_end[0] - math.sin(pose.theta) * local_end[1],
                pose.y + math.sin(pose.theta) * local_end[0] + math.cos(pose.theta) * local_end[1],
            )
            free_cells, occupied_cell = bresenham_cells(origin, end, resolution)
            for cell in free_cells:
                cells.setdefault(cell, 0)
            cells[occupied_cell] = 1
    return cells
