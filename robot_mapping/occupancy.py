from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import Pose


@dataclass
class OccupancyGrid:
    resolution: float
    width: int
    height: int
    origin_x: float
    origin_y: float
    occupied: set[tuple[int, int]]
    free: set[tuple[int, int]]

    @classmethod
    def create(cls, bounds: tuple[float, float, float, float], resolution: float = 0.1) -> "OccupancyGrid":
        min_x, min_y, max_x, max_y = bounds
        width = math.ceil((max_x - min_x) / resolution) + 2
        height = math.ceil((max_y - min_y) / resolution) + 2
        return cls(resolution, width, height, min_x - resolution, min_y - resolution, set(), set())

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        return int((x - self.origin_x) / self.resolution), int((y - self.origin_y) / self.resolution)

    def cell_center(self, cell_x: int, cell_y: int) -> tuple[float, float]:
        return (
            self.origin_x + (cell_x + 0.5) * self.resolution,
            self.origin_y + (cell_y + 0.5) * self.resolution,
        )

    def insert_scan(self, pose: Pose, points: list[tuple[float, float]], max_range: float = 8.0) -> None:
        start = self.world_to_cell(pose.x, pose.y)
        for local_x, local_y in points:
            x, y = pose.transform((local_x, local_y))
            end = self.world_to_cell(x, y)
            self._raycast(start, end, max_range)
            self.occupied.add(end)
            self.free.discard(end)

    def _raycast(self, start: tuple[int, int], end: tuple[int, int], max_range: float) -> None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        steps = max(abs(dx), abs(dy), 1)
        if math.hypot(dx * self.resolution, dy * self.resolution) > max_range:
            steps = int(max_range / self.resolution)
        for step in range(steps):
            ratio = step / steps
            cell = (int(round(start[0] + dx * ratio)), int(round(start[1] + dy * ratio)))
            if cell != end and cell not in self.occupied:
                self.free.add(cell)

    def to_pgm(self) -> str:
        header = f"P2\n{self.width} {self.height}\n255\n"
        rows = []
        for y in reversed(range(self.height)):
            row = []
            for x in range(self.width):
                cell = (x, y)
                if cell in self.occupied:
                    row.append("0")
                elif cell in self.free:
                    row.append("230")
                else:
                    row.append("255")
            rows.append(" ".join(row))
        return header + "\n".join(rows) + "\n"
