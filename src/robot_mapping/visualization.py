from __future__ import annotations

from .models import MapVersion, Pose2D


def svg_document(width: int, height: int, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
        + "".join(body)
        + "</svg>"
    )


def render_map_svg(
    version: MapVersion,
    truth: list[Pose2D] | None = None,
    width: int = 720,
    height: int = 720,
    title: str = "",
) -> str:
    points = [(pose.x, pose.y) for pose in version.poses]
    if truth is not None:
        points.extend((pose.x, pose.y) for pose in truth)
    for (cell_x, cell_y), occupied in version.cells.items():
        if occupied:
            points.append((cell_x * version.resolution, cell_y * version.resolution))
    min_x = min(point[0] for point in points) - 0.5
    max_x = max(point[0] for point in points) + 0.5
    min_y = min(point[1] for point in points) - 0.5
    max_y = max(point[1] for point in points) + 0.5
    scale = min((width - 80) / (max_x - min_x), (height - 100) / (max_y - min_y))

    def project(x: float, y: float) -> tuple[float, float]:
        return (40 + (x - min_x) * scale, height - 40 - (y - min_y) * scale)

    body = [f'<text x="18" y="26" font-family="sans-serif" font-size="18">{title}</text>']
    for (cell_x, cell_y), occupied in version.cells.items():
        x, y = project(cell_x * version.resolution, cell_y * version.resolution)
        size = max(1.0, version.resolution * scale)
        color = "#111827" if occupied else "#e5e7eb"
        body.append(f'<rect x="{x-size/2:.2f}" y="{y-size/2:.2f}" width="{size:.2f}" height="{size:.2f}" fill="{color}"/>')
    body.append(_path_svg(truth or [], project, "#2563eb", 2.0, "truth path"))
    body.append(_path_svg(list(version.poses), project, "#dc2626", 2.4, "estimated path"))
    return svg_document(width, height, body)


def _path_svg(poses: list[Pose2D], project, color: str, width: float, label: str) -> str:
    if not poses:
        return ""
    coords = [project(pose.x, pose.y) for pose in poses]
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    start = coords[0]
    return (
        f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"/>'
        f'<circle cx="{start[0]:.1f}" cy="{start[1]:.1f}" r="4" fill="{color}"><title>{label} start</title></circle>'
    )
