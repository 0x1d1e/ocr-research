"""Annotation polygon conversion to the model's six-value box contract."""

from __future__ import annotations

import math
from collections.abc import Sequence


def polygon_to_box(
    polygon_px: Sequence[Sequence[float]], image_size: tuple[int, int]
) -> tuple[float, float, float, float, float, float]:
    """Return the minimum-area oriented rectangle in normalized page space.

    ``image_size`` is ``(width, height)``. The orientation is pixel-space and
    canonicalized to ``[-pi/2, pi/2)``. This conversion approximates arbitrary
    polygons by one rectangle; source polygons remain the annotation truth.
    """
    width, height = image_size
    if width <= 0 or height <= 0 or len(polygon_px) < 3:
        raise ValueError("positive image dimensions and at least three points are required")
    points = [(float(p[0]), float(p[1])) for p in polygon_px]
    if any(not math.isfinite(v) for p in points for v in p):
        raise ValueError("polygon coordinates must be finite")

    # A minimum-area enclosing rectangle is supported by a convex-hull edge.
    # Original annotation edges are insufficient for concave polygons.
    ordered = sorted(set(points))

    def cross(origin: tuple[float, float], first: tuple[float, float], second: tuple[float, float]) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1] if len(ordered) > 1 else ordered

    candidates: list[tuple[float, float, float, float, float, float]] = []
    for i, first in enumerate(hull):
        second = hull[(i + 1) % len(hull)]
        dx, dy = second[0] - first[0], second[1] - first[1]
        if math.hypot(dx, dy) < 1e-9:
            continue
        theta = math.atan2(dy, dx)
        cosine, sine = math.cos(theta), math.sin(theta)
        projected = [(cosine * x + sine * y, -sine * x + cosine * y) for x, y in points]
        min_x = min(p[0] for p in projected)
        max_x = max(p[0] for p in projected)
        min_y = min(p[1] for p in projected)
        max_y = max(p[1] for p in projected)
        box_w, box_h = max_x - min_x, max_y - min_y
        area = box_w * box_h
        center_rotated = ((min_x + max_x) / 2, (min_y + max_y) / 2)
        center = (
            cosine * center_rotated[0] - sine * center_rotated[1],
            sine * center_rotated[0] + cosine * center_rotated[1],
        )
        # Recognition always advances along the pixel-space long axis. This
        # removes the 90-degree ambiguity when equivalent vertices start on
        # different edges or are supplied in reverse winding order.
        if box_h > box_w:
            box_w, box_h = box_h, box_w
            theta += math.pi / 2
        theta = (theta + math.pi / 2) % math.pi - math.pi / 2
        candidates.append((area, center[0], center[1], box_w, box_h, theta))
    if not candidates:
        raise ValueError("polygon has zero area")
    minimum_area = min(candidate[0] for candidate in candidates)
    area_tolerance = max(1.0, minimum_area) * 1e-10
    minimum_candidates = [
        candidate for candidate in candidates
        if candidate[0] <= minimum_area + area_tolerance
    ]
    _, cx, cy, box_w, box_h, theta = min(
        minimum_candidates,
        key=lambda candidate: (
            round(candidate[5], 12),
            round(candidate[1], 9),
            round(candidate[2], 9),
            round(candidate[3], 9),
            round(candidate[4], 9),
        ),
    )
    if box_w <= 0 or box_h <= 0:
        raise ValueError("polygon has zero area")
    return (
        cx / width,
        cy / height,
        box_w / width,
        box_h / height,
        math.sin(theta),
        math.cos(theta),
    )
