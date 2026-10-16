"""Aspect-aware oriented geometry used by matching, OCR sampling and decoding."""

from __future__ import annotations

import math
from collections.abc import Sequence

Point = tuple[float, float]
Box = Sequence[float]
Polygon = Sequence[Point]


def normalize_orientation(sin_theta: float, cos_theta: float) -> tuple[float, float]:
    norm = math.hypot(float(sin_theta), float(cos_theta))
    if not math.isfinite(norm) or norm < 1e-8:
        return 0.0, 1.0
    return float(sin_theta) / norm, float(cos_theta) / norm


def rotated_box_to_polygon(
    box: Box, image_size: tuple[int, int] = (1, 1)
) -> list[Point]:
    """Convert normalized ``cx,cy,w,h,sin,cos`` to pixel-space corners.

    ``image_size`` is ``(width, height)``. Angles are measured in pixel space,
    counter-clockwise in the usual Cartesian convention. The four corners are
    returned in consistent positive signed-area order.
    """
    if len(box) != 6:
        raise ValueError("oriented boxes must contain exactly six values")
    width_px, height_px = image_size
    if width_px <= 0 or height_px <= 0:
        raise ValueError("image dimensions must be positive")
    cx, cy, width, height = (float(v) for v in box[:4])
    sin_theta, cos_theta = normalize_orientation(float(box[4]), float(box[5]))
    if not all(math.isfinite(v) for v in (cx, cy, width, height)):
        raise ValueError("box values must be finite")
    if width <= 0 or height <= 0:
        raise ValueError("box width and height must be positive")

    cx *= width_px
    cy *= height_px
    width *= width_px
    height *= height_px
    ux, uy = cos_theta, sin_theta
    vx, vy = -sin_theta, cos_theta
    return [
        (
            cx + sx * ux * width / 2 + sy * vx * height / 2,
            cy + sx * uy * width / 2 + sy * vy * height / 2,
        )
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]


def signed_polygon_area(polygon: Polygon) -> float:
    if len(polygon) < 3:
        return 0.0
    return 0.5 * sum(
        polygon[i][0] * polygon[(i + 1) % len(polygon)][1]
        - polygon[(i + 1) % len(polygon)][0] * polygon[i][1]
        for i in range(len(polygon))
    )


def polygon_area(polygon: Polygon) -> float:
    return abs(signed_polygon_area(polygon))


def _cross(a: Point, b: Point, p: Point) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _line_intersection(a: Point, b: Point, c: Point, d: Point) -> Point:
    abx, aby = b[0] - a[0], b[1] - a[1]
    cdx, cdy = d[0] - c[0], d[1] - c[1]
    denom = abx * cdy - aby * cdx
    if abs(denom) < 1e-12:
        return b
    t = ((c[0] - a[0]) * cdy - (c[1] - a[1]) * cdx) / denom
    return a[0] + t * abx, a[1] + t * aby


def polygon_intersection(subject: Polygon, clip: Polygon) -> list[Point]:
    """Convex polygon clipping. Inputs may use either winding direction."""
    output = list(subject)
    clip_points = list(clip)
    if len(output) < 3 or len(clip_points) < 3:
        return []
    if signed_polygon_area(clip_points) < 0:
        clip_points.reverse()

    for i, edge_start in enumerate(clip_points):
        edge_end = clip_points[(i + 1) % len(clip_points)]
        source = output
        output = []
        if not source:
            break
        previous = source[-1]
        previous_inside = _cross(edge_start, edge_end, previous) >= -1e-9
        for current in source:
            current_inside = _cross(edge_start, edge_end, current) >= -1e-9
            if current_inside:
                if not previous_inside:
                    output.append(_line_intersection(previous, current, edge_start, edge_end))
                output.append(current)
            elif previous_inside:
                output.append(_line_intersection(previous, current, edge_start, edge_end))
            previous, previous_inside = current, current_inside
    return output


def _point_in_triangle(point: Point, a: Point, b: Point, c: Point, epsilon: float = 1e-10) -> bool:
    return (
        _cross(a, b, point) >= -epsilon
        and _cross(b, c, point) >= -epsilon
        and _cross(c, a, point) >= -epsilon
    )


def polygon_is_simple(polygon: Polygon, *, max_vertices: int = 4096) -> bool:
    """Return whether the closed polygon is a simple, non-self-intersecting ring.

    Repeated consecutive vertices are collapsed first. A polygon with more than
    ``max_vertices`` distinct vertices raises ``ValueError`` so callers can
    reject pathological input instead of paying an unbounded quadratic scan.
    """
    cleaned: list[Point] = []
    for point in polygon:
        item = (float(point[0]), float(point[1]))
        if not cleaned or item != cleaned[-1]:
            cleaned.append(item)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    if len(cleaned) < 3:
        return False
    if len(cleaned) > max_vertices:
        raise ValueError(
            f"polygon has {len(cleaned)} vertices, above the {max_vertices}-vertex simplicity limit"
        )
    count = len(cleaned)
    for first in range(count):
        a, b = cleaned[first], cleaned[(first + 1) % count]
        for second in range(first + 1, count):
            if second == first or (first + 1) % count == second or (second + 1) % count == first:
                continue
            c, d = cleaned[second], cleaned[(second + 1) % count]
            if _segments_intersect(a, b, c, d):
                return False
    return True


def _segments_intersect(a: Point, b: Point, c: Point, d: Point, epsilon: float = 1e-9) -> bool:
    """Closed-segment intersection, including collinear overlap."""

    def orientation(p: Point, q: Point, r: Point) -> int:
        value = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        if value > epsilon:
            return 1
        if value < -epsilon:
            return -1
        return 0

    def on_segment(p: Point, q: Point, r: Point) -> bool:
        return (
            min(p[0], r[0]) - epsilon <= q[0] <= max(p[0], r[0]) + epsilon
            and min(p[1], r[1]) - epsilon <= q[1] <= max(p[1], r[1]) + epsilon
        )

    first, second = orientation(a, b, c), orientation(a, b, d)
    third, fourth = orientation(c, d, a), orientation(c, d, b)
    if first != second and third != fourth:
        return True
    if first == 0 and on_segment(a, c, b):
        return True
    if second == 0 and on_segment(a, d, b):
        return True
    if third == 0 and on_segment(c, a, d):
        return True
    if fourth == 0 and on_segment(c, b, d):
        return True
    return False


def _triangulate_simple_polygon(polygon: Polygon) -> list[list[Point]]:
    """Triangulate a simple polygon by clipping convex ears in O(n²)."""
    points: list[Point] = []
    for point in polygon:
        item = (float(point[0]), float(point[1]))
        if not points or item != points[-1]:
            points.append(item)
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    if len(points) < 3 or polygon_area(points) <= 0:
        return []
    if signed_polygon_area(points) < 0:
        points.reverse()

    # Collinear vertices add no area and can prevent ear clipping from making progress.
    changed = True
    while changed and len(points) > 3:
        changed = False
        for index in range(len(points)):
            previous = points[index - 1]
            current = points[index]
            following = points[(index + 1) % len(points)]
            if abs(_cross(previous, current, following)) <= 1e-10:
                points.pop(index)
                changed = True
                break
    if len(points) < 3:
        return []

    remaining = list(range(len(points)))
    triangles: list[list[Point]] = []
    while len(remaining) > 3:
        ear_found = False
        for position, current_index in enumerate(remaining):
            previous_index = remaining[position - 1]
            next_index = remaining[(position + 1) % len(remaining)]
            a, b, c = points[previous_index], points[current_index], points[next_index]
            if _cross(a, b, c) <= 1e-10:
                continue
            if any(
                _point_in_triangle(points[other], a, b, c)
                for other in remaining
                if other not in {previous_index, current_index, next_index}
            ):
                continue
            triangles.append([a, b, c])
            remaining.pop(position)
            ear_found = True
            break
        if not ear_found:
            raise ValueError("polygon must be simple and non-self-intersecting")
    triangles.append([points[index] for index in remaining])
    return triangles


def source_polygon_iou(first: Polygon, second: Polygon) -> float:
    """IoU of the supplied simple polygons, including concave annotation geometry."""
    area_a, area_b = polygon_area(first), polygon_area(second)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    triangles_a = _triangulate_simple_polygon(first)
    triangles_b = _triangulate_simple_polygon(second)
    if not triangles_a or not triangles_b:
        return 0.0
    intersection = sum(
        polygon_area(polygon_intersection(triangle_a, triangle_b))
        for triangle_a in triangles_a
        for triangle_b in triangles_b
    )
    intersection = min(area_a, area_b, intersection)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else max(0.0, min(1.0, intersection / union))


def polygon_iou(first: Polygon, second: Polygon) -> float:
    area_a, area_b = polygon_area(first), polygon_area(second)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    intersection = polygon_area(polygon_intersection(first, second))
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else max(0.0, min(1.0, intersection / union))


def rotated_iou(
    first: Box, second: Box, image_size: tuple[int, int] = (1, 1)
) -> float:
    try:
        return polygon_iou(
            rotated_box_to_polygon(first, image_size),
            rotated_box_to_polygon(second, image_size),
        )
    except (TypeError, ValueError, OverflowError):
        return 0.0
