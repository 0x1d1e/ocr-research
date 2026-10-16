"""Deterministic page tiling and geometric stitching for CPU inference."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from prismocr.geometry import polygon_area, polygon_iou

Point = tuple[float, float]
Polygon = Sequence[Sequence[float]]
Size = tuple[int, int]


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _size(value: Sequence[int], name: str) -> Size:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a (width, height) pair")
    try:
        if len(value) != 2:
            raise ValueError(f"{name} must be a (width, height) pair")
    except TypeError as error:
        raise ValueError(f"{name} must be a (width, height) pair") from error
    return (
        _integer(value[0], f"{name} width", minimum=1),
        _integer(value[1], f"{name} height", minimum=1),
    )


@dataclass(frozen=True, slots=True)
class Tile:
    """A page-space crop. Coordinates and dimensions are in pixels."""

    index: int
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        _integer(self.index, "tile index", minimum=0)
        _integer(self.x, "tile x", minimum=0)
        _integer(self.y, "tile y", minimum=0)
        _integer(self.width, "tile width", minimum=1)
        _integer(self.height, "tile height", minimum=1)

    @property
    def origin(self) -> tuple[int, int]:
        return self.x, self.y

    @property
    def size(self) -> Size:
        return self.width, self.height


def _overlap_pair(overlap: int | Sequence[int], tile_size: Size) -> Size:
    if isinstance(overlap, bool):
        raise ValueError("overlap must be a non-negative integer or pair")
    if isinstance(overlap, int):
        values = overlap, overlap
    elif isinstance(overlap, Sequence) and not isinstance(overlap, (str, bytes)) and len(overlap) == 2:
        values = overlap[0], overlap[1]
    else:
        raise ValueError("overlap must be a non-negative integer or pair")
    result = (
        _integer(values[0], "horizontal overlap", minimum=0),
        _integer(values[1], "vertical overlap", minimum=0),
    )
    if result[0] >= tile_size[0] or result[1] >= tile_size[1]:
        raise ValueError("overlap must be smaller than the corresponding tile dimension")
    return result


def _axis_starts(length: int, tile_length: int, overlap: int) -> tuple[int, ...]:
    stride = tile_length - overlap
    starts = [0]
    while starts[-1] + tile_length < length:
        starts.append(starts[-1] + stride)
    return tuple(starts)


def plan_tiles(
    image_size: Sequence[int],
    tile_size: Sequence[int],
    overlap: int | Sequence[int],
) -> tuple[Tile, ...]:
    """Plan row-major tiles with full page coverage and a stable index order.

    Sizes use ``(width, height)``. Edge tiles may be smaller than ``tile_size``.
    Each step is exactly tile-size minus overlap, so requested overlap is never
    reduced at the tail and no page pixels are skipped.
    """
    page_width, page_height = _size(image_size, "image_size")
    tile_width, tile_height = _size(tile_size, "tile_size")
    overlap_x, overlap_y = _overlap_pair(overlap, (tile_width, tile_height))
    xs = _axis_starts(page_width, tile_width, overlap_x)
    ys = _axis_starts(page_height, tile_height, overlap_y)
    tiles = []
    for y in ys:
        for x in xs:
            tiles.append(
                Tile(
                    index=len(tiles),
                    x=x,
                    y=y,
                    width=min(tile_width, page_width - x),
                    height=min(tile_height, page_height - y),
                )
            )
    return tuple(tiles)


def map_tile_polygon(polygon: Polygon, tile: Tile) -> list[list[float]]:
    """Translate a tile-local polygon into page pixel coordinates."""
    if not isinstance(tile, Tile):
        raise ValueError("tile must be a Tile")
    if isinstance(polygon, (str, bytes)):
        raise ValueError("a polygon must have at least three points")
    try:
        if len(polygon) < 3:
            raise ValueError("a polygon must have at least three points")
    except TypeError as error:
        raise ValueError("a polygon must have at least three points") from error
    mapped: list[list[float]] = []
    for point in polygon:
        if isinstance(point, (str, bytes)):
            raise ValueError("polygon points must be (x, y) pairs")
        try:
            if len(point) != 2:
                raise ValueError("polygon points must be (x, y) pairs")
        except TypeError as error:
            raise ValueError("polygon points must be (x, y) pairs") from error
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("polygon coordinates must be finite")
        mapped.append([x + tile.x, y + tile.y])
    return mapped


@dataclass(frozen=True, slots=True)
class TileResult:
    """Detections emitted for one tile plus capacity accounting from its head."""

    tile: Tile
    detections: Sequence[Mapping[str, Any]]
    candidate_count: int | None = None
    capacity: int | None = None
    overflow: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.tile, Tile):
            raise ValueError("tile must be a Tile")
        if isinstance(self.detections, (str, bytes)):
            raise ValueError("detections must be a sequence of mappings")
        try:
            detections = tuple(self.detections)
        except TypeError as error:
            raise ValueError("detections must be a sequence of mappings") from error
        if any(not isinstance(item, Mapping) for item in detections):
            raise ValueError("each detection must be a mapping")
        object.__setattr__(self, "detections", detections)
        if self.candidate_count is not None:
            _integer(self.candidate_count, "candidate_count", minimum=0)
        if self.capacity is not None:
            _integer(self.capacity, "capacity", minimum=0)
        if not isinstance(self.overflow, bool):
            raise ValueError("overflow must be a boolean")


@dataclass(frozen=True, slots=True)
class StitchResult:
    """Stitched page detections and explicit partial-result diagnostics."""

    detections: tuple[dict[str, Any], ...]
    warnings: tuple[dict[str, Any], ...]
    ambiguous_merges: tuple[dict[str, Any], ...]
    overflow: bool


@dataclass(frozen=True, slots=True)
class _Node:
    tile: Tile
    detection_index: int
    raw: Mapping[str, Any]
    polygon: tuple[Point, ...]
    area: float
    center: Point
    angle: float
    long_side: float
    short_side: float
    clipped_to_tile: bool

    @property
    def ref(self) -> tuple[int, int]:
        return self.tile.index, self.detection_index


def _polygon_stats(polygon: Sequence[Sequence[float]]) -> tuple[float, Point, float, float, float]:
    if len(polygon) < 3:
        raise ValueError("detection polygons must have at least three points")
    points = tuple((float(point[0]), float(point[1])) for point in polygon)
    if any(not math.isfinite(value) for point in points for value in point):
        raise ValueError("detection polygon coordinates must be finite")
    area = polygon_area(points)
    if area <= 0:
        raise ValueError("detection polygons must have positive area")

    center = (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )
    xx = sum((point[0] - center[0]) ** 2 for point in points) / len(points)
    yy = sum((point[1] - center[1]) ** 2 for point in points) / len(points)
    xy = sum((point[0] - center[0]) * (point[1] - center[1]) for point in points) / len(points)
    angle = 0.5 * math.atan2(2.0 * xy, xx - yy)
    cosine, sine = math.cos(angle), math.sin(angle)
    along = [cosine * (x - center[0]) + sine * (y - center[1]) for x, y in points]
    across = [-sine * (x - center[0]) + cosine * (y - center[1]) for x, y in points]
    long_side = max(along) - min(along)
    short_side = max(across) - min(across)
    if short_side > long_side:
        long_side, short_side = short_side, long_side
        angle += math.pi / 2
    return area, center, angle % math.pi, long_side, short_side


def _tile_rectangles_overlap(first: Tile, second: Tile) -> bool:
    return (
        max(first.x, second.x) < min(first.x + first.width, second.x + second.width)
        and max(first.y, second.y) < min(first.y + first.height, second.y + second.height)
    )


def _text(node: _Node) -> str | None:
    value = node.raw.get("text")
    return value if isinstance(value, str) and value else None


def _touches_tile_boundary(polygon: Any, tile: Tile, tolerance: float = 1.0) -> bool:
    """Return whether an in-tile polygon was clipped to a crop boundary."""
    if isinstance(polygon, (str, bytes)):
        return False
    try:
        points = tuple((float(point[0]), float(point[1])) for point in polygon)
    except (TypeError, ValueError, IndexError, OverflowError):
        return False
    if not points or any(not math.isfinite(value) for point in points for value in point):
        return False
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    # Out-of-crop boxes are malformed input, not evidence of clipping.
    if min(xs) < -tolerance or min(ys) < -tolerance or max(xs) > tile.width + tolerance or max(ys) > tile.height + tolerance:
        return False
    return (
        min(xs) <= tolerance
        or min(ys) <= tolerance
        or max(xs) >= tile.width - tolerance
        or max(ys) >= tile.height - tolerance
    )


def _match_strength(
    first: _Node,
    second: _Node,
    *,
    iou_threshold: float,
    text_iou_threshold: float,
    area_ratio_threshold: float,
    center_distance_ratio: float,
    angle_threshold_radians: float,
) -> float | None:
    same_text = _text(first) is not None and _text(first) == _text(second)
    area_ratio = min(first.area, second.area) / max(first.area, second.area)
    first_elongation = first.long_side / first.short_side
    second_elongation = second.long_side / second.short_side
    elongated_pair = min(first_elongation, second_elongation) >= 1.2
    angle_delta = abs((first.angle - second.angle + math.pi / 2) % math.pi - math.pi / 2)
    if elongated_pair and angle_delta > angle_threshold_radians:
        return None

    dx = second.center[0] - first.center[0]
    dy = second.center[1] - first.center[1]
    center_distance = math.hypot(dx, dy)
    iou = polygon_iou(first.polygon, second.polygon)
    if iou <= 0:
        return None

    if same_text and elongated_pair and (first.clipped_to_tile or second.clipped_to_tile):
        # Relax IoU, center, and area checks only when a crop boundary clipped
        # one of the otherwise geometrically compatible text proposals.
        cosine, sine = math.cos(first.angle), math.sin(first.angle)
        along_distance = abs(dx * cosine + dy * sine)
        across_distance = abs(-dx * sine + dy * cosine)
        projected_overlap = (first.long_side + second.long_side) / 2.0 - along_distance
        if (
            area_ratio < min(area_ratio_threshold, 0.15)
            or across_distance > center_distance_ratio * max(first.short_side, second.short_side)
            or projected_overlap < 0.2 * min(first.long_side, second.long_side)
        ):
            return None
        return iou + 1e-6

    if area_ratio < area_ratio_threshold:
        return None
    if center_distance > center_distance_ratio * max(first.short_side, second.short_side):
        return None
    if iou < iou_threshold and not (same_text and iou >= text_iou_threshold):
        return None
    return iou + (1e-6 if same_text else 0.0)


def _confidence(node: _Node) -> float:
    for key in ("_tile_score", "detector_score", "score"):
        value = node.raw.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            continue
        return float(value)
    return -math.inf


def _connected_components(adjacency: Sequence[set[int]]) -> list[tuple[int, ...]]:
    unseen = set(range(len(adjacency)))
    components = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        stack = [root]
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current] & unseen, reverse=True):
                unseen.remove(neighbor)
                stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return components


def stitch_tile_results(
    tile_results: Sequence[TileResult],
    *,
    iou_threshold: float = 0.5,
    text_iou_threshold: float = 0.15,
    area_ratio_threshold: float = 0.5,
    center_distance_ratio: float = 0.75,
    angle_threshold_degrees: float = 15.0,
    id_prefix: str = "l",
) -> StitchResult:
    """Map tile-local polygons to page pixels and reconcile detections safely.

    Input detection polygons use coordinates local to their ``Tile``. Only
    cross-tile detections in geometrically overlapping tiles are compared.
    Strong polygon overlap can establish a duplicate. Identical text relaxes
    matching thresholds only for elongated proposals clipped to a tile boundary;
    angle and across-line geometry remain mandatory.
    Non-clique or same-tile multi-match components are reported as ambiguous and
    retained as separate detections. Output order and IDs are stable.
    """
    for name, value in (
        ("iou_threshold", iou_threshold),
        ("text_iou_threshold", text_iou_threshold),
        ("area_ratio_threshold", area_ratio_threshold),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if (
        isinstance(center_distance_ratio, bool)
        or not isinstance(center_distance_ratio, (int, float))
        or not math.isfinite(center_distance_ratio)
        or center_distance_ratio < 0
    ):
        raise ValueError("center_distance_ratio must be finite and non-negative")
    if (
        isinstance(angle_threshold_degrees, bool)
        or not isinstance(angle_threshold_degrees, (int, float))
        or not math.isfinite(angle_threshold_degrees)
        or not 0.0 <= angle_threshold_degrees <= 90.0
    ):
        raise ValueError("angle_threshold_degrees must be finite and in [0, 90]")
    if not isinstance(id_prefix, str) or not id_prefix:
        raise ValueError("id_prefix must be a non-empty string")

    try:
        results = tuple(tile_results)
    except TypeError as error:
        raise ValueError("tile_results must be a sequence of TileResult values") from error
    if any(not isinstance(item, TileResult) for item in results):
        raise ValueError("tile_results must contain TileResult values")
    results = tuple(sorted(results, key=lambda item: item.tile.index))
    if len({item.tile.index for item in results}) != len(results):
        raise ValueError("tile result indices must be unique")

    warnings: list[dict[str, Any]] = []
    overflow = False
    nodes: list[_Node] = []
    node_indices_by_tile: dict[int, list[int]] = {}
    for result in results:
        tile = result.tile
        tile_overflow = result.overflow or (
            result.candidate_count is not None
            and result.capacity is not None
            and result.candidate_count > result.capacity
        )
        if tile_overflow:
            overflow = True
            warnings.append(
                {
                    "code": "TILE_CAPACITY_OVERFLOW",
                    "tile_index": tile.index,
                    "candidate_count": result.candidate_count,
                    "capacity": result.capacity,
                }
            )
        for detection_index, detection in enumerate(result.detections):
            if "polygon" not in detection:
                raise ValueError(f"tile {tile.index} detection {detection_index} has no polygon")
            page_polygon = map_tile_polygon(detection["polygon"], tile)
            area, center, angle, long_side, short_side = _polygon_stats(page_polygon)
            node_indices_by_tile.setdefault(tile.index, []).append(len(nodes))
            nodes.append(
                _Node(
                    tile=tile,
                    detection_index=detection_index,
                    raw=detection,
                    polygon=tuple((point[0], point[1]) for point in page_polygon),
                    area=area,
                    center=center,
                    angle=angle,
                    long_side=long_side,
                    short_side=short_side,
                    clipped_to_tile=_touches_tile_boundary(detection["polygon"], tile),
                )
            )

    adjacency = [set() for _ in nodes]
    angle_limit = math.radians(angle_threshold_degrees)
    for first_position, first_result in enumerate(results):
        first_indices = node_indices_by_tile.get(first_result.tile.index, ())
        for second_result in results[first_position + 1 :]:
            if not _tile_rectangles_overlap(first_result.tile, second_result.tile):
                continue
            second_indices = node_indices_by_tile.get(second_result.tile.index, ())
            for first_index in first_indices:
                for second_index in second_indices:
                    first, second = nodes[first_index], nodes[second_index]
                    strength = _match_strength(
                        first,
                        second,
                        iou_threshold=iou_threshold,
                        text_iou_threshold=text_iou_threshold,
                        area_ratio_threshold=area_ratio_threshold,
                        center_distance_ratio=center_distance_ratio,
                        angle_threshold_radians=angle_limit,
                    )
                    if strength is not None:
                        adjacency[first_index].add(second_index)
                        adjacency[second_index].add(first_index)

    clusters: list[tuple[int, ...]] = []
    ambiguous: list[dict[str, Any]] = []
    for component in _connected_components(adjacency):
        if len(component) == 1:
            clusters.append(component)
            continue
        complete = all(
            second in adjacency[first]
            for offset, first in enumerate(component)
            for second in component[offset + 1 :]
        )
        distinct_tiles = len({nodes[index].tile.index for index in component}) == len(component)
        if complete and distinct_tiles:
            clusters.append(component)
            continue
        refs = [
            {"tile_index": nodes[index].tile.index, "detection_index": nodes[index].detection_index}
            for index in component
        ]
        ambiguous.append({"code": "AMBIGUOUS_TILE_MERGE", "detection_refs": refs})
        clusters.extend((index,) for index in component)

    merged: list[tuple[tuple[int, ...], dict[str, Any], tuple[float, ...]]] = []
    for cluster in clusters:
        members = tuple(sorted(cluster, key=lambda index: nodes[index].ref))
        representative_index = min(
            members,
            key=lambda index: (-_confidence(nodes[index]), nodes[index].tile.index, nodes[index].detection_index),
        )
        representative = nodes[representative_index]
        output = dict(representative.raw)
        output["polygon"] = [[x, y] for x, y in representative.polygon]
        output["source_tiles"] = sorted({nodes[index].tile.index for index in members})
        output["source_detections"] = [
            {
                "tile_index": node.tile.index,
                "tile_origin": [node.tile.x, node.tile.y],
                "tile_size": [node.tile.width, node.tile.height],
                "polygon_page": [[x, y] for x, y in node.polygon],
                "detection": dict(node.raw),
            }
            for node in (nodes[index] for index in members)
        ]
        min_x = min(point[0] for point in representative.polygon)
        min_y = min(point[1] for point in representative.polygon)
        merged.append((members, output, (representative.center[1], representative.center[0], min_y, min_x)))

    merged.sort(
        key=lambda item: (
            item[2],
            min(nodes[index].tile.index for index in item[0]),
            min(nodes[index].detection_index for index in item[0]),
        )
    )
    detections = []
    for number, (_, output, _) in enumerate(merged, start=1):
        output["id"] = f"{id_prefix}:{number}"
        detections.append(output)
    ambiguous.sort(key=lambda item: tuple(
        (ref["tile_index"], ref["detection_index"]) for ref in item["detection_refs"]
    ))
    warnings.extend(ambiguous)
    return StitchResult(
        detections=tuple(detections),
        warnings=tuple(warnings),
        ambiguous_merges=tuple(ambiguous),
        overflow=overflow,
    )
