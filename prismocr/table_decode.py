"""Deterministic geometric table reconstruction and safe rendering primitives."""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import median
from typing import Literal

from prismocr.decode import GridCell, TableRender, render_table
from prismocr.geometry import polygon_area, polygon_intersection
from prismocr.graph_decode import GraphNode, decode_graph

Point = tuple[float, float]
Polygon = tuple[Point, ...]
Record = Mapping[str, object]
_MAX_RENDERED_SLOTS = 100_000


@dataclass(frozen=True)
class DecodedCell:
    """A cell's solved grid placement, associated line text and source geometry."""

    id: str
    table_id: str
    row: int
    col: int
    rowspan: int
    colspan: int
    is_header: bool | None
    text: str
    line_ids: tuple[str, ...]
    polygon: Polygon | None
    status: Literal["valid", "invalid"]


@dataclass(frozen=True)
class UnassignedCell:
    """A detected cell that could not be assigned to one table."""

    id: str
    reason: str
    polygon: Polygon | None


@dataclass(frozen=True)
class DecodedTable:
    """A table grid and its derived, conflict-checked presentations."""

    id: str
    status: Literal["valid", "empty", "conflict"]
    rows: int
    cols: int
    cells: tuple[DecodedCell, ...]
    html: str | None
    markdown: str | None
    conflicts: tuple[str, ...]


@dataclass(frozen=True)
class TableDecodeResult:
    """Tables plus cells and lines that were deliberately left unassigned."""

    tables: tuple[DecodedTable, ...]
    unassigned_cells: tuple[UnassignedCell, ...]
    unassigned_line_ids: tuple[str, ...]


@dataclass
class _Cell:
    id: str
    source: Record
    polygon: Polygon | None
    bounds: tuple[float, float, float, float] | None
    explicit_table_id: str | None
    table_id: str | None = None
    row: int | None = None
    col: int | None = None
    rowspan: int | None = None
    colspan: int | None = None
    is_header: bool | None = None
    invalid: list[str] = field(default_factory=list)


def _instance_id(record: Record) -> str:
    value = record.get("id")
    return value if isinstance(value, str) else ""


def _record_sort_key(record: Record) -> tuple[object, ...]:
    polygon = _polygon(record) or ()
    return (
        _instance_id(record),
        polygon,
        str(record.get("table_id", "")),
        repr(record.get("row")),
        repr(record.get("col")),
        repr(record.get("rowspan")),
        repr(record.get("colspan")),
        repr(record.get("is_header")),
        repr(record.get("text", "")),
    )


def _polygon(record: Record) -> Polygon | None:
    raw = record.get("polygon_px", record.get("polygon"))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) < 3:
        return None
    points: list[Point] = []
    for point in raw:
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) != 2:
            return None
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        points.append((x, y))
    polygon = tuple(points)
    area = polygon_area(polygon)
    return polygon if math.isfinite(area) and area > 0 else None


def _bounds(polygon: Polygon) -> tuple[float, float, float, float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _intersection_area(first: Polygon, second: Polygon) -> float:
    try:
        return polygon_area(polygon_intersection(first, second))
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _overlap_fraction(first: Polygon, second: Polygon) -> float:
    area = polygon_area(first)
    intersection = _intersection_area(first, second)
    if not math.isfinite(area) or not math.isfinite(intersection) or area <= 0:
        return 0.0
    return min(1.0, intersection / area)


def _cluster_tracks(values: Sequence[float], tolerance: float) -> list[float]:
    tracks: list[list[float]] = []
    for value in sorted(values):
        if not tracks or value - tracks[-1][0] > tolerance:
            tracks.append([value])
        else:
            tracks[-1].append(value)
    return [sum(group) / len(group) for group in tracks]


def _grid_fallback(cells: Sequence[_Cell]) -> dict[int, tuple[int, int, int, int]]:
    """Infer monotonically ordered grid tracks from valid cell bounds."""
    valid = [(index, cell) for index, cell in enumerate(cells) if cell.bounds is not None]
    if not valid:
        return {}
    widths = [cell.bounds[2] - cell.bounds[0] for _, cell in valid]
    heights = [cell.bounds[3] - cell.bounds[1] for _, cell in valid]
    # Cluster ordinary edge jitter conservatively. Capping each tolerance by
    # the smallest cell extent prevents a narrow real row or column from being
    # collapsed merely because most cells are much larger.
    x_tolerance = max(0.25, min(median(widths) * 0.02, min(widths) * 0.1))
    y_tolerance = max(0.25, min(median(heights) * 0.02, min(heights) * 0.1))
    adjusted_bounds = {index: list(cell.bounds) for index, cell in valid}

    # Merge only facing edges of vertically/horizontally overlapping cells
    # when the gap is small relative to both cells. This treats border gutters
    # as a shared track without merging opposite edges of a narrow real cell.
    for position, (first_index, first) in enumerate(valid):
        first_bounds = first.bounds
        first_width = first_bounds[2] - first_bounds[0]
        first_height = first_bounds[3] - first_bounds[1]
        for second_index, second in valid[position + 1 :]:
            second_bounds = second.bounds
            second_width = second_bounds[2] - second_bounds[0]
            second_height = second_bounds[3] - second_bounds[1]
            if first_bounds[2] < second_bounds[0]:
                left_index, left_bounds, left_width = first_index, first_bounds, first_width
                right_index, right_bounds, right_width = second_index, second_bounds, second_width
            elif second_bounds[2] < first_bounds[0]:
                left_index, left_bounds, left_width = second_index, second_bounds, second_width
                right_index, right_bounds, right_width = first_index, first_bounds, first_width
            else:
                left_index = right_index = -1
                left_bounds = right_bounds = first_bounds
                left_width = right_width = 0.0
            if left_index >= 0:
                gap = right_bounds[0] - left_bounds[2]
                vertical_overlap = _axis_overlap_fraction(
                    (left_bounds[1], left_bounds[3]), (right_bounds[1], right_bounds[3])
                )
                left_center = (left_bounds[0] + left_bounds[2]) / 2
                right_center = (right_bounds[0] + right_bounds[2]) / 2
                has_intervening_cell = any(
                    other_index not in {left_index, right_index}
                    and left_center < (other.bounds[0] + other.bounds[2]) / 2 < right_center
                    and _axis_overlap_fraction(
                        (left_bounds[1], left_bounds[3]), (other.bounds[1], other.bounds[3])
                    ) >= 0.5
                    for other_index, other in valid
                )
                if (
                    0.0 < gap <= 0.08 * min(left_width, right_width)
                    and vertical_overlap >= 0.5
                    and not has_intervening_cell
                ):
                    track = (left_bounds[2] + right_bounds[0]) / 2
                    adjusted_bounds[left_index][2] = track
                    adjusted_bounds[right_index][0] = track

            if first_bounds[3] < second_bounds[1]:
                upper_index, upper_bounds, upper_height = first_index, first_bounds, first_height
                lower_index, lower_bounds, lower_height = second_index, second_bounds, second_height
            elif second_bounds[3] < first_bounds[1]:
                upper_index, upper_bounds, upper_height = second_index, second_bounds, second_height
                lower_index, lower_bounds, lower_height = first_index, first_bounds, first_height
            else:
                upper_index = lower_index = -1
                upper_bounds = lower_bounds = first_bounds
                upper_height = lower_height = 0.0
            if upper_index >= 0:
                gap = lower_bounds[1] - upper_bounds[3]
                horizontal_overlap = _axis_overlap_fraction(
                    (upper_bounds[0], upper_bounds[2]), (lower_bounds[0], lower_bounds[2])
                )
                upper_center = (upper_bounds[1] + upper_bounds[3]) / 2
                lower_center = (lower_bounds[1] + lower_bounds[3]) / 2
                has_intervening_cell = any(
                    other_index not in {upper_index, lower_index}
                    and upper_center < (other.bounds[1] + other.bounds[3]) / 2 < lower_center
                    and _axis_overlap_fraction(
                        (upper_bounds[0], upper_bounds[2]), (other.bounds[0], other.bounds[2])
                    ) >= 0.5
                    for other_index, other in valid
                )
                if (
                    0.0 < gap <= 0.08 * min(upper_height, lower_height)
                    and horizontal_overlap >= 0.5
                    and not has_intervening_cell
                ):
                    track = (upper_bounds[3] + lower_bounds[1]) / 2
                    adjusted_bounds[upper_index][3] = track
                    adjusted_bounds[lower_index][1] = track

    x_tracks = _cluster_tracks(
        [edge for bounds in adjusted_bounds.values() for edge in (bounds[0], bounds[2])], x_tolerance
    )
    y_tracks = _cluster_tracks(
        [edge for bounds in adjusted_bounds.values() for edge in (bounds[1], bounds[3])], y_tolerance
    )

    def nearest(tracks: Sequence[float], value: float) -> int:
        return min(range(len(tracks)), key=lambda index: (abs(tracks[index] - value), index))

    inferred: dict[int, tuple[int, int, int, int]] = {}
    for index, _cell in valid:
        left, top, right, bottom = adjusted_bounds[index]
        col, col_end = nearest(x_tracks, left), nearest(x_tracks, right)
        row, row_end = nearest(y_tracks, top), nearest(y_tracks, bottom)
        inferred[index] = (
            row,
            col,
            max(1, row_end - row),
            max(1, col_end - col),
        )
    return inferred


def _explicit_nonnegative_int(cell: _Cell, field: str) -> int | None:
    if field not in cell.source or cell.source[field] is None:
        return None
    value = cell.source[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        cell.invalid.append(f"invalid {field} for cell {cell.id!r}")
        return None
    return value


def _explicit_span(cell: _Cell, field: str) -> int | None:
    if field not in cell.source or cell.source[field] is None:
        return None
    value = cell.source[field]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        cell.invalid.append(f"invalid {field} for cell {cell.id!r}")
        return None
    return value


def _declared_dimension(record: Record | None, field: str) -> tuple[int | None, str | None]:
    if record is None or field not in record or record[field] is None:
        return None, None
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None, f"invalid declared {field}"
    return value, None


def _layout(cells: list[_Cell]) -> list[str]:
    conflicts: list[str] = []
    inferred = _grid_fallback(cells)
    for index, cell in enumerate(cells):
        fallback = inferred.get(index, (0, index, 1, 1))
        cell.row = _explicit_nonnegative_int(cell, "row")
        cell.col = _explicit_nonnegative_int(cell, "col")
        cell.rowspan = _explicit_span(cell, "rowspan")
        cell.colspan = _explicit_span(cell, "colspan")
        cell.row = fallback[0] if cell.row is None else cell.row
        cell.col = fallback[1] if cell.col is None else cell.col
        cell.rowspan = fallback[2] if cell.rowspan is None else cell.rowspan
        cell.colspan = fallback[3] if cell.colspan is None else cell.colspan
        if cell.source.get("is_header") is not None:
            header = cell.source["is_header"]
            if isinstance(header, bool):
                cell.is_header = header
            else:
                cell.invalid.append(f"invalid is_header for cell {cell.id!r}")
        if cell.polygon is None:
            cell.invalid.append(f"invalid or missing polygon for cell {cell.id!r}")
        conflicts.extend(cell.invalid)
    return conflicts


def _line_key(record: Record, polygon: Polygon, line_id: str) -> tuple[object, ...]:
    order = record.get("reading_order")
    bounds = _bounds(polygon)
    if isinstance(order, (int, float)) and not isinstance(order, bool):
        try:
            numeric_order = float(order)
        except (TypeError, ValueError, OverflowError):
            numeric_order = float("inf")
        if math.isfinite(numeric_order):
            return (0, numeric_order, bounds[1], bounds[0], line_id)
    return (1, bounds[1], bounds[0], line_id)


def _assign_lines(
    cells: Sequence[_Cell], lines: Sequence[Record]
) -> tuple[dict[int, list[tuple[tuple[object, ...], str, str]]], tuple[str, ...]]:
    line_groups: dict[str, list[Record]] = defaultdict(list)
    for line in lines:
        line_id = _instance_id(line)
        if line_id:
            line_groups[line_id].append(line)

    assigned: dict[int, list[tuple[tuple[object, ...], str, str]]] = defaultdict(list)
    unassigned: set[str] = set()
    for line_id in sorted(line_groups):
        records = line_groups[line_id]
        if len(records) != 1:
            unassigned.add(line_id)
            continue
        line = records[0]
        polygon = _polygon(line)
        if polygon is None:
            unassigned.add(line_id)
            continue
        candidates: list[tuple[float, int]] = []
        for index, cell in enumerate(cells):
            if cell.table_id is None or cell.polygon is None:
                continue
            fraction = _overlap_fraction(polygon, cell.polygon)
            if fraction >= 0.90:
                candidates.append((fraction, index))
        candidates.sort(key=lambda item: (-item[0], cells[item[1]].id, item[1]))
        if not candidates or (
            len(candidates) > 1 and candidates[0][0] - candidates[1][0] <= 1e-6
        ):
            unassigned.add(line_id)
            continue
        index = candidates[0][1]
        text = line.get("text", "")
        assigned[index].append(
            (_line_key(line, polygon, line_id), line_id, text if isinstance(text, str) else "")
        )

    for items in assigned.values():
        items.sort(key=lambda item: item[0])
    return assigned, tuple(sorted(unassigned))


def _bounded_render(table_id: str, cells: Sequence[GridCell]) -> TableRender:
    rows = max((cell.row + cell.rowspan for cell in cells), default=0)
    cols = max((cell.col + cell.colspan for cell in cells), default=0)
    occupied_slots = sum(cell.rowspan * cell.colspan for cell in cells)
    if rows * cols > _MAX_RENDERED_SLOTS or occupied_slots > _MAX_RENDERED_SLOTS:
        issue = "table grid exceeds renderer capacity"
        return TableRender("conflict", rows, cols, None, None, (issue,))
    return render_table(table_id, cells)


def _axis_overlap_fraction(first: tuple[float, float], second: tuple[float, float]) -> float:
    first_size = first[1] - first[0]
    second_size = second[1] - second[0]
    denominator = min(first_size, second_size)
    if denominator <= 0:
        return 0.0
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0])) / denominator


def _spatial_order_conflicts(cells: Sequence[_Cell]) -> list[str]:
    """Reject learned grid coordinates that reverse clearly separated geometry."""
    conflicts: list[str] = []
    for first_index, first in enumerate(cells):
        if first.bounds is None or first.row is None or first.col is None:
            continue
        for second in cells[first_index + 1 :]:
            if second.bounds is None or second.row is None or second.col is None:
                continue
            first_left, first_top, first_right, first_bottom = first.bounds
            second_left, second_top, second_right, second_bottom = second.bounds
            first_rowspan = first.rowspan or 1
            second_rowspan = second.rowspan or 1
            first_colspan = first.colspan or 1
            second_colspan = second.colspan or 1
            same_grid_row = (
                first.row < second.row + second_rowspan
                and second.row < first.row + first_rowspan
            )
            same_grid_col = (
                first.col < second.col + second_colspan
                and second.col < first.col + first_colspan
            )
            vertical_alignment = _axis_overlap_fraction(
                (first_top, first_bottom), (second_top, second_bottom)
            ) >= 0.8
            horizontal_alignment = _axis_overlap_fraction(
                (first_left, first_right), (second_left, second_right)
            ) >= 0.8

            if vertical_alignment and first_right <= second_left:
                if not same_grid_row:
                    conflicts.append(
                        f"cells {first.id!r} and {second.id!r} have reversed spatial row assignment"
                    )
                elif first.col + first_colspan > second.col:
                    conflicts.append(
                        f"cells {first.id!r} and {second.id!r} have reversed spatial column assignment"
                    )
            elif vertical_alignment and second_right <= first_left:
                if not same_grid_row:
                    conflicts.append(
                        f"cells {first.id!r} and {second.id!r} have reversed spatial row assignment"
                    )
                elif second.col + second_colspan > first.col:
                    conflicts.append(
                        f"cells {second.id!r} and {first.id!r} have reversed spatial column assignment"
                    )

            if horizontal_alignment and first_bottom <= second_top:
                if not same_grid_col:
                    conflicts.append(
                        f"cells {first.id!r} and {second.id!r} have reversed spatial column assignment"
                    )
                elif first.row + first_rowspan > second.row:
                    conflicts.append(
                        f"cells {first.id!r} and {second.id!r} have reversed spatial row assignment"
                    )
            elif horizontal_alignment and second_bottom <= first_top:
                if not same_grid_col:
                    conflicts.append(
                        f"cells {first.id!r} and {second.id!r} have reversed spatial column assignment"
                    )
                elif second.row + second_rowspan > first.row:
                    conflicts.append(
                        f"cells {second.id!r} and {first.id!r} have reversed spatial row assignment"
                    )
    return conflicts


def _apply_right_cell_order(
    cells: Sequence[_Cell],
    relation_scores: Mapping[str, Mapping[str, object]],
    null_scores: Mapping[str, object],
) -> list[str]:
    """Apply calibrated RIGHT_CELL preferences as row-local ordering constraints."""
    by_id: dict[str, _Cell] = {}
    for cell in cells:
        if cell.id in by_id:
            return [f"duplicate cell id {cell.id!r} prevents RIGHT_CELL placement"]
        by_id[cell.id] = cell

    groups: dict[int, list[_Cell]] = defaultdict(list)
    for cell in cells:
        if cell.row is not None and cell.col is not None:
            groups[cell.row].append(cell)

    conflicts: list[str] = []
    for row, group in sorted(groups.items()):
        baseline = sorted(
            group,
            key=lambda cell: (
                cell.col if cell.col is not None else 0,
                cell.bounds[0] if cell.bounds is not None else float("inf"),
                cell.id,
            ),
        )
        table_id = group[0].table_id
        if table_id is None:
            continue
        nodes = [GraphNode(table_id, "region", kind="TABLE")]
        nodes.extend(
            GraphNode(
                cell.id,
                "cell",
                polygon=cell.polygon,
                table_id=cell.table_id,
                row=cell.row,
                col=cell.col,
            )
            for cell in group
        )
        group_ids = {cell.id for cell in group}
        scores = {
            source_id: {
                target_id: score
                for target_id, score in relation_scores.get(source_id, {}).items()
                if target_id in group_ids
            }
            for source_id in sorted(group_ids)
        }
        nulls = {cell.id: null_scores.get(cell.id) for cell in group}
        affinity_graph = decode_graph(
            nodes,
            {"RIGHT_CELL": scores},
            {"RIGHT_CELL": nulls},
            allow_right_cell_column_repair=True,
        )
        adjacency: dict[str, set[str]] = {cell.id: set() for cell in group}
        indegree = {cell.id: 0 for cell in group}
        for prediction in affinity_graph.predictions:
            if prediction.relation != "RIGHT_CELL" or prediction.target_id is None:
                continue
            adjacency[prediction.source_id].add(prediction.target_id)
            indegree[prediction.target_id] += 1

        rank = {cell.id: index for index, cell in enumerate(baseline)}
        available = [(rank[cell.id], cell.id) for cell in group if indegree[cell.id] == 0]
        heapq.heapify(available)
        ordered_ids: list[str] = []
        while available:
            _, source_id = heapq.heappop(available)
            ordered_ids.append(source_id)
            for target_id in sorted(adjacency[source_id]):
                indegree[target_id] -= 1
                if indegree[target_id] == 0:
                    heapq.heappush(available, (rank[target_id], target_id))
        if len(ordered_ids) != len(group):
            conflicts.append(f"RIGHT_CELL preferences contain a cycle in row {row}")
            continue
        baseline_ids = [cell.id for cell in baseline]
        if ordered_ids == baseline_ids:
            continue
        next_col = 0
        for cell_id in ordered_ids:
            cell = by_id[cell_id]
            cell.col = next_col
            next_col += max(1, cell.colspan or 1)

    return conflicts


def _cell_overlaps(cells: Sequence[_Cell]) -> list[str]:
    conflicts: list[str] = []
    for first_index, first in enumerate(cells):
        if first.polygon is None:
            continue
        area = polygon_area(first.polygon)
        for second in cells[first_index + 1 :]:
            if second.polygon is None:
                continue
            overlap = _intersection_area(first.polygon, second.polygon)
            if overlap > max(1e-6, min(area, polygon_area(second.polygon)) * 1e-6):
                conflicts.append(f"cell polygons {first.id!r} and {second.id!r} overlap")
    return conflicts


def reconstruct_tables(
    tables: Sequence[Record],
    cells: Sequence[Record],
    lines: Sequence[Record],
    *,
    right_cell_scores: Mapping[str, Mapping[str, object]] | None = None,
    right_cell_null_scores: Mapping[str, object] | None = None,
    use_right_cell_constraints: bool = False,
) -> TableDecodeResult:
    """Reconstruct tables from pixel polygons and optional grid/header labels.

    Explicit table/grid/header labels take precedence when valid. Otherwise,
    positive polygon overlap assigns cells to tables, and sorted cell bounds
    define row/column tracks. Optional calibrated RIGHT_CELL scores constrain
    row-local ordering before occupied-slot validation. Ambiguous ownership,
    invalid geometry, duplicate origins, and overlapping spans remain structured
    conflicts. Recognized line text is assigned at most once and HTML is emitted
    only for valid grids.
    """
    table_records = sorted(tables, key=_record_sort_key)
    table_geometry: dict[str, Polygon | None] = {}
    table_sources: dict[str, Record | None] = {}
    table_conflicts: dict[str, list[str]] = defaultdict(list)
    for record in table_records:
        table_id = _instance_id(record)
        if not table_id:
            continue
        if table_id in table_geometry:
            table_geometry[table_id] = None
            table_sources[table_id] = None
            table_conflicts[table_id].append(f"duplicate table id {table_id!r}")
            continue
        polygon = _polygon(record)
        table_geometry[table_id] = polygon
        table_sources[table_id] = record
        if polygon is None:
            table_conflicts[table_id].append(f"invalid or missing polygon for table {table_id!r}")

    grouped: dict[str, list[_Cell]] = defaultdict(list)
    unassigned: list[UnassignedCell] = []
    assignment_conflicts: dict[str, list[str]] = defaultdict(list)
    cell_records = sorted(cells, key=_record_sort_key)
    for source in cell_records:
        cell_id = _instance_id(source)
        polygon = _polygon(source)
        explicit_table = source.get("table_id")
        explicit_table_id = explicit_table if isinstance(explicit_table, str) and explicit_table else None
        cell = _Cell(
            id=cell_id,
            source=source,
            polygon=polygon,
            bounds=_bounds(polygon) if polygon is not None else None,
            explicit_table_id=explicit_table_id,
        )
        if not cell_id:
            cell.invalid.append("missing cell id")
        if explicit_table_id is not None:
            if explicit_table_id not in table_geometry:
                unassigned.append(UnassignedCell(cell_id, "unknown_table_id", polygon))
                continue
            table_polygon = table_geometry[explicit_table_id]
            if polygon is not None and table_polygon is not None:
                assigned_fraction = _overlap_fraction(polygon, table_polygon)
                if assigned_fraction <= 1e-9:
                    unassigned.append(UnassignedCell(cell_id, "outside_explicit_table", polygon))
                    continue
                competitors = sorted(
                    (
                        _overlap_fraction(polygon, other_polygon),
                        other_id,
                    )
                    for other_id, other_polygon in table_geometry.items()
                    if other_id != explicit_table_id and other_polygon is not None
                )
                if competitors and competitors[-1][0] > assigned_fraction + 0.1:
                    assignment_conflicts[explicit_table_id].append(
                        f"cell {cell_id!r} explicit table assignment contradicts geometry"
                    )
            cell.table_id = explicit_table_id
        else:
            if polygon is None:
                unassigned.append(UnassignedCell(cell_id, "invalid_polygon", polygon))
                continue
            candidates = sorted(
                (
                    _overlap_fraction(polygon, table_polygon),
                    table_id,
                )
                for table_id, table_polygon in table_geometry.items()
                if table_polygon is not None
                and _overlap_fraction(polygon, table_polygon) > 1e-9
            )
            if not candidates:
                unassigned.append(UnassignedCell(cell_id, "no_table_overlap", polygon))
                continue
            candidates.sort(key=lambda item: (-item[0], item[1]))
            if len(candidates) == 1 and candidates[0][0] >= 0.5:
                cell.table_id = candidates[0][1]
            elif (
                len(candidates) > 1
                and candidates[0][0] >= 0.9
                and candidates[1][0] <= 0.05
            ):
                cell.table_id = candidates[0][1]
            else:
                unassigned.append(UnassignedCell(cell_id, "ambiguous_table_overlap", polygon))
                continue
        grouped[cell.table_id].append(cell)

    all_assigned = [cell for table_id in sorted(grouped) for cell in grouped[table_id]]
    assigned_positions = {id(cell): index for index, cell in enumerate(all_assigned)}
    line_assignments, unassigned_lines = _assign_lines(all_assigned, lines)
    table_outputs: list[DecodedTable] = []
    for table_id in sorted(table_geometry):
        table_cells = grouped.get(table_id, [])
        table_cells.sort(key=lambda cell: _record_sort_key(cell.source))
        conflicts = list(table_conflicts.get(table_id, ()))
        conflicts.extend(assignment_conflicts.get(table_id, ()))
        conflicts.extend(_layout(table_cells))
        if use_right_cell_constraints:
            conflicts.extend(_apply_right_cell_order(
                table_cells,
                right_cell_scores or {},
                right_cell_null_scores or {},
            ))
        conflicts.extend(_spatial_order_conflicts(table_cells))
        table_cells.sort(
            key=lambda cell: (
                cell.row if cell.row is not None else 0,
                cell.col if cell.col is not None else 0,
                cell.id,
                cell.bounds or (),
            )
        )
        conflicts.extend(_cell_overlaps(table_cells))

        decoded_cells: list[DecodedCell] = []
        grid_cells: list[GridCell] = []
        for cell_index, cell in enumerate(table_cells):
            line_items = line_assignments.get(assigned_positions[id(cell)], [])
            line_ids = tuple(item[1] for item in line_items)
            text = "\n".join(item[2] for item in line_items)
            row = int(cell.row if cell.row is not None else 0)
            col = int(cell.col if cell.col is not None else cell_index)
            rowspan = int(cell.rowspan if cell.rowspan is not None else 1)
            colspan = int(cell.colspan if cell.colspan is not None else 1)
            decoded_cells.append(
                DecodedCell(
                    id=cell.id,
                    table_id=table_id,
                    row=row,
                    col=col,
                    rowspan=rowspan,
                    colspan=colspan,
                    is_header=cell.is_header,
                    text=text,
                    line_ids=line_ids,
                    polygon=cell.polygon,
                    status="invalid" if cell.invalid else "valid",
                )
            )
            grid_cells.append(
                GridCell(
                    id=cell.id,
                    row=row,
                    col=col,
                    text=text,
                    rowspan=rowspan,
                    colspan=colspan,
                    is_header=cell.is_header is True,
                    line_ids=line_ids,
                )
            )

        render: TableRender = _bounded_render(table_id, grid_cells)
        conflicts.extend(render.conflicts)
        declared_rows, rows_error = _declared_dimension(table_sources.get(table_id), "rows")
        declared_cols, cols_error = _declared_dimension(table_sources.get(table_id), "cols")
        conflicts.extend(error for error in (rows_error, cols_error) if error is not None)
        if declared_rows is not None and render.rows > declared_rows:
            conflicts.append(f"table {table_id!r} cells exceed declared row count")
        if declared_cols is not None and render.cols > declared_cols:
            conflicts.append(f"table {table_id!r} cells exceed declared column count")
        if not conflicts:
            target_rows = render.rows if declared_rows is None else declared_rows
            target_cols = render.cols if declared_cols is None else declared_cols
            if (
                target_rows > _MAX_RENDERED_SLOTS
                or target_cols > _MAX_RENDERED_SLOTS
                or target_rows * target_cols > _MAX_RENDERED_SLOTS
            ):
                conflicts.append("table grid exceeds renderer capacity")
            elif target_rows > render.rows or target_cols > render.cols:
                occupied = {
                    (row, col)
                    for cell in grid_cells
                    for row in range(cell.row, cell.row + cell.rowspan)
                    for col in range(cell.col, cell.col + cell.colspan)
                }
                ids = {cell.id for cell in grid_cells}
                for row in range(target_rows):
                    for col in range(target_cols):
                        if (row, col) in occupied:
                            continue
                        placeholder = f"__empty__:{row}:{col}"
                        while placeholder in ids:
                            placeholder += "_"
                        ids.add(placeholder)
                        grid_cells.append(GridCell(placeholder, row, col))
                render = _bounded_render(table_id, grid_cells)
                conflicts.extend(render.conflicts)
        rows, cols = render.rows, render.cols
        conflicts = sorted(set(conflicts))
        has_cells = bool(table_cells)
        if conflicts:
            status: Literal["valid", "empty", "conflict"] = "conflict"
            html = markdown = None
        elif not has_cells:
            status = "empty"
            html, markdown = render.html, render.markdown
        else:
            status = "valid"
            html, markdown = render.html, render.markdown
        table_outputs.append(
            DecodedTable(
                id=table_id,
                status=status,
                rows=rows,
                cols=cols,
                cells=tuple(decoded_cells),
                html=html,
                markdown=markdown,
                conflicts=tuple(conflicts),
            )
        )

    unassigned.sort(key=lambda item: (item.id, item.reason))
    return TableDecodeResult(tuple(table_outputs), tuple(unassigned), unassigned_lines)
