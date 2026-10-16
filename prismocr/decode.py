"""Deterministic graph ordering and conflict-checked table presentation."""

from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from typing import Any


@dataclass(frozen=True)
class GridCell:
    id: str
    row: int
    col: int
    text: str = ""
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False
    line_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TableRender:
    status: str
    rows: int
    cols: int
    html: str | None
    markdown: str | None
    conflicts: tuple[str, ...]


def _geometry_key(instance_id: str, boxes: Mapping[str, Sequence[float]]) -> tuple[float, float, str]:
    box = boxes[instance_id]
    cx, cy, _, height = (float(v) for v in box[:4])
    row_scale = max(abs(height) * 0.75, 0.01)
    return round(cy / row_scale), cx, instance_id


def traverse_reading_order(
    active_ids: Sequence[str],
    next_scores: Mapping[str, Mapping[str, float]],
    null_scores: Mapping[str, float],
    boxes: Mapping[str, Sequence[float]],
    *,
    threshold: float = 0.0,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Resolve one outgoing NEXT edge per line, cut cycles and retain every node."""
    nodes = sorted(set(active_ids))
    if not nodes:
        return [], []
    node_set = set(nodes)
    edges: dict[str, tuple[str, float]] = {}
    for source in nodes:
        candidates = [
            (float(score), target)
            for target, score in next_scores.get(source, {}).items()
            if target in node_set and target != source
        ]
        if not candidates:
            continue
        score, target = max(candidates, key=lambda item: (item[0], item[1]))
        if score > float(null_scores.get(source, float("inf"))) and score >= threshold:
            edges[source] = (target, score)

    warnings: list[dict[str, Any]] = []
    settled: set[str] = set()
    for start in nodes:
        if start in settled:
            continue
        path: list[str] = []
        position: dict[str, int] = {}
        current: str | None = start
        while current is not None and current not in settled and current not in position:
            position[current] = len(path)
            path.append(current)
            current = edges[current][0] if current in edges else None
        if current is not None and current in position:
            cycle = path[position[current] :]
            weakest = min(cycle, key=lambda node: (edges[node][1], node))
            target, score = edges.pop(weakest)
            warnings.append({"code": "NEXT_CYCLE_BROKEN", "source": weakest, "target": target, "score": score})
        settled.update(path)

    indegree = {node: 0 for node in nodes}
    adjacency = {node: [] for node in nodes}
    for source, (target, _) in edges.items():
        adjacency[source].append(target)
        indegree[target] += 1
    heap = [(_geometry_key(node, boxes), node) for node in nodes if indegree[node] == 0]
    heapq.heapify(heap)
    ordered: list[str] = []
    while heap:
        _, source = heapq.heappop(heap)
        ordered.append(source)
        for target in adjacency[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(heap, (_geometry_key(target, boxes), target))
    if len(ordered) != len(nodes):
        # Defensive deterministic fallback. This should be unreachable after cycle cutting.
        remaining = sorted(node_set - set(ordered), key=lambda node: _geometry_key(node, boxes))
        warnings.append({"code": "ORDER_FALLBACK", "remaining": remaining})
        ordered.extend(remaining)
    return ordered, warnings


def validate_grid(cells: Sequence[GridCell]) -> tuple[tuple[str, ...], int, int]:
    conflicts: list[str] = []
    origins: dict[tuple[int, int], GridCell] = {}
    occupancy: dict[tuple[int, int], str] = {}
    max_row = max_col = 0
    seen_ids: set[str] = set()
    for cell in cells:
        if cell.id in seen_ids:
            conflicts.append(f"duplicate cell id {cell.id!r}")
        seen_ids.add(cell.id)
        if cell.row < 0 or cell.col < 0 or cell.rowspan <= 0 or cell.colspan <= 0:
            conflicts.append(f"invalid origin or span for cell {cell.id!r}")
            continue
        origin = (cell.row, cell.col)
        if origin in origins:
            conflicts.append(f"cells {origins[origin].id!r} and {cell.id!r} share origin {origin}")
        else:
            origins[origin] = cell
        max_row = max(max_row, cell.row + cell.rowspan)
        max_col = max(max_col, cell.col + cell.colspan)
        for row in range(cell.row, cell.row + cell.rowspan):
            for col in range(cell.col, cell.col + cell.colspan):
                previous = occupancy.get((row, col))
                if previous is not None:
                    conflicts.append(f"cells {previous!r} and {cell.id!r} overlap at ({row},{col})")
                else:
                    occupancy[(row, col)] = cell.id
    return tuple(conflicts), max_row, max_col


def render_table(table_id: str, cells: Sequence[GridCell]) -> TableRender:
    conflicts, rows, cols = validate_grid(cells)
    if conflicts:
        return TableRender("conflict", rows, cols, None, None, conflicts)
    if rows == 0 or cols == 0:
        return TableRender("empty", rows, cols, "<table></table>", "", ())

    by_origin = {(cell.row, cell.col): cell for cell in cells}
    occupied: set[tuple[int, int]] = set()
    table_attr = f' id="{escape(table_id, quote=True)}"' if table_id else ""
    html_rows: list[str] = [f"<table{table_attr}>"]
    for row in range(rows):
        html_rows.append("  <tr>")
        for col in range(cols):
            if (row, col) in occupied:
                continue
            cell = by_origin.get((row, col))
            if cell is None:
                html_rows.append("    <td></td>")
                occupied.add((row, col))
                continue
            tag = "th" if cell.is_header else "td"
            attrs = ""
            if cell.rowspan > 1:
                attrs += f' rowspan="{cell.rowspan}"'
            if cell.colspan > 1:
                attrs += f' colspan="{cell.colspan}"'
            html_rows.append(f"    <{tag}{attrs}>{escape(cell.text, quote=True)}</{tag}>")
            for r in range(row, row + cell.rowspan):
                for c in range(col, col + cell.colspan):
                    occupied.add((r, c))
        html_rows.append("  </tr>")
    html_rows.append("</table>")
    rendered_html = "\n".join(html_rows)

    first_row = [by_origin.get((0, col)) for col in range(cols)]
    simple_markdown_header = (
        all(cell is not None and cell.is_header for cell in first_row)
        and not any(cell.is_header and cell.row != 0 for cell in cells)
    )
    if any(cell.rowspan > 1 or cell.colspan > 1 for cell in cells) or not simple_markdown_header:
        markdown = rendered_html
    else:
        md_rows = []
        for row in range(rows):
            values = []
            for col in range(cols):
                cell = by_origin.get((row, col))
                text = "" if cell is None else escape(cell.text, quote=True).replace("|", "\\|").replace("\n", " ")
                values.append(text)
            md_rows.append("| " + " | ".join(values) + " |")
            if row == 0:
                md_rows.append("|" + "|".join("---" for _ in range(cols)) + "|")
        markdown = "\n".join(md_rows)
    return TableRender("valid", rows, cols, rendered_html, markdown, ())


def duplicate_preserving_key_values(
    pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep original display text and duplicate keys instead of overwriting."""
    result = []
    for pair in pairs:
        result.append(
            {
                "key": str(pair.get("key", "")),
                "value": str(pair.get("value", "")),
                "source_id": pair.get("source_id"),
                "target_id": pair.get("target_id"),
                "score": pair.get("score"),
                "page_id": pair.get("page_id"),
                "reading_order": pair.get("reading_order"),
            }
        )
    return result
