"""Validation for version 4.0 page annotations."""

from __future__ import annotations

import math
from collections import defaultdict
from numbers import Real
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from prismocr.geometry import polygon_is_simple, signed_polygon_area
from prismocr.graph_decode import GraphNode, relation_candidate_error

RELATION_TYPES = {"NEXT", "PARENT", "KV_VALUE", "RIGHT_CELL", "CONTAINS"}
REGION_KINDS = {"REGION", "TABLE", "LIST", "FORM_FIELD"}
TEXT_ROLES = {
    "TITLE", "H1", "H2", "PARA_LINE", "LIST_ITEM", "KEY", "VALUE",
    "CAPTION", "HEADER", "FOOTER", "FORMULA", "UNKNOWN",
}


@dataclass(frozen=True)
class AnnotationValidationError(ValueError):
    issues: tuple[str, ...]

    def __str__(self) -> str:
        return "annotation is invalid: " + "; ".join(self.issues)


def resolve_line_supervision(annotation: Mapping[str, Any]) -> str:
    """Resolve the shared annotation/target text-availability contract.

    Legacy nonempty lines retain the historical complete-label interpretation;
    legacy empty/missing lines are unknown, never certified blank. This function
    does not mutate annotations (including artifact-bound dataset fingerprints).
    """
    lines = annotation.get("lines", [])
    if not isinstance(lines, list):
        raise AnnotationValidationError(("lines must be an array",))
    if "supervision" not in annotation:
        return "complete" if lines else "unavailable"
    supervision = annotation["supervision"]
    if not isinstance(supervision, Mapping) or "lines" not in supervision:
        raise AnnotationValidationError(("supervision must be an object with lines availability",))
    state = supervision["lines"]
    if not isinstance(state, str) or state not in {"complete", "partial", "unavailable"}:
        raise AnnotationValidationError(("supervision.lines must be complete, partial, or unavailable",))
    if state == "unavailable" and lines:
        raise AnnotationValidationError(("supervision.lines unavailable conflicts with provided lines",))
    if state == "partial" and not lines:
        raise AnnotationValidationError(("supervision.lines partial requires provided positive lines",))
    if state == "complete" and "lines" not in annotation:
        raise AnnotationValidationError(("supervision.lines complete requires an explicit lines array",))
    return state


def _valid_polygon(value: Any, width: int, height: int, field: str, issues: list[str]) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        issues.append(f"{field} must contain at least three points")
        return
    points: list[tuple[float, float]] = []
    for i, point in enumerate(value):
        if (
            not isinstance(point, Sequence)
            or isinstance(point, (str, bytes))
            or len(point) != 2
        ):
            issues.append(f"{field}[{i}] must be an x,y pair")
            return
        if any(
            isinstance(coordinate, bool) or not isinstance(coordinate, Real)
            for coordinate in point
        ):
            issues.append(f"{field}[{i}] must contain numbers")
            return
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError, OverflowError):
            issues.append(f"{field}[{i}] must contain numbers")
            return
        if not math.isfinite(x) or not math.isfinite(y):
            issues.append(f"{field}[{i}] must be finite")
            return
        if x < 0 or y < 0 or x > width or y > height:
            issues.append(f"{field}[{i}] lies outside the image")
        points.append((x, y))
    if len(points) >= 3:
        area = signed_polygon_area(points)
        if abs(area) < 1e-6:
            issues.append(f"{field} has zero area")
        elif area < 0:
            issues.append(f"{field} winding must be positive in image coordinates")
        try:
            simple = polygon_is_simple(points)
        except ValueError as error:
            issues.append(f"{field} is not a valid polygon: {error}")
        else:
            if not simple:
                issues.append(f"{field} must be a simple non-self-intersecting polygon")


def validate_annotation(annotation: Mapping[str, Any]) -> None:
    """Raise one error containing all detectable schema and reference failures."""
    issues: list[str] = []
    if not isinstance(annotation, Mapping):
        raise AnnotationValidationError(("top-level value must be an object",))
    try:
        resolve_line_supervision(annotation)
    except AnnotationValidationError as error:
        issues.extend(error.issues)
    if annotation.get("schema_version") != "4.0":
        issues.append("schema_version must be '4.0'")
    if not isinstance(annotation.get("document_id"), str) or not annotation.get("document_id"):
        issues.append("document_id is required")
    if not isinstance(annotation.get("page_id"), str) or not annotation.get("page_id"):
        issues.append("page_id is required")

    image = annotation.get("image")
    if not isinstance(image, Mapping):
        issues.append("image must be an object")
        width = height = 0
    else:
        width, height = image.get("width"), image.get("height")
        if not isinstance(width, int) or width <= 0:
            issues.append("image.width must be a positive integer")
            width = 0
        if not isinstance(height, int) or height <= 0:
            issues.append("image.height must be a positive integer")
            height = 0
        digest = image.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            issues.append("image.sha256 must be a 64-character hex digest")
        else:
            try:
                int(digest, 16)
            except ValueError:
                issues.append("image.sha256 must be hexadecimal")

    provenance = annotation.get("provenance")
    if not isinstance(provenance, Mapping):
        issues.append("provenance must be an object")
    else:
        for field in ("source_class", "license", "permission_reference", "reviewer"):
            value = provenance.get(field)
            if not isinstance(value, str) or not value.strip():
                issues.append(f"provenance.{field} is required")
        source_class = provenance.get("source_class")
        if not isinstance(source_class, str) or source_class not in {"real", "synthetic", "derived"}:
            issues.append("provenance.source_class must be real, synthetic, or derived")
        if "template_family_id" in provenance and (
            not isinstance(provenance["template_family_id"], str)
            or not provenance["template_family_id"].strip()
        ):
            issues.append("provenance.template_family_id must be a non-empty string when supplied")

    records: dict[str, tuple[str, Mapping[str, Any]]] = {}
    line_ids: list[str] = []
    table_ids: set[str] = set()
    ambiguous_table_ids: set[str] = set()
    cells_by_table: dict[str, list[tuple[str, int, int, int, int]]] = defaultdict(list)
    cell_table_references: list[tuple[str, str]] = []

    def inspect_items(field: str, kind: str) -> list[Mapping[str, Any]]:
        items = annotation.get(field, [])
        if not isinstance(items, list):
            issues.append(f"{field} must be an array")
            return []
        for index, item in enumerate(items):
            prefix = f"{field}[{index}]"
            if not isinstance(item, Mapping):
                issues.append(f"{prefix} must be an object")
                continue
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id:
                issues.append(f"{prefix}.id is required")
                continue
            if item_id in records:
                issues.append(f"duplicate instance id {item_id!r}")
            else:
                records[item_id] = (kind, item)
            _valid_polygon(item.get("polygon_px"), width, height, f"{prefix}.polygon_px", issues)
            if kind == "line":
                line_ids.append(item_id)
                if not isinstance(item.get("text"), str):
                    issues.append(f"{prefix}.text must be a string")
                if not isinstance(item.get("role"), str) or item.get("role") not in TEXT_ROLES:
                    issues.append(f"{prefix}.role is not a supported text role")
                if "scope_id" in item and (
                    not isinstance(item["scope_id"], str) or not item["scope_id"].strip()
                ):
                    issues.append(f"{prefix}.scope_id must be a non-empty string when supplied")
            elif kind == "region":
                if not isinstance(item.get("kind"), str) or item.get("kind") not in REGION_KINDS:
                    issues.append(f"{prefix}.kind is not a supported region kind")
                if "annotation_ambiguous" in item and not isinstance(item["annotation_ambiguous"], bool):
                    issues.append(f"{prefix}.annotation_ambiguous must be boolean")
                if item.get("kind") == "TABLE":
                    table_ids.add(item_id)
                    if item.get("annotation_ambiguous") is True:
                        ambiguous_table_ids.add(item_id)
            elif kind == "cell":
                table_id = item.get("table_id")
                row, col = item.get("row"), item.get("col")
                rowspan, colspan = item.get("rowspan", 1), item.get("colspan", 1)
                if "table_id" not in item:
                    issues.append(f"{prefix}.table_id is required")
                elif table_id is not None:
                    if not isinstance(table_id, str) or not table_id:
                        issues.append(f"{prefix}.table_id must be a non-empty string or null")
                    else:
                        cell_table_references.append((item_id, table_id))
                if row is not None and (
                    isinstance(row, bool) or not isinstance(row, int) or row < 0
                ):
                    issues.append(f"{prefix}.row must be a nonnegative integer when supplied")
                if col is not None and (
                    isinstance(col, bool) or not isinstance(col, int) or col < 0
                ):
                    issues.append(f"{prefix}.col must be a nonnegative integer when supplied")
                if rowspan is not None and (
                    isinstance(rowspan, bool) or not isinstance(rowspan, int) or rowspan <= 0
                ):
                    issues.append(f"{prefix}.rowspan must be a positive integer when supplied")
                if colspan is not None and (
                    isinstance(colspan, bool) or not isinstance(colspan, int) or colspan <= 0
                ):
                    issues.append(f"{prefix}.colspan must be a positive integer when supplied")
                if (
                    isinstance(row, int) and not isinstance(row, bool) and row >= 0
                    and isinstance(col, int) and not isinstance(col, bool) and col >= 0
                    and (rowspan is None or (isinstance(rowspan, int) and not isinstance(rowspan, bool) and rowspan > 0))
                    and (colspan is None or (isinstance(colspan, int) and not isinstance(colspan, bool) and colspan > 0))
                    and isinstance(table_id, str)
                ):
                    cells_by_table[table_id].append((
                        item_id,
                        row,
                        col,
                        1 if rowspan is None else rowspan,
                        1 if colspan is None else colspan,
                    ))
                if "is_header" in item and not isinstance(item["is_header"], bool):
                    issues.append(f"{prefix}.is_header must be boolean")
        return [item for item in items if isinstance(item, Mapping)]

    inspect_items("lines", "line")
    inspect_items("regions", "region")
    inspect_items("cells", "cell")

    graph_nodes: dict[str, GraphNode] = {}
    for item_id, (node_type, item) in records.items():
        polygon = item.get("polygon_px")
        graph_nodes[item_id] = GraphNode(
            id=item_id,
            node_type=node_type,
            polygon=(
                polygon
                if isinstance(polygon, Sequence) and not isinstance(polygon, (str, bytes))
                else None
            ),
            role=item.get("role") if node_type == "line" else None,
            kind=item.get("kind") if node_type == "region" else None,
            scope_id=item.get("scope_id", "document") if node_type == "line" else None,
            table_id=item.get("table_id") if node_type == "cell" else None,
            row=item.get("row") if node_type == "cell" else None,
            col=item.get("col") if node_type == "cell" else None,
        )

    for _, table_id in cell_table_references:
        if table_id not in table_ids:
            issues.append(f"cells refer to missing TABLE region {table_id!r}")

    for table_id, cells in cells_by_table.items():
        if table_id in ambiguous_table_ids:
            continue
        for index, first in enumerate(cells):
            first_id, first_row, first_col, first_rowspan, first_colspan = first
            for second_index in range(index + 1, len(cells)):
                second_id, second_row, second_col, second_rowspan, second_colspan = cells[second_index]
                overlap_row = max(first_row, second_row)
                overlap_col = max(first_col, second_col)
                if (
                    overlap_row < min(first_row + first_rowspan, second_row + second_rowspan)
                    and overlap_col < min(first_col + first_colspan, second_col + second_colspan)
                ):
                    issues.append(
                        f"cells {first_id!r} and {second_id!r} overlap at grid slot "
                        f"({overlap_row},{overlap_col})"
                    )

    relations = annotation.get("relations", [])
    explicit_next_decisions: dict[str, str | None] = {}
    if not isinstance(relations, list):
        issues.append("relations must be an array")
    else:
        seen_edges: set[tuple[str, str, str | None]] = set()
        outgoing_next: set[str] = set()
        parent_decisions: dict[str, tuple[str | None, str]] = {}
        outgoing_kv_value: set[str] = set()
        outgoing_right_cell: set[str] = set()
        graph_by_type: dict[str, dict[str, list[str]]] = {
            graph_type: defaultdict(list)
            for graph_type in ("NEXT", "PARENT", "CONTAINS", "HIERARCHY")
        }
        def register_parent_decision(
            child_id: str, parent_id: str | None, relation_type: str, index: int
        ) -> None:
            previous = parent_decisions.get(child_id)
            if previous is None:
                parent_decisions[child_id] = (parent_id, relation_type)
                return
            previous_parent_id, previous_relation_type = previous
            equivalent_inverse_pair = (
                previous_parent_id == parent_id
                and {previous_relation_type, relation_type} == {"PARENT", "CONTAINS"}
            )
            if not equivalent_inverse_pair:
                issues.append(
                    f"relations[{index}] child {child_id!r} has multiple PARENT decisions"
                )

        for i, relation in enumerate(relations):
            if not isinstance(relation, Mapping):
                issues.append(f"relations[{i}] must be an object")
                continue
            rel_type = relation.get("type")
            source, target = relation.get("source"), relation.get("target")
            if not isinstance(rel_type, str) or rel_type not in RELATION_TYPES:
                issues.append(f"relations[{i}].type is unsupported")
                continue
            if not isinstance(source, str) or source not in records:
                issues.append(f"relations[{i}].source refers to a missing instance")
                continue
            if target is not None and (not isinstance(target, str) or target not in records):
                issues.append(f"relations[{i}].target refers to a missing instance")
                continue
            edge = (rel_type, source, target)
            if edge in seen_edges:
                issues.append(f"relations[{i}] duplicates edge {edge!r}")
            seen_edges.add(edge)
            if target is None:
                if relation.get("annotation_status") != "explicit_null":
                    issues.append(f"relations[{i}] NULL target must be marked explicit_null")
                source_kind, source_item = records[source]
                if rel_type == "NEXT":
                    if source in outgoing_next:
                        issues.append(f"relations[{i}] line {source!r} has multiple outgoing NEXT edges")
                    outgoing_next.add(source)
                if rel_type == "NEXT" and source_kind != "line":
                    issues.append(f"relations[{i}] NEXT source must be a line")
                elif rel_type == "NEXT":
                    explicit_next_decisions[source] = None
                elif rel_type == "PARENT":
                    if source_kind not in {"line", "cell", "region"}:
                        issues.append(f"relations[{i}] PARENT source must be a line/cell/region")
                    register_parent_decision(source, None, rel_type, i)
                elif rel_type == "KV_VALUE":
                    source_role = source_item.get("role") if source_kind == "line" else None
                    if source_role != "KEY":
                        issues.append(f"relations[{i}] KV_VALUE source must be a KEY line")
                    if source in outgoing_kv_value:
                        issues.append(f"relations[{i}] KEY line {source!r} has multiple outgoing KV_VALUE edges")
                    outgoing_kv_value.add(source)
                elif rel_type == "RIGHT_CELL":
                    if source_kind != "cell":
                        issues.append(f"relations[{i}] RIGHT_CELL source must be a cell")
                    if source in outgoing_right_cell:
                        issues.append(f"relations[{i}] cell {source!r} has multiple outgoing RIGHT_CELL decisions")
                    outgoing_right_cell.add(source)
                elif rel_type == "CONTAINS":
                    if source_kind not in {"cell", "region"}:
                        issues.append(f"relations[{i}] CONTAINS source must be a cell or region")
                    issues.append(
                        f"relations[{i}] CONTAINS cannot have a NULL target; use a PARENT NULL decision for a child"
                    )
                continue
            source_kind, source_item = records[source]
            target_kind, target_item = records[target]
            if source == target:
                issues.append(f"relations[{i}] cannot point to itself")

            if rel_type == "NEXT":
                if source_kind != "line" or target_kind != "line":
                    issues.append(f"relations[{i}] NEXT endpoints must both be lines")
                if source in outgoing_next:
                    issues.append(f"relations[{i}] line {source!r} has multiple outgoing NEXT edges")
                outgoing_next.add(source)
                if source_kind == "line" and target_kind == "line":
                    if source_item.get("scope_id", "document") != target_item.get("scope_id", "document"):
                        issues.append(f"relations[{i}] NEXT endpoints must share a reading-order scope")
                    else:
                        explicit_next_decisions[source] = target
                graph_by_type["NEXT"][source].append(target)
            elif rel_type == "PARENT":
                valid_parent = (
                    source_kind in {"line", "cell", "region"}
                    and target_kind in {"cell", "region"}
                )
                register_parent_decision(source, target, rel_type, i)
                if not valid_parent:
                    issues.append(f"relations[{i}] PARENT must link a line/cell/region to a cell/region")
                else:
                    decoder_error = relation_candidate_error(
                        "PARENT", graph_nodes[source], graph_nodes[target], graph_nodes
                    )
                    if decoder_error is not None:
                        issues.append(
                            f"relations[{i}] PARENT is not decoder-compatible: {decoder_error}"
                        )
                    graph_by_type["PARENT"][source].append(target)
                    graph_by_type["HIERARCHY"][source].append(target)
            elif rel_type == "KV_VALUE":
                source_role = source_item.get("role") if source_kind == "line" else None
                target_role = target_item.get("role") if target_kind == "line" else None
                value_region = target_kind == "region" and target_item.get("kind") == "FORM_FIELD"
                if source_role != "KEY" or (target_role != "VALUE" and not value_region):
                    issues.append(f"relations[{i}] KV_VALUE must link a KEY line to a VALUE line or FORM_FIELD region")
                if source in outgoing_kv_value:
                    issues.append(f"relations[{i}] KEY line {source!r} has multiple outgoing KV_VALUE edges")
                outgoing_kv_value.add(source)
            elif rel_type == "RIGHT_CELL":
                if source in outgoing_right_cell:
                    issues.append(f"relations[{i}] cell {source!r} has multiple outgoing RIGHT_CELL decisions")
                outgoing_right_cell.add(source)
                if source_kind != "cell" or target_kind != "cell":
                    issues.append(f"relations[{i}] RIGHT_CELL endpoints must both be cells")
                else:
                    decoder_error = relation_candidate_error(
                        "RIGHT_CELL", graph_nodes[source], graph_nodes[target], graph_nodes
                    )
                    if decoder_error is not None:
                        issues.append(
                            f"relations[{i}] RIGHT_CELL is not decoder-compatible: {decoder_error}"
                        )
            elif rel_type == "CONTAINS":
                valid_contains = (
                    source_kind in {"cell", "region"}
                    and target_kind in {"line", "cell", "region"}
                )
                if not valid_contains:
                    issues.append(f"relations[{i}] CONTAINS must link a cell/region to a line/cell/region")
                else:
                    decoder_error = relation_candidate_error(
                        "PARENT", graph_nodes[target], graph_nodes[source], graph_nodes
                    )
                    if decoder_error is not None:
                        issues.append(
                            f"relations[{i}] CONTAINS is not decoder-compatible: {decoder_error}"
                        )
                    register_parent_decision(target, source, rel_type, i)
                    graph_by_type["CONTAINS"][source].append(target)
                    # CONTAINS is normalized to the child-to-parent direction
                    # used by PARENT decoding before cycle validation.
                    graph_by_type["HIERARCHY"][target].append(source)

        for graph_type, graph in graph_by_type.items():
            state: dict[str, int] = {}
            for start in graph:
                if state.get(start, 0):
                    continue
                state[start] = 1
                stack = [(start, iter(graph.get(start, ())))]
                while stack:
                    node, successors = stack[-1]
                    try:
                        successor = next(successors)
                    except StopIteration:
                        state[node] = 2
                        stack.pop()
                        continue
                    successor_state = state.get(successor, 0)
                    if successor_state == 1:
                        issues.append(
                            f"{graph_type} graph contains a cycle through {successor!r}"
                        )
                    elif successor_state == 0:
                        state[successor] = 1
                        stack.append((successor, iter(graph.get(successor, ()))))

    order = annotation.get("reading_order")
    if not isinstance(order, list) or any(not isinstance(i, str) for i in order):
        issues.append("reading_order must be an array of line IDs")
    else:
        line_id_set = set(line_ids)
        if len(order) != len(set(order)):
            issues.append("reading_order contains duplicate line IDs")
        if any(i not in line_id_set for i in order):
            issues.append("reading_order contains a non-line or missing ID")
        if set(order) != line_id_set:
            issues.append("reading_order must include every line exactly once")
        if (
            len(order) == len(line_id_set)
            and len(order) == len(set(order))
            and set(order) == line_id_set
        ):
            ordered_by_scope: dict[str, list[str]] = defaultdict(list)
            for line_id in order:
                line = records.get(line_id)
                if line is not None and line[0] == "line":
                    scope_id = line[1].get("scope_id", "document")
                    if isinstance(scope_id, str) and scope_id.strip():
                        ordered_by_scope[scope_id].append(line_id)
            expected_next: dict[str, str | None] = {}
            for ordered_ids in ordered_by_scope.values():
                for index, line_id in enumerate(ordered_ids):
                    expected_next[line_id] = (
                        ordered_ids[index + 1] if index + 1 < len(ordered_ids) else None
                    )
            for source_id, target_id in explicit_next_decisions.items():
                if source_id in expected_next and target_id != expected_next[source_id]:
                    issues.append(
                        f"NEXT decision for {source_id!r} contradicts reading_order"
                    )

    if issues:
        raise AnnotationValidationError(tuple(issues))