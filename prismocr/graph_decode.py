"""Pure deterministic decoding for typed, NULL-aware document relations.

Scores are directed ``relation -> source ID -> target ID -> score`` mappings;
NULL scores are ``relation -> source ID -> score``. Scores for a source and
relation must be comparable to each other and to its NULL score. They may be
logits or probabilities, but are not calibrated by this module.

Each source chooses at most one target per relation, including KV_VALUE. Thus
a key line links to at most one value, while distinct duplicate-key lines are
preserved and multiple keys may link to the same value. Cycle breaks drop the
edge with the smallest score margin over that source's NULL score. CONTAINS is
derived only by reversing accepted PARENT edges; supplied CONTAINS scores are
reported as suppressed rather than used independently.

Suppression codes are ``SELF_LINK``, ``INVALID_ENDPOINT_TYPES``,
``INVALID_SOURCE_TYPE``, ``INVALID_TARGET_TYPE``,
``READING_SCOPE_MISMATCH``, ``PARENT_GEOMETRY_MISSING``,
``PARENT_GEOMETRY_MISMATCH``, ``AUTHORITATIVE_TABLE_MEMBERSHIP``,
``TABLE_UNKNOWN``, ``CROSS_TABLE``,
``GRID_POSITION_MISSING``, ``ROW_MISMATCH``, ``NOT_LATER_COLUMN``,
``NOT_HORIZONTAL``, ``UNKNOWN_SOURCE``, ``UNKNOWN_TARGET``,
``MISSING_NULL_SCORE``, ``INVALID_SCORE``, ``NON_FINITE_SCORE``,
``NULL_PREFERRED``, ``LOWER_SCORE``, ``AMBIGUOUS_TIE``,
``NEXT_CYCLE_BROKEN``, ``PARENT_CYCLE_BROKEN``,
``CONTAINS_DERIVED_ONLY``, and ``UNSUPPORTED_RELATION``. Equal-score
alternatives carry ``AMBIGUOUS_TIE`` on both the chosen decision and rejected
alternatives; selected NULL decisions carry ``NULL_SELECTED``. Derived
containment decisions carry ``DERIVED_FROM_PARENT``.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal, TypeAlias

from prismocr.geometry import polygon_area, polygon_intersection

NodeType: TypeAlias = Literal["line", "region", "cell"]
Point: TypeAlias = tuple[float, float]
Polygon: TypeAlias = Sequence[Point]

# Keep this in the same channel order as prismocr.model.network.RELATION_NAMES
# without importing the model (and its torch dependency) into the decoder.
RELATION_NAMES = ("NEXT", "PARENT", "KV_VALUE", "RIGHT_CELL", "CONTAINS")


@dataclass(frozen=True)
class GraphNode:
    """Stable typed graph instance and optional geometry/table metadata.

    ``polygon`` uses shared page coordinates and should be convex because
    containment uses the production convex-polygon clipping primitive.
    ``scope_id`` separates independent reading-order streams such as columns.
    Region ``kind`` and line ``role`` use the v4 schema vocabulary.
    """

    id: str
    node_type: NodeType
    polygon: Polygon | None = None
    role: str | None = None
    kind: str | None = None
    scope_id: str | None = None
    table_id: str | None = None
    row: int | None = None
    col: int | None = None


@dataclass(frozen=True)
class RelationPrediction:
    """Selected relation decision; ``target_id=None`` represents explicit NULL."""

    relation: str
    source_id: str
    target_id: str | None
    score: float
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SuppressedRelation:
    """Candidate that was rejected, lost to another candidate, or was ambiguous."""

    relation: str
    source_id: str
    target_id: str | None
    score: float | None
    reason_code: str


@dataclass(frozen=True)
class DecodedGraph:
    """All input nodes, chosen relation decisions, line order, and audit reasons."""

    nodes: tuple[GraphNode, ...]
    predictions: tuple[RelationPrediction, ...]
    reading_order: tuple[str, ...]
    suppressed: tuple[SuppressedRelation, ...]
    suppressed_count: int
    suppressed_reason_counts: tuple[tuple[str, int], ...]


class _SuppressedCollector:
    """Count every rejected edge while retaining only a bounded diagnostic sample."""

    def __init__(self, limit: int | None) -> None:
        self.limit = limit
        self.items: list[SuppressedRelation] = []
        self.reason_counts: Counter[str] = Counter()
        self.count = 0

    def append(self, item: SuppressedRelation) -> None:
        self.count += 1
        self.reason_counts[item.reason_code] += 1
        if self.limit is None or len(self.items) < self.limit:
            self.items.append(item)

    def extend(self, items: Sequence[SuppressedRelation]) -> None:
        for item in items:
            self.append(item)


def _polygon_points(node: GraphNode) -> tuple[Point, ...] | None:
    polygon = node.polygon
    if (
        not isinstance(polygon, Sequence)
        or isinstance(polygon, (str, bytes))
        or len(polygon) < 3
    ):
        return None
    try:
        points_list: list[Point] = []
        for point in polygon:
            if (
                not isinstance(point, Sequence)
                or isinstance(point, (str, bytes))
                or len(point) != 2
            ):
                return None
            if any(
                isinstance(coordinate, bool) or not isinstance(coordinate, Real)
                for coordinate in point
            ):
                return None
            points_list.append((float(point[0]), float(point[1])))
        points = tuple(points_list)
    except (TypeError, ValueError, IndexError, KeyError, OverflowError):
        return None
    if any(not math.isfinite(value) for point in points for value in point):
        return None
    if polygon_area(points) <= 0.0:
        return None
    return points


def _parent_geometry_compatible(child: GraphNode, parent: GraphNode) -> bool | None:
    """Return None if geometry is unavailable, otherwise full child containment."""
    child_polygon = _polygon_points(child)
    parent_polygon = _polygon_points(parent)
    if child_polygon is None or parent_polygon is None:
        return None
    child_area = polygon_area(child_polygon)
    intersection_area = polygon_area(polygon_intersection(child_polygon, parent_polygon))
    return intersection_area >= child_area * (1.0 - 1e-6)


def _center_x(node: GraphNode) -> float | None:
    polygon = _polygon_points(node)
    if polygon is None:
        return None
    return sum(point[0] for point in polygon) / len(polygon)


def relation_candidate_error(
    relation: str,
    source: GraphNode,
    target: GraphNode,
    nodes_by_id: Mapping[str, GraphNode],
    *,
    allow_right_cell_column_repair: bool = False,
) -> str | None:
    if not isinstance(allow_right_cell_column_repair, bool):
        raise ValueError("allow_right_cell_column_repair must be a boolean")
    if source.id == target.id:
        return "SELF_LINK"

    if relation == "NEXT":
        if source.node_type != "line" or target.node_type != "line":
            return "INVALID_ENDPOINT_TYPES"
        if source.scope_id != target.scope_id:
            return "READING_SCOPE_MISMATCH"
        return None

    if relation == "PARENT":
        if source.node_type not in {"line", "cell", "region"} or target.node_type not in {"cell", "region"}:
            return "INVALID_ENDPOINT_TYPES"
        contained = _parent_geometry_compatible(source, target)
        if contained is None:
            return "PARENT_GEOMETRY_MISSING"
        if not contained:
            return "PARENT_GEOMETRY_MISMATCH"
        if source.node_type == "cell" and target.kind == "TABLE":
            authoritative_table_id = source.table_id
            authoritative_table = (
                nodes_by_id.get(authoritative_table_id)
                if isinstance(authoritative_table_id, str) and authoritative_table_id
                else None
            )
            if (
                authoritative_table is not None
                and authoritative_table.node_type == "region"
                and authoritative_table.kind == "TABLE"
                and target.id != authoritative_table_id
            ):
                return "AUTHORITATIVE_TABLE_MEMBERSHIP"
        return None

    if relation == "KV_VALUE":
        if source.node_type != "line" or source.role != "KEY":
            return "INVALID_SOURCE_TYPE"
        value_line = target.node_type == "line" and target.role == "VALUE"
        value_region = target.node_type == "region" and target.kind == "FORM_FIELD"
        if not (value_line or value_region):
            return "INVALID_TARGET_TYPE"
        return None

    if relation == "RIGHT_CELL":
        if source.node_type != "cell" or target.node_type != "cell":
            return "INVALID_ENDPOINT_TYPES"
        if (
            not isinstance(source.table_id, str) or not source.table_id
            or not isinstance(target.table_id, str) or not target.table_id
        ):
            return "TABLE_UNKNOWN"
        source_table = nodes_by_id.get(source.table_id)
        target_table = nodes_by_id.get(target.table_id)
        if (
            source_table is None or source_table.node_type != "region" or source_table.kind != "TABLE"
            or target_table is None or target_table.node_type != "region" or target_table.kind != "TABLE"
        ):
            return "TABLE_UNKNOWN"
        if source.table_id != target.table_id:
            return "CROSS_TABLE"
        if not all(_valid_grid_index(value) for value in (source.row, source.col, target.row, target.col)):
            return "GRID_POSITION_MISSING"
        if source.row != target.row:
            return "ROW_MISMATCH"
        source_x, target_x = _center_x(source), _center_x(target)
        if source_x is not None and target_x is not None and target_x <= source_x:
            return "NOT_HORIZONTAL"
        if target.col <= source.col:
            if not allow_right_cell_column_repair or source_x is None or target_x is None:
                return "NOT_LATER_COLUMN"
        return None

    return "UNSUPPORTED_RELATION"


def _valid_grid_index(value: int | None) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _score(value: object) -> tuple[float | None, str | None]:
    if isinstance(value, bool):
        return None, "INVALID_SCORE"
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None, "INVALID_SCORE"
    if not math.isfinite(result):
        return None, "NON_FINITE_SCORE"
    return result, None


def _validate_score_maps(
    relation_scores: Mapping[str, Mapping[str, Mapping[str, object]]],
    null_scores: Mapping[str, Mapping[str, object]],
) -> None:
    if not isinstance(relation_scores, Mapping) or not isinstance(null_scores, Mapping):
        raise ValueError("relation_scores and null_scores must be mappings")
    for relation, sources in relation_scores.items():
        if not isinstance(relation, str) or not isinstance(sources, Mapping):
            raise ValueError("relation_scores must map relation names to source mappings")
        for source, targets in sources.items():
            if not isinstance(source, str) or not isinstance(targets, Mapping):
                raise ValueError("relation candidate rows must map string source IDs to target mappings")
            if any(not isinstance(target, str) for target in targets):
                raise ValueError("relation candidate target IDs must be strings")
    for relation, sources in null_scores.items():
        if not isinstance(relation, str) or not isinstance(sources, Mapping):
            raise ValueError("null_scores must map relation names to source mappings")
        if any(not isinstance(source, str) for source in sources):
            raise ValueError("NULL score source IDs must be strings")


def _invalid_source(relation: str, source: GraphNode) -> bool:
    if relation == "NEXT":
        return source.node_type != "line"
    if relation == "PARENT":
        return source.node_type not in {"line", "cell", "region"}
    if relation == "KV_VALUE":
        return source.node_type != "line" or source.role != "KEY"
    if relation == "RIGHT_CELL":
        return source.node_type != "cell"
    return True


def _resolve_relation(
    relation: str,
    score_rows: Mapping[str, Mapping[str, object]],
    null_rows: Mapping[str, object],
    nodes_by_id: Mapping[str, GraphNode],
    suppressed: _SuppressedCollector,
    *,
    allow_right_cell_column_repair: bool = False,
) -> tuple[dict[str, RelationPrediction], dict[str, float]]:
    predictions: dict[str, RelationPrediction] = {}
    finite_nulls: dict[str, float] = {}

    for source_id in sorted(set(score_rows) | set(null_rows)):
        source = nodes_by_id.get(source_id)
        raw_candidates = score_rows.get(source_id, {})
        if source is None:
            for target_id, raw_score in sorted(raw_candidates.items()):
                score, score_error = _score(raw_score)
                suppressed.append(SuppressedRelation(
                    relation, source_id, target_id, score,
                    score_error or "UNKNOWN_SOURCE",
                ))
            if source_id in null_rows:
                null_score, score_error = _score(null_rows[source_id])
                suppressed.append(SuppressedRelation(
                    relation, source_id, None, null_score,
                    score_error or "UNKNOWN_SOURCE",
                ))
            continue

        null_score: float | None = None
        if source_id in null_rows:
            null_score, null_error = _score(null_rows[source_id])
            if null_error is not None:
                suppressed.append(SuppressedRelation(relation, source_id, None, None, null_error))
            else:
                assert null_score is not None
                finite_nulls[source_id] = null_score

        valid_candidates: list[tuple[str, float]] = []
        for target_id, raw_score in sorted(raw_candidates.items()):
            candidate_score, score_error = _score(raw_score)
            if score_error is not None:
                suppressed.append(SuppressedRelation(relation, source_id, target_id, None, score_error))
                continue
            assert candidate_score is not None
            target = nodes_by_id.get(target_id)
            if target is None:
                suppressed.append(SuppressedRelation(relation, source_id, target_id, candidate_score, "UNKNOWN_TARGET"))
                continue
            error = relation_candidate_error(
                relation,
                source,
                target,
                nodes_by_id,
                allow_right_cell_column_repair=allow_right_cell_column_repair,
            )
            if error is not None:
                suppressed.append(SuppressedRelation(relation, source_id, target_id, candidate_score, error))
                continue
            valid_candidates.append((target_id, candidate_score))

        if _invalid_source(relation, source):
            if source_id in null_rows:
                suppressed.append(SuppressedRelation(
                    relation, source_id, None, finite_nulls.get(source_id), "INVALID_SOURCE_TYPE"
                ))
            continue

        if source_id not in finite_nulls:
            for target_id, candidate_score in valid_candidates:
                suppressed.append(SuppressedRelation(
                    relation, source_id, target_id, candidate_score, "MISSING_NULL_SCORE"
                ))
            continue

        null_score = finite_nulls[source_id]
        if not valid_candidates:
            predictions[source_id] = RelationPrediction(
                relation, source_id, None, null_score, ("NULL_SELECTED",)
            )
            continue

        best_score = max(score for _, score in valid_candidates)
        best_targets = sorted(target for target, score in valid_candidates if score == best_score)
        if best_score <= null_score:
            tied_with_null = best_score == null_score
            reasons = ("NULL_SELECTED", "AMBIGUOUS_TIE") if tied_with_null else ("NULL_SELECTED",)
            predictions[source_id] = RelationPrediction(relation, source_id, None, null_score, reasons)
            for target_id, candidate_score in valid_candidates:
                reason = "AMBIGUOUS_TIE" if candidate_score == null_score else "NULL_PREFERRED"
                suppressed.append(SuppressedRelation(relation, source_id, target_id, candidate_score, reason))
            continue

        chosen_target = best_targets[0]
        chosen_reasons = ("AMBIGUOUS_TIE",) if len(best_targets) > 1 else ()
        predictions[source_id] = RelationPrediction(
            relation, source_id, chosen_target, best_score, chosen_reasons
        )
        for target_id, candidate_score in valid_candidates:
            if target_id == chosen_target:
                continue
            if candidate_score == best_score:
                reason = "AMBIGUOUS_TIE"
            elif candidate_score <= null_score:
                reason = "NULL_PREFERRED"
            else:
                reason = "LOWER_SCORE"
            suppressed.append(SuppressedRelation(relation, source_id, target_id, candidate_score, reason))

    return predictions, finite_nulls


def _find_cycle(edges: Mapping[str, RelationPrediction]) -> tuple[str, ...] | None:
    settled: set[str] = set()
    for start in sorted(edges):
        current: str | None = start
        path: list[str] = []
        position: dict[str, int] = {}
        while current is not None and current in edges and current not in settled:
            if current in position:
                return tuple(path[position[current] :])
            position[current] = len(path)
            path.append(current)
            current = edges[current].target_id
        settled.update(path)
    return None


def _break_cycles(
    relation: str,
    predictions: dict[str, RelationPrediction],
    null_scores: Mapping[str, float],
    reason_code: str,
) -> list[SuppressedRelation]:
    edges = {
        source: prediction
        for source, prediction in predictions.items()
        if prediction.target_id is not None
    }
    suppressed: list[SuppressedRelation] = []
    while (cycle := _find_cycle(edges)) is not None:
        weakest_source = min(
            cycle,
            key=lambda source: (
                edges[source].score - null_scores[source],
                edges[source].score,
                source,
                edges[source].target_id or "",
            ),
        )
        dropped = edges.pop(weakest_source)
        suppressed.append(SuppressedRelation(
            relation, weakest_source, dropped.target_id, dropped.score, reason_code
        ))
        original_reasons = tuple(code for code in dropped.reason_codes if code != reason_code)
        predictions[weakest_source] = RelationPrediction(
            relation,
            weakest_source,
            None,
            null_scores[weakest_source],
            (reason_code, *original_reasons),
        )
    return suppressed


def _geometry_key(node: GraphNode) -> tuple[float, float]:
    polygon = _polygon_points(node)
    if polygon is None:
        return math.inf, math.inf
    return (
        sum(point[1] for point in polygon) / len(polygon),
        sum(point[0] for point in polygon) / len(polygon),
    )


def _reading_order(
    nodes: Sequence[GraphNode], predictions: Mapping[str, RelationPrediction]
) -> tuple[str, ...]:
    lines = {node.id: node for node in nodes if node.node_type == "line"}
    indegree = {line_id: 0 for line_id in lines}
    adjacency: dict[str, list[str]] = {line_id: [] for line_id in lines}
    for source_id, prediction in predictions.items():
        target_id = prediction.target_id
        if prediction.relation != "NEXT" or target_id is None:
            continue
        if source_id in lines and target_id in lines:
            adjacency[source_id].append(target_id)
            indegree[target_id] += 1

    def key(line_id: str) -> tuple[float, float, str]:
        # A scope constrains NEXT candidates, but must not become a page-level
        # sort key: doing so moves headings and later document blocks around
        # entire columns or tables. Use geometry to merge independent streams.
        y, x = _geometry_key(lines[line_id])
        return y, x, line_id

    queue = [(key(line_id), line_id) for line_id in lines if indegree[line_id] == 0]
    heapq.heapify(queue)
    ordered: list[str] = []
    while queue:
        _, source_id = heapq.heappop(queue)
        ordered.append(source_id)
        for target_id in sorted(adjacency[source_id]):
            indegree[target_id] -= 1
            if indegree[target_id] == 0:
                heapq.heappush(queue, (key(target_id), target_id))

    # Cycle resolution above should make this unreachable. Retain every line if
    # malformed future changes violate that invariant, in stable geometric order.
    if len(ordered) != len(lines):
        ordered.extend(sorted(set(lines) - set(ordered), key=key))
    return tuple(ordered)


def decode_graph(
    nodes: Sequence[GraphNode],
    relation_scores: Mapping[str, Mapping[str, Mapping[str, object]]],
    null_scores: Mapping[str, Mapping[str, object]],
    *,
    suppressed_limit: int | None = None,
    allow_right_cell_column_repair: bool = False,
) -> DecodedGraph:
    """Decode typed relation scores without model or accelerator dependencies.

    Candidate maps use the relation channel order in :data:`RELATION_NAMES`.
    NEXT edges must share a scope; PARENT edges require full geometric
    containment. Cycles are made acyclic by dropping the edge with the
    smallest margin over its source's NULL score.
    RIGHT_CELL requires a known TABLE node, same row, increasing column, and
    increasing geometric x when both polygons are available. Column repair is
    available only to calibrated affinity placement and requires rightward geometry.
    Reading order is
    topologically sorted across independent scopes, using geometry then ID to
    merge otherwise unordered lines. Every line appears once.
    """
    _validate_score_maps(relation_scores, null_scores)
    if suppressed_limit is not None and (
        isinstance(suppressed_limit, bool) or not isinstance(suppressed_limit, int) or suppressed_limit < 0
    ):
        raise ValueError("suppressed_limit must be a nonnegative integer or None")
    if not isinstance(allow_right_cell_column_repair, bool):
        raise ValueError("allow_right_cell_column_repair must be a boolean")
    node_list = tuple(nodes)
    if any(not isinstance(node, GraphNode) for node in node_list):
        raise ValueError("nodes must contain GraphNode records")
    if any(not isinstance(node.id, str) or not node.id for node in node_list):
        raise ValueError("every graph node must have a nonempty string ID")
    if len({node.id for node in node_list}) != len(node_list):
        raise ValueError("graph node IDs must be unique")
    if any(node.scope_id is not None and not isinstance(node.scope_id, str) for node in node_list):
        raise ValueError("reading-order scope IDs must be strings or None")

    sorted_nodes = tuple(sorted(node_list, key=lambda node: node.id))
    nodes_by_id = {node.id: node for node in sorted_nodes}
    predictions: list[RelationPrediction] = []
    suppressed = _SuppressedCollector(suppressed_limit)
    resolved_by_relation: dict[str, dict[str, RelationPrediction]] = {}

    known_relations = set(RELATION_NAMES)
    unknown_relations = (set(relation_scores) | set(null_scores)) - known_relations
    for relation in sorted(unknown_relations):
        for source_id, targets in sorted(relation_scores.get(relation, {}).items()):
            for target_id, raw_score in sorted(targets.items()):
                score, _ = _score(raw_score)
                suppressed.append(SuppressedRelation(
                    relation, source_id, target_id, score, "UNSUPPORTED_RELATION"
                ))
        for source_id, raw_score in sorted(null_scores.get(relation, {}).items()):
            score, _ = _score(raw_score)
            suppressed.append(SuppressedRelation(
                relation, source_id, None, score, "UNSUPPORTED_RELATION"
            ))

    for relation in RELATION_NAMES:
        score_rows = relation_scores.get(relation, {})
        null_rows = null_scores.get(relation, {})
        if not isinstance(score_rows, Mapping) or not isinstance(null_rows, Mapping):
            raise ValueError(f"scores for {relation} must be mappings")
        if relation == "CONTAINS":
            for source_id, targets in sorted(score_rows.items()):
                for target_id, raw_score in sorted(targets.items()):
                    score, _ = _score(raw_score)
                    suppressed.append(SuppressedRelation(
                        relation, source_id, target_id, score, "CONTAINS_DERIVED_ONLY"
                    ))
            for source_id, raw_score in sorted(null_rows.items()):
                score, _ = _score(raw_score)
                suppressed.append(SuppressedRelation(
                    relation, source_id, None, score, "CONTAINS_DERIVED_ONLY"
                ))
            resolved_by_relation[relation] = {}
            continue

        chosen, finite_nulls = _resolve_relation(
            relation,
            score_rows,
            null_rows,
            nodes_by_id,
            suppressed,
            allow_right_cell_column_repair=(
                allow_right_cell_column_repair and relation == "RIGHT_CELL"
            ),
        )
        if relation == "NEXT":
            suppressed.extend(_break_cycles(relation, chosen, finite_nulls, "NEXT_CYCLE_BROKEN"))
        elif relation == "PARENT":
            suppressed.extend(_break_cycles(relation, chosen, finite_nulls, "PARENT_CYCLE_BROKEN"))
        resolved_by_relation[relation] = chosen
        predictions.extend(chosen.values())

    for parent_prediction in resolved_by_relation["PARENT"].values():
        if parent_prediction.target_id is None:
            continue
        parent_id, child_id = parent_prediction.target_id, parent_prediction.source_id
        predictions.append(RelationPrediction(
            "CONTAINS",
            parent_id,
            child_id,
            parent_prediction.score,
            ("DERIVED_FROM_PARENT", *parent_prediction.reason_codes),
        ))

    relation_order = {relation: index for index, relation in enumerate(RELATION_NAMES)}
    predictions.sort(key=lambda item: (
        relation_order[item.relation], item.source_id, item.target_id is None,
        item.target_id or "",
    ))
    suppressed_items = sorted(suppressed.items, key=lambda item: (
        relation_order.get(item.relation, len(relation_order)),
        item.relation,
        item.source_id,
        item.target_id is None,
        item.target_id or "",
        item.reason_code,
    ))
    next_choices = resolved_by_relation["NEXT"]
    return DecodedGraph(
        nodes=sorted_nodes,
        predictions=tuple(predictions),
        reading_order=_reading_order(sorted_nodes, next_choices),
        suppressed=tuple(suppressed_items),
        suppressed_count=suppressed.count,
        suppressed_reason_counts=tuple(sorted(suppressed.reason_counts.items())),
    )
