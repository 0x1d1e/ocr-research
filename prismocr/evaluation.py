"""Unicode-aware text and assignment-based detection evaluation helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import unicodedata
from typing import Any

from prismocr.data.geometry import polygon_to_box
from prismocr.data.schema import REGION_KINDS, resolve_line_supervision
from prismocr.geometry import source_polygon_iou
from prismocr.matching import MatchSet, match_proposals


def edit_distance(reference: str, hypothesis: str) -> int:
    """Levenshtein distance over Unicode code points without implicit normalization."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, 1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def _sequence_edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, 1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def word_error_count(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return word edit errors and reference-word count using whitespace tokenization."""
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    return _sequence_edit_distance(reference_words, hypothesis_words), len(reference_words)


def word_error_rate(reference: str, hypothesis: str) -> float:
    errors, reference_count = word_error_count(reference, hypothesis)
    if not reference_count:
        return 0.0 if not hypothesis.split() else 1.0
    return errors / reference_count


@dataclass(frozen=True)
class DetectionCounts:
    matched: int
    assigned: int
    ground_truth: int
    valid_candidates: int
    unassigned: int
    missed: int

    @property
    def recall(self) -> float:
        return self.matched / self.ground_truth if self.ground_truth else 1.0

    @property
    def assignment_coverage(self) -> float:
        return self.assigned / self.ground_truth if self.ground_truth else 1.0


def match_detections(
    predicted_boxes: Sequence[Sequence[float]],
    ground_truth_boxes: Sequence[Sequence[float]],
    image_size: tuple[int, int],
    *,
    iou_threshold: float = 0.5,
) -> tuple[MatchSet, DetectionCounts]:
    """Report threshold-accepted recall separately from assignment coverage."""
    result = match_proposals(
        predicted_boxes,
        ground_truth_boxes,
        image_size=image_size,
        iou_threshold=iou_threshold,
    )
    counts = DetectionCounts(
        matched=len(result.accepted_pairs),
        assigned=len(result.assigned_gt_idx),
        ground_truth=result.gt_count,
        valid_candidates=result.valid_pred_count,
        unassigned=len(result.unassigned_gt_idx),
        missed=len(result.missed_gt_idx),
    )
    return result, counts


def aggregate_cer(pairs: Sequence[tuple[str, str]]) -> float:
    denominator = sum(len(reference) for reference, _ in pairs)
    if denominator == 0:
        return 0.0 if not any(hypothesis for _, hypothesis in pairs) else 1.0
    return sum(edit_distance(reference, hypothesis) for reference, hypothesis in pairs) / denominator


def _polygon(item: Mapping[str, Any], *, predicted: bool) -> Any:
    if not predicted:
        return item.get("polygon_px")
    return item.get("polygon") or item.get("polygon_px")


def _match_instances(
    ground_truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    image_size: tuple[int, int],
    iou_threshold: float,
) -> tuple[dict[str, int], dict[str, str], dict[str, str], list[tuple[Mapping[str, Any], Mapping[str, Any]]]]:
    """Match one typed instance stream and retain only IoU-accepted ID pairs."""
    gt_polygons = [_polygon(item, predicted=False) for item in ground_truth]
    pred_polygons = [_polygon(item, predicted=True) for item in predictions]
    valid_gt = [index for index, polygon in enumerate(gt_polygons) if polygon is not None]
    valid_pred = [index for index, polygon in enumerate(pred_polygons) if polygon is not None]
    gt_valid_polygons = [gt_polygons[index] for index in valid_gt]
    pred_valid_polygons = [pred_polygons[index] for index in valid_pred]
    gt_boxes = [polygon_to_box(polygon, image_size) for polygon in gt_valid_polygons]
    pred_boxes = [polygon_to_box(polygon, image_size) for polygon in pred_valid_polygons]
    polygon_quality = [
        [source_polygon_iou(gt_polygon, pred_polygon) for pred_polygon in pred_valid_polygons]
        for gt_polygon in gt_valid_polygons
    ]
    match = match_proposals(
        pred_boxes,
        gt_boxes,
        image_size=image_size,
        iou_threshold=iou_threshold,
        quality_matrix=polygon_quality,
    )
    gt_to_pred: dict[str, str] = {}
    pred_to_gt: dict[str, str] = {}
    matched: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for gt_index, pred_index in match.accepted_pairs:
        gt_item = ground_truth[valid_gt[gt_index]]
        pred_item = predictions[valid_pred[pred_index]]
        gt_id, pred_id = gt_item.get("id"), pred_item.get("id")
        if isinstance(gt_id, str) and isinstance(pred_id, str):
            gt_to_pred[gt_id] = pred_id
            pred_to_gt[pred_id] = gt_id
            matched.append((gt_item, pred_item))
    # Detection is a geometric decision. Missing IDs only prevent downstream
    # relation matching; they must not erase an otherwise accepted detection.
    true_positive = len(match.accepted_pairs)
    return (
        {
            "true_positive": true_positive,
            "false_positive": len(predictions) - true_positive,
            "false_negative": len(ground_truth) - true_positive,
        },
        gt_to_pred,
        pred_to_gt,
        matched,
    )


def _khmer_mark_contexts(text: str, *, coeng_only: bool = False) -> list[tuple[int, int, str]]:
    """Represent each Khmer mark with its word and preceding-base position."""
    contexts: list[tuple[int, int, str]] = []
    word_index = -1
    base_index = 0
    in_word = False
    for character in text:
        if character.isspace():
            in_word = False
            base_index = 0
            continue
        if not in_word:
            word_index += 1
            in_word = True
        codepoint = ord(character)
        is_mark = 0x17B6 <= codepoint <= 0x17D3 or codepoint == 0x17DD
        if is_mark:
            if not coeng_only or character == "\u17d2":
                contexts.append((word_index, base_index, character))
        else:
            base_index += 1
    return contexts


def _language_bucket(text: str, annotation_language: Any) -> str:
    if isinstance(annotation_language, str) and annotation_language.strip():
        return annotation_language.strip().lower()
    has_khmer = any(0x1780 <= ord(char) <= 0x17FF for char in text)
    has_latin = any(char.isascii() and char.isalpha() for char in text)
    if has_khmer and has_latin:
        return "mixed"
    if has_khmer:
        return "khmer"
    if has_latin:
        return "english"
    return "other"


_Tree = tuple[str, tuple["_Tree", ...]]
_MAX_TEDS_SLOTS = 100_000
_MAX_TEDS_NODES = 2_048


def _normalise_value(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _table_cell_text(
    cell: Mapping[str, Any],
    lines: Mapping[str, Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
    order: Mapping[str, int],
) -> str:
    direct_text = cell.get("text")
    if isinstance(direct_text, str):
        return direct_text
    cell_id = cell.get("id")
    line_ids = {
        line_id for line_id in cell.get("line_ids", ()) if isinstance(line_id, str)
    }
    for edge in relations:
        if edge.get("type") == "PARENT" and edge.get("target") == cell_id:
            source = edge.get("source")
            if isinstance(source, str) and source in lines:
                line_ids.add(source)
        elif edge.get("type") == "CONTAINS" and edge.get("source") == cell_id:
            target = edge.get("target")
            if isinstance(target, str) and target in lines:
                line_ids.add(target)
    ordered = sorted(line_ids, key=lambda item: (order.get(item, len(order)), item))
    return "\n".join(str(lines[line_id].get("text", "")) for line_id in ordered)


def _node_text(
    instance_id: str,
    records: Mapping[str, Mapping[str, Any]],
    lines: Mapping[str, Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
    order: Mapping[str, int],
) -> str:
    item = records.get(instance_id)
    if item is None:
        return ""
    text = item.get("text")
    if isinstance(text, str):
        return text
    if instance_id in lines:
        return str(lines[instance_id].get("text", ""))
    children = set(item.get("line_ids", ()))
    for edge in relations:
        if edge.get("type") == "PARENT" and edge.get("target") == instance_id:
            source = edge.get("source")
            if isinstance(source, str) and source in lines:
                children.add(source)
        elif edge.get("type") == "CONTAINS" and edge.get("source") == instance_id:
            target = edge.get("target")
            if isinstance(target, str) and target in lines:
                children.add(target)
    ordered = sorted(
        (child for child in children if isinstance(child, str) and child in lines),
        key=lambda child: (order.get(child, len(order)), child),
    )
    return "\n".join(str(lines[child].get("text", "")) for child in ordered)


def _table_tree(
    table_id: str,
    cells: Sequence[Mapping[str, Any]],
    table: Mapping[str, Any] | None,
    lines: Mapping[str, Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
    order: Mapping[str, int],
    *,
    include_text: bool,
) -> _Tree | None:
    """Build a deterministic HTML-like tree; None means GT grid labels are incomplete."""
    if table is not None and table.get("status") == "conflict":
        return None
    selected = [cell for cell in cells if cell.get("table_id") == table_id]
    origins: dict[tuple[int, int], tuple[Mapping[str, Any], int, int]] = {}
    inferred_rows = inferred_cols = 0
    for cell in selected:
        row, col = cell.get("row"), cell.get("col")
        if isinstance(row, bool) or not isinstance(row, int) or row < 0:
            return None
        if isinstance(col, bool) or not isinstance(col, int) or col < 0:
            return None
        rowspan = cell.get("rowspan", 1)
        colspan = cell.get("colspan", 1)
        rowspan = 1 if rowspan is None else rowspan
        colspan = 1 if colspan is None else colspan
        if (
            isinstance(rowspan, bool) or not isinstance(rowspan, int) or rowspan <= 0
            or isinstance(colspan, bool) or not isinstance(colspan, int) or colspan <= 0
        ):
            return None
        if (row, col) in origins:
            return None
        origins[(row, col)] = (cell, rowspan, colspan)
        inferred_rows = max(inferred_rows, row + rowspan)
        inferred_cols = max(inferred_cols, col + colspan)

    rows = inferred_rows
    cols = inferred_cols
    if table is not None:
        for field, inferred in (("rows", inferred_rows), ("cols", inferred_cols)):
            declared = table.get(field)
            if declared is None:
                continue
            if isinstance(declared, bool) or not isinstance(declared, int) or declared < inferred:
                return None
            if field == "rows":
                rows = declared
            else:
                cols = declared
    if rows * cols > _MAX_TEDS_SLOTS:
        return None

    occupied: set[tuple[int, int]] = set()
    for (row, col), (_, rowspan, colspan) in origins.items():
        if row + rowspan > rows or col + colspan > cols:
            return None
        slots = {
            (covered_row, covered_col)
            for covered_row in range(row, row + rowspan)
            for covered_col in range(col, col + colspan)
        }
        if occupied & slots:
            return None
        occupied.update(slots)
    occupied.clear()
    row_nodes: list[_Tree] = []
    for row in range(rows):
        cell_nodes: list[_Tree] = []
        for col in range(cols):
            if (row, col) in occupied:
                continue
            item = origins.get((row, col))
            if item is None:
                cell_nodes.append(("td|rowspan=1|colspan=1", ()))
                occupied.add((row, col))
                continue
            cell, rowspan, colspan = item
            header = cell.get("is_header") is True
            tag = "th" if header else "td"
            children: tuple[_Tree, ...] = ()
            if include_text:
                text = _table_cell_text(cell, lines, relations, order)
                if text:
                    children = ((f"#text:{text}", ()),)
            cell_nodes.append((f"{tag}|rowspan={rowspan}|colspan={colspan}", children))
            for covered_row in range(row, row + rowspan):
                for covered_col in range(col, col + colspan):
                    occupied.add((covered_row, covered_col))
        row_nodes.append(("tr", tuple(cell_nodes)))
    tree = ("table", tuple(row_nodes))
    return tree if _tree_size(tree) <= _MAX_TEDS_NODES else None


def _tree_size(tree: _Tree) -> int:
    return 1 + sum(_tree_size(child) for child in tree[1])


def _text_substitution_cost(first: str, second: str) -> float:
    if first == second:
        return 0.0
    first_text, second_text = first.removeprefix("#text:"), second.removeprefix("#text:")
    denominator = max(len(first_text), len(second_text))
    return edit_distance(first_text, second_text) / denominator if denominator else 0.0


def _tree_edit_distance(first: _Tree, second: _Tree) -> float:
    """Zhang-Shasha ordered tree edit distance with normalized text substitutions."""
    def postorder(root: _Tree) -> tuple[list[str], list[int], dict[int, int]]:
        labels = [""]
        leftmost = [0]
        keyroots_by_leftmost: dict[int, int] = {}

        def visit(node: _Tree) -> tuple[int, int]:
            child_indices = [visit(child) for child in node[1]]
            index = len(labels)
            labels.append(node[0])
            lmd = child_indices[0][0] if child_indices else index
            leftmost.append(lmd)
            keyroots_by_leftmost[lmd] = index
            return lmd, index

        visit(root)
        return labels, leftmost, keyroots_by_leftmost

    first_labels, first_leftmost, first_keyroots = postorder(first)
    second_labels, second_leftmost, second_keyroots = postorder(second)
    first_count, second_count = len(first_labels) - 1, len(second_labels) - 1
    tree_distance = [[0.0] * (second_count + 1) for _ in range(first_count + 1)]

    def substitute_cost(first_label: str, second_label: str) -> float:
        if first_label.startswith("#text:") and second_label.startswith("#text:"):
            return _text_substitution_cost(first_label, second_label)
        return 0.0 if first_label == second_label else 1.0

    for first_root in sorted(first_keyroots.values()):
        first_start = first_leftmost[first_root]
        for second_root in sorted(second_keyroots.values()):
            second_start = second_leftmost[second_root]
            first_offset, second_offset = first_start - 1, second_start - 1
            forest = [
                [0.0] * (second_root - second_offset + 1)
                for _ in range(first_root - first_offset + 1)
            ]

            def forest_get(first_index: int, second_index: int) -> float:
                return forest[first_index - first_offset][second_index - second_offset]

            def forest_set(first_index: int, second_index: int, value: float) -> None:
                forest[first_index - first_offset][second_index - second_offset] = value

            for first_index in range(first_start, first_root + 1):
                forest_set(
                    first_index, second_start - 1,
                    forest_get(first_index - 1, second_start - 1) + 1.0,
                )
            for second_index in range(second_start, second_root + 1):
                forest_set(
                    first_start - 1, second_index,
                    forest_get(first_start - 1, second_index - 1) + 1.0,
                )
            for first_index in range(first_start, first_root + 1):
                for second_index in range(second_start, second_root + 1):
                    delete_cost = forest_get(first_index - 1, second_index) + 1.0
                    insert_cost = forest_get(first_index, second_index - 1) + 1.0
                    if (
                        first_leftmost[first_index] == first_start
                        and second_leftmost[second_index] == second_start
                    ):
                        replace_cost = (
                            forest_get(first_index - 1, second_index - 1)
                            + substitute_cost(first_labels[first_index], second_labels[second_index])
                        )
                        value = min(delete_cost, insert_cost, replace_cost)
                        forest_set(first_index, second_index, value)
                        tree_distance[first_index][second_index] = value
                    else:
                        previous_forest = forest_get(
                            first_leftmost[first_index] - 1,
                            second_leftmost[second_index] - 1,
                        )
                        replace_cost = previous_forest + tree_distance[first_index][second_index]
                        forest_set(first_index, second_index, min(
                            delete_cost, insert_cost, replace_cost
                        ))
    return tree_distance[first_count][second_count]


def _tree_similarity(first: _Tree, second: _Tree) -> float:
    denominator = max(_tree_size(first), _tree_size(second))
    return max(0.0, 1.0 - _tree_edit_distance(first, second) / denominator)


def _multiset_scores(expected: Sequence[tuple[str, str]], actual: Sequence[tuple[str, str]]) -> dict[str, int | float]:
    expected_counts, actual_counts = Counter(expected), Counter(actual)
    true_positive = sum((expected_counts & actual_counts).values())
    false_positive = sum(actual_counts.values()) - true_positive
    false_negative = sum(expected_counts.values()) - true_positive
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative) if 2 * true_positive + false_positive + false_negative else 1.0
    return {"true_positive": true_positive, "false_positive": false_positive,
            "false_negative": false_negative, "precision": precision, "recall": recall, "f1": f1}


def evaluate_page(
    annotation: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute typed page metrics without conflating assignment and detection.

    Returned counts are additive across pages. Text normalization is deliberately
    absent: CER compares Unicode code points and WER uses whitespace tokens.
    """
    image = annotation["image"]
    image_size = int(image["width"]), int(image["height"])
    line_state = resolve_line_supervision(annotation)
    text_evaluable = line_state == "complete"
    emitted_lines = list(prediction.get("lines", ()))
    # Partial positives remain useful for training but do not define official
    # closed-world detection/transcription metrics. Emissions stay diagnostic.
    gt_lines = list(annotation.get("lines", ())) if text_evaluable else []
    pred_lines = emitted_lines if text_evaluable else []
    gt_regions = list(annotation.get("regions", ()))
    pred_regions = list(prediction.get("regions", ()))
    gt_cells = list(annotation.get("cells", ()))
    pred_cells = list(prediction.get("cells", ()))

    line_counts, gt_line_to_pred, pred_line_to_gt, line_matches = _match_instances(
        gt_lines, pred_lines, image_size, iou_threshold
    )
    detections: dict[str, dict[str, int]] = {"lines": line_counts, "regions": {}, "cells": {}}
    gt_to_pred = dict(gt_line_to_pred)
    pred_to_gt = dict(pred_line_to_gt)
    table_gt_to_pred: dict[str, str] = {}
    table_pred_to_gt: dict[str, str] = {}
    region_kinds = sorted({
        item.get("kind")
        for item in (*gt_regions, *pred_regions)
        if isinstance(item.get("kind"), str) and item["kind"] in REGION_KINDS
    })
    for kind in region_kinds:
        gt_kind = [item for item in gt_regions if item.get("kind") == kind]
        pred_kind = [item for item in pred_regions if item.get("kind") == kind]
        counts, gt_map, pred_map, _ = _match_instances(
            gt_kind, pred_kind, image_size, iou_threshold
        )
        detections["regions"][kind] = counts
        gt_to_pred.update(gt_map)
        pred_to_gt.update(pred_map)
        if kind == "TABLE":
            table_gt_to_pred.update(gt_map)
            table_pred_to_gt.update(pred_map)
    invalid_region_predictions = [
        item for item in pred_regions
        if not isinstance(item.get("kind"), str) or item["kind"] not in REGION_KINDS
    ]
    if invalid_region_predictions:
        detections["regions"]["INVALID"] = {
            "true_positive": 0,
            "false_positive": len(invalid_region_predictions),
            "false_negative": 0,
        }
    cell_counts, gt_cell_to_pred, pred_cell_to_gt, cell_matches = _match_instances(
        gt_cells, pred_cells, image_size, iou_threshold
    )
    detections["cells"] = cell_counts
    gt_to_pred.update(gt_cell_to_pred)
    pred_to_gt.update(pred_cell_to_gt)

    unmatched_relation_target = object()

    def canonical_relation(edge: Mapping[str, Any]) -> tuple[str, Any, Any]:
        relation_type = str(edge.get("type"))
        source = edge.get("source")
        if source is not unmatched_relation_target:
            source = str(source)
        target = edge.get("target")
        if relation_type == "CONTAINS" and target is not None:
            return "PARENT", target, source
        if target is not None and target is not unmatched_relation_target:
            target = str(target)
        return relation_type, source, target

    excluded_line_ids = {
        str(line["id"]) for line in annotation.get("lines", ()) if "id" in line
    } if not text_evaluable else set()
    gt_edges: set[tuple[str, str, str | None]] = set()
    known_relation_sources: dict[str, set[str]] = defaultdict(set)
    for edge in annotation.get("relations", ()):
        if edge.get("source") in excluded_line_ids or edge.get("target") in excluded_line_ids:
            continue
        relation_type, source, target = canonical_relation(edge)
        gt_edges.add((relation_type, source, target))
        known_relation_sources[relation_type].add(source)

    predicted_edges = set()
    for edge in prediction.get("relations", ()):
        source = str(edge.get("source"))
        target = edge.get("target")
        mapped_source = pred_to_gt.get(source, unmatched_relation_target)
        mapped_target = (
            None if target is None
            else pred_to_gt.get(str(target), unmatched_relation_target)
        )
        relation_type, mapped_source, mapped_target = canonical_relation({
            "type": edge.get("type"),
            "source": mapped_source,
            "target": mapped_target,
        })
        # Relation annotations are decision labels, not a closed-world claim
        # about every endpoint pair. Score predictions only for sources whose
        # decision was actually annotated (including explicit NULL labels).
        if mapped_source in known_relation_sources.get(relation_type, set()):
            predicted_edges.add((relation_type, mapped_source, mapped_target))

    relation_types = sorted({edge[0] for edge in gt_edges | predicted_edges})
    relation_counts = {}
    for relation_type in relation_types:
        expected = {edge for edge in gt_edges if edge[0] == relation_type}
        actual = {edge for edge in predicted_edges if edge[0] == relation_type}
        relation_counts[relation_type] = {
            "true_positive": len(expected & actual),
            "false_positive": len(actual - expected),
            "false_negative": len(expected - actual),
        }

    gt_order = [str(item) for item in annotation.get("reading_order", ())] if text_evaluable else []
    pred_order = [str(item) for item in prediction.get("reading_order", ())] if text_evaluable else []
    if not pred_order:
        pred_order = [
            str(line["id"])
            for line in sorted(pred_lines, key=lambda item: item.get("reading_order", 0))
            if isinstance(line.get("id"), str)
        ]
    gt_line_ids = {str(line["id"]) for line in gt_lines if isinstance(line.get("id"), str)}
    pred_line_ids = {str(line["id"]) for line in pred_lines if isinstance(line.get("id"), str)}
    gt_rank = {item_id: index for index, item_id in enumerate(gt_order)}
    pred_rank = {item_id: index for index, item_id in enumerate(pred_order)}
    ordered_gt_ids = [line_id for line_id in gt_order if line_id in gt_line_ids]
    order_pairs = order_correct = 0
    for left_index, left_id in enumerate(ordered_gt_ids):
        for right_id in ordered_gt_ids[left_index + 1 :]:
            order_pairs += 1
            left_pred = gt_line_to_pred.get(left_id)
            right_pred = gt_line_to_pred.get(right_id)
            if left_pred is None or right_pred is None:
                continue
            if left_pred not in pred_rank or right_pred not in pred_rank:
                continue
            order_correct += (pred_rank[left_pred] < pred_rank[right_pred]) == (
                gt_rank[left_id] < gt_rank[right_id]
            )
    predicted_gt_order = [pred_to_gt[item] for item in pred_order if item in pred_to_gt]
    exact_reading_order = (
        len(gt_line_ids) == len(gt_lines)
        and len(pred_line_ids) == len(pred_lines)
        and len(gt_line_to_pred) == len(gt_lines)
        and len(pred_line_to_gt) == len(pred_lines)
        and len(pred_lines) == len(gt_lines)
        and len(gt_order) == len(gt_lines)
        and set(gt_order) == gt_line_ids
        and len(pred_order) == len(pred_lines)
        and set(pred_order) == pred_line_ids
        and len(predicted_gt_order) == len(pred_lines)
        and predicted_gt_order == gt_order
    )

    pred_cell_by_id = {
        str(cell["id"]): cell for cell in pred_cells if isinstance(cell.get("id"), str)
    }
    span_correct = span_count = header_correct = header_count = 0
    for gt_cell in gt_cells:
        pred_cell_id = gt_cell_to_pred.get(str(gt_cell.get("id")))
        pred_cell = pred_cell_by_id.get(pred_cell_id) if pred_cell_id is not None else None
        if all(field in gt_cell and gt_cell[field] is not None for field in ("row", "col")):
            span_count += 1
            span_is_correct = pred_cell is not None and all(
                pred_cell.get(field) == gt_cell.get(field) for field in ("row", "col")
            )
            for field in ("rowspan", "colspan"):
                if field in gt_cell and gt_cell[field] is not None:
                    span_is_correct = span_is_correct and pred_cell is not None and (
                        pred_cell.get(field) == gt_cell[field]
                    )
            span_correct += bool(span_is_correct)
        if isinstance(gt_cell.get("is_header"), bool):
            header_count += 1
            header_correct += bool(
                pred_cell is not None and pred_cell.get("is_header") is gt_cell["is_header"]
            )

    exact_tables = eligible_tables = 0
    pred_table_metadata = {
        str(table.get("table_id")): table
        for table in prediction.get("tables", ())
        if isinstance(table.get("table_id"), str)
    }
    gt_lines_by_id = {
        str(line["id"]): line for line in gt_lines if isinstance(line.get("id"), str)
    }
    pred_lines_by_id = {
        str(line["id"]): line for line in pred_lines if isinstance(line.get("id"), str)
    }
    # TEDS may use independently annotated cell text even when official line
    # metrics are excluded; keep its line-resolution lookup intact on both sides.
    teds_gt_lines = {
        str(line["id"]): line for line in annotation.get("lines", ()) if isinstance(line.get("id"), str)
    }
    teds_pred_lines = {
        str(line["id"]): line for line in emitted_lines if isinstance(line.get("id"), str)
    }
    for gt_table in (region for region in gt_regions if region.get("kind") == "TABLE"):
        gt_table_id = str(gt_table.get("id"))
        expected_cells = [cell for cell in gt_cells if cell.get("table_id") == gt_table_id]
        if any(cell.get("row") is None or cell.get("col") is None for cell in expected_cells):
            continue
        eligible_tables += 1
        pred_table_id = table_gt_to_pred.get(gt_table_id)
        pred_table = pred_table_metadata.get(pred_table_id) if pred_table_id is not None else None
        predicted_cells = [cell for cell in pred_cells if cell.get("table_id") == pred_table_id]
        valid = (
            pred_table_id is not None
            and pred_table is not None
            and pred_table.get("status") != "conflict"
            and len(predicted_cells) == len(expected_cells)
            and len({str(cell.get("id")) for cell in predicted_cells}) == len(predicted_cells)
        )
        gt_grid = _table_tree(
            gt_table_id, gt_cells, gt_table, gt_lines_by_id,
            annotation.get("relations", ()), gt_rank, include_text=False,
        )
        pred_grid = (
            _table_tree(
                pred_table_id, pred_cells, pred_table, pred_lines_by_id,
                prediction.get("relations", ()), pred_rank, include_text=False,
            )
            if pred_table_id is not None else None
        )
        valid = valid and gt_grid is not None and pred_grid is not None and gt_grid == pred_grid
        for gt_cell in expected_cells:
            pred_cell_id = gt_cell_to_pred.get(str(gt_cell.get("id")))
            pred_cell = pred_cell_by_id.get(pred_cell_id) if pred_cell_id is not None else None
            if pred_cell is None or pred_cell.get("table_id") != pred_table_id:
                valid = False
                break
            for field, default in (("row", None), ("col", None), ("rowspan", 1), ("colspan", 1)):
                expected_value = gt_cell.get(field, default)
                predicted_value = pred_cell.get(field, default)
                if field in {"rowspan", "colspan"}:
                    expected_value = default if expected_value is None else expected_value
                    predicted_value = default if predicted_value is None else predicted_value
                if expected_value is not None and predicted_value != expected_value:
                    valid = False
                    break
            if not valid:
                break
            if isinstance(gt_cell.get("is_header"), bool) and pred_cell.get("is_header") is not gt_cell["is_header"]:
                valid = False
                break
        exact_tables += bool(valid)

    gt_records = {
        str(item["id"]): item
        for item in (*gt_lines, *gt_regions, *gt_cells)
        if isinstance(item.get("id"), str)
    }
    pred_records = {
        str(item["id"]): item
        for item in (*pred_lines, *pred_regions, *pred_cells)
        if isinstance(item.get("id"), str)
    }
    gt_relations = list(annotation.get("relations", ()))
    pred_relations = list(prediction.get("relations", ()))
    gt_order_map = {line_id: index for index, line_id in enumerate(gt_order)}
    pred_order_map = {line_id: index for index, line_id in enumerate(pred_order)}
    teds_pred_order = list(prediction.get("reading_order", ())) or [
        line["id"] for line in sorted(emitted_lines, key=lambda item: item.get("reading_order", 0))
        if isinstance(line.get("id"), str)
    ]
    teds_pred_order_map = {line_id: index for index, line_id in enumerate(teds_pred_order)}
    expected_kv = [
        (
            _node_text(str(edge.get("source")), gt_records, gt_lines_by_id, gt_relations, gt_order_map),
            _node_text(str(edge.get("target")), gt_records, gt_lines_by_id, gt_relations, gt_order_map),
        )
        for edge in gt_relations
        if edge.get("type") == "KV_VALUE" and edge.get("target") is not None
    ]
    predicted_kv = [
        (str(pair.get("key", "")), str(pair.get("value", "")))
        for pair in prediction.get("kv_pairs", ())
    ] if text_evaluable else []
    if not text_evaluable:
        expected_kv = []
    key_value_metrics = {
        "exact": _multiset_scores(expected_kv, predicted_kv),
        "normalized": _multiset_scores(
            [(_normalise_value(key), _normalise_value(value)) for key, value in expected_kv],
            [(_normalise_value(key), _normalise_value(value)) for key, value in predicted_kv],
        ),
        "normalization": "Unicode NFC, casefold, and collapse whitespace",
    }

    gt_table_metadata = {
        str(region.get("id")): region
        for region in gt_regions if region.get("kind") == "TABLE"
    }
    teds_sum = teds_structure_sum = 0.0
    teds_count = teds_text_count = teds_ineligible_tables = 0
    for gt_table_id, gt_table in gt_table_metadata.items():
        gt_structure = _table_tree(
            gt_table_id, gt_cells, gt_table, gt_lines_by_id, gt_relations,
            gt_order_map, include_text=False,
        )
        if gt_structure is None:
            teds_ineligible_tables += 1
            continue
        teds_count += 1
        # Direct cell transcripts are independently annotated; otherwise text
        # TEDS requires complete line supervision. Structure remains evaluable.
        table_cells = [cell for cell in gt_cells if cell.get("table_id") == gt_table_id]
        table_text_available = text_evaluable or bool(table_cells) and all(
            isinstance(cell.get("text"), str) for cell in table_cells
        )
        gt_tree = (
            _table_tree(gt_table_id, gt_cells, gt_table, teds_gt_lines, gt_relations,
                        gt_order_map, include_text=True)
            if table_text_available else None
        )
        teds_text_count += int(gt_tree is not None)
        pred_table_id = table_gt_to_pred.get(gt_table_id)
        if pred_table_id is None:
            continue
        pred_table = pred_table_metadata.get(pred_table_id)
        pred_tree = (
            _table_tree(pred_table_id, pred_cells, pred_table, teds_pred_lines,
                        pred_relations, teds_pred_order_map, include_text=True)
            if gt_tree is not None else None
        )
        pred_structure = _table_tree(
            pred_table_id, pred_cells, pred_table, pred_lines_by_id,
            pred_relations, pred_order_map, include_text=False,
        )
        if gt_tree is not None and pred_tree is not None:
            teds_sum += _tree_similarity(gt_tree, pred_tree)
        if pred_structure is not None:
            teds_structure_sum += _tree_similarity(gt_structure, pred_structure)

    table_structure_metrics = {
        "span_correct": span_correct,
        "span_count": span_count,
        "header_correct": header_correct,
        "header_count": header_count,
        "exact_tables": exact_tables,
        "eligible_tables": eligible_tables,
        "teds_sum": teds_sum,
        "teds_structure_sum": teds_structure_sum,
        "teds_count": teds_count,
        "teds_text_count": teds_text_count,
        "teds_ineligible_tables": teds_ineligible_tables,
    }

    language_slices: dict[str, dict[str, int]] = {}
    mark_slice: dict[str, int] = {
        "lines": 0, "character_errors": 0, "reference_characters": 0,
        "word_errors": 0, "reference_words": 0, "exact_lines": 0,
    }
    mark_characters = mark_errors = coeng_count = coeng_errors = 0
    pred_line_by_id = {
        str(line["id"]): line for line in pred_lines if isinstance(line.get("id"), str)
    }
    for gt_line in gt_lines:
        gt_id = str(gt_line.get("id"))
        pred_id = gt_line_to_pred.get(gt_id)
        pred_line = pred_line_by_id.get(pred_id) if pred_id is not None else None
        reference = str(gt_line.get("text", ""))
        hypothesis = "" if pred_line is None else str(pred_line.get("text", ""))
        errors = edit_distance(reference, hypothesis)
        word_errors, reference_words = word_error_count(reference, hypothesis)
        bucket = _language_bucket(reference, gt_line.get("language"))
        group = language_slices.setdefault(bucket, {
            "lines": 0, "character_errors": 0, "reference_characters": 0,
            "word_errors": 0, "reference_words": 0, "exact_lines": 0,
        })
        group["lines"] += 1
        group["character_errors"] += errors
        group["reference_characters"] += len(reference)
        group["word_errors"] += word_errors
        group["reference_words"] += reference_words
        group["exact_lines"] += reference == hypothesis
        has_khmer_mark = any(
            0x17B6 <= ord(char) <= 0x17D3 or ord(char) == 0x17DD
            for char in reference
        )
        if has_khmer_mark:
            mark_slice["lines"] += 1
            mark_slice["character_errors"] += errors
            mark_slice["reference_characters"] += len(reference)
            mark_slice["word_errors"] += word_errors
            mark_slice["reference_words"] += reference_words
            mark_slice["exact_lines"] += reference == hypothesis
        if any(0x1780 <= ord(char) <= 0x17FF for char in reference + hypothesis):
            reference_marks = _khmer_mark_contexts(reference)
            hypothesis_marks = _khmer_mark_contexts(hypothesis)
            reference_coeng = _khmer_mark_contexts(reference, coeng_only=True)
            hypothesis_coeng = _khmer_mark_contexts(hypothesis, coeng_only=True)
            mark_characters += len(reference_marks)
            mark_errors += _sequence_edit_distance(reference_marks, hypothesis_marks)
            coeng_count += len(reference_coeng)
            coeng_errors += _sequence_edit_distance(reference_coeng, hypothesis_coeng)

    # Unmatched predictions are insertions. Attribute them using their declared
    # language when present, otherwise infer from the emitted text so per-language
    # rates do not hide hallucinated lines.
    matched_prediction_objects = {id(predicted) for _, predicted in line_matches}
    for predicted_line in pred_lines:
        if id(predicted_line) in matched_prediction_objects:
            continue
        hypothesis = str(predicted_line.get("text", ""))
        errors = len(hypothesis)
        word_errors, reference_words = word_error_count("", hypothesis)
        bucket = _language_bucket(hypothesis, predicted_line.get("language"))
        group = language_slices.setdefault(bucket, {
            "lines": 0, "character_errors": 0, "reference_characters": 0,
            "word_errors": 0, "reference_words": 0, "exact_lines": 0,
        })
        group["lines"] += 1
        group["character_errors"] += errors
        group["word_errors"] += word_errors
        group["reference_words"] += reference_words
        mark_errors += len(_khmer_mark_contexts(hypothesis))
        coeng_errors += len(_khmer_mark_contexts(hypothesis, coeng_only=True))

    return {
        "line_supervision": line_state,
        "text_metrics_defined": text_evaluable,
        "text_emissions": {
            "line_count": len(emitted_lines),
            "character_count": sum(len(str(line.get("text", ""))) for line in emitted_lines),
            "excluded_from_text_metrics": not text_evaluable,
            "annotated_positive_line_count": len(annotation.get("lines", ())),
        },
        "detections": detections,
        "relations": relation_counts,
        "reading_order": {
            "correct_pairs": order_correct,
            "pair_count": order_pairs,
            "exact_page": int(exact_reading_order and text_evaluable),
            "page_count": int(text_evaluable),
        },
        "table_structure": table_structure_metrics,
        "key_value": key_value_metrics,
        "language_slices": language_slices,
        "khmer_mark_slice": mark_slice,
        "khmer_marks": {
            "character_errors": mark_errors,
            "reference_characters": mark_characters,
            "cer": mark_errors / mark_characters if mark_characters else (1.0 if mark_errors else 0.0),
            "coeng_errors": coeng_errors,
            "reference_coeng": coeng_count,
            "coeng_error_rate": coeng_errors / coeng_count if coeng_count else (1.0 if coeng_errors else 0.0),
            "selection_rule": "lines containing Khmer script; compare mark code points U+17B6..U+17D3 and U+17DD, with coeng U+17D2 reported separately",
        },
    }
