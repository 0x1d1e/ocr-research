"""Canonical decoded-document comparison for PyTorch and ONNX inference."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from prismocr.onnx_export import ONNXExportError


def _polygon(record: dict[str, Any]) -> tuple[tuple[float, float], ...] | None:
    points = record.get("polygon_px", record.get("polygon"))
    if points is None:
        return None
    return tuple(tuple(round(float(value), 4) for value in point) for point in points)


def _canonical_document(result: dict[str, Any], *, onnx: bool) -> tuple[dict[str, Any], dict[tuple[str, str, str | None], float]]:
    annotation = result["annotation"] if onnx else result
    line_records = annotation["lines"]
    region_records = annotation["regions"]
    cell_records = annotation["cells"]
    cell_metadata = {
        cell["id"]: cell for cell in result.get("cell_proposals", ())
    } if onnx else {}

    lines = tuple({
        "id": item["id"],
        "polygon": _polygon(item),
        "text": item.get("text", ""),
        "role": item.get("role"),
        "scope_id": item.get("scope_id"),
        "parent_id": item.get("parent_id"),
        "reading_order": item.get("reading_order"),
        "oov_positions": tuple(item.get("oov_positions", ())),
        "language": item.get("language"),
        "detector_score": item.get("detector_score"),
    } for item in sorted(line_records, key=lambda item: item["id"]))
    regions = tuple({
        "id": item["id"],
        "kind": item.get("kind"),
        "polygon": _polygon(item),
        "parent_id": item.get("parent_id"),
        "detector_score": item.get("detector_score"),
    } for item in sorted(region_records, key=lambda item: item["id"]))
    cells = tuple({
        "id": item["id"],
        "table_id": item.get("table_id"),
        "row": item.get("row"),
        "col": item.get("col"),
        "rowspan": item.get("rowspan"),
        "colspan": item.get("colspan"),
        "is_header": item.get("is_header"),
        "polygon": _polygon(item),
        "line_ids": tuple(item.get("line_ids", ())),
        "text": item.get("text", ""),
        "status": cell_metadata.get(item["id"], item).get("status"),
        "unassigned_reason": cell_metadata.get(item["id"], item).get("unassigned_reason"),
        "detector_score": item.get("detector_score"),
    } for item in sorted(cell_records, key=lambda item: item["id"]))

    if onnx:
        table_records = result["table_decode"]["tables"]
        relation_records = annotation["relations"]
        graph_records = result["graph"]["predictions"]
    else:
        table_records = result["tables"]
        relation_records = result["relations"]
        graph_records = relation_records
    tables = tuple({
        "id": item.get("id", item.get("table_id")),
        "status": item.get("status"),
        "rows": item.get("rows"),
        "cols": item.get("cols"),
        "cells": tuple(sorted(
            cell.get("id") if isinstance(cell, dict) else cell
            for cell in item.get("cells", ())
        )),
        "html": item.get("html"),
        "markdown": item.get("markdown"),
        "conflicts": tuple(item.get("conflicts", ())),
    } for item in sorted(table_records, key=lambda item: item.get("id", item.get("table_id", ""))))
    graph_reasons = {
        (
            item.get("type", item.get("relation")),
            item.get("source", item.get("source_id")),
            item.get("target", item.get("target_id")),
        ): tuple(item.get("reason_codes", ()))
        for item in graph_records
    } if onnx else {}
    relations = tuple(sorted((
        item.get("type", item.get("relation")),
        item.get("source", item.get("source_id")),
        item.get("target", item.get("target_id")),
        tuple(item.get("reason_codes", graph_reasons.get((
            item.get("type", item.get("relation")),
            item.get("source", item.get("source_id")),
            item.get("target", item.get("target_id")),
        ), ()))),
    ) for item in relation_records))
    reading_order = tuple(annotation.get("reading_order", ()))
    warning_details = tuple(sorted(
        json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        for item in result.get("warnings", ())
    ))
    diagnostics = result.get("diagnostics", {})
    suppressed = tuple(sorted((
        item.get("type"), item.get("source"), item.get("target"), item.get("reason_code")
    ) for item in diagnostics.get("relation_suppressed", ())))
    overflow = (
        bool(diagnostics.get("text_capacity_overflow", diagnostics.get("text_overflow", False))),
        bool(diagnostics.get("structural_capacity_overflow", False)),
        bool(diagnostics.get("overflow", False)),
    )
    kv_pairs = tuple({
        "key": item.get("key", ""),
        "value": item.get("value", ""),
        "source_id": item.get("source_id"),
        "target_id": item.get("target_id"),
        "reading_order": item.get("reading_order"),
    } for item in sorted(result.get("kv_pairs", ()), key=lambda item: (item.get("source_id", ""), item.get("target_id", ""))))

    scores: dict[tuple[str, str, str | None], float] = {}
    for item in graph_records:
        relation = item.get("type", item.get("relation"))
        source = item.get("source", item.get("source_id"))
        target = item.get("target", item.get("target_id"))
        score = item.get("score")
        if score is not None:
            scores[(str(relation), str(source), target)] = float(score)
    document = {
        "image_size": (
            annotation["image"]["width"], annotation["image"]["height"]
        ),
        "lines": lines,
        "regions": regions,
        "cells": cells,
        "tables": tables,
        "relations": relations,
        "reading_order": reading_order,
        "warnings": warning_details,
        "suppressed_relations": suppressed,
        "overflow": overflow,
        "kv_pairs": kv_pairs,
    }
    return document, scores


def _differing_semantic_fields(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    atol: float,
    rtol: float,
) -> list[str]:
    differing: list[str] = []
    for field, expected_value in expected.items():
        actual_value = actual[field]
        if field not in {"lines", "regions", "cells"}:
            if expected_value != actual_value:
                differing.append(field)
            continue
        if len(expected_value) != len(actual_value):
            differing.append(field)
            continue
        for expected_record, actual_record in zip(expected_value, actual_value):
            if expected_record.keys() != actual_record.keys():
                differing.append(field)
                break
            different = False
            for key in expected_record.keys() - {"polygon"}:
                expected_value, actual_value = expected_record[key], actual_record[key]
                if key == "detector_score":
                    if expected_value is None or actual_value is None:
                        different = expected_value is not actual_value
                    else:
                        different = not np.isclose(
                            expected_value, actual_value, atol=atol, rtol=rtol
                        )
                else:
                    different = expected_value != actual_value
                if different:
                    break
            if different:
                differing.append(field)
                break
            expected_polygon = expected_record["polygon"]
            actual_polygon = actual_record["polygon"]
            if expected_polygon is None or actual_polygon is None:
                if expected_polygon is not actual_polygon:
                    differing.append(field)
                    break
            elif not np.allclose(
                np.asarray(expected_polygon, dtype=np.float64),
                np.asarray(actual_polygon, dtype=np.float64),
                atol=atol,
                rtol=rtol,
            ):
                differing.append(field)
                break
    return differing


def assert_decoded_document_parity(
    pytorch_result: dict[str, Any],
    onnx_result: dict[str, Any],
    *,
    label: str,
    atol: float,
    rtol: float,
) -> None:
    """Fail unless semantic outputs agree and polygons are within numeric tolerance."""
    expected, expected_scores = _canonical_document(pytorch_result, onnx=False)
    actual, actual_scores = _canonical_document(onnx_result, onnx=True)
    differing = _differing_semantic_fields(expected, actual, atol=atol, rtol=rtol)
    if differing:
        raise ONNXExportError(
            f"decoded document parity failed at {label}: semantic fields differ: {differing}"
        )
    if expected_scores.keys() != actual_scores.keys():
        raise ONNXExportError(
            f"decoded document parity failed at {label}: selected relation edges differ"
        )
    for edge, expected_score in expected_scores.items():
        actual_score = actual_scores[edge]
        if not np.isclose(expected_score, actual_score, atol=atol, rtol=rtol):
            raise ONNXExportError(
                f"decoded document parity failed at {label}: relation score differs for {edge}"
            )
