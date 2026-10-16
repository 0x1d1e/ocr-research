"""Grouped-manifest evaluator for held-out text and line-detection quality."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from prismocr.api import InferenceError, InferenceOptions, PrismOCR
from prismocr.data.dataset import PrismPageDataset
from prismocr.data.geometry import polygon_to_box
from prismocr.data.schema import resolve_line_supervision
from prismocr.evaluation import edit_distance, evaluate_page, word_error_count
from prismocr.geometry import source_polygon_iou
from prismocr.matching import match_proposals


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _threshold_key(threshold: float) -> str:
    return str(float(threshold))


def _ratio(numerator: int, denominator: int, *, empty: float | None = 0.0) -> float | None:
    return numerator / denominator if denominator else empty


def _error_rate(errors: int, reference_units: int) -> float:
    """Use a capped 1.0 error rate for insertion-only empty references."""
    return errors / reference_units if reference_units else (1.0 if errors else 0.0)


def _score_counts(counts: dict[str, int]) -> dict[str, int | float]:
    true_positive = counts.get("true_positive", 0)
    false_positive = counts.get("false_positive", 0)
    false_negative = counts.get("false_negative", 0)
    precision = _ratio(
        true_positive, true_positive + false_positive,
        empty=1.0 if true_positive + false_negative == 0 else 0.0,
    )
    recall = _ratio(true_positive, true_positive + false_negative, empty=1.0)
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = 2 * true_positive / denominator if denominator else 1.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(
    manifest: str | Path,
    artifact_dir: str | Path,
    *,
    device: str = "cpu",
    iou_threshold: float = 0.5,
    page_resolution: int | None = None,
    allow_tiling: bool = True,
) -> dict[str, Any]:
    dataset = PrismPageDataset(manifest)
    if not len(dataset):
        raise ValueError("evaluation manifest must contain at least one page")
    api = PrismOCR.load(artifact_dir, device=device)
    transcript_pairs: list[tuple[str, str]] = []
    thresholds = sorted({0.5, 0.75, float(iou_threshold)})
    detection_totals: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        detection_totals[_threshold_key(threshold)] = {
            "lines": defaultdict(int),
            "regions": defaultdict(lambda: defaultdict(int)),
            "cells": defaultdict(int),
        }
    relation_totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    reading_order_totals: dict[str, int] = defaultdict(int)
    table_structure_totals: dict[str, float] = defaultdict(float)
    language_totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    khmer_mark_totals: dict[str, int] = defaultdict(int)
    orthographic_mark_totals: dict[str, int] = defaultdict(int)
    key_value_totals: dict[str, dict[str, int]] = {
        "exact": defaultdict(int),
        "normalized": defaultdict(int),
    }
    matched = assigned = ground_truth_count = 0
    pages_with_overflow = 0
    total_candidates = 0
    total_lines = 0
    unresolved_tables = 0
    table_conflict_warnings = unsupported_line_warnings = 0
    pages_with_table_conflicts = pages_with_unsupported_lines = 0
    inference_failures: list[dict[str, Any]] = []
    page_reports: list[dict[str, Any]] = []
    document_ids: set[str] = set()
    availability_counts = {state: 0 for state in ("complete", "partial", "unavailable")}
    excluded_text_emissions = 0
    for index, record in enumerate(dataset.records):
        annotation = record["_annotation"]
        document_ids.add(annotation["document_id"])
        image_path: Path = record["_image_path"]
        line_state = resolve_line_supervision(annotation)
        availability_counts[line_state] += 1
        text_evaluable = line_state == "complete"
        ground_truth_lines = annotation.get("lines", ()) if text_evaluable else ()
        inference_failed = False
        try:
            with Image.open(image_path) as image:
                result = api.predict(
                    image,
                    InferenceOptions(
                        page_resolution=page_resolution,
                        allow_tiling=allow_tiling,
                        output_coordinate_space="pixels",
                    ),
                ).to_dict()
        except Exception as error:
            inference_failed = True
            failure = {
                "page_index": index,
                "page_id": annotation["page_id"],
                "stage": error.stage if isinstance(error, InferenceError) else "page_prediction",
                "error_type": error.cause_type if isinstance(error, InferenceError) else type(error).__name__,
            }
            inference_failures.append(failure)
            result = {
                "lines": [],
                "warnings": [],
                "diagnostics": {"candidate_count": 0, "overflow": False},
            }
        predicted_lines = list(result.get("lines", ()))
        width = int(annotation["image"]["width"])
        height = int(annotation["image"]["height"])
        gt_boxes = [polygon_to_box(line["polygon_px"], (width, height)) for line in ground_truth_lines]
        scored_lines = predicted_lines if text_evaluable else []
        excluded_text_emissions += len(predicted_lines) if not text_evaluable else 0
        pred_boxes = [polygon_to_box(line["polygon"], (width, height)) for line in scored_lines]
        page_metrics_by_threshold = {
            threshold: evaluate_page(annotation, result, iou_threshold=threshold)
            for threshold in thresholds
        }
        for threshold, page_metrics in page_metrics_by_threshold.items():
            totals = detection_totals[_threshold_key(threshold)]
            for field, value in page_metrics["detections"]["lines"].items():
                totals["lines"][field] += int(value)
            for kind, counts in page_metrics["detections"]["regions"].items():
                for field, value in counts.items():
                    totals["regions"][kind][field] += int(value)
            for field, value in page_metrics["detections"]["cells"].items():
                totals["cells"][field] += int(value)
        page_metrics = page_metrics_by_threshold[float(iou_threshold)]
        for relation_type, counts in page_metrics["relations"].items():
            for field, value in counts.items():
                relation_totals[relation_type][field] += int(value)
        for field, value in page_metrics["reading_order"].items():
            reading_order_totals[field] += int(value)
        for field, value in page_metrics["table_structure"].items():
            table_structure_totals[field] += float(value)
        for metric_kind, counts in page_metrics["key_value"].items():
            if metric_kind not in key_value_totals:
                continue
            for field in ("true_positive", "false_positive", "false_negative"):
                key_value_totals[metric_kind][field] += int(counts[field])
        for language, counts in page_metrics["language_slices"].items():
            for field, value in counts.items():
                language_totals[language][field] += int(value)
        for field, value in page_metrics["khmer_mark_slice"].items():
            khmer_mark_totals[field] += int(value)
        for field in ("character_errors", "reference_characters", "coeng_errors", "reference_coeng"):
            orthographic_mark_totals[field] += int(page_metrics["khmer_marks"][field])
        polygon_quality = [
            [
                source_polygon_iou(
                    ground_truth_lines[gt_index]["polygon_px"],
                    predicted_lines[pred_index].get("polygon")
                    or predicted_lines[pred_index].get("polygon_px"),
                )
                for pred_index in range(len(scored_lines))
            ]
            for gt_index in range(len(ground_truth_lines))
        ]
        match = match_proposals(
            pred_boxes,
            gt_boxes,
            image_size=(width, height),
            iou_threshold=iou_threshold,
            quality_matrix=polygon_quality,
        )
        gt_for_prediction = {gt_index: pred_index for gt_index, pred_index in match.accepted_pairs}
        matched_prediction_indices = set(gt_for_prediction.values())
        for gt_index, line in enumerate(ground_truth_lines):
            prediction = scored_lines[gt_for_prediction[gt_index]] if gt_index in gt_for_prediction else None
            transcript_pairs.append((line["text"], "" if prediction is None else prediction["text"]))
        for pred_index, prediction in enumerate(scored_lines):
            if pred_index not in matched_prediction_indices:
                transcript_pairs.append(("", str(prediction.get("text", ""))))
        selected_page_metrics = page_metrics_by_threshold[float(iou_threshold)]
        page_reports.append({
            "page_index": index,
            "page_id": annotation["page_id"],
            "inference_failed": inference_failed,
            "line_supervision": line_state,
            "text_metrics_defined": text_evaluable,
            "text_emissions": page_metrics["text_emissions"],
            "detection": selected_page_metrics["detections"],
            "line_capacity": {
                "ground_truth": len(gt_boxes),
                "assigned": len(match.assigned_gt_idx),
                "geometrically_matched": len(match.accepted_pairs),
                "unassigned_ground_truth_indices": sorted(
                    set(range(len(gt_boxes))) - set(match.assigned_gt_idx)
                ),
                "missed_ground_truth_indices": sorted(match.missed_gt_idx),
            },
            "inference_diagnostics": {
                "candidate_count": int(result.get("diagnostics", {}).get("candidate_count", 0)),
                "overflow": bool(result.get("diagnostics", {}).get("overflow", False)),
            },
        })
        matched += len(match.accepted_pairs)
        assigned += len(match.assigned_gt_idx)
        ground_truth_count += len(gt_boxes)
        total_lines += len(predicted_lines)
        diagnostics = result.get("diagnostics", {})
        total_candidates += int(diagnostics.get("candidate_count", 0))
        pages_with_overflow += bool(diagnostics.get("overflow", False))
        warnings = result.get("warnings", ())
        table_conflicts = sum(warning.get("code") == "TABLE_GRID_CONFLICT" for warning in warnings)
        unsupported_lines = sum(warning.get("code") == "UNSUPPORTED_LINE_LENGTH" for warning in warnings)
        table_conflict_warnings += table_conflicts
        unsupported_line_warnings += unsupported_lines
        pages_with_table_conflicts += bool(table_conflicts)
        pages_with_unsupported_lines += bool(unsupported_lines)
        unresolved_tables += sum(warning.get("code") == "TABLE_GRID_UNRESOLVED" for warning in warnings)

    character_denominator = sum(len(reference) for reference, _ in transcript_pairs)
    character_errors = sum(edit_distance(reference, hypothesis) for reference, hypothesis in transcript_pairs)
    word_errors = reference_words = 0
    for reference, hypothesis in transcript_pairs:
        errors, count = word_error_count(reference, hypothesis)
        word_errors += errors
        reference_words += count
    detection_metrics: dict[str, Any] = {}
    for threshold_name, totals in detection_totals.items():
        region_metrics = {
            kind: _score_counts(dict(counts))
            for kind, counts in sorted(totals["regions"].items())
        }
        region_total = defaultdict(int, {
            "true_positive": 0, "false_positive": 0, "false_negative": 0,
        })
        for counts in totals["regions"].values():
            for field, value in counts.items():
                region_total[field] += value
        detection_metrics[threshold_name] = {
            "lines": _score_counts(dict(totals["lines"])),
            "regions": region_metrics,
            "regions_total": _score_counts(dict(region_total)),
            "cells": _score_counts(dict(totals["cells"])),
        }

    relation_metrics = {
        relation_type: _score_counts(dict(counts))
        for relation_type, counts in sorted(relation_totals.items())
    }
    language_metrics = {}
    for language, counts in sorted(language_totals.items()):
        language_metrics[language] = {
            **counts,
            "cer": _error_rate(counts["character_errors"], counts["reference_characters"]),
            "wer": _error_rate(counts["word_errors"], counts["reference_words"]),
            "exact_line_accuracy": _ratio(counts["exact_lines"], counts["lines"]),
        }
    khmer_mark_metrics = {
        **khmer_mark_totals,
        "cer": _error_rate(khmer_mark_totals["character_errors"], khmer_mark_totals["reference_characters"]),
        "wer": _error_rate(khmer_mark_totals["word_errors"], khmer_mark_totals["reference_words"]),
        "exact_line_accuracy": _ratio(khmer_mark_totals["exact_lines"], khmer_mark_totals["lines"], empty=1.0),
        "selection_rule": "reference contains U+17B6..U+17D3 or U+17DD",
    }
    artifact_root = Path(artifact_dir)
    artifact_manifest_path = artifact_root / "manifest.json"
    artifact_manifest: dict[str, Any] = {}
    if artifact_manifest_path.is_file():
        try:
            artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            artifact_manifest = {}
    calibration_path = artifact_root / "calibration-v4.json"
    selected_detection_metrics = detection_metrics[_threshold_key(iou_threshold)]
    reading_order_metrics = {
        **reading_order_totals,
        "pairwise_accuracy": _ratio(
            int(reading_order_totals["correct_pairs"]), int(reading_order_totals["pair_count"]), empty=1.0
        ),
        "exact_page_accuracy": _ratio(
            int(reading_order_totals["exact_page"]), int(reading_order_totals["page_count"]), empty=1.0
        ),
    }
    table_metrics = {
        **table_structure_totals,
        "span_exact_accuracy": _ratio(
            int(table_structure_totals["span_correct"]), int(table_structure_totals["span_count"]), empty=1.0
        ),
        "header_accuracy": _ratio(
            int(table_structure_totals["header_correct"]), int(table_structure_totals["header_count"]), empty=1.0
        ),
        "exact_table_accuracy": _ratio(
            int(table_structure_totals["exact_tables"]), int(table_structure_totals["eligible_tables"]), empty=1.0
        ),
        "teds": _ratio(
            float(table_structure_totals["teds_sum"]), int(table_structure_totals["teds_text_count"]), empty=None
        ),
        "teds_structure": _ratio(
            float(table_structure_totals["teds_structure_sum"]), int(table_structure_totals["teds_count"]), empty=None
        ),
        "teds_text_policy": "case-sensitive Unicode code points; whitespace preserved in table text nodes",
    }
    key_value_metrics = {
        mode: _score_counts(dict(counts))
        for mode, counts in key_value_totals.items()
    }
    artifact_hashes = {
        filename: _sha256_file(artifact_root / filename)
        if (artifact_root / filename).is_file() else None
        for filename in ("manifest.json", "model.pt", "model-config.json", "vocab-v4.json", "calibration-v4.json")
    }
    normalized_inputs = [
        {
            "document_id": record["_annotation"]["document_id"],
            "page_id": record["_annotation"]["page_id"],
            "image_sha256": record["_image_sha256"],
            "annotation_sha256": hashlib.sha256(json.dumps(
                record["_annotation"], sort_keys=True, ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest(),
        }
        for record in dataset.records
    ]
    input_inventory_sha256 = hashlib.sha256(json.dumps(
        normalized_inputs, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    calibration_settings = getattr(api, "calibration", {})
    if not isinstance(calibration_settings, dict):
        calibration_settings = {}
    # Zero-denominator rates read as perfect (1.0 recall, 0.0 CER) and must not be
    # mistaken for measured success.
    zero_denominator_metrics: list[str] = []
    if ground_truth_count == 0:
        zero_denominator_metrics.append("line_detection_recall_at_iou")
    if character_denominator == 0:
        zero_denominator_metrics.append("cer_on_matched_detections_with_misses_as_empty")
    if reference_words == 0:
        zero_denominator_metrics.append("wer_with_misses_as_empty")
    if reading_order_totals.get("pair_count", 0) == 0:
        zero_denominator_metrics.append("reading_order")
    if table_structure_totals.get("eligible_tables", 0) == 0:
        zero_denominator_metrics.append("table_structure")
    result = {
        "schema_version": "4.0",
        "model_version": api.model_version,
        "status": "measured_on_supplied_manifest_not_release_validation",
        "document_count": len(document_ids),
        "page_count": len(dataset),
        "supervision_contract": "lines-availability-v1",
        "line_supervision_page_counts": availability_counts,
        "text_metrics_defined": availability_counts["complete"] > 0,
        "text_emissions_excluded_from_metrics": excluded_text_emissions,
        "metrics_are_defined": not zero_denominator_metrics,
        "zero_denominator_metrics": zero_denominator_metrics,
        "inference_failure_count": len(inference_failures),
        "inference_failure_rate": _ratio(len(inference_failures), len(dataset)),
        "inference_failures": inference_failures,
        "protocol": {
            "configured_iou_threshold": float(iou_threshold),
            "detection_iou_thresholds": sorted(float(value) for value in thresholds),
            "page_resolution": page_resolution,
            "allow_tiling": allow_tiling,
            "output_coordinate_space": "pixels",
            "cer": "Unicode codepoint Levenshtein distance; no normalization; misses contribute empty hypotheses",
            "wer": "whitespace token Levenshtein distance; misses contribute empty hypotheses",
            "text_supervision": "only complete pages enter official text metrics; partial/unavailable emissions remain diagnostics; legacy empty lines are unavailable",
        },
        "provenance": {
            "evaluation_manifest_sha256": _sha256_file(dataset.manifest),
            "input_inventory_sha256": input_inventory_sha256,
            "artifact_sha256": artifact_hashes,
            "calibration": {
                "present": calibration_path.is_file(),
                "sidecar_sha256": artifact_hashes["calibration-v4.json"],
                "model_sha256": calibration_settings.get("model_sha256"),
                "split_sha256": calibration_settings.get("split_sha256"),
                "scores_calibrated": calibration_settings.get("scores_calibrated", False),
                "relation_scores_calibrated": calibration_settings.get("relation_scores_calibrated", False),
            },
        },
        "detection": detection_metrics,
        "per_page": page_reports,
        "line_detection_recall_at_iou": selected_detection_metrics["lines"]["recall"],
        "matched_lines": matched,
        "ground_truth_lines": ground_truth_count,
        "line_detection_precision_at_iou": selected_detection_metrics["lines"]["precision"],
        "line_detection_f1_at_iou": selected_detection_metrics["lines"]["f1"],
        "assignment_coverage": assigned / ground_truth_count if ground_truth_count else 1.0,
        "cer_on_geometrically_assigned_detections_with_misses_and_extra_lines_as_edits": _error_rate(character_errors, character_denominator),
        "cer_on_matched_detections_with_misses_as_empty": _error_rate(character_errors, character_denominator),
        "character_errors": character_errors,
        "ground_truth_characters": character_denominator,
        "wer_with_misses_and_extra_lines_as_edits": _error_rate(word_errors, reference_words),
        "wer_with_misses_as_empty": _error_rate(word_errors, reference_words),
        "word_errors": word_errors,
        "ground_truth_words": reference_words,
        "language_slices": language_metrics,
        "khmer_mark_slice": khmer_mark_metrics,
        "khmer_mark_bearing_lines": {
            **khmer_mark_totals,
            "cer": _error_rate(khmer_mark_totals["character_errors"], khmer_mark_totals["reference_characters"]),
            "wer": _error_rate(khmer_mark_totals["word_errors"], khmer_mark_totals["reference_words"]),
            "exact_line_accuracy": _ratio(khmer_mark_totals["exact_lines"], khmer_mark_totals["lines"], empty=1.0),
        },
        "khmer_marks": {
            **orthographic_mark_totals,
            "cer": _error_rate(orthographic_mark_totals["character_errors"], orthographic_mark_totals["reference_characters"]),
            "coeng_error_rate": _error_rate(orthographic_mark_totals["coeng_errors"], orthographic_mark_totals["reference_coeng"]),
            "mark_codepoints": "U+17B6..U+17D3 and U+17DD",
            "coeng_codepoint": "U+17D2",
        },
        "relations": relation_metrics,
        "reading_order": reading_order_metrics,
        "table_structure": table_metrics,
        "key_value": key_value_metrics,
        "predicted_lines": total_lines,
        "proposal_candidates": total_candidates,
        "pages_with_capacity_overflow": pages_with_overflow,
        "pages_with_unresolved_table_grid": unresolved_tables,
        "table_grid_conflict_warnings": table_conflict_warnings,
        "pages_with_table_grid_conflicts": pages_with_table_conflicts,
        "unsupported_line_warnings": unsupported_line_warnings,
        "pages_with_unsupported_lines": pages_with_unsupported_lines,
        "note": "This report measures supplied data only; corpus permission, split isolation, calibration fit quality, and release acceptance remain independent obligations.",
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSONL held-out manifest, grouped by document")
    parser.add_argument("--artifact", required=True, help="artifact directory from training/export pipeline")
    parser.add_argument("--output", help="optional metrics JSON path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--page-resolution", type=int)
    args = parser.parse_args(argv)
    metrics = evaluate(
        args.manifest,
        args.artifact,
        device=args.device,
        iou_threshold=args.iou_threshold,
        page_resolution=args.page_resolution,
    )
    rendered = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
