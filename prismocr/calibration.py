"""Validation-only detector confidence fitting and reliability reporting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from prismocr.api import InferenceOptions, PrismOCR, _validated_calibration
from prismocr.data.dataset import PrismPageDataset, dataset_content_sha256
from prismocr.data.vocabulary import Vocabulary
from prismocr.data.schema import resolve_line_supervision
from prismocr.data.tools import _group_records
from prismocr.geometry import source_polygon_iou
from prismocr.matching import linear_sum_assignment

_EPSILON = 1e-6


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sigmoid(value: float) -> float:
    value = max(-80.0, min(80.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _logit(probability: float) -> float:
    value = min(1.0 - _EPSILON, max(_EPSILON, float(probability)))
    return math.log(value / (1.0 - value))


def _fit_platt(scores: Sequence[float], labels: Sequence[bool]) -> tuple[float, float]:
    """Fit a positive-slope Platt transform with damped Newton updates."""
    if len(scores) != len(labels) or not scores:
        raise ValueError("calibration requires at least one scored validation proposal")
    if any(
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0.0 <= score <= 1.0
        for score in scores
    ):
        raise ValueError("calibration scores must be finite probabilities in [0,1]")
    if any(not isinstance(label, bool) for label in labels):
        raise ValueError("calibration labels must be booleans")
    positives = sum(labels)
    if positives == 0 or positives == len(labels):
        raise ValueError("calibration requires both positive and negative validation proposals")
    values = [_logit(score) for score in scores]
    targets = [1.0 if value else 0.0 for value in labels]
    slope, intercept = 1.0, 0.0
    regularization = 1e-6

    def objective(a: float, b: float) -> float:
        return sum(
            max(0.0, z) - target * z + math.log1p(math.exp(-abs(z)))
            for x, target in zip(values, targets, strict=True)
            for z in (a * x + b,)
        ) + regularization * ((a - 1.0) ** 2 + b * b)

    for _ in range(100):
        gradient_a = gradient_b = hessian_aa = hessian_ab = hessian_bb = 0.0
        for x, target in zip(values, targets, strict=True):
            probability = _sigmoid(slope * x + intercept)
            residual = probability - target
            weight = probability * (1.0 - probability)
            gradient_a += residual * x
            gradient_b += residual
            hessian_aa += weight * x * x
            hessian_ab += weight * x
            hessian_bb += weight
        gradient_a += 2.0 * regularization * (slope - 1.0)
        gradient_b += 2.0 * regularization * intercept
        hessian_aa += 2.0 * regularization
        hessian_bb += 2.0 * regularization
        at_slope_boundary = slope <= 1e-4 + 1e-12
        if at_slope_boundary and gradient_a > 0.0:
            # The positive-slope constraint is active. A coupled unconstrained
            # Newton step can pin the slope at its lower bound while leaving
            # the intercept far from its conditional optimum. Follow the
            # feasible tangent direction and optimize the intercept there.
            delta_a = 0.0
            delta_b = gradient_b / hessian_bb
        else:
            determinant = hessian_aa * hessian_bb - hessian_ab * hessian_ab
            if determinant <= 1e-18:
                break
            delta_a = (hessian_bb * gradient_a - hessian_ab * gradient_b) / determinant
            delta_b = (hessian_aa * gradient_b - hessian_ab * gradient_a) / determinant
        old_objective = objective(slope, intercept)
        step = 1.0
        while step >= 1.0 / 1024.0:
            next_slope = max(1e-4, slope - step * delta_a)
            next_intercept = intercept - step * delta_b
            if objective(next_slope, next_intercept) <= old_objective:
                break
            step /= 2.0
        else:
            break
        change = max(abs(next_slope - slope), abs(next_intercept - intercept))
        slope, intercept = next_slope, next_intercept
        if change < 1e-8:
            break
    if not math.isfinite(slope) or not math.isfinite(intercept) or slope <= 0.0:
        raise ValueError("validation confidence fitting did not produce a finite monotonic transform")
    return slope, intercept


def _reliability(
    probabilities: Sequence[float], labels: Sequence[bool], *, bins: int = 10
) -> dict[str, Any]:
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("reliability metrics require scored validation proposals")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("reliability bins must be a positive integer")
    if any(
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(probability)
        or not 0.0 <= probability <= 1.0
        for probability in probabilities
    ) or any(not isinstance(label, bool) for label in labels):
        raise ValueError("reliability inputs must be finite probabilities and boolean labels")
    brier = sum((probability - float(label)) ** 2 for probability, label in zip(probabilities, labels, strict=True)) / len(labels)
    bins_out = []
    weighted_error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [
            position for position, probability in enumerate(probabilities)
            if lower <= probability < upper or (index == bins - 1 and probability == 1.0)
        ]
        if not selected:
            continue
        confidence = sum(probabilities[position] for position in selected) / len(selected)
        accuracy = sum(bool(labels[position]) for position in selected) / len(selected)
        weighted_error += len(selected) / len(labels) * abs(confidence - accuracy)
        bins_out.append({
            "lower": lower,
            "upper": upper,
            "count": len(selected),
            "mean_confidence": confidence,
            "observed_precision": accuracy,
        })
    return {"brier": brier, "ece": weighted_error, "bins": bins_out}


def _precision_recall_curve(
    probabilities: Sequence[float], labels: Sequence[bool], ground_truth_count: int
) -> list[dict[str, float | int]]:
    if len(probabilities) != len(labels) or any(
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
        or not 0.0 <= score <= 1.0
        for score in probabilities
    ) or any(not isinstance(label, bool) for label in labels):
        raise ValueError("precision-recall inputs must be matching finite probabilities and boolean labels")
    if isinstance(ground_truth_count, bool) or not isinstance(ground_truth_count, int) or ground_truth_count < 0:
        raise ValueError("ground_truth_count must be a nonnegative integer")
    curve = []
    for step in range(11):
        threshold = step / 10.0
        selected = [index for index, score in enumerate(probabilities) if score >= threshold]
        true_positives = sum(bool(labels[index]) for index in selected)
        curve.append({
            "threshold": threshold,
            "predictions": len(selected),
            "true_positives": true_positives,
            "precision": true_positives / len(selected) if selected else 1.0,
            "recall": true_positives / ground_truth_count if ground_truth_count else 1.0,
        })
    return curve


def _operating_threshold(
    probabilities: Sequence[float],
    raw_scores: Sequence[float],
    labels: Sequence[bool],
    ground_truth_count: int,
) -> tuple[float, float]:
    if len(probabilities) != len(labels) or len(raw_scores) != len(labels) or not probabilities:
        raise ValueError("operating threshold inputs must be matching nonempty sequences")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
        for value in (*probabilities, *raw_scores)
    ) or any(not isinstance(label, bool) for label in labels):
        raise ValueError("operating threshold inputs must be finite probabilities and boolean labels")
    if isinstance(ground_truth_count, bool) or not isinstance(ground_truth_count, int) or ground_truth_count < 0:
        raise ValueError("ground_truth_count must be a nonnegative integer")
    order = sorted(range(len(raw_scores)), key=lambda index: (-raw_scores[index], index))
    selected = true_positives = 0
    best_threshold, best_raw_threshold, best_f1 = 1.0, 1.0, -1.0
    cursor = 0
    while cursor < len(order):
        raw_threshold = raw_scores[order[cursor]]
        end = cursor
        while end < len(order) and raw_scores[order[end]] == raw_threshold:
            selected += 1
            true_positives += labels[order[end]]
            end += 1
        f1 = (
            2.0 * true_positives / (ground_truth_count + selected)
            if ground_truth_count + selected else 1.0
        )
        threshold = min(probabilities[order[index]] for index in range(cursor, end))
        if f1 > best_f1 or (f1 == best_f1 and raw_threshold > best_raw_threshold):
            best_threshold, best_raw_threshold, best_f1 = threshold, raw_threshold, f1
        cursor = end
    return best_threshold, best_raw_threshold


def _ranked_labels(
    truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float,
    compare_kind: bool,
) -> list[bool]:
    """Assign each ranked prediction to its best unmatched source polygon."""
    truth_polygons = [item.get("polygon_px") for item in truth]
    prediction_polygons = [item.get("polygon") or item.get("polygon_px") for item in predictions]
    quality = [
        [
            source_polygon_iou(gt_polygon, pred_polygon)
            if gt_polygon is not None and pred_polygon is not None else 0.0
            for gt_polygon in truth_polygons
        ]
        for pred_polygon in prediction_polygons
    ]
    labels = [False] * len(predictions)
    used_truth: set[int] = set()
    ranked = sorted(
        range(len(predictions)),
        key=lambda index: (-float(predictions[index]["detector_score"]), str(predictions[index].get("id", index))),
    )
    for prediction_index in ranked:
        candidates = []
        for truth_index, overlap in enumerate(quality[prediction_index]):
            if truth_index in used_truth or overlap < iou_threshold:
                continue
            if compare_kind and truth[truth_index].get("kind") != predictions[prediction_index].get("kind"):
                continue
            candidates.append((overlap, str(truth[truth_index].get("id", truth_index)), truth_index))
        if candidates:
            _overlap, _stable_id, truth_index = max(candidates, key=lambda item: (item[0], item[1]))
            used_truth.add(truth_index)
            labels[prediction_index] = True
    return labels


def _text_detector_samples(
    annotation: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]], *, iou_threshold: float
) -> list[tuple[float, bool]]:
    """Only complete pages can label emitted instances as correct/incorrect."""
    if resolve_line_supervision(annotation) != "complete":
        return []
    labels = _ranked_labels(
        annotation.get("lines", ()), predictions, iou_threshold=iou_threshold, compare_kind=False
    )
    return [(float(item["detector_score"]), label)
            for item, label in zip(predictions, labels, strict=True)]


def _fit_stream(
    samples: Sequence[tuple[float, bool]], *, ground_truth_count: int, blank_predictions: int
) -> dict[str, Any]:
    scores = [score for score, _label in samples]
    labels = [label for _score, label in samples]
    slope, intercept = _fit_platt(scores, labels)
    probabilities = [_sigmoid(slope * _logit(score) + intercept) for score in scores]
    calibrated_threshold, raw_threshold = _operating_threshold(
        probabilities, scores, labels, ground_truth_count
    )
    positives = sum(labels)
    reliability = _reliability(probabilities, labels)
    return {
        "method": "platt_v1",
        "slope": slope,
        "intercept": intercept,
        "threshold": calibrated_threshold,
        "raw_threshold": raw_threshold,
        "sample_count": len(samples),
        "positive_count": positives,
        "negative_count": len(samples) - positives,
        "ground_truth_count": ground_truth_count,
        "blank_page_false_positives": blank_predictions,
        "measurement_scope": (
            "one-to-one confidence-ranked predicted instances; ECE and Brier exclude pages "
            "with no predicted instances, which remain included in the page and blank-page counts"
        ),
        "reliability": reliability,
        "precision_recall_curve": _precision_recall_curve(probabilities, labels, ground_truth_count),
    }


def _matched_node_ids(
    annotation: Mapping[str, Any], prediction: Mapping[str, Any], iou_threshold: float
) -> dict[str, str]:
    truth_nodes = []
    prediction_nodes = []
    for line in annotation.get("lines", ()):
        truth_nodes.append((line["id"], "line", None, line.get("polygon_px")))
    for region in annotation.get("regions", ()):
        truth_nodes.append((region["id"], "region", region.get("kind"), region.get("polygon_px")))
    for cell in annotation.get("cells", ()):
        truth_nodes.append((cell["id"], "cell", "CELL", cell.get("polygon_px")))
    for line in prediction.get("lines", ()):
        prediction_nodes.append((line["id"], "line", None, line.get("polygon", line.get("polygon_px"))))
    for region in prediction.get("regions", ()):
        prediction_nodes.append((region["id"], "region", region.get("kind"), region.get("polygon", region.get("polygon_px"))))
    for cell in prediction.get("cells", ()):
        prediction_nodes.append((cell["id"], "cell", "CELL", cell.get("polygon", cell.get("polygon_px"))))
    mapping: dict[str, str] = {}
    type_keys = sorted({(item[1], item[2]) for item in truth_nodes + prediction_nodes})
    for node_type, kind in type_keys:
        truths = sorted(
            (item for item in truth_nodes if item[1] == node_type and item[2] == kind and item[3] is not None),
            key=lambda item: item[0],
        )
        predictions = sorted(
            (item for item in prediction_nodes if item[1] == node_type and item[2] == kind and item[3] is not None),
            key=lambda item: item[0],
        )
        if not truths or not predictions:
            continue
        quality = np.asarray([
            [source_polygon_iou(truth[3], prediction[3]) for prediction in predictions]
            for truth in truths
        ],
            dtype=np.float64,
        )
        pair_count = min(quality.shape)
        cost = np.where(quality >= iou_threshold, 1.0 - quality, pair_count + 1.0)
        truth_indices, prediction_indices = linear_sum_assignment(cost)
        for truth_index, prediction_index in zip(truth_indices, prediction_indices, strict=True):
            if quality[truth_index, prediction_index] >= iou_threshold:
                mapping[predictions[int(prediction_index)][0]] = truths[int(truth_index)][0]
    return mapping


def _ground_truth_relation_decisions(
    annotation: Mapping[str, Any],
) -> dict[tuple[str, str], str | None]:
    """Return only annotated source decisions, preserving explicit NULL labels."""
    decisions: dict[tuple[str, str], str | None] = {}
    for item in annotation.get("relations", ()):
        relation = str(item["type"])
        source, target = item.get("source"), item.get("target")
        if relation == "CONTAINS":
            if target is not None:
                decisions[("PARENT", str(target))] = str(source)
        elif relation in {"NEXT", "PARENT", "KV_VALUE", "RIGHT_CELL"}:
            decisions[(relation, str(source))] = None if target is None else str(target)

    # Training derives NEXT decisions per source, but explicit edge or NULL labels
    # take precedence and unannotated relation families remain unknown.
    lines = {str(line["id"]): line for line in annotation.get("lines", ())}
    reading_order = (
        [str(item) for item in annotation.get("reading_order", ())]
        if resolve_line_supervision(annotation) == "complete" else []
    )
    scopes: dict[str, list[str]] = {}
    for line_id in reading_order:
        line = lines.get(line_id)
        if line is not None:
            scopes.setdefault(str(line.get("scope_id", "document")), []).append(line_id)
    for scoped_ids in scopes.values():
        for index, source in enumerate(scoped_ids):
            key = ("NEXT", source)
            if key not in decisions:
                decisions[key] = scoped_ids[index + 1] if index + 1 < len(scoped_ids) else None
    return decisions


def _validation_split_provenance(
    validation_manifest: Path, split_report_path: Path
) -> tuple[dict[str, Any], str, str]:
    if not split_report_path.is_file():
        raise ValueError("calibration requires a grouped split-report.json with a named validation manifest")
    try:
        report = json.loads(split_report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read grouped split report: {type(error).__name__}") from None
    manifests = report.get("manifests")
    manifest_hashes = report.get("manifest_sha256")
    split_fingerprints = report.get("split_fingerprints")
    if (
        report.get("status") != "grouped_by_document_template_family_and_exact_image_sha256"
        or not isinstance(manifests, dict)
        or not isinstance(manifest_hashes, dict)
        or not isinstance(split_fingerprints, dict)
        or set(("train", "validation", "test")) - set(manifests)
    ):
        raise ValueError("split report lacks frozen train/validation/test provenance")
    if Path(str(manifests["validation"])).resolve() != validation_manifest.resolve():
        raise ValueError("calibration input is not the split report's validation manifest")
    validation_hash = _sha256_file(validation_manifest)
    if manifest_hashes.get("validation") != validation_hash:
        raise ValueError("validation manifest checksum does not match the split report")
    for split_name in ("train", "validation", "test"):
        split_path = Path(str(manifests[split_name])).resolve()
        if not split_path.is_file() or manifest_hashes.get(split_name) != _sha256_file(split_path):
            raise ValueError(f"frozen {split_name} manifest is missing or changed")
    for field in ("document_sha256", "image_sha256", "group_sha256"):
        if any(not isinstance(split_fingerprints.get(name), dict) or not isinstance(split_fingerprints[name].get(field), list)
               for name in ("train", "validation", "test")):
            raise ValueError("split report is missing hashed split membership")
        memberships = [
            split_fingerprints[name][field] for name in ("train", "validation", "test")
        ]
        for values in memberships:
            if not values or any(
                not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value.lower())
                for value in values
            ):
                raise ValueError(f"split report has invalid hashed {field} membership")
        train, validation, test = (set(values) for values in memberships)
        if train & validation or train & test or validation & test:
            raise ValueError(f"split report contains {field} leakage across train/validation/test")
    return report, validation_hash, _sha256_file(split_report_path)


def _verified_split_datasets(report: Mapping[str, Any]) -> dict[str, PrismPageDataset]:
    """Load frozen split manifests and verify their reported memberships and grouping."""
    manifests = report["manifests"]
    datasets = {name: PrismPageDataset(Path(manifests[name])) for name in ("train", "validation", "test")}
    page_counts = report.get("page_counts")
    if not isinstance(page_counts, dict) or any(
        isinstance(page_counts.get(name), bool)
        or not isinstance(page_counts.get(name), int)
        or page_counts[name] != len(dataset)
        or not len(dataset)
        for name, dataset in datasets.items()
    ):
        raise ValueError("split report page counts do not match nonempty frozen manifests")

    records = [record for dataset in datasets.values() for record in dataset.records]
    combined = object.__new__(PrismPageDataset)
    combined.records = records
    hamming = report.get("near_duplicate_hamming")
    if hamming is not None and (
        isinstance(hamming, bool) or not isinstance(hamming, int) or not 0 <= hamming <= 64
    ):
        raise ValueError("split report has an invalid near-duplicate threshold")
    groups = _group_records(combined, hamming)
    owner_by_record: dict[int, str] = {}
    index = 0
    for name, dataset in datasets.items():
        for _record in dataset.records:
            owner_by_record[index] = name
            index += 1

    memberships = {
        name: {"document_sha256": set(), "image_sha256": set(), "group_sha256": set()}
        for name in datasets
    }
    for name, dataset in datasets.items():
        memberships[name]["document_sha256"].update(
            hashlib.sha256(record["_annotation"]["document_id"].encode("utf-8")).hexdigest()
            for record in dataset.records
        )
        memberships[name]["image_sha256"].update(record["_image_sha256"] for record in dataset.records)
    for group in groups:
        owners = {owner_by_record[item] for item in group}
        if len(owners) != 1:
            raise ValueError("frozen split manifests leak a document, family, or near-duplicate group")
        name = next(iter(owners))
        group_hash = hashlib.sha256(
            "\n".join(sorted(records[item]["_image_sha256"] for item in group)).encode("utf-8")
        ).hexdigest()
        memberships[name]["group_sha256"].add(group_hash)

    reported = report["split_fingerprints"]
    for name in datasets:
        for field, values in memberships[name].items():
            if reported[name].get(field) != sorted(values):
                raise ValueError(f"split report {name} {field} does not match its manifest")
    if report.get("independent_group_count") != len(groups):
        raise ValueError("split report independent group count does not match its manifests")
    return datasets


def calibrate_artifact(
    artifact_dir: str | Path,
    validation_manifest: str | Path,
    *,
    split_report: str | Path | None = None,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Fit detector and relation Platt transforms from a frozen validation split."""
    if isinstance(iou_threshold, bool) or not isinstance(iou_threshold, (int, float)) or not math.isfinite(iou_threshold) or not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be finite and in (0,1]")
    artifact = Path(artifact_dir).resolve()
    manifest_path = Path(validation_manifest).resolve()
    report_path = (
        Path(split_report).resolve()
        if split_report is not None else manifest_path.parent / "split-report.json"
    )
    report, validation_hash, split_report_hash = _validation_split_provenance(manifest_path, report_path)
    artifact_manifest_path = artifact / "manifest.json"
    required_files = (
        artifact / "manifest.json",
        artifact / "model.pt",
        artifact / "model-config.json",
        artifact / "vocab-v4.json",
    )
    if any(not path.is_file() for path in required_files):
        raise ValueError("calibration requires a complete PyTorch model artifact")
    artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    model_hash = _sha256_file(artifact / "model.pt")
    config_hash = _sha256_file(artifact / "model-config.json")
    vocabulary = Vocabulary.load(artifact / "vocab-v4.json")
    vocabulary_hash = vocabulary.sha256
    if artifact_manifest.get("schema_version") != "4.0":
        raise ValueError("calibration requires a v4 model artifact manifest")
    if artifact_manifest.get("model_sha256") != model_hash:
        raise ValueError("artifact model checksum does not match its manifest")
    if artifact_manifest.get("model_config_sha256") != config_hash:
        raise ValueError("artifact model configuration checksum does not match its manifest")
    if artifact_manifest.get("vocabulary_sha256") != vocabulary_hash:
        raise ValueError("artifact vocabulary checksum does not match its manifest")
    training_provenance = artifact_manifest.get("training_provenance")
    if not isinstance(training_provenance, dict):
        raise ValueError("calibration requires artifact-bound train and validation dataset provenance")
    split_datasets = _verified_split_datasets(report)
    train_dataset = split_datasets["train"]
    dataset = split_datasets["validation"]
    actual_train_hash = dataset_content_sha256(train_dataset)
    actual_validation_hash = dataset_content_sha256(dataset)
    if training_provenance.get("train_dataset_sha256") != actual_train_hash:
        raise ValueError("calibration split training manifest does not match the checkpoint training data")
    if training_provenance.get("validation_dataset_sha256") != actual_validation_hash:
        raise ValueError("calibration validation manifest does not match checkpoint validation data")
    sidecar = artifact / "calibration-v4.json"
    if sidecar.exists() or artifact_manifest.get("calibration_sha256") is not None:
        if not sidecar.is_file():
            raise ValueError("artifact manifest references a missing calibration sidecar")
        try:
            existing_calibration = _validated_calibration(
                json.loads(sidecar.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"existing calibration sidecar is invalid: {type(error).__name__}") from None
        if (
            existing_calibration.get("model_sha256") != model_hash
            or existing_calibration.get("model_config_sha256") != config_hash
            or existing_calibration.get("vocabulary_sha256") != vocabulary_hash
            or existing_calibration.get("split_sha256") != validation_hash
            or existing_calibration.get("split_report_sha256") != split_report_hash
            or existing_calibration.get("iou_threshold") != iou_threshold
        ):
            raise FileExistsError(
                "refusing to replace calibration bound to another model, config, vocabulary, split, or IoU threshold"
            )
        sidecar_hash = _sha256_file(sidecar)
        recorded_hash = artifact_manifest.get("calibration_sha256")
        if recorded_hash not in (None, sidecar_hash):
            raise ValueError("artifact manifest calibration checksum does not match the sidecar")
        if recorded_hash is None:
            artifact_manifest["calibration_sha256"] = sidecar_hash
            artifact_manifest["scores_calibrated"] = bool(
                existing_calibration.get("scores_calibrated", False)
            )
            artifact_manifest["relation_scores_calibrated"] = bool(
                existing_calibration.get("relation_scores_calibrated", False)
            )
            _atomic_json(artifact_manifest_path, artifact_manifest)
        return existing_calibration
    engine = PrismOCR.load(artifact, device="cpu")
    relation_names = ("NEXT", "PARENT", "KV_VALUE", "RIGHT_CELL")
    engine.calibration = {
        "method": "validation_only_platt_scaling_v1",
        "model_sha256": model_hash,
        "model_config_sha256": config_hash,
        "vocabulary_sha256": vocabulary_hash,
        "split_sha256": validation_hash,
        "split_report_sha256": split_report_hash,
        "validation_page_count": len(dataset.records),
        "scores_calibrated": True,
        "score_calibrators": {},
        "relation_scores_calibrated": False,
    }

    def collect_detector_samples() -> tuple[
        dict[str, list[tuple[float, bool]]], dict[str, int], dict[str, int]
    ]:
        samples: dict[str, list[tuple[float, bool]]] = {"text": [], "region": []}
        ground_truth = {"text": 0, "region": 0}
        blank_pages = {"page_count": 0, "text_predictions": 0, "region_predictions": 0,
                       "certified_text_blank_page_count": 0}
        availability = {state: 0 for state in ("complete", "partial", "unavailable")}
        excluded_emissions = 0
        failures: list[str] = []
        with torch.inference_mode():
            for record in dataset.records:
                annotation = record["_annotation"]
                with Image.open(record["_image_path"]) as source:
                    image = source.convert("RGB").copy()
                try:
                    prediction = engine.predict(image, InferenceOptions()).to_dict()
                except Exception as error:
                    failures.append(f"{annotation['page_id']}:{type(error).__name__}")
                    continue
                line_state = resolve_line_supervision(annotation)
                availability[line_state] += 1
                truth_text = list(annotation.get("lines", ()))
                truth_regions = [
                    *annotation.get("regions", ()),
                    *({**cell, "kind": "CELL"} for cell in annotation.get("cells", ())),
                ]
                predicted_text = list(prediction["lines"])
                predicted_regions = [
                    *({**item, "kind": item.get("kind")} for item in prediction["regions"]),
                    *({**item, "kind": "CELL"} for item in prediction["cells"]),
                ]
                for family, items in (("text", predicted_text), ("region", predicted_regions)):
                    for item in items:
                        score = item.get("detector_score")
                        if (
                            isinstance(score, bool)
                            or not isinstance(score, (int, float))
                            or not math.isfinite(score)
                            or not 0.0 <= score <= 1.0
                        ):
                            raise ValueError(
                                f"validation prediction has an invalid raw {family} detector score"
                            )
                text_complete = line_state == "complete"
                ground_truth["text"] += len(truth_text) if text_complete else 0
                if not text_complete:
                    excluded_emissions += len(predicted_text)
                if text_complete and not truth_text:
                    blank_pages["certified_text_blank_page_count"] += 1
                    blank_pages["text_predictions"] += len(predicted_text)
                ground_truth["region"] += len(truth_regions)
                if text_complete and not truth_text and not truth_regions:
                    blank_pages["page_count"] += 1
                    blank_pages["region_predictions"] += len(predicted_regions)
                for name, truth, predictions, compare_kind in (
                    ("text", truth_text, predicted_text, False),
                    ("region", truth_regions, predicted_regions, True),
                ):
                    if name == "text":
                        samples[name].extend(_text_detector_samples(
                            annotation, predictions, iou_threshold=iou_threshold
                        ))
                        continue
                    labels = _ranked_labels(
                        truth, predictions, iou_threshold=iou_threshold, compare_kind=compare_kind
                    )
                    samples[name].extend(
                        (float(item["detector_score"]), label)
                        for item, label in zip(predictions, labels, strict=True)
                    )
        if failures:
            raise RuntimeError(
                f"calibration aborted after {len(failures)} validation inference failures; "
                "failed pages must remain in the reliability denominator"
            )
        blank_pages["text_emissions_excluded_from_metrics"] = excluded_emissions
        for state, count in availability.items():
            blank_pages[f"text_{state}_page_count"] = count
        return samples, ground_truth, blank_pages

    stream_samples, ground_truth_counts, blank_page_counts = collect_detector_samples()
    text_report = _fit_stream(
        stream_samples["text"], ground_truth_count=ground_truth_counts["text"],
        blank_predictions=blank_page_counts["text_predictions"],
    )
    region_report = _fit_stream(
        stream_samples["region"], ground_truth_count=ground_truth_counts["region"],
        blank_predictions=blank_page_counts["region_predictions"],
    )
    text_report["blank_page_false_positives"] = blank_page_counts["text_predictions"]
    region_report["blank_page_false_positives"] = blank_page_counts["region_predictions"]
    engine.calibration["score_calibrators"] = {"text": text_report, "region": region_report}
    engine.calibration["text_threshold"] = float(text_report["raw_threshold"])
    engine.calibration["region_threshold"] = float(region_report["raw_threshold"])
    engine.calibration["relation_scores_calibrated"] = True
    engine.calibration["relation_score_calibrators"] = {}
    relation_samples: dict[str, list[tuple[float, bool]]] = {name: [] for name in relation_names}
    relation_ground_truth_counts = {name: 0 for name in relation_samples}
    relation_candidate_positive_counts = {name: 0 for name in relation_samples}
    relation_candidate_negative_counts = {name: 0 for name in relation_samples}
    relation_inference_failures: list[str] = []
    with torch.inference_mode():
        for record in dataset.records:
            annotation = record["_annotation"]
            with Image.open(record["_image_path"]) as source:
                image = source.convert("RGB").copy()
            try:
                prediction = engine.predict(
                    image, InferenceOptions(capture_calibration_data=True)
                ).to_dict()
            except Exception as error:
                relation_inference_failures.append(
                    f"{annotation['page_id']}:{type(error).__name__}"
                )
                continue
            predicted_to_truth = _matched_node_ids(annotation, prediction, iou_threshold)
            truth_decisions = _ground_truth_relation_decisions(annotation)
            for relation, _source in truth_decisions:
                if relation in relation_ground_truth_counts:
                    relation_ground_truth_counts[relation] += 1
            for candidate in prediction.get("diagnostics", {}).get(
                "relation_candidate_calibration_data", ()
            ):
                relation = str(candidate["type"])
                if relation not in relation_samples:
                    continue
                source_id = str(candidate["source"])
                target_id = candidate.get("target")
                source_truth = predicted_to_truth.get(source_id)
                if source_truth is None:
                    continue
                decision_key = (relation, source_truth)
                if decision_key not in truth_decisions:
                    continue
                expected_target = truth_decisions[decision_key]
                if target_id is None:
                    label = expected_target is None
                else:
                    target_truth = predicted_to_truth.get(str(target_id))
                    label = expected_target is not None and target_truth == expected_target
                relation_samples[relation].append((float(candidate["score"]), label))
                if label:
                    relation_candidate_positive_counts[relation] += 1
                else:
                    relation_candidate_negative_counts[relation] += 1
    if relation_inference_failures:
        raise RuntimeError(
            f"relation calibration aborted after {len(relation_inference_failures)} validation "
            "inference failures; all pages must contribute to reliability evidence"
        )
    relation_reports = {}
    for relation, samples in relation_samples.items():
        if relation_candidate_positive_counts[relation] == 0 or relation_candidate_negative_counts[relation] == 0:
            raise ValueError(
                f"validation split needs both correct and incorrect {relation} candidate decisions"
            )
        relation_reports[relation] = _fit_stream(
            samples,
            ground_truth_count=relation_ground_truth_counts[relation],
            blank_predictions=0,
        )
        relation_reports[relation]["edge_positive_count"] = relation_candidate_positive_counts[relation]
        relation_reports[relation]["edge_negative_count"] = relation_candidate_negative_counts[relation]
    calibration = {
        "schema_version": "4.0",
        "method": "validation_only_platt_scaling_v1",
        "scores_calibrated": True,
        "relation_scores_calibrated": True,
        "relation_score_calibrators": relation_reports,
        "grid_scores_calibrated": False,
        "model_sha256": model_hash,
        "model_config_sha256": config_hash,
        "vocabulary_sha256": vocabulary_hash,
        "training_dataset_sha256": actual_train_hash,
        "validation_dataset_sha256": actual_validation_hash,
        "split_sha256": validation_hash,
        "split_report_sha256": split_report_hash,
        "validation_manifest": str(manifest_path),
        "validation_page_count": len(dataset.records),
        "supervision_contract": "lines-availability-v1",
        "line_supervision_page_counts": {
            state: blank_page_counts[f"text_{state}_page_count"]
            for state in ("complete", "partial", "unavailable")
        },
        "iou_threshold": iou_threshold,
        "text_threshold": text_report["raw_threshold"],
        "region_threshold": region_report["raw_threshold"],
        "score_calibrators": {"text": text_report, "region": region_report},
        "blank_pages": blank_page_counts,
        "scope": (
            "Detector candidates are collected through public tiled inference at the artifact's "
            "configured raw-score thresholds; operating thresholds are selected from that fixed, "
            "stitched candidate set and applied after stitching at inference. Detector recall "
            "includes missed ground-truth "
            "instances on complete-text pages only; partial/unavailable text emissions are excluded "
            "and counted diagnostically. Relation candidates are labeled only for annotated source decisions, with "
            "explicit NULL preserved and unannotated sources excluded. ECE/Brier are conditional "
            "on emitted candidate instances. Grid and character confidence remain uncalibrated."
        ),
    }
    sidecar = artifact / "calibration-v4.json"
    temporary_sidecar = sidecar.with_suffix(sidecar.suffix + ".tmp")
    temporary_manifest = artifact_manifest_path.with_suffix(".json.tmp")
    if (
        _sha256_file(artifact / "model.pt") != model_hash
        or _sha256_file(artifact / "model-config.json") != config_hash
        or Vocabulary.load(artifact / "vocab-v4.json").sha256 != vocabulary_hash
    ):
        raise ValueError("artifact weights, configuration, or vocabulary changed during calibration")
    try:
        temporary_sidecar.write_text(
            json.dumps(calibration, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_sidecar.replace(sidecar)
        artifact_manifest["calibration_sha256"] = _sha256_file(sidecar)
        artifact_manifest["scores_calibrated"] = True
        artifact_manifest["relation_scores_calibrated"] = True
        temporary_manifest.write_text(
            json.dumps(artifact_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_manifest.replace(artifact_manifest_path)
    except BaseException:
        sidecar.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        raise
    return calibration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--split-report")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    result = calibrate_artifact(
        args.artifact,
        args.validation_manifest,
        split_report=args.split_report,
        iou_threshold=args.iou_threshold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
