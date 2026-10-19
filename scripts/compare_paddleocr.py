"""Compare PaddleOCR and PrismOCR on complete-line-labeled validation pages.

Detection is scored on complete pages with one-to-one polygon-IoU matching.
Recognition is scored separately on GT polygons. The default is a 12-page smoke;
--full or --limit 0 evaluates every eligible page. Outputs contain local corpus
identifiers, paths, transcripts, and predictions; keep them local.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "4")

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismocr.api import InferenceOptions, PrismOCR
from prismocr.data.geometry import polygon_to_box
from prismocr.data.orientation import exif_oriented
from prismocr.evaluation import edit_distance
from prismocr.geometry import polygon_area, polygon_intersection, source_polygon_iou

DEFAULT_MANIFEST = Path.home() / "data/prismocr-corpus/splits/val-select.jsonl"
DEFAULT_ARTIFACT = Path.home() / "data/prismocr-runs/real-v4-ep4-checkpoint/artifact"
DEFAULT_REC = Path.home() / "data/ppocrv6_tiny_rec_khmer_en_finetune_stage3_focalctc/best_accuracy/inference"
DEFAULT_DET = Path.home() / ".paddlex/official_models/PP-OCRv6_tiny_det"
IOU_THRESHOLDS = (0.5, 0.75)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_digest(path: Path) -> str:
    return file_sha256(path)


SOURCE_STRATA = ("fintech", "khmersynth", "wildkhmer", "khmerst")


def source_stratum(annotation: dict) -> str:
    provenance = annotation.get("provenance", {})
    identity = " ".join(str(value).lower() for value in (
        annotation.get("page_id", ""), annotation.get("document_id", ""),
        provenance.get("source_dataset", ""), provenance.get("template_family_id", ""),
    ))
    for stratum in SOURCE_STRATA:
        if stratum in identity:
            return stratum
    return str(provenance.get("source_class", "unknown"))


def load_records(path: Path, limit: int | None) -> list[dict]:
    candidates: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_no, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            ann = row.get("annotation")
            if isinstance(ann, str):
                ann = json.loads((path.parent / ann).read_text(encoding="utf-8"))
            elif ann is None and row.get("annotation_path"):
                ann = json.loads((path.parent / row["annotation_path"]).read_text(encoding="utf-8"))
            if not isinstance(ann, dict):
                raise ValueError(f"{path}:{line_no}: inline annotation is required")
            if ann.get("supervision", {}).get("lines") != "complete" or not ann.get("lines"):
                continue
            image_path = Path(row["image_path"])
            if not image_path.is_absolute():
                image_path = (path.parent / image_path).resolve()
            stratum = source_stratum(ann)
            candidates.setdefault(stratum, []).append(
                {"annotation": ann, "image_path": image_path, "manifest_row": line_no, "source_stratum": stratum}
            )
    if not candidates:
        raise ValueError("manifest has no eligible complete-label pages")

    # Take the first eligible manifest page from each available stratum per round.
    order = [name for name in SOURCE_STRATA if name in candidates]
    order += sorted(name for name in candidates if name not in SOURCE_STRATA)
    selected = []
    depth = 0
    while limit is None or len(selected) < limit:
        added = False
        for name in order:
            if depth < len(candidates[name]) and (limit is None or len(selected) < limit):
                selected.append(candidates[name][depth])
                added = True
        if not added:
            break
        depth += 1
    return selected


def polygon_iou(a, b) -> float:
    a = [(float(x), float(y)) for x, y in a]
    b = [(float(x), float(y)) for x, y in b]
    area_a, area_b = polygon_area(a), polygon_area(b)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    inter = polygon_area(polygon_intersection(a, b))
    return max(0.0, min(1.0, inter / max(1e-12, area_a + area_b - inter)))


def assign_max_iou(gt, pred, threshold: float) -> list[tuple[int, int, float]]:
    """Deterministic maximum-cardinality matching above threshold, then max IoU."""
    if not gt or not pred:
        return []
    from prismocr.matching import linear_sum_assignment
    quality = np.asarray([[polygon_iou(g, p) for p in pred] for g in gt], dtype=np.float64)
    # Large invalid-pair cost first maximizes accepted matches. IoU then selects quality.
    penalty = (min(len(gt), len(pred)) + 1) * 2
    rows, cols = linear_sum_assignment(np.where(quality >= threshold, 1.0 - quality, penalty))
    return [(int(i), int(j), float(quality[i, j])) for i, j in zip(rows, cols) if quality[i, j] >= threshold]


def detection_counts(gt_polys: list, pred_polys: list) -> dict:
    result = {"ground_truth": len(gt_polys), "predictions": len(pred_polys)}
    for threshold in IOU_THRESHOLDS:
        matches = assign_max_iou(gt_polys, pred_polys, threshold)
        tp = len(matches)
        fp, fn = len(pred_polys) - tp, len(gt_polys) - tp
        result[f"iou_{threshold:.2f}"] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        }
    return result


def language_bucket(text: str) -> str:
    if any("\u1780" <= c <= "\u17ff" for c in text):
        return "khmer"
    if any(c.isalpha() for c in text):
        return "english"
    return "other_or_empty"


def crop_polygon(image: np.ndarray, polygon) -> np.ndarray:
    points = np.asarray(polygon, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError("GT line polygon must have four points")
    width = max(1, int(round(max(np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[3])))))
    height = max(1, int(round(max(np.linalg.norm(points[3] - points[0]), np.linalg.norm(points[2] - points[1])))))
    target = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(points, target), (width, height),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def oov_count(text: str, vocabulary: set[str]) -> int:
    return sum(ch not in vocabulary for ch in text)


def aggregate_recognition(rows: list[dict]) -> dict:
    refs = sum(len(r["reference"]) for r in rows)
    errors = sum(edit_distance(r["reference"], r.get("hypothesis", "")) for r in rows)
    return {"lines": len(rows), "reference_characters": refs, "character_errors": errors,
            "cer": errors / refs if refs else None,
            "exact_lines": sum(r["reference"] == r.get("hypothesis", "") for r in rows),
            "exact_line_rate": sum(r["reference"] == r.get("hypothesis", "") for r in rows) / len(rows) if rows else None,
            "oov_reference_characters": sum(r["oov_reference_characters"] for r in rows),
            "inference_failures": sum(bool(r.get("failure")) for r in rows)}


def paddle_polygons(result) -> list[list[list[float]]]:
    data = result.json.get("res", result.json) if hasattr(result, "json") else result
    polys = data.get("dt_polys", data.get("polys", [])) if isinstance(data, dict) else []
    return [np.asarray(poly, dtype=float).reshape(-1, 2).tolist() for poly in polys]


def paddle_text(result) -> list[str]:
    data = result.json.get("res", result.json) if hasattr(result, "json") else result
    values = data.get("rec_text", data.get("texts", [])) if isinstance(data, dict) else []
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values]


def prism_oracle_recognize(engine, image: Image.Image, polygon, *, return_frame_stats: bool = False):
    """Run Prism's trained recognizer on a GT geometry box and page features."""
    import torch
    options = InferenceOptions()
    page, original_size, scale = engine._prepare_image(image, options.page_resolution)
    width, height = original_size
    padded, padded_size = engine._pad_tile(page)
    scaled = [(float(x) * scale[0], float(y) * scale[1]) for x, y in polygon]
    box_values = list(polygon_to_box(scaled, (width, height)))
    box_values[0] *= width / padded_size[0]
    box_values[1] *= height / padded_size[1]
    box_values[2] *= width / padded_size[0]
    box_values[3] *= height / padded_size[1]
    box = torch.tensor([box_values], device=engine.device, dtype=torch.float32)
    model = engine.model
    with torch.inference_mode():
        f4, f8, f16 = model.backbone(padded[None].to(engine.device))
        feature_map = model.forward_text_features(f4, f8, f16)
        raw = model.text_head(feature_map)
        box = box.to(dtype=raw["feature_map"].dtype)
        feature_height, feature_width = raw["feature_map"].shape[-2:]
        anchor_x = min(feature_width - 1, max(0, int(float(box[0, 0]) * feature_width)))
        anchor_y = min(feature_height - 1, max(0, int(float(box[0, 1]) * feature_height)))
        embedding = raw["feature_map"][0, :, anchor_y, anchor_x].unsqueeze(0)
        refs = torch.tensor([[0, 0]], device=engine.device, dtype=torch.long)
        glyph = model.lat4(f4) + torch.nn.functional.interpolate(
            model.lat8(f8), size=f4.shape[-2:], mode="bilinear", align_corners=False
        )
        logits = model.recognizer.forward_bucket(glyph, box, embedding, refs, max(model.strip_buckets))[0].float()
        token_ids = logits.argmax(-1)
        text = engine.vocabulary.decode_ctc(token_ids.cpu().tolist()).text
        if not return_frame_stats:
            return text
        return text, {
            "frames": int(token_ids.numel()),
            "greedy_blank_frames": int((token_ids == 0).sum().item()),
            "greedy_blank_frame_rate": float((token_ids == 0).float().mean().item()),
            "mean_blank_posterior": float(logits.softmax(-1)[:, 0].mean().item()),
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT, help="PrismOCR artifact used in validation")
    parser.add_argument("--det-model", type=Path, default=DEFAULT_DET, help="PaddleOCR TextDetection model directory")
    parser.add_argument("--rec-model", type=Path, default=DEFAULT_REC, help="PaddleOCR TextRecognition model directory")
    parser.add_argument("--limit", "--pages", dest="limit", type=int, default=12, help="complete-label pages sampled round-robin across source strata; 0 means all eligible pages")
    parser.add_argument("--full", action="store_true", help="evaluate all eligible complete-line-labeled validation pages (same as --limit 0)")
    parser.add_argument("--output", type=Path, default=Path.home() / "data/paddleocr-prismocr-smoke.json")
    parser.add_argument("--device", default="gpu:0", help="Paddle device, e.g. gpu:0 or cpu")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit/--pages must be nonnegative (0 means all eligible pages)")
    if args.full and args.limit not in (0, 12):
        parser.error("--full cannot be combined with a non-default --limit")
    limit = None if args.full or args.limit == 0 else args.limit
    for label, path in (("manifest", args.manifest), ("Prism artifact", args.artifact),
                        ("detection model", args.det_model), ("recognition model", args.rec_model)):
        if not path.exists():
            parser.error(f"{label} does not exist: {path}")

    records = load_records(args.manifest.resolve(), limit)
    if not records:
        parser.error("manifest has no eligible complete-label pages")
    from paddleocr import TextDetection, TextRecognition
    import paddle
    paddle.set_device(args.device)
    detector = TextDetection(model_name="PP-OCRv6_tiny_det", model_dir=str(args.det_model.resolve()), device=args.device)
    recognizer = TextRecognition(model_name="PP-OCRv6_tiny_rec", model_dir=str(args.rec_model.resolve()), device=args.device)
    prism = PrismOCR.load(args.artifact, device="cuda" if args.device.startswith("gpu") else "cpu")
    dictionary_path = Path.home() / "data/ppocrv6_khmer_en_dict.txt"
    import yaml
    paddle_config = yaml.safe_load((args.rec_model / "inference.yml").read_text(encoding="utf-8"))
    paddle_vocabulary = set(paddle_config["PostProcess"]["character_dict"])
    prism_vocabulary = set(prism.vocabulary.tokens[2:])
    page_reports, oracle_rows = [], []
    language_totals = {name: {"lines": 0, "characters": 0, "oov_characters": 0} for name in ("khmer", "english", "other_or_empty")}
    started = time.monotonic()
    for index, record in enumerate(records):
        ann, image_path = record["annotation"], record["image_path"]
        with Image.open(image_path) as source:
            image = exif_oriented(source).convert("RGB")
            rgb = np.asarray(image)
        if (image.width, image.height) != (ann["image"]["width"], ann["image"]["height"]):
            raise ValueError(f"image/annotation dimensions disagree: {image_path}")
        gt_polys = [line["polygon_px"] for line in ann.get("lines", [])]
        paddle_result = detector.predict(input=str(image_path))[0]
        paddle_preds = paddle_polygons(paddle_result)
        paddle_data = paddle_result.json.get("res", paddle_result.json) if hasattr(paddle_result, "json") else paddle_result
        detector_input = paddle_data.get("input_img") if isinstance(paddle_data, dict) else None
        if detector_input is not None and tuple(np.asarray(detector_input).shape[:2]) != (image.height, image.width):
            raise ValueError(f"Paddle detector input dimensions disagree with oriented image: {image_path}")
        for polygon in paddle_preds:
            points = np.asarray(polygon, dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                raise ValueError(f"invalid Paddle detector polygon: {image_path}")
            if (points[:, 0].min() < -1 or points[:, 1].min() < -1
                    or points[:, 0].max() > image.width + 1 or points[:, 1].max() > image.height + 1):
                raise ValueError(f"Paddle detector polygon outside original image coordinates {image.width}x{image.height}: {image_path}")
        prism_result = prism.predict(image, InferenceOptions(return_debug=False, allow_tiling=False)).to_dict()
        prism_preds = [poly for line in prism_result.get("lines", [])
                       if (poly := line.get("polygon")) and len(poly) >= 3 and polygon_area(poly) > 0]
        report = {"page_id": ann["page_id"], "document_id": ann["document_id"],
                  "manifest_row": record["manifest_row"], "source_class": ann.get("provenance", {}).get("source_class", "unknown"),
                  "source_stratum": record["source_stratum"], "image_dimensions": {"width": image.width, "height": image.height},
                  "paddle_detection_coordinate_space": "original_oriented_image_pixels",
                  "language_lines": {name: sum(language_bucket(line.get("text", "")) == name for line in ann.get("lines", []))
                                     for name in language_totals},
                  "ground_truth_lines": len(gt_polys), "paddle_detection": detection_counts(gt_polys, paddle_preds),
                  "prism_detection": detection_counts(gt_polys, prism_preds)}
        page_reports.append(report)

        # Recognition uses GT polygons and is evaluated separately from detector outputs.
        for line_index, line in enumerate(ann.get("lines", [])):
            reference = str(line.get("text", ""))
            bucket = language_bucket(reference)
            stats = language_totals[bucket]
            stats["lines"] += 1
            stats["characters"] += len(reference)
            oov = oov_count(reference, paddle_vocabulary)
            stats["oov_characters"] += oov
            row = {"page_id": ann["page_id"], "line_id": line.get("id", str(line_index)),
                   "language": bucket, "reference": reference, "paddle_oov_reference_characters": oov,
                   "prism_oov_reference_characters": oov_count(reference, prism_vocabulary)}
            try:
                crop = crop_polygon(rgb, line["polygon_px"])
                pp = recognizer.predict(input=crop)[0]
                row["paddle_hypothesis"] = paddle_text(pp)[0] if paddle_text(pp) else ""
            except Exception as exc:
                row["paddle_failure"] = f"{type(exc).__name__}: {exc}"
                row["paddle_hypothesis"] = ""
            try:
                row["prism_hypothesis"] = prism_oracle_recognize(prism, image, line["polygon_px"])
            except Exception as exc:
                row["prism_failure"] = f"{type(exc).__name__}: {exc}"
                row["prism_hypothesis"] = ""
            # Crop inference errors count as failed rows; each model summary keeps its failures explicit.
            oracle_rows.append(row)
        print(json.dumps({"page": index + 1, "page_id": ann["page_id"], "gt": len(gt_polys),
                          "paddle_detections": len(paddle_preds), "prism_detections": len(prism_preds)}, ensure_ascii=False), flush=True)

    summaries = {}
    for model in ("paddle", "prism"):
        rows = [{"reference": r["reference"], "hypothesis": r.get(f"{model}_hypothesis", ""),
                 "oov_reference_characters": r[f"{model}_oov_reference_characters"],
                 "failure": r.get(f"{model}_failure")} for r in oracle_rows]
        summaries[model] = {"overall": aggregate_recognition(rows), "by_language": {lang: aggregate_recognition([r for r in rows if language_bucket(r["reference"]) == lang]) for lang in language_totals}}
    det_summary = {}
    for model in ("paddle", "prism"):
        det_summary[model] = {f"iou_{threshold:.2f}": {} for threshold in IOU_THRESHOLDS}
        for threshold in IOU_THRESHOLDS:
            key = f"iou_{threshold:.2f}"
            tp = sum(p[f"{model}_detection"][key]["tp"] for p in page_reports)
            fp = sum(p[f"{model}_detection"][key]["fp"] for p in page_reports)
            fn = sum(p[f"{model}_detection"][key]["fn"] for p in page_reports)
            det_summary[model][key] = {"tp": tp, "fp": fp, "fn": fn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}
    source_breakdown = {}
    for source in sorted({p["source_stratum"] for p in page_reports}):
        source_pages = [p for p in page_reports if p["source_stratum"] == source]
        source_breakdown[source] = {"pages": len(source_pages), "ground_truth_lines": sum(p["ground_truth_lines"] for p in source_pages),
                                    "paddle_detections": sum(p["paddle_detection"]["predictions"] for p in source_pages),
                                    "prism_detections": sum(p["prism_detection"]["predictions"] for p in source_pages),
                                    "detection_metrics": {model: {f"iou_{threshold:.2f}": {
                                        key: sum(page[f"{model}_detection"][f"iou_{threshold:.2f}"][key] for page in source_pages)
                                        for key in ("tp", "fp", "fn")}
                                        for threshold in IOU_THRESHOLDS} for model in ("paddle", "prism")}}
        for model in ("paddle", "prism"):
            for threshold in IOU_THRESHOLDS:
                metrics = source_breakdown[source]["detection_metrics"][model][f"iou_{threshold:.2f}"]
                tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
                metrics.update({"precision": tp / (tp + fp) if tp + fp else None,
                                "recall": tp / (tp + fn) if tp + fn else None,
                                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None})
    selected_by_stratum = {name: sum(record["source_stratum"] == name for record in records)
                           for name in sorted({record["source_stratum"] for record in records})}
    recognition_by_source = {}
    page_source = {page["page_id"]: page["source_stratum"] for page in page_reports}
    for source in sorted(selected_by_stratum):
        source_rows = [row for row in oracle_rows if page_source[row["page_id"]] == source]
        recognition_by_source[source] = {model: aggregate_recognition([
            {"reference": row["reference"], "hypothesis": row.get(f"{model}_hypothesis", ""),
             "oov_reference_characters": row[f"{model}_oov_reference_characters"],
             "failure": row.get(f"{model}_failure")} for row in source_rows]) for model in ("paddle", "prism")}
    selection_identity = [{"manifest_row": record["manifest_row"], "page_id": record["annotation"]["page_id"],
                           "source_stratum": record["source_stratum"]} for record in records]
    selection_sha256 = hashlib.sha256(json.dumps(selection_identity, sort_keys=True, separators=(",", ":"),
                                          ensure_ascii=False).encode("utf-8")).hexdigest()
    output = {"status": "completed", "manifest": str(args.manifest.resolve()), "manifest_sha256": manifest_digest(args.manifest.resolve()),
              "requested_limit": "all" if limit is None else limit, "selected_pages": len(records),
              "selection": {"method": "deterministic round-robin over manifest-ordered eligible pages by source stratum; priority fintech, khmersynth, wildkhmer, khmerst, then remaining strata alphabetically",
                            "eligible": "manifest validation records with supervision.lines == complete and at least one labeled line",
                            "selected_page_ids": [record["annotation"]["page_id"] for record in records],
                            "selected_records_sha256": selection_sha256,
                            "stratum_page_counts": selected_by_stratum, "validation_only": True},
              "recognition_summary_by_source": recognition_by_source,
              "models": {"prism_artifact": str(args.artifact.resolve()), "paddle_detection": str(args.det_model.resolve()),
                         "paddle_recognition": str(args.rec_model.resolve()), "paddleocr_version": __import__("paddleocr").__version__,
                         "paddle_device": args.device}, "detection_summary": det_summary,
              "oracle_recognition_summary": summaries, "oracle_reference_by_language": language_totals,
              "source_breakdown": source_breakdown,
              "pages": page_reports, "oracle_lines": oracle_rows,
              "limitations": ["The local fine-tuned PaddleOCR recognition model is not an official Khmer baseline.",
                              "Detection and GT-polygon oracle recognition are separate. Paddle recognition receives perspective-cropped GT pixels; Prism oracle recognition samples GT polygon-derived rotated boxes from page feature maps, so the recognizers receive different inputs.",
                              "No test split is opened. Output includes private source paths and transcripts.",
                              "OOV counts compare reference code points to each model vocabulary; they do not measure model coverage.",
                              "Prism oracle recognition uses its trained recognizer with GT polygons converted to rotated boxes and anchor-cell embeddings. Paddle detector polygons are checked against the oriented input image dimensions and reported in original-image pixel coordinates."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"summary={args.output.resolve()} pages={len(records)} elapsed_seconds={time.monotonic()-started:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
