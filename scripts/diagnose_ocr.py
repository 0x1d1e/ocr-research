"""Bounded, observational detector/CTC diagnosis; never trains or changes an artifact.

Run from the repository with ``python -m scripts.diagnose_ocr --help``. Only selected
pages are image-validated/decoded. Text and predictions are written locally; do not
publish the line-details files without reviewing data permissions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import sys
import tempfile
import time

# Bound native pools before importing numerical libraries (also enforced by torch).
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from prismocr.api import InferenceOptions, PrismOCR
from prismocr.data.dataset import PrismPageDataset, collate_pages
from prismocr.data.geometry import polygon_to_box
from prismocr.data.orientation import exif_oriented
from prismocr.evaluation import edit_distance
from prismocr.geometry import polygon_area, polygon_intersection, rotated_box_to_polygon, source_polygon_iou
from prismocr.losses import PrismOCRLoss
from prismocr.matching import match_proposals
from prismocr.model.heads import TextLineHead
from prismocr.model.recognizer import choose_inference_bucket, minimum_ctc_frames

STRATA = ("fintech", "khmersynth", "wildkhmer-cropped", "wildkhmer-warped", "khmerst", "doclaynet")
THRESHOLDS = (0.55, 0.3, 0.1, 0.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def summary(values) -> dict:
    values = list(values)
    if not values:
        return {"count": 0}
    return {"count": len(values), "min": min(values), "median": float(np.median(values)),
            "mean": float(np.mean(values)), "max": max(values)}


def select_manifest(manifest: Path, output: Path, pages_per_stratum: int, limit: int) -> list[dict]:
    """First eligible manifest rows in each named stratum; no score/label-quality selection."""
    groups = {name: [] for name in STRATA}
    with manifest.open(encoding="utf-8") as stream:
        for row_index, raw in enumerate(stream):
            if not raw.strip():
                continue
            record = json.loads(raw)
            annotation = record.get("annotation")
            if not isinstance(annotation, dict):
                location = record.get("annotation_path", annotation)
                if location:
                    annotation = json.loads((manifest.parent / location).read_text(encoding="utf-8"))
                else:
                    annotation = {k: v for k, v in record.items() if k != "image_path"}
            document = annotation["document_id"]
            name = next((s for s in STRATA if document.startswith(s + "-") or document.startswith(s + ":")), None)
            if name is None or len(groups[name]) >= pages_per_stratum:
                continue
            lines = annotation.get("lines", [])
            if (name == "doclaynet" and lines) or (name != "doclaynet" and not any(x.get("text") for x in lines)):
                continue
            groups[name].append({"image_path": str((manifest.parent / record["image_path"]).resolve()),
                                 "annotation": annotation, "diagnosis_selection": {"manifest_row_zero_based": row_index, "stratum": name,
                                     "text_supervision": "unlabelled" if name == "doclaynet" else "annotated"}})
    chosen = [row for name in STRATA for row in groups[name]][:limit]
    if not chosen:
        raise ValueError("no eligible pages in the six named strata")
    with output.open("w", encoding="utf-8") as stream:
        for row in chosen:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return chosen


def box_quality(gt_boxes, pred_boxes, size) -> np.ndarray:
    """Exact rotated rectangle IoU, with cached polygons/AABB rejection."""
    gt_polys = [rotated_box_to_polygon(b, size) for b in gt_boxes]
    pred_polys = [rotated_box_to_polygon(b, size) for b in pred_boxes]
    def bounds(p):
        x, y = zip(*p)
        return min(x), min(y), max(x), max(y)
    pb = [bounds(p) for p in pred_polys]
    pa = [polygon_area(p) for p in pred_polys]
    quality = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, g in enumerate(gt_polys):
        gb, ga = bounds(g), polygon_area(g)
        for j, p in enumerate(pred_polys):
            b = pb[j]
            if gb[2] <= b[0] or b[2] <= gb[0] or gb[3] <= b[1] or b[3] <= gb[1]:
                continue
            intersection = polygon_area(polygon_intersection(g, p))
            quality[i, j] = min(1.0, max(0.0, intersection / max(1e-12, ga + pa[j] - intersection)))
    return quality


def decode_stats(logits, vocabulary) -> list[dict]:
    logits = logits.detach().float()
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("nonfinite recognizer logits")
    probabilities = logits.softmax(-1)
    ids = logits.argmax(-1)
    blank_mean = probabilities[..., vocabulary.blank_id].mean(-1).cpu().tolist()
    blank_top = (ids == vocabulary.blank_id).float().mean(-1).cpu().tolist()
    paths = ids.cpu().tolist()
    return [{"text": vocabulary.decode_ctc(path).text, "frames": len(path),
             "frame_blank_posterior_mean": mean, "frame_blank_top1_fraction": top}
            for path, mean, top in zip(paths, blank_mean, blank_top)]


def aggregate_recognition(rows: list[dict]) -> dict:
    refs = sum(len(r["reference"]) for r in rows)
    failures = [r for r in rows if r.get("failure")]
    available = [r for r in rows if not r.get("failure")]
    errors = sum(edit_distance(r["reference"], r.get("text", "")) for r in rows)
    frames = sum(r.get("frames", 0) for r in available)
    return {"lines": len(rows), "reference_characters": refs, "character_errors": errors,
            "cer_failures_as_empty": errors / refs if refs else None,
            "exact_lines": sum(r["reference"] == r.get("text", "") for r in rows),
            "blank_outputs": sum(not r.get("text", "") for r in available),
            "blank_output_rate_available": sum(not r.get("text", "") for r in available) / len(available) if available else None,
            "bucket_failures": len(failures), "minframes_exceeds_bucket": sum(r.get("minframes_exceeds_bucket", False) for r in rows),
            "frame_blank_posterior_mean": sum(r["frame_blank_posterior_mean"] * r["frames"] for r in available) / frames if frames else None,
            "frame_blank_top1_fraction": sum(r["frame_blank_top1_fraction"] * r["frames"] for r in available) / frames if frames else None}


def aggregate_public(pages: list[dict]) -> dict:
    """Retain all emission diagnostics; score text only where supervision is available."""
    supervised = [p for p in pages if p["text_supervision"] == "annotated"]
    counts = {f: sum(p[f] for p in pages) for f in
              ("ground_truth_lines", "detections", "matched_iou_0.5", "blank_outputs")}
    for field in ("reference_characters", "character_errors_including_unmatched_insertions"):
        counts[field] = sum(p[field] for p in supervised)
    counts["text_supervised_pages"] = len(supervised)
    counts["text_unlabelled_pages"] = len(pages) - len(supervised)
    counts["text_unlabelled_detections"] = sum(p["detections"] for p in pages if p["text_supervision"] == "unlabelled")
    counts["recall_iou_0.5"] = counts["matched_iou_0.5"] / counts["ground_truth_lines"] if counts["ground_truth_lines"] else None
    counts["e2e_cer"] = counts["character_errors_including_unmatched_insertions"] / counts["reference_characters"] if counts["reference_characters"] else None
    counts["blank_output_rate"] = counts["blank_outputs"] / counts["detections"] if counts["detections"] else None
    return counts


def public_probe(engine, image_path: Path, annotation: dict, resolution: int, allow_tiling: bool = True,
                 *, text_supervised: bool) -> tuple[dict, dict]:
    # An empty line list does not certify a blank page. The caller must explicitly
    # distinguish unavailable text labels from supervised (possibly blank) pages.
    frames = []
    def capture(module, inputs, outputs):
        for group in outputs["ocr_groups"].values():
            frames.extend(decode_stats(group["logits"], engine.vocabulary))
    handle = engine.model.register_forward_hook(capture)
    try:
        with Image.open(image_path) as image:
            result = engine.predict(exif_oriented(image), InferenceOptions(page_resolution=resolution, allow_tiling=allow_tiling, return_debug=True)).to_dict()
    finally:
        handle.remove()
    gt = annotation.get("lines", [])
    predictions = result.get("lines", [])
    size = (annotation["image"]["width"], annotation["image"]["height"])
    quality = np.array([[source_polygon_iou(g["polygon_px"], p["polygon"]) for p in predictions] for g in gt]).reshape(len(gt), len(predictions))
    match = match_proposals([polygon_to_box(p["polygon"], size) for p in predictions],
                            [polygon_to_box(g["polygon_px"], size) for g in gt], image_size=size, quality_matrix=quality)
    matched = dict(match.accepted_pairs)
    matched_predictions = set(matched.values())
    pairs = [(g["text"], predictions[matched[i]]["text"] if i in matched else "") for i, g in enumerate(gt)]
    pairs += [("", p["text"]) for i, p in enumerate(predictions) if i not in matched_predictions]
    chars = sum(len(g["text"]) for g in gt) if text_supervised else None
    errors = sum(edit_distance(ref, hyp) for ref, hyp in pairs) if text_supervised else None
    report = {"text_supervision": "annotated" if text_supervised else "unlabelled",
              "ground_truth_lines": len(gt), "detections": len(predictions), "matched_iou_0.5": len(matched),
              "reference_characters": chars, "character_errors_including_unmatched_insertions": errors,
              "e2e_cer": errors / chars if chars else None,
              "blank_outputs": sum(not p["text"] for p in predictions),
              "blank_output_rate": sum(not p["text"] for p in predictions) / len(predictions) if predictions else None,
              "recognized_tile_lines": len(frames),
              "tile_frame_stats": {k: v for k, v in aggregate_recognition([{**r, "reference": ""} for r in frames]).items()
                                   if k in {"lines", "blank_outputs", "blank_output_rate_available", "frame_blank_posterior_mean", "frame_blank_top1_fraction"}},
              "diagnostics": result.get("diagnostics"), "warnings": result.get("warnings")}
    return report, result


@torch.inference_mode()
def canvas_probe(engine, dataset, index: int) -> dict:
    """Same interior pixels, stride-aligned canvases; no BN/stat/weight mutation."""
    image, target = dataset[index]
    full, adjusted = collate_pages([(image, target)])
    full = full.to(engine.device)
    full_size = (full.shape[-1], full.shape[-2])
    eligible = [line for line in adjusted[0]["lines"]
                if 64 < line["box"][0] * full_size[0] < min(full_size[0], 1024) - 64
                and 64 < line["box"][1] * full_size[1] < min(full_size[1], 512) - 64]
    if not eligible or full_size[1] <= 1024:
        return {"status": "not_applicable", "reason": "no common interior GT anchor or full height <=1024"}
    line = min(eligible, key=lambda r: abs(r["box"][1] * full_size[1] - 256))
    x, y = int(line["box"][0] * full_size[0] / 4), int(line["box"][1] * full_size[1] / 4)
    rows = []
    for height in (full_size[1], 1024, 512):
        tensor = full if height == full_size[1] else full[:, :, :height, :min(1024, full_size[0])]
        f4, f8, f16 = engine.model.backbone(tensor)
        raw = engine.model.text_head(engine.model.forward_text_features(f4, f8, f16))
        size = (tensor.shape[-1], tensor.shape[-2])
        raw_box = raw["box_map"][0, :, y, x]
        normalized_size = TextLineHead.decode_sizes(
            raw_box[2:4],
            model_contract_version=engine.model.model_contract_version,
            image_size=size,
        )
        rows.append({"canvas_size": list(size), "anchor_grid_xy": [x, y],
                     "heatmap_logit": float(raw["heatmap_logits"][0, 0, y, x]),
                     "raw_box": raw_box.cpu().tolist(), "predicted_size_fraction": normalized_size.cpu().tolist(),
                     "predicted_size_pixels": (normalized_size * normalized_size.new_tensor(size)).cpu().tolist(),
                     "target_size_fraction": [line["box"][2] * full_size[0] / size[0], line["box"][3] * full_size[1] / size[1]]})
    base = np.array(rows[0]["raw_box"])
    for row in rows:
        row["raw_box_max_abs_difference_from_full"] = float(np.max(np.abs(np.array(row["raw_box"]) - base)))
        row["heatmap_logit_difference_from_full"] = row["heatmap_logit"] - rows[0]["heatmap_logit"]
    return {"status": "completed", "line_id": line["id"], "method": "same page pixels at corresponding interior f4 cell; top-left crops stride16-aligned; fused FP32 eval; no image resizing or BN updates",
            "observations": rows,
            "limitation": "deterministic canvas sensitivity for one interior patch, not proof of training root cause or heatmap-collapse explanation"}


@torch.inference_mode()
def direct_probe(engine, dataset, index: int, oracle_cap: int, heatmap_probe: bool = False) -> tuple[dict, list[dict]]:
    image, target_unpadded = dataset[index]
    images, targets = collate_pages([(image, target_unpadded)])
    images = images.to(engine.device)
    target = targets[0]
    size = (images.shape[-1], images.shape[-2])
    model = engine.model
    f4, f8, f16 = model.backbone(images)
    raw = model.text_head(model.forward_text_features(f4, f8, f16))
    heatmap = raw["heatmap_logits"][0, 0].sigmoid().cpu()
    gh, gw = heatmap.shape
    gt_boxes = [line["box"] for line in target["lines"]]
    gradient_probe = None
    if heatmap_probe:
        # Only the detached head logits require gradients. No model parameter or
        # BN state changes. Sum dL/dlogit equals dL/dshared_heatmap_bias.
        with torch.inference_mode(False), torch.enable_grad():
            logits = raw["heatmap_logits"].detach().clone().requires_grad_(True)
            rendered, collisions, anchors = PrismOCRLoss._heatmap_targets([target], (gh, gw), logits.device, logits.dtype)
            probability = logits.sigmoid().clamp(1e-6, 1 - 1e-6)
            denominator = anchors.sum().clamp_min(1)
            positive = (-probability.log() * (1 - probability).square() * anchors).sum() / denominator
            negative = (-(1 - probability).log() * probability.square() * (1 - rendered).pow(4) * ~anchors).sum() / denominator
            gradient_probe = {"method": "FP32 eval logits detached; exact shared-head-bias derivative via sum(dLoss/dLogit); not optimizer update or train-mode batch",
                "anchor_count": int(anchors.sum()), "anchor_collisions": collisions, "denominator": int(denominator),
                "positive_loss": float(positive.detach()), "negative_loss": float(negative.detach()),
                "positive_dloss_dhead_bias": float(torch.autograd.grad(positive, logits, retain_graph=True)[0].sum()),
                "negative_dloss_dhead_bias": float(torch.autograd.grad(negative, logits)[0].sum())}
    detections, per_gt = {}, [{} for _ in gt_boxes]
    zero_proposals = zero_quality = zero_match = None
    for threshold in THRESHOLDS:
        proposals = TextLineHead.proposals(raw, max_proposals=model.text_proposals, score_threshold=threshold,
            model_contract_version=model.model_contract_version, image_size=size)
        boxes = proposals["boxes"][0].cpu().tolist()
        mask = proposals["mask"][0].cpu().numpy()
        quality = box_quality(gt_boxes, boxes, size)
        match = match_proposals(boxes, gt_boxes, mask, image_size=size, quality_matrix=quality)
        best = quality[:, mask].max(axis=1) if mask.any() else np.zeros(len(gt_boxes))
        detections[str(threshold)] = {"accepted_proposals": int(mask.sum()),
            "candidate_count_before_capacity": int(proposals["candidate_count"][0]),
            "capacity": model.text_proposals, "overflow": bool(proposals["capacity_overflow"][0]),
            "one_to_one_matched_iou_0.5": len(match.accepted_pairs),
            "gt_any_iou_0.5": int((best >= 0.5).sum()), "gt_any_iou_0.3": int((best >= 0.3).sum()),
            "best_iou": summary(best.tolist()),
            "scores": summary(proposals["scores"][0][proposals["mask"][0]].cpu().tolist()),
            "predicted_width_px": summary(b[2] * size[0] for b, active in zip(boxes, mask) if active),
            "predicted_height_px": summary(b[3] * size[1] for b, active in zip(boxes, mask) if active)}
        for i in range(len(gt_boxes)):
            per_gt[i][str(threshold)] = float(best[i])
        if threshold == 0.0:
            zero_proposals, zero_quality, zero_match = proposals, quality, match
    features = model.lat4(f4) + F.interpolate(model.lat8(f8), size=f4.shape[-2:], mode="bilinear", align_corners=False)
    # Evenly spaced original annotation indices, fixed across every mode.
    eligible = [i for i, line in enumerate(target["lines"]) if line.get("text")]
    selected = [eligible[i] for i in np.linspace(0, len(eligible) - 1, min(oracle_cap, len(eligible)), dtype=int)] if eligible else []
    anchor_embeddings, slot_embeddings, details = [], [], []
    accepted = dict(zero_match.accepted_pairs)
    proposal_boxes = zero_proposals["boxes"][0].cpu().numpy()
    valid = zero_proposals["candidate_mask"][0].cpu().numpy()
    for gt_index in selected:
        line, box = target["lines"][gt_index], gt_boxes[gt_index]
        x, y = min(gw - 1, max(0, int(box[0] * gw))), min(gh - 1, max(0, int(box[1] * gh)))
        slot = accepted.get(gt_index)
        method = "iou_accepted_one_to_one_slot_at_threshold_0"
        if slot is None and valid.any():
            distance = ((proposal_boxes[:, :2] - np.array(box[:2])) * np.array(size)) ** 2
            slot = int(np.where(valid, distance.sum(-1), np.inf).argmin())
            method = "nearest_valid_proposal_center_pixels_at_threshold_0_not_a_detection_match"
        if slot is None:
            raise ValueError("no valid proposal embedding available")
        anchor_embedding = raw["feature_map"][0, :, y, x]
        slot_embedding = zero_proposals["embeddings"][0, slot]
        anchor_embeddings.append(anchor_embedding)
        slot_embeddings.append(slot_embedding)
        encoded = engine.vocabulary.encode(line["text"])
        anchor_raw = raw["box_map"][0, :, y, x]
        anchor_size = TextLineHead.decode_sizes(anchor_raw[2:4],
            model_contract_version=model.model_contract_version, image_size=size).cpu().tolist()
        details.append({"gt_index": gt_index, "id": line["id"], "reference": line["text"], "box_padded_normalized": box,
            "gt_width_px": box[2] * size[0], "gt_height_px": box[3] * size[1],
            "oov_positions": list(encoded.oov_positions), "minimum_ctc_frames": minimum_ctc_frames(encoded.ids),
            "geometry_bucket": choose_inference_bucket(torch.tensor(box), size, model.strip_buckets,
                model_contract_version=model.model_contract_version),
            "label_derived_training_bucket_DIAGNOSTIC_NOT_DEPLOYABLE": model._training_bucket(line["text"]),
            "anchor_grid_xy": [x, y], "anchor_heatmap_score": float(heatmap[y, x]),
            "anchor_heatmap_3x3_max": float(heatmap[max(0, y-1):y+2, max(0, x-1):x+2].max()),
            "anchor_predicted_width_over_gt": anchor_size[0] / box[2], "anchor_predicted_height_over_gt": anchor_size[1] / box[3],
            "best_iou_by_threshold": per_gt[gt_index], "slot": slot, "slot_embedding_method": method,
            "slot_iou": float(zero_quality[gt_index, slot]),
            "slot_predicted_width_over_gt": float(proposal_boxes[slot, 2] / box[2]),
            "slot_predicted_height_over_gt": float(proposal_boxes[slot, 3] / box[3]),
            "anchor_embedding_norm": float(anchor_embedding.norm()), "slot_embedding_norm": float(slot_embedding.norm()),
            "embedding_cosine": float(F.cosine_similarity(anchor_embedding[None], slot_embedding[None])), "modes": {}})
    if selected:
        boxes = torch.tensor([gt_boxes[i] for i in selected], device=engine.device, dtype=features.dtype)
        refs = torch.tensor([[0, i] for i in range(len(selected))], device=engine.device)
        for name, embeddings in (("anchor", anchor_embeddings), ("slot", slot_embeddings)):
            cache = {}
            for bucket in model.strip_buckets:
                logits = model.recognizer.forward_bucket(features, boxes, torch.stack(embeddings), refs, bucket)
                cache[bucket] = decode_stats(logits, engine.vocabulary)
            for i, detail in enumerate(details):
                choices = {f"fixed_{bucket}": bucket for bucket in model.strip_buckets}
                choices["geometry"] = detail["geometry_bucket"]
                choices["label_derived_DIAGNOSTIC_NOT_DEPLOYABLE"] = detail["label_derived_training_bucket_DIAGNOSTIC_NOT_DEPLOYABLE"]
                for mode, bucket in choices.items():
                    value = dict(cache[bucket][i]) if bucket is not None else {"text": "", "failure": "no_supported_bucket"}
                    value["bucket"] = bucket
                    value["minframes_exceeds_bucket"] = bucket is not None and detail["minimum_ctc_frames"] > bucket
                    value["character_errors"] = edit_distance(detail["reference"], value["text"])
                    value["cer"] = value["character_errors"] / len(detail["reference"])
                    value["exact"] = detail["reference"] == value["text"]
                    value["blank_output"] = not value["text"] and not value.get("failure")
                    detail["modes"][f"{name}/{mode}"] = value
    return {"scaled_size": list(target_unpadded["_original_size"]), "padded_size": list(size),
            "ground_truth_lines": len(gt_boxes), "oracle_lines": len(selected), "direct_thresholds": detections,
            "heatmap_gradient_probe": gradient_probe,
            "heatmap_page": summary(heatmap.flatten().tolist()),
            "gt_width_px": summary(b[2] * size[0] for b in gt_boxes),
            "gt_height_px": summary(b[3] * size[1] for b in gt_boxes)}, details


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="NEW directory; defaults to /tmp/prismocr-v4-diagnosis-*")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--pages-per-stratum", type=int, default=2)
    parser.add_argument("--oracle-lines-per-page", type=int, default=16)
    parser.add_argument("--max-page-side", type=int, default=1600)
    parser.add_argument("--timeout-seconds", type=int, default=540)
    parser.add_argument("--heatmap-gradient-probe", action="store_true", help="detached logits probe on first text and first region-only page")
    parser.add_argument("--canvas-probe", action="store_true", help="same interior raw-head cell on full/1024/512 canvases for first text page")
    args = parser.parse_args(argv)
    if not (1 <= args.limit <= 24 and 1 <= args.pages_per_stratum <= 4 and 1 <= args.oracle_lines_per_page <= 16):
        parser.error("bounded limits: pages <=24, per stratum <=4, oracle per page <=16; all positive")
    if not (16 <= args.max_page_side <= 1600 and 1 <= args.timeout_seconds <= 600):
        parser.error("page side must be 16..1600 and timeout 1..600 seconds")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    if args.device.startswith("cuda"):
        cuda_device = torch.device(args.device)
        torch.cuda.set_per_process_memory_fraction(0.5, cuda_device.index if cuda_device.index is not None else torch.cuda.current_device())
        torch.cuda.reset_peak_memory_stats(args.device)
    output = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="prismocr-v4-diagnosis-"))
    if args.output:
        output.mkdir(parents=True, exist_ok=False)
    os.chmod(output, 0o700)
    (output / "diagnose_ocr.py").write_bytes(Path(__file__).read_bytes())
    def timeout_handler(signum, frame):
        raise TimeoutError("bounded diagnostic timeout exceeded")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout_seconds)
    started = time.monotonic()
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "running", "argv": sys.argv,
              "artifact": str(args.artifact.resolve()), "manifest": str(args.manifest.resolve()),
              "artifact_hashes": {p.name: sha256(p) for p in sorted(args.artifact.iterdir()) if p.name in
                                  {"model.pt", "model-config.json", "vocab-v4.json", "manifest.json", "training-run.json"}},
              "manifest_sha256": sha256(args.manifest), "script_sha256": sha256(Path(__file__)),
              "model_config": json.loads((args.artifact / "model-config.json").read_text()),
              "artifact_manifest": json.loads((args.artifact / "manifest.json").read_text()),
              "training_run": json.loads((args.artifact / "training-run.json").read_text()) if (args.artifact / "training-run.json").is_file() else None,
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": args.device,
                  "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
                  "cpu_threads": 4, "cuda_memory_fraction_cap": 0.5},
              "method": {"selection": "first eligible rows in each named stratum, stratum order then --limit; no score selection",
                         "oracle_selection": "evenly spaced nonempty GT annotation indices, capped per page, identical across modes",
                         "scaling": args.max_page_side, "padding": "per-page zero right/bottom to stride16; target center/size rescaled to padded axes",
                         "public": "unmodified PrismOCR.predict, default thresholds, page_resolution set; both default tiling True and False",
                         "model_precision": "FP32 eval; public and direct/oracle all use engine.model, API fused_for_inference copy of safely loaded checkpoint; no AMP, no source or weights changed", 
                         "direct": "UNTILED backbone/text head, capacity unchanged; geometry-only one-to-one and any-proposal rotated IoU",
                         "oracle": "full page lat4(f4)+bilinear_upsampled_lat8(f8); forward_bucket on GT boxes, no crop resizing",
                         "label_bucket_warning": "uses GT text to choose training bucket; diagnostic only, NOT deployable",
                         "embedding": "compare raw text-head feature at target anchor vs IoU-accepted slot, else nearest valid proposal center at threshold0",
                         "metrics": "raw Unicode codepoint CER; no normalization; public CER includes unmatched prediction insertions only on text-supervised pages (including certified blanks); selected DocLayNet pages are text-unlabelled, not certified blank, and excluded from text-error/reference aggregation; emission diagnostics retained for all pages",
                         "gradient_probe": "optional first text and first region-only page; detached eval logits only, exact positive/negative bias derivatives; not training-batch replay or causal attribution", 
                         "causality_warning": "observational probes do not identify training root cause; selected subset not a release evaluation"},
              "pages": []}
    all_lines = []
    try:
        chosen = select_manifest(args.manifest.resolve(), output / "selected-manifest.jsonl", args.pages_per_stratum, args.limit)
        report["selection"] = [{**r["diagnosis_selection"], "page_id": r["annotation"]["page_id"],
                                "image_sha256": r["annotation"]["image"]["sha256"]} for r in chosen]
        report["selected_manifest_sha256"] = sha256(output / "selected-manifest.jsonl")
        write_json(output / "summary.json", report)
        engine = PrismOCR.load(args.artifact, device=args.device)
        dataset = PrismPageDataset(output / "selected-manifest.jsonl", max_page_side=args.max_page_side)
        probed_types = set()
        with (output / "line-details.jsonl").open("w", encoding="utf-8") as line_stream:
            for index, record in enumerate(dataset.records):
                page_started = time.monotonic()
                annotation = record["_annotation"]
                page = {"index": index, "page_id": annotation["page_id"], "stratum": chosen[index]["diagnosis_selection"]["stratum"]}
                text_supervised = chosen[index]["diagnosis_selection"]["text_supervision"] == "annotated"
                public, result = public_probe(engine, record["_image_path"], annotation, args.max_page_side,
                                              text_supervised=text_supervised)
                write_json(output / f"public-page-{index:02d}.json", result)
                page["public"] = public
                untiled, untiled_result = public_probe(engine, record["_image_path"], annotation, args.max_page_side, allow_tiling=False,
                                                       text_supervised=text_supervised)
                write_json(output / f"public-untiled-page-{index:02d}.json", untiled_result)
                page["public_untiled"] = untiled
                page_type = "text" if annotation.get("lines") else "region_only"
                probe = args.heatmap_gradient_probe and page_type not in probed_types
                direct, details = direct_probe(engine, dataset, index, args.oracle_lines_per_page, heatmap_probe=probe)
                if probe:
                    probed_types.add(page_type)
                page.update(direct)
                if args.canvas_probe and index == 0:
                    report["canvas_probe"] = canvas_probe(engine, dataset, index)
                for detail in details:
                    detail.update(page_id=page["page_id"], stratum=page["stratum"])
                    line_stream.write(json.dumps(detail, ensure_ascii=False, allow_nan=False) + "\n")
                line_stream.flush()
                all_lines.extend(details)
                page["elapsed_seconds"] = time.monotonic() - page_started
                report["pages"].append(page)
                write_json(output / "summary.json", report)
                print(json.dumps({"index": index, "stratum": page["stratum"], "gt": page["ground_truth_lines"],
                                  "public_detections": public["detections"], "matched": public["matched_iou_0.5"],
                                  "oracle_lines": len(details), "seconds": round(page["elapsed_seconds"], 2)}), flush=True)
        modes = defaultdict(list)
        by_stratum = defaultdict(lambda: defaultdict(list))
        for detail in all_lines:
            for mode, value in detail["modes"].items():
                row = {**value, "reference": detail["reference"]}
                modes[mode].append(row)
                by_stratum[detail["stratum"]][mode].append(row)
        report["oracle_summary"] = {mode: aggregate_recognition(rows) for mode, rows in modes.items()}
        report["oracle_by_stratum"] = {s: {m: aggregate_recognition(rows) for m, rows in group.items()} for s, group in by_stratum.items()}
        report["oracle_gt_summary"] = {"lines": len(all_lines), "oov_characters": sum(len(r["oov_positions"]) for r in all_lines),
            "minimum_ctc_frames": summary(r["minimum_ctc_frames"] for r in all_lines),
            "anchor_heatmap_scores": summary(r["anchor_heatmap_score"] for r in all_lines),
            "anchor_predicted_width_over_gt": summary(r["anchor_predicted_width_over_gt"] for r in all_lines),
            "anchor_predicted_height_over_gt": summary(r["anchor_predicted_height_over_gt"] for r in all_lines),
            "slot_embedding_norm": summary(r["slot_embedding_norm"] for r in all_lines),
            "anchor_embedding_norm": summary(r["anchor_embedding_norm"] for r in all_lines)}
        report["public_totals"] = {key: aggregate_public([p[key] for p in report["pages"]])
                                   for key in ("public", "public_untiled")}
        report["direct_totals"] = {}
        for threshold in THRESHOLDS:
            rows = [p["direct_thresholds"][str(threshold)] for p in report["pages"]]
            counts = {f: sum(r[f] for r in rows) for f in ("accepted_proposals", "one_to_one_matched_iou_0.5", "gt_any_iou_0.5", "gt_any_iou_0.3")}
            counts["ground_truth_lines"] = sum(p["ground_truth_lines"] for p in report["pages"])
            counts["recall_iou_0.5"] = counts["one_to_one_matched_iou_0.5"] / counts["ground_truth_lines"] if counts["ground_truth_lines"] else None
            report["direct_totals"][str(threshold)] = counts
        report["limitations"] = [
            "Small fixed manifest-order strata are diagnostic, not random-sample quality estimates or release evidence.",
            "Oracle boxes and anchor embeddings use GT geometry; label-derived buckets additionally use GT text and are not deployable.",
            "Nearest slot embeddings may not be IoU matches; identities, method, IoU and nonzero norms are retained per line.",
            "Low thresholds retain original top128 capacity; candidate overflow is reported, not resolved or tuned.",
            "Public blank rate denominator is emitted detections, not GT lines; no detections is undefined, not evidence of blank CTC.",
            "Selected region-only DocLayNet pages lack text supervision, not certified blank; their emissions remain diagnostic but are excluded from text-error/reference aggregation. Supervised blank pages still incur insertion errors.",
            "FP32 fused eval probes do not reproduce AMP training, mixed-page batch padding, optimizer history or annotation exhaustiveness.",
            "Differences between detector thresholds, oracle geometry, embeddings or buckets are observational, not causal root-cause proof.",
            "No baseline recognition, annotation visual QA, production fix, retraining or full evaluation is performed by this CLI."]
        report["status"] = "completed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic() - started
        report["timing_note"] = "wall-clock bookkeeping only; GPU timings are not explicitly synchronized and are NOT latency/performance evidence"
        if args.device.startswith("cuda"):
            report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
        write_json(output / "summary.json", report)
        print(f"evidence={output} status={report['status']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
