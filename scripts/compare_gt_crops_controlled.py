"""Bounded validation-only PaddleOCR vs PrismOCR GT-geometry recognition comparison.

Paddle receives shared RGB perspective crops. PrismOCR has no crop encoder; it
receives the same GT quadrilateral as geometry on page feature maps. The routes
are therefore explicitly not pixel-identical. Keep output local/private.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_paddleocr import (  # noqa: E402
    DEFAULT_ARTIFACT, DEFAULT_MANIFEST, DEFAULT_REC, aggregate_recognition,
    crop_polygon, file_sha256, language_bucket, load_records, oov_count,
    paddle_text, prism_oracle_recognize,
)
from prismocr.api import PrismOCR  # noqa: E402
from prismocr.data.orientation import exif_oriented  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--rec-model", type=Path, default=DEFAULT_REC)
    parser.add_argument("--pages-per-source", type=int, default=12)
    parser.add_argument("--max-lines-per-page", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path.home() / "data/paddleocr-prismocr-controlled-48.json")
    parser.add_argument("--device", default="gpu:0")
    args = parser.parse_args(argv)
    if args.pages_per_source < 1 or args.max_lines_per_page < 1:
        parser.error("page and line limits must be positive")
    for label, path in (("manifest", args.manifest), ("Prism artifact", args.artifact), ("Paddle recognition model", args.rec_model)):
        if not path.exists():
            parser.error(f"{label} does not exist: {path}")

    records = load_records(args.manifest.resolve(), args.pages_per_source * 4)
    counts = {}
    for record in records:
        counts[record["source_stratum"]] = counts.get(record["source_stratum"], 0) + 1
    if len(counts) != 4 or any(n != args.pages_per_source for n in counts.values()):
        parser.error(f"could not select requested pages from each source stratum: {counts}")

    import paddle
    import yaml
    from paddleocr import TextRecognition
    paddle.set_device(args.device)
    rec = TextRecognition(model_name="PP-OCRv6_tiny_rec", model_dir=str(args.rec_model.resolve()), device=args.device)
    prism = PrismOCR.load(args.artifact, device="cuda" if args.device.startswith("gpu") else "cpu")
    paddle_cfg = yaml.safe_load((args.rec_model / "inference.yml").read_text(encoding="utf-8"))
    paddle_vocab = set(paddle_cfg["PostProcess"]["character_dict"])
    prism_vocab = set(prism.vocabulary.tokens[2:])

    rows, page_info, prism_frame_stats = [], [], []
    started = time.monotonic()
    for page_idx, record in enumerate(records, 1):
        ann = record["annotation"]
        with Image.open(record["image_path"]) as src:
            image = exif_oriented(src).convert("RGB")
            rgb = np.asarray(image)
        if (image.width, image.height) != (ann["image"]["width"], ann["image"]["height"]):
            raise ValueError(f"image/annotation dimensions disagree for {ann['page_id']}")
        selected_lines = ann["lines"][:args.max_lines_per_page]
        page_info.append({"page_id": ann["page_id"], "source": record["source_stratum"],
                          "manifest_row": record["manifest_row"], "labeled_lines": len(ann["lines"]),
                          "evaluated_lines": len(selected_lines)})
        for line_i, line in enumerate(selected_lines):
            ref = str(line.get("text", ""))
            row = {"page_id": ann["page_id"], "line_id": line.get("id", str(line_i)),
                   "source": record["source_stratum"], "language": language_bucket(ref),
                   "reference": ref, "paddle_oov": oov_count(ref, paddle_vocab),
                   "prism_oov": oov_count(ref, prism_vocab)}
            try:
                crop = crop_polygon(rgb, line["polygon_px"])
                result = rec.predict(input=crop)[0]
                texts = paddle_text(result)
                row["paddle_hypothesis"] = texts[0] if texts else ""
            except Exception as exc:
                row["paddle_hypothesis"] = ""
                row["paddle_failure"] = f"{type(exc).__name__}: {exc}"
            try:
                hypothesis, frame_stats = prism_oracle_recognize(prism, image, line["polygon_px"], return_frame_stats=True)
                row["prism_hypothesis"] = hypothesis
                row["prism_ctc_frames"] = frame_stats["frames"]
                row["prism_greedy_blank_frames"] = frame_stats["greedy_blank_frames"]
                row["prism_mean_blank_posterior"] = frame_stats["mean_blank_posterior"]
                prism_frame_stats.append(frame_stats)
            except Exception as exc:
                row["prism_hypothesis"] = ""
                row["prism_failure"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        print(json.dumps({"page": page_idx, "page_id": ann["page_id"], "source": record["source_stratum"],
                          "lines": len(selected_lines)}, ensure_ascii=False), flush=True)

    page_sources = {p["page_id"]: p["source"] for p in page_info}
    def summarize(subset, model):
        result = aggregate_recognition([{"reference": r["reference"], "hypothesis": r.get(model + "_hypothesis", ""),
            "oov_reference_characters": r[model + "_oov"], "failure": r.get(model + "_failure")} for r in subset])
        blanks = sum(not (r.get(model + "_hypothesis") or "").strip() for r in subset)
        result["blank_lines"] = blanks
        result["blank_line_rate"] = blanks / len(subset) if subset else None
        return result
    summary = {model: {"overall": summarize(rows, model),
        "by_source": {s: summarize([r for r in rows if r["source"] == s], model) for s in sorted(counts)},
        "by_language": {l: summarize([r for r in rows if r["language"] == l], model)
                        for l in ("khmer", "english", "other_or_empty")}} for model in ("paddle", "prism")}
    total_frames = sum(x["frames"] for x in prism_frame_stats)
    total_blank_frames = sum(x["greedy_blank_frames"] for x in prism_frame_stats)
    frame_stats = {"available": True, "sequence_length_per_line": max(prism.model.strip_buckets),
                   "lines_with_logits": len(prism_frame_stats), "total_frames": total_frames,
                   "greedy_blank_frames": total_blank_frames,
                   "greedy_blank_frame_rate": total_blank_frames / total_frames if total_frames else None,
                   "mean_per_line_blank_posterior": float(np.mean([x["mean_blank_posterior"] for x in prism_frame_stats])) if prism_frame_stats else None}
    # The shared-crop invariant is established at crop creation, then Paddle transforms it internally.
    # Prism takes page pixels plus GT geometry through its trained feature path.
    selection = [{"page_id": p["page_id"], "source": p["source"], "manifest_row": p["manifest_row"]} for p in page_info]
    selection_hash = hashlib.sha256(json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output = {"status": "completed", "protocol": "GT-quadrilateral geometry-controlled recognition; same-crop Prism route unavailable",
        "manifest": str(args.manifest.resolve()), "manifest_sha256": file_sha256(args.manifest.resolve()),
        "validation_only": True, "test_split_opened": False, "device": args.device,
        "selection": {"method": "manifest-order round-robin; equal page count from four source strata",
                      "pages_per_source": args.pages_per_source, "page_ids": selection,
                      "selection_sha256": selection_hash, "source_page_counts": counts},
        "line_cap_per_page": args.max_lines_per_page, "evaluated_lines": len(rows),
        "models": {"prism_artifact": str(args.artifact.resolve()), "paddle_recognition": str(args.rec_model.resolve()),
                   "paddleocr_version": __import__("paddleocr").__version__},
        "routes": {"paddle": {"input": "GT quadrilateral perspective-warped RGB crop, then Paddle RecResizeImg to 3x48x320",
                                 "crop": "cv2.warpPerspective, INTER_CUBIC, BORDER_REPLICATE"},
                   "prism": {"input": "full page resized by Prism page_resolution (if configured), padded to multiple of 16; GT polygon converted to normalized rotated box; backbone/text feature map and lat4+lat8 strip sampled; text-head anchor-cell embedding added",
                                "same_crop_route": "unavailable: model has no RGB crop encoder; trained recognizer consumes page feature strips and text-head node embedding"}},
        "pixel_identity": {"gt_crop_rgb_sha256_recorded": False,
                           "explanation": "Paddle receives the perspective crop pixels. Prism does not accept that crop; it independently reads the page and samples page features using the GT geometry. Pixel-identical recognizer inputs cannot be claimed."},
        "metrics": summary, "ctc_frame_blank_stats": frame_stats, "pages": page_info, "lines": rows,
        "limitations": ["This controls GT line geometry and validation sample, not recognizer input representation.",
                        "Paddle model preprocessing follows its inference.yml RecResizeImg shape 3x48x320; Prism uses page RGB normalization/resize, padding and feature-strip sampling.",
                        "Prism OOV vocabulary and blank outputs are reported descriptively; no crop encoder was added or trained.",
                        "Outputs contain private corpus IDs and transcripts; keep local."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"summary={args.output.resolve()} lines={len(rows)} elapsed_seconds={time.monotonic()-started:.1f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
