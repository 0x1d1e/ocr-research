"""Diagnose PrismOCR text-detection boxes on real corpus pages.

Runs the trained artifact's text head at a low score threshold and reports, for
ground-truth lines, the heatmap score at the anchor and the geometry of the
nearest proposals. This is a diagnostic, not a release metric.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

from prismocr.api import PrismOCR
from prismocr.data.dataset import PrismPageDataset, collate_pages
from prismocr.geometry import rotated_iou
from prismocr.model.network import _target_box


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pages", type=int, default=6)
    parser.add_argument("--max-page-side", type=int, default=1600)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    engine = PrismOCR.load(args.artifact, device=args.device)
    model = engine.model
    dataset = PrismPageDataset(Path(args.manifest), max_page_side=args.max_page_side)

    chosen = []
    for index in range(len(dataset)):
        if dataset.records[index]["_annotation"].get("lines"):
            chosen.append(index)
        if len(chosen) >= args.pages:
            break
    items = [dataset[index] for index in chosen]
    images, targets = collate_pages(items)
    images = images.to(args.device)

    with torch.no_grad():
        outputs = model(images, targets, teacher_force_ratio=0.0, text_threshold=0.0)
    text = outputs["text"]
    heatmap = torch.sigmoid(outputs["text_heatmap_logits"]).float().cpu()
    boxes = text["boxes"].float().cpu()
    scores = text["scores"].float().cpu()
    valid = text["candidate_mask"].cpu()

    anchor_best, anchor_mean, size_ratios, ious, top_scores = [], [], [], [], []
    for batch_index, target in enumerate(targets):
        height, width = heatmap.shape[-2:]
        for line in target.get("lines", ()):
            gt = _target_box(line, torch.device("cpu"), torch.float32)
            grid_x = min(width - 1, max(0, int(float(gt[0]) * width)))
            grid_y = min(height - 1, max(0, int(float(gt[1]) * height)))
            window = heatmap[batch_index, 0, max(0, grid_y - 1):grid_y + 2, max(0, grid_x - 1):grid_x + 2]
            anchor_best.append(float(window.max()))
            anchor_mean.append(float(heatmap[batch_index, 0].mean()))
            candidates = [
                (float(scores[batch_index, i]), i)
                for i in range(scores.shape[1])
                if bool(valid[batch_index, i])
            ]
            if not candidates:
                continue
            top_score, top_index = max(candidates)
            top_scores.append(top_score)
            ious.append(rotated_iou(boxes[batch_index, top_index].tolist(), gt.tolist(), (images.shape[-1], images.shape[-2])))
            size_ratios.append(
                (
                    float(boxes[batch_index, top_index, 3]) / max(float(gt[3]), 1e-6),
                    float(boxes[batch_index, top_index, 2]) / max(float(gt[2]), 1e-6),
                )
            )

    def summary(values):
        if not values:
            return "none"
        return (
            f"min={min(values):.4f} median={statistics.median(values):.4f} "
            f"max={max(values):.4f}"
        )

    print(f"pages={len(items)} ground_truth_lines={len(anchor_best)}")
    print(f"heatmap_at_anchor_best: {summary(anchor_best)}")
    print(f"heatmap_page_mean: {summary(anchor_mean)}")
    print(f"top_proposal_score: {summary(top_scores)}")
    print(f"top_proposal_iou_vs_gt_box: {summary(ious)}")
    if size_ratios:
        heights = [pair[0] for pair in size_ratios]
        widths = [pair[1] for pair in size_ratios]
        print(f"predicted_height/gt_height: {summary(heights)}")
        print(f"predicted_width/gt_width: {summary(widths)}")
    print(f"accepted_at_0.55: {int((scores >= 0.55).sum())} of {int(valid.sum())} candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
