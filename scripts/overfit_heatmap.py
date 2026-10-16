"""Overfit a single PrismOCR page to test whether the heatmap can reach confidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from prismocr.data.dataset import PrismPageDataset, collate_pages
from prismocr.data.vocabulary import Vocabulary
from prismocr.losses import PrismOCRLoss
from prismocr.model.network import PrismOCRModel, validated_model_state


def build_model(config: dict, vocabulary: Vocabulary | None = None) -> PrismOCRModel:
    return PrismOCRModel(
        model_contract_version=config.get("model_contract_version", 1),
        vocabulary=vocabulary,
        dim=config["dim"],
        text_proposals=config["text_proposals"],
        region_queries=config["region_queries"],
        vocab_size=config["vocab_size"],
        strip_height=config["strip_height"],
        recognizer=config["recognizer"],
        text_threshold=config["text_threshold"],
        region_threshold=config["region_threshold"],
        max_table_rows=config["max_table_rows"],
        max_table_cols=config["max_table_cols"],
        max_cell_span=config["max_cell_span"],
        strip_buckets=config["strip_buckets"],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--page-index", type=int, default=0)
    parser.add_argument("--max-page-side", type=int, default=256)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--init", choices=("random", "checkpoint"), default="random")
    parser.add_argument("--box-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    artifact = Path(args.artifact)
    config = json.loads((artifact / "model-config.json").read_text())
    torch.manual_seed(0)
    vocabulary = Vocabulary.load(artifact / "vocab-v4.json")
    model = build_model(config, vocabulary)
    if args.init == "checkpoint":
        checkpoint = torch.load(artifact / "model.pt", map_location="cpu", weights_only=True)
        state = validated_model_state(checkpoint, config)
        model.load_state_dict(state, strict=True)
    model.to(args.device).train()

    dataset = PrismPageDataset(Path(args.manifest), max_page_side=args.max_page_side)
    index = args.page_index
    while not dataset.records[index]["_annotation"].get("lines"):
        index += 1
    images, targets = collate_pages([dataset[index]])
    images = images.to(args.device)

    criterion = PrismOCRLoss(
        vocabulary, loss_weights={"box": args.box_weight}
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images, targets, teacher_force_ratio=0.0)
        losses = criterion(outputs, targets)
        losses["total"].backward()
        optimizer.step()
        if step % 50 == 0 or step == args.steps - 1:
            logits = outputs["text_heatmap_logits"].detach()
            _, _, anchors = criterion._heatmap_targets(
                targets, logits.shape[-2:], logits.device, logits.dtype
            )
            probability = logits.float().sigmoid()
            anchor_values = probability[anchors.to(probability.device)]
            print(
                f"step={step:4d} heatmap={float(losses['heatmap']):.4f} box={float(losses['box']):.4f} "
                f"total={float(losses['total']):.4f} "
                f"p_anchor_mean={float(anchor_values.mean()):.4f} p_anchor_max={float(anchor_values.max()):.4f} "
                f"p_bg={float(probability[~anchors.to(probability.device)].mean()):.5f} "
                f"detect_recall={losses['detection_recall_text']:.3f}"
            )
    if index != args.page_index:
        print(f"page_index {args.page_index} has no lines; using {index}")
    match = outputs["assignments"]["text"][0]
    from prismocr.geometry import rotated_iou

    boxes = outputs["text"]["boxes"][0].detach().float().cpu()
    lines = list(targets[0].get("lines", ()))
    image_size = (images.shape[-1], images.shape[-2])
    print(f"image_size={image_size} text_proposals={boxes.shape[0]}")
    for pair_index, (gt_i, pred_i) in enumerate(match.assignment_pairs[:4]):
        gt = torch.as_tensor(lines[gt_i]["box"], dtype=torch.float32)
        pred = boxes[pred_i]
        iou = rotated_iou(pred.tolist(), gt.tolist(), image_size)
        print(f"pair{pair_index} iou={iou:.3f} pred={[round(float(v), 4) for v in pred]} gt={[round(float(v), 4) for v in gt]}")
    qualities = list(match.candidate_quality)
    print(
        f"gt={match.gt_count} assigned={len(match.assigned_gt_idx)} accepted={len(match.accepted_gt_idx)} "
        f"quality_mean={sum(qualities) / max(len(qualities), 1):.4f} "
        f"quality_max={max(qualities) if qualities else 0.0:.4f} "
        f"quality_ge_0.5={sum(q >= 0.5 for q in qualities)}/{len(qualities)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
