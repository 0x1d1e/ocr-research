"""Diagnose PrismOCR heatmap supervision equilibrium on a real training batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from prismocr.data.dataset import PrismPageDataset, collate_pages
from prismocr.data.vocabulary import Vocabulary
from prismocr.losses import PrismOCRLoss
from prismocr.model.network import PrismOCRModel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--max-page-side", type=int, default=1600)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    artifact = Path(args.artifact)
    config = json.loads((artifact / "model-config.json").read_text())
    model = PrismOCRModel(
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
    state = torch.load(artifact / "model.pt", map_location="cpu", weights_only=True)["model_state"]
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval()

    dataset = PrismPageDataset(Path(args.manifest), max_page_side=args.max_page_side)
    items = [dataset[index] for index in range(args.pages)]
    images, targets = collate_pages(items)
    images = images.to(args.device)

    outputs = model(images, targets, teacher_force_ratio=0.0)
    criterion = PrismOCRLoss(Vocabulary.load(artifact / "vocab-v4.json"))
    losses = criterion(outputs, targets)
    logits = outputs["text_heatmap_logits"]
    rendered, collisions, anchors = criterion._heatmap_targets(
        targets, logits.shape[-2:], logits.device, logits.dtype
    )
    probability = logits.float().sigmoid()
    anchor_values = probability[anchors.to(probability.device)]
    rendered = rendered.float()
    positive_loss = (
        -torch.log(probability.clamp(1e-6, 1 - 1e-6)) * (1 - probability).pow(2) * anchors.to(probability.dtype)
    ).sum() / anchors.sum()
    negative_loss = (
        -torch.log((1 - probability).clamp(1e-6, 1))
        * probability.pow(2)
        * (1 - rendered).pow(4)
        * (~anchors).to(probability.dtype)
    ).sum() / anchors.sum()
    print(f"anchors={int(anchors.sum())} collisions={collisions} grid={tuple(logits.shape[-2:])}")
    print(f"p_at_anchor: mean={anchor_values.mean():.4f} median={anchor_values.median():.4f} max={anchor_values.max():.4f}")
    print(f"p_background_mean={probability[~anchors.to(probability.device)].mean():.4f}")
    print(f"positive_loss={float(positive_loss):.4f} negative_loss={float(negative_loss):.4f} total_heatmap={float(losses['heatmap']):.4f}")

    bias = model.text_head.heatmap.bias
    for label, term in (("full_heatmap", losses["heatmap"]), ("positive_only", positive_loss), ("negative_only", negative_loss)):
        model.zero_grad(set_to_none=True)
        term.backward(retain_graph=True)
        print(f"grad[{label}] heatmap.bias={float(bias.grad):.6f} weight_norm={float(model.text_head.heatmap.weight.grad.norm()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
