"""Reproducible manifest-driven research training entry point.

Validation checkpoints are selected lexicographically: maximize line-detection
recall, then minimize CER. Exact metric ties keep the earliest checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import tomllib
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from prismocr.api import save_artifact
from prismocr.data.dataset import PrismPageDataset, collate_pages, dataset_content_sha256
from prismocr.data.schema import resolve_line_supervision
from prismocr.data.vocabulary import Vocabulary
from prismocr.eval import evaluate
from prismocr.geometry import rotated_iou
from prismocr.losses import PrismOCRLoss
from prismocr.model.heads import validate_model_contract_version
from prismocr.model.network import PrismOCRModel
from prismocr.model.recognizer import STRIP_BUCKETS, validate_strip_buckets


# Format 5 binds the repaired objective and explicit model-contract settings.
# Old optimizer states must not silently resume under changed semantics.
CHECKPOINT_FORMAT_VERSION = 5
SELECTION_POLICY = (
    "minimize inference_failure_count, then maximize line_detection_recall_at_iou, then "
    "minimize cer_on_matched_detections_with_misses_as_empty; exact ties keep the earliest epoch"
)


def _config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _dataset_sha256(dataset: PrismPageDataset) -> str:
    """Fingerprint canonical annotations and exact image bytes for safe resume."""
    return dataset_content_sha256(dataset)


def _assert_split_disjoint(train: PrismPageDataset, validation: PrismPageDataset) -> None:
    """Reject document, template-family, and exact-image leakage across splits."""
    train_docs = {record["_annotation"]["document_id"] for record in train.records}
    validation_docs = {record["_annotation"]["document_id"] for record in validation.records}
    overlap = train_docs & validation_docs
    if overlap:
        raise ValueError(f"training and validation manifests share document IDs: {sorted(overlap)[:8]}")

    def families(dataset: PrismPageDataset) -> set[str]:
        return {
            annotation["provenance"]["template_family_id"]
            for annotation in (record["_annotation"] for record in dataset.records)
            if annotation["provenance"].get("template_family_id")
        }

    family_overlap = families(train) & families(validation)
    if family_overlap:
        raise ValueError(f"training and validation manifests share template families: {sorted(family_overlap)[:8]}")
    image_overlap = {record["_image_sha256"] for record in train.records} & {
        record["_image_sha256"] for record in validation.records
    }
    if image_overlap:
        raise ValueError("training and validation manifests contain identical image bytes")


def _resolved_settings(
    model_config: dict[str, Any],
    train_config: dict[str, Any],
    vocabulary: Vocabulary,
    *,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    """Return the effective settings that must remain stable across resume."""
    return {
        "model": {
            "model_contract_version": validate_model_contract_version(model_config.get("model_contract_version", 1)),
            "dim": int(model_config.get("dim", 160)),
            "text_proposals": int(model_config.get("text_proposals", 128)),
            "region_queries": int(model_config.get("region_queries", 64)),
            "vocab_size": len(vocabulary),
            "strip_height": int(model_config.get("strip_height", 5)),
            "strip_buckets": list(validate_strip_buckets(model_config.get("strip_buckets", STRIP_BUCKETS))),
            "recognizer": str(model_config.get("recognizer", "transformer")),
            "text_threshold": float(model_config.get("text_threshold", 0.55)),
            "region_threshold": float(model_config.get("region_threshold", 0.5)),
            "max_table_rows": int(model_config.get("max_table_rows", 128)),
            "max_table_cols": int(model_config.get("max_table_cols", 128)),
            "max_cell_span": int(model_config.get("max_cell_span", 32)),
            "fused_text_head": bool(model_config.get("fused_text_head", False)),
        },
        "training": {
            # Raw annotation hashes remain compatible; optimizer/loss semantics do not.
            "objective_contract": "lines-availability-v1-page-balanced-focal",
            "seed": int(train_config.get("seed", 42)),
            "teacher_forcing_start": float(train_config.get("teacher_forcing_start", 1.0)),
            "teacher_forcing_end": float(train_config.get("teacher_forcing_end", 0.0)),
            "strict_ctc_length": bool(train_config.get("strict_ctc_length", True)),
            "batch_size": batch_size,
            "num_workers": num_workers,
            "max_page_side": train_config.get("max_page_side"),
            "learning_rate": float(train_config.get("learning_rate", 1e-4)),
            "lr_warmup_steps": int(train_config.get("lr_warmup_steps", train_config.get("warmup_steps", 0))),
            "weight_decay": float(train_config.get("weight_decay", 1e-4)),
            "max_grad_norm": float(train_config.get("max_grad_norm", 5.0)),
            "gradient_accumulation_steps": int(train_config.get("gradient_accumulation_steps", 1)),
            "amp": bool(train_config.get("amp", False)),
            "amp_dtype": str(train_config.get("amp_dtype", "float16")),
            "activation_checkpointing": bool(train_config.get("activation_checkpointing", False)),
            "loss_weights": {
                str(name): float(value)
                for name, value in sorted(train_config.get("loss_weights", {}).items())
            },
        },
    }


def _numpy_rng_state() -> dict[str, Any]:
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "name": name,
        "keys": keys.tolist(),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _capture_rng(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": _numpy_rng_state(),
        "torch": torch.get_rng_state().cpu(),
        "cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        "data_loader": generator.get_state().cpu(),
    }


def _restore_rng(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(tuple(state["python"]))
    numpy_state = state["numpy"]
    np.random.set_state((
        str(numpy_state["name"]),
        np.asarray(numpy_state["keys"], dtype=np.uint32),
        int(numpy_state["position"]),
        int(numpy_state["has_gauss"]),
        float(numpy_state["cached_gaussian"]),
    ))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    generator.set_state(state["data_loader"].cpu())


def _cpu_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _atomic_checkpoint_save(state: dict[str, Any], path: Path) -> None:
    """Write a complete checkpoint beside its destination, then atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # The file replacement is atomic even on filesystems that cannot fsync directories.
            pass
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _checkpoint_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    scheduler: Any = None,
    epoch: int,
    step: int,
    generator: torch.Generator,
    compatibility: dict[str, Any],
    best_model_state: dict[str, torch.Tensor] | None,
    best_metrics: dict[str, float] | None,
    best_epoch: int | None,
    log_offset: int,
    scaler_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state": _cpu_model_state(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler_state,
        "epoch": epoch,
        "step": step,
        "rng_state": _capture_rng(generator),
        "compatibility": compatibility,
        "best_model_state": best_model_state,
        "best_metrics": best_metrics,
        "best_epoch": best_epoch,
        "log_offset": log_offset,
        "selection_policy": SELECTION_POLICY if compatibility["validation_examples"] is not None else None,
    }


def _validate_resume(
    checkpoint: dict[str, Any],
    *,
    compatibility: dict[str, Any],
    requested_epochs: int,
) -> None:
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported training checkpoint format")
    saved = checkpoint.get("compatibility", {})
    if saved.get("vocabulary_sha256") != compatibility["vocabulary_sha256"]:
        raise ValueError("resume checkpoint vocabulary is incompatible")
    if saved.get("settings") != compatibility["settings"]:
        raise ValueError("resume checkpoint configuration is incompatible")
    if saved.get("train_examples") != compatibility["train_examples"]:
        raise ValueError("resume checkpoint training data cardinality is incompatible")
    if saved.get("validation_examples") != compatibility["validation_examples"]:
        raise ValueError("resume checkpoint validation data cardinality is incompatible")
    if saved.get("train_dataset_sha256") != compatibility["train_dataset_sha256"]:
        raise ValueError("resume checkpoint training data contents are incompatible")
    if saved.get("validation_dataset_sha256") != compatibility["validation_dataset_sha256"]:
        raise ValueError("resume checkpoint validation data contents are incompatible")
    if saved.get("output_dir") != compatibility["output_dir"]:
        raise ValueError("resume checkpoint output directory is incompatible")
    if not isinstance(checkpoint.get("log_offset"), int) or checkpoint["log_offset"] < 0:
        raise ValueError("resume checkpoint has an invalid training log offset")
    if saved.get("target_epochs") != compatibility["target_epochs"]:
        raise ValueError("resume checkpoint target epoch count is incompatible")
    if checkpoint.get("selection_policy") != (
        SELECTION_POLICY if compatibility["validation_examples"] is not None else None
    ):
        raise ValueError("resume checkpoint validation policy is incompatible")
    if not isinstance(checkpoint.get("epoch"), int) or checkpoint["epoch"] < 0:
        raise ValueError("resume checkpoint has an invalid epoch")
    if requested_epochs < checkpoint["epoch"]:
        raise ValueError("requested epochs cannot be less than the checkpoint epoch")


def _validate_training_log_boundary(path: Path, offset: int) -> None:
    """Check that a checkpoint points into an existing complete training log."""
    if not path.is_file():
        if offset == 0:
            return
        raise ValueError(f"resume checkpoint training log is missing: {path}")
    with path.open("rb") as log:
        log.seek(0, os.SEEK_END)
        if log.tell() < offset:
            raise ValueError("training log is shorter than the resume checkpoint boundary")
        if offset > 0:
            log.seek(offset - 1)
            if log.read(1) != b"\n":
                raise ValueError("resume checkpoint log offset is not at a complete record boundary")


def _restore_training_log_boundary(path: Path, offset: int) -> None:
    """Discard logged updates newer than the last atomically saved epoch checkpoint."""
    _validate_training_log_boundary(path, offset)
    if not path.is_file():
        return
    with path.open("r+b") as log:
        log.seek(offset)
        log.truncate()
        log.flush()
        os.fsync(log.fileno())


def _is_better(candidate: dict[str, Any], current: dict[str, float] | None) -> bool:
    """Rank epochs by failed pages, then recall, then CER; exact ties keep the earliest epoch.

    Recall and CER are undefined on a validation manifest with no ground-truth
    lines, where zero-denominator recall reads as 1.0. Comparing failure counts
    first keeps an epoch that cannot run inference from displacing a working one.
    When operating threshold yields no accepted detections (e.g. recall == 0.0) or
    on exact ties, threshold-free detection metrics (mean proposal IoU) track progress.
    """
    recall = float(candidate["line_detection_recall_at_iou"])
    cer = float(candidate["cer_on_matched_detections_with_misses_as_empty"])
    failures = int(candidate.get("inference_failure_count", 0))
    if failures < 0:
        raise ValueError("inference failure counts must be non-negative")
    if not np.isfinite(recall) or not np.isfinite(cer):
        raise ValueError("validation metrics must be finite")
    if current is None:
        return True
    best_failures = int(current.get("inference_failure_count", 0))
    if failures != best_failures:
        return failures < best_failures
    best_recall = float(current["line_detection_recall_at_iou"])
    best_cer = float(current["cer_on_matched_detections_with_misses_as_empty"])
    if recall != best_recall:
        return recall > best_recall
    if recall > 0.0:
        if cer != best_cer:
            return cer < best_cer
    # Break ties or evaluate when recall == 0.0 using threshold-free detection metrics
    cand_tf = float(candidate.get("mean_proposal_iou", candidate.get("proposal_ap_50", 0.0)))
    curr_tf = float(current.get("mean_proposal_iou", current.get("proposal_ap_50", 0.0)))
    if cand_tf != curr_tf:
        return cand_tf > curr_tf
    if cer != best_cer:
        return cer < best_cer
    return False


def _compute_threshold_free_metrics(
    model: Any,
    validation_dataset: PrismPageDataset,
    device: str | torch.device,
) -> dict[str, float]:
    if not hasattr(model, "backbone") or not hasattr(model, "text_head"):
        return {"mean_proposal_iou": 0.0, "proposal_ap_50": 0.0}

    dev = torch.device(device)
    was_training = model.training
    model.eval()

    all_scores: list[float] = []
    all_is_tp: list[bool] = []
    max_ious_per_gt: list[float] = []

    try:
        with torch.inference_mode():
            for idx in range(len(validation_dataset)):
                image, target = validation_dataset[idx]
                if resolve_line_supervision(target) == "unavailable":
                    continue
                gt_lines = [line for line in target.get("lines", ()) if "box" in line]
                if not gt_lines:
                    continue
                images, _ = collate_pages([(image, target)])
                images = images.to(dev)
                outputs = model(images, text_threshold=0.0)
                pred_boxes = outputs["text"]["boxes"][0].cpu().numpy()
                pred_scores = outputs["text"]["scores"][0].cpu().numpy()
                gt_boxes = np.array([line["box"] for line in gt_lines], dtype=np.float64)
                img_size = (images.shape[-1], images.shape[-2])

                iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float64)
                for p_idx in range(len(pred_boxes)):
                    for g_idx in range(len(gt_boxes)):
                        iou_matrix[p_idx, g_idx] = rotated_iou(
                            pred_boxes[p_idx], gt_boxes[g_idx], image_size=img_size
                        )

                for g_idx in range(len(gt_boxes)):
                    max_ious_per_gt.append(float(np.max(iou_matrix[:, g_idx])) if len(pred_boxes) else 0.0)

                gt_matched = [False] * len(gt_boxes)
                order = np.argsort(-pred_scores)
                for p_idx in order:
                    score = float(pred_scores[p_idx])
                    all_scores.append(score)
                    best_iou = 0.0
                    best_gt = -1
                    for g_idx in range(len(gt_boxes)):
                        if iou_matrix[p_idx, g_idx] > best_iou:
                            best_iou = float(iou_matrix[p_idx, g_idx])
                            best_gt = g_idx
                    if best_iou >= 0.5 and best_gt >= 0 and not gt_matched[best_gt]:
                        gt_matched[best_gt] = True
                        all_is_tp.append(True)
                    else:
                        all_is_tp.append(False)
    finally:
        if was_training:
            model.train()

    mean_iou = float(np.mean(max_ious_per_gt)) if max_ious_per_gt else 0.0

    total_gt = len(max_ious_per_gt)
    if total_gt == 0 or not all_scores:
        ap = 0.0
    else:
        order = np.argsort(-np.array(all_scores))
        sorted_tp = np.array(all_is_tp)[order]
        tp_cumsum = np.cumsum(sorted_tp)
        fp_cumsum = np.cumsum(~sorted_tp)
        recalls = tp_cumsum / total_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            prec_at_rec = precisions[recalls >= t]
            ap += float(np.max(prec_at_rec)) if len(prec_at_rec) > 0 else 0.0
        ap /= 11.0

    return {
        "mean_proposal_iou": round(mean_iou, 6),
        "proposal_ap_50": round(float(ap), 6),
    }


def _evaluate_validation(
    manifest: str | Path,
    model: PrismOCRModel,
    vocabulary: Vocabulary,
    *,
    device: str,
    page_resolution: int | None = None,
    validation_dataset: PrismPageDataset | None = None,
) -> dict[str, Any]:
    # Reuse the public evaluator to keep checkpoint selection aligned with reported metrics.
    with tempfile.TemporaryDirectory(prefix="prismocr-validation-") as temporary_dir:
        save_artifact(temporary_dir, model, vocabulary, model_version="v4.0.0-unvalidated")
        metrics = evaluate(manifest, temporary_dir, device=device, page_resolution=page_resolution)

    # Compute threshold-free detection metrics (e.g. average precision AP or mean proposal IoU to ground truth)
    if validation_dataset is None:
        try:
            validation_dataset = PrismPageDataset(manifest, max_page_side=page_resolution)
        except Exception:
            validation_dataset = None

    if validation_dataset is not None:
        threshold_free = _compute_threshold_free_metrics(model, validation_dataset, device=device)
        metrics.update(threshold_free)
    else:
        metrics.setdefault("mean_proposal_iou", 0.0)
        metrics.setdefault("proposal_ap_50", 0.0)
    return metrics


def train(
    manifest: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path,
    device: str = "cpu",
    epochs: int | None = None,
    batch_size: int | None = None,
    vocabulary_path: str | Path | None = None,
    validation_manifest: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    config = _config(Path(config_path))
    train_config = config.get("training", {})
    model_config = config.get("model", {})
    seed = int(train_config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    max_page_side = train_config.get("max_page_side")
    if max_page_side is not None and (
        isinstance(max_page_side, bool) or not isinstance(max_page_side, int) or max_page_side <= 0
    ):
        raise ValueError("training.max_page_side must be a positive integer or null")
    dataset = PrismPageDataset(manifest, max_page_side=max_page_side)
    if len(dataset) == 0:
        raise ValueError("training manifest is empty")
    validation_dataset = (
        PrismPageDataset(validation_manifest, max_page_side=max_page_side)
        if validation_manifest is not None
        else None
    )
    if validation_dataset is not None and len(validation_dataset) == 0:
        raise ValueError("validation manifest is empty")
    if validation_dataset is not None:
        _assert_split_disjoint(dataset, validation_dataset)
    vocabulary = (
        Vocabulary.load(vocabulary_path)
        if vocabulary_path is not None
        else Vocabulary.from_texts(dataset.texts(), max_size=int(model_config.get("vocabulary_size", 512)))
    )
    selected_device = torch.device(device)
    selected_batch_size = int(batch_size if batch_size is not None else train_config.get("batch_size", 1))
    if selected_batch_size <= 0:
        raise ValueError("batch_size must be positive")
    num_workers = int(train_config.get("num_workers", 0))
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    num_workers = min(12, num_workers)
    accumulation_steps = int(train_config.get("gradient_accumulation_steps", 1))
    if accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    use_amp = bool(train_config.get("amp", False))
    amp_dtype_name = str(train_config.get("amp_dtype", "float16"))
    if amp_dtype_name not in {"float16", "bfloat16"}:
        raise ValueError("amp_dtype must be float16 or bfloat16")
    if use_amp and selected_device.type != "cuda":
        raise ValueError("AMP training is currently supported only on CUDA")
    amp_dtype = torch.float16 if amp_dtype_name == "float16" else torch.bfloat16
    loss_weights = train_config.get("loss_weights", {})
    if not isinstance(loss_weights, dict) or any(
        not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0
        for value in loss_weights.values()
    ):
        raise ValueError("training.loss_weights must map loss names to finite nonnegative numbers")
    selected_epochs = int(epochs if epochs is not None else train_config.get("epochs", 1))
    if selected_epochs <= 0:
        raise ValueError("epochs must be positive")

    resolved_settings = _resolved_settings(
        model_config,
        train_config,
        vocabulary,
        batch_size=selected_batch_size,
        num_workers=num_workers,
    )
    max_grad_norm = resolved_settings["training"]["max_grad_norm"]
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("training.max_grad_norm must be finite and positive")
    output = Path(output_dir).resolve()
    compatibility = {
        "vocabulary_sha256": vocabulary.sha256,
        "settings": resolved_settings,
        "train_examples": len(dataset),
        "validation_examples": len(validation_dataset) if validation_dataset is not None else None,
        "train_dataset_sha256": _dataset_sha256(dataset),
        "validation_dataset_sha256": (
            _dataset_sha256(validation_dataset) if validation_dataset is not None else None
        ),
        "target_epochs": selected_epochs,
        "output_dir": str(output),
    }

    model = PrismOCRModel(
        model_contract_version=resolved_settings["model"]["model_contract_version"],
        dim=resolved_settings["model"]["dim"],
        text_proposals=resolved_settings["model"]["text_proposals"],
        region_queries=resolved_settings["model"]["region_queries"],
        vocab_size=len(vocabulary),
        vocabulary=vocabulary,
        strip_height=resolved_settings["model"]["strip_height"],
        recognizer=resolved_settings["model"]["recognizer"],
        text_threshold=resolved_settings["model"]["text_threshold"],
        region_threshold=resolved_settings["model"]["region_threshold"],
        max_table_rows=resolved_settings["model"]["max_table_rows"],
        max_table_cols=resolved_settings["model"]["max_table_cols"],
        max_cell_span=resolved_settings["model"]["max_cell_span"],
        strip_buckets=resolved_settings["model"]["strip_buckets"],
        fused_text_head=resolved_settings["model"]["fused_text_head"],
    ).to(selected_device)
    activation_checkpointing = bool(train_config.get("activation_checkpointing", False))
    if activation_checkpointing:
        backbone = getattr(model, "backbone", None)
        if backbone is None or not hasattr(backbone, "activation_checkpointing"):
            raise ValueError("activation checkpointing requires a model with checkpointable backbone stages")
        backbone.activation_checkpointing = True
    criterion = PrismOCRLoss(
        vocabulary,
        strict_ctc_length=resolved_settings["training"]["strict_ctc_length"],
        loss_weights=loss_weights,
    )
    generator = torch.Generator().manual_seed(seed)
    data_loader = DataLoader(
        dataset,
        batch_size=selected_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_pages,
        generator=generator,
        pin_memory=selected_device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=resolved_settings["training"]["learning_rate"],
        weight_decay=resolved_settings["training"]["weight_decay"],
    )
    warmup_steps = int(train_config.get("lr_warmup_steps", train_config.get("warmup_steps", 0)))
    steps_per_epoch = (len(data_loader) + accumulation_steps - 1) // accumulation_steps
    total_steps = max(1, steps_per_epoch * selected_epochs)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    checkpoint_path = output / "training-state.pt"
    start_epoch = 0
    step = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_epoch: int | None = None
    if resume_from is not None:
        checkpoint = torch.load(Path(resume_from), map_location="cpu", weights_only=True)
        _validate_resume(checkpoint, compatibility=compatibility, requested_epochs=selected_epochs)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint and checkpoint["scheduler_state"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        else:
            for _ in range(int(checkpoint["step"])):
                scheduler.step()
        start_epoch = checkpoint["epoch"]
        step = int(checkpoint["step"])
        best_model_state = checkpoint.get("best_model_state")
        best_metrics = checkpoint.get("best_metrics")
        best_epoch = checkpoint.get("best_epoch")
        if validation_dataset is not None and (best_model_state is None or best_metrics is None or best_epoch is None):
            raise ValueError("resume checkpoint is missing validation selection state")
        log_path = output / "training-log.jsonl"
        _validate_training_log_boundary(log_path, checkpoint["log_offset"])
        if Path(resume_from).resolve() != checkpoint_path.resolve():
            _atomic_checkpoint_save(checkpoint, checkpoint_path)
        scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        _restore_training_log_boundary(log_path, checkpoint["log_offset"])
        _restore_rng(checkpoint["rng_state"], generator)

    output.mkdir(parents=True, exist_ok=True)
    total_microbatches = selected_epochs * len(data_loader)
    log_path = output / "training-log.jsonl"
    log_mode = "ab" if resume_from is not None else "wb"
    run_manifest = {
        "schema_version": "4.0",
        "settings": resolved_settings,
        "loss_weights": criterion.loss_weights,
        "mixed_precision": {"enabled": use_amp, "dtype": amp_dtype_name if use_amp else None},
        "gradient_accumulation_steps": accumulation_steps,
        "activation_checkpointing": activation_checkpointing,
    }
    run_path = output / "training-run.json"
    if resume_from is None or not run_path.exists():
        run_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    start_ratio = resolved_settings["training"]["teacher_forcing_start"]
    end_ratio = resolved_settings["training"]["teacher_forcing_end"]
    max_grad_norm = resolved_settings["training"]["max_grad_norm"]

    with log_path.open(log_mode) as log:
        for epoch in range(start_epoch, selected_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for batch_index, (images, targets) in enumerate(data_loader):
                images = images.to(selected_device, non_blocking=True)
                micro_step = epoch * len(data_loader) + batch_index
                progress = micro_step / max(1, total_microbatches - 1)
                teacher_ratio = start_ratio + (end_ratio - start_ratio) * progress
                autocast = (
                    torch.autocast(device_type="cuda", dtype=amp_dtype)
                    if use_amp
                    else nullcontext()
                )
                with autocast:
                    outputs = model(images, targets, teacher_force_ratio=teacher_ratio)
                losses = criterion(outputs, targets)
                total_loss = losses.get("total")
                non_finite_losses = [
                    name for name, value in losses.items()
                    if isinstance(value, torch.Tensor)
                    and value.is_floating_point()
                    and not bool(torch.isfinite(value).all())
                ]
                if not isinstance(total_loss, torch.Tensor) or total_loss.numel() != 1:
                    raise ValueError("training criterion must return a scalar tensor named 'total'")
                if non_finite_losses:
                    optimizer.zero_grad(set_to_none=True)
                    page_ids = [str(target.get("page_id", "unknown")) for target in targets]
                    raise FloatingPointError(
                        f"non-finite training loss at epoch {epoch + 1}, batch {batch_index}, "
                        f"pages {page_ids}: {non_finite_losses}"
                    )
                group_start = (batch_index // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, len(data_loader) - group_start)
                scaler.scale(total_loss / group_size).backward()
                update_now = (
                    batch_index + 1 == len(data_loader)
                    or (batch_index + 1) % accumulation_steps == 0
                )
                optimizer_updated = False
                scaler_overflow_skipped = False
                current_lr = float(optimizer.param_groups[0]["lr"])
                if update_now:
                    scaler.unscale_(optimizer)
                    non_finite_gradients = [
                        name for name, parameter in model.named_parameters()
                        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                    ]
                    if non_finite_gradients:
                        if scaler.is_enabled():
                            # unscale_ has recorded the overflow for this
                            # optimizer. GradScaler.step skips the update and
                            # update lowers its scale for the next attempt.
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)
                            scaler_overflow_skipped = True
                        else:
                            optimizer.zero_grad(set_to_none=True)
                            raise FloatingPointError(
                                f"non-finite gradients before optimizer step at epoch {epoch + 1}, "
                                f"batch {batch_index}: {non_finite_gradients[:8]}"
                            )
                    else:
                        try:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), max_grad_norm, error_if_nonfinite=True
                            )
                        except RuntimeError as error:
                            optimizer.zero_grad(set_to_none=True)
                            raise FloatingPointError(
                                f"non-finite gradient norm before optimizer step at epoch {epoch + 1}, "
                                f"batch {batch_index}"
                            ) from error
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                        optimizer_updated = True
                        step += 1
                row = {
                    "epoch": epoch + 1,
                    "step": step,
                    "micro_step": micro_step + 1,
                    "learning_rate": current_lr,
                    "optimizer_updated": optimizer_updated,
                    "scaler_overflow_skipped": scaler_overflow_skipped,
                    "teacher_force_ratio": teacher_ratio,
                    **{
                        key: float(value.detach().float().cpu())
                        for key, value in losses.items()
                        if isinstance(value, torch.Tensor) and value.ndim == 0
                    },
                    **{
                        key: value
                        for key, value in losses.items()
                        if not isinstance(value, torch.Tensor)
                    },
                }
                log.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
                log.flush()

            epoch_metrics: dict[str, Any] | None = None
            improved = False
            if validation_dataset is not None:
                epoch_metrics = _evaluate_validation(
                    validation_manifest,
                    model,
                    vocabulary,
                    device=device,
                    page_resolution=(
                        int(max_page_side) if max_page_side is not None else None
                    ),
                    validation_dataset=validation_dataset,
                )
                improved = _is_better(epoch_metrics, best_metrics)
                if improved:
                    best_metrics = {
                        "line_detection_recall_at_iou": float(epoch_metrics["line_detection_recall_at_iou"]),
                        "cer_on_matched_detections_with_misses_as_empty": float(
                            epoch_metrics["cer_on_matched_detections_with_misses_as_empty"]
                        ),
                        "inference_failure_count": int(epoch_metrics["inference_failure_count"]),
                        "inference_failure_rate": float(epoch_metrics["inference_failure_rate"]),
                        "ground_truth_lines": int(epoch_metrics["ground_truth_lines"]),
                        "mean_proposal_iou": float(epoch_metrics.get("mean_proposal_iou", 0.0)),
                        "proposal_ap_50": float(epoch_metrics.get("proposal_ap_50", 0.0)),
                    }
                    best_model_state = _cpu_model_state(model)
                    best_epoch = epoch + 1
                log.write(
                    (
                        json.dumps(
                            {
                                "epoch": epoch + 1,
                                "validation_metrics": epoch_metrics,
                                "selected_best": improved,
                                "best_epoch": best_epoch,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    ).encode("utf-8")
                )
                log.flush()

            log.flush()
            os.fsync(log.fileno())
            state = _checkpoint_state(
                model,
                optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                step=step,
                generator=generator,
                compatibility=compatibility,
                best_model_state=best_model_state,
                best_metrics=best_metrics,
                best_epoch=best_epoch,
                log_offset=log.tell(),
                scaler_state=scaler.state_dict(),
            )
            _atomic_checkpoint_save(state, checkpoint_path)
            if improved:
                save_artifact(
                    output,
                    model,
                    vocabulary,
                    model_version="v4.0.0-unvalidated",
                    training_provenance={
                        "train_dataset_sha256": compatibility["train_dataset_sha256"],
                        "validation_dataset_sha256": compatibility["validation_dataset_sha256"],
                        "selection_policy": SELECTION_POLICY,
                    },
                )

    if validation_dataset is not None:
        if best_model_state is None:
            raise ValueError("validation did not produce a selectable checkpoint")
        model.load_state_dict(best_model_state, strict=True)
    save_artifact(
        output,
        model,
        vocabulary,
        model_version="v4.0.0-unvalidated",
        training_provenance={
            "train_dataset_sha256": compatibility["train_dataset_sha256"],
            "validation_dataset_sha256": compatibility["validation_dataset_sha256"],
            "selection_policy": SELECTION_POLICY if validation_dataset is not None else None,
        },
    )
    return {
        "output_dir": str(output.resolve()),
        "examples": len(dataset),
        "epochs": selected_epochs,
        "steps": step,
        "vocab_size": len(vocabulary),
        "model_tensor_bytes": model.tensor_bytes(),
        "status": "training_completed_without_validation_claim" if validation_dataset is None else "training_completed_with_validation",
        "validation_status": "unvalidated" if validation_dataset is None else "measured_on_supplied_manifest",
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "selection_policy": SELECTION_POLICY if validation_dataset is not None else None,
        "checkpoint_path": str(checkpoint_path.resolve()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSONL training manifest")
    parser.add_argument("--output", required=True, help="output artifact directory")
    parser.add_argument("--config", default="configs/v4_detection.toml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--vocabulary", help="reuse a frozen vocab-v4.json")
    parser.add_argument("--validation-manifest", help="optional JSONL validation manifest")
    parser.add_argument("--resume", "--resume-from", dest="resume_from", help="training-state.pt checkpoint")
    args = parser.parse_args(argv)
    result = train(
        args.manifest,
        args.output,
        config_path=args.config,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        vocabulary_path=args.vocabulary,
        validation_manifest=args.validation_manifest,
        resume_from=args.resume_from,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
