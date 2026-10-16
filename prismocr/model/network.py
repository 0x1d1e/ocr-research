"""Unified typed detector, immutable matching snapshot and sparse OCR forward."""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from prismocr.data.vocabulary import Vocabulary
from prismocr.matching import MatchSet, match_proposals
from prismocr.model.backbone import PrismBackbone
from prismocr.model.heads import PairRelationHead, RegionQueryHead, TextLineHead, validate_model_contract_version
from prismocr.model.recognizer import (
    STRIP_BUCKETS,
    SparseCTCRecognizer,
    choose_inference_bucket,
    minimum_ctc_frames,
    validate_strip_buckets,
)

TEXT_ROLES = (
    "TITLE", "H1", "H2", "PARA_LINE", "LIST_ITEM", "KEY", "VALUE",
    "CAPTION", "HEADER", "FOOTER", "FORMULA", "UNKNOWN",
)
TEXT_ROLE_TO_ID = {name: index for index, name in enumerate(TEXT_ROLES)}
REGION_KINDS = ("NO_OBJECT", "REGION", "TABLE", "CELL", "LIST", "FORM_FIELD")
REGION_KIND_TO_ID = {name: index for index, name in enumerate(REGION_KINDS)}
RELATION_NAMES = ("NEXT", "PARENT", "KV_VALUE", "RIGHT_CELL", "CONTAINS")


def _target_box(item: Mapping[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    value = item.get("box")
    if value is None:
        raise ValueError("converted model targets must contain a six-value 'box'")
    box = torch.as_tensor(value, device=device, dtype=dtype)
    if box.shape != (6,):
        raise ValueError("target box must have shape [6]")
    return box


def _interpolate_teacher_box(
    predicted: torch.Tensor, target: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Blend equivalent oriented boxes without cancelling antipodal angles."""
    target_orientation = target[4:6]
    predicted_orientation = predicted[4:6]
    sign = torch.where(
        torch.dot(predicted_orientation, target_orientation) < 0,
        target_orientation.new_tensor(-1.0),
        target_orientation.new_tensor(1.0),
    )
    aligned_target = torch.cat((target[:4], target_orientation * sign))
    blended = (1.0 - alpha) * predicted + alpha * aligned_target
    orientation = blended[4:6]
    norm = torch.linalg.vector_norm(orientation)
    fallback_norm = torch.linalg.vector_norm(predicted_orientation).clamp_min(1e-6)
    normalized = orientation / norm.clamp_min(1e-6)
    fallback = predicted_orientation / fallback_norm
    orientation = torch.where(norm > 1e-6, normalized, fallback)
    return torch.cat((blended[:4], orientation))


def _role_id(value: Any) -> int:
    if isinstance(value, str):
        if value not in TEXT_ROLE_TO_ID:
            raise ValueError(f"unsupported text role {value!r}")
        return TEXT_ROLE_TO_ID[value]
    result = int(value)
    if not 0 <= result < len(TEXT_ROLES):
        raise ValueError(f"text role id {result} is out of range")
    return result


def _kind_id(value: Any) -> int:
    if isinstance(value, str):
        if value not in REGION_KIND_TO_ID or value == "NO_OBJECT":
            raise ValueError(f"unsupported region kind {value!r}")
        return REGION_KIND_TO_ID[value]
    result = int(value)
    if not 1 <= result < len(REGION_KINDS):
        raise ValueError(f"region kind id {result} is out of range")
    return result


def validated_model_state(checkpoint: Mapping[str, Any], config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Bind checkpoint semantics to config, including missing-field legacy assets."""
    configured = validate_model_contract_version(config.get("model_contract_version", 1))
    stored = validate_model_contract_version(checkpoint.get("model_contract_version", 1))
    if configured != stored:
        raise ValueError("checkpoint and configuration model_contract_version disagree")
    return checkpoint.get("model_state", checkpoint)


class PrismOCRModel(nn.Module):
    """V4 research model with text and structure proposals kept as separate streams."""

    def __init__(
        self,
        *,
        dim: int = 160,
        text_proposals: int = 128,
        region_queries: int = 64,
        vocab_size: int = 512,
        vocabulary: Vocabulary | None = None,
        strip_height: int = 5,
        recognizer: str = "transformer",
        text_threshold: float = 0.55,
        region_threshold: float = 0.5,
        match_iou_threshold: float = 0.5,
        max_table_rows: int = 128,
        max_table_cols: int = 128,
        max_cell_span: int = 32,
        strip_buckets: Sequence[int] = STRIP_BUCKETS,
        model_contract_version: int = 1,
    ) -> None:
        super().__init__()
        if dim % 5:
            raise ValueError("dim must be divisible by 5 for the reference transformer")
        if text_proposals <= 0 or region_queries <= 0:
            raise ValueError("proposal capacities must be positive")
        if vocabulary is not None and len(vocabulary) != vocab_size:
            raise ValueError("vocabulary length must equal vocab_size")
        self.model_contract_version = validate_model_contract_version(model_contract_version)
        self.dim = dim
        self.text_proposals = text_proposals
        self.region_queries = region_queries
        self.vocab_size = vocab_size
        self.vocabulary = vocabulary
        self.text_threshold = text_threshold
        self.region_threshold = region_threshold
        self.match_iou_threshold = match_iou_threshold
        self.max_table_rows = max_table_rows
        self.max_table_cols = max_table_cols
        self.max_cell_span = max_cell_span
        self.strip_buckets = validate_strip_buckets(strip_buckets)

        self.backbone = PrismBackbone(dim)
        self.text_head = TextLineHead(64, dim, len(TEXT_ROLES), model_contract_version=self.model_contract_version)
        self.region_head = RegionQueryHead(
            dim,
            region_queries,
            len(REGION_KINDS),
            max_table_rows=max_table_rows,
            max_table_cols=max_table_cols,
            max_cell_span=max_cell_span,
        )
        self.lat4 = nn.Conv2d(64, dim, 1, bias=False)
        self.lat8 = nn.Conv2d(112, dim, 1, bias=False)
        self.recognizer = SparseCTCRecognizer(dim, vocab_size, strip_height, recognizer)
        self.relation_head = PairRelationHead(dim, len(RELATION_NAMES))

    def _matches(
        self,
        text: dict[str, torch.Tensor],
        regions: dict[str, torch.Tensor],
        targets: Sequence[Mapping[str, Any]],
        image_size: tuple[int, int],
    ) -> dict[str, tuple[MatchSet, ...]]:
        text_matches: list[MatchSet] = []
        region_matches: list[MatchSet] = []
        for batch_index, target in enumerate(targets):
            lines = list(target.get("lines", ()))
            region_items = list(target.get("regions", ())) + [
                {**cell, "kind": cell.get("kind", "CELL")}
                for cell in target.get("cells", ())
            ]
            line_boxes = torch.stack([_target_box(item, torch.device("cpu"), torch.float32) for item in lines]) if lines else torch.empty((0, 6))
            line_roles = torch.tensor([_role_id(item["role"]) for item in lines], dtype=torch.long)
            text_matches.append(
                match_proposals(
                    text["boxes"][batch_index],
                    line_boxes,
                    text["candidate_mask"][batch_index],
                    image_size=image_size,
                    pred_class_logits=text["role_logits"][batch_index],
                    gt_classes=line_roles,
                    iou_threshold=self.match_iou_threshold,
                )
            )

            region_boxes = torch.stack([_target_box(item, torch.device("cpu"), torch.float32) for item in region_items]) if region_items else torch.empty((0, 6))
            region_kinds = torch.tensor([_kind_id(item["kind"]) for item in region_items], dtype=torch.long)
            region_matches.append(
                match_proposals(
                    regions["boxes"][batch_index],
                    region_boxes,
                    regions["candidate_mask"][batch_index],
                    image_size=image_size,
                    pred_class_logits=regions["class_logits"][batch_index],
                    gt_classes=region_kinds,
                    iou_threshold=self.match_iou_threshold,
                )
            )
        return {"text": tuple(text_matches), "regions": tuple(region_matches)}

    @property
    def ctc_bucket_policy(self) -> str:
        return "fixed_max" if self.model_contract_version == 2 else "legacy_label_train_geometry_infer"

    def _training_bucket(self, text: str) -> int | None:
        if self.model_contract_version == 2:
            return max(self.strip_buckets)
        if self.vocabulary is not None:
            ids = self.vocabulary.encode(text).ids
        else:
            # Exact codepoints preserve repeated-character alignment estimates.
            ids = tuple(ord(char) for char in text)
        required = minimum_ctc_frames(ids)
        return next((bucket for bucket in self.strip_buckets if bucket >= required), None)

    def forward(
        self,
        images: torch.Tensor,
        targets: Sequence[Mapping[str, Any]] | None = None,
        *,
        teacher_force_ratio: float = 0.0,
        text_threshold: float | None = None,
        region_threshold: float | None = None,
        recognition_threshold: float | None = None,
    ) -> dict[str, Any]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B,3,H,W]")
        text_threshold = self.text_threshold if text_threshold is None else text_threshold
        region_threshold = self.region_threshold if region_threshold is None else region_threshold
        f4, f8, f16 = self.backbone(images)
        text_raw = self.text_head(f4)
        text = TextLineHead.proposals(
            text_raw,
            max_proposals=self.text_proposals,
            score_threshold=text_threshold,
            model_contract_version=self.model_contract_version,
            image_size=(images.shape[-1], images.shape[-2]),
        )
        if f16.dtype == self.region_head.query.weight.dtype:
            regions = self.region_head(f16, region_threshold)
        else:
            # The query decoder is numerically sensitive in autocast, especially
            # its attention backward. Keep it in parameter precision while the
            # backbone, text heads, and recognizer retain the mixed-precision path.
            with torch.autocast(device_type=images.device.type, enabled=False):
                regions = self.region_head(
                    f16.to(dtype=self.region_head.query.weight.dtype), region_threshold
                )
        assignments = None
        if targets is not None:
            if len(targets) != images.shape[0]:
                raise ValueError("one target mapping is required per image")
            assignments = self._matches(
                text, regions, targets, (images.shape[-1], images.shape[-2])
            )

        sample_boxes = text["boxes"].clone()
        ocr_items: list[tuple[int, int, int]] = []
        unsupported: list[tuple[int, int, str]] = []
        if targets is not None and assignments is not None:
            alpha = min(1.0, max(0.0, float(teacher_force_ratio)))
            for batch_index, match in enumerate(assignments["text"]):
                lines = list(targets[batch_index].get("lines", ()))
                for gt_index, proposal_index in match.accepted_pairs:
                    gt_box = _target_box(lines[gt_index], images.device, images.dtype)
                    sample_boxes[batch_index, proposal_index] = _interpolate_teacher_box(
                        text["boxes"][batch_index, proposal_index], gt_box, alpha
                    )
                    transcript = str(lines[gt_index].get("text", ""))
                    bucket = self._training_bucket(transcript)
                    rejection = None
                    if self.model_contract_version == 2:
                        if self.vocabulary is None:
                            raise ValueError("contract 2 training requires the actual vocabulary")
                        encoded = self.vocabulary.encode(transcript)
                        if encoded.oov_positions:
                            rejection = "ctc_oov_characters"
                        elif minimum_ctc_frames(encoded.ids) > bucket:
                            rejection = f"requires_{minimum_ctc_frames(encoded.ids)}_frames_but_bucket_is_{bucket}"
                    if rejection is not None:
                        unsupported.append((batch_index, proposal_index, rejection))
                    elif bucket is None:
                        unsupported.append((
                            batch_index,
                            proposal_index,
                            f"ctc_length_exceeds_{self.strip_buckets[-1]}_frames",
                        ))
                    else:
                        ocr_items.append((batch_index, proposal_index, bucket))
        else:
            for batch_index in range(images.shape[0]):
                recognition_mask = text["mask"][batch_index]
                if recognition_threshold is not None:
                    recognition_mask = recognition_mask & (
                        text["scores"][batch_index] >= recognition_threshold
                    )
                for proposal_index in torch.where(recognition_mask)[0].tolist():
                    bucket = choose_inference_bucket(
                        text["boxes"][batch_index, proposal_index],
                        (images.shape[-1], images.shape[-2]),
                        self.strip_buckets,
                        model_contract_version=self.model_contract_version,
                    )
                    if bucket is None:
                        unsupported.append((
                            batch_index,
                            proposal_index,
                            f"geometry_exceeds_{self.strip_buckets[-1]}_frame_bucket",
                        ))
                    else:
                        ocr_items.append((batch_index, proposal_index, bucket))

        f_glyph = self.lat4(f4) + torch.nn.functional.interpolate(
            self.lat8(f8), size=f4.shape[-2:], mode="bilinear", align_corners=False
        )
        grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for batch_index, proposal_index, bucket in ocr_items:
            grouped[bucket].append((batch_index, proposal_index))
        ocr_groups: dict[int, dict[str, torch.Tensor]] = {}
        for bucket, refs_list in grouped.items():
            refs = torch.tensor(refs_list, device=images.device, dtype=torch.long)
            batch_idx, proposal_idx = refs[:, 0], refs[:, 1]
            boxes = sample_boxes[batch_idx, proposal_idx]
            embeddings = text["embeddings"][batch_idx, proposal_idx]
            logits = self.recognizer.forward_bucket(f_glyph, boxes, embeddings, refs, bucket)
            ocr_groups[bucket] = {"refs": refs, "logits": logits}

        node_embeddings = torch.cat((text["embeddings"], regions["embeddings"]), dim=1)
        graph_logits = self.relation_head(node_embeddings)
        graph_mask = torch.cat((text["mask"], regions["mask"]), dim=1)
        return {
            "text": text,
            "regions": regions,
            "text_heatmap_logits": text_raw["heatmap_logits"],
            "assignments": assignments,
            "sample_boxes": sample_boxes,
            "ocr_groups": ocr_groups,
            "ocr_unsupported": tuple(unsupported),
            "graph_logits": graph_logits,
            "graph_mask": graph_mask,
            "image_size": (images.shape[-1], images.shape[-2]),
            "model_contract_version": self.model_contract_version,
            "ctc_bucket_policy": self.ctc_bucket_policy,
        }

    def fused_for_inference(self) -> "PrismOCRModel":
        """Return an explicitly fused eval copy, leaving the trainable model intact."""
        fused = copy.deepcopy(self).eval()
        fused.backbone.fuse_for_inference()
        return fused

    def tensor_bytes(self) -> int:
        """Count serialized model state tensors, including registered buffers."""
        return sum(tensor.numel() * tensor.element_size() for tensor in self.state_dict().values())
