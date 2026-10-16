"""Shared-MatchSet supervision for detection, typed roles, CTC and graph links."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from prismocr.data.schema import resolve_line_supervision
from prismocr.data.vocabulary import Vocabulary
from prismocr.graph_decode import GraphNode, relation_candidate_error
from prismocr.model.network import (
    RELATION_NAMES,
    _kind_id,
    _role_id,
    _target_box,
)
from prismocr.model.recognizer import minimum_ctc_frames


def _relation_source_allowed(relation: str, source: Mapping[str, Any]) -> bool:
    kind = source.get("_node_type")
    if relation == "NEXT":
        return kind == "line"
    if relation == "PARENT":
        return kind in {"line", "cell", "region"}
    if relation == "KV_VALUE":
        return kind == "line" and source.get("role") == "KEY"
    if relation == "RIGHT_CELL":
        return kind == "cell"
    if relation == "CONTAINS":
        return kind in {"cell", "region"}
    return False


def _validate_grid_annotations(
    targets: Sequence[Mapping[str, Any]], regions: Mapping[str, Any]
) -> None:
    field_specs = (
        ("row", "row_logits", 0, False),
        ("col", "col_logits", 0, False),
        ("rowspan", "rowspan_logits", -1, True),
        ("colspan", "colspan_logits", -1, True),
    )
    for target in targets:
        page_id = target.get("page_id", "unknown")
        ambiguous_table_ids = {
            region.get("id")
            for region in target.get("regions", ())
            if region.get("kind") == "TABLE" and region.get("annotation_ambiguous") is True
        }
        for index, cell in enumerate(target.get("cells", ())):
            # Ambiguous table grids are excluded from grid supervision, so their
            # labels must not be capacity- or type-validated before exclusion.
            if cell.get("table_id") in ambiguous_table_ids or cell.get("annotation_ambiguous") is True:
                continue
            cell_id = cell.get("id", index)
            for field, logits_field, offset, positive in field_specs:
                value = cell.get(field)
                if value is None and positive:
                    value = 1
                if value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, int) or (value <= 0 if positive else value < 0):
                    expected = "a positive integer" if positive else "a nonnegative integer"
                    raise ValueError(f"cell {cell_id!r} on {page_id!r} has invalid {field}; expected {expected}")
                logits = regions.get(logits_field)
                if not isinstance(logits, torch.Tensor):
                    raise ValueError(f"model outputs are missing learned table head {logits_field!r}")
                label = value + offset
                if not 0 <= label < logits.shape[-1]:
                    raise ValueError(
                        f"cell {cell_id!r} on {page_id!r} has {field}={value} "
                        f"outside configured grid capacity {logits.shape[-1]}"
                    )
            header = cell.get("is_header")
            if header is not None:
                if not isinstance(header, bool):
                    raise ValueError(f"cell {cell_id!r} on {page_id!r} has invalid is_header label")
                if not isinstance(regions.get("header_logits"), torch.Tensor):
                    raise ValueError("model outputs are missing learned table head 'header_logits'")


def _graph_node(item: Mapping[str, Any]) -> GraphNode | None:
    item_id = item.get("id")
    node_type = item.get("_node_type")
    if not isinstance(item_id, str) or node_type not in {"line", "region", "cell"}:
        return None
    polygon = item.get("polygon_px")
    return GraphNode(
        id=item_id,
        node_type=node_type,
        polygon=(
            polygon
            if isinstance(polygon, Sequence) and not isinstance(polygon, (str, bytes))
            else None
        ),
        role=item.get("role") if node_type == "line" else None,
        kind=item.get("kind") if node_type == "region" else None,
        scope_id=item.get("scope_id", "document") if node_type == "line" else None,
        table_id=item.get("table_id") if node_type == "cell" else None,
        row=item.get("row") if node_type == "cell" else None,
        col=item.get("col") if node_type == "cell" else None,
    )


def _relation_target_allowed(
    relation: str,
    source: Mapping[str, Any],
    candidate: Mapping[str, Any],
    graph_nodes: Mapping[str, GraphNode],
) -> bool:
    source_id, candidate_id = source.get("id"), candidate.get("id")
    if not isinstance(source_id, str) or not isinstance(candidate_id, str):
        return False
    source_node = graph_nodes.get(source_id)
    candidate_node = graph_nodes.get(candidate_id)
    if source_node is None or candidate_node is None:
        return False
    return relation_candidate_error(
        relation, source_node, candidate_node, graph_nodes
    ) is None


class PrismOCRLoss(nn.Module):
    def __init__(
        self,
        vocabulary: Vocabulary,
        *,
        strict_ctc_length: bool = True,
        loss_weights: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.vocabulary = vocabulary
        self.strict_ctc_length = strict_ctc_length
        self.loss_weights = {
            "heatmap": 1.0,
            "role": 1.0,
            "region": 1.0,
            "box": 1.0,
            "ctc": 1.0,
            "graph": 1.0,
            "grid": 1.0,
        }
        supplied_weights = dict(loss_weights or {})
        unknown_weights = set(supplied_weights) - set(self.loss_weights)
        if unknown_weights:
            raise ValueError(f"unknown loss weights: {sorted(unknown_weights)}")
        for name, weight in supplied_weights.items():
            if not isinstance(weight, (int, float)) or not math.isfinite(float(weight)) or weight < 0:
                raise ValueError(f"loss weight {name!r} must be finite and nonnegative")
            self.loss_weights[name] = float(weight)

    @staticmethod
    def _heatmap_targets(
        targets: Sequence[Mapping[str, Any]],
        shape: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        height, width = shape
        rendered = torch.zeros((len(targets), 1, height, width), device="cpu", dtype=torch.float32)
        # Only the integer anchor cell is foreground. Gaussian shoulders spread
        # beyond the anchor for large lines, and a value threshold would count
        # them as positives, training extra detection centers for one line.
        anchors = torch.zeros((len(targets), 1, height, width), dtype=torch.bool)
        collision_count = 0
        ys = torch.arange(height, device="cpu", dtype=torch.float32).view(height, 1)
        xs = torch.arange(width, device="cpu", dtype=torch.float32).view(1, width)
        for batch_index, target in enumerate(targets):
            if resolve_line_supervision(target) == "unavailable":
                continue
            occupied: set[tuple[int, int]] = set()
            for line in target.get("lines", ()):
                box = _target_box(line, torch.device("cpu"), torch.float32)
                # The proposal head decodes centers within the selected grid
                # cell as (index + sigmoid(offset)) / grid_size. Floor to that
                # same cell interval so center supervision remains representable.
                x = min(width - 1, max(0, int(float(box[0]) * width)))
                y = min(height - 1, max(0, int(float(box[1]) * height)))
                if (y, x) in occupied:
                    collision_count += 1
                occupied.add((y, x))
                sigma = max(1.0, float(torch.sqrt((box[2] * width * box[3] * height).clamp_min(0)) * 0.25))
                gaussian = torch.exp(-((xs - x).square() + (ys - y).square()) / (2 * sigma * sigma))
                rendered[batch_index, 0] = torch.maximum(rendered[batch_index, 0], gaussian)
                rendered[batch_index, 0, y, x] = 1.0
                anchors[batch_index, 0, y, x] = True
        return rendered.to(device=device, dtype=dtype), collision_count, anchors.to(device=device)

    @staticmethod
    def _heatmap_masks(
        targets: Sequence[Mapping[str, Any]], anchors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = torch.zeros_like(anchors)
        negative = torch.zeros_like(anchors)
        height, width = anchors.shape[-2:]
        for index, target in enumerate(targets):
            state = resolve_line_supervision(target)
            extent = target.get("_valid_image_extent", (1.0, 1.0))
            if (
                not isinstance(extent, (tuple, list)) or len(extent) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or not 0 < value <= 1 for value in extent)
            ):
                raise ValueError("_valid_image_extent must contain two finite fractions in (0,1]")
            xs = torch.arange(width, device=anchors.device) / width
            ys = torch.arange(height, device=anchors.device) / height
            page = (ys[:, None] < extent[1]) & (xs[None, :] < extent[0])
            if state != "unavailable":
                valid[index, 0] = page
            if state == "complete":
                negative[index, 0] = page
        if bool((anchors & ~valid).any()):
            raise ValueError("text anchors must lie inside supervised valid image extent")
        return valid, negative

    @staticmethod
    def _focal_heatmap(
        logits: torch.Tensor, target: torch.Tensor, anchors: torch.Tensor,
        valid: torch.Tensor | None = None, negative_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = logits.float()
        target = target.float()
        anchors = anchors.to(device=logits.device, dtype=torch.bool)
        if anchors.shape != target.shape or logits.shape != target.shape:
            raise ValueError("heatmap logits and anchors must match the target shape")
        valid = torch.ones_like(anchors) if valid is None else valid
        negative_valid = valid if negative_valid is None else negative_valid
        for mask in (valid, negative_valid):
            if mask.dtype != torch.bool or mask.shape != target.shape or mask.device != logits.device:
                raise ValueError("heatmap validity masks must be boolean and match target shape/device")
        if bool((anchors & ~valid).any()) or bool((negative_valid & ~valid).any()):
            raise ValueError("heatmap anchors and negative validity must be subsets of valid extent")
        probability = logits.sigmoid().clamp(1e-6, 1.0 - 1e-6)
        positive = anchors.to(logits.dtype)
        negative = ((~anchors) & negative_valid).to(logits.dtype)
        negative_weight = (1.0 - target).pow(4)
        positive_loss = -torch.log(probability) * (1.0 - probability).pow(2) * positive
        negative_loss = -torch.log(1.0 - probability) * probability.pow(2) * negative_weight * negative
        # Page-balanced means: no grid-size cliff at zero positives and no
        # dilution/contribution from unavailable pages. Partial pages have only
        # the annotated-positive term, never an unlabelled-background negative.
        dims = (1, 2, 3)
        per_page = (
            positive_loss.sum(dims) / positive.sum(dims).clamp_min(1.0)
            + negative_loss.sum(dims) / negative.sum(dims).clamp_min(1.0)
        )
        eligible = valid.flatten(1).any(1).to(logits.dtype)
        return (per_page * eligible).sum() / eligible.sum().clamp_min(1.0)

    @staticmethod
    def _box_loss(predicted: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        if predicted.numel() == 0:
            return predicted.float().sum() * 0.0
        predicted = predicted.float()
        truth = truth.float()
        centers = F.smooth_l1_loss(predicted[:, :2], truth[:, :2], beta=0.05)
        # Regress logarithms of the normalized side lengths. The proposal head
        # decodes ``size = softplus(raw - 2)``, whose slope vanishes as raw goes
        # negative, so a linear-size smooth L1 lets the head collapse small line
        # heights to near zero with almost no gradient. In log space the chain
        # rule contributes sigmoid(x)/softplus(x) -> 1, so small sizes stay
        # trainable while the term remains a scale-invariant side ratio.
        predicted_log_size = torch.log(predicted[:, 2:4].clamp_min(1e-6))
        truth_log_size = torch.log(truth[:, 2:4].clamp_min(1e-6))
        sizes = F.smooth_l1_loss(predicted_log_size, truth_log_size, beta=1.0)
        pred_orientation = F.normalize(predicted[:, 4:6], dim=-1, eps=1e-6)
        gt_orientation = F.normalize(truth[:, 4:6], dim=-1, eps=1e-6)
        angle = (1.0 - (pred_orientation * gt_orientation).sum(-1).abs()).mean()
        return centers + sizes + angle

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        assignments = outputs.get("assignments")
        if assignments is None:
            raise ValueError("training loss requires MatchSets created by the same model forward")
        if len(targets) != len(assignments["text"]):
            raise ValueError("target batch size does not match MatchSets")
        _validate_grid_annotations(targets, outputs["regions"])
        device = outputs["text_heatmap_logits"].device
        dtype = outputs["text_heatmap_logits"].dtype
        batch_size = len(targets)
        # Accumulate in float32 before the empty-loss zero. A large FP16
        # feature map can overflow its sum to infinity, making ``sum() * 0`` NaN.
        zero = outputs["text_heatmap_logits"].float().sum() * 0.0

        heatmap_target, collision_count, heatmap_anchors = self._heatmap_targets(
            targets,
            outputs["text_heatmap_logits"].shape[-2:],
            device,
            dtype,
        )
        line_states = [resolve_line_supervision(target) for target in targets]
        heatmap_valid, heatmap_negative = self._heatmap_masks(targets, heatmap_anchors)
        heatmap_loss = self._focal_heatmap(
            outputs["text_heatmap_logits"], heatmap_target, heatmap_anchors,
            heatmap_valid, heatmap_negative,
        )
        role_loss = zero
        region_loss = zero
        box_loss = zero
        graph_loss = zero
        grid_terms: list[torch.Tensor] = []
        text_box_terms: list[torch.Tensor] = []
        region_box_terms: list[torch.Tensor] = []
        accepted_gt_for_proposal: list[dict[int, int]] = []
        grid_exclusions: list[dict[str, Any]] = []
        line_items_by_batch: list[list[Mapping[str, Any]]] = []
        oov_count = 0
        rejected_ctc: list[dict[str, Any]] = []
        unsupported_ctc = tuple(outputs.get("ocr_unsupported", ()))
        unmatched_gt = 0
        missed_gt = 0
        unmatched_relations = 0
        text_gt_count = 0
        text_assigned_count = 0
        text_accepted_count = 0

        for batch_index, target in enumerate(targets):
            line_items = list(target.get("lines", ()))
            line_items_by_batch.append(line_items)
            table_ambiguous = {
                region.get("id")
                for region in target.get("regions", ())
                if region.get("kind") == "TABLE" and region.get("annotation_ambiguous") is True
            }
            region_items = list(target.get("regions", ())) + [
                {**cell, "kind": cell.get("kind", "CELL")}
                for cell in target.get("cells", ())
            ]
            text_match = assignments["text"][batch_index]
            region_match = assignments["regions"][batch_index]
            unmatched_gt += len(text_match.unassigned_gt_idx) + len(region_match.unassigned_gt_idx)
            missed_gt += len(text_match.missed_gt_idx) + len(region_match.missed_gt_idx)
            text_gt_count += text_match.gt_count
            text_assigned_count += len(text_match.assigned_gt_idx)
            text_accepted_count += len(text_match.accepted_gt_idx)

            for gt_index, proposal_index in text_match.assignment_pairs:
                item = line_items[gt_index]
                role = _role_id(item["role"])
                role_loss = role_loss + F.cross_entropy(
                    outputs["text"]["role_logits"][batch_index, proposal_index].float()[None],
                    torch.tensor([role], device=device),
                )
                truth = _target_box(item, device, dtype)[None]
                prediction = outputs["text"]["boxes"][batch_index, proposal_index][None]
                text_box_terms.append(self._box_loss(prediction, truth))

            region_logits = outputs["regions"]["class_logits"][batch_index]
            region_targets = torch.zeros(region_logits.shape[0], device=device, dtype=torch.long)
            for gt_index, proposal_index in region_match.assignment_pairs:
                item = region_items[gt_index]
                region_targets[proposal_index] = _kind_id(item["kind"])
                truth = _target_box(item, device, dtype)[None]
                prediction = outputs["regions"]["boxes"][batch_index, proposal_index][None]
                region_box_terms.append(self._box_loss(prediction, truth))
                if item.get("kind") != "CELL":
                    continue
                if item.get("table_id") in table_ambiguous or item.get("annotation_ambiguous") is True:
                    grid_exclusions.append({
                        "page_id": str(target.get("page_id", f"batch:{batch_index}")),
                        "cell_id": str(item.get("id", gt_index)),
                        "table_id": item.get("table_id"),
                        "reason": "annotation_ambiguous",
                    })
                    continue
                for field, logits_field, offset in (
                    ("row", "row_logits", 0),
                    ("col", "col_logits", 0),
                    ("rowspan", "rowspan_logits", -1),
                    ("colspan", "colspan_logits", -1),
                ):
                    value = item.get(field, 1 if field in {"rowspan", "colspan"} else None)
                    if value is None and field in {"rowspan", "colspan"}:
                        value = 1
                    if value is None:
                        continue
                    if isinstance(value, bool) or not isinstance(value, int):
                        raise ValueError(f"cell {item.get('id', gt_index)!r} has invalid {field} label")
                    label = value + offset
                    logits = outputs["regions"].get(logits_field)
                    if not isinstance(logits, torch.Tensor):
                        raise ValueError(f"model outputs are missing learned table head {logits_field!r}")
                    if not 0 <= label < logits.shape[-1]:
                        page_id = target.get("page_id", f"batch:{batch_index}")
                        raise ValueError(
                            f"cell {item.get('id', gt_index)!r} on {page_id!r} has {field}={value} "
                            f"outside configured grid capacity {logits.shape[-1]}"
                        )
                    grid_terms.append(F.cross_entropy(
                        logits[batch_index, proposal_index].float()[None],
                        torch.tensor([label], device=device),
                    ))
                header = item.get("is_header")
                if header is not None:
                    if not isinstance(header, bool):
                        raise ValueError(f"cell {item.get('id', gt_index)!r} has invalid is_header label")
                    header_logits = outputs["regions"].get("header_logits")
                    if not isinstance(header_logits, torch.Tensor):
                        raise ValueError("model outputs are missing learned table head 'header_logits'")
                    grid_terms.append(F.binary_cross_entropy_with_logits(
                        header_logits[batch_index, proposal_index].float(),
                        torch.tensor(float(header), device=device),
                    ))
            if region_logits.shape[0]:
                weights = torch.ones(region_logits.shape[-1], device=device, dtype=torch.float32)
                weights[0] = 0.1
                region_loss = region_loss + F.cross_entropy(
                    region_logits.float(), region_targets, weight=weights
                )

            # CTC targets attach only through quality-accepted slots from the forward MatchSet.
            accepted_gt_for_proposal.append(
                {pred_i: gt_i for gt_i, pred_i in text_match.accepted_pairs}
            )

            # Relations without annotations are ignored; explicit NULL uses endpoint N.
            node_map: dict[str, int] = {}
            node_metadata: dict[int, Mapping[str, Any]] = {}
            for gt_index, proposal_index in text_match.accepted_pairs:
                item = line_items[gt_index]
                instance_id = item.get("id")
                if isinstance(instance_id, str):
                    node_map[instance_id] = proposal_index
                    node_metadata[proposal_index] = {**item, "_node_type": "line"}
            region_offset = outputs["text"]["boxes"].shape[1]
            for gt_index, proposal_index in region_match.accepted_pairs:
                item = region_items[gt_index]
                instance_id = item.get("id")
                if isinstance(instance_id, str):
                    node_index = region_offset + proposal_index
                    node_map[instance_id] = node_index
                    node_metadata[node_index] = {**item, "_node_type": "cell" if item.get("kind") == "CELL" else "region"}
            graph_nodes = {
                item_id: graph_node
                for item in node_metadata.values()
                if isinstance(item_id := item.get("id"), str)
                if (graph_node := _graph_node(item)) is not None
            }
            node_count = outputs["graph_logits"].shape[-1] - 1
            relations = list(target.get("relations", ()))
            explicit_next_sources = {
                relation.get("source")
                for relation in relations
                if relation.get("type") == "NEXT" and isinstance(relation.get("source"), str)
            }
            line_by_id = {
                line.get("id"): line
                for line in line_items
                if isinstance(line.get("id"), str)
            }
            reading_order = target.get("reading_order", ())
            if line_states[batch_index] == "complete" and isinstance(reading_order, Sequence) and not isinstance(reading_order, (str, bytes)):
                ordered_by_scope: dict[str, list[str]] = {}
                for line_id in reading_order:
                    line = line_by_id.get(line_id) if isinstance(line_id, str) else None
                    if line is None:
                        continue
                    scope = line.get("scope_id", "document")
                    ordered_by_scope.setdefault(scope, []).append(line_id)
                for ordered_ids in ordered_by_scope.values():
                    for order_index, source_id in enumerate(ordered_ids):
                        if source_id in explicit_next_sources:
                            continue
                        target_id = (
                            ordered_ids[order_index + 1]
                            if order_index + 1 < len(ordered_ids)
                            else None
                        )
                        relations.append({"type": "NEXT", "source": source_id, "target": target_id})
            supervised_decisions: set[tuple[str, str, str | None]] = set()
            for relation in relations:
                rel_type = relation.get("type")
                if rel_type not in RELATION_NAMES:
                    continue
                source_id, target_id = relation.get("source"), relation.get("target")
                if rel_type == "CONTAINS":
                    if target_id is None:
                        # CONTAINS is derived from child-to-parent PARENT decisions.
                        # An explicit parent-level NULL has no child endpoint to invert.
                        unmatched_relations += 1
                        continue
                    source_id, target_id = target_id, source_id
                    rel_type = "PARENT"
                decision = (rel_type, source_id, target_id)
                if decision in supervised_decisions:
                    continue
                supervised_decisions.add(decision)
                if not isinstance(source_id, str) or source_id not in node_map:
                    unmatched_relations += 1
                    continue
                source_metadata = node_metadata[node_map[source_id]]
                if not _relation_source_allowed(rel_type, source_metadata):
                    unmatched_relations += 1
                    continue
                target_metadata = None
                if target_id is not None:
                    if target_id == source_id or not isinstance(target_id, str) or target_id not in node_map:
                        unmatched_relations += 1
                        continue
                    target_metadata = node_metadata[node_map[target_id]]
                    if not _relation_target_allowed(
                        rel_type, source_metadata, target_metadata, graph_nodes
                    ):
                        unmatched_relations += 1
                        continue
                rel_index = RELATION_NAMES.index(rel_type)
                source_index = node_map[source_id]
                target_index = node_count if target_id is None else node_map[target_id]
                valid_targets = torch.zeros(node_count + 1, device=device, dtype=torch.bool)
                valid_targets[node_count] = True  # Explicit NULL is valid for every allowed source.
                for candidate_index, candidate_metadata in node_metadata.items():
                    if candidate_index != source_index and _relation_target_allowed(
                        rel_type, source_metadata, candidate_metadata, graph_nodes
                    ):
                        valid_targets[candidate_index] = True
                relation_logits = outputs["graph_logits"][batch_index, rel_index, source_index].float()
                relation_logits = relation_logits.masked_fill(~valid_targets, float("-inf"))
                graph_loss = graph_loss + F.cross_entropy(
                    relation_logits[None],
                    torch.tensor([target_index], device=device),
                )

        def record_ctc_rejection(batch_index: int, proposal_index: int, reason: str, gt_index: int | None = None) -> None:
            target = targets[batch_index]
            line_items = line_items_by_batch[batch_index]
            if gt_index is None:
                gt_index = accepted_gt_for_proposal[batch_index].get(proposal_index)
            line_id = None
            if gt_index is not None and 0 <= gt_index < len(line_items):
                value = line_items[gt_index].get("id")
                line_id = value if isinstance(value, str) else None
            page_id = target.get("page_id", f"batch:{batch_index}")
            rejected_ctc.append({
                "page_id": str(page_id),
                "line_id": line_id,
                "proposal_index": int(proposal_index),
                "reason": str(reason),
            })

        for unsupported_batch, unsupported_proposal, unsupported_reason in unsupported_ctc:
            if (0 <= int(unsupported_batch) < batch_size
                    and line_states[int(unsupported_batch)] != "unavailable"):
                record_ctc_rejection(
                    int(unsupported_batch), int(unsupported_proposal), str(unsupported_reason)
                )

        ctc_by_bucket: dict[int, list[tuple[torch.Tensor, tuple[int, ...]]]] = {}
        for bucket, group in outputs["ocr_groups"].items():
            bucket_samples = ctc_by_bucket.setdefault(int(bucket), [])
            refs = group["refs"].detach().cpu().tolist()
            logits = group["logits"]
            for local_index, (batch_index, proposal_index) in enumerate(refs):
                gt_for_proposal = accepted_gt_for_proposal[batch_index]
                if proposal_index not in gt_for_proposal:
                    continue
                line = line_items_by_batch[batch_index][gt_for_proposal[proposal_index]]
                encoded = self.vocabulary.encode(str(line.get("text", "")))
                oov_count += len(encoded.oov_positions)
                if not encoded.ids:
                    continue
                required = minimum_ctc_frames(encoded.ids)
                if required > int(bucket):
                    reason = f"requires_{required}_frames_but_bucket_is_{bucket}"
                    record_ctc_rejection(batch_index, proposal_index, reason)
                    continue
                bucket_samples.append((logits[local_index].float().log_softmax(-1), encoded.ids))

        if text_box_terms:
            box_loss = box_loss + torch.stack(text_box_terms).mean()
        if region_box_terms:
            box_loss = box_loss + torch.stack(region_box_terms).mean()
        grid_loss = torch.stack(grid_terms).mean() if grid_terms else zero
        if batch_size:
            role_loss = role_loss / max(1, sum(state != "unavailable" for state in line_states))
            region_loss = region_loss / batch_size
            graph_loss = graph_loss / batch_size

        ctc_terms: list[torch.Tensor] = []
        for bucket, samples in ctc_by_bucket.items():
            if not samples:
                continue
            target_lengths = torch.tensor([len(ids) for _, ids in samples], device=device, dtype=torch.long)
            flat_targets = torch.tensor(
                [token for _, ids in samples for token in ids], device=device, dtype=torch.long
            )
            log_probs = torch.stack([logits for logits, _ in samples], dim=1)
            raw_losses = F.ctc_loss(
                log_probs,
                flat_targets,
                torch.full((len(samples),), bucket, device=device, dtype=torch.long),
                target_lengths,
                blank=self.vocabulary.blank_id,
                reduction="none",
                zero_infinity=False,
            )
            ctc_terms.append(raw_losses / target_lengths.clamp_min(1))
        ctc_loss = torch.cat(ctc_terms).mean() if ctc_terms else zero

        total = sum(
            getattr(loss, "sum", lambda: loss)() * self.loss_weights[name]
            for name, loss in {
                "heatmap": heatmap_loss,
                "role": role_loss,
                "region": region_loss,
                "box": box_loss,
                "ctc": ctc_loss,
                "graph": graph_loss,
                "grid": grid_loss,
            }.items()
        )
        return {
            "total": total,
            "heatmap": heatmap_loss,
            "role": role_loss,
            "region": region_loss,
            "box": box_loss,
            "ctc": ctc_loss,
            "graph": graph_loss,
            "grid": grid_loss,
            "unassigned_gt_count": unmatched_gt,
            "geometry_missed_gt_count": missed_gt,
            "detection_recall_text": text_accepted_count / text_gt_count if text_gt_count else 1.0,
            "assignment_coverage_text": text_assigned_count / text_gt_count if text_gt_count else 1.0,
            "heatmap_collision_count": collision_count,
            "line_supervision_page_counts": {
                state: line_states.count(state) for state in ("complete", "partial", "unavailable")
            },
            "rejected_ctc_samples": tuple(sorted(
                rejected_ctc,
                key=lambda item: (item["page_id"], item["line_id"] or "", item["proposal_index"], item["reason"]),
            )),
            "ctc_rejected_sample_count": len(rejected_ctc),
            "oov_character_count": oov_count,
            "unmatched_relation_count": unmatched_relations,
            "grid_supervision_exclusions": tuple(sorted(
                grid_exclusions,
                key=lambda item: (item["page_id"], item["table_id"] or "", item["cell_id"]),
            )),
            "grid_supervision_excluded_cell_count": len(grid_exclusions),
        }
