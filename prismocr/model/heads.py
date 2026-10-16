"""Typed text/structure proposal heads and sparse relation scoring."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def validate_model_contract_version(version: int) -> int:
    if type(version) is not int or version not in (1, 2):
        raise ValueError("model_contract_version must be integer 1 or 2")
    return version


def _multihead_attention(
    attention: nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Run batch-first self/cross attention with sequence-dynamic tensor shapes."""
    if attention.in_proj_weight is None or not attention._qkv_same_embed_dim:
        raise ValueError("dynamic attention requires shared query/key/value dimensions")
    embed_dim = attention.embed_dim
    heads = attention.num_heads
    head_dim = embed_dim // heads
    if attention.in_proj_bias is None:
        query_bias = key_bias = value_bias = None
    else:
        query_bias, key_bias, value_bias = attention.in_proj_bias.chunk(3)
    query_weight, key_weight, value_weight = attention.in_proj_weight.chunk(3)
    batch = query.shape[0]
    q_len, kv_len = query.shape[1], key.shape[1]
    q = F.linear(query, query_weight, query_bias).reshape(batch, q_len, heads, head_dim).transpose(1, 2)
    k = F.linear(key, key_weight, key_bias).reshape(batch, kv_len, heads, head_dim).transpose(1, 2)
    v = F.linear(value, value_weight, value_bias).reshape(batch, kv_len, heads, head_dim).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)
    probabilities = F.softmax(scores, dim=-1)
    probabilities = F.dropout(probabilities, p=attention.dropout, training=attention.training)
    attended = torch.matmul(probabilities, v).transpose(1, 2).reshape(batch, q_len, embed_dim)
    return attention.out_proj(attended)


def _decode_region_queries(
    decoder: nn.TransformerDecoder,
    target: torch.Tensor,
    memory: torch.Tensor,
) -> torch.Tensor:
    """Match norm-first TransformerDecoderLayer without trace-fixed MHA lengths."""
    hidden = target
    for layer in decoder.layers:
        if not layer.norm_first:
            raise ValueError("dynamic region decoder requires norm_first=True")
        normalized = layer.norm1(hidden)
        hidden = hidden + layer.dropout1(
            _multihead_attention(layer.self_attn, normalized, normalized, normalized)
        )
        hidden = hidden + layer.dropout2(
            _multihead_attention(layer.multihead_attn, layer.norm2(hidden), memory, memory)
        )
        hidden = hidden + layer._ff_block(layer.norm3(hidden))
    if decoder.norm is not None:
        hidden = decoder.norm(hidden)
    return hidden


class TextLineHead(nn.Module):
    def __init__(self, in_channels: int, dim: int, num_roles: int, *, model_contract_version: int = 1) -> None:
        super().__init__()
        self.neck = nn.Sequential(nn.Conv2d(in_channels, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.SiLU())
        self.heatmap = nn.Conv2d(dim, 1, 1)
        self.box = nn.Conv2d(dim, 6, 1)
        self.role = nn.Conv2d(dim, num_roles, 1)
        self.model_contract_version = validate_model_contract_version(model_contract_version)
        if self.model_contract_version == 2:
            # Fresh heads only: loading a state dict always overwrites these values.
            nn.init.zeros_(self.box.weight)
            with torch.no_grad():
                self.box.bias.copy_(self.box.bias.new_tensor([
                    0.0, 0.0, 2 + math.log(math.expm1(64 / 256)),
                    2 + math.log(math.expm1(16 / 256)), 0.0, 1.0,
                ]))

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.neck(features)
        return {
            "heatmap_logits": self.heatmap(hidden),
            "box_map": self.box(hidden),
            "role_map": self.role(hidden),
            "feature_map": hidden,
        }

    @staticmethod
    def decode_sizes(
        values: torch.Tensor, *, model_contract_version: int = 1,
        image_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Decode final normalized sides once for proposals and suppression.

        Keep actual input dimensions symbolic for dynamic ONNX graphs.
        """
        version = validate_model_contract_version(model_contract_version)
        sizes = F.softplus(values - 2.0).clamp(min=1e-4)
        if version == 1:
            return sizes
        if image_size is None:
            raise ValueError("contract 2 size decoding requires actual input image_size")
        return torch.stack((sizes[..., 0] * 256 / image_size[0],
                            sizes[..., 1] * 256 / image_size[1]), dim=-1)

    @staticmethod
    def proposals(
        raw: dict[str, torch.Tensor],
        *,
        max_proposals: int,
        score_threshold: float,
        model_contract_version: int = 1,
        image_size: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        version = validate_model_contract_version(model_contract_version)
        logits = raw["heatmap_logits"]
        box_map = raw["box_map"]
        role_map = raw["role_map"]
        embeddings = raw["feature_map"]
        batch, _, height, width = logits.shape
        total = height * width
        probabilities = logits.sigmoid()
        local_max = probabilities == F.max_pool2d(probabilities, kernel_size=3, stride=1, padding=1)
        # Keep one deterministic representative from equal-valued plateaus. Without
        # this tie break a flat heatmap makes every pixel a proposal peak.
        padded = F.pad(probabilities, (1, 1, 1, 1), value=float("-inf"))
        has_earlier_equal_neighbor = torch.zeros_like(local_max)
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1)):
            neighbor = padded[..., 1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
            has_earlier_equal_neighbor = has_earlier_equal_neighbor | (probabilities == neighbor)
        local_max = local_max & ~has_earlier_equal_neighbor

        # A 3x3 maximum suppresses a weaker adjacent line even when both peaks
        # decode to distinct centers. Recover those neighbors with a local center
        # sweep, while retaining ordinary same-line suppression. Equal-valued
        # plateaus remain represented by their stable local-maximum winner.
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=logits.device, dtype=box_map.dtype),
            torch.arange(width, device=logits.device, dtype=box_map.dtype),
            indexing="ij",
        )
        centers_x = (grid_x[None] + box_map[:, 0].sigmoid()) / max(width, 1)
        centers_y = (grid_y[None] + box_map[:, 1].sigmoid()) / max(height, 1)
        line_heights = TextLineHead.decode_sizes(
            box_map[:, 2:4].permute(0, 2, 3, 1),
            model_contract_version=version, image_size=image_size,
        )[..., 1]
        padded_scores = F.pad(probabilities[:, 0], (1, 1, 1, 1), value=float("-inf"))
        padded_local_max = F.pad(local_max[:, 0], (1, 1, 1, 1), value=False)
        padded_centers_x = F.pad(centers_x, (1, 1, 1, 1), value=0.0)
        padded_centers_y = F.pad(centers_y, (1, 1, 1, 1), value=0.0)
        padded_heights = F.pad(line_heights, (1, 1, 1, 1), value=0.0)
        # Suppress a neighbor only when a higher-ranked proposal decodes to the
        # same center. This center-aware local NMS retains arbitrarily long chains
        # of distinct adjacent lines without iterative, size-dependent sweeps.
        blocked_by_same_center = torch.zeros_like(local_max[:, 0])
        for dy, dx in (
            (-1, -1), (-1, 0), (-1, 1), (0, -1),
            (0, 1), (1, -1), (1, 0), (1, 1),
        ):
            neighbor_scores = padded_scores[:, 1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
            neighbor_peaks = padded_local_max[:, 1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
            neighbor_x = padded_centers_x[:, 1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
            neighbor_y = padded_centers_y[:, 1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
            neighbor_height = padded_heights[:, 1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
            is_stronger_peak = neighbor_peaks & (neighbor_scores >= probabilities[:, 0])
            distance_width, distance_height = (width, height) if version == 1 else image_size
            delta_x = (centers_x - neighbor_x) * distance_width
            delta_y = (centers_y - neighbor_y) * distance_height
            same_center_radius = 0.5 * torch.minimum(line_heights * distance_height, neighbor_height * distance_height)
            same_center = delta_x.square() + delta_y.square() <= same_center_radius.square()
            blocked_by_same_center = blocked_by_same_center | (is_stronger_peak & same_center)
        recovered = (probabilities[:, 0] >= score_threshold) & ~blocked_by_same_center
        proposal_peaks = local_max[:, 0] | recovered
        qualified = proposal_peaks & (probabilities[:, 0] >= score_threshold)
        peaks = probabilities[:, 0].masked_fill(~proposal_peaks, float("-inf")).flatten(1)
        candidate_count = qualified.flatten(1).sum(dim=1)
        # Always provide at least K sortable entries without tracing a shape-dependent
        # Python branch. Runtime inputs with fewer than K spatial sites select invalid
        # -inf sentinels that are masked and mapped to safe gather index zero below.
        sentinels = peaks.new_full((batch, max_proposals), float("-inf"))
        ranked_peaks = torch.cat((peaks, sentinels), dim=1)
        boundary = torch.topk(ranked_peaks, max_proposals, dim=1, sorted=True).values[:, -1:]
        strictly_better = ranked_peaks > boundary
        tied = ranked_peaks == boundary
        needed_from_tie = max_proposals - strictly_better.sum(dim=1)
        tie_ranks = tied.to(torch.int64).cumsum(dim=1)
        keep = strictly_better | (tied & (tie_ranks <= needed_from_tie[:, None]))
        selected = keep.nonzero(as_tuple=False)
        indices = selected[:, 1].reshape(batch, max_proposals)
        scores = ranked_peaks.gather(1, indices)
        valid = (indices < total) & torch.isfinite(scores)
        scores = torch.where(valid, scores, torch.zeros_like(scores))
        # Stable descending ranks use pairwise comparisons rather than argsort,
        # which keeps the same proposal order and exports to the fixed ONNX matrix.
        order = torch.arange(max_proposals, device=indices.device).view(1, max_proposals, 1)
        score_i = scores.unsqueeze(-1)
        score_j = scores.unsqueeze(-2)
        precedes = (score_j > score_i) | ((score_j == score_i) & (order.transpose(1, 2) < order))
        stable_order = precedes.sum(dim=-1)
        indices = torch.zeros_like(indices).scatter(1, stable_order, indices)
        scores = torch.zeros_like(scores).scatter(1, stable_order, scores)
        valid = torch.zeros_like(valid).scatter(1, stable_order, valid)

        indices = torch.where(valid, indices, torch.zeros_like(indices))
        x = (indices % width).to(box_map.dtype)
        y = (indices // width).to(box_map.dtype)
        box_values = box_map.flatten(2).gather(2, indices[:, None, :].expand(-1, 6, -1)).transpose(1, 2)
        role_values = role_map.flatten(2).gather(
            2, indices[:, None, :].expand(-1, role_map.shape[1], -1)
        ).transpose(1, 2)
        embedding_values = embeddings.flatten(2).gather(
            2, indices[:, None, :].expand(-1, embeddings.shape[1], -1)
        ).transpose(1, 2)

        center_x = (x + box_values[..., 0].sigmoid()) / max(width, 1)
        center_y = (y + box_values[..., 1].sigmoid()) / max(height, 1)
        # Oriented side lengths are normalized against page axes, so valid
        # rectangles may exceed 0.5 or even 1.0 on an axis.
        sizes = TextLineHead.decode_sizes(
            box_values[..., 2:4], model_contract_version=version, image_size=image_size
        )
        orientation = box_values[..., 4:6]
        norm = torch.linalg.vector_norm(orientation, dim=-1, keepdim=True)
        fallback = torch.zeros_like(orientation)
        fallback[..., 1] = 1.0
        orientation = torch.where(norm > 1e-6, orientation / norm.clamp_min(1e-6), fallback)
        boxes = torch.cat((center_x[..., None], center_y[..., None], sizes, orientation), dim=-1)
        accepted = valid & (scores >= score_threshold)
        return {
            "boxes": boxes,
            "scores": scores,
            "candidate_mask": valid,
            "mask": accepted,
            "role_logits": role_values,
            "embeddings": embedding_values,
            "proposal_indices": indices,
            "candidate_count": candidate_count,
            "capacity_overflow": candidate_count > max_proposals,
        }


class RegionQueryHead(nn.Module):
    """Typed query decoder; nested region and cell boxes are never cross-suppressed."""

    def __init__(
        self,
        dim: int,
        num_queries: int,
        num_kinds: int,
        *,
        max_table_rows: int = 128,
        max_table_cols: int = 128,
        max_cell_span: int = 32,
    ) -> None:
        super().__init__()
        if min(max_table_rows, max_table_cols, max_cell_span) <= 0:
            raise ValueError("table grid class counts must be positive")
        self.num_queries = num_queries
        self.max_table_rows = max_table_rows
        self.max_table_cols = max_table_cols
        self.max_cell_span = max_cell_span
        self.query = nn.Embedding(num_queries, dim)
        self.memory_position = nn.Sequential(nn.Linear(2, dim), nn.SiLU(), nn.Linear(dim, dim))
        layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=5 if dim % 5 == 0 else 4,
            dim_feedforward=dim * 2,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=2)
        self.classifier = nn.Linear(dim, num_kinds)
        self.box = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 6))
        self.row_classifier = nn.Linear(dim, max_table_rows)
        self.col_classifier = nn.Linear(dim, max_table_cols)
        self.rowspan_classifier = nn.Linear(dim, max_cell_span)
        self.colspan_classifier = nn.Linear(dim, max_cell_span)
        self.header_classifier = nn.Linear(dim, 1)

    def forward(self, features: torch.Tensor, score_threshold: float) -> dict[str, torch.Tensor]:
        batch, channels, height, width = features.shape
        memory = features.flatten(2).transpose(1, 2)
        y_positions = torch.arange(height, device=features.device, dtype=features.dtype)
        x_positions = torch.arange(width, device=features.device, dtype=features.dtype)
        ys = y_positions / y_positions[-1].clamp_min(1.0)
        xs = x_positions / x_positions[-1].clamp_min(1.0)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        position = self.memory_position(torch.stack((xx, yy), dim=-1).reshape(-1, 2))
        memory = memory + position.unsqueeze(0)
        queries = self.query.weight.unsqueeze(0).expand(batch, -1, -1)
        hidden = _decode_region_queries(self.decoder, queries, memory)

        raw_box = self.box(hidden)
        center = raw_box[..., :2].sigmoid()
        size = F.softplus(raw_box[..., 2:4] - 2.0).clamp(min=1e-4)
        orient = raw_box[..., 4:6]
        norm = torch.linalg.vector_norm(orient, dim=-1, keepdim=True)
        fallback = torch.zeros_like(orient)
        fallback[..., 1] = 1.0
        orient = torch.where(norm > 1e-6, orient / norm.clamp_min(1e-6), fallback)
        logits = self.classifier(hidden)
        probabilities = logits.softmax(dim=-1)
        scores, kinds = probabilities[..., 1:].max(dim=-1)
        kinds = kinds + 1
        accepted = (scores >= score_threshold) & (logits.argmax(dim=-1) != 0)
        candidate_mask = torch.ones_like(accepted, dtype=torch.bool)
        return {
            "boxes": torch.cat((center, size, orient), dim=-1),
            "class_logits": logits,
            "scores": scores,
            "kinds": kinds,
            "row_logits": self.row_classifier(hidden),
            "col_logits": self.col_classifier(hidden),
            "rowspan_logits": self.rowspan_classifier(hidden),
            "colspan_logits": self.colspan_classifier(hidden),
            "header_logits": self.header_classifier(hidden).squeeze(-1),
            "candidate_mask": candidate_mask,
            "mask": accepted,
            "embeddings": hidden,
        }


class PairRelationHead(nn.Module):
    """Score typed relation channels and one explicit NULL endpoint per source."""

    def __init__(self, dim: int, num_relations: int = 5) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.query = nn.Linear(dim, dim * num_relations, bias=False)
        self.key = nn.Linear(dim, dim * num_relations, bias=False)
        self.null = nn.Linear(dim, num_relations)
        self.scale = dim**-0.5

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        batch, count, dim = nodes.shape
        q = self.query(nodes).reshape(batch, count, self.num_relations, dim).permute(0, 2, 1, 3)
        k = self.key(nodes).reshape(batch, count, self.num_relations, dim).permute(0, 2, 1, 3)
        pair_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        null_score = self.null(nodes).permute(0, 2, 1).unsqueeze(-1)
        return torch.cat((pair_scores, null_score), dim=-1)
