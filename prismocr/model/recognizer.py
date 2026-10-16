"""Compact, length-bucketed oriented strip sampling and CTC recognizers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from prismocr.model.heads import validate_model_contract_version

STRIP_BUCKETS = (64, 128, 160, 256)


def validate_strip_buckets(buckets: Sequence[int]) -> tuple[int, ...]:
    values = tuple(buckets)
    if not values or any(type(bucket) is not int or bucket <= 0 for bucket in values):
        raise ValueError("strip_buckets must contain positive integers")
    if tuple(sorted(set(values))) != values:
        raise ValueError("strip_buckets must be strictly increasing")
    return values


def minimum_ctc_frames(token_ids: Sequence[int]) -> int:
    return len(token_ids) + sum(token_ids[i] == token_ids[i - 1] for i in range(1, len(token_ids)))


def choose_inference_bucket(
    box: torch.Tensor,
    image_size: tuple[int, int],
    buckets: Sequence[int] = STRIP_BUCKETS,
    *, model_contract_version: int = 1,
) -> int | None:
    """Legacy geometry policy or contract-2 fixed maximum; never reads labels."""
    if validate_model_contract_version(model_contract_version) == 2:
        return max(validate_strip_buckets(buckets))
    width_px, height_px = image_size
    line_width = float(box[2].detach().cpu()) * width_px
    line_height = float(box[3].detach().cpu()) * height_px
    estimated_chars = max(1.0, line_width / max(1.0, 0.55 * line_height))
    estimate = int(2 * estimated_chars + 2)
    return next((bucket for bucket in validate_strip_buckets(buckets) if bucket >= estimate), None)


def sample_oriented_strips(
    features: torch.Tensor,
    boxes: torch.Tensor,
    refs: torch.Tensor,
    length: int,
    *,
    strip_height: int = 5,
) -> torch.Tensor:
    """Sample selected oriented lines only, respecting align_corners=False geometry.

    Boxes use normalized center/size and pixel-space orientation. ``refs`` stores
    (batch_index, proposal_index) pairs and is used to gather only the requested
    feature maps. The result has shape [N, L, D].
    """
    if boxes.ndim != 2 or boxes.shape[-1] != 6 or refs.ndim != 2 or refs.shape[-1] != 2:
        raise ValueError("boxes and refs must have shapes [N,6] and [N,2]")
    proposal_count = boxes.shape[0]
    if refs.shape[0] != proposal_count:
        raise ValueError("boxes and refs must have the same proposal count")
    if length <= 0 or strip_height <= 0:
        raise ValueError("strip dimensions must be positive")
    if proposal_count == 0:
        return features.new_zeros((0, length, features.shape[1]))

    batch_indices = refs[:, 0].to(device=features.device, dtype=torch.long)
    boxes = boxes.to(device=features.device, dtype=features.dtype)
    image_h = features.shape[-2] * 4
    image_w = features.shape[-1] * 4

    sin_theta, cos_theta = boxes[:, 4], boxes[:, 5]
    norm = torch.sqrt(sin_theta.square() + cos_theta.square()).clamp_min(1e-6)
    sin_theta, cos_theta = sin_theta / norm, cos_theta / norm
    # Rotated boxes are geometrically equivalent after a 180-degree turn, but
    # OCR is directional. Use the dataset's canonical [-pi/2, pi/2) scan axis
    # so antipodal detector predictions cannot reverse the transcript.
    reverse_axis = (cos_theta < 0) | ((cos_theta == 0) & (sin_theta > 0))
    direction = torch.where(reverse_axis, -torch.ones_like(cos_theta), torch.ones_like(cos_theta))
    sin_theta, cos_theta = sin_theta * direction, cos_theta * direction
    theta = torch.zeros((proposal_count, 2, 3), device=features.device, dtype=features.dtype)
    theta[:, 0, 0] = cos_theta * boxes[:, 2]
    theta[:, 0, 1] = -sin_theta * boxes[:, 3] * (image_h / image_w)
    theta[:, 0, 2] = 2 * boxes[:, 0] - 1
    theta[:, 1, 0] = sin_theta * boxes[:, 2] * (image_w / image_h)
    theta[:, 1, 1] = cos_theta * boxes[:, 3]
    theta[:, 1, 2] = 2 * boxes[:, 1] - 1

    ys = torch.linspace(-1.0, 1.0, strip_height, device=features.device, dtype=features.dtype)
    xs = torch.linspace(-1.0, 1.0, length, device=features.device, dtype=features.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    local = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=-1).to(dtype=features.dtype)
    grid = torch.einsum("nhwc,nkc->nhwk", local.unsqueeze(0).expand(proposal_count, -1, -1, -1), theta)

    # Gather only the sampled pixels. Expanding an entire page feature map per
    # proposal makes dense pages consume O(lines * channels * page pixels) memory.
    # This is the align_corners=False bilinear coordinate transform used by
    # grid_sample with zero padding, expressed as four sparse gathers instead.
    feature_height, feature_width = features.shape[-2:]
    x = ((grid[..., 0] + 1) * feature_width - 1) * 0.5
    y = ((grid[..., 1] + 1) * feature_height - 1) * 0.5
    x0, y0 = torch.floor(x).to(torch.long), torch.floor(y).to(torch.long)
    x1, y1 = x0 + 1, y0 + 1
    dx, dy = x - x0.to(x.dtype), y - y0.to(y.dtype)
    flattened = features.flatten(2)
    channel_indices = torch.arange(features.shape[1], device=features.device)
    output_height, output_width = grid.shape[1:3]

    def gather(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
        valid = (px >= 0) & (px < feature_width) & (py >= 0) & (py < feature_height)
        locations = py.clamp(0, feature_height - 1) * feature_width + px.clamp(0, feature_width - 1)
        values = flattened[
            batch_indices[:, None, None],
            channel_indices[None, :, None],
            locations.flatten(1)[:, None, :],
        ]
        values = values * valid.flatten(1)[:, None, :].to(values.dtype)
        return values.reshape(proposal_count, features.shape[1], output_height, output_width)

    top_left = gather(x0, y0) * ((1 - dx) * (1 - dy))[:, None]
    top_right = gather(x1, y0) * (dx * (1 - dy))[:, None]
    bottom_left = gather(x0, y1) * ((1 - dx) * dy)[:, None]
    bottom_right = gather(x1, y1) * (dx * dy)[:, None]
    return top_left + top_right + bottom_left + bottom_right


class TransformerSequenceBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 5) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(value)
        value = value + self.attention(normalized, normalized, normalized, need_weights=False)[0]
        return value + self.ffn(self.norm2(value))


class GatedConvSequenceBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.mix = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.SiLU(), nn.Linear(dim * 2, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        mixed = self.mix(self.depthwise(value.transpose(1, 2))).transpose(1, 2)
        left, gate = mixed.chunk(2, dim=-1)
        value = value + left * gate.sigmoid()
        return value + self.ffn(self.norm2(value))


class SparseCTCRecognizer(nn.Module):
    """Recognize only the compact per-bucket list supplied by the model."""

    def __init__(self, dim: int, vocab_size: int, strip_height: int = 5, variant: str = "transformer") -> None:
        super().__init__()
        if variant not in {"transformer", "gated_conv"}:
            raise ValueError("recognizer variant must be 'transformer' or 'gated_conv'")
        self.dim = dim
        self.strip_height = strip_height
        self.variant = variant
        self.vertical_collapse = nn.Conv2d(dim, dim, kernel_size=(strip_height, 1), groups=dim)
        block_type = TransformerSequenceBlock if variant == "transformer" else GatedConvSequenceBlock
        self.blocks = nn.ModuleList(block_type(dim) for _ in range(2))
        self.classifier = nn.Linear(dim, vocab_size)

    def forward_bucket(
        self,
        features: torch.Tensor,
        boxes: torch.Tensor,
        node_embeddings: torch.Tensor,
        refs: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        if len(refs) == 0:
            return features.new_zeros((0, length, self.classifier.out_features))
        patches = sample_oriented_strips(
            features, boxes, refs, length, strip_height=self.strip_height
        )
        strips = F.silu(self.vertical_collapse(patches)).squeeze(2).transpose(1, 2)
        strips = strips + node_embeddings.unsqueeze(1)
        for block in self.blocks:
            strips = block(strips)
        return self.classifier(strips)
