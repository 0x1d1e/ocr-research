"""Rectangular assignment and immutable per-forward match results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from prismocr.geometry import (
    polygon_area,
    polygon_intersection,
    rotated_box_to_polygon,
    rotated_iou,
)


def linear_sum_assignment(cost: np.ndarray | Sequence[Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Solve a finite rectangular assignment with deterministic tie handling.

    Returns ``min(rows, columns)`` pairs and supports empty dimensions without
    importing SciPy. Non-finite costs are rejected rather than made attractive
    through an arbitrary sentinel value.
    """
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("cost must be a two-dimensional matrix")
    rows, cols = matrix.shape
    if not np.isfinite(matrix).all():
        raise ValueError("assignment costs must all be finite")
    if rows == 0 or cols == 0:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty.copy()

    transposed = rows > cols
    work = matrix.T.copy() if transposed else matrix.copy()
    n, m = work.shape
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(m + 1, dtype=np.float64)
    p = np.zeros(m + 1, dtype=np.int64)
    way = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        minv = np.full(m + 1, np.inf, dtype=np.float64)
        used = np.zeros(m + 1, dtype=np.bool_)
        j0 = 0
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                current = work[i0 - 1, j - 1] - u[i0] - v[j]
                if current < minv[j]:
                    minv[j] = current
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs = [(int(p[j] - 1), j - 1) for j in range(1, m + 1) if p[j] != 0]
    pairs.sort()
    if transposed:
        original_pairs = sorted((col, row) for row, col in pairs)
        return (
            np.asarray([row for row, _ in original_pairs], dtype=np.int64),
            np.asarray([col for _, col in original_pairs], dtype=np.int64),
        )
    return (
        np.asarray([row for row, col in pairs], dtype=np.int64),
        np.asarray([col for row, col in pairs], dtype=np.int64),
    )


def _to_numpy(value: torch.Tensor | np.ndarray | Sequence[float]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        detached = value.detach().to(device="cpu")
        if detached.dtype == torch.bfloat16:
            detached = detached.float()
        return detached.numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class MatchSet:
    """One assignment snapshot, with detection-quality acceptance kept separate."""

    assigned_gt_idx: tuple[int, ...]
    assigned_pred_idx: tuple[int, ...]
    accepted_gt_idx: tuple[int, ...]
    accepted_pred_idx: tuple[int, ...]
    candidate_quality: tuple[float, ...]
    unassigned_gt_idx: tuple[int, ...]
    missed_gt_idx: tuple[int, ...]
    gt_count: int
    valid_pred_count: int

    @property
    def assignment_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(zip(self.assigned_gt_idx, self.assigned_pred_idx))

    @property
    def accepted_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(zip(self.accepted_gt_idx, self.accepted_pred_idx))

    @property
    def assignment_coverage(self) -> float:
        return 1.0 if self.gt_count == 0 else len(self.assigned_gt_idx) / self.gt_count

    @property
    def detection_recall(self) -> float:
        return 1.0 if self.gt_count == 0 else len(self.accepted_gt_idx) / self.gt_count


def match_proposals(
    pred_boxes: torch.Tensor | np.ndarray,
    gt_boxes: torch.Tensor | np.ndarray,
    pred_mask: torch.Tensor | np.ndarray | Sequence[bool] | None = None,
    *,
    image_size: tuple[int, int] = (1, 1),
    pred_class_logits: torch.Tensor | np.ndarray | None = None,
    gt_classes: torch.Tensor | np.ndarray | Sequence[int] | None = None,
    iou_threshold: float = 0.5,
    quality_matrix: torch.Tensor | np.ndarray | None = None,
) -> MatchSet:
    """Match geometry once and separate assignments from accepted detections.

    Candidate indices returned by this function are original proposal indices,
    even when ``pred_mask`` excludes padding. Class logits are an optional cost
    term; a geometry-poor assignment never becomes an accepted detection.
    """
    preds = _to_numpy(pred_boxes).astype(np.float64, copy=False)
    truths = _to_numpy(gt_boxes).astype(np.float64, copy=False)
    if preds.ndim == 1 and preds.size == 0:
        preds = preds.reshape(0, 6)
    if truths.ndim == 1 and truths.size == 0:
        truths = truths.reshape(0, 6)
    if preds.ndim != 2 or preds.shape[-1] != 6:
        raise ValueError("pred_boxes must have shape [K, 6]")
    if truths.ndim != 2 or truths.shape[-1] != 6:
        raise ValueError("gt_boxes must have shape [M, 6]")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")

    if pred_mask is None:
        valid = np.ones(len(preds), dtype=np.bool_)
    else:
        valid = _to_numpy(pred_mask).astype(np.bool_, copy=False)
        if valid.shape != (len(preds),):
            raise ValueError("pred_mask must have shape [K]")
    candidate_indices = np.flatnonzero(valid)
    gt_count = len(truths)
    if gt_count == 0 or len(candidate_indices) == 0:
        unassigned = tuple(range(gt_count))
        return MatchSet((), (), (), (), (), unassigned, unassigned, gt_count, len(candidate_indices))

    logits = None if pred_class_logits is None else _to_numpy(pred_class_logits).astype(np.float64)
    classes = None if gt_classes is None else _to_numpy(gt_classes).astype(np.int64)
    if logits is not None and logits.shape[0] != len(preds):
        raise ValueError("pred_class_logits first dimension must equal K")
    if classes is not None and classes.shape != (gt_count,):
        raise ValueError("gt_classes must have shape [M]")
    if (logits is None) != (classes is None):
        raise ValueError("pred_class_logits and gt_classes must be supplied together")
    supplied_quality = None if quality_matrix is None else _to_numpy(quality_matrix).astype(np.float64)
    if supplied_quality is not None and (
        supplied_quality.shape != (gt_count, len(preds))
        or not np.isfinite(supplied_quality).all()
        or np.any(supplied_quality < 0)
        or np.any(supplied_quality > 1)
    ):
        raise ValueError("quality_matrix must be finite [M,K] values in [0,1]")

    cost = np.zeros((gt_count, len(candidate_indices)), dtype=np.float64)
    qualities = np.zeros_like(cost)
    width_px, height_px = image_size
    if supplied_quality is None:
        gt_polys = []
        gt_aabbs = []
        gt_areas = []
        for gt in truths:
            try:
                poly = rotated_box_to_polygon(gt, image_size)
                area = polygon_area(poly)
                if area > 0:
                    xs = (poly[0][0], poly[1][0], poly[2][0], poly[3][0])
                    ys = (poly[0][1], poly[1][1], poly[2][1], poly[3][1])
                    aabb = (min(xs), min(ys), max(xs), max(ys))
                else:
                    poly, aabb, area = None, None, 0.0
            except (TypeError, ValueError, OverflowError):
                poly, aabb, area = None, None, 0.0
            gt_polys.append(poly)
            gt_aabbs.append(aabb)
            gt_areas.append(area)

        pred_polys = []
        pred_aabbs = []
        pred_areas = []
        for pred_i in candidate_indices:
            pred = preds[pred_i]
            try:
                poly = rotated_box_to_polygon(pred, image_size)
                area = polygon_area(poly)
                if area > 0:
                    xs = (poly[0][0], poly[1][0], poly[2][0], poly[3][0])
                    ys = (poly[0][1], poly[1][1], poly[2][1], poly[3][1])
                    aabb = (min(xs), min(ys), max(xs), max(ys))
                else:
                    poly, aabb, area = None, None, 0.0
            except (TypeError, ValueError, OverflowError):
                poly, aabb, area = None, None, 0.0
            pred_polys.append(poly)
            pred_aabbs.append(aabb)
            pred_areas.append(area)

    for gt_i, gt in enumerate(truths):
        gt_cx, gt_cy, gt_w, gt_h = gt[:4]
        gt_aabb = None if supplied_quality is not None else gt_aabbs[gt_i]
        gt_poly = None if supplied_quality is not None else gt_polys[gt_i]
        gt_area = 0.0 if supplied_quality is not None else gt_areas[gt_i]
        for local_i, pred_i in enumerate(candidate_indices):
            pred = preds[pred_i]
            if supplied_quality is not None:
                iou = float(supplied_quality[gt_i, pred_i])
            elif gt_aabb is None or pred_aabbs[local_i] is None:
                iou = 0.0
            else:
                gt_min_x, gt_min_y, gt_max_x, gt_max_y = gt_aabb
                pred_min_x, pred_min_y, pred_max_x, pred_max_y = pred_aabbs[local_i]
                if (
                    gt_max_x < pred_min_x
                    or pred_max_x < gt_min_x
                    or gt_max_y < pred_min_y
                    or pred_max_y < gt_min_y
                ):
                    iou = 0.0
                else:
                    intersection = polygon_area(polygon_intersection(gt_poly, pred_polys[local_i]))
                    iou = min(1.0, max(0.0, intersection / max(1e-12, gt_area + pred_areas[local_i] - intersection)))
            qualities[gt_i, local_i] = iou
            dx = (pred[0] - gt_cx) * width_px
            dy = (pred[1] - gt_cy) * height_px
            scale = max(1.0, math_sqrt_positive(gt_w * width_px * gt_h * height_px))
            geometric = 1.0 - iou + 0.05 * min(10.0, math_sqrt_positive(dx * dx + dy * dy) / scale)
            class_cost = 0.0
            if logits is not None and classes is not None:
                class_id = int(classes[gt_i])
                if class_id < 0 or class_id >= logits.shape[1]:
                    raise ValueError(f"ground-truth class {class_id} is out of range")
                row = logits[pred_i]
                shifted = row - np.max(row)
                log_probs = shifted - np.log(np.exp(shifted).sum())
                class_cost = -float(log_probs[class_id]) * 0.05
            cost[gt_i, local_i] = geometric + class_cost

    # First maximize the number of pairs that satisfy the geometric detection
    # threshold, then minimize the original assignment cost among those pairs.
    pair_count = min(cost.shape)
    cost_range = float(cost.max() - cost.min()) if cost.size else 0.0
    invalid_pair_penalty = (cost_range + 1.0) * (pair_count + 1)
    assignment_cost = cost + (qualities < iou_threshold) * invalid_pair_penalty
    gt_rows, candidate_cols = linear_sum_assignment(assignment_cost)
    assigned_gt = tuple(int(i) for i in gt_rows)
    assigned_pred = tuple(int(candidate_indices[j]) for j in candidate_cols)
    matched_quality = tuple(float(qualities[i, j]) for i, j in zip(gt_rows, candidate_cols))
    accepted_pairs = [
        (gt_i, pred_i, quality)
        for gt_i, pred_i, quality in zip(assigned_gt, assigned_pred, matched_quality)
        if quality >= iou_threshold
    ]
    assigned_set = set(assigned_gt)
    accepted_set = {gt_i for gt_i, _, _ in accepted_pairs}
    unassigned_gt = tuple(i for i in range(gt_count) if i not in assigned_set)
    missed_gt = tuple(i for i in range(gt_count) if i not in accepted_set)
    return MatchSet(
        assigned_gt,
        assigned_pred,
        tuple(item[0] for item in accepted_pairs),
        tuple(item[1] for item in accepted_pairs),
        matched_quality,
        unassigned_gt,
        missed_gt,
        gt_count,
        len(candidate_indices),
    )


def math_sqrt_positive(value: float) -> float:
    return float(np.sqrt(max(0.0, float(value))))
