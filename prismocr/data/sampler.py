"""Aspect-ratio / page-shape batch bucketing for training.

Prevents GPU utilization oscillations and memory fragmentation caused by mixing
arbitrary aspect ratios (e.g., portrait, square, landscape) in the same batch,
which otherwise forces `collate_pages` to apply excessive empty canvas padding.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch.utils.data import Sampler


DEFAULT_ASPECT_RATIO_BOUNDARIES: tuple[float, ...] = (0.85, 1.15)


def validate_bucket_boundaries(boundaries: Sequence[float]) -> tuple[float, ...]:
    """Validate that bucket boundaries are positive, finite, and strictly increasing."""
    if not isinstance(boundaries, (list, tuple)) or not boundaries:
        raise ValueError("bucket boundaries must be a non-empty sequence of numbers")
    parsed: list[float] = []
    for b in boundaries:
        if isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(float(b)) or float(b) <= 0:
            raise ValueError(f"bucket boundary must be a positive finite number, got {b!r}")
        parsed.append(float(b))
    if parsed != sorted(parsed) or len(set(parsed)) != len(parsed):
        raise ValueError(f"bucket boundaries must be strictly increasing, got {boundaries!r}")
    return tuple(parsed)


def compute_scaled_dimensions(
    raw_width: int,
    raw_height: int,
    max_page_side: int | None = None,
) -> tuple[int, int]:
    """Compute (width, height) after applying optional max_page_side downscaling."""
    if raw_width <= 0 or raw_height <= 0:
        raise ValueError(f"dimensions must be positive, got width={raw_width}, height={raw_height}")
    if max_page_side is None or max(raw_width, raw_height) <= max_page_side:
        return raw_width, raw_height
    scale = max_page_side / max(raw_width, raw_height)
    return max(1, round(raw_width * scale)), max(1, round(raw_height * scale))


def extract_record_aspect_ratio(
    record: dict[str, Any] | Any,
    max_page_side: int | None = None,
) -> float:
    """Extract post-scaling aspect ratio (width / height) from a dataset record."""
    annotation = record.get("_annotation") if isinstance(record, dict) else None
    if annotation is None and isinstance(record, dict):
        annotation = record.get("annotation", record)
    image_info = annotation.get("image", {}) if isinstance(annotation, dict) else {}
    raw_w = int(image_info.get("width", 1))
    raw_h = int(image_info.get("height", 1))
    scaled_w, scaled_h = compute_scaled_dimensions(raw_w, raw_h, max_page_side)
    return float(scaled_w) / float(scaled_h)


def extract_aspect_ratios_from_dataset(
    dataset: Any,
    *,
    max_page_side: int | None = None,
) -> list[float]:
    """Extract aspect ratios from PrismPageDataset, Subset, or sequence of records."""
    if isinstance(dataset, torch.utils.data.Subset):
        parent_ratios = extract_aspect_ratios_from_dataset(dataset.dataset, max_page_side=max_page_side)
        return [parent_ratios[i] for i in dataset.indices]

    mps = getattr(dataset, "max_page_side", None)
    if mps is None:
        mps = max_page_side
    records = getattr(dataset, "records", None)
    if records is not None and isinstance(records, Sequence):
        return [extract_record_aspect_ratio(record, mps) for record in records]

    if isinstance(dataset, Sequence) and len(dataset) > 0 and isinstance(dataset[0], dict):
        return [extract_record_aspect_ratio(item, mps) for item in dataset]

    # Fallback for generic datasets with known length:
    return [1.0] * len(dataset)


def assign_bucket(aspect_ratio: float, boundaries: Sequence[float]) -> int:
    """Assign aspect ratio to a bucket index based on ascending boundaries.

    For boundaries (b_0, b_1, ..., b_{k-1}):
      bucket 0: ar < b_0
      bucket i: b_{i-1} <= ar < b_i
      bucket k: ar >= b_{k-1}
    """
    for index, boundary in enumerate(boundaries):
        if aspect_ratio < boundary:
            return index
    return len(boundaries)


class AspectRatioBatchSampler(Sampler[list[int]]):
    """Batch sampler that groups samples by aspect ratio into buckets before batching.

    Modeled after fintech_ocr's AspectRatioBatchSampler, but operates on page
    aspect ratios and supports deterministic PyTorch generator shuffling for
    exact checkpoint resume and RNG reproducibility.

    Samples within each bucket are shuffled, chunked into batches of `batch_size`,
    and the resulting batches across all buckets are shuffled together so that
    batches of different page shapes interleave evenly across training steps.
    """

    DEFAULT_BOUNDARIES: tuple[float, ...] = DEFAULT_ASPECT_RATIO_BOUNDARIES

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        *,
        aspect_ratios: Sequence[float] | None = None,
        boundaries: Sequence[float] | None = None,
        drop_last: bool = False,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
        max_page_side: int | None = None,
    ) -> None:
        super().__init__()
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
        self.batch_size = batch_size
        self.drop_last = bool(drop_last)
        self.shuffle = bool(shuffle)
        self.generator = generator

        if boundaries is None:
            self.boundaries = self.DEFAULT_BOUNDARIES
        else:
            self.boundaries = validate_bucket_boundaries(boundaries)

        if isinstance(dataset, int):
            if dataset < 0:
                raise ValueError("dataset size cannot be negative")
            self.dataset_size = dataset
            if aspect_ratios is None:
                self.aspect_ratios = [1.0] * dataset
            else:
                self.aspect_ratios = list(aspect_ratios)
        else:
            self.dataset_size = len(dataset)
            if aspect_ratios is None:
                self.aspect_ratios = extract_aspect_ratios_from_dataset(dataset, max_page_side=max_page_side)
            else:
                self.aspect_ratios = list(aspect_ratios)

        if len(self.aspect_ratios) != self.dataset_size:
            raise ValueError(
                f"aspect_ratios length ({len(self.aspect_ratios)}) does not match dataset size ({self.dataset_size})"
            )

        for idx, ar in enumerate(self.aspect_ratios):
            if isinstance(ar, bool) or not isinstance(ar, (int, float)) or not math.isfinite(float(ar)) or float(ar) <= 0:
                raise ValueError(f"aspect ratio at index {idx} must be a positive finite number, got {ar!r}")

        self._bucket_indices: dict[int, list[int]] = {i: [] for i in range(len(self.boundaries) + 1)}
        for idx, ar in enumerate(self.aspect_ratios):
            bucket = assign_bucket(float(ar), self.boundaries)
            self._bucket_indices[bucket].append(idx)

    @property
    def bucket_indices(self) -> dict[int, list[int]]:
        """Mapping from bucket index to dataset indices belonging to that bucket."""
        return {b: list(indices) for b, indices in self._bucket_indices.items()}

    @property
    def bucket_counts(self) -> dict[int, int]:
        """Count of samples in each aspect ratio bucket."""
        return {b: len(indices) for b, indices in self._bucket_indices.items()}

    def __len__(self) -> int:
        """Accurate batch count mirroring per-bucket chunking and drop_last."""
        total = 0
        for indices in self._bucket_indices.values():
            n = len(indices)
            if n == 0:
                continue
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total

    def __iter__(self) -> Iterator[list[int]]:
        all_batches: list[list[int]] = []
        for bucket in sorted(self._bucket_indices.keys()):
            indices = self._bucket_indices[bucket]
            if not indices:
                continue
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=self.generator).tolist()
                bucket_order = [indices[i] for i in perm]
            else:
                bucket_order = list(indices)

            for i in range(0, len(bucket_order), self.batch_size):
                chunk = bucket_order[i : i + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                all_batches.append(chunk)

        if self.shuffle:
            batch_perm = torch.randperm(len(all_batches), generator=self.generator).tolist()
            all_batches = [all_batches[i] for i in batch_perm]

        yield from all_batches


# Alias for explicit page bucket naming
PageBucketSampler = AspectRatioBatchSampler
