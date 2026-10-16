"""JSONL page dataset with image padding and polygon conversion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from prismocr.data.geometry import polygon_to_box
from prismocr.data.orientation import exif_oriented, exif_oriented_size
from prismocr.data.schema import resolve_line_supervision, validate_annotation


def dataset_content_sha256(dataset: "PrismPageDataset") -> str:
    """Fingerprint canonical annotations and exact image bytes for artifact provenance."""
    digest = hashlib.sha256(b"prismocr-dataset-content-v1\0")
    for record in dataset.records:
        annotation = json.dumps(
            record["_annotation"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        record_digest = hashlib.sha256()
        record_digest.update(b"annotation\0")
        record_digest.update(len(annotation).to_bytes(8, "big"))
        record_digest.update(annotation)
        image_path: Path = record["_image_path"]
        try:
            record_digest.update(b"image\0")
            record_digest.update(image_path.stat().st_size.to_bytes(8, "big"))
            with image_path.open("rb") as image:
                for chunk in iter(lambda: image.read(1024 * 1024), b""):
                    record_digest.update(chunk)
        except OSError as exc:
            raise ValueError(f"cannot fingerprint dataset image {image_path}: {exc}") from exc
        digest.update(record_digest.digest())
    return digest.hexdigest()


def _load_records(manifest: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        record = json.loads(raw)
        if not isinstance(record, dict):
            raise ValueError(f"{manifest}:{line_number}: each manifest row must be an object")
        annotation = record.get("annotation")
        if annotation is None and record.get("annotation_path"):
            annotation_path = (manifest.parent / record["annotation_path"]).resolve()
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        elif annotation is None:
            annotation = {key: value for key, value in record.items() if key not in {"image_path", "annotation_path"}}
        elif isinstance(annotation, str):
            annotation_path = (manifest.parent / annotation).resolve()
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        validate_annotation(annotation)
        image_path = record.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            raise ValueError(f"{manifest}:{line_number}: image_path is required")
        resolved_image_path = (manifest.parent / image_path).resolve()
        try:
            hasher = hashlib.sha256()
            with resolved_image_path.open("rb") as image_file:
                for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                    hasher.update(chunk)
            image_digest = hasher.hexdigest()
        except OSError as exc:
            raise ValueError(f"{manifest}:{line_number}: cannot read image {resolved_image_path}: {exc}") from exc
        expected_digest = annotation["image"]["sha256"].lower()
        if image_digest != expected_digest:
            raise ValueError(
                f"{manifest}:{line_number}: image SHA-256 mismatch for {resolved_image_path}"
            )
        try:
            with Image.open(resolved_image_path) as image:
                actual_width, actual_height = exif_oriented_size(image)
        except OSError as exc:
            raise ValueError(
                f"{manifest}:{line_number}: cannot decode image {resolved_image_path}: {exc}"
            ) from exc
        expected_size = (annotation["image"]["width"], annotation["image"]["height"])
        if (actual_width, actual_height) != expected_size:
            raise ValueError(
                f"{manifest}:{line_number}: image dimensions for {resolved_image_path} are "
                f"{(actual_width, actual_height)}, annotation says {expected_size}"
            )
        record["_annotation"] = annotation
        record["_image_path"] = resolved_image_path
        record["_image_sha256"] = image_digest
        records.append(record)
    return records


class PrismPageDataset(Dataset):
    """Loads one page and canonical v4 line/region/cell targets per item.

    ``max_page_side`` optionally downscales a loaded page so its longer side fits
    the given pixel count. Normalized targets stay in page fractions, so supervision
    is unchanged while large scans stop dominating batch tensor size and compute.
    """

    def __init__(self, manifest: str | Path, *, max_page_side: int | None = None) -> None:
        if max_page_side is not None and (
            isinstance(max_page_side, bool) or not isinstance(max_page_side, int) or max_page_side <= 0
        ):
            raise ValueError("max_page_side must be a positive integer or None")
        self.manifest = Path(manifest).resolve()
        self.max_page_side = max_page_side
        self.records = _load_records(self.manifest)

    def __len__(self) -> int:
        return len(self.records)

    def texts(self) -> list[str]:
        return [
            line["text"]
            for record in self.records
            for line in record["_annotation"].get("lines", ())
            if line.get("text")
        ]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        record = self.records[index]
        annotation = record["_annotation"]
        image_path: Path = record["_image_path"]
        with Image.open(image_path) as image:
            rgb = exif_oriented(image).convert("RGB")
        expected = annotation["image"]
        if (rgb.width, rgb.height) != (expected["width"], expected["height"]):
            raise ValueError(
                f"image dimensions for {image_path} are {(rgb.width, rgb.height)}, "
                f"annotation says {(expected['width'], expected['height'])}"
            )
        tensor = torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1).float().div_(255.0)
        del rgb
        tensor, scale_x, scale_y = _cap_page_side(tensor, self.max_page_side)
        actual_height, actual_width = tensor.shape[-2:]

        def box_for(item: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
            polygon = item["polygon_px"]
            if scale_x != 1.0 or scale_y != 1.0:
                polygon = [(point[0] * scale_x, point[1] * scale_y) for point in polygon]
            return polygon_to_box(polygon, (actual_width, actual_height))

        target = {
            "document_id": annotation["document_id"],
            "page_id": annotation["page_id"],
            "supervision": {"lines": resolve_line_supervision(annotation)},
            "lines": [
                {**line, "box": box_for(line)}
                for line in annotation.get("lines", ())
            ],
            "regions": [
                {**region, "box": box_for(region)}
                for region in annotation.get("regions", ())
            ],
            "cells": [
                {**cell, "kind": "CELL", "box": box_for(cell)}
                for cell in annotation.get("cells", ())
            ],
            "relations": list(annotation.get("relations", ())),
            "reading_order": list(annotation.get("reading_order", ())),
            "_original_size": (actual_width, actual_height),
        }
        return tensor, target


def _cap_page_side(
    tensor: torch.Tensor, max_page_side: int | None
) -> tuple[torch.Tensor, float, float]:
    """Downscale a page tensor so its longer side fits ``max_page_side`` pixels.

    Bilinear interpolation matches ``PrismOCR._prepare_image`` page resolution, and
    the returned per-axis factors match the resized pixels so polygon scaling stays
    exact. Normalized targets are unchanged by a uniform downscale.
    """
    height, width = tensor.shape[-2:]
    if max_page_side is None or max(width, height) <= max_page_side:
        return tensor, 1.0, 1.0
    scale = max_page_side / max(width, height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = F.interpolate(
        tensor[None], size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )[0]
    return resized, resized_width / width, resized_height / height


def collate_pages(batch: list[tuple[torch.Tensor, dict[str, Any]]]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    max_height = max(image.shape[-2] for image, _ in batch)
    max_width = max(image.shape[-1] for image, _ in batch)
    padded_height = (max_height + 15) // 16 * 16
    padded_width = (max_width + 15) // 16 * 16
    images = batch[0][0].new_zeros((len(batch), 3, padded_height, padded_width))
    targets: list[dict[str, Any]] = []
    for batch_index, (image, target) in enumerate(batch):
        height, width = image.shape[-2:]
        images[batch_index, :, :height, :width] = image
        adjusted = {key: value for key, value in target.items() if key != "_original_size"}
        adjusted["supervision"] = {"lines": resolve_line_supervision(target)}
        scale_x, scale_y = width / padded_width, height / padded_height
        # Valid top-left page extent in padded-canvas fractions, not a text mask.
        adjusted["_valid_image_extent"] = (scale_x, scale_y)
        for field in ("lines", "regions", "cells"):
            adjusted[field] = []
            for item in target[field]:
                converted = dict(item)
                box = list(converted["box"])
                box[0] *= scale_x
                box[1] *= scale_y
                box[2] *= scale_x
                box[3] *= scale_y
                converted["box"] = box
                adjusted[field].append(converted)
        targets.append(adjusted)
    return images, targets
