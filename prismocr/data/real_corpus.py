"""Converters from local permissioned corpora into v4 JSONL page manifests.

Two converters cover the corpus shapes available on the training host:

``compose_recognition_pages``
    Pastes text-line crops from a PaddleOCR-style ``labels.txt`` (``path<TAB>text``)
    onto page canvases. Every line polygon is the exact pasted (optionally
    rotated) crop rectangle, so detection geometry and recognition transcripts
    stay pixel-consistent. Paragraph-level ``REGION`` instances group vertically
    adjacent lines inside each column.

``convert_coco_layout``
    Converts a COCO layout annotation file (for example DocLayNet) into typed
    region supervision on the original page images. No text is invented: pages
    carry regions only.

``convert_khmerst_scene``
    Converts the KhmerST page-level VIA annotations (polygon plus transcript per
    text line) into line supervision on the original scene images. These are
    real annotated pages, not composed crops; each line is its own reading scope.

All converters write provenance (license, permission reference, reviewer) and
report ``converted_requires_human_label_qa``; structural conversion is not
annotation review.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import unicodedata
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from prismocr.data.orientation import exif_oriented_size
from prismocr.data.schema import TEXT_ROLES, validate_annotation
from prismocr.geometry import polygon_is_simple, signed_polygon_area

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_DEFAULT_CATEGORY_KINDS = {"Table": "TABLE", "List-item": "LIST"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_recognition_labels(
    labels_file: str | Path,
    *,
    prefix_filter: str | None = None,
    max_characters: int = 160,
) -> tuple[list[tuple[Path, str]], dict[str, int]]:
    """Read ``path<TAB>text`` rows, dropping unusable crops with counted reasons."""
    if isinstance(max_characters, bool) or not isinstance(max_characters, int) or max_characters <= 0:
        raise ValueError("max_characters must be a positive integer")
    source = Path(labels_file).resolve()
    base = source.parent
    rows: list[tuple[Path, str]] = []
    skipped: dict[str, int] = {"missing_file": 0, "empty_text": 0, "multiline_text": 0, "too_long": 0}
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        if "\t" not in raw:
            raise ValueError(f"{source}:{line_number}: expected 'image_path<TAB>text'")
        image_field, text = raw.split("\t", 1)
        image_field = image_field.strip()
        if prefix_filter and not image_field.startswith(prefix_filter):
            continue
        text = unicodedata.normalize("NFC", text).strip()
        if not image_field:
            raise ValueError(f"{source}:{line_number}: image_path is empty")
        if not text:
            skipped["empty_text"] += 1
            continue
        if "\n" in text or "\r" in text:
            skipped["multiline_text"] += 1
            continue
        if len(text) > max_characters:
            skipped["too_long"] += 1
            continue
        image_path = (base / image_field).resolve()
        if not image_path.is_file():
            skipped["missing_file"] += 1
            continue
        rows.append((image_path, text))
    if not rows:
        raise ValueError(f"{source}: no usable recognition rows were found")
    rows.sort(key=lambda item: str(item[0]))
    return rows, skipped


def _rotate_crop(crop: Image.Image, angle_degrees: float) -> tuple[Image.Image, list[list[float]]]:
    """Rotate about the crop center and return the tile plus rotated corner offsets.

    The polygon is the forward transform of the crop rectangle under the same
    affine used to build the tile, so polygon geometry and pasted pixels agree.
    """
    width, height = crop.size
    if width <= 0 or height <= 0:
        raise ValueError("crop has no area")
    if angle_degrees == 0.0:
        return crop, [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]]
    theta = math.radians(angle_degrees)
    cos, sin = math.cos(theta), math.sin(theta)
    corners = [(0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height))]
    center_x, center_y = width / 2.0, height / 2.0

    def forward(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point[0] - center_x, point[1] - center_y
        return cos * x - sin * y, sin * x + cos * y

    moved = [forward(point) for point in corners]
    min_x = min(point[0] for point in moved)
    min_y = min(point[1] for point in moved)
    max_x = max(point[0] for point in moved)
    max_y = max(point[1] for point in moved)
    tile_width = max(1, math.ceil(max_x - min_x))
    tile_height = max(1, math.ceil(max_y - min_y))
    offset_x, offset_y = -min_x, -min_y
    polygon = [[x + offset_x, y + offset_y] for x, y in moved]
    # PIL samples the destination from the source, so the inverse affine is
    # built from the same forward rotation: X = R (p - c_in) + offset.
    data = (
        cos,
        sin,
        center_x - cos * offset_x - sin * offset_y,
        -sin,
        cos,
        center_y + sin * offset_x - cos * offset_y,
    )
    tile = crop.transform((tile_width, tile_height), Image.AFFINE, data, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    return tile, polygon


def compose_recognition_pages(
    labels_file: str | Path,
    output_dir: str | Path,
    *,
    pages: int,
    license_name: str,
    permission_reference: str,
    reviewer: str,
    template_family_id: str,
    source_class: str = "derived",
    source_dataset: str | None = None,
    seed: int = 42,
    image_size: tuple[int, int] = (1024, 1400),
    columns: int = 2,
    lines_per_page: tuple[int, int] = (16, 44),
    blank_page_fraction: float = 0.02,
    rotation_degrees: float = 2.0,
    scale_range: tuple[float, float] = (0.85, 1.15),
    margin: int = 28,
    gutter: int = 24,
    line_padding: int = 2,
    max_characters: int = 160,
    prefix_filter: str | None = None,
    document_prefix: str | None = None,
    page_offset: int = 0,
    image_format: str = "PNG",
    jpeg_quality: int = 90,
) -> dict[str, Any]:
    """Compose page canvases from text-line crops and return a manifest summary."""
    if isinstance(pages, bool) or not isinstance(pages, int) or pages <= 0:
        raise ValueError("pages must be a positive integer")
    if source_class not in {"real", "synthetic", "derived"}:
        raise ValueError("source_class must be real, synthetic, or derived")
    if source_dataset is not None and (not isinstance(source_dataset, str) or not source_dataset.strip()):
        raise ValueError("source_dataset must be a non-empty string when supplied")
    for field, value in (
        ("license_name", license_name),
        ("permission_reference", permission_reference),
        ("reviewer", reviewer),
        ("template_family_id", template_family_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} is required")
    if (
        not isinstance(image_size, tuple)
        or len(image_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in image_size)
    ):
        raise ValueError("image_size must be a pair of positive integers (width, height)")
    if isinstance(columns, bool) or not isinstance(columns, int) or columns <= 0:
        raise ValueError("columns must be a positive integer")
    if (
        not isinstance(lines_per_page, tuple)
        or len(lines_per_page) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in lines_per_page)
        or lines_per_page[0] > lines_per_page[1]
    ):
        raise ValueError("lines_per_page must be a non-decreasing pair of nonnegative integers")
    if not 0.0 <= blank_page_fraction < 1.0:
        raise ValueError("blank_page_fraction must be in [0, 1)")
    if not 0.0 <= rotation_degrees <= 15.0:
        raise ValueError("rotation_degrees must be in [0, 15]")
    if (
        not isinstance(scale_range, tuple)
        or len(scale_range) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0 for value in scale_range)
        or scale_range[0] > scale_range[1]
    ):
        raise ValueError("scale_range must be an increasing positive pair")
    for field, value in (("margin", margin), ("gutter", gutter), ("line_padding", line_padding)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
    if isinstance(page_offset, bool) or not isinstance(page_offset, int) or page_offset < 0:
        raise ValueError("page_offset must be a nonnegative integer")
    image_format = image_format.upper()
    if image_format not in {"PNG", "JPEG"}:
        raise ValueError("image_format must be PNG or JPEG")
    if isinstance(jpeg_quality, bool) or not isinstance(jpeg_quality, int) or not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be an integer in [1, 100]")
    width, height = image_size
    column_width = (width - 2 * margin - (columns - 1) * gutter) / columns
    if column_width < 16:
        raise ValueError("image_size, margin, gutter and columns leave no usable column width")

    labels_resolved = Path(labels_file).resolve()
    labels_parent = labels_resolved.parent
    if source_dataset is not None:
        resolved_source_dataset = source_dataset.strip()
    else:
        parent_name = labels_parent.name
        resolved_source_dataset = f"{parent_name}/{labels_resolved.name}" if parent_name else labels_resolved.name

    pool, skipped = read_recognition_labels(
        labels_file, prefix_filter=prefix_filter, max_characters=max_characters
    )

    output = Path(output_dir).resolve()
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".png" if image_format == "PNG" else ".jpg"
    save_options = {"format": image_format}
    if image_format == "JPEG":
        save_options["quality"] = jpeg_quality
    rows: list[dict[str, Any]] = []
    document_stem = document_prefix or template_family_id
    total_lines = 0
    total_regions = 0
    total_relations = 0
    total_characters = 0
    unreadable = 0
    crop_hashes: dict[Path, str] = {}

    for page_index in range(pages):
        # A per-page stream keeps every page reproducible regardless of process
        # partitioning, so page ranges can be composed in parallel.
        rng = random.Random(f"{seed}:{page_offset + page_index}")
        document_id = f"{document_stem}:{page_offset + page_index:06d}"
        page_id = f"{document_id}:page:0"
        canvas = Image.new("RGB", image_size, (255, 255, 255))
        target_lines = 0 if rng.random() < blank_page_fraction else rng.randint(*lines_per_page)
        quotas = [target_lines // columns + (1 if column < target_lines % columns else 0) for column in range(columns)]
        column_lines: list[list[dict[str, Any]]] = []
        page_line_index = 0
        for column in range(columns):
            remaining = quotas[column]
            scope_id = f"{page_id}:col{column}"
            placed: list[dict[str, Any]] = []
            column_x = margin + column * (column_width + gutter)
            cursor_y = float(margin)
            attempts = 0
            while remaining > 0 and attempts < 8:
                attempts += 1
                crop_path, text = pool[rng.randrange(len(pool))]
                try:
                    with Image.open(crop_path) as handle:
                        crop = handle.convert("RGB")
                except OSError:
                    unreadable += 1
                    remaining -= 1
                    continue
                # Rotation expands the crop bounding box; budget the scaled crop
                # so the rotated tile still fits the column exactly.
                angle = rng.uniform(-rotation_degrees, rotation_degrees)
                radians = math.radians(angle)
                expansion = abs(math.cos(radians)) + abs(math.sin(radians))
                scale = rng.uniform(*scale_range)
                scale = min(scale, column_width / (expansion * crop.width))
                resized = (
                    max(3, int(round(crop.width * scale))),
                    max(3, int(round(crop.height * scale))),
                )
                crop = crop.resize(resized, Image.Resampling.BILINEAR)
                if line_padding:
                    padded = Image.new(
                        "RGB",
                        (crop.width + 2 * line_padding, crop.height + 2 * line_padding),
                        (255, 255, 255),
                    )
                    padded.paste(crop, (line_padding, line_padding))
                    crop = padded
                tile, offsets = _rotate_crop(crop, angle)
                tile_y = cursor_y
                if tile_y + tile.height > height - margin:
                    break
                if tile.width > column_width + 0.5 or tile.width > width:
                    remaining -= 1
                    continue
                tile_x = column_x + rng.uniform(0.0, max(0.0, column_width - tile.width))
                if tile_x + tile.width > width:
                    tile_x = float(width - tile.width)
                polygon = [[round(x + tile_x, 3), round(y + tile_y, 3)] for x, y in offsets]
                if any(
                    point[0] < 0 or point[1] < 0 or point[0] > width or point[1] > height
                    for point in polygon
                ):
                    remaining -= 1
                    continue
                canvas.paste(tile, (int(round(tile_x)), int(round(tile_y))))
                line_id = f"{page_id}:l{page_line_index:03d}"
                page_line_index += 1
                try:
                    resolved_crop = Path(crop_path).resolve()
                    source_crop_id = str(resolved_crop.relative_to(labels_parent))
                except ValueError:
                    source_crop_id = str(crop_path)
                if resolved_crop not in crop_hashes:
                    crop_hashes[resolved_crop] = _sha256(resolved_crop)
                source_crop_hash = crop_hashes[resolved_crop]
                placed.append(
                    {
                        "id": line_id,
                        "polygon_px": polygon,
                        "text": text,
                        "role": "PARA_LINE",
                        "scope_id": scope_id,
                        "source_crop_id": source_crop_id,
                        "source_crop_hash": source_crop_hash,
                    }
                )
                total_lines += 1
                total_characters += len(text)
                cursor_y = tile_y + tile.height + rng.uniform(2.0, 8.0)
                remaining -= 1
                attempts = 0
            column_lines.append(placed)

        line_records = [line for placed in column_lines for line in placed]
        regions: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []
        for column, placed in enumerate(column_lines):
            block: list[dict[str, Any]] = []
            previous_bottom: float | None = None
            block_index = 0
            for line in placed:
                top = min(point[1] for point in line["polygon_px"])
                if previous_bottom is not None and (top - previous_bottom) > 12.0:
                    region = _block_region(page_id, column, block_index, block)
                    regions.append(region)
                    for bline in block:
                        relations.append(
                            {"type": "PARENT", "source": bline["id"], "target": region["id"]}
                        )
                    block_index += 1
                    block = []
                block.append(line)
                previous_bottom = max(point[1] for point in line["polygon_px"])
            if block:
                region = _block_region(page_id, column, block_index, block)
                regions.append(region)
                for bline in block:
                    relations.append(
                        {"type": "PARENT", "source": bline["id"], "target": region["id"]}
                    )
        total_regions += len(regions)
        total_relations += len(relations)

        image_name = f"page-{page_offset + page_index:06d}{suffix}"
        image_path = image_dir / image_name
        canvas.save(image_path, **save_options)
        annotation = {
            "schema_version": "4.0",
            "document_id": document_id,
            "page_id": page_id,
            "image": {"width": width, "height": height, "sha256": _sha256(image_path)},
            "provenance": {
                "source_class": source_class,
                "source_dataset": resolved_source_dataset,
                "license": license_name,
                "permission_reference": permission_reference,
                "reviewer": reviewer,
                "template_family_id": template_family_id,
            },
            "supervision": {"lines": "complete"},
            "lines": line_records,
            "regions": regions,
            "cells": [],
            "relations": relations,
            "reading_order": [line["id"] for line in line_records],
            "annotation": {
                "source": "prismocr.data.real_corpus.compose_recognition_pages",
                "composition_seed": seed,
            },
        }
        validate_annotation(annotation)
        rows.append({"image_path": str(image_path), "annotation": annotation})

    manifest = output / "manifest.jsonl"
    _write_jsonl(manifest, rows)
    return {
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "pages": len(rows),
        "lines": total_lines,
        "regions": total_regions,
        "relations": total_relations,
        "characters": total_characters,
        "recognized_pool_rows": len(pool),
        "skipped_rows": skipped,
        "unreadable_crops": unreadable,
        "seed": seed,
        "page_offset": page_offset,
        "image_format": image_format,
        "status": "converted_requires_human_label_qa",
    }


def compose_table_pages(
    labels_file: str | Path,
    output_dir: str | Path,
    *,
    pages: int,
    license_name: str,
    permission_reference: str,
    reviewer: str,
    template_family_id: str,
    source_class: str = "derived",
    source_dataset: str | None = None,
    seed: int = 42,
    image_size: tuple[int, int] = (1024, 1400),
    rows: tuple[int, int] | int = (2, 5),
    cols: tuple[int, int] | int = (2, 4),
    columns: tuple[int, int] | int | None = None,
    header_probability: float = 0.5,
    margin: int = 28,
    cell_padding: int = 4,
    rotation_degrees: float = 0.0,
    scale_range: tuple[float, float] = (0.85, 1.15),
    prefix_filter: str | None = None,
    document_prefix: str | None = None,
    page_offset: int = 0,
    image_format: str = "PNG",
    jpeg_quality: int = 90,
    max_characters: int = 160,
    draw_grid: bool = True,
) -> dict[str, Any]:
    """Compose page canvases containing structured table grids with typed relations."""
    if isinstance(pages, bool) or not isinstance(pages, int) or pages <= 0:
        raise ValueError("pages must be a positive integer")
    if source_class not in {"real", "synthetic", "derived"}:
        raise ValueError("source_class must be real, synthetic, or derived")
    if source_dataset is not None and (not isinstance(source_dataset, str) or not source_dataset.strip()):
        raise ValueError("source_dataset must be a non-empty string when supplied")
    for field, value in (
        ("license_name", license_name),
        ("permission_reference", permission_reference),
        ("reviewer", reviewer),
        ("template_family_id", template_family_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} is required")
    if (
        not isinstance(image_size, tuple)
        or len(image_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in image_size)
    ):
        raise ValueError("image_size must be a pair of positive integers (width, height)")

    if columns is not None:
        cols = columns

    if isinstance(rows, int) and not isinstance(rows, bool):
        rows_range = (rows, rows)
    elif isinstance(rows, tuple) and len(rows) == 2 and all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in rows) and rows[0] <= rows[1]:
        rows_range = rows
    else:
        raise ValueError("rows must be a positive integer or non-decreasing pair of positive integers")

    if isinstance(cols, int) and not isinstance(cols, bool):
        cols_range = (cols, cols)
    elif isinstance(cols, tuple) and len(cols) == 2 and all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in cols) and cols[0] <= cols[1]:
        cols_range = cols
    else:
        raise ValueError("cols must be a positive integer or non-decreasing pair of positive integers")

    if isinstance(header_probability, bool) or not isinstance(header_probability, (int, float)) or not 0.0 <= header_probability <= 1.0:
        raise ValueError("header_probability must be a float in [0.0, 1.0]")
    if not 0.0 <= rotation_degrees <= 15.0:
        raise ValueError("rotation_degrees must be in [0, 15]")
    if (
        not isinstance(scale_range, tuple)
        or len(scale_range) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0 for value in scale_range)
        or scale_range[0] > scale_range[1]
    ):
        raise ValueError("scale_range must be an increasing positive pair")
    for field, value in (("margin", margin), ("cell_padding", cell_padding), ("page_offset", page_offset)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
    image_format = image_format.upper()
    if image_format not in {"PNG", "JPEG"}:
        raise ValueError("image_format must be PNG or JPEG")
    if isinstance(jpeg_quality, bool) or not isinstance(jpeg_quality, int) or not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be an integer in [1, 100]")

    width, height = image_size
    labels_resolved = Path(labels_file).resolve()
    labels_parent = labels_resolved.parent
    if source_dataset is not None:
        resolved_source_dataset = source_dataset.strip()
    else:
        parent_name = labels_parent.name
        resolved_source_dataset = f"{parent_name}/{labels_resolved.name}" if parent_name else labels_resolved.name

    pool, skipped = read_recognition_labels(
        labels_file, prefix_filter=prefix_filter, max_characters=max_characters
    )
    if not pool:
        raise ValueError(f"no valid recognition crops found in {labels_file}")

    output = Path(output_dir).resolve()
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".png" if image_format == "PNG" else ".jpg"
    save_options = {"format": image_format}
    if image_format == "JPEG":
        save_options["quality"] = jpeg_quality
    rows_records: list[dict[str, Any]] = []
    document_stem = document_prefix or template_family_id
    total_lines = 0
    total_regions = 0
    total_cells = 0
    total_relations = 0
    total_characters = 0
    unreadable = 0
    crop_hashes: dict[Path, str] = {}

    for page_index in range(pages):
        rng = random.Random(f"{seed}:{page_offset + page_index}")
        document_id = f"{document_stem}:{page_offset + page_index:06d}"
        page_id = f"{document_id}:page:0"
        canvas = Image.new("RGB", image_size, (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        num_rows = rng.randint(*rows_range)
        num_cols = rng.randint(*cols_range)
        has_header = rng.random() < header_probability

        tbl_x0 = float(margin)
        tbl_y0 = float(margin)
        tbl_x1 = float(width - margin)
        tbl_y1 = float(height - margin)

        if tbl_x1 <= tbl_x0 + num_cols * 8 or tbl_y1 <= tbl_y0 + num_rows * 8:
            raise ValueError("image_size and margin leave insufficient space for table grid")

        cw = (tbl_x1 - tbl_x0) / num_cols
        rh = (tbl_y1 - tbl_y0) / num_rows

        table_id = f"{page_id}:table:0"
        table_polygon = [
            [round(tbl_x0, 2), round(tbl_y0, 2)],
            [round(tbl_x1, 2), round(tbl_y0, 2)],
            [round(tbl_x1, 2), round(tbl_y1, 2)],
            [round(tbl_x0, 2), round(tbl_y1, 2)],
        ]

        if draw_grid:
            draw.rectangle(
                [int(round(tbl_x0)), int(round(tbl_y0)), int(round(tbl_x1)), int(round(tbl_y1))],
                outline=(0, 0, 0),
                width=1,
            )
            for c in range(1, num_cols):
                line_x = int(round(tbl_x0 + c * cw))
                draw.line([(line_x, int(round(tbl_y0))), (line_x, int(round(tbl_y1)))], fill=(180, 180, 180), width=1)
            for r in range(1, num_rows):
                line_y = int(round(tbl_y0 + r * rh))
                draw.line([(int(round(tbl_x0)), line_y), (int(round(tbl_x1)), line_y)], fill=(180, 180, 180), width=1)
            if has_header and num_rows > 1:
                hdr_y = int(round(tbl_y0 + rh))
                draw.line([(int(round(tbl_x0)), hdr_y), (int(round(tbl_x1)), hdr_y)], fill=(0, 0, 0), width=2)

        page_lines: list[dict[str, Any]] = []
        page_cells: list[dict[str, Any]] = []
        page_relations: list[dict[str, Any]] = []
        page_line_index = 0

        for r in range(num_rows):
            for c in range(num_cols):
                cx0 = tbl_x0 + c * cw
                cx1 = tbl_x0 + (c + 1) * cw
                cy0 = tbl_y0 + r * rh
                cy1 = tbl_y0 + (r + 1) * rh
                cell_id = f"{page_id}:c{r}_{c}"
                is_header_cell = bool(has_header and r == 0)

                cell_poly = [
                    [round(cx0, 2), round(cy0, 2)],
                    [round(cx1, 2), round(cy0, 2)],
                    [round(cx1, 2), round(cy1, 2)],
                    [round(cx0, 2), round(cy1, 2)],
                ]
                cell_record = {
                    "id": cell_id,
                    "table_id": table_id,
                    "row": r,
                    "col": c,
                    "rowspan": 1,
                    "colspan": 1,
                    "header": is_header_cell,
                    "is_header": is_header_cell,
                    "polygon_px": cell_poly,
                }
                page_cells.append(cell_record)

                # PARENT: cell -> table
                page_relations.append({"type": "PARENT", "source": cell_id, "target": table_id})

                # RIGHT_CELL: cell (r, c) -> cell (r, c+1)
                if c < num_cols - 1:
                    next_cell_id = f"{page_id}:c{r}_{c + 1}"
                    page_relations.append({"type": "RIGHT_CELL", "source": cell_id, "target": next_cell_id})

                pad = max(1, cell_padding)
                avail_w = (cx1 - cx0) - 2 * pad
                avail_h = (cy1 - cy0) - 2 * pad

                crop_placed = False
                attempts = 0
                while not crop_placed and attempts < 8:
                    attempts += 1
                    crop_path, text = pool[rng.randrange(len(pool))]
                    try:
                        with Image.open(crop_path) as handle:
                            crop = handle.convert("RGB")
                    except OSError:
                        unreadable += 1
                        continue

                    target_scale = rng.uniform(*scale_range)
                    scale = min(target_scale, avail_w / max(1, crop.width), avail_h / max(1, crop.height))
                    resized_w = max(3, int(round(crop.width * scale)))
                    resized_h = max(3, int(round(crop.height * scale)))
                    if resized_w > avail_w:
                        resized_w = max(1, int(avail_w))
                    if resized_h > avail_h:
                        resized_h = max(1, int(avail_h))

                    crop = crop.resize((resized_w, resized_h), Image.Resampling.BILINEAR)

                    offset_x = rng.uniform(0.0, max(0.0, avail_w - resized_w))
                    offset_y = rng.uniform(0.0, max(0.0, avail_h - resized_h))
                    paste_x = cx0 + pad + offset_x
                    paste_y = cy0 + pad + offset_y

                    canvas.paste(crop, (int(round(paste_x)), int(round(paste_y))))
                    line_poly = [
                        [round(paste_x, 2), round(paste_y, 2)],
                        [round(paste_x + resized_w, 2), round(paste_y, 2)],
                        [round(paste_x + resized_w, 2), round(paste_y + resized_h, 2)],
                        [round(paste_x, 2), round(paste_y + resized_h, 2)],
                    ]

                    line_id = f"{page_id}:l{page_line_index:03d}"
                    page_line_index += 1

                    try:
                        resolved_crop = Path(crop_path).resolve()
                        source_crop_id = str(resolved_crop.relative_to(labels_parent))
                    except ValueError:
                        source_crop_id = str(crop_path)
                    if resolved_crop not in crop_hashes:
                        crop_hashes[resolved_crop] = _sha256(resolved_crop)
                    source_crop_hash = crop_hashes[resolved_crop]

                    line_record = {
                        "id": line_id,
                        "polygon_px": line_poly,
                        "text": text,
                        "role": "PARA_LINE",
                        "scope_id": table_id,
                        "source_crop_id": source_crop_id,
                        "source_crop_hash": source_crop_hash,
                    }
                    page_lines.append(line_record)
                    total_lines += 1
                    total_characters += len(text)

                    # PARENT: line -> cell
                    page_relations.append({"type": "PARENT", "source": line_id, "target": cell_id})
                    crop_placed = True

        total_cells += len(page_cells)
        total_regions += 1
        total_relations += len(page_relations)

        table_region = {
            "id": table_id,
            "kind": "TABLE",
            "polygon_px": table_polygon,
            "annotation_ambiguous": False,
        }

        image_name = f"page-{page_offset + page_index:06d}{suffix}"
        image_path = image_dir / image_name
        canvas.save(image_path, **save_options)
        annotation = {
            "schema_version": "4.0",
            "document_id": document_id,
            "page_id": page_id,
            "image": {"width": width, "height": height, "sha256": _sha256(image_path)},
            "provenance": {
                "source_class": source_class,
                "source_dataset": resolved_source_dataset,
                "license": license_name,
                "permission_reference": permission_reference,
                "reviewer": reviewer,
                "template_family_id": template_family_id,
            },
            "supervision": {"lines": "complete"},
            "lines": page_lines,
            "regions": [table_region],
            "cells": page_cells,
            "relations": page_relations,
            "reading_order": [line["id"] for line in page_lines],
            "annotation": {
                "source": "prismocr.data.real_corpus.compose_table_pages",
                "composition_seed": seed,
            },
        }
        validate_annotation(annotation)
        rows_records.append({"image_path": str(image_path), "annotation": annotation})

    manifest = output / "manifest.jsonl"
    _write_jsonl(manifest, rows_records)
    return {
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "pages": len(rows_records),
        "lines": total_lines,
        "regions": total_regions,
        "cells": total_cells,
        "relations": total_relations,
        "characters": total_characters,
        "recognized_pool_rows": len(pool),
        "skipped_rows": skipped,
        "unreadable_crops": unreadable,
        "seed": seed,
        "page_offset": page_offset,
        "image_format": image_format,
        "status": "converted_requires_human_label_qa",
    }


def _block_region(page_id: str, column: int, index: int, lines: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    points = [point for line in lines for point in line["polygon_px"]]
    left = min(point[0] for point in points)
    top = min(point[1] for point in points)
    right = max(point[0] for point in points)
    bottom = max(point[1] for point in points)
    return {
        "id": f"{page_id}:r{column}-{index}",
        "kind": "REGION",
        "polygon_px": [[left, top], [right, top], [right, bottom], [left, bottom]],
        "annotation_ambiguous": False,
    }


def _segmentation_ring(segmentation: Any) -> list[list[float]] | None:
    if isinstance(segmentation, Mapping):
        return None
    if not isinstance(segmentation, Sequence) or isinstance(segmentation, (str, bytes)) or not segmentation:
        return None
    first = segmentation[0]
    if not isinstance(first, Sequence) or isinstance(first, (str, bytes)):
        return None
    if first and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in first):
        if len(first) < 6 or len(first) % 2:
            return None
        points = [[float(first[i]), float(first[i + 1])] for i in range(0, len(first), 2)]
    else:
        points = []
        for point in first:
            if (
                not isinstance(point, Sequence)
                or isinstance(point, (str, bytes))
                or len(point) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in point)
            ):
                return None
            points.append([float(point[0]), float(point[1])])
        if len(points) < 3:
            return None
    return points


def convert_coco_layout(
    coco_json: str | Path,
    image_dir: str | Path,
    output_manifest: str | Path,
    *,
    license_name: str,
    permission_reference: str,
    reviewer: str,
    document_prefix: str,
    category_kinds: Mapping[str, str] | None = None,
    limit: int | None = None,
    min_area: float = 1.0,
) -> dict[str, Any]:
    """Convert COCO region polygons into v4 region supervision on page images."""
    for field, value in (
        ("license_name", license_name),
        ("permission_reference", permission_reference),
        ("reviewer", reviewer),
        ("document_prefix", document_prefix),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} is required")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("limit must be a positive integer when supplied")
    mapping = {**_DEFAULT_CATEGORY_KINDS, **(dict(category_kinds) if category_kinds else {})}
    for name, kind in mapping.items():
        if kind not in {"REGION", "TABLE", "LIST", "FORM_FIELD"}:
            raise ValueError(f"category {name!r} maps to unsupported region kind {kind!r}")

    source = Path(coco_json).resolve()
    image_root = Path(image_dir).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    for key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"{source}: COCO payload must contain a {key} array")
    category_names = {
        category["id"]: category.get("name", str(category["id"]))
        for category in payload["categories"]
        if isinstance(category, Mapping) and "id" in category
    }
    by_image: dict[Any, list[Mapping[str, Any]]] = {}
    for annotation in payload["annotations"]:
        if isinstance(annotation, Mapping):
            by_image.setdefault(annotation.get("image_id"), []).append(annotation)

    rows: list[dict[str, Any]] = []
    region_counts: dict[str, int] = {}
    skipped = {"unreadable_image": 0, "unsupported_segmentation": 0, "degenerate_polygon": 0, "no_regions": 0}
    for image in sorted(payload["images"], key=lambda item: item.get("id", 0)):
        if limit is not None and len(rows) >= limit:
            break
        if not isinstance(image, Mapping):
            continue
        file_name = image.get("file_name")
        width, height = image.get("width"), image.get("height")
        if not isinstance(file_name, str) or not isinstance(width, int) or not isinstance(height, int):
            continue
        image_path = (image_root / file_name).resolve()
        if not image_path.is_file():
            skipped["unreadable_image"] += 1
            continue
        with Image.open(image_path) as opened:
            actual_width, actual_height = exif_oriented_size(opened)
        if (actual_width, actual_height) != (width, height):
            raise ValueError(
                f"{image_path}: image dimensions {(actual_width, actual_height)} do not match "
                f"COCO annotation {(width, height)}"
            )
        document = str(image.get("doc_name") or Path(file_name).stem)
        page_number = image.get("page_no")
        document_id = f"{document_prefix}:{document}:{page_number if page_number is not None else Path(file_name).stem}"
        regions: list[dict[str, Any]] = []
        for annotation in by_image.get(image.get("id"), ()):
            name = category_names.get(annotation.get("category_id"))
            kind = mapping.get(name, "REGION")
            ring = _segmentation_ring(annotation.get("segmentation"))
            if ring is None:
                skipped["unsupported_segmentation"] += 1
                continue
            points = [
                [min(max(point[0], 0.0), float(width)), min(max(point[1], 0.0), float(height))]
                for point in ring
            ]
            if len(points) < 3 or abs(signed_polygon_area(points)) < min_area:
                skipped["degenerate_polygon"] += 1
                continue
            if signed_polygon_area(points) < 0:
                points.reverse()
            regions.append({
                "id": f"{document_id}:region:{annotation.get('id')}",
                "kind": kind,
                "polygon_px": [[round(point[0], 3), round(point[1], 3)] for point in points],
                "annotation_ambiguous": False,
            })
            region_counts[kind] = region_counts.get(kind, 0) + 1
        if not regions:
            skipped["no_regions"] += 1
            continue
        annotation = {
            "schema_version": "4.0",
            "document_id": document_id,
            "page_id": f"{document_id}:page:0",
            "image": {"width": width, "height": height, "sha256": _sha256(image_path)},
            "provenance": {
                "source_class": "real",
                "source_dataset": source.name,
                "license": license_name,
                "permission_reference": permission_reference,
                "reviewer": reviewer,
                "template_family_id": document,
            },
            "supervision": {"lines": "unavailable"},
            "lines": [],
            "regions": regions,
            "cells": [],
            "relations": [],
            "reading_order": [],
            "annotation": {
                "source": "prismocr.data.real_corpus.convert_coco_layout",
                "category_kinds": mapping,
            },
        }
        validate_annotation(annotation)
        rows.append({"image_path": str(image_path), "annotation": annotation})

    if not rows:
        raise ValueError(f"{source}: no pages converted; skipped={skipped}")
    output = Path(output_manifest).resolve()
    _write_jsonl(output, rows)
    return {
        "manifest": str(output),
        "manifest_sha256": _sha256(output),
        "pages": len(rows),
        "regions": sum(region_counts.values()),
        "region_kinds": dict(sorted(region_counts.items())),
        "skipped": skipped,
        "status": "converted_requires_human_label_qa",
    }


def _khmerst_payload_entries(
    payload: Any, skipped: dict[str, int]
) -> list[tuple[str, list[tuple[str, list[list[float]]]], bool]]:
    """Extract image lines and a per-image known-omission flag from VIA JSON."""
    if not isinstance(payload, Mapping):
        raise ValueError("KhmerST annotation payload must be an object")
    entries: list[tuple[str, list[tuple[str, list[list[float]]]], bool]] = []
    for value in payload.values():
        if not isinstance(value, Mapping):
            continue
        filename = value.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            continue
        skipped_before = sum(skipped.values())
        regions: list[tuple[str, list[list[float]]]] = []
        raw_regions = value.get("regions", [])
        if isinstance(raw_regions, Mapping):
            raw_regions = list(raw_regions.values())
        if not isinstance(raw_regions, Sequence) or isinstance(raw_regions, (str, bytes)):
            raise ValueError(f"{filename}: VIA regions must be an array or object")
        if isinstance(raw_regions, Sequence):
            for region in raw_regions:
                if not isinstance(region, Mapping):
                    skipped["malformed_polygon"] += 1
                    continue
                shape = region.get("shape_attributes")
                if not isinstance(shape, Mapping) or shape.get("name") != "polygon":
                    skipped["unsupported_shape"] += 1
                    continue
                xs, ys = shape.get("all_points_x"), shape.get("all_points_y")
                if not isinstance(xs, Sequence) or not isinstance(ys, Sequence):
                    skipped["malformed_polygon"] += 1
                    continue
                if isinstance(xs, (str, bytes)) or isinstance(ys, (str, bytes)) or len(xs) != len(ys):
                    skipped["malformed_polygon"] += 1
                    continue
                if len(xs) < 3:
                    skipped["malformed_polygon"] += 1
                    continue
                points: list[list[float]] = []
                malformed = False
                for x, y in zip(xs, ys):
                    if (
                        isinstance(x, bool)
                        or isinstance(y, bool)
                        or not isinstance(x, (int, float))
                        or not isinstance(y, (int, float))
                    ):
                        malformed = True
                        break
                    points.append([float(x), float(y)])
                if malformed:
                    skipped["malformed_polygon"] += 1
                    continue
                attributes = region.get("region_attributes")
                text = ""
                if isinstance(attributes, Mapping):
                    raw_text = attributes.get("Name", attributes.get("text", ""))
                    if isinstance(raw_text, str):
                        text = raw_text
                regions.append((unicodedata.normalize("NFC", text).strip(), points))
        entries.append((filename, regions, sum(skipped.values()) != skipped_before))
    return entries


def read_khmerst_annotations(
    annotations: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Read per-image VIA annotations from a zip or directory of JSON files.

    Returns image records (``filename`` plus non-empty ``lines`` of text and
    polygon points) together with counts of every skipped region, so callers
    can see exactly how much annotation content was dropped and why.
    """
    source = Path(annotations).resolve()
    skipped = {"unsupported_shape": 0, "malformed_polygon": 0, "empty_text": 0, "empty_image_entry": 0}
    payloads: list[tuple[str, Any]] = []
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            for name in sorted(entry for entry in archive.namelist() if entry.lower().endswith(".json")):
                payloads.append((name, json.loads(archive.read(name).decode("utf-8"))))
    elif source.is_dir():
        for path in sorted(source.glob("*.json")):
            payloads.append((path.name, json.loads(path.read_text(encoding="utf-8"))))
    else:
        raise ValueError(f"{source}: expected a VIA annotation zip or directory of JSON files")
    if not payloads:
        raise ValueError(f"{source}: no KhmerST annotations found")

    merged: dict[str, dict[str, Any]] = {}
    ordered: list[str] = []
    for label, payload in payloads:
        for filename, regions, partial in _khmerst_payload_entries(payload, skipped):
            record = merged.get(filename)
            if record is None:
                record = {"filename": filename, "lines": [], "sources": [], "partial": False}
                merged[filename] = record
                ordered.append(filename)
            record["sources"].append(label)
            record["partial"] |= partial
            for text, points in regions:
                if len(points) < 3 or abs(signed_polygon_area(points)) < 1e-9:
                    skipped["malformed_polygon"] += 1
                    record["partial"] = True
                    continue
                if not polygon_is_simple(points):
                    skipped["unsupported_shape"] += 1
                    record["partial"] = True
                    continue
                if not text:
                    skipped["empty_text"] += 1
                    record["partial"] = True
                    continue
                record["lines"].append((text, points))
    records = [merged[filename] for filename in ordered]
    empty = sum(1 for record in records if not record["lines"])
    skipped["empty_image_entry"] = empty
    return [record for record in records if record["lines"]], skipped


def assign_scene_splits(
    filenames: Sequence[str],
    *,
    seed: int = 0,
    ratios: Mapping[str, float] | None = None,
) -> dict[str, str]:
    """Assign each source image to exactly one split by stable hash of its name.

    Scene images have no document grouping above the image itself, so hashing
    the image name (not a crop or region) keeps every annotation of an image in
    a single split and makes the assignment reproducible across processes.
    """
    plan = dict(ratios) if ratios else {"train": 0.8, "val": 0.1, "test": 0.1}
    if not plan:
        raise ValueError("ratios must assign at least one split")
    for name, fraction in plan.items():
        if not isinstance(name, str) or not name or isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or fraction <= 0:
            raise ValueError("ratios must map non-empty split names to positive fractions")
    if abs(sum(plan.values()) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")
    ordered_splits = list(plan.items())
    assignment: dict[str, str] = {}
    for filename in filenames:
        digest = hashlib.sha256(f"{seed}:{filename}".encode("utf-8")).hexdigest()
        position = int(digest[:12], 16) / float(16**12)
        cumulative = 0.0
        chosen = ordered_splits[-1][0]
        for name, fraction in ordered_splits:
            cumulative += fraction
            if position < cumulative:
                chosen = name
                break
        assignment[filename] = chosen
    return assignment


def _duplicate_scene_filenames(
    records: Sequence[Mapping[str, Any]], image_root: Path
) -> dict[str, str]:
    """Map each duplicate scan to the first filename holding the same image bytes.

    The KhmerST release ships a few byte-identical pages under different names. Keeping
    both copies would place identical image bytes in two splits, so the corpus keeps only
    the first filename per content and reports the rest as skipped.
    """
    by_size: dict[int, list[str]] = defaultdict(list)
    for record in records:
        path = image_root / str(record["filename"])
        if path.is_file():
            by_size[path.stat().st_size].append(str(record["filename"]))
    first_by_digest: dict[str, str] = {}
    duplicate_of: dict[str, str] = {}
    for filenames in by_size.values():
        if len(filenames) < 2:
            continue
        for filename in sorted(filenames):
            keeper = first_by_digest.setdefault(_sha256(image_root / filename), filename)
            if keeper != filename:
                duplicate_of[filename] = keeper
    return duplicate_of


def convert_khmerst_scene(
    annotations: str | Path,
    image_dir: str | Path,
    output_manifest: str | Path,
    *,
    license_name: str,
    permission_reference: str,
    reviewer: str,
    document_prefix: str = "khmerst",
    split: str | None = None,
    split_assignment: Mapping[str, str] | None = None,
    line_role: str = "UNKNOWN",
    min_area: float = 4.0,
) -> dict[str, Any]:
    """Convert KhmerST page-level scene annotations into line supervision.

    Every polygon is a real annotated text line on the original scene image, so
    these pages contribute detection geometry and recognition transcripts that
    were not composed or augmented by this repository. Scene text lines are
    independent, so each line gets its own reading scope: the global reading
    order is top-to-bottom then left-to-right, but no NEXT relation is implied
    between unrelated signs.
    """
    for field, value in (
        ("license_name", license_name),
        ("permission_reference", permission_reference),
        ("reviewer", reviewer),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} is required")
    if not isinstance(document_prefix, str) or not document_prefix.strip():
        raise ValueError("document_prefix is required")
    if line_role not in TEXT_ROLES:
        raise ValueError(f"line_role {line_role!r} is not a supported text role")
    if isinstance(min_area, bool) or not isinstance(min_area, (int, float)) or min_area < 0:
        raise ValueError("min_area must be a nonnegative number")

    records, skipped = read_khmerst_annotations(annotations)
    if split is not None and not isinstance(split, str):
        raise ValueError("split must be a string when supplied")
    if split_assignment is not None:
        if not all(isinstance(name, str) and name for name in split_assignment.values()):
            raise ValueError("split_assignment values must be non-empty split names")
    image_root = Path(image_dir).resolve()
    duplicate_of = _duplicate_scene_filenames(records, image_root)
    rows: list[dict[str, Any]] = []
    line_role_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for record in records:
        filename = record["filename"]
        if split_assignment is not None and filename not in split_assignment:
            raise ValueError(f"{filename}: no split assignment")
        assigned = split_assignment[filename] if split_assignment is not None else (split or "unspecified")
        if split is not None and assigned != split:
            continue
        if filename in duplicate_of:
            skipped["duplicate_image_bytes"] = skipped.get("duplicate_image_bytes", 0) + 1
            continue
        image_path = (image_root / filename).resolve()
        if not image_path.is_file():
            skipped["unreadable_image"] = skipped.get("unreadable_image", 0) + 1
            continue
        with Image.open(image_path) as opened:
            width, height = exif_oriented_size(opened)
        stem = Path(filename).stem
        document_id = f"{document_prefix}:{stem}"
        page_id = f"{document_id}:page:0"
        partial = record["partial"]
        candidates: list[tuple[float, float, dict[str, Any]]] = []
        for index, (text, points) in enumerate(record["lines"]):
            polygon = [
                [round(min(max(point[0], 0.0), float(width)), 3), round(min(max(point[1], 0.0), float(height)), 3)]
                for point in points
            ]
            if not polygon_is_simple(polygon) or abs(signed_polygon_area(polygon)) < min_area:
                skipped["degenerate_polygon"] = skipped.get("degenerate_polygon", 0) + 1
                partial = True
                continue
            if signed_polygon_area(polygon) < 0:
                polygon.reverse()
            line_id = f"{page_id}:l{index:03d}"
            centroid_x = sum(point[0] for point in polygon) / len(polygon)
            centroid_y = sum(point[1] for point in polygon) / len(polygon)
            candidates.append((centroid_y, centroid_x, {
                "id": line_id,
                "polygon_px": polygon,
                "text": text,
                "role": line_role,
                "scope_id": f"{page_id}:scene{index:03d}",
            }))
        if not candidates:
            skipped["no_lines"] = skipped.get("no_lines", 0) + 1
            continue
        candidates.sort(key=lambda item: (item[0], item[1]))
        lines = [item[2] for item in candidates]
        annotation = {
            "schema_version": "4.0",
            "document_id": document_id,
            "page_id": page_id,
            "image": {"width": width, "height": height, "sha256": _sha256(image_path)},
            "provenance": {
                "source_class": "real",
                "source_dataset": Path(annotations).name,
                "license": license_name,
                "permission_reference": permission_reference,
                "reviewer": reviewer,
                "template_family_id": stem,
            },
            "supervision": {"lines": "partial" if partial else "complete"},
            "lines": lines,
            "regions": [],
            "cells": [],
            "relations": [],
            "reading_order": [line["id"] for line in lines],
            "annotation": {
                "source": "prismocr.data.real_corpus.convert_khmerst_scene",
                "line_role": line_role,
                "split": assigned,
            },
        }
        validate_annotation(annotation)
        rows.append({"image_path": str(image_path), "annotation": annotation})
        line_role_counts[line_role] = line_role_counts.get(line_role, 0) + len(lines)
        split_counts[assigned] = split_counts.get(assigned, 0) + 1

    if not rows:
        raise ValueError(f"{annotations}: no pages converted; skipped={skipped}")
    output = Path(output_manifest).resolve()
    _write_jsonl(output, rows)
    return {
        "manifest": str(output),
        "manifest_sha256": _sha256(output),
        "pages": len(rows),
        "lines": sum(line_role_counts.values()),
        "line_roles": dict(sorted(line_role_counts.items())),
        "split_pages": dict(sorted(split_counts.items())),
        "skipped": skipped,
        "status": "converted_requires_human_label_qa",
    }


def concat_manifests(manifests: Sequence[str | Path], output_manifest: str | Path) -> dict[str, Any]:
    """Concatenate JSONL manifests row-wise for a combined training split."""
    if not manifests:
        raise ValueError("at least one manifest is required")
    output = Path(output_manifest).resolve()
    rows = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for manifest in manifests:
            for raw in Path(manifest).read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                handle.write(raw + "\n")
                rows += 1
    if rows == 0:
        raise ValueError("no manifest rows were concatenated")
    return {"manifest": str(output), "manifest_sha256": _sha256(output), "pages": rows}
