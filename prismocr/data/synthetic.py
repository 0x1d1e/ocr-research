"""Deterministic synthetic page fixtures, not realistic OCR training data."""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def _load_font(
    font: str | Path | ImageFont.ImageFont | ImageFont.FreeTypeFont,
    font_size: int,
) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    if isinstance(font, (str, Path)):
        if isinstance(font_size, bool) or not isinstance(font_size, int) or font_size <= 0:
            raise ValueError("font_size must be a positive integer when font is a path")
        try:
            return ImageFont.truetype(str(font), size=font_size)
        except OSError as exc:
            raise ValueError(f"cannot load caller-provided font {font!s}: {exc}") from exc
    if isinstance(font, (ImageFont.ImageFont, ImageFont.FreeTypeFont)):
        return font
    raise ValueError("font must be a caller-provided font object or font path")


def _validate_pages(pages: Sequence[Sequence[str]]) -> None:
    if isinstance(pages, (str, bytes)) or not isinstance(pages, Sequence) or not pages:
        raise ValueError("pages must be a non-empty sequence of non-empty text-line sequences")
    for page_index, lines in enumerate(pages):
        if isinstance(lines, (str, bytes)) or not isinstance(lines, Sequence) or not lines:
            raise ValueError(f"page {page_index} must contain at least one text line")
        for line_index, text in enumerate(lines):
            prefix = f"page {page_index} line {line_index}"
            if not isinstance(text, str):
                raise ValueError(f"{prefix} must be a string")
            if not text.strip():
                raise ValueError(f"{prefix} must not be empty or whitespace-only")
            if any(unicodedata.category(char) in {"Cc", "Zl", "Zp"} for char in text):
                raise ValueError(f"{prefix} must contain one line without control or line-separator characters")
            try:
                text.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError(f"{prefix} is not valid Unicode text encodable as UTF-8") from exc


def generate_synthetic_dataset(
    output_dir: str | Path,
    pages: Sequence[Sequence[str]],
    *,
    font: str | Path | ImageFont.ImageFont | ImageFont.FreeTypeFont,
    seed: int = 0,
    image_size: tuple[int, int] = (640, 480),
    font_size: int = 20,
) -> Path:
    """Render text-only pages and return the generated JSONL manifest path.

    Each input page is a sequence of lines in reading order. ``font`` must be
    supplied by the caller as a Pillow font object or font path; ``font_size``
    applies only to a path. No font or text corpus is bundled. Images are plain
    synthetic black-on-white pages with seeded line spacing. Annotations use
    the rendered text bounds as axis-aligned polygons and stable IDs derived
    from ``seed`` and page/line indices. This utility does not simulate realistic
    Khmer rendering or imply OCR quality.
    """
    _validate_pages(pages)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if (
        not isinstance(image_size, tuple)
        or len(image_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in image_size)
    ):
        raise ValueError("image_size must be a pair of positive integers (width, height)")
    width, height = image_size
    loaded_font = _load_font(font, font_size)
    rng = random.Random(seed)
    document_id = f"synthetic-seed-{seed}"
    rendered: list[tuple[str, bytes, dict[str, Any]]] = []

    for page_index, texts in enumerate(pages):
        page_id = f"{document_id}:p{page_index:04d}"
        image = Image.new("RGB", image_size, (255, 255, 255))
        draw = ImageDraw.Draw(image)
        cursor_y = 16
        line_records: list[dict[str, Any]] = []

        for line_index, text in enumerate(texts):
            prefix = f"page {page_index} line {line_index}"
            try:
                bounds = draw.textbbox((0, 0), text, font=loaded_font)
            except (OSError, UnicodeError, ValueError) as exc:
                raise ValueError(f"{prefix} cannot be measured with the selected font: {exc}") from exc
            left_bearing, top_bearing, right_bearing, bottom_bearing = bounds
            text_width = right_bearing - left_bearing
            text_height = bottom_bearing - top_bearing
            if text_width <= 0 or text_height <= 0:
                raise ValueError(f"{prefix} has no measurable area with the selected font")

            left = 16
            top = cursor_y
            right = left + text_width
            bottom = top + text_height
            if right > width or bottom > height:
                raise ValueError(
                    f"{prefix} overflows image size {image_size}: "
                    f"text bounds {(left, top, right, bottom)}"
                )
            try:
                draw.text(
                    (left - left_bearing, top - top_bearing),
                    text,
                    font=loaded_font,
                    fill=(0, 0, 0),
                )
            except (OSError, UnicodeError, ValueError) as exc:
                raise ValueError(f"{prefix} cannot be rendered with the selected font: {exc}") from exc

            line_id = f"{page_id}:l{line_index:04d}"
            line_records.append(
                {
                    "id": line_id,
                    "polygon_px": [[left, top], [right, top], [right, bottom], [left, bottom]],
                    "text": text,
                    "role": "PARA_LINE",
                }
            )
            cursor_y = bottom + rng.randint(4, 10)

        png = BytesIO()
        image.save(png, format="PNG", optimize=False)
        image_bytes = png.getvalue()
        annotation = {
            "schema_version": "4.0",
            "document_id": document_id,
            "page_id": page_id,
            "image": {
                "width": width,
                "height": height,
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
            },
            "supervision": {"lines": "complete"},
            "lines": line_records,
            "regions": [],
            "cells": [],
            "relations": [],
            "reading_order": [line["id"] for line in line_records],
            "provenance": {
                "source_class": "synthetic",
                "license": "generated",
                "permission_reference": "not_applicable",
                "reviewer": "synthetic_generator",
            },
            "annotation": {"source": "synthetic", "generator": "prismocr.data.synthetic"},
        }
        rendered.append((f"page-{page_index:04d}", image_bytes, annotation))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[str] = []
    for stem, image_bytes, annotation in rendered:
        image_name = f"{stem}.png"
        annotation_name = f"{stem}.json"
        (output_path / image_name).write_bytes(image_bytes)
        (output_path / annotation_name).write_text(
            json.dumps(annotation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_rows.append(
            json.dumps(
                {"image_path": image_name, "annotation_path": annotation_name},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    manifest_path = output_path / "manifest.jsonl"
    manifest_path.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    return manifest_path
