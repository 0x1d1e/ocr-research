"""Interactive Gradio demo application for testing PrismOCR checkpoints."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import socket
import time
from typing import Any

import gradio as gr
from PIL import Image, ImageDraw
import torch

from prismocr.api import DocumentResult, InferenceOptions, PrismOCR

DEFAULT_RUNS_DIR = "/home/kosal/data/prismocr-runs"
REQUIRED_ARTIFACT_FILES = (
    "model.pt",
    "model-config.json",
    "vocab-v4.json",
    "manifest.json",
)

_MODEL_CACHE: dict[tuple[str, str], PrismOCR] = {}


def is_port_available(port: int, host: str = "0.0.0.0") -> bool:
    """Check if a given port is available for binding on the specified host."""
    if not (1 <= port < 65536):
        return False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
            return True
    except OSError:
        return False


def find_free_port(start_port: int = 7860, host: str = "0.0.0.0") -> int:
    """Find the first available port >= start_port using a stream socket."""
    port = max(1, int(start_port))
    while port < 65536:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((host, port))
                return port
        except OSError:
            port += 1
    raise RuntimeError(f"No available port found starting from {start_port} on host {host}")


def discover_artifacts(base_dir: str | Path | None = None) -> dict[str, str]:
    """Scan base_dir (or DEFAULT_RUNS_DIR) for directories containing PrismOCR checkpoint artifacts.

    Returns a mapping of model label -> absolute path string.
    """
    if base_dir is None:
        base_dir = DEFAULT_RUNS_DIR
    base_path = Path(base_dir).expanduser().resolve()
    if not base_path.exists() or not base_path.is_dir():
        return {}

    artifacts: dict[str, str] = {}
    for root, _dirs, files in os.walk(base_path, followlinks=True):
        file_set = set(files)
        if all(req in file_set for req in REQUIRED_ARTIFACT_FILES):
            dir_path = Path(root).resolve()
            if dir_path.name == "artifact" and dir_path.parent != dir_path:
                label = dir_path.parent.name
            else:
                label = dir_path.name

            if label in artifacts:
                try:
                    rel = dir_path.relative_to(base_path)
                    label = str(rel.parent if rel.name == "artifact" else rel)
                except ValueError:
                    label = str(dir_path)

            artifacts[label] = str(dir_path)

    return dict(sorted(artifacts.items()))


def clear_model_cache() -> None:
    """Clear all cached PrismOCR model instances."""
    _MODEL_CACHE.clear()


def get_cached_model(path: str | Path, device: str = "cpu") -> PrismOCR:
    """Load or retrieve a cached PrismOCR model instance for the given path and device."""
    canonical_path = str(Path(path).expanduser().resolve())
    canonical_device = str(device).lower()
    key = (canonical_path, canonical_device)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = PrismOCR.load(canonical_path, device=device)
    return _MODEL_CACHE[key]


# Alias for load caching
get_model = get_cached_model


def _extract_pixel_points(
    polygon: Sequence[Sequence[float]] | None,
    width: int,
    height: int,
    is_normalized: bool,
) -> list[tuple[float, float]]:
    if not polygon:
        return []
    points: list[tuple[float, float]] = []
    for pt in polygon:
        if len(pt) >= 2:
            x, y = float(pt[0]), float(pt[1])
            if is_normalized:
                x *= width
                y *= height
            points.append((x, y))
    return points


def annotate_document(image: Image.Image, result: DocumentResult | dict[str, Any]) -> Image.Image:
    """Draw line polygons (green outline) and region polygons (orange/blue outline) on a copy of image."""
    if hasattr(result, "to_dict"):
        payload = result.to_dict()
    elif isinstance(result, dict):
        payload = result
    else:
        payload = getattr(result, "payload", {})

    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size

    coordinate_space = payload.get("coordinate_space", "pixels")
    is_normalized = coordinate_space == "normalized"

    # Fallback auto-detection if coordinate_space is missing or default but coords are in [0, 1]
    if not is_normalized and (width > 1 and height > 1):
        all_coords: list[float] = []
        for line in payload.get("lines", []):
            for pt in line.get("polygon", []):
                all_coords.extend(pt[:2])
        for region in payload.get("regions", []):
            for pt in region.get("polygon", []):
                all_coords.extend(pt[:2])
        if all_coords and all(0.0 <= c <= 1.0 for c in all_coords):
            is_normalized = True

    # Draw regions first (orange for general regions, blue for tables)
    regions = payload.get("regions", [])
    for region in regions:
        pts = _extract_pixel_points(region.get("polygon"), width, height, is_normalized)
        kind = str(region.get("kind", "")).upper()
        outline = "blue" if kind == "TABLE" else "orange"
        if len(pts) >= 3:
            draw.polygon(pts, outline=outline, width=2)
        elif len(pts) == 2:
            draw.line(pts, fill=outline, width=2)

    # Draw cells if present
    cells = payload.get("cells", [])
    for cell in cells:
        pts = _extract_pixel_points(cell.get("polygon"), width, height, is_normalized)
        if len(pts) >= 3:
            draw.polygon(pts, outline="blue", width=1)
        elif len(pts) == 2:
            draw.line(pts, fill="blue", width=1)

    # Draw lines (green outline)
    lines = payload.get("lines", [])
    for line in lines:
        pts = _extract_pixel_points(line.get("polygon"), width, height, is_normalized)
        if len(pts) >= 3:
            draw.polygon(pts, outline="green", width=2)
        elif len(pts) == 2:
            draw.line(pts, fill="green", width=2)

    return annotated


def process_document(
    image: Image.Image | None,
    model_name: str | None,
    device: str = "cpu",
    resolution: str | int | None = None,
    allow_tiling: bool = True,
    base_dir: str | Path | None = None,
) -> tuple[Image.Image | None, str, dict[str, Any], str]:
    """Execute inference on image and return (annotated_image, extracted_text, json_payload, diagnostics_markdown)."""
    if image is None:
        return None, "", {}, "### No Image Provided\nPlease upload a document image to run OCR."

    artifacts = discover_artifacts(base_dir)
    model_path: str | None = None
    if model_name and model_name in artifacts:
        model_path = artifacts[model_name]
    elif model_name and Path(model_name).exists():
        model_path = str(Path(model_name).resolve())
    elif artifacts:
        # Default to first discovered artifact if none specified
        first_label = next(iter(artifacts))
        model_path = artifacts[first_label]

    if not model_path:
        return None, "", {}, f"### Error\nModel '{model_name}' not found and no checkpoints discovered."

    if resolution in (None, "None", "Original", "", 0):
        page_resolution = None
    else:
        try:
            page_resolution = int(resolution)
        except (ValueError, TypeError):
            page_resolution = None

    options = InferenceOptions(
        page_resolution=page_resolution,
        allow_tiling=bool(allow_tiling),
    )

    try:
        model = get_cached_model(model_path, device=device)
        start_time = time.perf_counter()
        result = model.predict(image, options=options)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
    except Exception as exc:
        return (
            None,
            "",
            {"error": str(exc), "stage": getattr(exc, "stage", "prediction")},
            f"### Inference Error\n**Error type**: `{type(exc).__name__}`\n\n```\n{exc}\n```",
        )

    annotated = annotate_document(image, result)
    payload = result.to_dict()

    # Extract text with lines ordered by reading_order
    lines = payload.get("lines", [])
    ordered_lines = sorted(
        lines,
        key=lambda l: (
            l.get("reading_order") is None,
            l.get("reading_order", 0) if l.get("reading_order") is not None else 0,
            l.get("id", ""),
        ),
    )
    extracted_text = "\n".join(
        str(l["text"]) for l in ordered_lines if str(l.get("text", "")).strip()
    )

    diag = payload.get("diagnostics", {})
    warnings = payload.get("warnings", [])

    diag_lines = [
        "### Latency & Diagnostics",
        f"- **Latency**: {latency_ms:.1f} ms",
        f"- **Model Version**: `{payload.get('model_version', 'unknown')}`",
        f"- **Lines detected**: {len(lines)}",
        f"- **Regions detected**: {len(payload.get('regions', []))}",
        f"- **Tables detected**: {len(payload.get('tables', []))}",
        f"- **Tiles processed**: {diag.get('tiles', 1)} (model calls: {diag.get('tile_model_calls', 1)})",
    ]
    if warnings:
        diag_lines.append("\n**Warnings:**")
        for w in warnings:
            code = w.get("code", str(w)) if isinstance(w, dict) else str(w)
            diag_lines.append(f"- `{code}`")

    diagnostics_md = "\n".join(diag_lines)
    return annotated, extracted_text, payload, diagnostics_md


def build_demo(base_dir: str | Path | None = None) -> gr.Blocks:
    """Build the Gradio demo UI blocks for PrismOCR."""
    artifacts = discover_artifacts(base_dir)
    model_choices = list(artifacts.keys())
    default_model = model_choices[0] if model_choices else None

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device_choices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu", "cuda"]

    def run_ocr(
        model_name: str | None,
        device: str,
        resolution: str | int | None,
        allow_tiling: bool,
        image: Image.Image | None,
    ) -> tuple[Image.Image | None, str, dict[str, Any], str]:
        return process_document(
            image=image,
            model_name=model_name,
            device=device,
            resolution=resolution,
            allow_tiling=allow_tiling,
            base_dir=base_dir,
        )

    with gr.Blocks(title="PrismOCR Interactive Demo") as demo:
        gr.Markdown("# PrismOCR Interactive Demo")
        gr.Markdown(
            "Upload a document image to test PrismOCR checkpoints for end-to-end OCR, "
            "line recognition, reading order reassembly, and layout analysis."
        )

        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    label="Model Checkpoint",
                    choices=model_choices,
                    value=default_model,
                    interactive=True,
                )
                device_dropdown = gr.Dropdown(
                    label="Device",
                    choices=device_choices,
                    value=default_device,
                    interactive=True,
                )
                resolution_dropdown = gr.Dropdown(
                    label="Page Resolution",
                    choices=["Original", "1024", "1536", "2048"],
                    value="Original",
                    interactive=True,
                )
                tiling_checkbox = gr.Checkbox(
                    label="Allow Tiling",
                    value=True,
                    interactive=True,
                )
                image_input = gr.Image(
                    type="pil",
                    label="Input Document Image",
                )
                run_btn = gr.Button("Run OCR", variant="primary")

            with gr.Column(scale=1):
                annotated_output = gr.Image(
                    type="pil",
                    label="Annotated Document",
                )
                text_output = gr.Textbox(
                    label="Extracted Text (Reading Order)",
                    lines=10,
                    interactive=False,
                )
                diagnostics_output = gr.Markdown(
                    label="Latency & Diagnostics",
                )
                json_output = gr.JSON(
                    label="Document Result JSON",
                )

        run_btn.click(
            fn=run_ocr,
            inputs=[
                model_dropdown,
                device_dropdown,
                resolution_dropdown,
                tiling_checkbox,
                image_input,
            ],
            outputs=[
                annotated_output,
                text_output,
                json_output,
                diagnostics_output,
            ],
        )

    return demo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PrismOCR Gradio Demo Application")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the demo server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind the demo server to (default: first available port >= 7860)",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default=None,
        help="Base directory containing PrismOCR run artifacts",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Generate a publicly shareable Gradio link",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for launching the PrismOCR demo."""
    args = parse_args(argv)

    port = args.port
    if port is None or not is_port_available(port, host=args.host):
        start_port = port if port is not None else 7860
        port = find_free_port(start_port=start_port, host=args.host)

    demo = build_demo(base_dir=args.runs_dir)
    demo.launch(server_name=args.host, server_port=port, share=args.share)


if __name__ == "__main__":
    main()
