"""Export and verify the complete PrismOCR inference tensor path on ONNX Runtime CPU.

The core graph contains the backbone, both proposal heads, and pair-relation
logits. Four recognizer graphs keep line count dynamic while fixing CTC sequence
length to the model's supported buckets. CTC, graph, table, and schema decoding
remain deterministic Python operations in :class:`PrismONNXRuntime`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from prismocr.api import (
    _RELATION_DIAGNOSTIC_LIMIT,
    InferenceError,
    InferenceOptions,
    PrismOCR,
    _calibrated_grid_score,
    _calibrated_relation_score,
    _calibrated_score,
    _validated_calibration,
)
from prismocr.data.vocabulary import Vocabulary
from prismocr.model.heads import validate_model_contract_version
from prismocr.model.network import (
    validated_model_state,
    REGION_KINDS,
    RELATION_NAMES,
    TEXT_ROLES,
    PrismOCRModel,
)
from prismocr.graph_decode import relation_candidate_error
from prismocr.model.recognizer import (
    STRIP_BUCKETS,
    choose_inference_bucket,
    validate_strip_buckets,
)
from prismocr.tiling import Tile, TileResult, stitch_tile_results

PRISMOCR_VERSION = "0.1.0"
FORMAT_NAME = "prismocr-onnx"
FORMAT_VERSION = 2
ONNX_OPSET = 18
ONNX_PACKAGE_VERSION = "1.17.0"
ONNX_RUNTIME_VERSION = "1.24.3"
CORE_FILENAME = "core.onnx"
VOCAB_FILENAME = "vocab-v4.json"
CORE_OUTPUT_NAMES = (
    "glyph_features",
    "text_heatmap_logits",
    "text_boxes",
    "text_scores",
    "text_candidate_mask",
    "text_mask",
    "text_role_logits",
    "text_embeddings",
    "text_proposal_indices",
    "text_candidate_count",
    "text_capacity_overflow",
    "region_boxes",
    "region_class_logits",
    "region_scores",
    "region_kinds",
    "region_candidate_mask",
    "region_mask",
    "region_embeddings",
    "region_row_logits",
    "region_col_logits",
    "region_rowspan_logits",
    "region_colspan_logits",
    "region_header_logits",
    "graph_logits",
    "graph_mask",
)
RECOGNIZER_INPUT_NAMES = ("glyph_features", "boxes", "node_embeddings", "refs")
RECOGNIZER_OUTPUT_NAMES = ("logits",)
DEFAULT_PARITY_SIZES = ((64, 80), (96, 128))
PARITY_SEED = 1703
PARITY_ATOL = 2e-4
PARITY_RTOL = 2e-3
MAX_FP32_TENSOR_BYTES = 10_000_000
ONNX_WEIGHTS_FILENAME = "model-weights.bin"


class ONNXExportError(RuntimeError):
    """Export, runtime compatibility, artifact, or parity failure."""


def _optional_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ONNXExportError(
            f"ONNX export requires the optional '{name}' dependency; "
            "install PrismOCR with its export extra"
        ) from exc


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_source(
    artifact_dir: str | Path,
) -> tuple[PrismOCRModel, Vocabulary, dict[str, Any], dict[str, Any], dict[str, str]]:
    root = Path(artifact_dir)
    config_path = root / "model-config.json"
    weights_path = root / "model.pt"
    vocabulary_path = root / VOCAB_FILENAME
    for path in (config_path, weights_path, vocabulary_path):
        if not path.is_file():
            raise ONNXExportError(f"required PyTorch artifact file is missing: {path}")
    try:
        engine = PrismOCR.load(root, device="cpu")
    except (ValueError, FileNotFoundError, RuntimeError, KeyError, TypeError) as exc:
        raise ONNXExportError(f"cannot load source PyTorch artifact: {exc}") from exc
    vocabulary = engine.vocabulary
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["model_contract_version"] = engine.model.model_contract_version
    calibration_path = root / "calibration-v4.json"
    calibration = dict(engine.calibration)
    checksums = {
        "model_sha256": _sha256_file(weights_path),
        "vocabulary_sha256": vocabulary.sha256,
        "config_sha256": _sha256_file(config_path),
        "calibration_sha256": _sha256_file(calibration_path) if calibration_path.is_file() else None,
    }
    return engine.model.eval(), vocabulary, config, calibration, checksums


def _load_unfused_source_model(
    artifact_dir: str | Path,
    vocabulary: Vocabulary,
    config: dict[str, Any],
) -> PrismOCRModel:
    """Load the exact trainable checkpoint form for fusion parity verification."""
    root = Path(artifact_dir)
    source_config = dict(config)
    try:
        checkpoint = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
        model = PrismOCRModel(vocabulary=vocabulary, **source_config)
        model.load_state_dict(validated_model_state(checkpoint, source_config), strict=True)
    except (OSError, RuntimeError, TypeError, KeyError, ValueError) as exc:
        raise ONNXExportError(f"cannot load unfused source model for parity: {exc}") from exc
    return model.eval()


class _CoreGraph(nn.Module):
    """Tensor-only view of the shared PrismOCR inference computation."""

    def __init__(self, model: PrismOCRModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        model = self.model
        f4, f8, f16 = model.backbone(images)
        raw_text = model.text_head(f4)
        text = model.text_head.proposals(
            raw_text,
            max_proposals=model.text_proposals,
            score_threshold=model.text_threshold,
            model_contract_version=model.model_contract_version,
            image_size=(images.shape[-1], images.shape[-2]),
        )
        regions = model.region_head(f16, model.region_threshold)
        glyph_features = model.lat4(f4) + F.interpolate(
            model.lat8(f8), size=f4.shape[-2:], mode="bilinear", align_corners=False
        )
        nodes = torch.cat((text["embeddings"], regions["embeddings"]), dim=1)
        graph_logits = model.relation_head(nodes)
        graph_mask = torch.cat((text["mask"], regions["mask"]), dim=1)
        return (
            glyph_features,
            raw_text["heatmap_logits"],
            text["boxes"],
            text["scores"],
            text["candidate_mask"],
            text["mask"],
            text["role_logits"],
            text["embeddings"],
            text["proposal_indices"],
            text["candidate_count"],
            text["capacity_overflow"],
            regions["boxes"],
            regions["class_logits"],
            regions["scores"],
            regions["kinds"],
            regions["candidate_mask"],
            regions["mask"],
            regions["embeddings"],
            regions["row_logits"],
            regions["col_logits"],
            regions["rowspan_logits"],
            regions["colspan_logits"],
            regions["header_logits"],
            graph_logits,
            graph_mask,
        )


class _RecognizerGraph(nn.Module):
    def __init__(self, model: PrismOCRModel, length: int) -> None:
        super().__init__()
        self.recognizer = model.recognizer
        self.length = length

    def forward(
        self,
        glyph_features: torch.Tensor,
        boxes: torch.Tensor,
        node_embeddings: torch.Tensor,
        refs: torch.Tensor,
    ) -> torch.Tensor:
        return self.recognizer.forward_bucket(
            glyph_features, boxes, node_embeddings, refs, self.length
        )


def _graph_record(path: Path, inputs: Sequence[str], outputs: Sequence[str]) -> dict[str, Any]:
    return {
        "file": path.name,
        "sha256": _sha256_file(path),
        "inputs": list(inputs),
        "outputs": list(outputs),
    }


def _externalize_shared_initializers(
    graph_paths: Sequence[Path], onnx: Any, output_dir: Path
) -> dict[str, Any]:
    """Store ONNX initializers once across graphs and enforce the FP32 inventory cap."""
    models = [onnx.load(str(path), load_external_data=False) for path in graph_paths]
    shared: dict[tuple[int, tuple[int, ...], str], tuple[int, int]] = {}
    payload = bytearray()
    fp32_unique_bytes = 0
    initializer_count = 0
    for model in models:
        for tensor in model.graph.initializer:
            array = onnx.numpy_helper.to_array(tensor)
            raw = array.tobytes(order="C")
            if not raw:
                continue
            key = (tensor.data_type, tuple(tensor.dims), _sha256(raw))
            location = shared.get(key)
            if location is None:
                location = (len(payload), len(raw))
                shared[key] = location
                payload.extend(raw)
                if tensor.data_type == onnx.TensorProto.FLOAT:
                    fp32_unique_bytes += len(raw)
            initializer_count += 1
            tensor.raw_data = raw
            for field in (
                "float_data", "int32_data", "string_data", "int64_data",
                "double_data", "uint64_data",
            ):
                tensor.ClearField(field)
            onnx.external_data_helper.set_external_data(
                tensor,
                location=ONNX_WEIGHTS_FILENAME,
                offset=location[0],
                length=location[1],
            )
            tensor.ClearField("raw_data")

    if fp32_unique_bytes > MAX_FP32_TENSOR_BYTES:
        raise ONNXExportError(
            f"ONNX FP32 tensor inventory is {fp32_unique_bytes} bytes, exceeding the "
            f"{MAX_FP32_TENSOR_BYTES}-byte limit"
        )
    weights_path = output_dir / ONNX_WEIGHTS_FILENAME
    temporary_weights = weights_path.with_suffix(weights_path.suffix + ".tmp")
    temporary_weights.write_bytes(payload)
    temporary_weights.replace(weights_path)
    for path, model in zip(graph_paths, models):
        path.write_bytes(model.SerializeToString())
        onnx.checker.check_model(str(path))
    return {
        "max_fp32_bytes": MAX_FP32_TENSOR_BYTES,
        "fp32_unique_bytes": fp32_unique_bytes,
        "unique_initializer_bytes": len(payload),
        "initializer_reference_count": initializer_count,
        "unique_initializer_count": len(shared),
        "external_data_file": weights_path.name,
        "external_data_file_bytes": weights_path.stat().st_size,
        "external_data_file_sha256": _sha256_file(weights_path),
    }


def _validate_export_stack(onnx: Any, ort: Any, opset: int) -> None:
    if opset != ONNX_OPSET:
        raise ONNXExportError(
            f"unsupported ONNX opset {opset}; this exporter supports opset {ONNX_OPSET}"
        )
    if onnx.__version__ != ONNX_PACKAGE_VERSION or ort.__version__ != ONNX_RUNTIME_VERSION:
        raise ONNXExportError(
            "unsupported ONNX export matrix: expected "
            f"onnx=={ONNX_PACKAGE_VERSION}, onnxruntime=={ONNX_RUNTIME_VERSION}; "
            f"found onnx=={onnx.__version__}, onnxruntime=={ort.__version__}"
        )


def _validate_model(model: PrismOCRModel, opset: int) -> None:
    if opset != ONNX_OPSET:
        raise ONNXExportError(
            f"unsupported ONNX opset {opset}; this exporter supports opset {ONNX_OPSET}"
        )
    if model.recognizer.variant not in {"transformer", "gated_conv"}:
        raise ONNXExportError(
            f"unsupported recognizer variant {model.recognizer.variant!r}"
        )
    if model.dim % 5 or model.recognizer.strip_height <= 0:
        raise ONNXExportError("model dimensions are unsupported by the reference ONNX graphs")


def export_graphs(
    artifact_dir: str | Path,
    output_dir: str | Path,
    *,
    opset: int = ONNX_OPSET,
) -> dict[str, Any]:
    """Export core and four dynamic-N recognizer graphs and write their manifest."""
    source = Path(artifact_dir).resolve()
    output_root = Path(output_dir).resolve()
    if source == output_root:
        raise ONNXExportError("ONNX output directory must differ from the source artifact directory")
    onnx = _optional_module("onnx")
    ort = _optional_module("onnxruntime")
    _validate_export_stack(onnx, ort, opset)
    model, vocabulary, config, calibration, source_checksums = _load_source(artifact_dir)
    _validate_model(model, opset)
    output = output_root
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").unlink(missing_ok=True)
    vocabulary_path = output / VOCAB_FILENAME
    shutil.copyfile(source / VOCAB_FILENAME, vocabulary_path)
    source_calibration_path = source / "calibration-v4.json"
    calibration_path = output / "calibration-v4.json"
    if source_calibration_path.is_file():
        shutil.copyfile(source_calibration_path, calibration_path)
    else:
        calibration_path.unlink(missing_ok=True)

    core = _CoreGraph(model).eval()
    example_images = torch.zeros((1, 3, 64, 80), dtype=torch.float32)
    core_dynamic_axes: dict[str, dict[int, str]] = {
        "images": {2: "image_height", 3: "image_width"},
        "glyph_features": {2: "glyph_height", 3: "glyph_width"},
        "text_heatmap_logits": {2: "heatmap_height", 3: "heatmap_width"},
    }
    try:
        torch.onnx.export(
            core,
            (example_images,),
            str(output / CORE_FILENAME),
            input_names=["images"],
            output_names=list(CORE_OUTPUT_NAMES),
            opset_version=opset,
            dynamic_axes=core_dynamic_axes,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
        )
        core_model = onnx.load(str(output / CORE_FILENAME))
        onnx.checker.check_model(core_model)

        glyph = torch.randn((1, model.dim, 16, 20), dtype=torch.float32)
        boxes = torch.tensor(
            [[0.3, 0.5, 0.45, 0.12, 0.0, 1.0], [0.7, 0.5, 0.45, 0.12, 0.0, 1.0]],
            dtype=torch.float32,
        )
        embeddings = torch.randn((2, model.dim), dtype=torch.float32)
        refs = torch.tensor([[0, 0], [0, 1]], dtype=torch.int64)
        recognizer_records: dict[str, Any] = {}
        for bucket in model.strip_buckets:
            filename = f"recognizer-{bucket}.onnx"
            recognizer = _RecognizerGraph(model, bucket).eval()
            torch.onnx.export(
                recognizer,
                (glyph, boxes, embeddings, refs),
                str(output / filename),
                input_names=list(RECOGNIZER_INPUT_NAMES),
                output_names=list(RECOGNIZER_OUTPUT_NAMES),
                opset_version=opset,
                dynamic_axes={
                    "glyph_features": {2: "glyph_height", 3: "glyph_width"},
                    "boxes": {0: "proposal_count"},
                    "node_embeddings": {0: "proposal_count"},
                    "refs": {0: "proposal_count"},
                    "logits": {0: "proposal_count"},
                },
                do_constant_folding=True,
                export_params=True,
                dynamo=False,
            )
            recognizer_model = onnx.load(str(output / filename))
            onnx.checker.check_model(recognizer_model)
            recognizer_records[str(bucket)] = _graph_record(
                output / filename, RECOGNIZER_INPUT_NAMES, RECOGNIZER_OUTPUT_NAMES
            )
    except Exception as exc:
        if isinstance(exc, ONNXExportError):
            raise
        raise ONNXExportError(f"failed to export or validate ONNX graphs: {exc}") from exc

    graph_paths = [output / CORE_FILENAME, *(output / record["file"] for record in recognizer_records.values())]
    try:
        tensor_inventory = _externalize_shared_initializers(graph_paths, onnx, output)
    except ONNXExportError:
        raise
    except Exception as exc:
        raise ONNXExportError(f"failed to deduplicate ONNX initializer storage: {exc}") from exc

    # Externalizing initializers rewrites each ONNX protobuf, so checksums
    # captured during graph export are stale. Bind the manifest to final bytes.
    recognizer_records = {
        str(bucket): _graph_record(output / f"recognizer-{bucket}.onnx", RECOGNIZER_INPUT_NAMES, RECOGNIZER_OUTPUT_NAMES)
        for bucket in model.strip_buckets
    }

    manifest: dict[str, Any] = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "opset": opset,
        "versions": {
            "prismocr": PRISMOCR_VERSION,
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "runtime_matrix": {
            "opset": ONNX_OPSET,
            "onnx": ONNX_PACKAGE_VERSION,
            "onnxruntime": ONNX_RUNTIME_VERSION,
            "execution_provider": "CPUExecutionProvider",
        },
        "runtime": {
            "name": "onnxruntime",
            "version": ort.__version__,
            "execution_provider": "CPUExecutionProvider",
        },
        "source": source_checksums,
        "tensor_inventory": {
            **tensor_inventory,
            "source_model_tensor_bytes": model.tensor_bytes(),
        },
        "model": {
            "config": config,
            "recognizer": model.recognizer.variant,
            "buckets": list(model.strip_buckets),
            "text_roles": list(TEXT_ROLES),
            "region_kinds": list(REGION_KINDS),
            "relation_names": list(RELATION_NAMES),
        },
        "calibration": {
            "file": calibration_path.name if source_calibration_path.is_file() else None,
            "sha256": source_checksums["calibration_sha256"],
            "settings": calibration,
        },
        "vocabulary": {
            "file": VOCAB_FILENAME,
            "sha256": _sha256_file(vocabulary_path),
            "size": len(vocabulary),
            "version": vocabulary.version,
        },
        "graphs": {
            "core": _graph_record(output / CORE_FILENAME, ("images",), CORE_OUTPUT_NAMES),
            "recognizers": recognizer_records,
        },
        "input": {
            "name": "images",
            "dtype": "float32",
            "shape": [1, 3, "height", "width"],
            "layout": "NCHW RGB in [0,1]",
            "height_width_multiple": 16,
            "preprocessing": "caller supplies RGB float32 pixels scaled to [0,1]; no resize or normalization is applied",
        },
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _torch_core_reference(model: PrismOCRModel, images: torch.Tensor) -> dict[str, torch.Tensor]:
    outputs = model(images)
    text = outputs["text"]
    regions = outputs["regions"]
    f4, f8, _ = model.backbone(images)
    glyph_features = model.lat4(f4) + F.interpolate(
        model.lat8(f8), size=f4.shape[-2:], mode="bilinear", align_corners=False
    )
    return {
        "glyph_features": glyph_features,
        "text_heatmap_logits": outputs["text_heatmap_logits"],
        "text_boxes": text["boxes"],
        "text_scores": text["scores"],
        "text_candidate_mask": text["candidate_mask"],
        "text_mask": text["mask"],
        "text_role_logits": text["role_logits"],
        "text_embeddings": text["embeddings"],
        "text_proposal_indices": text["proposal_indices"],
        "text_candidate_count": text["candidate_count"],
        "text_capacity_overflow": text["capacity_overflow"],
        "region_boxes": regions["boxes"],
        "region_class_logits": regions["class_logits"],
        "region_scores": regions["scores"],
        "region_kinds": regions["kinds"],
        "region_candidate_mask": regions["candidate_mask"],
        "region_mask": regions["mask"],
        "region_embeddings": regions["embeddings"],
        "region_row_logits": regions["row_logits"],
        "region_col_logits": regions["col_logits"],
        "region_rowspan_logits": regions["rowspan_logits"],
        "region_colspan_logits": regions["colspan_logits"],
        "region_header_logits": regions["header_logits"],
        "graph_logits": outputs["graph_logits"],
        "graph_mask": outputs["graph_mask"],
    }


def _assert_close(
    actual: np.ndarray,
    expected: torch.Tensor | np.ndarray,
    *,
    label: str,
    atol: float = PARITY_ATOL,
    rtol: float = PARITY_RTOL,
) -> None:
    actual_array = actual.detach().cpu().numpy() if isinstance(actual, torch.Tensor) else actual
    reference = expected.detach().cpu().numpy() if isinstance(expected, torch.Tensor) else expected
    if actual_array.shape != reference.shape:
        raise ONNXExportError(
            f"parity failed for {label}: actual shape {actual_array.shape}, reference shape {reference.shape}"
        )
    if (
        np.issubdtype(actual_array.dtype, np.inexact)
        and not np.isfinite(actual_array).all()
    ) or (
        np.issubdtype(reference.dtype, np.inexact)
        and not np.isfinite(reference).all()
    ):
        raise ONNXExportError(f"parity failed for {label}: non-finite values in model output")
    if actual_array.dtype.kind in "biu" or reference.dtype.kind in "biu":
        if not np.array_equal(actual_array, reference):
            raise ONNXExportError(f"parity failed for {label}: discrete outputs differ")
        return
    try:
        np.testing.assert_allclose(actual_array, reference, atol=atol, rtol=rtol)
    except AssertionError as exc:
        difference = float(np.max(np.abs(actual_array - reference))) if actual_array.size else 0.0
        raise ONNXExportError(
            f"parity failed for {label}: max absolute error {difference:.6g} "
            f"(atol={atol}, rtol={rtol})"
        ) from exc


def _fusion_parity_report(
    fused_model: PrismOCRModel,
    unfused_model: PrismOCRModel,
    image_sizes: Sequence[tuple[int, int]],
    *,
    seed: int,
) -> dict[str, Any]:
    """Compare the fused inference model with its checkpoint form in FP32 and FP16."""
    precision_cases = (
        ("float32", torch.float32, PARITY_ATOL, PARITY_RTOL),
        ("float16", torch.float16, 2e-3, 1e-2),
    )
    reports: list[dict[str, Any]] = []
    for precision_index, (name, dtype, atol, rtol) in enumerate(precision_cases):
        fused = copy.deepcopy(fused_model).to(dtype=dtype).eval()
        unfused = copy.deepcopy(unfused_model).to(dtype=dtype).eval()
        generator = torch.Generator(device="cpu").manual_seed(seed + precision_index)
        checked_outputs = 0
        max_absolute_error = 0.0
        max_relative_error = 0.0
        with torch.inference_mode():
            for height, width in image_sizes:
                images = torch.rand((1, 3, height, width), generator=generator, dtype=dtype)
                expected = _torch_core_reference(unfused, images)
                actual = _torch_core_reference(fused, images)
                for output_name, expected_tensor in expected.items():
                    actual_tensor = actual[output_name]
                    _assert_close(
                        actual_tensor,
                        expected_tensor,
                        label=f"fusion:{name}:{height}x{width}:{output_name}",
                        atol=atol,
                        rtol=rtol,
                    )
                    if expected_tensor.is_floating_point():
                        expected_values = expected_tensor.float().cpu().numpy()
                        actual_values = actual_tensor.float().cpu().numpy()
                        if expected_values.size:
                            difference = np.abs(actual_values - expected_values)
                            max_absolute_error = max(
                                max_absolute_error, float(difference.max())
                            )
                            relative = difference / np.maximum(
                                np.abs(expected_values), np.finfo(np.float32).tiny
                            )
                            max_relative_error = max(
                                max_relative_error, float(relative.max())
                            )
                    checked_outputs += 1

                expected_glyph = expected["glyph_features"]
                actual_glyph = actual["glyph_features"]
                boxes = torch.tensor(
                    [[0.3, 0.5, 0.45, 0.12, 0.0, 1.0], [0.7, 0.5, 0.45, 0.12, 0.0, 1.0]],
                    dtype=dtype,
                )
                embeddings = torch.randn(
                    (2, fused.dim), generator=generator, dtype=dtype
                )
                refs = torch.tensor([[0, 0], [0, 1]], dtype=torch.int64)
                for bucket in fused.strip_buckets:
                    expected_logits = unfused.recognizer.forward_bucket(
                        expected_glyph, boxes, embeddings, refs, bucket
                    )
                    actual_logits = fused.recognizer.forward_bucket(
                        actual_glyph, boxes, embeddings, refs, bucket
                    )
                    _assert_close(
                        actual_logits,
                        expected_logits,
                        label=f"fusion:{name}:{height}x{width}:recognizer-{bucket}",
                        atol=atol,
                        rtol=rtol,
                    )
                    difference = np.abs(
                        actual_logits.float().cpu().numpy()
                        - expected_logits.float().cpu().numpy()
                    )
                    if difference.size:
                        max_absolute_error = max(
                            max_absolute_error, float(difference.max())
                        )
                        relative = difference / np.maximum(
                            np.abs(expected_logits.float().cpu().numpy()),
                            np.finfo(np.float32).tiny,
                        )
                        max_relative_error = max(
                            max_relative_error, float(relative.max())
                        )
                    checked_outputs += 1
        reports.append(
            {
                "precision": name,
                "image_sizes_hw": [list(size) for size in image_sizes],
                "tensor_outputs_checked": checked_outputs,
                "atol": atol,
                "rtol": rtol,
                "max_absolute_error": max_absolute_error,
                "max_relative_error": max_relative_error,
                "status": "passed",
            }
        )
    return {"status": "passed", "cases": reports}


def verify_parity(
    artifact_dir: str | Path,
    export_dir: str | Path,
    *,
    image_sizes: Sequence[tuple[int, int]] = DEFAULT_PARITY_SIZES,
    seed: int = PARITY_SEED,
) -> dict[str, Any]:
    """Compare all neural graph outputs and each bucket against eager PyTorch."""
    if not image_sizes:
        raise ONNXExportError("parity verification requires at least one image size")
    for height, width in image_sizes:
        if height <= 0 or width <= 0 or height % 16 or width % 16:
            raise ONNXExportError(
                f"unsupported parity image size {(height, width)}; dimensions must be positive multiples of 16"
            )

    from prismocr.onnx_parity import assert_decoded_document_parity

    model, vocabulary, config, calibration, source_checksums = _load_source(artifact_dir)
    pytorch_runtime = PrismOCR.load(artifact_dir, device="cpu")
    unfused_model = _load_unfused_source_model(
        artifact_dir, vocabulary, config
    )
    fusion_parity = _fusion_parity_report(
        model, unfused_model, image_sizes, seed=seed
    )
    runtime = PrismONNXRuntime(export_dir, _allow_unverified=True)
    manifest_path = Path(export_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source") != source_checksums:
        raise ONNXExportError("export source checksums do not match the PyTorch artifact")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    images_checked = 0
    decoded_document_cases: list[dict[str, Any]] = []
    recognizer_cases: list[dict[str, int]] = []
    with torch.inference_mode():
        for height, width in image_sizes:
            images = torch.rand((1, 3, height, width), generator=generator, dtype=torch.float32)
            reference_outputs = _torch_core_reference(model, images)
            runtime_outputs = runtime.run_tensors(images)
            for name, reference in reference_outputs.items():
                _assert_close(runtime_outputs["core"][name], reference, label=f"{height}x{width}:{name}")

            recognition_threshold = (
                max(float(config["text_threshold"]), float(calibration["text_threshold"]))
                if calibration.get("scores_calibrated") is True
                and "text_threshold" in calibration
                else None
            )
            pytorch_groups = model(
                images, recognition_threshold=recognition_threshold
            )["ocr_groups"]
            if set(pytorch_groups) != set(runtime_outputs["ocr_groups"]):
                raise ONNXExportError(
                    f"parity failed at {height}x{width}: OCR bucket assignments differ"
                )
            for bucket, group in pytorch_groups.items():
                actual = runtime_outputs["ocr_groups"][bucket]
                _assert_close(actual["refs"], group["refs"], label=f"{height}x{width}:bucket-{bucket}:refs")
                _assert_close(actual["logits"], group["logits"], label=f"{height}x{width}:bucket-{bucket}:logits")
            pytorch_document = pytorch_runtime.predict(images[0]).to_dict()
            onnx_document = decode_onnx_outputs(runtime_outputs, manifest, vocabulary)
            assert_decoded_document_parity(
                pytorch_document,
                onnx_document,
                label=f"{height}x{width}",
                atol=PARITY_ATOL,
                rtol=PARITY_RTOL,
            )
            decoded_document_cases.append({
                "image_size_hw": [height, width],
                "status": "passed",
                "lines": len(onnx_document["annotation"]["lines"]),
                "regions": len(onnx_document["annotation"]["regions"]),
                "cells": len(onnx_document["annotation"]["cells"]),
                "relations": len(onnx_document["annotation"]["relations"]),
            })
            images_checked += 1

            glyph = torch.from_numpy(runtime_outputs["core"]["glyph_features"])
            refs = torch.tensor([[0, 0], [0, 1]], dtype=torch.int64)
            boxes = torch.tensor(
                [[0.3, 0.5, 0.45, 0.12, 0.0, 1.0], [0.7, 0.5, 0.45, 0.12, 0.0, 1.0]],
                dtype=torch.float32,
            )
            embeddings = torch.randn((2, model.dim), generator=generator, dtype=torch.float32)
            for bucket in model.strip_buckets:
                torch_logits = model.recognizer.forward_bucket(
                    glyph, boxes, embeddings, refs, bucket
                )
                onnx_logits = runtime.run_recognizer(
                    bucket,
                    runtime_outputs["core"]["glyph_features"],
                    boxes.numpy(),
                    embeddings.numpy(),
                    refs.numpy(),
                )
                _assert_close(
                    onnx_logits,
                    torch_logits,
                    label=f"{height}x{width}:bucket-{bucket}:dynamic-N=2",
                )
                recognizer_cases.append({"bucket": bucket, "proposal_count": 2})

                dynamic_boxes = torch.cat(
                    (boxes, torch.tensor([[0.5, 0.7, 0.3, 0.08, 0.0, 1.0]], dtype=torch.float32))
                )
                dynamic_refs = torch.tensor([[0, 0], [0, 1], [0, 2]], dtype=torch.int64)
                dynamic_embeddings = torch.randn((3, model.dim), generator=generator, dtype=torch.float32)
                torch_dynamic = model.recognizer.forward_bucket(
                    glyph, dynamic_boxes, dynamic_embeddings, dynamic_refs, bucket
                )
                onnx_dynamic = runtime.run_recognizer(
                    bucket,
                    runtime_outputs["core"]["glyph_features"],
                    dynamic_boxes.numpy(),
                    dynamic_embeddings.numpy(),
                    dynamic_refs.numpy(),
                )
                _assert_close(
                    onnx_dynamic,
                    torch_dynamic,
                    label=f"{height}x{width}:bucket-{bucket}:dynamic-N=3",
                )
                recognizer_cases.append({"bucket": bucket, "proposal_count": 3})

                empty_boxes = boxes[:0]
                empty_embeddings = embeddings[:0]
                empty_refs = refs[:0]
                torch_empty = model.recognizer.forward_bucket(
                    glyph, empty_boxes, empty_embeddings, empty_refs, bucket
                )
                onnx_empty = runtime.run_recognizer(
                    bucket,
                    runtime_outputs["core"]["glyph_features"],
                    empty_boxes.numpy(),
                    empty_embeddings.numpy(),
                    empty_refs.numpy(),
                )
                _assert_close(
                    onnx_empty,
                    torch_empty,
                    label=f"{height}x{width}:bucket-{bucket}:dynamic-N=0",
                )
                recognizer_cases.append({"bucket": bucket, "proposal_count": 0})

    report = {
        "status": "passed",
        "source_model_sha256": source_checksums["model_sha256"],
        "seed": seed,
        "image_sizes_hw": [list(size) for size in image_sizes],
        "core_images_checked": images_checked,
        "decoded_document_status": "passed",
        "decoded_document_cases": decoded_document_cases,
        "recognizer_cases": recognizer_cases,
        "atol": PARITY_ATOL,
        "rtol": PARITY_RTOL,
        "empty_recognizer_inputs_checked": True,
        "fusion_parity": fusion_parity,
    }
    manifest["parity"] = report
    _write_json(manifest_path, manifest)
    return report


def export_artifact(
    artifact_dir: str | Path,
    output_dir: str | Path,
    *,
    opset: int = ONNX_OPSET,
    image_sizes: Sequence[tuple[int, int]] = DEFAULT_PARITY_SIZES,
) -> dict[str, Any]:
    """Export a PyTorch artifact, verify CPU parity, and return its manifest."""
    export_graphs(artifact_dir, output_dir, opset=opset)
    try:
        verify_parity(artifact_dir, output_dir, image_sizes=image_sizes)
    except Exception:
        # A graph-only manifest is useful for diagnostics but must not look like
        # a completed deployment bundle after an acceptance check fails.
        (Path(output_dir) / "manifest.json").unlink(missing_ok=True)
        raise
    return json.loads((Path(output_dir) / "manifest.json").read_text(encoding="utf-8"))


class PrismONNXRuntime:
    """CPU inference stages for the exported model; this does not alter PrismOCR API."""

    def __init__(
        self, export_dir: str | Path, *, _allow_unverified: bool = False
    ) -> None:
        ort = _optional_module("onnxruntime")
        root = Path(export_dir)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise ONNXExportError(f"ONNX manifest is missing: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ONNXExportError(f"cannot read ONNX manifest: {exc}") from exc
        if (manifest.get("format") != FORMAT_NAME
                or type(manifest.get("format_version")) is not int
                or manifest["format_version"] not in (1, FORMAT_VERSION)):
            raise ONNXExportError("unsupported PrismOCR ONNX manifest format or version")
        contract_config = manifest.get("model", {}).get("config", {})
        if manifest["format_version"] == 2 and "model_contract_version" not in contract_config:
            raise ONNXExportError("format 2 requires explicit model_contract_version")
        try:
            self.model_contract_version = validate_model_contract_version(contract_config.get("model_contract_version", 1))
        except ValueError as exc:
            raise ONNXExportError(str(exc)) from exc
        if manifest["format_version"] == 1 and self.model_contract_version != 1:
            raise ONNXExportError("format 1 supports only legacy model contract 1")
        parity = manifest.get("parity")
        if not _allow_unverified and (
            not isinstance(parity, dict)
            or parity.get("status") != "passed"
            or not isinstance(parity.get("fusion_parity"), dict)
            or parity["fusion_parity"].get("status") != "passed"
            or not parity.get("image_sizes_hw")
        ):
            raise ONNXExportError("ONNX artifact has no completed parity verification")
        if manifest.get("opset") != ONNX_OPSET:
            raise ONNXExportError(
                f"unsupported exported opset {manifest.get('opset')!r}; expected {ONNX_OPSET}"
            )
        if ort.__version__ != ONNX_RUNTIME_VERSION:
            raise ONNXExportError(
                f"unsupported ONNX Runtime {ort.__version__!r}; "
                f"this artifact supports {ONNX_RUNTIME_VERSION!r}"
            )
        expected_matrix = {
            "opset": ONNX_OPSET,
            "onnx": ONNX_PACKAGE_VERSION,
            "onnxruntime": ONNX_RUNTIME_VERSION,
            "execution_provider": "CPUExecutionProvider",
        }
        if manifest.get("runtime_matrix") != expected_matrix:
            raise ONNXExportError("ONNX artifact runtime matrix is missing or unsupported")
        runtime = manifest.get("runtime", {})
        if runtime.get("name") != "onnxruntime" or runtime.get("version") != ort.__version__:
            raise ONNXExportError(
                f"ONNX artifact requires ONNX Runtime {runtime.get('version')!r}; "
                f"installed version is {ort.__version__!r}"
            )
        if runtime.get("execution_provider") != "CPUExecutionProvider":
            raise ONNXExportError("only the CPUExecutionProvider is supported")
        if "CPUExecutionProvider" not in ort.get_available_providers():
            raise ONNXExportError("installed ONNX Runtime has no CPUExecutionProvider")

        tensor_inventory = manifest.get("tensor_inventory")
        if not isinstance(tensor_inventory, dict):
            raise ONNXExportError("ONNX tensor inventory is missing or invalid")
        fp32_bytes = tensor_inventory.get("fp32_unique_bytes")
        max_fp32_bytes = tensor_inventory.get("max_fp32_bytes")
        external_data_file = tensor_inventory.get("external_data_file")
        weights_path = root / ONNX_WEIGHTS_FILENAME
        if (
            type(fp32_bytes) is not int
            or type(max_fp32_bytes) is not int
            or max_fp32_bytes != MAX_FP32_TENSOR_BYTES
            or fp32_bytes < 0
            or fp32_bytes > max_fp32_bytes
            or external_data_file != ONNX_WEIGHTS_FILENAME
            or not weights_path.is_file()
            or weights_path.stat().st_size != tensor_inventory.get("external_data_file_bytes")
            or _sha256_file(weights_path) != tensor_inventory.get("external_data_file_sha256")
        ):
            raise ONNXExportError("ONNX external tensor data is missing, over limit, or has a checksum mismatch")

        model_info = manifest.get("model", {})
        try:
            self.buckets = validate_strip_buckets(model_info.get("buckets", ()))
        except (TypeError, ValueError) as exc:
            raise ONNXExportError(f"ONNX manifest strip buckets are invalid: {exc}") from exc
        graphs = manifest.get("graphs", {})
        core_record = graphs.get("core", {})
        self._verify_graph_record(root, core_record, CORE_OUTPUT_NAMES, ("images",))
        recognizer_records = graphs.get("recognizers", {})
        if set(recognizer_records) != {str(bucket) for bucket in self.buckets}:
            raise ONNXExportError("manifest recognizer graphs do not match configured strip buckets")
        for bucket in self.buckets:
            self._verify_graph_record(
                root,
                recognizer_records[str(bucket)],
                RECOGNIZER_OUTPUT_NAMES,
                RECOGNIZER_INPUT_NAMES,
            )
        calibration_info = manifest.get("calibration", {})
        if not isinstance(calibration_info, dict) or not isinstance(calibration_info.get("settings"), dict):
            raise ONNXExportError("ONNX calibration manifest entry is invalid")
        calibration_file = calibration_info.get("file")
        if calibration_file is None:
            if calibration_info.get("sha256") is not None or (root / "calibration-v4.json").exists():
                raise ONNXExportError("ONNX calibration file and manifest identity disagree")
            self.calibration = _validated_calibration(dict(calibration_info["settings"]))
        else:
            calibration_path = root / str(calibration_file)
            if (
                calibration_file != "calibration-v4.json"
                or not calibration_path.is_file()
                or _sha256_file(calibration_path) != calibration_info.get("sha256")
            ):
                raise ONNXExportError("ONNX calibration file is missing or its SHA-256 does not match the manifest")
            try:
                self.calibration = _validated_calibration(
                    json.loads(calibration_path.read_text(encoding="utf-8"))
                )
            except (json.JSONDecodeError, OSError) as exc:
                raise ONNXExportError(f"invalid ONNX calibration artifact: {exc}") from exc
            if self.calibration != calibration_info["settings"]:
                raise ONNXExportError("ONNX calibration settings do not match their manifest copy")
        calibration_model_hash = self.calibration.get("model_sha256")
        if calibration_model_hash is not None and calibration_model_hash != manifest.get("source", {}).get("model_sha256"):
            raise ONNXExportError("ONNX calibration is not bound to the exported source model")
        vocabulary_info = manifest.get("vocabulary", {})
        vocabulary_path = root / str(vocabulary_info.get("file", ""))
        if not vocabulary_path.is_file() or _sha256_file(vocabulary_path) != vocabulary_info.get("sha256"):
            raise ONNXExportError("exported vocabulary is missing or its SHA-256 does not match the manifest")
        try:
            self.vocabulary = Vocabulary.load(vocabulary_path)
        except (ValueError, KeyError, OSError) as exc:
            raise ONNXExportError(f"invalid exported vocabulary: {exc}") from exc
        if len(self.vocabulary) != vocabulary_info.get("size"):
            raise ONNXExportError("manifest vocabulary size does not match its contents")

        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self.core_session = ort.InferenceSession(
                str(root / core_record["file"]),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self.recognizer_sessions = {
                bucket: ort.InferenceSession(
                    str(root / recognizer_records[str(bucket)]["file"]),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
                for bucket in self.buckets
            }
        except Exception as exc:
            raise ONNXExportError(f"ONNX Runtime could not load the CPU graphs: {exc}") from exc
        self.manifest = manifest

    @staticmethod
    def _verify_graph_record(
        root: Path,
        record: dict[str, Any],
        outputs: Sequence[str],
        inputs: Sequence[str],
    ) -> None:
        if record.get("inputs") != list(inputs) or record.get("outputs") != list(outputs):
            raise ONNXExportError("ONNX graph input/output contract does not match its manifest")
        path = root / str(record.get("file", ""))
        if not path.is_file() or _sha256_file(path) != record.get("sha256"):
            raise ONNXExportError(f"ONNX graph is missing or has a checksum mismatch: {path}")

    @staticmethod
    def _input_array(images: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(images, torch.Tensor):
            if images.device.type != "cpu":
                raise ONNXExportError("ONNX Runtime CPU input tensors must be on CPU")
            images = images.detach().numpy()
        array = np.asarray(images)
        if array.dtype != np.float32:
            raise ONNXExportError("ONNX input must have dtype float32")
        if array.ndim != 4 or array.shape[0] != 1 or array.shape[1] != 3:
            raise ONNXExportError("ONNX input must have shape [1,3,H,W]; batch and channels are fixed")
        height, width = array.shape[-2:]
        if height <= 0 or width <= 0 or height % 16 or width % 16:
            raise ONNXExportError("ONNX input height and width must be positive multiples of 16")
        if not np.isfinite(array).all() or (array.size and (array.min() < 0.0 or array.max() > 1.0)):
            raise ONNXExportError("ONNX RGB input values must be finite and in [0,1]")
        return np.ascontiguousarray(array)

    def run_recognizer(
        self,
        bucket: int,
        glyph_features: np.ndarray,
        boxes: np.ndarray,
        node_embeddings: np.ndarray,
        refs: np.ndarray,
    ) -> np.ndarray:
        if bucket not in self.buckets:
            raise ONNXExportError(
                f"unsupported OCR sequence bucket {bucket!r}; supported buckets are {self.buckets}"
            )
        if glyph_features.dtype != np.float32 or boxes.dtype != np.float32 or node_embeddings.dtype != np.float32:
            raise ONNXExportError("recognizer feature, box, and embedding inputs must be float32")
        if refs.dtype != np.int64:
            raise ONNXExportError("recognizer refs must have dtype int64")
        if glyph_features.ndim != 4 or glyph_features.shape[0] != 1:
            raise ONNXExportError("glyph features must have shape [1,D,H/4,W/4]")
        count = len(boxes)
        if boxes.shape != (count, 6) or node_embeddings.shape != (count, glyph_features.shape[1]):
            raise ONNXExportError("recognizer boxes and embeddings have incompatible proposal dimensions")
        if refs.shape != (count, 2):
            raise ONNXExportError("recognizer refs must have shape [N,2]")
        if count == 0:
            # ORT CPU GridSample does not safely handle a zero proposal batch;
            # the recognizer contract defines this case as empty zero logits.
            return np.zeros((0, bucket, len(self.vocabulary)), dtype=np.float32)
        session = self.recognizer_sessions[bucket]
        try:
            (logits,) = session.run(
                list(RECOGNIZER_OUTPUT_NAMES),
                {
                    "glyph_features": np.ascontiguousarray(glyph_features),
                    "boxes": np.ascontiguousarray(boxes),
                    "node_embeddings": np.ascontiguousarray(node_embeddings),
                    "refs": np.ascontiguousarray(refs),
                },
            )
        except Exception as exc:
            raise ONNXExportError(f"ONNX Runtime recognizer bucket {bucket} failed: {exc}") from exc
        return logits

    def run_tensors(self, images: torch.Tensor | np.ndarray) -> dict[str, Any]:
        """Run core and sparse recognizers; return tensors before Python decoding."""
        array = self._input_array(images)
        try:
            values = self.core_session.run(
                list(CORE_OUTPUT_NAMES), {"images": array}
            )
        except Exception as exc:
            raise ONNXExportError(f"ONNX Runtime core graph failed: {exc}") from exc
        core = dict(zip(CORE_OUTPUT_NAMES, values))
        height, width = array.shape[-2:]
        active_mask = core["text_mask"][0]
        if (
            self.calibration.get("scores_calibrated") is True
            and "text_threshold" in self.calibration
        ):
            active_mask = active_mask & (
                core["text_scores"][0] >= float(self.calibration["text_threshold"])
            )
        active = np.flatnonzero(active_mask)
        grouped: dict[int, list[int]] = {bucket: [] for bucket in self.buckets}
        unsupported: list[tuple[int, int, str]] = []
        boxes = torch.from_numpy(core["text_boxes"][0])
        for proposal_index in active.tolist():
            bucket = choose_inference_bucket(
                boxes[proposal_index], (width, height), self.buckets,
                model_contract_version=self.model_contract_version
            )
            if bucket is None:
                unsupported.append((
                    0,
                    proposal_index,
                    f"geometry_exceeds_{self.buckets[-1]}_frame_bucket",
                ))
            else:
                grouped[bucket].append(proposal_index)

        ocr_groups: dict[int, dict[str, np.ndarray]] = {}
        for bucket, proposal_indices in grouped.items():
            if not proposal_indices:
                continue
            refs = np.asarray([(0, index) for index in proposal_indices], dtype=np.int64)
            selected = np.asarray(proposal_indices, dtype=np.int64)
            logits = self.run_recognizer(
                bucket,
                core["glyph_features"],
                core["text_boxes"][0, selected],
                core["text_embeddings"][0, selected],
                refs,
            )
            ocr_groups[bucket] = {"refs": refs, "logits": logits}
        return {
            "core": core,
            "ocr_groups": ocr_groups,
            "ocr_unsupported": tuple(unsupported),
            "input_sha256": _sha256(array.tobytes(order="C")),
            "input_shape": list(array.shape),
        }

    def predict(self, images: torch.Tensor | np.ndarray) -> dict[str, Any]:
        """Return decoded predictions with the public inference error contract."""
        try:
            array = self._input_array(images)
        except Exception as error:
            raise InferenceError("preprocessing", error, code="INVALID_IMAGE") from None
        try:
            result = self.run_tensors(array)
            return decode_onnx_outputs(result, self.manifest, self.vocabulary)
        except InferenceError:
            raise
        except Exception as error:
            raise InferenceError("onnx_inference", error) from None


def _box_polygon(box: Sequence[float], image_size: tuple[int, int]) -> list[list[float]] | None:
    from prismocr.geometry import polygon_area, polygon_intersection, rotated_box_to_polygon

    width, height = image_size
    points = rotated_box_to_polygon(box, image_size)
    clipped = polygon_intersection(
        points,
        ((0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height))),
    )
    if len(clipped) < 3 or polygon_area(clipped) <= 1e-6:
        return None
    return [[round(float(x), 4), round(float(y), 4)] for x, y in clipped]


def _json_table_result(result: Any) -> dict[str, Any]:
    return {
        "tables": [
            {
                "id": table.id,
                "status": table.status,
                "rows": table.rows,
                "cols": table.cols,
                "html": table.html,
                "markdown": table.markdown,
                "conflicts": list(table.conflicts),
                "cells": [
                    {
                        "id": cell.id,
                        "table_id": cell.table_id,
                        "row": cell.row,
                        "col": cell.col,
                        "rowspan": cell.rowspan,
                        "colspan": cell.colspan,
                        "is_header": cell.is_header,
                        "text": cell.text,
                        "line_ids": list(cell.line_ids),
                        "status": cell.status,
                        "polygon_px": None if cell.polygon is None else [list(point) for point in cell.polygon],
                    }
                    for cell in table.cells
                ],
            }
            for table in result.tables
        ],
        "unassigned_cells": [
            {
                "id": cell.id,
                "reason": cell.reason,
                "polygon_px": None if cell.polygon is None else [list(point) for point in cell.polygon],
            }
            for cell in result.unassigned_cells
        ],
        "unassigned_line_ids": list(result.unassigned_line_ids),
    }


def decode_onnx_outputs(
    output: dict[str, Any], manifest: dict[str, Any], vocabulary: Vocabulary
) -> dict[str, Any]:
    """Decode CTC, typed relations, grids, and v4 annotation shape deterministically."""
    from prismocr.data.schema import AnnotationValidationError, validate_annotation
    from prismocr.graph_decode import GraphNode, decode_graph
    from prismocr.table_decode import reconstruct_tables

    core = output["core"]
    image_height = int(core["glyph_features"].shape[-2] * 4)
    image_width = int(core["glyph_features"].shape[-1] * 4)
    image_size = (image_width, image_height)
    calibration = manifest.get("calibration", {}).get("settings", {})
    text_ids = np.flatnonzero(core["text_mask"][0]).tolist()
    region_ids = np.flatnonzero(core["region_mask"][0]).tolist()
    decoded_by_proposal: dict[int, Any] = {}
    for group in output["ocr_groups"].values():
        for ref, logits in zip(group["refs"], group["logits"]):
            decoded_by_proposal[int(ref[1])] = vocabulary.decode_ctc(logits.argmax(axis=-1))

    lines: list[dict[str, Any]] = []
    line_node_data: list[tuple[str, int, GraphNode]] = []
    invalid_geometry: list[dict[str, Any]] = []
    text_role_names = tuple(manifest["model"]["text_roles"])
    region_kind_names = tuple(manifest["model"]["region_kinds"])
    tile = Tile(0, 0, 0, image_width, image_height)
    text_capacity = int(manifest["model"]["config"]["text_proposals"])
    text_candidates: list[dict[str, Any]] = []
    for proposal_index in text_ids:
        polygon = _box_polygon(core["text_boxes"][0, proposal_index], image_size)
        if polygon is None:
            invalid_geometry.append({"stream": "text", "proposal_index": proposal_index})
            continue
        role_id = int(core["text_role_logits"][0, proposal_index].argmax())
        decoded_text = decoded_by_proposal.get(proposal_index)
        score = float(core["text_scores"][0, proposal_index])
        confidence = _calibrated_score(score, calibration, "text")
        text_candidates.append({
            "proposal_index": proposal_index,
            "polygon": polygon,
            "text": "" if decoded_text is None else decoded_text.text,
            "role": text_role_names[role_id],
            "detector_score": confidence if calibration.get("scores_calibrated") is True else None,
            "_tile_score": score,
            "oov_positions": [] if decoded_text is None else list(decoded_text.oov_positions),
        })
    text_stitch = stitch_tile_results((TileResult(
        tile,
        text_candidates,
        candidate_count=int(core["text_candidate_count"][0]),
        capacity=text_capacity,
        overflow=bool(core["text_capacity_overflow"][0]),
    ),), id_prefix="l")
    stitch_warnings = list(text_stitch.warnings)
    text_operating_threshold = (
        float(calibration["text_threshold"])
        if calibration.get("scores_calibrated") is True and "text_threshold" in calibration
        else None
    )
    for detection in text_stitch.detections:
        if (
            text_operating_threshold is not None
            and float(detection["_tile_score"]) < text_operating_threshold
        ):
            continue
        proposal_index = int(detection["proposal_index"])
        polygon = detection["polygon"]
        item_id = str(detection["id"])
        role = str(detection["role"])
        line = {
            "id": item_id,
            "polygon_px": polygon,
            "text": str(detection["text"]),
            "role": role,
            "detector_score": detection.get("detector_score"),
            "oov_positions": list(detection.get("oov_positions", [])),
        }
        lines.append(line)
        line_node_data.append(
            (item_id, proposal_index, GraphNode(item_id, "line", polygon=polygon, role=role))
        )

    region_detections_by_kind: dict[str, list[dict[str, Any]]] = {}
    for proposal_index in region_ids:
        polygon = _box_polygon(core["region_boxes"][0, proposal_index], image_size)
        if polygon is None:
            invalid_geometry.append({"stream": "region", "proposal_index": proposal_index})
            continue
        kind_id = int(core["region_kinds"][0, proposal_index])
        kind = region_kind_names[kind_id]
        score = float(core["region_scores"][0, proposal_index])
        confidence = _calibrated_score(score, calibration, "region")
        region_detections_by_kind.setdefault(kind, []).append({
            "proposal_index": proposal_index,
            "polygon": polygon,
            "kind": kind,
            "detector_score": confidence if calibration.get("scores_calibrated") is True else None,
            "_tile_score": score,
        })
    region_candidates: list[tuple[int, str, list[list[float]]]] = []
    region_operating_threshold = (
        float(calibration["region_threshold"])
        if calibration.get("scores_calibrated") is True and "region_threshold" in calibration
        else None
    )
    for kind in sorted(region_detections_by_kind):
        stitched = stitch_tile_results((TileResult(
            tile, region_detections_by_kind[kind]
        ),), id_prefix="r")
        stitch_warnings.extend(stitched.warnings)
        region_candidates.extend(
            (int(item["proposal_index"]), kind, item["polygon"])
            for item in stitched.detections
            if region_operating_threshold is None
            or float(item["_tile_score"]) >= region_operating_threshold
        )
    region_candidates.sort(key=lambda item: (
        item[1],
        sum(point[1] for point in item[2]) / len(item[2]),
        sum(point[0] for point in item[2]) / len(item[2]),
        min(point[1] for point in item[2]),
        min(point[0] for point in item[2]),
        item[0],
    ))

    regions: list[dict[str, Any]] = []
    cells_for_table: list[dict[str, Any]] = []
    cell_proposal_indices: dict[str, int] = {}
    region_node_data: list[tuple[str, int, GraphNode]] = []
    region_index = cell_index = 0
    for proposal_index, kind, polygon in region_candidates:
        if kind == "CELL":
            cell_index += 1
            item_id = f"c:{cell_index}"
        else:
            region_index += 1
            item_id = f"r:{region_index}"
        record = {
            "id": item_id,
            "kind": kind,
            "polygon_px": polygon,
            "detector_score": (
                _calibrated_score(
                    float(core["region_scores"][0, proposal_index]), calibration, "region"
                )
                if calibration.get("scores_calibrated") is True else None
            ),
        }
        if kind == "CELL":
            if calibration.get("grid_scores_calibrated") is True:
                grid_threshold = float(calibration.get("grid_threshold", 0.0))
                for field, logits_field, offset in (
                    ("row", "region_row_logits", 0),
                    ("col", "region_col_logits", 0),
                    ("rowspan", "region_rowspan_logits", 1),
                    ("colspan", "region_colspan_logits", 1),
                ):
                    logits = core[logits_field][0, proposal_index].astype(np.float32)
                    logits = logits - logits.max()
                    probabilities = np.exp(logits) / np.exp(logits).sum()
                    index = int(probabilities.argmax())
                    calibrated_confidence = _calibrated_grid_score(
                        float(probabilities[index]), calibration, field
                    )
                    if calibrated_confidence >= grid_threshold:
                        record[field] = index + offset
                header_logit = float(core["region_header_logits"][0, proposal_index])
                header_probability = 1.0 / (1.0 + np.exp(-np.clip(header_logit, -80.0, 80.0)))
                header_confidence = _calibrated_grid_score(
                    max(header_probability, 1.0 - header_probability), calibration, "header"
                )
                if header_confidence >= grid_threshold:
                    record["is_header"] = bool(
                        header_probability >= float(calibration.get("header_threshold", 0.5))
                    )
            cells_for_table.append(record)
            cell_proposal_indices[item_id] = proposal_index
        else:
            regions.append(record)
            region_node_data.append(
                (item_id, len(core["text_boxes"][0]) + proposal_index,
                 GraphNode(item_id, "region", polygon=polygon, kind=kind))
            )

    tables_for_decode = [record for record in regions if record["kind"] == "TABLE"]
    line_scopes = PrismOCR._reading_order_scopes(lines, tables_for_decode, image_width)
    for line in lines:
        line["scope_id"] = line_scopes.get(line["id"], "document")
    line_node_data = [
        (
            line["id"],
            proposal_index,
            GraphNode(
                line["id"], "line", polygon=line["polygon_px"],
                role=line["role"], scope_id=line["scope_id"],
            ),
        )
        for line, (_, proposal_index, _) in zip(lines, line_node_data)
    ]
    lines_in_tables = [
        line for line in lines if PrismOCR._line_in_table(line, tables_for_decode)
    ]
    table_result = reconstruct_tables(tables_for_decode, cells_for_table, lines_in_tables)
    initial_cells = {
        cell.id: cell
        for table in table_result.tables
        for cell in table.cells
    }
    cell_node_data: list[tuple[str, int, GraphNode]] = []
    for record in cells_for_table:
        cell = initial_cells.get(record["id"])
        proposal_index = cell_proposal_indices[record["id"]]
        cell_node_data.append((
            record["id"],
            len(core["text_boxes"][0]) + proposal_index,
            GraphNode(
                record["id"],
                "cell",
                polygon=record["polygon_px"],
                kind="CELL",
                table_id=None if cell is None else cell.table_id,
                row=None if cell is None else cell.row,
                col=None if cell is None else cell.col,
            ),
        ))

    node_data = line_node_data + region_node_data + cell_node_data
    graph_nodes = [entry[2] for entry in node_data]
    nodes_by_id = {node.id: node for node in graph_nodes}
    relation_names = tuple(manifest["model"]["relation_names"][:4])
    relation_scores: dict[str, dict[str, dict[str, float]]] = {}
    null_scores: dict[str, dict[str, float]] = {}
    graph_values = core["graph_logits"][0].astype(np.float64)
    relation_calibrated = calibration.get("relation_scores_calibrated") is True
    allowed_sources = {
        "NEXT": {node.id for node in graph_nodes if node.node_type == "line"},
        "PARENT": {node.id for node in graph_nodes},
        "KV_VALUE": {
            node.id for node in graph_nodes
            if node.node_type == "line" and node.role == "KEY"
        },
        "RIGHT_CELL": {node.id for node in graph_nodes if node.node_type == "cell"},
    }
    if relation_calibrated:
        for relation_index, relation in enumerate(relation_names):
            relation_scores[relation] = {}
            null_scores[relation] = {}
            for source_id in sorted(allowed_sources[relation]):
                source = nodes_by_id[source_id]
                source_index = next(index for node_id, index, _ in node_data if node_id == source_id)
                valid_targets = [
                    (target_id, target_index)
                    for target_id, target_index, target_node in node_data
                    if relation_candidate_error(
                        relation,
                        source,
                        target_node,
                        nodes_by_id,
                        allow_right_cell_column_repair=relation == "RIGHT_CELL",
                    ) is None
                ]
                candidate_logits = [
                    graph_values[relation_index, source_index, target_index]
                    for _, target_index in valid_targets
                ]
                candidate_logits.append(graph_values[relation_index, source_index, -1])
                values = np.asarray(candidate_logits, dtype=np.float64)
                values -= values.max()
                probabilities = np.exp(values)
                probabilities /= probabilities.sum()
                target_scores = {node.id: 0.0 for node in graph_nodes}
                for (target_id, _), probability in zip(valid_targets, probabilities[:-1]):
                    target_scores[target_id] = _calibrated_relation_score(
                        float(probability), calibration, relation
                    )
                relation_scores[relation][source_id] = target_scores
                null_scores[relation][source_id] = _calibrated_relation_score(
                    float(probabilities[-1]), calibration, relation
                )

    decoded_graph = decode_graph(
        graph_nodes,
        relation_scores,
        null_scores,
        suppressed_limit=_RELATION_DIAGNOSTIC_LIMIT,
    )
    reading_order = list(decoded_graph.reading_order)
    reading_order_by_id = {line_id: rank for rank, line_id in enumerate(reading_order)}
    parent_by_child = {
        prediction.source_id: prediction.target_id
        for prediction in decoded_graph.predictions
        if prediction.relation == "PARENT" and prediction.target_id is not None
    }
    for line in lines:
        line["reading_order"] = reading_order_by_id[line["id"]]
        line["parent_id"] = parent_by_child.get(line["id"])
    for region in regions:
        region["parent_id"] = parent_by_child.get(region["id"])

    # Rebuild table text and cells after logical reading order and learned affinities.
    table_result = reconstruct_tables(
        tables_for_decode,
        cells_for_table,
        lines_in_tables,
        right_cell_scores=relation_scores.get("RIGHT_CELL", {}),
        right_cell_null_scores=null_scores.get("RIGHT_CELL", {}),
        use_right_cell_constraints=relation_calibrated,
    )
    affinity_cells = {
        cell.id: cell
        for table in table_result.tables
        for cell in table.cells
    }
    graph_nodes = [
        GraphNode(
            id=node.id,
            node_type=node.node_type,
            polygon=node.polygon,
            role=node.role,
            kind=node.kind,
            scope_id=node.scope_id,
            table_id=(affinity_cells[node.id].table_id if node.id in affinity_cells else node.table_id),
            row=(affinity_cells[node.id].row if node.id in affinity_cells else node.row),
            col=(affinity_cells[node.id].col if node.id in affinity_cells else node.col),
        )
        for node in graph_nodes
    ]
    decoded_graph = decode_graph(
        graph_nodes,
        relation_scores,
        null_scores,
        suppressed_limit=_RELATION_DIAGNOSTIC_LIMIT,
    )
    reading_order = list(decoded_graph.reading_order)
    reading_order_by_id = {line_id: rank for rank, line_id in enumerate(reading_order)}
    parent_by_child = {
        prediction.source_id: prediction.target_id
        for prediction in decoded_graph.predictions
        if prediction.relation == "PARENT" and prediction.target_id is not None
    }
    for line in lines:
        line["reading_order"] = reading_order_by_id[line["id"]]
        line["parent_id"] = parent_by_child.get(line["id"])
    for region in regions:
        region["parent_id"] = parent_by_child.get(region["id"])

    decoded_cells = {
        cell.id: (cell, table.status)
        for table in table_result.tables
        for cell in table.cells
    }
    unassigned_cells = {cell.id: cell for cell in table_result.unassigned_cells}
    cell_records: list[dict[str, Any]] = []
    cell_proposals: list[dict[str, Any]] = []
    for record in cells_for_table:
        cell_state = decoded_cells.get(record["id"])
        if cell_state is not None:
            cell, table_status = cell_state
            cell_record: dict[str, Any] = {
                "id": cell.id,
                "table_id": cell.table_id,
                "row": cell.row,
                "col": cell.col,
                "rowspan": cell.rowspan,
                "colspan": cell.colspan,
                "polygon_px": (
                    [list(point) for point in cell.polygon]
                    if cell.polygon is not None else record["polygon_px"]
                ),
                "text": cell.text,
                "detector_score": record.get("detector_score"),
                "line_ids": list(cell.line_ids),
            }
            if cell.is_header is not None:
                cell_record["is_header"] = bool(cell.is_header)
            cell_records.append(cell_record)
            cell_proposals.append({**cell_record, "status": cell.status, "table_status": table_status})
            continue
        unassigned = unassigned_cells.get(record["id"])
        unassigned_reason = (
            unassigned.reason if unassigned is not None else "not_returned_by_table_decoder"
        )
        cell_record = {
            "id": record["id"],
            "table_id": None,
            "row": None,
            "col": None,
            "rowspan": None,
            "colspan": None,
            "polygon_px": record["polygon_px"],
            "text": "",
            "line_ids": [],
            "detector_score": record.get("detector_score"),
        }
        cell_records.append(cell_record)
        cell_proposals.append({
            **cell_record,
            "status": "unassigned",
            "unassigned_reason": unassigned_reason,
        })
    relations = [
        {
            "type": prediction.relation,
            "source": prediction.source_id,
            "target": prediction.target_id,
            **({"annotation_status": "explicit_null"} if prediction.target_id is None else {}),
        }
        for prediction in decoded_graph.predictions
    ]
    line_by_id = {line["id"]: line for line in lines}
    region_by_id = {region["id"]: region for region in regions}
    kv_pairs: list[dict[str, Any]] = []
    for prediction in decoded_graph.predictions:
        if prediction.relation != "KV_VALUE" or prediction.target_id is None:
            continue
        key_line = line_by_id.get(prediction.source_id)
        if key_line is None:
            continue
        target_text = ""
        target_line = line_by_id.get(prediction.target_id)
        if target_line is not None:
            target_text = str(target_line.get("text", ""))
        else:
            target_region = region_by_id.get(prediction.target_id)
            if target_region is not None and target_region.get("kind") == "FORM_FIELD":
                contained = [
                    line for line in lines
                    if line.get("parent_id") == prediction.target_id
                ]
                contained.sort(key=lambda line: line.get("reading_order", 0))
                target_text = "\n".join(str(line.get("text", "")) for line in contained)
        kv_pairs.append({
            "key": str(key_line.get("text", "")),
            "value": target_text,
            "source_id": prediction.source_id,
            "target_id": prediction.target_id,
            "score": float(prediction.score),
            "page_id": "onnx-prediction:page-0",
            "reading_order": key_line.get("reading_order"),
        })
    annotation = {
        "schema_version": "4.0",
        "document_id": "onnx-prediction",
        "page_id": "onnx-prediction:page-0",
        "image": {
            "width": image_width,
            "height": image_height,
            "sha256": str(output.get("input_sha256", "0" * 64)),
        },
        "provenance": {
            "source_class": "derived",
            "license": "not_provided",
            "permission_reference": "not_provided",
            "reviewer": "not_applicable_prediction_output",
            "input_hash_kind": "normalized_nchw_float32_tensor_sha256",
        },
        "lines": lines,
        "regions": regions,
        "cells": cell_records,
        "relations": relations,
        "reading_order": reading_order,
    }
    schema_issues: tuple[str, ...] = ()
    try:
        validate_annotation(annotation)
    except AnnotationValidationError as exc:
        schema_issues = exc.issues

    unsupported_records = [list(item) for item in output["ocr_unsupported"]]
    warnings: list[dict[str, Any]] = list(stitch_warnings)
    if decoded_graph.suppressed_count:
        warnings.append({
            "code": "RELATIONS_SUPPRESSED",
            "count": decoded_graph.suppressed_count,
            "reason_counts": dict(decoded_graph.suppressed_reason_counts),
        })
    proposal_to_line = {proposal_index: line_id for line_id, proposal_index, _ in line_node_data}
    for batch_index, proposal_index, reason in output["ocr_unsupported"]:
        if batch_index == 0:
            line_id = proposal_to_line.get(int(proposal_index))
            if line_id is None:
                continue
            warnings.append({
                "code": "UNSUPPORTED_LINE_LENGTH",
                "line_id": line_id,
                "reason": str(reason),
            })
    for line in lines:
        if line["oov_positions"]:
            warnings.append({
                "code": "UNSUPPORTED_CHARSET",
                "line_id": line["id"],
                "oov_count": len(line["oov_positions"]),
                "oov_positions": list(line["oov_positions"]),
            })
    if calibration.get("scores_calibrated") is not True:
        warnings.append({"code": "SCORES_UNCALIBRATED"})
    if not relation_calibrated and graph_nodes:
        warnings.append({"code": "RELATION_SCORES_UNCALIBRATED"})
        if lines:
            warnings.append({"code": "READING_ORDER_GEOMETRIC_FALLBACK"})
    for table in table_result.tables:
        if table.status == "conflict":
            warnings.append({
                "code": "TABLE_GRID_CONFLICT",
                "table_id": table.id,
                "conflicts": list(table.conflicts),
            })
    for cell in cell_proposals:
        if cell.get("table_id") is None:
            warnings.append({
                "code": "CELL_UNASSIGNED",
                "cell_id": cell["id"],
                "reason": cell.get("unassigned_reason", "table_grid_conflict"),
            })
    if table_result.unassigned_line_ids:
        warnings.append({
            "code": "TABLE_LINES_UNASSIGNED",
            "line_ids": list(table_result.unassigned_line_ids),
        })

    text_candidates = int(core["text_candidate_count"][0])
    model_config = manifest["model"]["config"]
    text_capacity = int(model_config["text_proposals"])
    text_overflow = bool(core["text_capacity_overflow"][0])
    structural_count = len(region_ids)
    structural_capacity = int(model_config["region_queries"])
    structural_overflow = structural_capacity > 0 and structural_count >= structural_capacity
    if text_overflow or structural_overflow:
        warnings.append({
            "code": "CAPACITY_OVERFLOW",
            "streams": [
                *(["text"] if text_overflow else []),
                *(["structure"] if structural_overflow else []),
            ],
            "partial_result": True,
            **({"candidate_count": text_candidates} if text_overflow else {}),
        })
    if structural_overflow:
        warnings.append({
            "code": "STRUCTURE_CAPACITY_OVERFLOW",
            "partial_result": True,
            "saturated_tile_count": 1,
            "candidate_count": structural_count,
            "capacity": structural_capacity,
        })
    source_identity = manifest.get("source", {})
    artifact_identity = {
        key: source_identity.get(key)
        for key in ("model_sha256", "vocabulary_sha256", "calibration_sha256")
    }

    return {
        "annotation": annotation,
        "ocr_unsupported": unsupported_records,
        "kv_pairs": kv_pairs,
        "artifact_identity": artifact_identity,
        "warnings": warnings,
        "cell_proposals": cell_proposals,
        "invalid_geometry": invalid_geometry,
        "graph": {
            "predictions": [
                {
                    "relation": prediction.relation,
                    "source_id": prediction.source_id,
                    "target_id": prediction.target_id,
                    "score": prediction.score,
                    "reason_codes": list(prediction.reason_codes),
                }
                for prediction in decoded_graph.predictions
            ],
            "suppressed": [
                {
                    "relation": item.relation,
                    "source_id": item.source_id,
                    "target_id": item.target_id,
                    "score": item.score,
                    "reason_code": item.reason_code,
                }
                for item in decoded_graph.suppressed
            ],
        },
        "table_decode": _json_table_result(table_result),
        "schema_valid": not schema_issues,
        "schema_issues": list(schema_issues),
        "diagnostics": {
            "text_candidates": text_candidates,
            "text_capacity": text_capacity,
            "text_overflow": text_overflow,
            "active_text_proposals": len(text_ids),
            "text_capacity_overflow": text_overflow,
            "active_region_proposals": len(region_ids),
            "region_capacity": structural_capacity,
            "structural_capacity_overflow": structural_overflow,
            "overflow": text_overflow or structural_overflow,
            "ocr_buckets": {
                str(bucket): len(group["refs"])
                for bucket, group in sorted(output["ocr_groups"].items())
            },
            "relation_suppressed_count": decoded_graph.suppressed_count,
            "relation_suppressed": [
                {
                    "type": item.relation,
                    "source": item.source_id,
                    "target": item.target_id,
                    "score": item.score,
                    "reason_code": item.reason_code,
                }
                for item in decoded_graph.suppressed
            ],
            "relation_diagnostics_truncated": (
                decoded_graph.suppressed_count > len(decoded_graph.suppressed)
            ),
        },
    }


def _parse_size(value: str) -> tuple[int, int]:
    try:
        height_text, width_text = value.lower().split("x", 1)
        height, width = int(height_text), int(width_text)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError("size must be HEIGHTxWIDTH, for example 64x80") from exc
    if height <= 0 or width <= 0 or height % 16 or width % 16:
        raise argparse.ArgumentTypeError("size dimensions must be positive multiples of 16")
    return height, width


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export PrismOCR to ONNX and verify CPU parity")
    parser.add_argument("--artifact", required=True, help="directory containing model.pt and v4 metadata")
    parser.add_argument("--output", required=True, help="destination directory for ONNX graphs and manifest")
    parser.add_argument("--opset", type=int, default=ONNX_OPSET)
    parser.add_argument(
        "--parity-size",
        type=_parse_size,
        action="append",
        dest="parity_sizes",
        help="aligned parity image size HEIGHTxWIDTH; may be repeated (defaults: 64x80, 96x128)",
    )
    args = parser.parse_args(argv)
    try:
        manifest = export_artifact(
            args.artifact,
            args.output,
            opset=args.opset,
            image_sizes=args.parity_sizes or DEFAULT_PARITY_SIZES,
        )
    except (ONNXExportError, OSError, ValueError) as exc:
        print(f"prismocr ONNX export failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": "exported_and_parity_checked",
        "output": str(Path(args.output).resolve()),
        "opset": manifest["opset"],
        "onnxruntime": manifest["runtime"]["version"],
        "parity": manifest["parity"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
