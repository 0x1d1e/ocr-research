"""CPU-first page prediction API and versioned checkpoint loader."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from prismocr.data.orientation import exif_oriented
from prismocr.data.vocabulary import DecodedText, Vocabulary
from prismocr.geometry import polygon_area, polygon_intersection, rotated_box_to_polygon
from prismocr.graph_decode import GraphNode, decode_graph, relation_candidate_error
from prismocr.model.network import REGION_KINDS, RELATION_NAMES, TEXT_ROLES, PrismOCRModel, validated_model_state
from prismocr.table_decode import TableDecodeResult, reconstruct_tables
from prismocr.tiling import Tile, TileResult, plan_tiles, stitch_tile_results

_MAX_TILE_SIZE = (1024, 1024)
_INITIAL_TILE_OVERLAP = 128
_MIN_RECOVERY_TILE = 256
_RELATION_DIAGNOSTIC_LIMIT = 128
_MAX_FP32_TENSOR_BYTES = 10_000_000
_CALIBRATION_FLAGS = (
    "scores_calibrated",
    "relation_scores_calibrated",
    "grid_scores_calibrated",
)
_CALIBRATION_THRESHOLDS = (
    "text_threshold",
    "region_threshold",
    "grid_threshold",
    "header_threshold",
)


def _validated_calibration(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("calibration artifact must be a JSON object")
    calibration = dict(value)
    for key in _CALIBRATION_FLAGS:
        if key in calibration and not isinstance(calibration[key], bool):
            raise ValueError(f"calibration {key} must be a boolean")
    for key in _CALIBRATION_THRESHOLDS:
        if key not in calibration:
            continue
        threshold = calibration[key]
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0 <= threshold <= 1
        ):
            raise ValueError(f"calibration {key} must be finite and in [0,1]")
    calibrated_flags = (
        "scores_calibrated", "relation_scores_calibrated", "grid_scores_calibrated"
    )
    if any(calibration.get(key) is True for key in calibrated_flags):
        if calibration.get("method") != "validation_only_platt_scaling_v1":
            raise ValueError("calibrated scores require validation-only fitting provenance")
        for key in (
            "model_sha256",
            "model_config_sha256",
            "vocabulary_sha256",
            "split_sha256",
            "split_report_sha256",
        ):
            digest = calibration.get(key)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"calibrated scores require a valid {key}")
            try:
                int(digest, 16)
            except ValueError:
                raise ValueError(f"calibrated scores require a valid {key}") from None
        page_count = calibration.get("validation_page_count")
        if isinstance(page_count, bool) or not isinstance(page_count, int) or page_count <= 0:
            raise ValueError("calibrated scores require a positive validation_page_count")
        iou_threshold = calibration.get("iou_threshold")
        if (
            isinstance(iou_threshold, bool)
            or not isinstance(iou_threshold, (int, float))
            or not math.isfinite(iou_threshold)
            or not 0.0 < iou_threshold <= 1.0
        ):
            raise ValueError("calibrated scores require a valid polygon IoU threshold")
        if calibration.get("scores_calibrated") is True:
            transforms = calibration.get("score_calibrators")
            _validate_calibrator_set(transforms, ("text", "region"), "detector")
        if calibration.get("relation_scores_calibrated") is True:
            transforms = calibration.get("relation_score_calibrators")
            _validate_calibrator_set(transforms, ("NEXT", "PARENT", "KV_VALUE", "RIGHT_CELL"), "relation")
        if calibration.get("grid_scores_calibrated") is True:
            transforms = calibration.get(
                "grid_score_calibrators", {"row": calibration.get("grid_calibrator")}
            )
            _validate_calibrator_set(
                transforms, ("row", "col", "rowspan", "colspan", "header"), "grid"
            )
    return calibration


def _validate_calibrator_set(
    transforms: object, names: Sequence[str], category: str
) -> None:
    if not isinstance(transforms, dict):
        raise ValueError(f"calibrated {category} scores require fitted calibrators")
    for name in names:
        transform = transforms.get(name)
        if not isinstance(transform, dict) or transform.get("method") != "platt_v1":
            raise ValueError(f"calibrated {category} {name} scores require a platt_v1 transform")
        slope, intercept = transform.get("slope"), transform.get("intercept")
        if (
            isinstance(slope, bool)
            or not isinstance(slope, (int, float))
            or not math.isfinite(slope)
            or slope <= 0
            or isinstance(intercept, bool)
            or not isinstance(intercept, (int, float))
            or not math.isfinite(intercept)
        ):
            raise ValueError(f"calibrated {category} {name} transform must be finite and monotonic")
        positive, negative = transform.get("positive_count"), transform.get("negative_count")
        if (
            isinstance(positive, bool) or not isinstance(positive, int) or positive <= 0
            or isinstance(negative, bool) or not isinstance(negative, int) or negative <= 0
        ):
            raise ValueError(f"calibrated {category} {name} transform lacks positive/negative validation samples")
        reliability = transform.get("reliability")
        if not isinstance(reliability, dict):
            raise ValueError(f"calibrated {category} {name} transform lacks reliability metrics")
        for metric in ("ece", "brier"):
            value = reliability.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"calibrated {category} {name} reliability {metric} must be in [0,1]")
        if not isinstance(transform.get("precision_recall_curve"), list):
            raise ValueError(f"calibrated {category} {name} transform lacks a precision-recall curve")


def _apply_platt(score: float, transform: Mapping[str, Any] | None) -> float:
    value = float(score)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("confidence score must be finite and in [0,1]")
    if not isinstance(transform, Mapping):
        return value
    raw = min(1.0 - 1e-6, max(1e-6, value))
    logit = math.log(raw / (1.0 - raw))
    calibrated_logit = float(transform["slope"]) * logit + float(transform["intercept"])
    if calibrated_logit >= 0:
        return 1.0 / (1.0 + math.exp(-min(calibrated_logit, 80.0)))
    exp_value = math.exp(max(calibrated_logit, -80.0))
    return exp_value / (1.0 + exp_value)


def _calibrated_score(score: float, calibration: Mapping[str, Any], stream: str) -> float:
    """Map a raw detector probability through its validation-fitted Platt transform."""
    transforms = calibration.get("score_calibrators")
    transform = transforms.get(stream) if isinstance(transforms, dict) else None
    return _apply_platt(score, transform)


def _calibrated_relation_score(
    score: float, calibration: Mapping[str, Any], relation: str
) -> float:
    transforms = calibration.get("relation_score_calibrators")
    transform = transforms.get(relation) if isinstance(transforms, dict) else None
    return _apply_platt(score, transform)


def _calibrated_grid_score(
    score: float, calibration: Mapping[str, Any], field: str
) -> float:
    transforms = calibration.get("grid_score_calibrators")
    transform = transforms.get(field) if isinstance(transforms, dict) else None
    return _apply_platt(score, transform)


@dataclass(frozen=True)
class InferenceOptions:
    page_resolution: int | None = None
    allow_tiling: bool = True
    emit_character_confidence: bool = False
    return_debug: bool = False
    capture_calibration_data: bool = False
    output_coordinate_space: Literal["pixels", "normalized"] = "pixels"

    def __post_init__(self) -> None:
        if self.page_resolution is not None and (
            isinstance(self.page_resolution, bool)
            or not isinstance(self.page_resolution, int)
            or self.page_resolution <= 0
        ):
            raise ValueError("page_resolution must be a positive integer or None")
        for name in (
            "allow_tiling", "emit_character_confidence", "return_debug", "capture_calibration_data"
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.output_coordinate_space not in {"pixels", "normalized"}:
            raise ValueError("output_coordinate_space must be 'pixels' or 'normalized'")


@dataclass(frozen=True)
class DocumentResult:
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.payload


class InferenceError(RuntimeError):
    """A redacted, machine-readable failure at the public prediction boundary."""

    def __init__(
        self, stage: str, cause: BaseException, *, code: str = "INFERENCE_ERROR"
    ) -> None:
        if code not in {"INVALID_IMAGE", "INFERENCE_ERROR"}:
            raise ValueError(f"unsupported inference error code {code!r}")
        self.code = code
        self.stage = stage
        self.cause_type = type(cause).__name__
        self.message = "invalid image" if code == "INVALID_IMAGE" else "inference failed"
        super().__init__(f"{self.message} during {stage} ({self.code})")

    def to_dict(self) -> dict[str, str]:
        """Return only stable, non-sensitive failure metadata."""
        return {
            "code": self.code,
            "stage": self.stage,
            "message": self.message,
        }


@dataclass(frozen=True)
class _TileInference:
    tile: Tile
    text_detections: tuple[dict[str, Any], ...]
    region_detections: tuple[dict[str, Any], ...]
    candidate_count: int
    capacity: int
    overflow: bool
    structural_count: int
    structural_capacity: int
    structural_overflow: bool
    unsupported: tuple[tuple[int, str], ...]


class PrismOCR:
    def __init__(
        self,
        model: PrismOCRModel,
        vocabulary: Vocabulary,
        *,
        device: str | torch.device = "cpu",
        model_version: str = "unversioned-experiment",
        calibration: dict[str, Any] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.fused_for_inference().to(self.device).eval()
        self.vocabulary = vocabulary
        self.model_version = model_version
        self.calibration = _validated_calibration({} if calibration is None else calibration)

    @classmethod
    def load(cls, artifact_dir: str | Path, device: str = "cpu") -> "PrismOCR":
        root = Path(artifact_dir)
        config_path = root / "model-config.json"
        vocab_path = root / "vocab-v4.json"
        weights_path = root / "model.pt"
        manifest_path = root / "manifest.json"
        for path in (config_path, vocab_path, weights_path):
            if not path.is_file():
                raise FileNotFoundError(f"required artifact is missing: {path}")
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
        if not isinstance(config, dict):
            raise ValueError("model configuration must be a JSON object")
        vocabulary = Vocabulary.load(vocab_path)
        if config.get("vocab_size") != len(vocabulary):
            raise ValueError("model configuration and vocabulary size do not match")
        has_manifest = manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if has_manifest else {}
        if not isinstance(manifest, dict):
            raise ValueError("artifact manifest must be a JSON object")
        if has_manifest and manifest.get("schema_version") != "4.0":
            raise ValueError("unsupported artifact manifest schema_version")
        weights_sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        if has_manifest:
            expected_hashes = {
                "model_sha256": weights_sha256,
                "vocabulary_sha256": vocabulary.sha256,
                "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            }
            for field, actual in expected_hashes.items():
                if manifest.get(field) != actual:
                    raise ValueError(f"artifact {field} is missing or does not match manifest")
        calibration_path = root / "calibration-v4.json"
        if calibration_path.is_file():
            calibration_bytes = calibration_path.read_bytes()
            if not has_manifest or manifest.get("calibration_sha256") != hashlib.sha256(calibration_bytes).hexdigest():
                raise ValueError("calibration SHA-256 is missing or does not match manifest")
            calibration = _validated_calibration(json.loads(calibration_bytes))
        else:
            if has_manifest and manifest.get("calibration_sha256") is not None:
                raise ValueError("calibration manifest identity has no matching sidecar")
            calibration = {}
        if calibration:
            if calibration.get("model_sha256") != weights_sha256:
                raise ValueError("calibration artifact is missing a matching model_sha256")
            if calibration.get("model_config_sha256") != manifest.get("model_config_sha256"):
                raise ValueError("calibration artifact does not match model configuration")
            if calibration.get("vocabulary_sha256") != vocabulary.sha256:
                raise ValueError("calibration artifact does not match vocabulary")
        model_kwargs = dict(config)
        model_kwargs["fused_text_head"] = config.get("fused_text_head", False)
        model = PrismOCRModel(vocabulary=vocabulary, **model_kwargs)
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        state_dict = validated_model_state(checkpoint, config)
        model.load_state_dict(state_dict, strict=True)
        tensor_bytes = model.tensor_bytes()
        if tensor_bytes > _MAX_FP32_TENSOR_BYTES:
            raise ValueError(
                f"model tensor inventory is {tensor_bytes} bytes, exceeding the "
                f"{_MAX_FP32_TENSOR_BYTES}-byte FP32 limit"
            )
        declared_tensor_bytes = manifest.get("model_tensor_bytes")
        if declared_tensor_bytes is not None and declared_tensor_bytes != tensor_bytes:
            raise ValueError("model tensor inventory does not match artifact manifest")
        if not has_manifest:
            raise ValueError("artifact manifest is required")
        return cls(
            model.eval(),
            vocabulary,
            device=device,
            model_version=manifest.get("model_version", "unversioned-experiment"),
            calibration=calibration,
        )

    from_artifact = load

    @staticmethod
    def _tensor_image(image: Any) -> tuple[torch.Tensor, int, int]:
        if isinstance(image, Image.Image):
            array = np.asarray(exif_oriented(image).convert("RGB"))
            tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
        elif isinstance(image, np.ndarray):
            if image.ndim != 3:
                raise ValueError("image array must have three dimensions")
            if image.shape[-1] in (3, 4):
                array = image[..., :3]
                tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)
            elif image.shape[0] == 3:
                tensor = torch.from_numpy(np.ascontiguousarray(image))
            else:
                raise ValueError("image array must have RGB channels")
        elif isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim != 3:
                raise ValueError("image tensor must have three dimensions")
            if tensor.shape[0] != 3 and tensor.shape[-1] in (3, 4):
                tensor = tensor[..., :3].permute(2, 0, 1)
            if tensor.shape[0] != 3:
                raise ValueError("image tensor must have RGB channels")
        else:
            raise TypeError("image must be a PIL image, numpy array or torch tensor")
        if tensor.dtype == torch.uint8:
            tensor = tensor.float().div_(255.0)
        else:
            tensor = tensor.float()
            if not torch.isfinite(tensor).all():
                raise ValueError("image contains non-finite values")
            if tensor.numel() and float(tensor.max()) > 1.0:
                tensor = tensor / 255.0
        if tensor.numel() and (float(tensor.min()) < 0.0 or float(tensor.max()) > 1.0):
            raise ValueError("RGB input values must be in [0,1] or [0,255]")
        height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
        if height <= 0 or width <= 0:
            raise ValueError("image dimensions must be positive")
        return tensor, width, height

    @staticmethod
    def _prepare_image(
        image: Any, page_resolution: int | None
    ) -> tuple[torch.Tensor, tuple[int, int], tuple[float, float]]:
        tensor, original_width, original_height = PrismOCR._tensor_image(image)
        scale = 1.0
        if page_resolution is not None:
            scale = min(1.0, page_resolution / max(original_width, original_height))
        resized_width = max(1, round(original_width * scale))
        resized_height = max(1, round(original_height * scale))
        if (resized_width, resized_height) != (original_width, original_height):
            tensor = F.interpolate(
                tensor[None], size=(resized_height, resized_width), mode="bilinear", align_corners=False
            )[0]
        return tensor, (original_width, original_height), (
            resized_width / original_width,
            resized_height / original_height,
        )

    @staticmethod
    def _pad_tile(tile_image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        height, width = int(tile_image.shape[-2]), int(tile_image.shape[-1])
        padded_width = (width + 15) // 16 * 16
        padded_height = (height + 15) // 16 * 16
        padded = tile_image.new_zeros((3, padded_height, padded_width))
        padded[:, :height, :width] = tile_image
        return padded, (padded_width, padded_height)

    @staticmethod
    def _polygon(
        box: torch.Tensor,
        padded_size: tuple[int, int],
        tile_size: tuple[int, int],
    ) -> list[list[float]]:
        padded_width, padded_height = padded_size
        tile_width, tile_height = tile_size
        points = rotated_box_to_polygon(
            box.detach().to(device="cpu", dtype=torch.float64).tolist(),
            (padded_width, padded_height),
        )
        clipped = polygon_intersection(
            points,
            ((0.0, 0.0), (float(tile_width), 0.0), (float(tile_width), float(tile_height)), (0.0, float(tile_height))),
        )
        if len(clipped) < 3 or polygon_area(clipped) <= 0:
            return []
        return [[round(x, 4), round(y, 4)] for x, y in clipped]

    @staticmethod
    def _decode_path(
        vocabulary: Vocabulary, token_ids: list[int], frame_confidences: list[float] | None = None
    ) -> tuple[DecodedText, list[float] | None]:
        decoded = vocabulary.decode_ctc(token_ids)
        if frame_confidences is None:
            return decoded, None
        collapsed_confidences: list[float] = []
        current_token: int | None = None
        current_values: list[float] = []
        previous: int | None = None
        collapsed_count = 0
        for token_id, confidence in zip(token_ids, frame_confidences):
            if token_id == 0:
                if current_token is not None:
                    collapsed_confidences.append(sum(current_values) / len(current_values))
                    current_token, current_values = None, []
                previous = token_id
                continue
            if token_id != previous:
                if current_token is not None:
                    collapsed_confidences.append(sum(current_values) / len(current_values))
                current_token, current_values = token_id, [confidence]
                collapsed_count += 1
            else:
                current_values.append(confidence)
            previous = token_id
        if current_token is not None:
            collapsed_confidences.append(sum(current_values) / len(current_values))
        if len(collapsed_confidences) != len(decoded.text):
            raise RuntimeError("greedy CTC confidence alignment does not match decoded codepoints")
        return decoded, collapsed_confidences

    def _tile_output(self, tile: Tile, page: torch.Tensor, options: InferenceOptions) -> _TileInference:
        crop = page[:, tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
        padded, padded_size = self._pad_tile(crop)
        model_options: dict[str, float] = {}
        if (
            options.capture_calibration_data is False
            and self.calibration.get("scores_calibrated") is True
            and "text_threshold" in self.calibration
        ):
            model_options["recognition_threshold"] = max(
                self.model.text_threshold,
                float(self.calibration["text_threshold"]),
            )
        try:
            outputs = self.model(padded[None].to(self.device), **model_options)
        except Exception as error:
            raise InferenceError(f"model_forward:tile={tile.index}", error) from None

        calibrated = bool(self.calibration.get("scores_calibrated", False))
        grid_calibrated = bool(self.calibration.get("grid_scores_calibrated", False))
        grid_threshold = float(self.calibration.get("grid_threshold", 0.0))
        header_threshold = float(self.calibration.get("header_threshold", 0.5))
        decoded_by_proposal: dict[int, tuple[DecodedText, list[float] | None]] = {}
        for group in outputs["ocr_groups"].values():
            refs = group["refs"].detach().cpu().tolist()
            logits = group["logits"].detach().float().cpu()
            for row, ref in enumerate(refs):
                token_ids = logits[row].argmax(dim=-1).tolist()
                frame_confidences = None
                if options.emit_character_confidence:
                    frame_confidences = logits[row].softmax(dim=-1).amax(dim=-1).tolist()
                decoded_by_proposal[int(ref[1])] = self._decode_path(
                    self.vocabulary, token_ids, frame_confidences
                )
        unsupported_by_proposal = {
            int(proposal_index): str(reason)
            for batch_index, proposal_index, reason in outputs["ocr_unsupported"]
            if batch_index == 0
        }

        text = outputs["text"]
        active_text = torch.where(text["mask"][0])[0].tolist()
        roles = text["role_logits"][0].argmax(dim=-1)
        text_scores = text["scores"][0]
        relation_embeddings = bool(self.calibration.get("relation_scores_calibrated", False))
        text_detections: list[dict[str, Any]] = []
        for proposal_index in active_text:
            polygon = self._polygon(text["boxes"][0, proposal_index], padded_size, tile.size)
            if len(polygon) < 3:
                continue
            decoded_and_conf = decoded_by_proposal.get(proposal_index)
            decoded = decoded_and_conf[0] if decoded_and_conf is not None else None
            detection: dict[str, Any] = {
                "polygon": polygon,
                "text": "" if decoded is None else decoded.text,
                "role": TEXT_ROLES[int(roles[proposal_index])],
                "detector_score": (
                    _calibrated_score(float(text_scores[proposal_index]), self.calibration, "text")
                    if calibrated else None
                ),
                "recognition_score": None,
                "language": None,
                "oov_positions": [] if decoded is None else list(decoded.oov_positions),
                "_tile_score": float(text_scores[proposal_index]),
            }
            if options.emit_character_confidence:
                detection["character_confidences"] = (
                    [] if decoded_and_conf is None or decoded_and_conf[1] is None else decoded_and_conf[1]
                )
                detection["character_confidence_method"] = "greedy_ctc_frame_posterior_mean_uncalibrated"
            if proposal_index in unsupported_by_proposal:
                detection["_unsupported_reason"] = unsupported_by_proposal[proposal_index]
            if relation_embeddings:
                detection["_embedding"] = text["embeddings"][0, proposal_index].detach().cpu()
            text_detections.append(detection)

        regions = outputs["regions"]
        active_regions = torch.where(regions["mask"][0])[0].tolist()
        region_detections: list[dict[str, Any]] = []
        for proposal_index in active_regions:
            polygon = self._polygon(regions["boxes"][0, proposal_index], padded_size, tile.size)
            if len(polygon) < 3:
                continue
            score = float(regions["scores"][0, proposal_index])
            kind = REGION_KINDS[int(regions["kinds"][0, proposal_index])]
            detection = {
                "polygon": polygon,
                "kind": kind,
                "detector_score": (
                    _calibrated_score(score, self.calibration, "region") if calibrated else None
                ),
                "_tile_score": score,
            }
            if kind == "CELL" and grid_calibrated:
                for field, logits_field, offset in (
                    ("row", "row_logits", 0),
                    ("col", "col_logits", 0),
                    ("rowspan", "rowspan_logits", 1),
                    ("colspan", "colspan_logits", 1),
                ):
                    logits = regions.get(logits_field)
                    if isinstance(logits, torch.Tensor):
                        probabilities = logits[0, proposal_index].float().softmax(dim=-1)
                        confidence, index = probabilities.max(dim=-1)
                        calibrated_confidence = _calibrated_grid_score(
                            float(confidence), self.calibration, field
                        )
                        if calibrated_confidence >= grid_threshold:
                            detection[field] = int(index) + offset
                header_logits = regions.get("header_logits")
                if isinstance(header_logits, torch.Tensor):
                    header_probability = float(header_logits[0, proposal_index].float().sigmoid())
                    header_confidence = _calibrated_grid_score(
                        max(header_probability, 1.0 - header_probability),
                        self.calibration,
                        "header",
                    )
                    if header_confidence >= grid_threshold:
                        detection["is_header"] = header_probability >= header_threshold
            if relation_embeddings:
                detection["_embedding"] = regions["embeddings"][0, proposal_index].detach().cpu()
            region_detections.append(detection)

        candidate_count = int(text["candidate_count"][0])
        capacity = int(self.model.text_proposals)
        overflow = bool(text["capacity_overflow"][0])
        structural_count = len(active_regions)
        structural_capacity = int(self.model.region_queries)
        structural_overflow = structural_capacity > 0 and structural_count >= structural_capacity
        return _TileInference(
            tile=tile,
            text_detections=tuple(text_detections),
            region_detections=tuple(region_detections),
            candidate_count=candidate_count,
            capacity=capacity,
            overflow=overflow,
            structural_count=structural_count,
            structural_capacity=structural_capacity,
            structural_overflow=structural_overflow,
            unsupported=tuple(sorted(unsupported_by_proposal.items())),
        )

    @staticmethod
    def _tile_plan(page_size: tuple[int, int], allow_tiling: bool) -> tuple[Tile, ...]:
        width, height = page_size
        if not allow_tiling:
            return (Tile(0, 0, 0, width, height),)
        tile_size = min(width, _MAX_TILE_SIZE[0]), min(height, _MAX_TILE_SIZE[1])
        overlap = (
            min(_INITIAL_TILE_OVERLAP, max(0, (tile_size[0] - 1) // 4)),
            min(_INITIAL_TILE_OVERLAP, max(0, (tile_size[1] - 1) // 4)),
        )
        return plan_tiles(page_size, tile_size, overlap)

    def _run_tiles(
        self, page: torch.Tensor, options: InferenceOptions
    ) -> tuple[tuple[_TileInference, ...], tuple[dict[str, Any], ...], int]:
        height, width = int(page.shape[-2]), int(page.shape[-1])
        initial = self._tile_plan((width, height), options.allow_tiling)
        pending = deque((tile.x, tile.y, tile.width, tile.height) for tile in initial)
        next_tile_index = 0
        final: list[_TileInference] = []
        warnings: list[dict[str, Any]] = []
        model_calls = 0
        while pending:
            x, y, tile_width, tile_height = pending.popleft()
            tile = Tile(next_tile_index, x, y, tile_width, tile_height)
            next_tile_index += 1
            result = self._tile_output(tile, page, options)
            model_calls += 1
            saturated_streams = []
            if result.overflow:
                saturated_streams.append("text")
            if result.structural_overflow:
                saturated_streams.append("structure")
            if saturated_streams and options.allow_tiling and max(tile.width, tile.height) > _MIN_RECOVERY_TILE:
                child_width = min(tile.width, max(_MIN_RECOVERY_TILE, (tile.width + 1) // 2))
                child_height = min(tile.height, max(_MIN_RECOVERY_TILE, (tile.height + 1) // 2))
                if child_width < tile.width or child_height < tile.height:
                    overlap = (
                        min(_INITIAL_TILE_OVERLAP // 4, max(0, (child_width - 1) // 4)),
                        min(_INITIAL_TILE_OVERLAP // 4, max(0, (child_height - 1) // 4)),
                    )
                    children = plan_tiles((tile.width, tile.height), (child_width, child_height), overlap)
                    pending.extend(
                        (tile.x + child.x, tile.y + child.y, child.width, child.height)
                        for child in children
                    )
                    warning = {
                        "code": "CAPACITY_OVERFLOW_RETRIED_WITH_TILES",
                        "tile_index": tile.index,
                        "streams": saturated_streams,
                        "retry_tile_count": len(children),
                    }
                    if result.overflow:
                        warning.update({
                            "candidate_count": result.candidate_count,
                            "capacity": result.capacity,
                        })
                    if result.structural_overflow:
                        warning.update({
                            "structural_count": result.structural_count,
                            "structural_capacity": result.structural_capacity,
                        })
                    warnings.append(warning)
                    preserved = result
                    if result.overflow:
                        preserved = replace(
                            preserved, text_detections=(), candidate_count=0, overflow=False
                        )
                    if result.structural_overflow:
                        preserved = replace(
                            preserved, region_detections=(), structural_count=0,
                            structural_overflow=False,
                        )
                    if preserved.text_detections or preserved.region_detections:
                        final.append(preserved)
                    continue
            final.append(result)
        final.sort(key=lambda result: result.tile.index)
        return tuple(final), tuple(warnings), model_calls

    @staticmethod
    def _tile_results(
        tile_inferences: tuple[_TileInference, ...], detections_field: Literal["text_detections", "region_detections"],
        *, region_kind: str | None = None,
    ) -> tuple[TileResult, ...]:
        results = []
        for result in tile_inferences:
            detections = getattr(result, detections_field)
            if region_kind is not None:
                detections = tuple(item for item in detections if item.get("kind") == region_kind)
            results.append(TileResult(
                tile=result.tile,
                detections=detections,
                candidate_count=result.candidate_count if detections_field == "text_detections" else None,
                capacity=result.capacity if detections_field == "text_detections" else None,
                overflow=result.overflow if detections_field == "text_detections" else False,
            ))
        return tuple(results)

    @staticmethod
    def _original_polygon(
        polygon: Any,
        scale: tuple[float, float],
        original_size: tuple[int, int],
    ) -> list[list[float]]:
        width, height = original_size
        scale_x, scale_y = scale
        points = [(float(point[0]) / scale_x, float(point[1]) / scale_y) for point in polygon]
        clipped = polygon_intersection(
            points,
            ((0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height))),
        )
        if len(clipped) < 3 or polygon_area(clipped) <= 0:
            return []
        return [[round(x, 4), round(y, 4)] for x, y in clipped]

    @staticmethod
    def _coordinate_polygon(
        polygon_px: list[list[float]],
        coordinate_space: str,
        original_size: tuple[int, int],
    ) -> list[list[float]]:
        if coordinate_space == "pixels":
            return polygon_px
        width, height = original_size
        return [[round(x / width, 6), round(y / height, 6)] for x, y in polygon_px]

    @staticmethod
    def _clean_detection(detection: dict[str, Any], return_debug: bool) -> dict[str, Any]:
        result = dict(detection)
        for key in ("_embedding", "_tile_score", "_unsupported_reason", "score"):
            result.pop(key, None)
        if not return_debug:
            result.pop("source_tiles", None)
            result.pop("source_detections", None)
        else:
            for source in result.get("source_detections", []):
                nested = source.get("detection")
                if isinstance(nested, dict):
                    for key in ("_embedding", "_tile_score", "_unsupported_reason", "score"):
                        nested.pop(key, None)
        return result

    @staticmethod
    def _reading_order_scopes(
        lines: list[dict[str, Any]], tables: list[dict[str, Any]], page_width: int
    ) -> dict[str, str]:
        """Assign table streams and conservatively detected text columns to scopes."""
        scopes: dict[str, str] = {}
        page_lines: list[tuple[dict[str, Any], tuple[float, float, float, float]]] = []
        for line in lines:
            line_id = line.get("id")
            polygon = line.get("polygon_px")
            if not isinstance(line_id, str):
                continue
            try:
                points = [(float(point[0]), float(point[1])) for point in polygon]
                if len(points) < 3 or any(not math.isfinite(value) for point in points for value in point):
                    continue
                area = polygon_area(points)
                if area <= 0:
                    continue
            except (TypeError, ValueError, IndexError, OverflowError):
                continue
            bounds = (
                min(point[0] for point in points), min(point[1] for point in points),
                max(point[0] for point in points), max(point[1] for point in points),
            )
            matching_tables: list[str] = []
            for table in tables:
                table_id = table.get("id")
                table_polygon = table.get("polygon_px", table.get("polygon"))
                if not isinstance(table_id, str) or not isinstance(table_polygon, list):
                    continue
                try:
                    overlap = polygon_area(polygon_intersection(points, table_polygon)) / area
                except (TypeError, ValueError, IndexError, OverflowError):
                    continue
                if overlap >= 0.01:
                    matching_tables.append(table_id)
            if len(matching_tables) == 1:
                scopes[line_id] = f"table:{matching_tables[0]}"
            elif len(matching_tables) > 1:
                scopes[line_id] = "table:ambiguous"
            else:
                page_lines.append((line, bounds))
                scopes[line_id] = "document"

        # Split only clearly separated, vertically interleaved line groups. This
        # avoids treating short labels or full-width headings as columns.
        column_lines = [entry for entry in page_lines if entry[1][2] - entry[1][0] <= page_width * 0.65]
        ordered = sorted(column_lines, key=lambda item: ((item[1][0] + item[1][2]) / 2.0, item[0]["id"]))
        groups: list[list[tuple[dict[str, Any], tuple[float, float, float, float]]]] = []
        for item in ordered:
            center = (item[1][0] + item[1][2]) / 2.0
            if not groups:
                groups.append([item])
                continue
            previous = groups[-1][-1][1]
            previous_center = (previous[0] + previous[2]) / 2.0
            if center - previous_center > max(1.0, page_width * 0.18):
                groups.append([item])
            else:
                groups[-1].append(item)
        # Isolated centered headings can form singleton x clusters between
        # otherwise well-populated columns. Keep them in document scope rather
        # than letting them disable the column evidence.
        column_groups = [group for group in groups if len(group) >= 2]
        has_column_evidence = len(column_groups) >= 2
        if has_column_evidence:
            vertical_overlap = False
            for first_index, first_group in enumerate(column_groups):
                for second_group in column_groups[first_index + 1 :]:
                    for _, first in first_group:
                        first_height = first[3] - first[1]
                        for _, second in second_group:
                            second_height = second[3] - second[1]
                            overlap = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
                            if overlap >= 0.25 * min(first_height, second_height):
                                vertical_overlap = True
                                break
                        if vertical_overlap:
                            break
                    if vertical_overlap:
                        break
                if vertical_overlap:
                    break
            centers = [
                (min((item[1][0] + item[1][2]) / 2.0 for item in group),
                 max((item[1][0] + item[1][2]) / 2.0 for item in group))
                for group in column_groups
            ]
            separated = all(
                centers[index + 1][0] - centers[index][1] > page_width * 0.18
                for index in range(len(centers) - 1)
            )
            if vertical_overlap and separated:
                for index, group in enumerate(column_groups):
                    for line, _ in group:
                        scopes[line["id"]] = f"column:{index:03d}"
        return scopes

    @staticmethod
    def _line_in_table(line: dict[str, Any], tables: list[dict[str, Any]]) -> bool:
        line_polygon = line.get("polygon_px")
        if not isinstance(line_polygon, list) or len(line_polygon) < 3:
            return False
        line_area = polygon_area(line_polygon)
        if line_area <= 0:
            return False
        for table in tables:
            table_polygon = table.get("polygon_px", table.get("polygon"))
            if not isinstance(table_polygon, list) or len(table_polygon) < 3:
                continue
            intersection = polygon_area(polygon_intersection(line_polygon, table_polygon))
            if intersection / line_area >= 0.01:
                return True
        return False

    @staticmethod
    def _public_cell(cell: Any, coordinate_space: str, original_size: tuple[int, int]) -> dict[str, Any]:
        polygon_px = None if cell.polygon is None else [[round(x, 4), round(y, 4)] for x, y in cell.polygon]
        return {
            "id": cell.id,
            "table_id": cell.table_id,
            "row": cell.row,
            "col": cell.col,
            "rowspan": cell.rowspan,
            "colspan": cell.colspan,
            "polygon": None if polygon_px is None else PrismOCR._coordinate_polygon(polygon_px, coordinate_space, original_size),
            "is_header": cell.is_header,
            "line_ids": list(cell.line_ids),
            "text": cell.text,
            "status": cell.status,
        }

    @staticmethod
    def _validate_result(payload: dict[str, Any]) -> None:
        if payload.get("schema_version") != "4.0":
            raise ValueError("internal result validation failed: unsupported schema version")
        image = payload.get("image", {})
        width, height = image.get("width"), image.get("height")
        if not isinstance(width, int) or width <= 0 or not isinstance(height, int) or height <= 0:
            raise ValueError("internal result validation failed: invalid image dimensions")
        coordinate_space = payload.get("coordinate_space")
        if coordinate_space not in {"pixels", "normalized"}:
            raise ValueError("internal result validation failed: invalid coordinate space")
        instances: dict[str, str] = {}
        for collection_name, kind in (("lines", "line"), ("regions", "region"), ("cells", "cell")):
            collection = payload.get(collection_name)
            if not isinstance(collection, list):
                raise ValueError(f"internal result validation failed: {collection_name} must be an array")
            for record in collection:
                instance_id = record.get("id")
                if not isinstance(instance_id, str) or not instance_id or instance_id in instances:
                    raise ValueError("internal result validation failed: instance IDs must be unique strings")
                instances[instance_id] = kind
                polygon = record.get("polygon")
                if kind == "cell" and polygon is None:
                    continue
                if not isinstance(polygon, list) or len(polygon) < 3:
                    raise ValueError(f"internal result validation failed: invalid polygon for {instance_id}")
                for point in polygon:
                    if not isinstance(point, list) or len(point) != 2 or any(
                        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                        for value in point
                    ):
                        raise ValueError(f"internal result validation failed: invalid polygon point for {instance_id}")
                    x, y = point
                    xmax, ymax = (1.0, 1.0) if coordinate_space == "normalized" else (width, height)
                    if x < -1e-6 or y < -1e-6 or x > xmax + 1e-6 or y > ymax + 1e-6:
                        raise ValueError(f"internal result validation failed: polygon outside page for {instance_id}")
        line_ids = {key for key, kind in instances.items() if kind == "line"}
        if payload.get("reading_order") is None or len(payload["reading_order"]) != len(line_ids) or set(payload["reading_order"]) != line_ids:
            raise ValueError("internal result validation failed: reading_order must include each line exactly once")
        table_ids = {region["id"] for region in payload["regions"] if region.get("kind") == "TABLE"}
        for cell in payload["cells"]:
            table_id = cell.get("table_id")
            if table_id is not None and table_id not in table_ids:
                raise ValueError(f"internal result validation failed: unknown table for cell {cell['id']}")
            if cell.get("is_header") is not None and not isinstance(cell["is_header"], bool):
                raise ValueError("internal result validation failed: invalid cell header flag")
            for field in ("row", "col"):
                value = cell.get(field)
                if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                    raise ValueError(f"internal result validation failed: invalid cell {field}")
            for field in ("rowspan", "colspan"):
                value = cell.get(field)
                if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                    raise ValueError(f"internal result validation failed: invalid cell {field}")
        for relation in payload.get("relations", []):
            if relation.get("source") not in instances:
                raise ValueError("internal result validation failed: unknown relation source")
            target = relation.get("target")
            if target is not None and target not in instances:
                raise ValueError("internal result validation failed: unknown relation target")
            score = relation.get("score")
            if score is not None and (
                isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score)
            ):
                raise ValueError("internal result validation failed: invalid relation score")
        json.dumps(payload, ensure_ascii=False, allow_nan=False)

    def predict(self, image: Any, options: InferenceOptions | None = None) -> DocumentResult:
        options = options or InferenceOptions()
        if not isinstance(options, InferenceOptions):
            raise TypeError("options must be an InferenceOptions value or None")
        try:
            return self._predict_page(image, options)
        except InferenceError:
            raise
        except Exception as error:
            raise InferenceError("prediction", error) from None

    @torch.inference_mode()
    def _predict_page(self, image: Any, options: InferenceOptions) -> DocumentResult:
        try:
            page, original_size, scale = self._prepare_image(image, options.page_resolution)
        except Exception as error:
            raise InferenceError("preprocessing", error, code="INVALID_IMAGE") from None
        self.model.eval()
        tiles, retry_warnings, model_calls = self._run_tiles(page, options)
        tile_warnings: list[dict[str, Any]] = list(retry_warnings)
        width, height = original_size
        page_size = (int(page.shape[-1]), int(page.shape[-2]))
        was_tiled = len(tiles) > 1 or model_calls > 1

        text_stitch = stitch_tile_results(
            self._tile_results(tiles, "text_detections"), id_prefix="l"
        )
        tile_warnings.extend(text_stitch.warnings)
        lines_internal: list[dict[str, Any]] = []
        unsupported_warnings: list[dict[str, Any]] = []
        text_operating_threshold = (
            float(self.calibration["text_threshold"])
            if self.calibration.get("scores_calibrated") is True
            and "text_threshold" in self.calibration
            else None
        )
        for detection in text_stitch.detections:
            if (
                text_operating_threshold is not None
                and float(detection["_tile_score"]) < text_operating_threshold
            ):
                continue
            record = dict(detection)
            polygon_px = self._original_polygon(record["polygon"], scale, original_size)
            if len(polygon_px) < 3:
                continue
            record["polygon_px"] = polygon_px
            record["polygon"] = self._coordinate_polygon(polygon_px, options.output_coordinate_space, original_size)
            reason = record.pop("_unsupported_reason", None)
            if reason is not None:
                unsupported_warnings.append({"code": "UNSUPPORTED_LINE_LENGTH", "line_id": record["id"], "reason": reason})
            if record.get("oov_positions"):
                unsupported_warnings.append({
                    "code": "UNSUPPORTED_CHARSET",
                    "line_id": record["id"],
                    "oov_count": len(record["oov_positions"]),
                    "oov_positions": list(record["oov_positions"]),
                })
            lines_internal.append(record)

        region_kinds = sorted({
            str(detection.get("kind"))
            for tile_result in tiles
            for detection in tile_result.region_detections
        })
        region_internal: list[dict[str, Any]] = []
        region_operating_threshold = (
            float(self.calibration["region_threshold"])
            if self.calibration.get("scores_calibrated") is True
            and "region_threshold" in self.calibration
            else None
        )
        for kind in region_kinds:
            stitched = stitch_tile_results(
                self._tile_results(tiles, "region_detections", region_kind=kind), id_prefix="r"
            )
            tile_warnings.extend(stitched.warnings)
            for detection in stitched.detections:
                if (
                    region_operating_threshold is not None
                    and float(detection["_tile_score"]) < region_operating_threshold
                ):
                    continue
                record = dict(detection)
                polygon_px = self._original_polygon(record["polygon"], scale, original_size)
                if len(polygon_px) < 3:
                    continue
                record["polygon_px"] = polygon_px
                record["polygon"] = self._coordinate_polygon(polygon_px, options.output_coordinate_space, original_size)
                region_internal.append(record)
        region_internal.sort(key=lambda item: (
            item["kind"],
            sum(point[1] for point in item["polygon_px"]) / len(item["polygon_px"]),
            sum(point[0] for point in item["polygon_px"]) / len(item["polygon_px"]),
            item.get("source_tiles", [0])[0],
        ))

        regions: list[dict[str, Any]] = []
        cells_internal: list[dict[str, Any]] = []
        region_embedding: dict[str, torch.Tensor] = {}
        cell_embedding: dict[str, torch.Tensor] = {}
        region_index = cell_index = 0
        for detection in region_internal:
            if detection["kind"] == "CELL":
                cell_index += 1
                detection["id"] = f"c:{cell_index}"
                detection.update({
                    "table_id": None,
                    "row": detection.get("row"),
                    "col": detection.get("col"),
                    "rowspan": detection.get("rowspan"),
                    "colspan": detection.get("colspan"),
                    "is_header": detection.get("is_header"),
                    "line_ids": [],
                    "text": "",
                    "status": "unassigned",
                })
                cells_internal.append(detection)
                if isinstance(detection.get("_embedding"), torch.Tensor):
                    cell_embedding[detection["id"]] = detection["_embedding"]
            else:
                region_index += 1
                detection["id"] = f"r:{region_index}"
                detection["parent_id"] = None
                regions.append(detection)
                if isinstance(detection.get("_embedding"), torch.Tensor):
                    region_embedding[detection["id"]] = detection["_embedding"]

        table_regions = [record for record in regions if record.get("kind") == "TABLE"]
        table_decode: TableDecodeResult = reconstruct_tables(
            table_regions,
            cells_internal,
            [line for line in lines_internal if self._line_in_table(line, table_regions)],
        )
        reconstructed_cells: list[dict[str, Any]] = []
        for table in table_decode.tables:
            for decoded_cell in table.cells:
                public_cell = self._public_cell(decoded_cell, options.output_coordinate_space, original_size)
                prior = next((cell for cell in cells_internal if cell["id"] == public_cell["id"]), None)
                if prior is not None:
                    public_cell["detector_score"] = prior.get("detector_score")
                    if prior.get("polygon_px") is not None:
                        public_cell["polygon_px"] = prior["polygon_px"]
                    if isinstance(prior.get("_embedding"), torch.Tensor):
                        cell_embedding[public_cell["id"]] = prior["_embedding"]
                reconstructed_cells.append(public_cell)
        assigned_cell_ids = {cell["id"] for cell in reconstructed_cells}
        for unassigned in table_decode.unassigned_cells:
            prior = next((cell for cell in cells_internal if cell["id"] == unassigned.id), None)
            polygon_px = None if unassigned.polygon is None else [[round(x, 4), round(y, 4)] for x, y in unassigned.polygon]
            record = {
                "id": unassigned.id,
                "table_id": None,
                "row": None,
                "col": None,
                "rowspan": None,
                "colspan": None,
                "polygon": None if polygon_px is None else self._coordinate_polygon(polygon_px, options.output_coordinate_space, original_size),
                "polygon_px": polygon_px,
                "is_header": None,
                "line_ids": [],
                "text": "",
                "status": "unassigned",
                "unassigned_reason": unassigned.reason,
            }
            if prior is not None:
                record["detector_score"] = prior.get("detector_score")
                if isinstance(prior.get("_embedding"), torch.Tensor):
                    cell_embedding[record["id"]] = prior["_embedding"]
            reconstructed_cells.append(record)
        for cell in cells_internal:
            if cell["id"] not in assigned_cell_ids and cell["id"] not in {item.id for item in table_decode.unassigned_cells}:
                reconstructed_cells.append({
                    "id": cell["id"],
                    "table_id": None,
                    "row": None,
                    "col": None,
                    "rowspan": None,
                    "colspan": None,
                    "polygon": cell["polygon"],
                    "polygon_px": cell["polygon_px"],
                    "is_header": None,
                    "line_ids": [],
                    "text": "",
                    "status": "unassigned",
                    "unassigned_reason": "not_returned_by_table_decoder",
                })
        reconstructed_cells.sort(key=lambda cell: cell["id"])
        table_outputs = []
        for table in table_decode.tables:
            table_outputs.append({
                "table_id": table.id,
                "status": table.status,
                "rows": table.rows,
                "cols": table.cols,
                "cells": [cell["id"] for cell in reconstructed_cells if cell.get("table_id") == table.id],
                "html": table.html,
                "markdown": table.markdown,
                "conflicts": list(table.conflicts),
            })
            if table.status == "conflict":
                tile_warnings.append({"code": "TABLE_GRID_CONFLICT", "table_id": table.id, "conflicts": list(table.conflicts)})
        for cell in reconstructed_cells:
            if cell.get("table_id") is None:
                tile_warnings.append({
                    "code": "CELL_UNASSIGNED",
                    "cell_id": cell["id"],
                    "reason": cell.get("unassigned_reason", "table_grid_conflict"),
                })
        if table_decode.unassigned_line_ids:
            tile_warnings.append({
                "code": "TABLE_LINES_UNASSIGNED",
                "line_ids": list(table_decode.unassigned_line_ids),
            })

        reading_order_scopes = self._reading_order_scopes(lines_internal, table_regions, width)
        graph_nodes: list[GraphNode] = []
        graph_node_ids: list[str] = []
        graph_embeddings: list[torch.Tensor] = []
        line_by_id = {line["id"]: line for line in lines_internal}
        region_by_id = {region["id"]: region for region in regions}
        cell_by_id = {cell["id"]: cell for cell in reconstructed_cells}
        for line in lines_internal:
            graph_nodes.append(GraphNode(
                id=line["id"],
                node_type="line",
                polygon=line["polygon_px"],
                role=line["role"],
                scope_id=reading_order_scopes.get(line["id"], "document"),
            ))
            graph_node_ids.append(line["id"])
            if isinstance(line.get("_embedding"), torch.Tensor):
                graph_embeddings.append(line["_embedding"])
            else:
                graph_embeddings.append(torch.zeros(self.model.dim))
        for region in regions:
            graph_nodes.append(GraphNode(
                id=region["id"], node_type="region", polygon=region["polygon_px"], kind=region["kind"]
            ))
            graph_node_ids.append(region["id"])
            graph_embeddings.append(region_embedding.get(region["id"], torch.zeros(self.model.dim)))
        for cell in reconstructed_cells:
            graph_nodes.append(GraphNode(
                id=cell["id"],
                node_type="cell",
                polygon=cell.get("polygon_px"),
                table_id=cell.get("table_id"),
                row=cell.get("row"),
                col=cell.get("col"),
            ))
            graph_node_ids.append(cell["id"])
            graph_embeddings.append(cell_embedding.get(cell["id"], torch.zeros(self.model.dim)))

        relation_scores: dict[str, dict[str, dict[str, float]]] = {}
        null_scores: dict[str, dict[str, float]] = {}
        relation_candidate_calibration_data: list[dict[str, Any]] = []
        relation_calibrated = bool(self.calibration.get("relation_scores_calibrated", False))
        if relation_calibrated and graph_node_ids:
            ordered_ids = [node.id for node in graph_nodes]
            try:
                embeddings = torch.stack(graph_embeddings).to(self.device).unsqueeze(0)
                relation_logits = self.model.relation_head(embeddings)[0].float().cpu()
            except Exception as error:
                raise InferenceError("relation_scoring", error) from None
            indices = {node_id: index for index, node_id in enumerate(ordered_ids)}
            nodes_by_id = {node.id: node for node in graph_nodes}
            allowed_sources = {
                "NEXT": {node.id for node in graph_nodes if node.node_type == "line"},
                "PARENT": {node.id for node in graph_nodes},
                "KV_VALUE": {node.id for node in graph_nodes if node.node_type == "line" and node.role == "KEY"},
                "RIGHT_CELL": {node.id for node in graph_nodes if node.node_type == "cell"},
            }
            for relation_index, relation in enumerate(RELATION_NAMES[:4]):
                relation_scores[relation] = {}
                null_scores[relation] = {}
                for source_id in sorted(allowed_sources[relation]):
                    source_index = indices[source_id]
                    source = nodes_by_id[source_id]
                    valid_target_ids = [
                        target_id for target_id in ordered_ids
                        if relation_candidate_error(
                            relation,
                            source,
                            nodes_by_id[target_id],
                            nodes_by_id,
                            allow_right_cell_column_repair=relation == "RIGHT_CELL",
                        ) is None
                    ]
                    candidate_indices = [indices[target_id] for target_id in valid_target_ids]
                    candidate_logits = relation_logits[relation_index, source_index, candidate_indices]
                    null_logit = relation_logits[relation_index, source_index, -1]
                    probabilities = torch.cat((candidate_logits, null_logit.reshape(1))).softmax(dim=0)
                    if options.capture_calibration_data:
                        relation_candidate_calibration_data.extend(
                            {
                                "type": relation,
                                "source": source_id,
                                "target": target_id,
                                "score": float(probability),
                            }
                            for target_id, probability in zip(valid_target_ids, probabilities[:-1])
                        )
                        relation_candidate_calibration_data.append({
                            "type": relation,
                            "source": source_id,
                            "target": None,
                            "score": float(probabilities[-1]),
                        })
                    target_scores = {target_id: 0.0 for target_id in ordered_ids}
                    for target_id, probability in zip(valid_target_ids, probabilities[:-1]):
                        target_scores[target_id] = _calibrated_relation_score(
                            float(probability), self.calibration, relation
                        )
                    relation_scores[relation][source_id] = target_scores
                    null_scores[relation][source_id] = _calibrated_relation_score(
                        float(probabilities[-1]), self.calibration, relation
                    )
        if relation_calibrated:
            try:
                affinity_table_decode = reconstruct_tables(
                    table_regions,
                    cells_internal,
                    [line for line in lines_internal if self._line_in_table(line, table_regions)],
                    right_cell_scores=relation_scores.get("RIGHT_CELL", {}),
                    right_cell_null_scores=null_scores.get("RIGHT_CELL", {}),
                    use_right_cell_constraints=True,
                )
            except Exception as error:
                raise InferenceError("table_placement", error) from None
            affinity_cells = {
                cell.id: cell
                for table in affinity_table_decode.tables
                for cell in table.cells
            }
            for cell in reconstructed_cells:
                decoded_cell = affinity_cells.get(cell["id"])
                if decoded_cell is None:
                    continue
                original_polygon = cell.get("polygon_px")
                cell.update(self._public_cell(
                    decoded_cell, options.output_coordinate_space, original_size
                ))
                if original_polygon is not None:
                    cell["polygon_px"] = original_polygon
            affinity_by_id = affinity_cells
            graph_nodes = [
                GraphNode(
                    id=node.id,
                    node_type=node.node_type,
                    polygon=node.polygon,
                    role=node.role,
                    kind=node.kind,
                    scope_id=node.scope_id,
                    table_id=(affinity_by_id[node.id].table_id if node.id in affinity_by_id else node.table_id),
                    row=(affinity_by_id[node.id].row if node.id in affinity_by_id else node.row),
                    col=(affinity_by_id[node.id].col if node.id in affinity_by_id else node.col),
                )
                for node in graph_nodes
            ]

        decoded_graph = decode_graph(
            graph_nodes,
            relation_scores,
            null_scores,
            suppressed_limit=_RELATION_DIAGNOSTIC_LIMIT,
        )
        if not relation_calibrated and lines_internal:
            tile_warnings.append({"code": "READING_ORDER_GEOMETRIC_FALLBACK"})
        predictions = list(decoded_graph.predictions)
        parent_by_child = {
            prediction.source_id: prediction.target_id
            for prediction in predictions
            if prediction.relation == "PARENT" and prediction.target_id is not None
        }
        reading_order = list(decoded_graph.reading_order)
        for index, line_id in enumerate(reading_order):
            line_by_id[line_id]["reading_order"] = index
            line_by_id[line_id]["parent_id"] = parent_by_child.get(line_id)
        for region in regions:
            region["parent_id"] = parent_by_child.get(region["id"])

        # Reassemble presentation text only after learned NEXT edges have
        # populated each line's logical reading order.
        ordered_table_decode = reconstruct_tables(
            table_regions,
            cells_internal,
            [line for line in lines_internal if self._line_in_table(line, table_regions)],
            right_cell_scores=relation_scores.get("RIGHT_CELL", {}),
            right_cell_null_scores=null_scores.get("RIGHT_CELL", {}),
            use_right_cell_constraints=relation_calibrated,
        )
        ordered_cells = {
            cell.id: cell
            for table in ordered_table_decode.tables
            for cell in table.cells
        }
        for cell in reconstructed_cells:
            decoded_cell = ordered_cells.get(cell["id"])
            if decoded_cell is not None:
                cell.update(self._public_cell(
                    decoded_cell, options.output_coordinate_space, original_size
                ))
                if decoded_cell.polygon is not None:
                    cell["polygon_px"] = [[round(x, 4), round(y, 4)] for x, y in decoded_cell.polygon]
        ordered_tables = {table.id: table for table in ordered_table_decode.tables}
        for output in table_outputs:
            table = ordered_tables.get(output["table_id"])
            if table is not None:
                if table.status == "conflict" and not any(
                    warning.get("code") == "TABLE_GRID_CONFLICT"
                    and warning.get("table_id") == table.id
                    for warning in tile_warnings
                ):
                    tile_warnings.append({
                        "code": "TABLE_GRID_CONFLICT",
                        "table_id": table.id,
                        "conflicts": list(table.conflicts),
                    })
                output.update({
                    "status": table.status,
                    "rows": table.rows,
                    "cols": table.cols,
                    "cells": [cell["id"] for cell in reconstructed_cells if cell.get("table_id") == table.id],
                    "html": table.html,
                    "markdown": table.markdown,
                    "conflicts": list(table.conflicts),
                })
        table_decode = ordered_table_decode
        table_warning_codes = {
            "TABLE_GRID_CONFLICT",
            "CELL_UNASSIGNED",
            "TABLE_LINES_UNASSIGNED",
        }
        tile_warnings[:] = [
            warning for warning in tile_warnings
            if warning.get("code") not in table_warning_codes
        ]
        for table in table_outputs:
            if table["status"] == "conflict":
                tile_warnings.append({
                    "code": "TABLE_GRID_CONFLICT",
                    "table_id": table["table_id"],
                    "conflicts": list(table["conflicts"]),
                })
        for cell in reconstructed_cells:
            if cell.get("table_id") is None:
                tile_warnings.append({
                    "code": "CELL_UNASSIGNED",
                    "cell_id": cell["id"],
                    "reason": cell.get("unassigned_reason", "table_grid_conflict"),
                })
        if table_decode.unassigned_line_ids:
            tile_warnings.append({
                "code": "TABLE_LINES_UNASSIGNED",
                "line_ids": list(table_decode.unassigned_line_ids),
            })

        public_relations = [{
            "type": prediction.relation,
            "source": prediction.source_id,
            "target": prediction.target_id,
            "score": float(prediction.score),
            "reason_codes": list(prediction.reason_codes),
        } for prediction in predictions]
        suppressed = decoded_graph.suppressed
        suppressed_count = decoded_graph.suppressed_count
        suppressed_counts = dict(decoded_graph.suppressed_reason_counts)
        if suppressed_count:
            tile_warnings.append({
                "code": "RELATIONS_SUPPRESSED",
                "count": suppressed_count,
                "reason_counts": suppressed_counts,
            })
        relation_diagnostics = [{
            "type": item.relation,
            "source": item.source_id,
            "target": item.target_id,
            "score": item.score,
            "reason_code": item.reason_code,
        } for item in suppressed[:_RELATION_DIAGNOSTIC_LIMIT]]

        def target_text(node_id: str | None) -> str:
            if node_id is None:
                return ""
            if node_id in line_by_id:
                return str(line_by_id[node_id].get("text", ""))
            target_region = region_by_id.get(node_id)
            if target_region is None:
                return ""
            contained = [line for line in lines_internal if line.get("parent_id") == node_id]
            contained.sort(key=lambda line: line.get("reading_order", 0))
            return "\n".join(str(line.get("text", "")) for line in contained)

        kv_pairs = []
        for prediction in predictions:
            if prediction.relation != "KV_VALUE" or prediction.target_id is None:
                continue
            key_record = line_by_id.get(prediction.source_id)
            if key_record is None:
                continue
            kv_pairs.append({
                "key": str(key_record.get("text", "")),
                "value": target_text(prediction.target_id),
                "source_id": prediction.source_id,
                "target_id": prediction.target_id,
                "score": float(prediction.score),
                "page_id": "prediction",
                "reading_order": key_record.get("reading_order"),
            })

        warnings: list[dict[str, Any]] = []
        if was_tiled:
            warnings.append({"code": "TILING_USED", "tile_count": len(tiles), "model_calls": model_calls})
        if options.allow_tiling is False and max(page_size) > max(_MAX_TILE_SIZE):
            warnings.append({"code": "TILING_DISABLED_PAGE_PROCESSED_WHOLE"})
        warnings.extend(tile_warnings)
        warnings.extend(unsupported_warnings)
        if options.emit_character_confidence:
            warnings.append({"code": "CHARACTER_CONFIDENCE_UNCALIBRATED"})
        if not bool(self.calibration.get("scores_calibrated", False)):
            warnings.append({"code": "SCORES_UNCALIBRATED"})
        if not relation_calibrated and graph_nodes:
            warnings.append({"code": "RELATION_SCORES_UNCALIBRATED"})

        final_lines = []
        for line in lines_internal:
            cleaned = self._clean_detection(line, options.return_debug)
            final_lines.append({
                "id": cleaned["id"],
                "polygon": cleaned["polygon"],
                "text": cleaned["text"],
                "role": cleaned["role"],
                "scope_id": reading_order_scopes.get(cleaned["id"], "document"),
                "detector_score": cleaned.get("detector_score"),
                "recognition_score": None,
                "language": None,
                "parent_id": cleaned.get("parent_id"),
                "reading_order": cleaned["reading_order"],
                "oov_positions": cleaned.get("oov_positions", []),
                **({
                    "character_confidences": cleaned.get("character_confidences", []),
                    "character_confidence_method": cleaned.get("character_confidence_method"),
                } if options.emit_character_confidence else {}),
                **({
                    "source_tiles": cleaned.get("source_tiles", []),
                    "source_detections": cleaned.get("source_detections", []),
                } if options.return_debug else {}),
            })
        final_regions = []
        for region in regions:
            cleaned = self._clean_detection(region, options.return_debug)
            final_regions.append({
                "id": cleaned["id"],
                "kind": cleaned["kind"],
                "polygon": cleaned["polygon"],
                "parent_id": cleaned.get("parent_id"),
                "detector_score": cleaned.get("detector_score"),
                **({
                    "source_tiles": cleaned.get("source_tiles", []),
                    "source_detections": cleaned.get("source_detections", []),
                } if options.return_debug else {}),
            })
        final_cells = [{
            key: value for key, value in cell.items() if key != "polygon_px"
        } for cell in reconstructed_cells]

        candidate_count = sum(result.candidate_count for result in tiles)
        text_overflow = any(result.overflow for result in tiles)
        structural_overflow = any(result.structural_overflow for result in tiles)
        overflow = text_overflow or structural_overflow
        if overflow:
            overflow_warning: dict[str, Any] = {
                "code": "CAPACITY_OVERFLOW",
                "partial_result": True,
                "streams": [
                    *(["text"] if text_overflow else []),
                    *(["structure"] if structural_overflow else []),
                ],
            }
            if text_overflow:
                overflow_warning["candidate_count"] = candidate_count
            warnings.append(overflow_warning)
        if structural_overflow:
            saturated_tiles = [result for result in tiles if result.structural_overflow]
            warnings.append({
                "code": "STRUCTURE_CAPACITY_OVERFLOW",
                "partial_result": True,
                "saturated_tile_count": len(saturated_tiles),
                "candidate_count": sum(result.structural_count for result in saturated_tiles),
                "capacity": sum(result.structural_capacity for result in saturated_tiles),
            })
        payload: dict[str, Any] = {
            "schema_version": "4.0",
            "model_version": self.model_version,
            "document_id": "prediction",
            "image": {"width": width, "height": height},
            "coordinate_space": options.output_coordinate_space,
            "lines": final_lines,
            "regions": final_regions,
            "cells": final_cells,
            "tables": table_outputs,
            "relations": public_relations,
            "kv_pairs": kv_pairs,
            "reading_order": reading_order,
            "warnings": warnings,
            "diagnostics": {
                "tiles": len(tiles),
                "tile_model_calls": model_calls,
                "text_proposals": len(final_lines),
                "region_proposals": len(final_regions) + len(final_cells),
                "candidate_count": candidate_count,
                "overflow": overflow,
                "text_capacity_overflow": text_overflow,
                "structural_capacity_overflow": structural_overflow,
                "structural_candidate_count": sum(result.structural_count for result in tiles),
                "structural_query_capacity": sum(result.structural_capacity for result in tiles),
                "execution_ms": None,
                "relation_suppressed_count": suppressed_count,
                "relation_suppressed": relation_diagnostics,
                "relation_diagnostics_truncated": suppressed_count > len(relation_diagnostics),
                "table_conflict_count": sum(table["status"] == "conflict" for table in table_outputs),
            },
        }
        if options.capture_calibration_data:
            payload["diagnostics"]["relation_candidate_calibration_data"] = (
                relation_candidate_calibration_data
            )
        if options.return_debug:
            payload["diagnostics"]["tile_plan"] = [
                {"index": result.tile.index, "origin": list(result.tile.origin), "size": list(result.tile.size)}
                for result in tiles
            ]
            payload["diagnostics"]["unassigned_table_line_ids"] = list(table_decode.unassigned_line_ids)
        self._validate_result(payload)
        return DocumentResult(payload)

    def predict_batch(
        self, images: list[Any], options: InferenceOptions | None = None
    ) -> list[DocumentResult]:
        if not isinstance(images, list):
            raise TypeError("images must be a list")
        return [self.predict(image, options) for image in images]


def save_artifact(
    artifact_dir: str | Path,
    model: PrismOCRModel,
    vocabulary: Vocabulary,
    *,
    model_version: str = "v4.0.0-experiment",
    training_provenance: Mapping[str, Any] | None = None,
) -> None:
    """Write model assets and invalidate calibration bound to prior weights."""
    backbone = getattr(model, "backbone", None)
    if backbone is not None and any(
        getattr(module, "fused_conv", None) is not None
        for module in backbone.modules()
    ):
        raise ValueError(
            "save_artifact requires an unfused training model; loaded inference models are fused"
        )
    if model.vocab_size != len(vocabulary):
        raise ValueError("model and vocabulary sizes do not match")
    tensor_bytes = model.tensor_bytes()
    if tensor_bytes > _MAX_FP32_TENSOR_BYTES:
        raise ValueError(
            f"model tensor inventory is {tensor_bytes} bytes, exceeding the "
            f"{_MAX_FP32_TENSOR_BYTES}-byte FP32 limit"
        )
    provenance = None if training_provenance is None else dict(training_provenance)
    if provenance is not None:
        for field in ("train_dataset_sha256", "validation_dataset_sha256"):
            digest = provenance.get(field)
            if digest is not None and (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest.lower())
            ):
                raise ValueError(f"training provenance {field} must be a SHA-256 digest or null")
        if provenance.get("train_dataset_sha256") is None:
            raise ValueError("training provenance requires train_dataset_sha256")
    root = Path(artifact_dir)
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{root.name}-", dir=root.parent) as temporary:
        staged = Path(temporary)
        vocabulary_path = staged / "vocab-v4.json"
        config_path = staged / "model-config.json"
        weights_path = staged / "model.pt"
        manifest_path = staged / "manifest.json"
        vocabulary.save(vocabulary_path)
        config = {
            "model_contract_version": getattr(model, "model_contract_version", 1),
            "dim": model.dim,
            "text_proposals": model.text_proposals,
            "region_queries": model.region_queries,
            "vocab_size": model.vocab_size,
            "strip_height": model.recognizer.strip_height,
            "strip_buckets": list(model.strip_buckets),
            "recognizer": model.recognizer.variant,
            "text_threshold": model.text_threshold,
            "region_threshold": model.region_threshold,
            "match_iou_threshold": model.match_iou_threshold,
            "max_table_rows": getattr(model, "max_table_rows", 128),
            "max_table_cols": getattr(model, "max_table_cols", 128),
            "max_cell_span": getattr(model, "max_cell_span", 32),
            "fused_text_head": getattr(model, "fused_text_head", False),
        }
        config_bytes = (json.dumps(config, indent=2) + "\n").encode("utf-8")
        config_path.write_bytes(config_bytes)
        torch.save({"model_state": model.state_dict(),
                    "model_contract_version": getattr(model, "model_contract_version", 1)}, weights_path)
        manifest = {
            "schema_version": "4.0",
            "model_version": model_version,
            "model_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
            "vocabulary_sha256": vocabulary.sha256,
            "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "calibration_sha256": None,
            "model_tensor_bytes": tensor_bytes,
            "training_provenance": provenance,
            "scores_calibrated": False,
            "relation_scores_calibrated": False,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        filenames = (
            "vocab-v4.json", "model-config.json", "model.pt", "calibration-v4.json", "manifest.json"
        )
        previous = {name: (root / name).exists() for name in filenames}
        backup = staged / "previous"
        backup.mkdir()
        for name, existed in previous.items():
            if existed:
                if not (root / name).is_file():
                    raise ValueError(f"artifact asset is not a regular file: {root / name}")
                shutil.copy2(root / name, backup / name)

        published: list[str] = []
        try:
            for name in ("vocab-v4.json", "model-config.json", "model.pt"):
                published.append(name)
                os.replace(staged / name, root / name)
            if previous["calibration-v4.json"]:
                published.append("calibration-v4.json")
                (root / "calibration-v4.json").unlink()
            published.append("manifest.json")
            os.replace(manifest_path, root / "manifest.json")
        except BaseException as error:
            rollback_errors = []
            for name in reversed(published):
                target = root / name
                try:
                    if previous[name]:
                        os.replace(backup / name, target)
                    else:
                        target.unlink(missing_ok=True)
                except OSError as rollback_error:
                    rollback_errors.append(f"{name}: {type(rollback_error).__name__}")
            if rollback_errors:
                raise RuntimeError(
                    "artifact publication failed and rollback was incomplete: "
                    + ", ".join(rollback_errors)
                ) from error
            raise
