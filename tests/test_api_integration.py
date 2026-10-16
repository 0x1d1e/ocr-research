from __future__ import annotations

import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path

import torch
from PIL import Image

from prismocr.api import InferenceError, InferenceOptions, PrismOCR, save_artifact
from prismocr.data.vocabulary import Vocabulary
from prismocr.model.network import REGION_KIND_TO_ID, TEXT_ROLE_TO_ID, PrismOCRModel


def make_model(vocabulary: Vocabulary, *, text_proposals: int, region_queries: int) -> PrismOCRModel:
    return PrismOCRModel(
        dim=10,
        text_proposals=text_proposals,
        region_queries=region_queries,
        vocab_size=len(vocabulary),
        vocabulary=vocabulary,
    )


def text_box(cx: float, cy: float, width: float, height: float) -> list[float]:
    return [cx, cy, width, height, 0.0, 1.0]


def _calibrator_fixture() -> dict[str, object]:
    return {
        "method": "platt_v1",
        "slope": 1.0,
        "intercept": 0.0,
        "positive_count": 1,
        "negative_count": 1,
        "reliability": {"ece": 0.0, "brier": 0.0},
        "precision_recall_curve": [],
    }


def _calibration_provenance_fixture() -> dict[str, object]:
    return {
        "method": "validation_only_platt_scaling_v1",
        "model_sha256": "3" * 64,
        "model_config_sha256": "4" * 64,
        "vocabulary_sha256": "5" * 64,
        "split_sha256": "1" * 64,
        "split_report_sha256": "2" * 64,
        "validation_page_count": 1,
        "iou_threshold": 0.5,
    }


class PublicInferenceIntegrationTests(unittest.TestCase):
    def test_nonfinite_calibrated_detector_score_fails_as_inference_error(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        model = make_model(vocabulary, text_proposals=1, region_queries=1)
        api = PrismOCR(
            model,
            vocabulary,
            calibration={
                **_calibration_provenance_fixture(),
                "scores_calibrated": True,
                "score_calibrators": {
                    "text": _calibrator_fixture(),
                    "region": _calibrator_fixture(),
                },
            },
        )
        original_forward = api.model.forward

        def nonfinite_forward(_self: PrismOCRModel, images: torch.Tensor) -> dict[str, object]:
            output = original_forward(images)
            output["text"]["mask"][0, 0] = True
            output["text"]["boxes"][0, 0] = torch.tensor(
                [0.5, 0.5, 0.5, 0.1, 0.0, 1.0], device=images.device
            )
            output["text"]["scores"][0, 0] = float("nan")
            return output

        api.model.forward = types.MethodType(nonfinite_forward, api.model)
        with self.assertRaises(InferenceError) as caught:
            api.predict(torch.zeros((3, 64, 64), dtype=torch.float32))
        self.assertEqual(caught.exception.code, "INFERENCE_ERROR")

    def test_calibrated_operating_threshold_filters_after_stitching(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        model = make_model(vocabulary, text_proposals=2, region_queries=1)

        received: dict[str, float | None] = {}

        def fake_forward(
            _self: PrismOCRModel,
            images: torch.Tensor,
            *,
            recognition_threshold: float | None = None,
        ) -> dict[str, object]:
            received["recognition_threshold"] = recognition_threshold
            device = images.device
            role_logits = torch.zeros((1, 2, len(TEXT_ROLE_TO_ID)), device=device)
            return {
                "text": {
                    "mask": torch.ones((1, 2), dtype=torch.bool, device=device),
                    "boxes": torch.tensor([[
                        text_box(0.5, 0.25, 0.4, 0.08),
                        text_box(0.5, 0.65, 0.4, 0.08),
                    ]], device=device),
                    "scores": torch.tensor([[0.8, 0.95]], device=device),
                    "role_logits": role_logits,
                    "embeddings": torch.zeros((1, 2, model.dim), device=device),
                    "candidate_count": torch.tensor([2], device=device),
                    "capacity_overflow": torch.tensor([False], device=device),
                },
                "regions": {
                    "mask": torch.zeros((1, 1), dtype=torch.bool, device=device),
                    "boxes": torch.zeros((1, 1, 6), device=device),
                    "scores": torch.zeros((1, 1), device=device),
                    "kinds": torch.zeros((1, 1), dtype=torch.long, device=device),
                    "row_logits": torch.zeros((1, 1, 128), device=device),
                    "col_logits": torch.zeros((1, 1, 128), device=device),
                    "rowspan_logits": torch.zeros((1, 1, 32), device=device),
                    "colspan_logits": torch.zeros((1, 1, 32), device=device),
                    "header_logits": torch.zeros((1, 1), device=device),
                    "embeddings": torch.zeros((1, 1, model.dim), device=device),
                },
                "ocr_groups": {},
                "ocr_unsupported": ((0, 0, "overlong"),),
            }

        model.forward = types.MethodType(fake_forward, model)
        api = PrismOCR(
            model,
            vocabulary,
            calibration={
                **_calibration_provenance_fixture(),
                "scores_calibrated": True,
                "score_calibrators": {
                    "text": _calibrator_fixture(),
                    "region": _calibrator_fixture(),
                },
                "text_threshold": 0.949,
                "region_threshold": 0.5,
            },
        )
        result = api.predict(torch.zeros((3, 64, 64), dtype=torch.float32)).to_dict()

        self.assertEqual(model.text_threshold, 0.55)
        self.assertEqual(received["recognition_threshold"], 0.949)
        self.assertEqual([line["id"] for line in result["lines"]], ["l:2"])
        self.assertAlmostEqual(result["lines"][0]["detector_score"], 0.95, places=6)
        self.assertFalse(any(
            warning["code"] == "UNSUPPORTED_LINE_LENGTH" for warning in result["warnings"]
        ))

    def test_public_failures_are_structured_and_redacted(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        model = make_model(vocabulary, text_proposals=1, region_queries=1)
        api = PrismOCR(model, vocabulary)
        secret = "/private/customer/key.txt"

        def fail_forward(_self: PrismOCRModel, _images: torch.Tensor) -> dict[str, object]:
            raise RuntimeError(f"failed while opening {secret}")

        api.model.forward = types.MethodType(fail_forward, api.model)
        with self.assertRaises(InferenceError) as caught:
            api.predict(torch.zeros((3, 32, 32)))
        error = caught.exception
        self.assertEqual(error.code, "INFERENCE_ERROR")
        self.assertEqual(error.stage, "model_forward:tile=0")
        self.assertEqual(error.to_dict(), {
            "code": "INFERENCE_ERROR",
            "stage": "model_forward:tile=0",
            "message": "inference failed",
        })
        self.assertNotIn(secret, str(error))
        self.assertNotIn(secret, json.dumps(error.to_dict()))

        with self.assertRaises(InferenceError) as preprocessing_error:
            api.predict(object())
        self.assertEqual(preprocessing_error.exception.code, "INVALID_IMAGE")
        self.assertEqual(preprocessing_error.exception.stage, "preprocessing")
        self.assertEqual(preprocessing_error.exception.to_dict()["message"], "invalid image")
        self.assertNotIn(secret, str(preprocessing_error.exception))

    def test_graph_table_and_character_confidence_are_exposed_by_predict(self) -> None:
        vocabulary = Vocabulary.from_texts(["ab"], max_size=8)
        model = make_model(vocabulary, text_proposals=2, region_queries=2)
        char_a = vocabulary.tokens.index("a")
        char_b = vocabulary.tokens.index("b")

        def fake_forward(_self: PrismOCRModel, images: torch.Tensor) -> dict[str, object]:
            device = images.device
            text_boxes = torch.tensor([[
                text_box(0.5, 0.35, 0.30, 0.08),
                text_box(0.5, 0.55, 0.30, 0.08),
            ]], device=device)
            role_logits = torch.full((1, 2, len(TEXT_ROLE_TO_ID)), -5.0, device=device)
            role_logits[0, 0, TEXT_ROLE_TO_ID["KEY"]] = 5.0
            role_logits[0, 1, TEXT_ROLE_TO_ID["VALUE"]] = 5.0
            region_boxes = torch.tensor([[
                text_box(0.5, 0.5, 0.95, 0.85),
                text_box(0.5, 0.46, 0.80, 0.55),
            ]], device=device)
            ctc = torch.full((2, 64, len(vocabulary)), -10.0, device=device)
            ctc[:, :, vocabulary.blank_id] = 10.0
            ctc[0, 0, char_a] = 12.0
            ctc[1, 0, char_b] = 12.0
            return {
                "text": {
                    "mask": torch.ones((1, 2), dtype=torch.bool, device=device),
                    "boxes": text_boxes,
                    "scores": torch.tensor([[0.95, 0.94]], device=device),
                    "role_logits": role_logits,
                    "embeddings": torch.randn((1, 2, model.dim), device=device),
                    "candidate_count": torch.tensor([2], device=device),
                    "capacity_overflow": torch.tensor([False], device=device),
                },
                "regions": {
                    "mask": torch.ones((1, 2), dtype=torch.bool, device=device),
                    "boxes": region_boxes,
                    "scores": torch.tensor([[0.92, 0.91]], device=device),
                    "kinds": torch.tensor([[
                        REGION_KIND_TO_ID["TABLE"], REGION_KIND_TO_ID["CELL"]
                    ]], device=device),
                    "row_logits": torch.zeros((1, 2, 128), device=device),
                    "col_logits": torch.zeros((1, 2, 128), device=device),
                    "rowspan_logits": torch.zeros((1, 2, 32), device=device),
                    "colspan_logits": torch.zeros((1, 2, 32), device=device),
                    "header_logits": torch.full((1, 2), -10.0, device=device),
                    "embeddings": torch.randn((1, 2, model.dim), device=device),
                },
                "ocr_groups": {
                    64: {
                        "refs": torch.tensor([[0, 0], [0, 1]], device=device),
                        "logits": ctc,
                    }
                },
                "ocr_unsupported": (),
            }

        original_fake_forward = fake_forward

        def grid_forward(_self: PrismOCRModel, images: torch.Tensor) -> dict[str, object]:
            output = original_fake_forward(_self, images)
            region = output["regions"]
            region["row_logits"][0, 1, 2] = 10.0
            region["col_logits"][0, 1, 3] = 10.0
            region["rowspan_logits"][0, 1, 1] = 10.0
            region["colspan_logits"][0, 1, 2] = 10.0
            region["header_logits"][0, 1] = 10.0
            return output

        model.forward = types.MethodType(grid_forward, model)

        def fake_relations(_self: torch.nn.Module, nodes: torch.Tensor) -> torch.Tensor:
            count = nodes.shape[1]
            logits = torch.full((1, 5, count, count + 1), -20.0, device=nodes.device)
            for source in range(count):
                logits[0, :, source, -1] = 0.0
            # Stable graph node order is line 1, line 2, table, cell.
            logits[0, 0, 1, 0] = 10.0
            logits[0, 1, 0, 3] = 10.0
            logits[0, 1, 1, 3] = 10.0
            logits[0, 1, 2, -1] = 10.0
            logits[0, 1, 3, 2] = 10.0
            logits[0, 2, 0, 1] = 10.0
            return logits

        model.relation_head.forward = types.MethodType(fake_relations, model.relation_head)
        api = PrismOCR(
            model,
            vocabulary,
            calibration={
                **_calibration_provenance_fixture(),
                "scores_calibrated": True,
                "score_calibrators": {"text": _calibrator_fixture(), "region": _calibrator_fixture()},
                "relation_scores_calibrated": True,
                "relation_score_calibrators": {
                    name: _calibrator_fixture()
                    for name in ("NEXT", "PARENT", "KV_VALUE", "RIGHT_CELL")
                },
                "grid_scores_calibrated": True,
                "grid_score_calibrators": {
                    name: _calibrator_fixture()
                    for name in ("row", "col", "rowspan", "colspan", "header")
                },
                "grid_threshold": 0.5,
            },
        )
        result = api.predict(
            torch.zeros((3, 64, 64), dtype=torch.float32),
            InferenceOptions(emit_character_confidence=True, return_debug=True),
        ).to_dict()

        self.assertEqual([line["text"] for line in result["lines"]], ["a", "b"])
        self.assertEqual([line["reading_order"] for line in result["lines"]], [1, 0])
        self.assertEqual(result["lines"][0]["parent_id"], "c:1")
        self.assertEqual(result["cells"][0]["line_ids"], ["l:2", "l:1"])
        self.assertEqual(result["cells"][0]["text"], "b\na")
        self.assertEqual(result["tables"][0]["status"], "valid")
        self.assertEqual((result["cells"][0]["row"], result["cells"][0]["col"]), (2, 3))
        self.assertEqual((result["cells"][0]["rowspan"], result["cells"][0]["colspan"]), (2, 3))
        self.assertTrue(result["cells"][0]["is_header"])
        self.assertIn("<table", result["tables"][0]["html"])
        self.assertIn({"type": "KV_VALUE", "source": "l:1", "target": "l:2", "score": unittest.mock.ANY, "reason_codes": []}, result["relations"])
        self.assertEqual(result["kv_pairs"][0]["key"], "a")
        self.assertEqual(result["kv_pairs"][0]["value"], "b")
        self.assertEqual(len(result["lines"][0]["character_confidences"]), 1)
        self.assertEqual(result["diagnostics"]["tiles"], 1)
        self.assertEqual(result["reading_order"], ["l:2", "l:1"])

        normalized = api.predict(
            torch.zeros((3, 64, 64), dtype=torch.float32),
            InferenceOptions(output_coordinate_space="normalized"),
        ).to_dict()
        self.assertEqual(normalized["cells"][0]["line_ids"], result["cells"][0]["line_ids"])
        self.assertEqual(normalized["cells"][0]["text"], result["cells"][0]["text"])
        self.assertEqual(normalized["tables"][0]["html"], result["tables"][0]["html"])
        self.assertNotEqual(normalized["cells"][0]["polygon"], result["cells"][0]["polygon"])

    def test_calibrated_right_cell_affinity_guides_public_table_placement(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        model = make_model(vocabulary, text_proposals=1, region_queries=3)

        def fake_forward(_self: PrismOCRModel, images: torch.Tensor) -> dict[str, object]:
            device = images.device
            return {
                "text": {
                    "mask": torch.zeros((1, 1), dtype=torch.bool, device=device),
                    "boxes": torch.zeros((1, 1, 6), device=device),
                    "scores": torch.zeros((1, 1), device=device),
                    "role_logits": torch.zeros((1, 1, len(TEXT_ROLE_TO_ID)), device=device),
                    "embeddings": torch.zeros((1, 1, model.dim), device=device),
                    "candidate_count": torch.tensor([0], device=device),
                    "capacity_overflow": torch.tensor([False], device=device),
                },
                "regions": {
                    "mask": torch.ones((1, 3), dtype=torch.bool, device=device),
                    "boxes": torch.tensor([[
                        text_box(0.5, 0.5, 0.96, 0.9),
                        text_box(0.3, 0.5, 0.4, 0.7),
                        text_box(0.7, 0.5, 0.4, 0.7),
                    ]], device=device),
                    "scores": torch.full((1, 3), 0.95, device=device),
                    "kinds": torch.tensor([[
                        REGION_KIND_TO_ID["TABLE"],
                        REGION_KIND_TO_ID["CELL"],
                        REGION_KIND_TO_ID["CELL"],
                    ]], device=device),
                    "row_logits": torch.zeros((1, 3, 128), device=device),
                    "col_logits": torch.zeros((1, 3, 128), device=device),
                    "rowspan_logits": torch.zeros((1, 3, 32), device=device),
                    "colspan_logits": torch.zeros((1, 3, 32), device=device),
                    "header_logits": torch.full((1, 3), -10.0, device=device),
                    "embeddings": torch.zeros((1, 3, model.dim), device=device),
                },
                "ocr_groups": {},
                "ocr_unsupported": (),
            }

        def reversed_column_affinity(_self: torch.nn.Module, nodes: torch.Tensor) -> torch.Tensor:
            count = nodes.shape[1]
            logits = torch.full((1, 5, count, count + 1), -20.0, device=nodes.device)
            logits[:, :, :, -1] = 0.0
            # Graph order: table, left cell, right cell. The predicted columns
            # are intentionally reversed; RIGHT_CELL favors left -> right.
            logits[0, 3, 1, 2] = 10.0
            return logits

        model.forward = types.MethodType(fake_forward, model)
        model.relation_head.forward = types.MethodType(reversed_column_affinity, model.relation_head)
        result = PrismOCR(
            model,
            vocabulary,
            calibration={
                **_calibration_provenance_fixture(),
                "relation_scores_calibrated": True,
                "relation_score_calibrators": {
                    name: _calibrator_fixture()
                    for name in ("NEXT", "PARENT", "KV_VALUE", "RIGHT_CELL")
                },
                "grid_scores_calibrated": True,
                "grid_score_calibrators": {
                    name: _calibrator_fixture()
                    for name in ("row", "col", "rowspan", "colspan", "header")
                },
                "grid_threshold": 0.5,
            },
        ).predict(torch.zeros((3, 64, 64), dtype=torch.float32)).to_dict()

        self.assertEqual(result["tables"][0]["status"], "valid")
        self.assertFalse(any(
            warning["code"] == "TABLE_GRID_CONFLICT" for warning in result["warnings"]
        ))
        self.assertEqual([(cell["id"], cell["col"]) for cell in result["cells"]], [("c:1", 0), ("c:2", 1)])
        self.assertIn(
            {"type": "RIGHT_CELL", "source": "c:1", "target": "c:2", "score": unittest.mock.ANY, "reason_codes": []},
            result["relations"],
        )

    def test_public_inference_keeps_column_reading_order_scopes_independent(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        model = make_model(vocabulary, text_proposals=5, region_queries=1)
        line_boxes = (
            (0.5, 0.08, 0.18),  # centered heading, intentionally a singleton x cluster
            (0.25, 0.3, 0.20),
            (0.75, 0.3, 0.20),
            (0.25, 0.65, 0.20),
            (0.75, 0.65, 0.20),
        )

        def fake_forward(_self: PrismOCRModel, images: torch.Tensor) -> dict[str, object]:
            device = images.device
            boxes = torch.tensor([[
                text_box(cx, cy, width, 0.08) for cx, cy, width in line_boxes
            ]], device=device)
            logits = torch.full((len(line_boxes), 64, len(vocabulary)), -10.0, device=device)
            logits[:, :, vocabulary.blank_id] = 10.0
            return {
                "text": {
                    "mask": torch.ones((1, len(line_boxes)), dtype=torch.bool, device=device),
                    "boxes": boxes,
                    "scores": torch.full((1, len(line_boxes)), 0.9, device=device),
                    "role_logits": torch.zeros((1, len(line_boxes), len(TEXT_ROLE_TO_ID)), device=device),
                    "embeddings": torch.zeros((1, len(line_boxes), model.dim), device=device),
                    "candidate_count": torch.tensor([len(line_boxes)], device=device),
                    "capacity_overflow": torch.tensor([False], device=device),
                },
                "regions": {
                    "mask": torch.zeros((1, 1), dtype=torch.bool, device=device),
                    "boxes": torch.zeros((1, 1, 6), device=device),
                    "scores": torch.zeros((1, 1), device=device),
                    "kinds": torch.zeros((1, 1), dtype=torch.long, device=device),
                    "embeddings": torch.zeros((1, 1, model.dim), device=device),
                },
                "ocr_groups": {
                    64: {"refs": torch.tensor([[0, i] for i in range(len(line_boxes))], device=device), "logits": logits}
                },
                "ocr_unsupported": (),
            }

        model.forward = types.MethodType(fake_forward, model)

        def prefer_every_next_target(_self: torch.nn.Module, nodes: torch.Tensor) -> torch.Tensor:
            count = nodes.shape[1]
            logits = torch.full((1, 5, count, count + 1), -20.0, device=nodes.device)
            logits[:, :, :, -1] = 0.0
            logits[0, 0, :, :-1] = 10.0
            return logits

        model.relation_head.forward = types.MethodType(prefer_every_next_target, model.relation_head)
        result = PrismOCR(
            model,
            vocabulary,
            calibration={
                **_calibration_provenance_fixture(),
                "relation_scores_calibrated": True,
                "relation_score_calibrators": {
                    name: _calibrator_fixture()
                    for name in ("NEXT", "PARENT", "KV_VALUE", "RIGHT_CELL")
                },
            },
        ).predict(torch.zeros((3, 64, 128), dtype=torch.float32)).to_dict()

        self.assertEqual(len(result["lines"]), 5)
        scopes = {line["scope_id"] for line in result["lines"]}
        self.assertEqual(scopes, {"document", "column:000", "column:001"})
        first_line = min(result["lines"], key=lambda line: line["reading_order"])
        heading_y = sum(point[1] for point in first_line["polygon"]) / len(first_line["polygon"])
        self.assertLess(heading_y, 20.0)
        scope_by_id = {line["id"]: line["scope_id"] for line in result["lines"]}
        next_edges = [
            relation for relation in result["relations"]
            if relation["type"] == "NEXT" and relation["target"] is not None
        ]
        self.assertTrue(next_edges)
        self.assertTrue(all(scope_by_id[edge["source"]] == scope_by_id[edge["target"]] for edge in next_edges))
        self.assertTrue(any(
            item["reason_code"] == "READING_SCOPE_MISMATCH"
            for item in result["diagnostics"]["relation_suppressed"]
        ))

    def test_region_query_saturation_retries_with_smaller_tiles(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        model = make_model(vocabulary, text_proposals=1, region_queries=2)

        def fake_forward(_self: PrismOCRModel, images: torch.Tensor) -> dict[str, object]:
            device = images.device
            saturated = images.shape[-2] > 256 and images.shape[-1] > 256
            region_mask = torch.full((1, 2), saturated, dtype=torch.bool, device=device)
            boxes = torch.tensor([[
                text_box(0.3, 0.5, 0.2, 0.4), text_box(0.7, 0.5, 0.2, 0.4)
            ]], device=device)
            return {
                "text": {
                    "mask": torch.zeros((1, 1), dtype=torch.bool, device=device),
                    "boxes": torch.zeros((1, 1, 6), device=device),
                    "scores": torch.zeros((1, 1), device=device),
                    "role_logits": torch.zeros((1, 1, len(TEXT_ROLE_TO_ID)), device=device),
                    "embeddings": torch.zeros((1, 1, model.dim), device=device),
                    "candidate_count": torch.zeros((1,), dtype=torch.long, device=device),
                    "capacity_overflow": torch.zeros((1,), dtype=torch.bool, device=device),
                },
                "regions": {
                    "mask": region_mask,
                    "boxes": boxes,
                    "scores": torch.full((1, 2), 0.9, device=device),
                    "kinds": torch.full((1, 2), REGION_KIND_TO_ID["REGION"], device=device),
                    "embeddings": torch.zeros((1, 2, model.dim), device=device),
                },
                "ocr_groups": {},
                "ocr_unsupported": (),
            }

        model.forward = types.MethodType(fake_forward, model)
        result = PrismOCR(model, vocabulary).predict(
            torch.zeros((3, 512, 512), dtype=torch.float32)
        ).to_dict()

        self.assertEqual(result["diagnostics"]["tile_model_calls"], 10)
        self.assertFalse(result["diagnostics"]["structural_capacity_overflow"])
        self.assertTrue(any(
            warning["code"] == "CAPACITY_OVERFLOW_RETRIED_WITH_TILES"
            and warning["streams"] == ["structure"]
            for warning in result["warnings"]
        ))

    def test_terminal_structural_overflow_uses_canonical_capacity_warning(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        model = make_model(vocabulary, text_proposals=1, region_queries=2)

        def fake_forward(_self: PrismOCRModel, images: torch.Tensor) -> dict[str, object]:
            device = images.device
            return {
                "text": {
                    "mask": torch.zeros((1, 1), dtype=torch.bool, device=device),
                    "boxes": torch.zeros((1, 1, 6), device=device),
                    "scores": torch.zeros((1, 1), device=device),
                    "role_logits": torch.zeros((1, 1, len(TEXT_ROLE_TO_ID)), device=device),
                    "embeddings": torch.zeros((1, 1, model.dim), device=device),
                    "candidate_count": torch.zeros((1,), dtype=torch.long, device=device),
                    "capacity_overflow": torch.zeros((1,), dtype=torch.bool, device=device),
                },
                "regions": {
                    "mask": torch.ones((1, 2), dtype=torch.bool, device=device),
                    "boxes": torch.tensor([[
                        text_box(0.3, 0.5, 0.2, 0.4), text_box(0.7, 0.5, 0.2, 0.4)
                    ]], device=device),
                    "scores": torch.full((1, 2), 0.9, device=device),
                    "kinds": torch.full((1, 2), REGION_KIND_TO_ID["REGION"], device=device),
                    "embeddings": torch.zeros((1, 2, model.dim), device=device),
                },
                "ocr_groups": {},
                "ocr_unsupported": (),
            }

        model.forward = types.MethodType(fake_forward, model)
        result = PrismOCR(model, vocabulary).predict(
            torch.zeros((3, 64, 64), dtype=torch.float32),
            InferenceOptions(allow_tiling=False),
        ).to_dict()

        warning_codes = {warning["code"] for warning in result["warnings"]}
        self.assertTrue(result["diagnostics"]["structural_capacity_overflow"])
        self.assertTrue(result["diagnostics"]["overflow"])
        self.assertIn("CAPACITY_OVERFLOW", warning_codes)
        self.assertIn("STRUCTURE_CAPACITY_OVERFLOW", warning_codes)

    def test_large_page_is_tiled_and_boundary_duplicate_is_stitched(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        model = make_model(vocabulary, text_proposals=1, region_queries=1)
        char_a = vocabulary.tokens.index("a")

        def fake_forward(_self: PrismOCRModel, images: torch.Tensor) -> dict[str, object]:
            device = images.device
            width = images.shape[-1]
            center_x = 960.0 if width > 1000 else 64.0
            text_box_value = torch.tensor([[
                text_box(center_x / width, 0.5, 160.0 / width, 10.0 / images.shape[-2])
            ]], device=device)
            logits = torch.full((1, 64, len(vocabulary)), -10.0, device=device)
            logits[:, :, vocabulary.blank_id] = 10.0
            logits[:, 0, char_a] = 12.0
            return {
                "text": {
                    "mask": torch.ones((1, 1), dtype=torch.bool, device=device),
                    "boxes": text_box_value,
                    "scores": torch.tensor([[0.9]], device=device),
                    "role_logits": torch.zeros((1, 1, len(TEXT_ROLE_TO_ID)), device=device),
                    "embeddings": torch.zeros((1, 1, model.dim), device=device),
                    "candidate_count": torch.tensor([1], device=device),
                    "capacity_overflow": torch.tensor([False], device=device),
                },
                "regions": {
                    "mask": torch.zeros((1, 1), dtype=torch.bool, device=device),
                    "boxes": torch.zeros((1, 1, 6), device=device),
                    "scores": torch.zeros((1, 1), device=device),
                    "kinds": torch.zeros((1, 1), dtype=torch.long, device=device),
                    "embeddings": torch.zeros((1, 1, model.dim), device=device),
                },
                "ocr_groups": {64: {"refs": torch.tensor([[0, 0]], device=device), "logits": logits}},
                "ocr_unsupported": (),
            }

        model.forward = types.MethodType(fake_forward, model)
        api = PrismOCR(model, vocabulary)
        result = api.predict(
            torch.zeros((3, 64, 1800), dtype=torch.float32),
            InferenceOptions(return_debug=True),
        ).to_dict()
        self.assertEqual(len(result["lines"]), 1)
        self.assertEqual(result["lines"][0]["text"], "a")
        self.assertEqual(result["lines"][0]["source_tiles"], [0, 1])
        self.assertEqual(result["diagnostics"]["tiles"], 2)
        self.assertTrue(any(warning["code"] == "TILING_USED" for warning in result["warnings"]))

    def test_text_overflow_retry_preserves_complete_parent_structure(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        model = make_model(vocabulary, text_proposals=1, region_queries=2)

        def fake_forward(_self: PrismOCRModel, images: torch.Tensor) -> dict[str, object]:
            device = images.device
            parent = images.shape[-2] > 256 or images.shape[-1] > 256
            return {
                "text": {
                    "mask": torch.tensor([[parent]], dtype=torch.bool, device=device),
                    "boxes": torch.tensor([[text_box(0.5, 0.5, 0.3, 0.08)]], device=device),
                    "scores": torch.tensor([[0.9 if parent else 0.0]], device=device),
                    "role_logits": torch.zeros((1, 1, len(TEXT_ROLE_TO_ID)), device=device),
                    "embeddings": torch.zeros((1, 1, model.dim), device=device),
                    "candidate_count": torch.tensor([2 if parent else 0], device=device),
                    "capacity_overflow": torch.tensor([parent], device=device),
                },
                "regions": {
                    "mask": torch.tensor([[parent, False]], dtype=torch.bool, device=device),
                    "boxes": torch.tensor([[
                        text_box(0.5, 0.5, 0.8, 0.8), text_box(0.2, 0.2, 0.1, 0.1)
                    ]], device=device),
                    "scores": torch.tensor([[0.95, 0.0]], device=device),
                    "kinds": torch.tensor([[
                        REGION_KIND_TO_ID["TABLE"], REGION_KIND_TO_ID["REGION"]
                    ]], device=device),
                    "embeddings": torch.zeros((1, 2, model.dim), device=device),
                },
                "ocr_groups": {},
                "ocr_unsupported": (),
            }

        model.forward = types.MethodType(fake_forward, model)
        result = PrismOCR(model, vocabulary).predict(torch.zeros((3, 512, 512))).to_dict()

        self.assertEqual([region["kind"] for region in result["regions"]], ["TABLE"])
        retry = next(
            warning for warning in result["warnings"]
            if warning["code"] == "CAPACITY_OVERFLOW_RETRIED_WITH_TILES"
        )
        self.assertEqual(retry["streams"], ["text"])

    def test_capacity_overflow_retries_with_smaller_tiles(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        model = make_model(vocabulary, text_proposals=1, region_queries=1)

        def fake_forward(_self: PrismOCRModel, images: torch.Tensor) -> dict[str, object]:
            device = images.device
            width = images.shape[-1]
            large = width > 256
            return {
                "text": {
                    "mask": torch.tensor([[large]], dtype=torch.bool, device=device),
                    "boxes": torch.tensor([[text_box(0.5, 0.5, 0.2, 0.1)]], device=device),
                    "scores": torch.tensor([[0.9 if large else 0.0]], device=device),
                    "role_logits": torch.zeros((1, 1, len(TEXT_ROLE_TO_ID)), device=device),
                    "embeddings": torch.zeros((1, 1, model.dim), device=device),
                    "candidate_count": torch.tensor([2 if large else 0], device=device),
                    "capacity_overflow": torch.tensor([large], device=device),
                },
                "regions": {
                    "mask": torch.zeros((1, 1), dtype=torch.bool, device=device),
                    "boxes": torch.zeros((1, 1, 6), device=device),
                    "scores": torch.zeros((1, 1), device=device),
                    "kinds": torch.zeros((1, 1), dtype=torch.long, device=device),
                    "embeddings": torch.zeros((1, 1, model.dim), device=device),
                },
                "ocr_groups": {},
                "ocr_unsupported": (),
            }

        model.forward = types.MethodType(fake_forward, model)
        result = PrismOCR(model, vocabulary).predict(torch.zeros((3, 512, 512))).to_dict()
        self.assertEqual(result["lines"], [])
        self.assertFalse(result["diagnostics"]["overflow"])
        self.assertEqual(result["diagnostics"]["tile_model_calls"], 10)
        self.assertTrue(any(warning["code"] == "CAPACITY_OVERFLOW_RETRIED_WITH_TILES" for warning in result["warnings"]))

    def test_calibration_must_be_bound_to_the_checkpoint(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        model = make_model(vocabulary, text_proposals=1, region_queries=1)
        with tempfile.TemporaryDirectory() as tmp:
            save_artifact(tmp, model, vocabulary)
            PrismOCR(model, vocabulary)
            self.assertIsNone(model.backbone.stage4[0].dw.fused_conv)
            self.assertTrue(hasattr(model.backbone.stage4[0].dw, "dw_3x3"))
            PrismOCR.load(tmp)
            manifest_path = Path(tmp) / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            def write_calibration(value: dict[str, object]) -> None:
                calibration_bytes = json.dumps(value).encode("utf-8")
                (Path(tmp) / "calibration-v4.json").write_bytes(calibration_bytes)
                manifest["calibration_sha256"] = hashlib.sha256(calibration_bytes).hexdigest()
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            identity_calibrators = {
                "text": _calibrator_fixture(),
                "region": _calibrator_fixture(),
            }
            calibration_provenance = _calibration_provenance_fixture()
            unbound_calibration = {
                **calibration_provenance,
                "scores_calibrated": True,
                "score_calibrators": identity_calibrators,
                "text_threshold": 0.9,
            }
            write_calibration(unbound_calibration)
            with self.assertRaisesRegex(ValueError, "matching model_sha256"):
                PrismOCR.load(tmp)

            bound_calibration = {
                **calibration_provenance,
                "model_sha256": manifest["model_sha256"],
                "model_config_sha256": manifest["model_config_sha256"],
                "vocabulary_sha256": manifest["vocabulary_sha256"],
                "scores_calibrated": True,
                "score_calibrators": identity_calibrators,
                "relation_scores_calibrated": True,
                "relation_score_calibrators": {
                    name: _calibrator_fixture()
                    for name in ("NEXT", "PARENT", "KV_VALUE", "RIGHT_CELL")
                },
                "grid_scores_calibrated": True,
                "grid_score_calibrators": {
                    name: _calibrator_fixture()
                    for name in ("row", "col", "rowspan", "colspan", "header")
                },
                "text_threshold": 0.7,
                "region_threshold": 0.6,
                "grid_threshold": 0.8,
                "header_threshold": 0.4,
            }
            write_calibration(bound_calibration)
            loaded = PrismOCR.load(tmp)
            self.assertEqual(loaded.model.text_threshold, 0.55)
            self.assertEqual(loaded.model.region_threshold, 0.5)
            self.assertEqual(loaded.calibration["text_threshold"], 0.7)
            self.assertEqual(loaded.calibration["region_threshold"], 0.6)
            self.assertEqual(loaded.calibration["grid_threshold"], 0.8)
            self.assertTrue(loaded.calibration["grid_scores_calibrated"])
            config_path = Path(tmp) / "model-config.json"
            original_config = config_path.read_bytes()
            changed_config = json.loads(original_config)
            changed_config["text_proposals"] = 2
            changed_bytes = json.dumps(changed_config, indent=2).encode("utf-8") + b"\n"
            config_path.write_bytes(changed_bytes)
            manifest["model_config_sha256"] = hashlib.sha256(changed_bytes).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match model configuration"):
                PrismOCR.load(tmp)
            config_path.write_bytes(original_config)
            manifest["model_config_sha256"] = hashlib.sha256(original_config).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            bound_calibration["grid_scores_calibrated"] = "false"
            write_calibration(bound_calibration)
            with self.assertRaisesRegex(ValueError, "grid_scores_calibrated must be a boolean"):
                PrismOCR.load(tmp)

        with self.assertRaisesRegex(ValueError, "grid_scores_calibrated must be a boolean"):
            PrismOCR(model, vocabulary, calibration={"grid_scores_calibrated": "false"})

    def test_exif_orientation_tag_6_transposes_to_displayed_frame(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        model = make_model(vocabulary, text_proposals=1, region_queries=1)
        api = PrismOCR(model, vocabulary)

        # Raw size: width=30, height=40.
        # Orientation 6 specifies a 90-degree CCW rotation, so displayed frame has width=40, height=30.
        raw = Image.new("RGB", (30, 40), (255, 255, 255))
        exif = Image.Exif()
        exif[0x0112] = 6
        raw._getexif = lambda: dict(exif)  # in-memory EXIF attachment
        raw.getexif = lambda: exif

        tensor, width, height = PrismOCR._tensor_image(raw)
        self.assertEqual((width, height), (40, 30))
        self.assertEqual(tuple(tensor.shape), (3, 30, 40))

        result = api.predict(raw, InferenceOptions(return_debug=True)).to_dict()
        self.assertEqual((result["image"]["width"], result["image"]["height"]), (40, 30))



if __name__ == "__main__":
    unittest.main()
