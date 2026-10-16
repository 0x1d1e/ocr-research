from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from prismocr.api import InferenceError, save_artifact
from prismocr.data.vocabulary import Vocabulary
from prismocr.model.network import PrismOCRModel
from prismocr.onnx_export import (
    ONNXExportError,
    ONNX_OPSET,
    CORE_OUTPUT_NAMES,
    PrismONNXRuntime,
    export_artifact,
    export_graphs,
    decode_onnx_outputs,
)
from prismocr.onnx_parity import assert_decoded_document_parity

_HAS_ONNX_EXPORT_DEPS = all(
    importlib.util.find_spec(name) is not None for name in ("onnx", "onnxruntime")
)


class ExportableHeadTests(unittest.TestCase):
    def test_export_refuses_source_directory_without_removing_its_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            manifest = source / "manifest.json"
            manifest.write_text('{"source": true}\n', encoding="utf-8")

            with self.assertRaisesRegex(ONNXExportError, "must differ from the source artifact"):
                export_graphs(source, source)

            self.assertEqual(manifest.read_text(encoding="utf-8"), '{"source": true}\n')

    def test_decoded_document_parity_uses_numeric_polygon_tolerance(self) -> None:
        polygon = [[1.0, 2.0], [8.0, 2.0], [8.0, 9.0], [1.0, 9.0]]
        pytorch = {
            "image": {"width": 10, "height": 12},
            "lines": [],
            "regions": [{"id": "r:1", "kind": "TABLE", "polygon": polygon, "parent_id": None}],
            "cells": [], "tables": [], "relations": [], "reading_order": [],
            "warnings": [], "diagnostics": {}, "kv_pairs": [],
        }
        onnx = {
            "annotation": {
                "image": {"width": 10, "height": 12},
                "lines": [],
                "regions": [{"id": "r:1", "kind": "TABLE", "polygon_px": [
                    [1.0001, 2.0], [8.0, 2.0], [8.0, 9.0], [1.0, 9.0],
                ], "parent_id": None}],
                "cells": [], "relations": [], "reading_order": [],
            },
            "table_decode": {"tables": []}, "graph": {"predictions": []},
            "warnings": [], "diagnostics": {}, "kv_pairs": [],
        }
        assert_decoded_document_parity(pytorch, onnx, label="numeric-tolerance", atol=2e-4, rtol=2e-3)
        onnx["annotation"]["regions"][0]["kind"] = "FORM_FIELD"
        with self.assertRaisesRegex(ONNXExportError, "semantic fields differ: \\['regions'\\]"):
            assert_decoded_document_parity(pytorch, onnx, label="semantic-mismatch", atol=2e-4, rtol=2e-3)

    def test_text_proposal_pads_when_spatial_sites_are_below_capacity(self) -> None:
        from prismocr.model.heads import TextLineHead

        heatmap_logits = torch.full((1, 1, 2, 2), -10.0)
        heatmap_logits[0, 0, 0, 0] = 3.0
        raw = {
            "heatmap_logits": heatmap_logits,
            "box_map": torch.zeros(1, 6, 2, 2),
            "role_map": torch.zeros(1, 2, 2, 2),
            "feature_map": torch.zeros(1, 10, 2, 2),
        }
        proposals = TextLineHead.proposals(raw, max_proposals=8, score_threshold=0.5)
        self.assertEqual(tuple(proposals["boxes"].shape), (1, 8, 6))
        self.assertEqual(int(proposals["candidate_mask"].sum()), 1)
        self.assertEqual(proposals["proposal_indices"].shape, (1, 8))
        self.assertTrue(torch.all(proposals["proposal_indices"] == 0))

    def test_decoded_document_parity_checks_confidence_warnings_and_relation_reasons(self) -> None:
        polygon = [[1.0, 2.0], [8.0, 2.0], [8.0, 9.0], [1.0, 9.0]]
        line = {
            "id": "l:1", "polygon": polygon, "text": "x", "detector_score": 0.95,
            "role": "PARA_LINE", "oov_positions": [],
        }
        onnx_line = {**line, "polygon_px": polygon}
        onnx_line.pop("polygon")
        relation = {
            "type": "NEXT", "source": "l:1", "target": None,
            "score": 0.7, "reason_codes": ["NULL_SELECTED"],
        }
        pytorch = {
            "image": {"width": 10, "height": 12},
            "lines": [line], "regions": [], "cells": [], "tables": [],
            "relations": [relation], "reading_order": ["l:1"],
            "warnings": [{"code": "UNSUPPORTED_LINE_LENGTH", "line_id": "l:1", "reason": "long"}],
            "diagnostics": {
                "relation_suppressed": [{
                    "type": "NEXT", "source": "l:1", "target": "l:2", "reason_code": "LOWER_SCORE",
                }],
            },
            "kv_pairs": [],
        }
        onnx = {
            "annotation": {
                "image": {"width": 10, "height": 12},
                "lines": [onnx_line], "regions": [], "cells": [],
                "relations": [{"type": "NEXT", "source": "l:1", "target": None}],
                "reading_order": ["l:1"],
            },
            "table_decode": {"tables": []},
            "graph": {"predictions": [{
                "relation": "NEXT", "source_id": "l:1", "target_id": None,
                "score": 0.7, "reason_codes": ["NULL_SELECTED"],
            }], "suppressed": []},
            "warnings": [{"code": "UNSUPPORTED_LINE_LENGTH", "line_id": "l:1", "reason": "long"}],
            "diagnostics": {
                "relation_suppressed": [{
                    "type": "NEXT", "source": "l:1", "target": "l:2", "reason_code": "LOWER_SCORE",
                }],
            },
            "kv_pairs": [],
        }
        assert_decoded_document_parity(pytorch, onnx, label="metadata", atol=2e-4, rtol=2e-3)

        onnx["annotation"]["lines"][0]["detector_score"] = None
        with self.assertRaisesRegex(ONNXExportError, "semantic fields differ: \\['lines'\\]"):
            assert_decoded_document_parity(pytorch, onnx, label="confidence", atol=2e-4, rtol=2e-3)
        onnx["annotation"]["lines"][0]["detector_score"] = 0.95
        onnx["warnings"][0]["line_id"] = "l:999"
        with self.assertRaisesRegex(ONNXExportError, "semantic fields differ: \\['warnings'\\]"):
            assert_decoded_document_parity(pytorch, onnx, label="warning-target", atol=2e-4, rtol=2e-3)
        onnx["warnings"][0]["line_id"] = "l:1"
        onnx["graph"]["predictions"][0]["reason_codes"] = ["NEXT_CYCLE_BROKEN"]
        with self.assertRaisesRegex(ONNXExportError, "semantic fields differ: \\['relations'\\]"):
            assert_decoded_document_parity(pytorch, onnx, label="relation-reason", atol=2e-4, rtol=2e-3)
        onnx["graph"]["predictions"][0]["reason_codes"] = ["NULL_SELECTED"]
        onnx["diagnostics"]["relation_suppressed"][0]["reason_code"] = "INVALID_ENDPOINT_TYPES"
        with self.assertRaisesRegex(ONNXExportError, "semantic fields differ: \\['suppressed_relations'\\]"):
            assert_decoded_document_parity(pytorch, onnx, label="suppression-reason", atol=2e-4, rtol=2e-3)

    def test_parity_rejects_identically_nonfinite_tensors(self) -> None:
        from prismocr.onnx_export import _assert_close

        with self.assertRaisesRegex(ONNXExportError, "non-finite"):
            _assert_close(np.array([np.nan], dtype=np.float32), np.array([np.nan], dtype=np.float32), label="nan")

    def test_text_proposal_equal_score_order_uses_flattened_indices(self) -> None:
        from prismocr.model.heads import TextLineHead

        raw = {
            "heatmap_logits": torch.tensor([[[[3.0, 0.0, 3.0, 0.0]]]]),
            "box_map": torch.zeros(1, 6, 1, 4),
            "role_map": torch.zeros(1, 2, 1, 4),
            "feature_map": torch.zeros(1, 10, 1, 4),
        }
        proposals = TextLineHead.proposals(raw, max_proposals=2, score_threshold=0.0)
        self.assertEqual(proposals["proposal_indices"].tolist(), [[0, 2]])

    def test_dynamic_memory_attention_matches_standard_decoder(self) -> None:
        from prismocr.model.heads import RegionQueryHead

        torch.manual_seed(17)
        head = RegionQueryHead(dim=10, num_queries=3, num_kinds=6).eval()
        features = torch.randn(1, 10, 4, 5)
        with torch.inference_mode():
            actual = head(features, score_threshold=0.0)["embeddings"]
            _, _, height, width = features.shape
            ys = torch.arange(height, dtype=features.dtype) / max(height - 1, 1)
            xs = torch.arange(width, dtype=features.dtype) / max(width - 1, 1)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            position = head.memory_position(torch.stack((xx, yy), dim=-1).reshape(-1, 2))
            memory = features.flatten(2).transpose(1, 2) + position.unsqueeze(0)
            queries = head.query.weight.unsqueeze(0)
            expected = head.decoder(queries, memory)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@unittest.skipUnless(
    _HAS_ONNX_EXPORT_DEPS,
    "install PrismOCR with the export extra to run ONNX CPU parity tests",
)
class ONNXExportTest(unittest.TestCase):
    contract_version = 1

    @classmethod
    def setUpClass(cls) -> None:
        cls.old_num_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        torch.manual_seed(731)
        cls.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(cls.temporary_directory.name)
        cls.artifact_dir = root / "pytorch"
        cls.export_dir = root / "onnx"
        vocabulary = Vocabulary.from_texts(["abc"])
        model = PrismOCRModel(
            model_contract_version=cls.contract_version,
            dim=10,
            text_proposals=32,
            region_queries=3,
            vocab_size=len(vocabulary),
            vocabulary=vocabulary,
            text_threshold=0.0,
            region_threshold=0.15,
        )
        save_artifact(cls.artifact_dir, model, vocabulary)
        cls.manifest = export_artifact(
            cls.artifact_dir,
            cls.export_dir,
            image_sizes=((16, 16), (64, 80), (96, 128)),
        )
        cls.runtime = PrismONNXRuntime(cls.export_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()
        torch.set_num_threads(cls.old_num_threads)

    def test_full_model_parity_manifest_and_empty_buckets(self) -> None:
        self.assertEqual(self.manifest["opset"], ONNX_OPSET)
        self.assertEqual(self.manifest["runtime"]["execution_provider"], "CPUExecutionProvider")
        self.assertEqual(
            self.manifest["runtime_matrix"],
            {
                "opset": ONNX_OPSET,
                "onnx": "1.17.0",
                "onnxruntime": "1.24.3",
                "execution_provider": "CPUExecutionProvider",
            },
        )
        self.assertEqual(
            set(self.manifest["versions"]), {"prismocr", "torch", "onnx", "onnxruntime"}
        )
        source = self.manifest["source"]
        for key in ("model_sha256", "vocabulary_sha256", "config_sha256"):
            self.assertEqual(len(source[key]), 64)
        calibration_sha256 = source["calibration_sha256"]
        self.assertTrue(calibration_sha256 is None or len(calibration_sha256) == 64)
        self.assertEqual(set(self.manifest["graphs"]["recognizers"]), {"64", "128", "160", "256"})
        self.assertEqual(
            self.manifest["parity"]["image_sizes_hw"],
            [[16, 16], [64, 80], [96, 128]],
        )
        parity = self.manifest["parity"]
        self.assertEqual(parity["status"], "passed")
        self.assertTrue(parity["empty_recognizer_inputs_checked"])
        self.assertEqual(parity["core_images_checked"], 3)
        fusion = parity["fusion_parity"]
        self.assertEqual(fusion["status"], "passed")
        self.assertEqual(
            [case["precision"] for case in fusion["cases"]], ["float32", "float16"]
        )
        self.assertTrue(all(case["status"] == "passed" for case in fusion["cases"]))
        self.assertTrue(all(case["image_sizes_hw"] == [[16, 16], [64, 80], [96, 128]] for case in fusion["cases"]))
        cases = parity["recognizer_cases"]
        for bucket in (64, 128, 160, 256):
            self.assertIn({"bucket": bucket, "proposal_count": 2}, cases)
            self.assertIn({"bucket": bucket, "proposal_count": 3}, cases)
            self.assertIn({"bucket": bucket, "proposal_count": 0}, cases)
        self.assertEqual(json.loads((self.export_dir / "manifest.json").read_text())["parity"], parity)

    def test_model_contract_manifest_dispatch_and_legacy_format_read(self) -> None:
        self.assertEqual(self.manifest["format_version"], 2)
        self.assertEqual(self.manifest["model"]["config"]["model_contract_version"], self.contract_version)
        path = self.export_dir / "manifest.json"
        original = path.read_bytes()
        try:
            if self.contract_version == 1:
                legacy = json.loads(original)
                legacy["format_version"] = 1
                del legacy["model"]["config"]["model_contract_version"]
                path.write_text(json.dumps(legacy))
                runtime = PrismONNXRuntime(self.export_dir)
                self.assertEqual(runtime.model_contract_version, 1)
                self.assertIn("core", runtime.run_tensors(np.zeros((1, 3, 32, 48), dtype=np.float32)))
            for format_version, contract in ((2, None), (2, 3), (2, True), (1, 2), (True, 1), (3, 1)):
                malformed = json.loads(original)
                malformed["format_version"] = format_version
                if contract is None:
                    del malformed["model"]["config"]["model_contract_version"]
                else:
                    malformed["model"]["config"]["model_contract_version"] = contract
                path.write_text(json.dumps(malformed))
                with self.assertRaisesRegex(ONNXExportError, "version|contract"):
                    PrismONNXRuntime(self.export_dir)
        finally:
            path.write_bytes(original)

    def test_contract2_host_uses_max_bucket_even_when_legacy_geometry_rejects(self) -> None:
        if self.contract_version != 2:
            return
        image = np.zeros((1, 3, 32, 48), dtype=np.float32)
        values = self.runtime.core_session.run(list(CORE_OUTPUT_NAMES), {"images": image})
        core = dict(zip(CORE_OUTPUT_NAMES, values))
        core["text_mask"][:] = False
        core["text_mask"][0, 0] = True
        core["text_boxes"][0, 0] = [0.5, 0.5, 100, 0.0001, 0, 1]
        session = SimpleNamespace(run=lambda names, _inputs: [core[name] for name in names])
        with patch.object(self.runtime, "core_session", session):
            result = self.runtime.run_tensors(image)
        self.assertEqual(set(result["ocr_groups"]), {256})
        self.assertEqual(result["ocr_unsupported"], ())

    def test_runtime_rejects_same_size_external_weight_corruption(self) -> None:
        weights_path = self.export_dir / "model-weights.bin"
        original = weights_path.read_bytes()
        self.assertTrue(original)
        corrupted = bytes([original[0] ^ 1]) + original[1:]
        weights_path.write_bytes(corrupted)
        try:
            with self.assertRaisesRegex(ONNXExportError, "external tensor data.*checksum"):
                PrismONNXRuntime(self.export_dir)
        finally:
            weights_path.write_bytes(original)

    def test_runtime_rejects_artifacts_without_completed_parity(self) -> None:
        manifest_path = self.export_dir / "manifest.json"
        original = manifest_path.read_bytes()
        manifest = json.loads(original)
        manifest.pop("parity")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        try:
            with self.assertRaisesRegex(ONNXExportError, "no completed parity verification"):
                PrismONNXRuntime(self.export_dir)
        finally:
            manifest_path.write_bytes(original)

    def test_runtime_decoding_emits_schema_valid_hashed_annotations(self) -> None:
        images = np.zeros((1, 3, 64, 80), dtype=np.float32)
        result = self.runtime.predict(images)
        annotation = result["annotation"]
        self.assertTrue(result["schema_valid"], result["schema_issues"])
        self.assertEqual(
            annotation["image"]["sha256"], hashlib.sha256(images.tobytes()).hexdigest()
        )
        self.assertEqual(
            annotation["provenance"]["input_hash_kind"],
            "normalized_nchw_float32_tensor_sha256",
        )
        self.assertEqual(result["kv_pairs"], [])
        self.assertEqual(result["artifact_identity"], {
            key: self.manifest["source"][key]
            for key in ("model_sha256", "vocabulary_sha256", "calibration_sha256")
        })
        self.assertIn("region_row_logits", self.manifest["graphs"]["core"]["outputs"])

    def test_onnx_proposals_recover_long_dense_center_chains_and_overflow(self) -> None:
        from prismocr.model.heads import TextLineHead

        class ProposalModule(torch.nn.Module):
            def forward(self, heatmap, boxes, roles, features):
                proposals = TextLineHead.proposals(
                    {
                        "heatmap_logits": heatmap,
                        "box_map": boxes,
                        "role_map": roles,
                        "feature_map": features,
                    },
                    max_proposals=8,
                    score_threshold=0.55,
                )
                return (
                    proposals["proposal_indices"],
                    proposals["candidate_count"],
                    proposals["capacity_overflow"],
                    proposals["mask"],
                )

        def inputs_for(line_count: int) -> tuple[torch.Tensor, ...]:
            height = line_count + 4
            heatmap = torch.full((1, 1, height, 3), -10.0)
            heatmap[0, 0, 1 : line_count + 1, 1] = 3.0
            boxes = torch.zeros(1, 6, height, 3)
            boxes[:, 3] = -3.0
            return (
                heatmap,
                boxes,
                torch.zeros(1, 2, height, 3),
                torch.zeros(1, 10, height, 3),
            )

        def plateau_inputs() -> tuple[torch.Tensor, ...]:
            heatmap = torch.full((1, 1, 24, 24), -10.0)
            heatmap[:, :, 2:22, 2:22] = 3.0
            boxes = torch.zeros(1, 6, 24, 24)
            boxes[:, 2:4] = -3.0
            return (
                heatmap,
                boxes,
                torch.zeros(1, 2, 24, 24),
                torch.zeros(1, 10, 24, 24),
            )

        def duplicate_shoulder_inputs() -> tuple[torch.Tensor, ...]:
            heatmap = torch.full((1, 1, 6, 3), -10.0)
            heatmap[0, 0, 1, 1] = 4.0
            heatmap[0, 0, 2, 1] = 3.0
            heatmap[0, 0, 3, 1] = 2.0
            boxes = torch.zeros(1, 6, 6, 3)
            boxes[0, 1, 1, 1] = 8.0
            boxes[0, 1, 2, 1] = -8.0
            boxes[0, 3, 1, 1] = -1.386
            boxes[0, 3, 2, 1] = 2.087
            boxes[0, 3, 3, 1] = 2.087
            return (
                heatmap,
                boxes,
                torch.zeros(1, 2, 6, 3),
                torch.zeros(1, 10, 6, 3),
            )

        module = ProposalModule().eval()
        with tempfile.TemporaryDirectory() as temporary:
            onnx_path = Path(temporary) / "dense-proposals.onnx"
            torch.onnx.export(
                module,
                inputs_for(20),
                onnx_path,
                opset_version=ONNX_OPSET,
                input_names=["heatmap", "boxes", "roles", "features"],
                output_names=["indices", "count", "overflow", "mask"],
                dynamic_axes={
                    name: {2: f"{name}_height", 3: f"{name}_width"}
                    for name in ("heatmap", "boxes", "roles", "features")
                },
                dynamo=False,
            )
            session = __import__("onnxruntime").InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
            cases = [
                (inputs_for(20), 20, True),
                (inputs_for(40), 40, True),
                (plateau_inputs(), 400, True),
                (duplicate_shoulder_inputs(), 2, False),
            ]
            accepted_inputs = {item.name for item in session.get_inputs()}
            for raw_inputs, expected_count, expected_overflow in cases:
                reference = module(*raw_inputs)
                actual = session.run(None, {
                    name: tensor.numpy()
                    for name, tensor in zip(("heatmap", "boxes", "roles", "features"), raw_inputs)
                    if name in accepted_inputs
                })
                for expected, observed in zip(reference, actual):
                    np.testing.assert_array_equal(expected.detach().numpy(), observed)
                self.assertEqual(int(actual[1][0]), expected_count)
                self.assertEqual(bool(actual[2][0]), expected_overflow)

    def test_calibrated_operating_threshold_filters_after_stitching(self) -> None:
        model_config = self.manifest["model"]["config"]
        text_count = int(model_config["text_proposals"])
        region_count = int(model_config["region_queries"])
        node_count = text_count + region_count
        text_roles = self.manifest["model"]["text_roles"]
        region_kinds = self.manifest["model"]["region_kinds"]
        core = {
            "glyph_features": np.zeros((1, int(model_config["dim"]), 4, 4), dtype=np.float32),
            "text_mask": np.zeros((1, text_count), dtype=np.bool_),
            "text_boxes": np.zeros((1, text_count, 6), dtype=np.float32),
            "text_scores": np.zeros((1, text_count), dtype=np.float32),
            "text_role_logits": np.zeros((1, text_count, len(text_roles)), dtype=np.float32),
            "text_candidate_count": np.array([2], dtype=np.int64),
            "text_capacity_overflow": np.array([False]),
            "region_mask": np.zeros((1, region_count), dtype=np.bool_),
            "region_boxes": np.zeros((1, region_count, 6), dtype=np.float32),
            "region_scores": np.zeros((1, region_count), dtype=np.float32),
            "region_kinds": np.zeros((1, region_count), dtype=np.int64),
            "graph_logits": np.zeros((1, 5, node_count, node_count + 1), dtype=np.float32),
        }
        core["text_mask"][0, :2] = True
        core["text_boxes"][0, 0] = [0.5, 0.25, 0.4, 0.08, 0.0, 1.0]
        core["text_boxes"][0, 1] = [0.5, 0.65, 0.4, 0.08, 0.0, 1.0]
        core["text_scores"][0, :2] = [0.8, 0.95]
        core["region_mask"][0, :2] = True
        core["region_boxes"][0, 0] = [0.5, 0.5, 0.9, 0.9, 0.0, 1.0]
        core["region_boxes"][0, 1] = [0.5, 0.5, 0.2, 0.2, 0.0, 1.0]
        core["region_scores"][0, :2] = [0.95, 0.8]
        core["region_kinds"][0, 0] = region_kinds.index("TABLE")
        core["region_kinds"][0, 1] = region_kinds.index("REGION")
        manifest = json.loads(json.dumps(self.manifest))
        manifest["calibration"]["settings"].update({
            "scores_calibrated": True,
            "score_calibrators": {
                "text": {"method": "platt_v1", "slope": 1.0, "intercept": 0.0},
                "region": {"method": "platt_v1", "slope": 1.0, "intercept": 0.0},
            },
            "text_threshold": 0.949,
            "region_threshold": 0.949,
        })

        result = decode_onnx_outputs({
            "core": core,
            "ocr_groups": {},
            "ocr_unsupported": ((0, 0, "overlong"),),
            "input_sha256": "1" * 64,
        }, manifest, self.runtime.vocabulary)

        self.assertTrue(result["schema_valid"], result["schema_issues"])
        self.assertEqual(len(result["annotation"]["lines"]), 1)
        self.assertEqual(result["annotation"]["lines"][0]["id"], "l:2")
        self.assertEqual(len(result["annotation"]["regions"]), 1)
        self.assertFalse(any(item["code"] == "UNSUPPORTED_LINE_LENGTH" for item in result["warnings"]))
        self.assertEqual(result["annotation"]["regions"][0]["kind"], "TABLE")

    def test_onnx_runtime_compacts_only_calibrated_recognition_candidates(self) -> None:
        image = np.zeros((1, 3, 64, 80), dtype=np.float32)
        values = self.runtime.core_session.run(
            list(CORE_OUTPUT_NAMES), {"images": image}
        )
        core = dict(zip(CORE_OUTPUT_NAMES, values))
        core["text_mask"][:] = False
        core["text_mask"][0, :2] = True
        core["text_scores"][0, :2] = [0.8, 0.95]
        core["text_boxes"][0, 0] = [0.3, 0.5, 0.8, 0.1, 0.0, 1.0]
        core["text_boxes"][0, 1] = [0.7, 0.5, 0.8, 0.1, 0.0, 1.0]
        calls: list[np.ndarray] = []
        session = SimpleNamespace(run=lambda names, _inputs: [core[name] for name in names])
        calibration = {"scores_calibrated": True, "text_threshold": 0.9}

        def recognize(_bucket: int, _features: np.ndarray, _boxes: np.ndarray,
                      _embeddings: np.ndarray, refs: np.ndarray) -> np.ndarray:
            calls.append(refs.copy())
            return np.zeros((len(refs), 64, len(self.runtime.vocabulary)), dtype=np.float32)

        with patch.object(self.runtime, "core_session", session), patch.object(
            self.runtime, "calibration", calibration
        ), patch.object(self.runtime, "run_recognizer", side_effect=recognize):
            result = self.runtime.run_tensors(image)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].tolist(), [[0, 1]])
        bucket = 256 if self.contract_version == 2 else 64
        self.assertEqual(result["ocr_groups"][bucket]["refs"].tolist(), [[0, 1]])

    def test_unassigned_cell_relation_keeps_cell_in_canonical_annotation(self) -> None:
        model_config = self.manifest["model"]["config"]
        text_count = int(model_config["text_proposals"])
        region_count = int(model_config["region_queries"])
        node_count = text_count + region_count
        region_kinds = self.manifest["model"]["region_kinds"]
        relation_names = self.manifest["model"]["relation_names"]
        form_field_kind = region_kinds.index("FORM_FIELD")
        cell_kind = region_kinds.index("CELL")
        core = {
            "glyph_features": np.zeros((1, int(model_config["dim"]), 4, 4), dtype=np.float32),
            "text_mask": np.zeros((1, text_count), dtype=np.bool_),
            "text_boxes": np.zeros((1, text_count, 6), dtype=np.float32),
            "text_role_logits": np.zeros((1, text_count, len(self.manifest["model"]["text_roles"])), dtype=np.float32),
            "text_scores": np.zeros((1, text_count), dtype=np.float32),
            "text_candidate_count": np.array([0], dtype=np.int64),
            "text_capacity_overflow": np.array([False]),
            "region_mask": np.array([[True, True] + [False] * (region_count - 2)]),
            "region_boxes": np.zeros((1, region_count, 6), dtype=np.float32),
            "region_kinds": np.zeros((1, region_count), dtype=np.int64),
            "region_scores": np.zeros((1, region_count), dtype=np.float32),
            "graph_logits": np.full((1, len(relation_names), node_count, node_count + 1), -5.0, dtype=np.float32),
        }
        core["region_boxes"][0, 0] = [0.5, 0.5, 0.9, 0.9, 0.0, 1.0]
        core["region_boxes"][0, 1] = [0.5, 0.5, 0.2, 0.2, 0.0, 1.0]
        core["region_kinds"][0, 0] = form_field_kind
        core["region_kinds"][0, 1] = cell_kind
        core["region_scores"][0, 0] = 0.6
        core["region_scores"][0, 1] = 0.8
        core["graph_logits"][0, :, :, -1] = 0.0
        parent_channel = relation_names.index("PARENT")
        core["graph_logits"][0, parent_channel, text_count + 1, text_count] = 10.0
        settings = dict(self.manifest["calibration"]["settings"])
        settings.update({
            "scores_calibrated": True,
            "score_calibrators": {
                "text": {"method": "platt_v1", "slope": 1.0, "intercept": 0.0},
                "region": {"method": "platt_v1", "slope": 2.0, "intercept": -1.0},
            },
            "relation_scores_calibrated": True,
        })
        manifest = dict(self.manifest)
        manifest["calibration"] = dict(self.manifest["calibration"])
        manifest["calibration"]["settings"] = settings
        result = decode_onnx_outputs({
            "core": core,
            "ocr_groups": {},
            "ocr_unsupported": (),
            "input_sha256": "1" * 64,
        }, manifest, self.runtime.vocabulary)
        self.assertTrue(result["schema_valid"], result["schema_issues"])
        self.assertEqual(len(result["annotation"]["cells"]), 1)
        self.assertAlmostEqual(
            result["annotation"]["cells"][0]["detector_score"], 0.854, delta=0.002
        )
        cell_id = result["annotation"]["cells"][0]["id"]
        self.assertIsNone(result["annotation"]["cells"][0]["table_id"])
        self.assertTrue(any(
            relation["type"] == "PARENT"
            and relation["source"] == cell_id
            and relation["target"] == "r:1"
            for relation in result["annotation"]["relations"]
        ))

    def test_runtime_public_errors_distinguish_invalid_image_from_execution(self) -> None:
        with self.assertRaises(InferenceError) as invalid:
            self.runtime.predict(np.zeros((1, 3, 65, 80), dtype=np.float32))
        self.assertEqual(invalid.exception.code, "INVALID_IMAGE")

        run_tensors = self.runtime.run_tensors
        self.runtime.run_tensors = lambda _images: (_ for _ in ()).throw(ONNXExportError("private runtime detail"))
        try:
            with self.assertRaises(InferenceError) as failed:
                self.runtime.predict(np.zeros((1, 3, 64, 80), dtype=np.float32))
        finally:
            self.runtime.run_tensors = run_tensors
        self.assertEqual(failed.exception.code, "INFERENCE_ERROR")
        self.assertNotIn("private runtime detail", str(failed.exception))
        self.assertNotIn("private runtime detail", json.dumps(failed.exception.to_dict()))

    def test_unsupported_opset_bucket_and_image_size_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ONNXExportError, "unsupported ONNX opset"):
            export_graphs(self.artifact_dir, self.export_dir / "wrong-opset", opset=ONNX_OPSET - 1)
        with self.assertRaisesRegex(ONNXExportError, "unsupported OCR sequence bucket"):
            self.runtime.run_recognizer(
                32,
                np.zeros((1, 10, 16, 20), dtype=np.float32),
                np.zeros((1, 6), dtype=np.float32),
                np.zeros((1, 10), dtype=np.float32),
                np.zeros((1, 2), dtype=np.int64),
            )
        with self.assertRaisesRegex(ONNXExportError, "multiples of 16"):
            self.runtime.run_tensors(np.zeros((1, 3, 65, 80), dtype=np.float32))


class ONNXContract2ExportTest(ONNXExportTest):
    contract_version = 2


if __name__ == "__main__":
    unittest.main()
