from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from prismocr.api import PrismOCR, save_artifact
from prismocr.data.dataset import collate_pages
from prismocr.data.vocabulary import Vocabulary
from prismocr.geometry import rotated_box_to_polygon
from prismocr.losses import PrismOCRLoss
from prismocr.model.heads import TextLineHead
from prismocr.model.network import PrismOCRModel, validated_model_state
from prismocr.model.recognizer import choose_inference_bucket
from prismocr.onnx_export import ONNXExportError, _load_unfused_source_model
from prismocr.train import _resolved_settings
from scripts.overfit_heatmap import build_model


class ModelContractTests(unittest.TestCase):
    @staticmethod
    def model(version=2):
        vocabulary = Vocabulary.from_texts(["abc"], max_size=8)
        return PrismOCRModel(dim=10, text_proposals=1, region_queries=1,
                             vocab_size=len(vocabulary), vocabulary=vocabulary,
                             model_contract_version=version).eval()

    @staticmethod
    def raw(width, height, line_height=16, angle=0.3):
        box = torch.zeros(1, 6, height // 4, width // 4)
        box[:, 2] = 2 + math.log(math.expm1(64 / 256))
        box[:, 3] = 2 + math.log(math.expm1(line_height / 256))
        box[:, 4] = math.sin(angle)
        box[:, 5] = math.cos(angle)
        logits = torch.full((1, 1, height // 4, width // 4), -20.0)
        logits[0, 0, 32, 32] = 3
        return {"box_map": box, "heatmap_logits": logits,
                "role_map": torch.zeros(1, 12, height // 4, width // 4),
                "feature_map": torch.zeros(1, 10, height // 4, width // 4)}

    def test_contract2_pixel_sizes_polygons_and_centers_invariant_rectangular_canvases(self):
        polygons = []
        for width, height in ((768, 512), (768, 1024), (1536, 1024)):
            result = TextLineHead.proposals(self.raw(width, height), max_proposals=1,
                score_threshold=0.5, model_contract_version=2, image_size=(width, height))
            box = result["boxes"][0, 0]
            torch.testing.assert_close(box[:4] * torch.tensor([width, height, width, height]),
                                       torch.tensor([130., 130., 64., 16.]))
            polygons.append(torch.tensor(rotated_box_to_polygon(box.tolist(), (width, height))))
        for polygon in polygons[1:]:
            torch.testing.assert_close(polygon, polygons[0])

    def test_contract2_nms_uses_same_physical_height_across_canvas_padding(self):
        for line_height, count in ((4, 2), (16, 1)):
            for width, height in ((768, 512), (768, 1024), (1536, 1024)):
                raw = self.raw(width, height, line_height)
                raw["heatmap_logits"][0, 0, 33, 32] = 2
                result = TextLineHead.proposals(raw, max_proposals=4, score_threshold=0.5,
                    model_contract_version=2, image_size=(width, height))
                self.assertEqual(int(result["candidate_count"][0]), count)
                self.assertEqual(int(result["mask"].sum()), count)
                physical = result["boxes"][0, result["mask"][0], :2] * torch.tensor([width, height])
                expected = torch.tensor([[130., 130.], [130., 134.]])[:count]
                torch.testing.assert_close(physical, expected)

    def test_fresh_contract2_box_initialization_and_size_gradients(self):
        model = self.model()
        raw = model.text_head(torch.randn(1, 64, 8, 12))
        self.assertEqual(int(torch.count_nonzero(model.text_head.box.weight)), 0)
        torch.testing.assert_close(raw["box_map"][:, :2], torch.zeros(1, 2, 8, 12))
        torch.testing.assert_close(raw["box_map"][:, 4], torch.zeros(1, 8, 12))
        torch.testing.assert_close(raw["box_map"][:, 5], torch.ones(1, 8, 12))
        sizes = TextLineHead.decode_sizes(raw["box_map"][:, 2:4].permute(0, 2, 3, 1),
                                        model_contract_version=2, image_size=(48, 32))
        torch.testing.assert_close(sizes * torch.tensor([48, 32]),
                                  torch.tensor([64., 16.]).expand(1, 8, 12, 2))
        sizes.sum().backward()
        self.assertTrue(bool((model.text_head.box.bias.grad[2:4] > 0).all()))
        self.assertTrue(bool(torch.isfinite(model.text_head.box.bias.grad).all()))

    def test_default_and_explicit_legacy_same_seed_initialization_and_geometry(self):
        torch.manual_seed(19)
        implicit = PrismOCRModel(dim=10, text_proposals=1, region_queries=1)
        torch.manual_seed(19)
        explicit = PrismOCRModel(dim=10, text_proposals=1, region_queries=1, model_contract_version=1)
        for key, value in implicit.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
        heights = []
        for width, height in ((768, 512), (768, 1024), (1536, 1024)):
            raw = self.raw(width, height)
            raw["box_map"][:, 2:4] = 0
            default = TextLineHead.proposals(raw, max_proposals=1, score_threshold=0.5)
            explicit = TextLineHead.proposals(raw, max_proposals=1, score_threshold=0.5,
                                             model_contract_version=1)
            torch.testing.assert_close(default["boxes"], explicit["boxes"], rtol=0, atol=0)
            heights.append(float(default["boxes"][0, 0, 3]) * height)
        self.assertAlmostEqual(heights[0], 64.987143, places=4)
        self.assertAlmostEqual(heights[1], heights[0] * 2, places=4)
        self.assertAlmostEqual(heights[2], heights[1], places=4)

    def test_strict_versions_and_contract2_requires_actual_dimensions(self):
        vocabulary = Vocabulary.from_texts(["a"])
        for version in (True, False, 0, 3, 2.0, "2", None):
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, "model_contract_version"):
                    PrismOCRModel(model_contract_version=version)
                with self.assertRaisesRegex(ValueError, "model_contract_version"):
                    _resolved_settings({"model_contract_version": version}, {}, vocabulary, batch_size=1, num_workers=0)
        with self.assertRaisesRegex(ValueError, "actual input"):
            TextLineHead.decode_sizes(torch.zeros(2), model_contract_version=2)

    def test_padding_preserves_controlled_prediction_and_gt_pixel_geometry_and_loss(self):
        target = {"lines": [{"box": [0.5, 0.5, 64/128, 16/64, 0., 1.]}], "regions": [], "cells": []}
        _, solo = collate_pages([(torch.zeros(3, 64, 128), target)])
        _, mixed = collate_pages([(torch.zeros(3, 64, 128), target),
                                 (torch.zeros(3, 128, 256), {"lines": [], "regions": [], "cells": []})])
        losses = []
        for targets, size in ((solo, (128, 64)), (mixed, (256, 128))):
            truth = torch.tensor(targets[0]["lines"][0]["box"])
            sides = TextLineHead.decode_sizes(torch.tensor([0.741308451, -0.741175967]),
                                              model_contract_version=2, image_size=size)
            torch.testing.assert_close(sides * torch.tensor(size), torch.tensor([64., 16.]))
            torch.testing.assert_close(truth[2:4] * torch.tensor(size), torch.tensor([64., 16.]))
            predicted = truth.clone()
            predicted[2:4] = sides * 0.5
            losses.append(PrismOCRLoss._box_loss(predicted[None], truth[None]))
        torch.testing.assert_close(losses[0], losses[1])

    def test_fixed_max_training_inference_and_rejections_preserve_identity(self):
        model = self.model()
        image = torch.zeros(1, 3, 64, 128)
        with torch.no_grad():
            base = model(image, text_threshold=0.0)
        line = {"id": "line", "box": base["text"]["boxes"][0, 0].tolist(), "text": "abc", "role": "PARA_LINE"}
        target = {"page_id": "page", "lines": [line], "regions": [], "cells": []}
        for text in ("a", "aa", "abc"):
            line["text"] = text
            for ratio in (0., 0.5, 1.):
                outputs = model(image, [target], teacher_force_ratio=ratio)
                self.assertEqual(set(outputs["ocr_groups"]), {256})
                self.assertEqual(outputs["ctc_bucket_policy"], "fixed_max")
        self.assertEqual(set(base["ocr_groups"]), {256})
        enormous = torch.tensor([0.5, 0.5, 100., 0.0001, 0., 1.])
        self.assertIsNone(choose_inference_bucket(enormous, (128, 64)))
        self.assertEqual(choose_inference_bucket(enormous, (128, 64), model_contract_version=2), 256)
        for text, reason in (("a" * 129, "requires_257_frames_but_bucket_is_256"), ("a?", "ctc_oov_characters")):
            line["text"] = text
            outputs = model(image, [target])
            self.assertEqual(outputs["ocr_groups"], {})
            self.assertEqual(outputs["ocr_unsupported"], ((0, 0, reason),))
            rejected = PrismOCRLoss(model.vocabulary)(outputs, [target])["rejected_ctc_samples"]
            self.assertEqual(rejected[0]["page_id"], "page")
            self.assertEqual(rejected[0]["line_id"], "line")
            self.assertEqual(rejected[0]["reason"], reason)

    def test_both_contract_artifact_roundtrip_and_legacy_missing_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            for version in (1, 2):
                root = Path(temporary) / str(version)
                model = self.model(version)
                save_artifact(root, model, model.vocabulary)
                loaded = PrismOCR.load(root)
                self.assertEqual(loaded.model.model_contract_version, version)
                image = torch.randn(1, 3, 32, 48)
                with torch.no_grad():
                    expected = model.fused_for_inference()(image)
                    actual = loaded.model(image)
                torch.testing.assert_close(actual["text"]["boxes"], expected["text"]["boxes"], rtol=0, atol=0)
                config = json.loads((root / "model-config.json").read_text())
                checkpoint = torch.load(root / "model.pt", weights_only=True)
                self.assertEqual(config["model_contract_version"], version)
                self.assertEqual(checkpoint["model_contract_version"], version)
                if version == 1:
                    del config["model_contract_version"]
                    del checkpoint["model_contract_version"]
                    (root / "model-config.json").write_text(json.dumps(config))
                    torch.save(checkpoint, root / "model.pt")
                    self.rehash(root)
                    self.assertEqual(PrismOCR.load(root).model.model_contract_version, 1)

    @staticmethod
    def rehash(root):
        manifest = json.loads((root / "manifest.json").read_text())
        for name, field in (("model.pt", "model_sha256"), ("model-config.json", "model_config_sha256")):
            manifest[field] = hashlib.sha256((root / name).read_bytes()).hexdigest()
        (root / "manifest.json").write_text(json.dumps(manifest))

    def test_checkpoint_config_conflicts_unknown_and_config_only_relabel_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = self.model(1)
            save_artifact(root, model, model.vocabulary)
            config = json.loads((root / "model-config.json").read_text())
            for configured, stored in ((2, 1), (1, 2), (2, None), (3, 3), (True, True)):
                altered = {**config, "model_contract_version": configured}
                checkpoint = {"model_state": model.state_dict()}
                if stored is not None:
                    checkpoint["model_contract_version"] = stored
                torch.save(checkpoint, root / "model.pt")
                (root / "model-config.json").write_text(json.dumps(altered))
                self.rehash(root)
                with self.assertRaisesRegex(ValueError, "model_contract_version"):
                    PrismOCR.load(root)
                with self.assertRaisesRegex(ONNXExportError, "model_contract_version"):
                    _load_unfused_source_model(root, model.vocabulary, altered)
            # Hash mismatch still rejects before trusting otherwise supported metadata.
            save_artifact(root, model, model.vocabulary)
            (root / "model-config.json").write_text(json.dumps({**config, "model_contract_version": 2}))
            with self.assertRaisesRegex(ValueError, "model_config_sha256"):
                PrismOCR.load(root)

    def test_overfit_helper_retains_contract_and_actual_vocabulary(self):
        model = self.model(2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_artifact(root, model, model.vocabulary)
            config = json.loads((root / "model-config.json").read_text())
            rebuilt = build_model(config, model.vocabulary)
            self.assertEqual(rebuilt.model_contract_version, 2)
            self.assertIs(rebuilt.vocabulary, model.vocabulary)
            checkpoint = torch.load(root / "model.pt", weights_only=True)
            rebuilt.load_state_dict(validated_model_state(checkpoint, config))


if __name__ == "__main__":
    unittest.main()
