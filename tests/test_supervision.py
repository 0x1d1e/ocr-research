from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from prismocr.calibration import _text_detector_samples, calibrate_artifact
from prismocr.data.dataset import PrismPageDataset, collate_pages, dataset_content_sha256
from prismocr.data.schema import AnnotationValidationError, resolve_line_supervision, validate_annotation
from prismocr.data.vocabulary import Vocabulary
from prismocr.eval import evaluate
from prismocr.evaluation import evaluate_page
from prismocr.losses import PrismOCRLoss
from prismocr.model.network import PrismOCRModel
from prismocr.train import CHECKPOINT_FORMAT_VERSION, _resolved_settings, _validate_resume


POLYGON = [[4, 4], [20, 4], [20, 12], [4, 12]]
LINE = {"id": "line", "text": "abc", "role": "PARA_LINE", "polygon_px": POLYGON}
EMISSION = {"id": "pred", "text": "abc", "polygon": POLYGON, "detector_score": 0.8}


class SupervisionTests(unittest.TestCase):
    def test_resolver_legacy_and_invalid_states(self) -> None:
        self.assertEqual(resolve_line_supervision({}), "unavailable")
        self.assertEqual(resolve_line_supervision({"lines": []}), "unavailable")
        legacy = {"lines": [LINE]}
        self.assertEqual(resolve_line_supervision(legacy), "complete")
        self.assertNotIn("supervision", legacy)
        for state in ("complete", "partial"):
            self.assertEqual(resolve_line_supervision({"lines": [LINE], "supervision": {"lines": state}}), state)
        for annotation in (
            {"lines": [], "supervision": {"lines": True}},
            {"lines": [], "supervision": {"lines": 1}},
            {"lines": [], "supervision": {"lines": "bogus"}},
            {"lines": [], "supervision": []},
            {"lines": [], "supervision": {}},
            {"lines": [LINE], "supervision": {"lines": "unavailable"}},
            {"lines": [], "supervision": {"lines": "partial"}},
            {"supervision": {"lines": "complete"}},
        ):
            with self.subTest(annotation=annotation), self.assertRaises(AnnotationValidationError):
                resolve_line_supervision(annotation)

    @staticmethod
    def _heatmap(targets, shape=(4, 4)):
        rendered, _, anchors = PrismOCRLoss._heatmap_targets(
            targets, shape, torch.device("cpu"), torch.float32
        )
        valid, negative = PrismOCRLoss._heatmap_masks(targets, anchors)
        logits = torch.zeros_like(rendered, requires_grad=True)
        loss = PrismOCRLoss._focal_heatmap(logits, rendered, anchors, valid, negative)
        loss.backward()
        return loss.detach(), logits.grad, anchors

    def test_unknown_zero_blank_negative_and_mixed_batch_no_dilution(self) -> None:
        unknown = {"lines": [], "supervision": {"lines": "unavailable"}}
        blank = {"lines": [], "supervision": {"lines": "complete"}}
        positive = {"lines": [{"box": [0.5, 0.5, 0.25, 0.125, 0, 1]}]}
        for targets in ([unknown], [unknown, {"lines": []}], [{}]):
            loss, gradient, _ = self._heatmap(targets)
            self.assertEqual(float(loss), 0.0)
            self.assertEqual(int(torch.count_nonzero(gradient)), 0)
        blank_loss, blank_gradient, _ = self._heatmap([blank])
        self.assertGreater(float(blank_loss), 0.0)
        self.assertTrue(bool((blank_gradient > 0).all()))
        for supervised in (blank, positive):
            solo, gradient, _ = self._heatmap([supervised])
            mixed, mixed_gradient, _ = self._heatmap([supervised, unknown])
            torch.testing.assert_close(solo, mixed, rtol=0, atol=0)
            torch.testing.assert_close(gradient[0], mixed_gradient[0], rtol=0, atol=0)
            self.assertEqual(int(torch.count_nonzero(mixed_gradient[1])), 0)

    def test_partial_annotations_have_positive_gradient_without_background_negative(self) -> None:
        partial = {"lines": [{"box": [0.5, 0.5, 0.25, 0.125, 0, 1]}],
                   "supervision": {"lines": "partial"}}
        loss, gradient, anchors = self._heatmap([partial])
        self.assertGreater(float(loss), 0)
        self.assertTrue(bool((gradient[anchors] < 0).all()))
        self.assertEqual(int(torch.count_nonzero(gradient[~anchors])), 0)

    def test_padding_and_zero_positive_grid_size_invariance(self) -> None:
        line = {"box": [0.5, 0.5, 0.25, 0.125, 0, 1]}
        for lines in ([], [line]):
            target = {"lines": lines, "regions": [], "cells": [], "supervision": {"lines": "complete"}}
            _, small_targets = collate_pages([(torch.zeros(3, 16, 16), target)])
            _, large_targets = collate_pages([
                (torch.zeros(3, 16, 16), target),
                (torch.zeros(3, 32, 32), {"lines": [], "regions": [], "cells": []}),
            ])
            small, small_grad, _ = self._heatmap(small_targets)
            large, large_grad, _ = self._heatmap(large_targets, shape=(8, 8))
            torch.testing.assert_close(small, large)
            torch.testing.assert_close(small_grad[0, :, :4, :4], large_grad[0, :, :4, :4])
            self.assertEqual(int(torch.count_nonzero(large_grad[0, :, 4:, :])), 0)
            self.assertEqual(int(torch.count_nonzero(large_grad[0, :, :, 4:])), 0)
        blank = {"lines": [], "supervision": {"lines": "complete"}}
        small, _, _ = self._heatmap([blank], (4, 4))
        large, _, _ = self._heatmap([blank], (40, 40))
        torch.testing.assert_close(small, large)

    def test_page_balanced_focal_exact_class_normalization(self) -> None:
        logits = torch.tensor([[[[0.2, -0.3], [0.7, -0.9]]], [[[0.1, -0.2], [0.3, -0.4]]]],
                              requires_grad=True)
        target = torch.tensor([[[[1.0, 0.3], [0.1, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]])
        anchors = target == 1
        valid = torch.ones_like(anchors)
        probability = logits.sigmoid()
        positive = -probability.log() * (1 - probability).square() * anchors
        negative = -(1 - probability).log() * probability.square() * (1 - target).pow(4) * ~anchors
        expected = (positive[0].sum() + negative[0].sum() / 3 + negative[1].sum() / 4) / 2
        actual = PrismOCRLoss._focal_heatmap(logits, target, anchors, valid, valid)
        torch.testing.assert_close(actual, expected)
        expected_gradient, = torch.autograd.grad(expected, logits, retain_graph=True)
        actual_gradient, = torch.autograd.grad(actual, logits)
        torch.testing.assert_close(actual_gradient, expected_gradient)

    def test_invalid_masks_and_extents_fail_closed(self) -> None:
        anchors = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
        for extent in ((0, 1), (1.1, 1), (True, 1), (float("nan"), 1), [1]):
            with self.subTest(extent=extent), self.assertRaisesRegex(ValueError, "extent"):
                PrismOCRLoss._heatmap_masks([{"_valid_image_extent": extent}], anchors)
        logits = torch.zeros(1, 1, 4, 4)
        with self.assertRaisesRegex(ValueError, "boolean"):
            PrismOCRLoss._focal_heatmap(logits, logits, anchors, logits)
        anchors[0, 0, 0, 0] = True
        with self.assertRaisesRegex(ValueError, "subsets"):
            PrismOCRLoss._focal_heatmap(logits, logits, anchors, torch.zeros_like(anchors))

    def test_real_forward_region_only_text_gradient_zero_in_mixed_batch(self) -> None:
        vocabulary = Vocabulary.from_texts(["abc"], max_size=8)
        model = PrismOCRModel(dim=10, text_proposals=4, region_queries=2,
                              vocab_size=len(vocabulary), vocabulary=vocabulary)
        targets = [
            {"lines": [], "regions": [{"id": "region", "kind": "REGION", "box": [0.5, 0.5, 0.5, 0.5, 0, 1]}],
             "cells": [], "relations": [], "supervision": {"lines": "unavailable"}},
            {"lines": [], "regions": [], "cells": [], "relations": [], "supervision": {"lines": "complete"}},
        ]
        outputs = model(torch.zeros(2, 3, 32, 32), targets)
        outputs["text_heatmap_logits"].retain_grad()
        losses = PrismOCRLoss(vocabulary)(outputs, targets)
        self.assertEqual(float(losses["role"].detach()), 0)
        self.assertEqual(float(losses["ctc"].detach()), 0)
        self.assertEqual(outputs["ocr_groups"], {})
        losses["total"].backward()
        gradient = outputs["text_heatmap_logits"].grad
        self.assertEqual(int(torch.count_nonzero(gradient[0])), 0)
        self.assertTrue(bool((gradient[1] > 0).all()))
        unknown = [targets[0]]
        outputs = model(torch.zeros(1, 3, 32, 32), unknown)
        loss = PrismOCRLoss(vocabulary)(outputs, unknown)["heatmap"]
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                            for parameter in model.parameters()))

    def test_partial_positive_role_geometry_ctc_remain_usable(self) -> None:
        # Use the established MatchSet fixture, not independently re-matched loss slots.
        try:
            from tests.test_training_contracts import TrainingContractTests
        except ModuleNotFoundError:
            from test_training_contracts import TrainingContractTests
        vocabulary = Vocabulary.from_texts(["abc"], max_size=8)
        line = {**LINE, "box": [0.5, 0.5, 0.25, 0.125, 0, 1]}
        target = {"lines": [line], "regions": [], "cells": [], "reading_order": ["line"],
                  "supervision": {"lines": "partial"}}
        outputs = TrainingContractTests()._outputs(vocabulary, [line], [], ctc_bucket=64)
        outputs["text"]["boxes"][0, 0, 2] *= 0.5
        outputs["text"]["boxes"].requires_grad_()
        losses = PrismOCRLoss(vocabulary)(outputs, [target])
        self.assertGreater(float(losses["role"].detach()), 0)
        self.assertGreater(float(losses["ctc"].detach()), 0)
        self.assertEqual(float(losses["graph"].detach()), 0)  # no invented terminal NEXT NULL
        losses["total"].backward()
        self.assertGreater(int(torch.count_nonzero(outputs["text"]["role_logits"].grad)), 0)
        self.assertGreater(int(torch.count_nonzero(outputs["ocr_groups"][64]["logits"].grad)), 0)
        self.assertGreater(int(torch.count_nonzero(outputs["text"]["boxes"].grad)), 0)

    @staticmethod
    def _write_manifest(root: Path, states) -> Path:
        rows = []
        for index, state in enumerate(states):
            image = root / f"image-{index}.png"
            Image.new("RGB", (31, 17), (index, 0, 0)).save(image)
            lines = [copy.deepcopy(LINE)] if state in ("positive", "partial") else []
            annotation = {
                "schema_version": "4.0", "document_id": f"doc-{index}", "page_id": f"page-{index}",
                "image": {"width": 31, "height": 17, "sha256": hashlib.sha256(image.read_bytes()).hexdigest()},
                "provenance": {"source_class": "synthetic", "license": "generated",
                               "permission_reference": "not_applicable", "reviewer": "test"},
                "lines": lines, "regions": [], "cells": [], "relations": [],
                "reading_order": [line["id"] for line in lines],
            }
            if state != "legacy":
                annotation["supervision"] = {"lines": "complete" if state in ("positive", "blank") else state}
            validate_annotation(annotation)
            rows.append({"image_path": str(image), "annotation": annotation})
        manifest = root / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return manifest

    def test_dataset_resolves_without_mutating_raw_annotation_or_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._write_manifest(root, ["legacy", "blank", "partial"])
            original = manifest.read_bytes()
            dataset = PrismPageDataset(manifest, max_page_side=16)
            before = copy.deepcopy([record["_annotation"] for record in dataset.records])
            digest_before = dataset_content_sha256(dataset)
            batch, targets = collate_pages([dataset[index] for index in range(3)])
            self.assertEqual([target["supervision"]["lines"] for target in targets],
                             ["unavailable", "complete", "partial"])
            self.assertEqual(targets[0]["_valid_image_extent"], (1.0, 9 / 16))
            self.assertEqual(tuple(batch.shape), (3, 3, 16, 16))
            self.assertEqual(before, [record["_annotation"] for record in dataset.records])
            self.assertEqual(digest_before, dataset_content_sha256(dataset))
            self.assertEqual(original, manifest.read_bytes())

    def test_eval_and_calibration_exclude_unknown_partial_but_penalize_blank(self) -> None:
        prediction = {"lines": [EMISSION], "regions": [], "cells": [], "reading_order": ["pred"]}
        for state in ("unavailable", "partial", "complete"):
            annotation = {"image": {"width": 31, "height": 17}, "lines": [LINE] if state == "partial" else [],
                          "supervision": {"lines": state}, "reading_order": []}
            metrics = evaluate_page(annotation, prediction)
            self.assertEqual(metrics["detections"]["lines"]["false_positive"], int(state == "complete"))
            self.assertEqual(metrics["text_emissions"]["line_count"], 1)
            self.assertEqual(metrics["reading_order"]["page_count"], int(state == "complete"))
            self.assertEqual(_text_detector_samples(annotation, [EMISSION], iou_threshold=0.5),
                             [(0.8, False)] if state == "complete" else [])
            if state != "complete":
                self.assertEqual(metrics["language_slices"], {})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._write_manifest(root, ["positive", "legacy", "unavailable", "partial", "blank"])
            api = SimpleNamespace(model_version="fixture", predict=lambda *_: SimpleNamespace(to_dict=lambda: prediction))
            with patch("prismocr.eval.PrismOCR.load", return_value=api):
                metrics = evaluate(manifest, root / "artifact")
            self.assertEqual(metrics["line_supervision_page_counts"], {"complete": 2, "partial": 1, "unavailable": 2})
            self.assertEqual(metrics["detection"]["0.5"]["lines"]["false_positive"], 1)
            self.assertEqual(metrics["character_errors"], 3)
            self.assertEqual(metrics["ground_truth_characters"], 3)
            self.assertEqual(metrics["predicted_lines"], 5)
            self.assertEqual(metrics["text_emissions_excluded_from_metrics"], 3)
            manifest = self._write_manifest(root, ["legacy", "unavailable", "partial"])
            with patch("prismocr.eval.PrismOCR.load", return_value=api):
                unknown_metrics = evaluate(manifest, root / "artifact")
            self.assertFalse(unknown_metrics["text_metrics_defined"])
            self.assertEqual(unknown_metrics["character_errors"], 0)
            self.assertEqual(unknown_metrics["detection"]["0.5"]["lines"]["false_positive"], 0)
            self.assertFalse(unknown_metrics["metrics_are_defined"])

    def test_partial_table_structure_and_independent_cell_text_do_not_resolve_empty_lookups(self) -> None:
        annotation = {
            "image": {"width": 31, "height": 17}, "lines": [LINE],
            "supervision": {"lines": "partial"}, "reading_order": ["line"],
            "regions": [{"id": "table", "kind": "TABLE", "polygon_px": POLYGON}],
            "cells": [{"id": "cell", "table_id": "table", "row": 0, "col": 0,
                       "polygon_px": POLYGON, "line_ids": ["line"]}],
        }
        prediction = {
            "lines": [EMISSION], "reading_order": ["pred"],
            "regions": [{"id": "p-table", "kind": "TABLE", "polygon": POLYGON}],
            "cells": [{"id": "p-cell", "table_id": "p-table", "row": 0, "col": 0,
                       "polygon": POLYGON, "line_ids": ["pred"]}],
            "tables": [{"table_id": "p-table", "status": "complete", "rows": 1, "cols": 1}],
        }
        metrics = evaluate_page(annotation, prediction)["table_structure"]
        self.assertEqual(metrics["teds_text_count"], 0)
        self.assertEqual(metrics["teds_structure_sum"], 1)
        annotation["cells"][0]["text"] = "abc"
        metrics = evaluate_page(annotation, prediction)["table_structure"]
        self.assertEqual(metrics["teds_text_count"], 1)
        self.assertEqual(metrics["teds_sum"], 1)
        self.assertEqual(metrics["teds_structure_sum"], 1)
        annotation["cells"][0]["text"] = "b\na"
        prediction["lines"] = [{**EMISSION, "id": "pa", "text": "a"},
                               {**EMISSION, "id": "pb", "text": "b"}]
        prediction["reading_order"] = ["pb", "pa"]
        prediction["cells"][0]["line_ids"] = ["pa", "pb"]
        metrics = evaluate_page(annotation, prediction)["table_structure"]
        self.assertEqual(metrics["teds_sum"], 1)

    def test_resume_rejects_legacy_objective_even_when_raw_data_hash_matches(self) -> None:
        vocabulary = Vocabulary.from_texts(["abc"], max_size=8)
        settings = _resolved_settings({}, {}, vocabulary, batch_size=1, num_workers=0)
        self.assertEqual(settings["training"]["objective_contract"], "lines-availability-v1-page-balanced-focal")
        legacy_settings = copy.deepcopy(settings)
        del legacy_settings["training"]["objective_contract"]
        compatibility = {"settings": settings, "vocabulary_sha256": vocabulary.sha256}
        checkpoint = {"format_version": CHECKPOINT_FORMAT_VERSION,
                      "compatibility": {**compatibility, "settings": legacy_settings}}
        with self.assertRaisesRegex(ValueError, "configuration is incompatible"):
            _validate_resume(checkpoint, compatibility=compatibility, requested_epochs=1)

    def test_calibration_public_collection_never_labels_unavailable_emissions_negative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._write_manifest(root, ["positive", "legacy", "unavailable", "partial", "blank"])
            dataset = PrismPageDataset(manifest)
            artifact = root / "artifact"
            artifact.mkdir()
            vocabulary = Vocabulary.from_texts(["abc"], max_size=8)
            vocabulary.save(artifact / "vocab-v4.json")
            (artifact / "model.pt").write_bytes(b"fixture")
            (artifact / "model-config.json").write_text("{}")
            (artifact / "manifest.json").write_text(json.dumps({
                "schema_version": "4.0", "model_sha256": hashlib.sha256(b"fixture").hexdigest(),
                "model_config_sha256": hashlib.sha256(b"{}").hexdigest(), "vocabulary_sha256": vocabulary.sha256,
                "training_provenance": {"train_dataset_sha256": dataset_content_sha256(dataset),
                                        "validation_dataset_sha256": dataset_content_sha256(dataset)},
            }))
            candidates = [{"type": name, "source": "source", "target": target, "score": 0.8}
                          for name in ("NEXT", "PARENT", "KV_VALUE", "RIGHT_CELL") for target in (None, "wrong")]
            prediction = {"lines": [EMISSION], "regions": [], "cells": [],
                          "diagnostics": {"relation_candidate_calibration_data": candidates}}
            engine = SimpleNamespace(calibration={}, predict=lambda *_: SimpleNamespace(to_dict=lambda: prediction))
            fit_calls = []
            def fit(samples, **kwargs):
                fit_calls.append((list(samples), kwargs))
                return {"raw_threshold": 0.5}
            decisions = {(name, "source"): None for name in ("NEXT", "PARENT", "KV_VALUE", "RIGHT_CELL")}
            with patch("prismocr.calibration._validation_split_provenance", return_value=({}, "v" * 64, "s" * 64)), \
                 patch("prismocr.calibration._verified_split_datasets", return_value={"train": dataset, "validation": dataset}), \
                 patch("prismocr.calibration.PrismOCR.load", return_value=engine), \
                 patch("prismocr.calibration._fit_stream", side_effect=fit), \
                 patch("prismocr.calibration._matched_node_ids", return_value={"source": "source"}), \
                 patch("prismocr.calibration._ground_truth_relation_decisions", return_value=decisions):
                report = calibrate_artifact(artifact, manifest)
            self.assertEqual(fit_calls[0][0], [(0.8, True), (0.8, False)])
            self.assertEqual(fit_calls[0][1], {"ground_truth_count": 1, "blank_predictions": 1})
            self.assertEqual(report["line_supervision_page_counts"], {"complete": 2, "partial": 1, "unavailable": 2})
            self.assertEqual(report["blank_pages"]["text_emissions_excluded_from_metrics"], 3)
            self.assertEqual(report["blank_pages"]["certified_text_blank_page_count"], 1)


if __name__ == "__main__":
    unittest.main()
