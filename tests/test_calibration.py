from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from prismocr.api import PrismOCR, _calibrated_score, _validated_calibration
from prismocr.calibration import (
    _fit_platt,
    _ground_truth_relation_decisions,
    _matched_node_ids,
    _operating_threshold,
    _reliability,
    calibrate_artifact,
)
from prismocr.data.dataset import PrismPageDataset, dataset_content_sha256
from prismocr.data.tools import build_document_splits
from prismocr.data.vocabulary import Vocabulary
from prismocr.geometry import rotated_box_to_polygon
from prismocr.model.network import REGION_KIND_TO_ID, TEXT_ROLE_TO_ID, PrismOCRModel


class CalibrationTests(unittest.TestCase):
    def test_invalid_calibration_samples_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite probabilities"):
            _fit_platt([float("nan"), 0.1], [True, False])
        with self.assertRaisesRegex(ValueError, "finite probabilities"):
            _reliability([float("inf")], [False])

    def test_raw_threshold_ranking_survives_equal_saturated_probabilities(self) -> None:
        threshold, raw_threshold = _operating_threshold(
            [1.0, 1.0, 0.1], [1.0, 0.9999995, 0.1],
            [True, False, False], ground_truth_count=1,
        )
        self.assertEqual(raw_threshold, 1.0)
        self.assertEqual(threshold, 1.0)

    def test_relation_node_matching_maximizes_threshold_qualified_pairs(self) -> None:
        annotation = {
            "lines": [
                {"id": "gt-a", "polygon_px": [[0.0, 0.0]]},
                {"id": "gt-b", "polygon_px": [[1.0, 1.0]]},
            ],
        }
        prediction = {
            "lines": [
                {"id": "pred-a", "polygon": [[10.0, 10.0]]},
                {"id": "pred-b", "polygon": [[11.0, 11.0]]},
            ],
        }
        overlaps = {
            ((0.0, 0.0), (10.0, 10.0)): 0.9,
            ((1.0, 1.0), (10.0, 10.0)): 0.8,
            ((0.0, 0.0), (11.0, 11.0)): 0.85,
            ((1.0, 1.0), (11.0, 11.0)): 0.0,
        }
        with patch(
            "prismocr.calibration.source_polygon_iou",
            side_effect=lambda truth, pred: overlaps[(tuple(truth[0]), tuple(pred[0]))],
        ):
            matched = _matched_node_ids(annotation, prediction, 0.5)
        self.assertEqual(matched, {"pred-a": "gt-b", "pred-b": "gt-a"})

    def test_relation_decisions_preserve_explicit_nulls_and_partial_annotations(self) -> None:
        decisions = _ground_truth_relation_decisions({
            "lines": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            "reading_order": ["a", "b", "c"],
            "relations": [
                {"type": "NEXT", "source": "a", "target": None, "explicit_null": True},
                {"type": "NEXT", "source": "b", "target": "c"},
            ],
        })
        self.assertIsNone(decisions[("NEXT", "a")])
        self.assertEqual(decisions[("NEXT", "b")], "c")
        self.assertIsNone(decisions[("NEXT", "c")])
        self.assertNotIn(("PARENT", "a"), decisions)

    def test_platt_fit_optimizes_intercept_when_positive_slope_constraint_is_active(self) -> None:
        slope, intercept = _fit_platt([0.6] + [0.99] * 9, [True] + [False] * 9)
        self.assertAlmostEqual(slope, 1e-4, places=8)
        false_positive_probability = 1.0 / (1.0 + np.exp(-(
            slope * np.log(0.99 / 0.01) + intercept
        )))
        self.assertLess(false_positive_probability, 0.2)

    def test_platt_fit_is_monotonic_and_reliability_uses_calibrated_probabilities(self) -> None:
        scores = [0.8, 0.7, 0.2, 0.1]
        labels = [True, True, False, False]
        slope, intercept = _fit_platt(scores, labels)
        self.assertGreater(slope, 0.0)
        transform = {"text": {"slope": slope, "intercept": intercept}}
        calibrated = [_calibrated_score(score, {"score_calibrators": transform}, "text") for score in scores]
        self.assertEqual(calibrated, sorted(calibrated, reverse=True))
        reliability = _reliability(calibrated, labels)
        self.assertGreaterEqual(reliability["brier"], 0.0)
        self.assertLessEqual(reliability["brier"], 1.0)
        self.assertGreaterEqual(reliability["ece"], 0.0)
        self.assertLessEqual(reliability["ece"], 1.0)
        self.assertEqual(sum(item["count"] for item in reliability["bins"]), len(scores))

    def test_operating_point_uses_observed_raw_score_for_saturated_platt_outputs(self) -> None:
        scores = [0.8, 0.1] * 100
        labels = [True, False] * 100
        slope, intercept = _fit_platt(scores, labels)
        probabilities = [
            1.0 / (1.0 + np.exp(-np.clip(slope * np.log(score / (1.0 - score)) + intercept, -80, 80)))
            for score in scores
        ]
        from prismocr.calibration import _operating_threshold
        probability_threshold, raw_threshold = _operating_threshold(
            probabilities, scores, labels, ground_truth_count=100
        )
        self.assertEqual(raw_threshold, 0.8)
        self.assertTrue(all(
            (score >= raw_threshold) == (probability >= probability_threshold)
            for score, probability in zip(scores, probabilities, strict=True)
        ))

    def test_calibration_binds_to_frozen_validation_split_and_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            def box(cx: float, cy: float, width: float, height: float) -> list[float]:
                return [cx, cy, width, height, 0.0, 1.0]

            def polygon(value: list[float]) -> list[list[float]]:
                return [list(point) for point in rotated_box_to_polygon(value, (64, 64))]

            truth_line_1 = polygon(box(0.25, 0.20, 0.40, 0.08))
            truth_line_2 = polygon(box(0.25, 0.35, 0.40, 0.08))
            truth_region = polygon(box(0.50, 0.55, 0.90, 0.80))
            truth_field = polygon(box(0.25, 0.20, 0.46, 0.14))
            truth_cells = [
                polygon(box(center, 0.72, 0.20, 0.20))
                for center in (0.25, 0.50, 0.75)
            ]
            for index in range(3):
                image_path = root / f"page-{index}.png"
                pixels = np.random.default_rng(100 + index).integers(0, 256, (64, 64, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(image_path)
                document_id = f"calibration-doc-{index}"
                annotation = {
                    "schema_version": "4.0",
                    "document_id": document_id,
                    "page_id": f"{document_id}:page-0",
                    "image": {
                        "width": 64,
                        "height": 64,
                        "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    },
                    "provenance": {
                        "source_class": "synthetic",
                        "license": "generated",
                        "permission_reference": "not_applicable",
                        "reviewer": "test",
                    },
                    "lines": [
                        {"id": "gt-line-1", "polygon_px": truth_line_1, "text": "key", "role": "KEY"},
                        {"id": "gt-line-2", "polygon_px": truth_line_2, "text": "value", "role": "VALUE"},
                    ],
                    "regions": [
                        {"id": "gt-table", "kind": "TABLE", "polygon_px": truth_region},
                        {"id": "gt-field", "kind": "FORM_FIELD", "polygon_px": truth_field},
                    ],
                    "cells": [
                        {
                            "id": f"gt-cell-{cell_index + 1}",
                            "table_id": "gt-table",
                            "row": 0,
                            "col": cell_index,
                            "polygon_px": polygon,
                        }
                        for cell_index, polygon in enumerate(truth_cells)
                    ],
                    "relations": [
                        {"type": "NEXT", "source": "gt-line-1", "target": "gt-line-2"},
                        {"type": "PARENT", "source": "gt-line-1", "target": "gt-table"},
                        {"type": "KV_VALUE", "source": "gt-line-1", "target": "gt-line-2"},
                        {"type": "RIGHT_CELL", "source": "gt-cell-1", "target": "gt-cell-2"},
                        {"type": "RIGHT_CELL", "source": "gt-cell-2", "target": "gt-cell-3"},
                    ],
                    "reading_order": ["gt-line-1", "gt-line-2"],
                }
                rows.append({"image_path": str(image_path), "annotation": annotation})
            source_manifest = root / "source.jsonl"
            source_manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            report = build_document_splits(
                source_manifest,
                root / "frozen-splits",
                ratios=(1 / 3, 1 / 3, 1 / 3),
                near_duplicate_hamming=0,
            )
            validation_manifest = Path(report["manifests"]["validation"])
            artifact = root / "artifact"
            artifact.mkdir()
            vocabulary = Vocabulary.from_texts(["key", "value", "text"], max_size=16)
            vocabulary.save(artifact / "vocab-v4.json")
            model_config = {
                "dim": 10,
                "text_proposals": 3,
                "region_queries": 6,
                "vocab_size": len(vocabulary),
            }
            config_bytes = (json.dumps(model_config, indent=2) + "\n").encode("utf-8")
            (artifact / "model-config.json").write_bytes(config_bytes)
            train_dataset = PrismPageDataset(report["manifests"]["train"])
            validation_dataset = PrismPageDataset(report["manifests"]["validation"])
            model_weights = artifact / "model.pt"
            model_weights.write_bytes(b"synthetic-calibration-model")
            model_hash = hashlib.sha256(model_weights.read_bytes()).hexdigest()
            (artifact / "manifest.json").write_text(
                json.dumps({
                    "schema_version": "4.0",
                    "model_sha256": model_hash,
                    "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                    "vocabulary_sha256": vocabulary.sha256,
                    "training_provenance": {
                        "train_dataset_sha256": dataset_content_sha256(train_dataset),
                        "validation_dataset_sha256": dataset_content_sha256(validation_dataset),
                    },
                }),
                encoding="utf-8",
            )

            model = PrismOCRModel(
                dim=10,
                text_proposals=3,
                region_queries=6,
                vocab_size=len(vocabulary),
                vocabulary=vocabulary,
            )
            engine = PrismOCR(model, vocabulary)
            text_boxes = [
                box(0.25, 0.20, 0.40, 0.08),
                box(0.25, 0.35, 0.40, 0.08),
                box(0.25, 0.50, 0.40, 0.08),
            ]
            region_boxes = [
                box(0.50, 0.55, 0.90, 0.80),
                box(0.25, 0.20, 0.46, 0.14),
                *(box(center, 0.72, 0.20, 0.20) for center in (0.25, 0.50, 0.75)),
                box(0.85, 0.20, 0.12, 0.08),
            ]
            text_scores = [0.90, 0.85, 0.86]
            region_scores = [0.95, 0.94, 0.92, 0.91, 0.90, 0.86]

            def fake_forward(_self: PrismOCRModel, images: torch.Tensor) -> dict[str, object]:
                device = images.device
                role_logits = torch.zeros((1, 3, len(TEXT_ROLE_TO_ID)), device=device)
                role_logits[0, 0, TEXT_ROLE_TO_ID["KEY"]] = 10.0
                role_logits[0, 1, TEXT_ROLE_TO_ID["VALUE"]] = 10.0
                region_kinds = torch.tensor([[
                    REGION_KIND_TO_ID["TABLE"],
                    REGION_KIND_TO_ID["FORM_FIELD"],
                    REGION_KIND_TO_ID["CELL"],
                    REGION_KIND_TO_ID["CELL"],
                    REGION_KIND_TO_ID["CELL"],
                    REGION_KIND_TO_ID["REGION"],
                ]], device=device)
                return {
                    "text": {
                        "mask": torch.tensor([text_scores], dtype=torch.float32, device=device)
                        >= _self.text_threshold,
                        "boxes": torch.tensor(text_boxes, dtype=torch.float32, device=device).unsqueeze(0),
                        "scores": torch.tensor([text_scores], dtype=torch.float32, device=device),
                        "role_logits": role_logits,
                        "embeddings": torch.zeros((1, 3, model.dim), device=device),
                        "candidate_count": torch.tensor([3], device=device),
                        "capacity_overflow": torch.tensor([False], device=device),
                    },
                    "regions": {
                        "mask": torch.tensor([region_scores], dtype=torch.float32, device=device)
                        >= _self.region_threshold,
                        "boxes": torch.tensor(region_boxes, dtype=torch.float32, device=device).unsqueeze(0),
                        "scores": torch.tensor([region_scores], dtype=torch.float32, device=device),
                        "kinds": region_kinds,
                        "row_logits": torch.zeros((1, 6, 128), device=device),
                        "col_logits": torch.zeros((1, 6, 128), device=device),
                        "rowspan_logits": torch.zeros((1, 6, 32), device=device),
                        "colspan_logits": torch.zeros((1, 6, 32), device=device),
                        "header_logits": torch.zeros((1, 6), device=device),
                        "embeddings": torch.zeros((1, 6, model.dim), device=device),
                    },
                    "ocr_groups": {},
                    "ocr_unsupported": (),
                }

            def neutral_relations(
                _self: torch.nn.Module, nodes: torch.Tensor
            ) -> torch.Tensor:
                count = nodes.shape[1]
                return torch.zeros((1, 5, count, count + 1), device=nodes.device)

            observed_thresholds: list[tuple[float, float]] = []

            def record_thresholds(self: PrismOCRModel, images: torch.Tensor) -> dict[str, object]:
                observed_thresholds.append((self.text_threshold, self.region_threshold))
                return fake_forward(self, images)

            engine.model.forward = __import__("types").MethodType(record_thresholds, engine.model)
            engine.model.relation_head.forward = __import__("types").MethodType(
                neutral_relations, engine.model.relation_head
            )
            with patch("prismocr.calibration.PrismOCR.load", return_value=engine):
                result = calibrate_artifact(artifact, validation_manifest)

            self.assertEqual(result["model_sha256"], model_hash)
            self.assertTrue(observed_thresholds)
            self.assertTrue(all(pair == (0.55, 0.5) for pair in observed_thresholds))
            self.assertAlmostEqual(result["region_threshold"], 0.9, places=6)
            self.assertEqual(result["score_calibrators"]["region"]["positive_count"], 5)
            self.assertEqual(result["score_calibrators"]["region"]["negative_count"], 1)
            self.assertEqual(result["split_sha256"], report["manifest_sha256"]["validation"])
            self.assertEqual(result["score_calibrators"]["text"]["positive_count"], 2)
            self.assertEqual(result["score_calibrators"]["text"]["negative_count"], 1)
            self.assertIn("ece", result["score_calibrators"]["text"]["reliability"])
            self.assertIn("brier", result["score_calibrators"]["region"]["reliability"])
            manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
            sidecar = artifact / "calibration-v4.json"
            self.assertEqual(manifest["calibration_sha256"], hashlib.sha256(sidecar.read_bytes()).hexdigest())
            self.assertTrue(manifest["scores_calibrated"])
            sidecar_data = _validated_calibration(
                json.loads(sidecar.read_text(encoding="utf-8"))
            )
            self.assertTrue(sidecar_data["relation_scores_calibrated"])
            self.assertGreater(
                sidecar_data["relation_score_calibrators"]["RIGHT_CELL"]["edge_positive_count"],
                0,
            )
            manifest["calibration_sha256"] = None
            (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch("prismocr.calibration.PrismOCR.load", side_effect=AssertionError("must recover without fitting")):
                recovered = calibrate_artifact(artifact, validation_manifest)
            self.assertEqual(recovered, sidecar_data)
            recovered_manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                recovered_manifest["calibration_sha256"],
                hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            )

            with self.assertRaisesRegex(
                FileExistsError, "another model, config, vocabulary, split, or IoU threshold"
            ):
                calibrate_artifact(artifact, validation_manifest, iou_threshold=0.75)

            with self.assertRaisesRegex(ValueError, "not the split report's validation manifest"):
                from prismocr.calibration import _validation_split_provenance
                _validation_split_provenance(Path(report["manifests"]["train"]), root / "frozen-splits" / "split-report.json")


if __name__ == "__main__":
    unittest.main()
