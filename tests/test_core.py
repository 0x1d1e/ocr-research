from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from prismocr.api import InferenceOptions, PrismOCR, save_artifact
from prismocr.data.dataset import PrismPageDataset, collate_pages
from prismocr.data.geometry import polygon_to_box
from prismocr.data.schema import AnnotationValidationError, validate_annotation
from prismocr.data.vocabulary import Vocabulary
from prismocr.decode import GridCell, render_table, traverse_reading_order
from prismocr.evaluation import character_error_rate, match_detections, word_error_rate
from prismocr.geometry import polygon_area, rotated_box_to_polygon, rotated_iou
from prismocr.losses import PrismOCRLoss
from prismocr.matching import linear_sum_assignment, match_proposals
from prismocr.model.backbone import AnisotropicRepDW, PrismBackbone
from prismocr.model.heads import RegionQueryHead, TextLineHead
from prismocr.model.network import PrismOCRModel, _interpolate_teacher_box
from prismocr.model.recognizer import minimum_ctc_frames, sample_oriented_strips


class GeometryAndMatchingTests(unittest.TestCase):
    def test_aspect_aware_box_and_iou(self) -> None:
        box = (0.5, 0.5, 0.4, 0.2, 0.0, 1.0)
        polygon = rotated_box_to_polygon(box, (400, 200))
        self.assertAlmostEqual(polygon_area(polygon), 6400.0)
        self.assertAlmostEqual(rotated_iou(box, box, (400, 200)), 1.0)

    def test_concave_polygon_uses_convex_hull_edges_for_box(self) -> None:
        polygon = [(10, 10), (4, 1), (11, 4), (12, 12), (14, 16), (12, 20)]
        hull = [(4, 1), (11, 4), (14, 16), (12, 20)]

        box = polygon_to_box(polygon, (32, 32))
        hull_box = polygon_to_box(hull, (32, 32))

        self.assertEqual(box, hull_box)

    def test_assignment_coverage_is_not_detection_recall(self) -> None:
        gt = [(0.8, 0.8, 0.1, 0.1, 0.0, 1.0)]
        pred = [(0.1, 0.1, 0.1, 0.1, 0.0, 1.0)]
        match = match_proposals(pred, gt, image_size=(100, 100), iou_threshold=0.5)
        self.assertEqual(match.assignment_coverage, 1.0)
        self.assertEqual(match.detection_recall, 0.0)
        self.assertEqual(match.missed_gt_idx, (0,))

    def test_empty_and_rectangular_assignment(self) -> None:
        self.assertEqual(match_proposals([], [], image_size=(10, 10)).assignment_pairs, ())
        rows, cols = linear_sum_assignment([[5.0, 1.0], [2.0, 3.0], [1.0, 4.0]])
        self.assertEqual(list(zip(rows.tolist(), cols.tolist())), [(0, 1), (2, 0)])

    def test_strict_ctc_minimum_counts_repeats(self) -> None:
        self.assertEqual(minimum_ctc_frames([1, 1, 2, 2, 2]), 8)

    def test_selected_strip_sampler_shape_and_empty_path(self) -> None:
        features = torch.randn(2, 4, 8, 8)
        boxes = torch.tensor([[0.5, 0.5, 0.5, 0.2, 0.0, 1.0]])
        sample = sample_oriented_strips(features, boxes, torch.tensor([[1, 0]]), 64)
        self.assertEqual(tuple(sample.shape), (1, 4, 5, 64))
        empty = sample_oriented_strips(features, boxes[:0], torch.empty((0, 2), dtype=torch.long), 64)
        self.assertEqual(tuple(empty.shape), (0, 64, 4))

    def test_antipodal_teacher_and_inference_boxes_share_the_canonical_ocr_direction(self) -> None:
        features = torch.arange(8.0).view(1, 1, 1, 8).expand(1, 1, 8, 8)
        target = torch.tensor([0.5, 0.5, 0.75, 0.25, 0.0, 1.0])
        prediction = target.clone()
        prediction[5] = -1.0
        teacher_forced = _interpolate_teacher_box(prediction, target, 1.0)
        boxes = torch.stack((target, prediction, teacher_forced))
        strips = sample_oriented_strips(
            features, boxes, torch.tensor([[0, 0], [0, 1], [0, 2]]), 4
        )

        torch.testing.assert_close(strips[0], strips[1])
        torch.testing.assert_close(strips[0], strips[2])
        self.assertEqual(strips[0, 0, 2].tolist(), [0.5, 2.5, 4.5, 6.5])

    def test_sparse_strip_sampler_matches_grid_sample_without_page_map_expansion(self) -> None:
        torch.manual_seed(19)
        features = torch.randn(2, 3, 7, 8, requires_grad=True)
        boxes = torch.tensor([
            [0.05, 0.12, 0.7, 0.3, 0.3, 0.953939],
            [0.55, 0.48, 0.42, 0.18, -0.2, 0.979796],
        ], requires_grad=True)
        refs = torch.tensor([[0, 1], [1, 0]])
        output = sample_oriented_strips(features, boxes, refs, 11, strip_height=3)

        image_h, image_w = features.shape[-2] * 4, features.shape[-1] * 4
        sin_theta, cos_theta = boxes[:, 4], boxes[:, 5]
        norm = torch.sqrt(sin_theta.square() + cos_theta.square()).clamp_min(1e-6)
        sin_theta, cos_theta = sin_theta / norm, cos_theta / norm
        theta = torch.zeros((2, 2, 3), dtype=features.dtype)
        theta[:, 0, 0] = cos_theta * boxes[:, 2]
        theta[:, 0, 1] = -sin_theta * boxes[:, 3] * (image_h / image_w)
        theta[:, 0, 2] = 2 * boxes[:, 0] - 1
        theta[:, 1, 0] = sin_theta * boxes[:, 2] * (image_w / image_h)
        theta[:, 1, 1] = cos_theta * boxes[:, 3]
        theta[:, 1, 2] = 2 * boxes[:, 1] - 1
        ys = torch.linspace(-1.0, 1.0, 3)
        xs = torch.linspace(-1.0, 1.0, 11)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        local = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=-1)
        grid = torch.einsum("nhwc,nkc->nhwk", local.unsqueeze(0).expand(2, -1, -1, -1), theta)
        expected = torch.nn.functional.grid_sample(
            features.index_select(0, refs[:, 0]), grid,
            mode="bilinear", padding_mode="zeros", align_corners=False,
        )
        torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-6)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertTrue(torch.isfinite(boxes.grad).all())

    def test_low_score_local_max_remains_available_to_training_matcher(self) -> None:
        proposals = TextLineHead.proposals(
            {
                "heatmap_logits": torch.full((1, 1, 4, 4), -10.0),
                "box_map": torch.zeros(1, 6, 4, 4),
                "role_map": torch.zeros(1, 2, 4, 4),
                "feature_map": torch.zeros(1, 3, 4, 4),
            },
            max_proposals=1,
            score_threshold=0.99,
        )
        self.assertTrue(bool(proposals["candidate_mask"][0, 0]))
        self.assertFalse(bool(proposals["mask"][0, 0]))
        match = match_proposals(
            proposals["boxes"][0],
            proposals["boxes"][0, :1],
            proposals["candidate_mask"][0],
            image_size=(100, 100),
        )
        self.assertEqual(match.accepted_pairs, ((0, 0),))

    def test_center_sweep_recovers_adjacent_distinct_line_centers(self) -> None:
        logits = torch.full((1, 1, 4, 4), -10.0)
        logits[0, 0, 1, 1] = 3.0
        logits[0, 0, 2, 1] = 3.0
        proposals = TextLineHead.proposals(
            {
                "heatmap_logits": logits,
                "box_map": torch.zeros(1, 6, 4, 4),
                "role_map": torch.zeros(1, 2, 4, 4),
                "feature_map": torch.zeros(1, 3, 4, 4),
            },
            max_proposals=4,
            score_threshold=0.5,
        )

        self.assertEqual(int(proposals["candidate_count"][0]), 2)
        self.assertEqual(int(proposals["mask"].sum()), 2)
        self.assertEqual(proposals["proposal_indices"][0, :2].tolist(), [5, 9])

    def test_center_sweep_recovers_long_chains_of_adjacent_distinct_line_centers(self) -> None:
        line_count = 20
        logits = torch.full((1, 1, line_count + 4, 3), -10.0)
        logits[0, 0, 1 : line_count + 1, 1] = 3.0
        box_map = torch.zeros(1, 6, line_count + 4, 3)
        box_map[:, 3] = -3.0
        proposals = TextLineHead.proposals(
            {
                "heatmap_logits": logits,
                "box_map": box_map,
                "role_map": torch.zeros(1, 2, line_count + 4, 3),
                "feature_map": torch.zeros(1, 3, line_count + 4, 3),
            },
            max_proposals=32,
            score_threshold=0.5,
        )

        self.assertEqual(int(proposals["candidate_count"][0]), line_count)
        self.assertEqual(int(proposals["mask"].sum()), line_count)
        self.assertEqual(
            proposals["proposal_indices"][0, :line_count].tolist(),
            [row * 3 + 1 for row in range(1, line_count + 1)],
        )

    def test_long_dense_center_chain_reports_capacity_overflow(self) -> None:
        line_count = 20
        logits = torch.full((1, 1, line_count + 4, 3), -10.0)
        logits[0, 0, 1 : line_count + 1, 1] = 3.0
        box_map = torch.zeros(1, 6, line_count + 4, 3)
        box_map[:, 3] = -3.0
        proposals = TextLineHead.proposals(
            {
                "heatmap_logits": logits,
                "box_map": box_map,
                "role_map": torch.zeros(1, 2, line_count + 4, 3),
                "feature_map": torch.zeros(1, 3, line_count + 4, 3),
            },
            max_proposals=8,
            score_threshold=0.5,
        )

        self.assertEqual(int(proposals["candidate_count"][0]), line_count)
        self.assertTrue(bool(proposals["capacity_overflow"][0]))

    def test_center_sweep_retains_dense_equal_score_distinct_centers_and_reports_overflow(self) -> None:
        logits = torch.full((1, 1, 24, 24), -10.0)
        logits[:, :, 2:22, 2:22] = 3.0
        box_map = torch.zeros(1, 6, 24, 24)
        box_map[:, 2:4] = -3.0
        proposals = TextLineHead.proposals(
            {
                "heatmap_logits": logits,
                "box_map": box_map,
                "role_map": torch.zeros(1, 2, 24, 24),
                "feature_map": torch.zeros(1, 3, 24, 24),
            },
            max_proposals=128,
            score_threshold=0.55,
        )

        self.assertEqual(int(proposals["candidate_count"][0]), 400)
        self.assertEqual(int(proposals["mask"].sum()), 128)
        self.assertTrue(bool(proposals["capacity_overflow"][0]))

    def test_suppressed_same_center_shoulder_cannot_suppress_a_distinct_line(self) -> None:
        logits = torch.full((1, 1, 6, 3), -10.0)
        logits[0, 0, 1, 1] = 4.0
        logits[0, 0, 2, 1] = 3.0
        logits[0, 0, 3, 1] = 2.0
        box_map = torch.zeros(1, 6, 6, 3)
        box_map[0, 1, 1, 1] = 8.0
        box_map[0, 1, 2, 1] = -8.0
        box_map[0, 3, 1, 1] = -1.386
        box_map[0, 3, 2, 1] = 2.087
        box_map[0, 3, 3, 1] = 2.087
        proposals = TextLineHead.proposals(
            {
                "heatmap_logits": logits,
                "box_map": box_map,
                "role_map": torch.zeros(1, 2, 6, 3),
                "feature_map": torch.zeros(1, 3, 6, 3),
            },
            max_proposals=4,
            score_threshold=0.55,
        )

        self.assertEqual(int(proposals["candidate_count"][0]), 2)
        self.assertEqual(proposals["proposal_indices"][0, :2].tolist(), [4, 10])

    def test_uniform_zero_logit_map_has_no_high_threshold_text_candidates(self) -> None:
        logits = torch.zeros(1, 1, 16, 16)
        proposals = TextLineHead.proposals(
            {
                "heatmap_logits": logits,
                "box_map": torch.zeros(1, 6, 16, 16),
                "role_map": torch.zeros(1, 2, 16, 16),
                "feature_map": torch.zeros(1, 3, 16, 16),
            },
            max_proposals=16,
            score_threshold=0.55,
        )

        self.assertEqual(int(proposals["candidate_count"][0]), 0)
        self.assertFalse(bool(proposals["mask"].any()))

    def test_center_sweep_suppresses_adjacent_peaks_decoding_to_one_center(self) -> None:
        logits = torch.full((1, 1, 4, 4), -10.0)
        logits[0, 0, 1, 1] = 3.0
        logits[0, 0, 2, 1] = 2.0
        box_map = torch.zeros(1, 6, 4, 4)
        box_map[0, 1, 1, 1] = 8.0
        box_map[0, 1, 2, 1] = -8.0
        proposals = TextLineHead.proposals(
            {
                "heatmap_logits": logits,
                "box_map": box_map,
                "role_map": torch.zeros(1, 2, 4, 4),
                "feature_map": torch.zeros(1, 3, 4, 4),
            },
            max_proposals=4,
            score_threshold=0.5,
        )

        self.assertEqual(int(proposals["candidate_count"][0]), 1)
        self.assertEqual(int(proposals["mask"].sum()), 1)


class VocabularyAndDecoderTests(unittest.TestCase):
    def test_unicode_oov_is_visible_and_ctc_collapses_repeats(self) -> None:
        vocab = Vocabulary.from_texts(["a"], max_size=8)
        encoded = vocab.encode("a🙂a")
        self.assertEqual(encoded.oov_positions, (1,))
        decoded = vocab.decode_ctc([vocab.tokens.index("a"), vocab.tokens.index("a"), 0, 1, 1])
        self.assertEqual(decoded.text, "a\ufffd")
        self.assertEqual(decoded.oov_positions, (1,))

    def test_cer_does_not_normalize_unicode_implicitly(self) -> None:
        self.assertGreater(character_error_rate("é", "e\u0301"), 0.0)

    def test_word_error_rate_uses_token_edits_and_assignment_coverage_uses_ground_truth(self) -> None:
        self.assertEqual(word_error_rate("alpha beta gamma", "alpha gamma"), 1 / 3)
        _, counts = match_detections(
            [(0.1, 0.1, 0.1, 0.1, 0.0, 1.0)],
            [(0.8, 0.8, 0.1, 0.1, 0.0, 1.0)],
            (100, 100),
        )
        self.assertEqual(counts.recall, 0.0)
        self.assertEqual(counts.assignment_coverage, 1.0)

    def test_cycle_break_keeps_every_node(self) -> None:
        ids = ["a", "b", "c"]
        scores = {
            "a": {"b": 0.9},
            "b": {"c": 0.8},
            "c": {"a": 0.1},
        }
        boxes = {key: (0.2 + index * 0.2, 0.5, 0.1, 0.1, 0.0, 1.0) for index, key in enumerate(ids)}
        order, warnings = traverse_reading_order(ids, scores, {key: 0.0 for key in ids}, boxes)
        self.assertEqual(set(order), set(ids))
        self.assertEqual(len(order), 3)
        self.assertTrue(any(item["code"] == "NEXT_CYCLE_BROKEN" for item in warnings))

    def test_table_escaping_and_overlap_rejection(self) -> None:
        table = render_table("safe-id", [
            GridCell("h", 0, 0, "<header>", is_header=True),
            GridCell("c", 1, 0, "value"),
        ])
        self.assertEqual(table.status, "valid")
        self.assertIn("&lt;header&gt;", table.html or "")
        self.assertIn("&lt;header&gt;", table.markdown or "")
        conflict = render_table("t", [
            GridCell("a", 0, 0, rowspan=2),
            GridCell("b", 1, 0),
        ])
        self.assertEqual(conflict.status, "conflict")
        self.assertIsNone(conflict.html)


class DataAndModelTests(unittest.TestCase):
    @staticmethod
    def _annotation() -> dict[str, object]:
        return {
            "schema_version": "4.0",
            "document_id": "d1",
            "page_id": "d1:p0",
            "image": {"width": 19, "height": 17, "sha256": "0" * 64},
            "provenance": {
                "source_class": "synthetic",
                "license": "generated",
                "permission_reference": "not_applicable",
                "reviewer": "unit_test",
            },
            "lines": [{
                "id": "l1",
                "polygon_px": [[1, 1], [10, 1], [10, 5], [1, 5]],
                "text": "aa",
                "role": "TITLE",
            }],
            "regions": [],
            "cells": [],
            "relations": [],
            "reading_order": ["l1"],
        }

    def test_schema_accepts_partially_labeled_cell_grid(self) -> None:
        annotation = self._annotation()
        annotation["regions"] = [{
            "id": "t1",
            "polygon_px": [[1, 6], [18, 6], [18, 16], [1, 16]],
            "kind": "TABLE",
        }]
        annotation["cells"] = [{
            "id": "c1",
            "polygon_px": [[1, 6], [10, 6], [10, 16], [1, 16]],
            "table_id": "t1",
            "row": 0,
        }]
        validate_annotation(annotation)

        annotation["cells"][0]["rowspan"] = True
        with self.assertRaisesRegex(AnnotationValidationError, "rowspan must be a positive integer"):
            validate_annotation(annotation)

    def test_unassigned_cell_may_have_null_table_reference(self) -> None:
        annotation = self._annotation()
        annotation["cells"] = [{
            "id": "c1",
            "polygon_px": [[1, 6], [10, 6], [10, 16], [1, 16]],
            "table_id": None,
        }]
        validate_annotation(annotation)

    def test_partial_grid_cell_still_requires_an_existing_table(self) -> None:
        annotation = self._annotation()
        annotation["regions"] = [{
            "id": "t1",
            "polygon_px": [[1, 6], [18, 6], [18, 16], [1, 16]],
            "kind": "TABLE",
        }]
        annotation["cells"] = [{
            "id": "c1",
            "polygon_px": [[1, 6], [10, 6], [10, 16], [1, 16]],
            "table_id": "missing-table",
            "row": 0,
        }]

        with self.assertRaisesRegex(AnnotationValidationError, "missing TABLE region 'missing-table'"):
            validate_annotation(annotation)

    def test_schema_rejects_cycles_in_combined_parent_contains_hierarchy(self) -> None:
        annotation = self._annotation()
        annotation["lines"] = []
        annotation["reading_order"] = []
        annotation["regions"] = [
            {"id": "r1", "polygon_px": [[1, 1], [10, 1], [10, 10], [1, 10]], "kind": "REGION"},
            {"id": "r2", "polygon_px": [[10, 10], [18, 10], [18, 16], [10, 16]], "kind": "REGION"},
        ]
        annotation["relations"] = [
            {"type": "PARENT", "source": "r1", "target": "r2"},
            {"type": "CONTAINS", "source": "r1", "target": "r2"},
        ]

        with self.assertRaisesRegex(AnnotationValidationError, "HIERARCHY graph contains a cycle"):
            validate_annotation(annotation)

    def test_schema_limits_each_key_to_one_value_target(self) -> None:
        annotation = self._annotation()
        annotation["lines"] = [
            {"id": "key", "polygon_px": [[1, 1], [5, 1], [5, 5], [1, 5]], "text": "k", "role": "KEY"},
            {"id": "value-a", "polygon_px": [[6, 1], [10, 1], [10, 5], [6, 5]], "text": "a", "role": "VALUE"},
            {"id": "value-b", "polygon_px": [[11, 1], [15, 1], [15, 5], [11, 5]], "text": "b", "role": "VALUE"},
        ]
        annotation["reading_order"] = ["key", "value-a", "value-b"]
        annotation["relations"] = [
            {"type": "KV_VALUE", "source": "key", "target": "value-a"},
            {"type": "KV_VALUE", "source": "key", "target": "value-b"},
        ]

        with self.assertRaisesRegex(AnnotationValidationError, "multiple outgoing KV_VALUE"):
            validate_annotation(annotation)

    def test_schema_allows_conflicting_grid_labels_only_when_table_is_ambiguous(self) -> None:
        annotation = self._annotation()
        annotation["lines"] = []
        annotation["reading_order"] = []
        annotation["regions"] = [{
            "id": "t1",
            "polygon_px": [[1, 6], [18, 6], [18, 16], [1, 16]],
            "kind": "TABLE",
        }]
        annotation["cells"] = [
            {
                "id": cell_id,
                "polygon_px": [[x, 7], [x + 5, 7], [x + 5, 15], [x, 15]],
                "table_id": "t1",
                "row": 0,
                "col": 0,
            }
            for cell_id, x in (("c1", 2), ("c2", 10))
        ]
        with self.assertRaisesRegex(AnnotationValidationError, "overlap at grid slot"):
            validate_annotation(annotation)

        annotation["regions"][0]["annotation_ambiguous"] = True
        validate_annotation(annotation)

        annotation["regions"][0]["annotation_ambiguous"] = "true"
        with self.assertRaisesRegex(AnnotationValidationError, "annotation_ambiguous must be boolean"):
            validate_annotation(annotation)

    def test_schema_rejects_invalid_next_endpoint_types(self) -> None:
        annotation = self._annotation()
        annotation["relations"] = [{"type": "NEXT", "source": "l1", "target": "missing"}]
        with self.assertRaises(AnnotationValidationError):
            validate_annotation(annotation)

    def test_schema_rejects_invalid_cell_table_ids_without_crashing(self) -> None:
        for table_id in (["table"], {"table": "table"}):
            with self.subTest(table_id=table_id):
                annotation = self._annotation()
                annotation["regions"] = [{
                    "id": "table",
                    "polygon_px": [[0, 0], [12, 0], [12, 8], [0, 8]],
                    "kind": "TABLE",
                }]
                annotation["cells"] = [{
                    "id": "cell",
                    "polygon_px": [[2, 2], [6, 2], [6, 5], [2, 5]],
                    "table_id": table_id,
                }]
                annotation["relations"] = [{
                    "type": "PARENT", "source": "cell", "target": "table",
                }]

                with self.assertRaisesRegex(
                    AnnotationValidationError,
                    "table_id must be a non-empty string or null",
                ):
                    validate_annotation(annotation)

    def test_schema_rejects_malformed_polygon_in_hierarchy_without_crashing(self) -> None:
        for relation in ("PARENT", "CONTAINS"):
            with self.subTest(relation=relation):
                annotation = self._annotation()
                annotation["lines"][0]["polygon_px"][0] = {}
                annotation["regions"] = [{
                    "id": "parent",
                    "polygon_px": [[0, 0], [12, 0], [12, 8], [0, 8]],
                    "kind": "FORM_FIELD",
                }]
                annotation["relations"] = [{
                    "type": relation,
                    "source": "l1" if relation == "PARENT" else "parent",
                    "target": "parent" if relation == "PARENT" else "l1",
                }]

                with self.assertRaisesRegex(AnnotationValidationError, "polygon_px\\[0\\] must be an x,y pair"):
                    validate_annotation(annotation)

    def test_schema_rejects_malformed_polygon_point_shapes(self) -> None:
        for polygon in (
            ["11", "21", "22", "12"],
            [[1, 1, 0], [10, 1, 0], [10, 5, 0], [1, 5, 0]],
            [["1", 1], [10, 1], [10, 5], [1, 5]],
            [[True, 1], [10, 1], [10, 5], [1, 5]],
            [{"x": 1, "y": 1}, {"x": 10, "y": 1}, {"x": 10, "y": 5}, {"x": 1, "y": 5}],
        ):
            with self.subTest(polygon=polygon):
                annotation = self._annotation()
                annotation["lines"][0]["polygon_px"] = polygon
                with self.assertRaises(AnnotationValidationError):
                    validate_annotation(annotation)

    def test_schema_rejects_self_intersecting_polygons(self) -> None:
        annotation = self._annotation()
        annotation["lines"][0]["polygon_px"] = [[1, 1], [9, 9], [1, 9], [7, 1]]

        with self.assertRaisesRegex(
            AnnotationValidationError, "must be a simple non-self-intersecting polygon"
        ):
            validate_annotation(annotation)

    def test_schema_rejects_excessive_polygon_vertex_counts(self) -> None:
        annotation = self._annotation()
        annotation["lines"][0]["polygon_px"] = [
            [1 + index % 4, 1 + index // 4] for index in range(5000)
        ]

        with self.assertRaisesRegex(AnnotationValidationError, "simplicity limit"):
            validate_annotation(annotation)

    def test_schema_rejects_huge_polygon_coordinates_without_crashing(self) -> None:
        annotation = self._annotation()
        annotation["lines"][0]["polygon_px"][0][0] = 10**400

        with self.assertRaisesRegex(AnnotationValidationError, "polygon_px\\[0\\] must contain numbers"):
            validate_annotation(annotation)

    def test_schema_checks_huge_cell_spans_without_expanding_grid_slots(self) -> None:
        annotation = self._annotation()
        annotation["cells"] = [{
            "id": "huge-span",
            "polygon_px": [[1, 6], [5, 6], [5, 10], [1, 10]],
            "table_id": "missing-table",
            "row": 0,
            "col": 0,
            "rowspan": 10**400,
        }]

        with self.assertRaisesRegex(
            AnnotationValidationError, "missing TABLE region 'missing-table'"
        ):
            validate_annotation(annotation)

        annotation["regions"] = [{
            "id": "table",
            "polygon_px": [[0, 0], [18, 0], [18, 16], [0, 16]],
            "kind": "TABLE",
        }]
        annotation["cells"] = [
            {
                "id": cell_id,
                "polygon_px": polygon,
                "table_id": "table",
                "row": 0,
                "col": 0,
                "rowspan": 10**400,
                "colspan": 10**400,
            }
            for cell_id, polygon in (
                ("huge-a", [[1, 6], [5, 6], [5, 10], [1, 10]]),
                ("huge-b", [[6, 6], [10, 6], [10, 10], [6, 10]]),
            )
        ]
        with self.assertRaisesRegex(AnnotationValidationError, "overlap at grid slot \\(0,0\\)"):
            validate_annotation(annotation)

    def test_schema_accepts_long_acyclic_reading_order_without_recursion(self) -> None:
        annotation = self._annotation()
        line_ids = [f"line-{index}" for index in range(1100)]
        polygon = [[1, 1], [10, 1], [10, 5], [1, 5]]
        annotation["lines"] = [
            {
                "id": line_id,
                "polygon_px": polygon,
                "text": "line",
                "role": "PARA_LINE",
            }
            for line_id in line_ids
        ]
        annotation["reading_order"] = line_ids
        annotation["relations"] = [
            {"type": "NEXT", "source": source, "target": target}
            for source, target in zip(line_ids, line_ids[1:])
        ]

        validate_annotation(annotation)

    def test_schema_aggregates_non_array_relations(self) -> None:
        annotation = self._annotation()
        annotation["relations"] = {}

        with self.assertRaisesRegex(AnnotationValidationError, "relations must be an array"):
            validate_annotation(annotation)

    def test_schema_aggregates_unhashable_line_scopes(self) -> None:
        annotation = self._annotation()
        annotation["lines"][0]["scope_id"] = []

        with self.assertRaisesRegex(AnnotationValidationError, "scope_id must be a non-empty string"):
            validate_annotation(annotation)

    def test_schema_reports_malformed_enum_types(self) -> None:
        annotation = self._annotation()
        annotation["lines"][0]["role"] = ["TITLE"]
        with self.assertRaises(AnnotationValidationError):
            validate_annotation(annotation)

        annotation = self._annotation()
        annotation["relations"] = [{"type": ["NEXT"], "source": "l1", "target": None}]
        with self.assertRaises(AnnotationValidationError):
            validate_annotation(annotation)

    def test_schema_rejects_conflicting_parent_positive_and_null_decisions(self) -> None:
        annotation = self._annotation()
        annotation["regions"] = [{
            "id": "parent",
            "polygon_px": [[1, 6], [18, 6], [18, 16], [1, 16]],
            "kind": "FORM_FIELD",
        }]
        annotation["relations"] = [
            {"type": "PARENT", "source": "l1", "target": "parent"},
            {
                "type": "PARENT", "source": "l1", "target": None,
                "annotation_status": "explicit_null",
            },
        ]
        with self.assertRaisesRegex(AnnotationValidationError, "multiple PARENT decisions"):
            validate_annotation(annotation)

    def test_schema_accepts_consistent_parent_and_contains_views(self) -> None:
        annotation = self._annotation()
        annotation["regions"] = [{
            "id": "parent",
            "polygon_px": [[1, 6], [18, 6], [18, 16], [1, 16]],
            "kind": "FORM_FIELD",
        }]
        annotation["regions"][0]["polygon_px"] = [[0, 0], [12, 0], [12, 8], [0, 8]]
        annotation["relations"] = [
            {"type": "PARENT", "source": "l1", "target": "parent"},
            {"type": "CONTAINS", "source": "parent", "target": "l1"},
        ]
        validate_annotation(annotation)

    def test_schema_rejects_inconsistent_parent_and_contains_views(self) -> None:
        annotation = self._annotation()
        annotation["regions"] = [
            {"id": "parent-a", "polygon_px": [[1, 6], [9, 6], [9, 16], [1, 16]], "kind": "FORM_FIELD"},
            {"id": "parent-b", "polygon_px": [[10, 6], [18, 6], [18, 16], [10, 16]], "kind": "FORM_FIELD"},
        ]
        annotation["relations"] = [
            {"type": "PARENT", "source": "l1", "target": "parent-a"},
            {"type": "CONTAINS", "source": "parent-b", "target": "l1"},
        ]
        with self.assertRaisesRegex(AnnotationValidationError, "multiple PARENT decisions"):
            validate_annotation(annotation)

    def test_schema_rejects_parent_annotations_outside_decoder_geometry(self) -> None:
        annotation = self._annotation()
        annotation["regions"] = [{
            "id": "parent",
            "polygon_px": [[1, 6], [18, 6], [18, 16], [1, 16]],
            "kind": "FORM_FIELD",
        }]
        annotation["relations"] = [{
            "type": "CONTAINS", "source": "parent", "target": "l1",
        }]

        with self.assertRaisesRegex(AnnotationValidationError, "CONTAINS is not decoder-compatible: PARENT_GEOMETRY_MISMATCH"):
            validate_annotation(annotation)

    def test_schema_requires_complete_known_table_grid_for_right_cell_edges(self) -> None:
        for table_id, col_labels, expected_error in (
            (None, (0, 1), "TABLE_UNKNOWN"),
            ("table", (None, None), "GRID_POSITION_MISSING"),
        ):
            with self.subTest(table_id=table_id, col_labels=col_labels):
                annotation = self._annotation()
                annotation["regions"] = [{
                    "id": "table",
                    "polygon_px": [[1, 6], [18, 6], [18, 16], [1, 16]],
                    "kind": "TABLE",
                }]
                annotation["cells"] = [
                    {
                        "id": cell_id,
                        "polygon_px": polygon,
                        "table_id": table_id,
                        "row": 0,
                        **({"col": col} if col is not None else {}),
                    }
                    for cell_id, polygon, col in (
                        ("left", [[2, 8], [7, 8], [7, 14], [2, 14]], col_labels[0]),
                        ("right", [[10, 8], [15, 8], [15, 14], [10, 14]], col_labels[1]),
                    )
                ]
                annotation["relations"] = [{
                    "type": "RIGHT_CELL", "source": "left", "target": "right",
                }]

                with self.assertRaisesRegex(
                    AnnotationValidationError,
                    f"RIGHT_CELL is not decoder-compatible: {expected_error}",
                ):
                    validate_annotation(annotation)

    def test_schema_accepts_decoder_compatible_right_cell_edge(self) -> None:
        annotation = self._annotation()
        annotation["regions"] = [{
            "id": "table",
            "polygon_px": [[1, 6], [18, 6], [18, 16], [1, 16]],
            "kind": "TABLE",
        }]
        annotation["cells"] = [
            {
                "id": cell_id,
                "polygon_px": polygon,
                "table_id": "table",
                "row": 0,
                "col": col,
            }
            for cell_id, polygon, col in (
                ("left", [[2, 8], [7, 8], [7, 14], [2, 14]], 0),
                ("right", [[10, 8], [15, 8], [15, 14], [10, 14]], 1),
            )
        ]
        annotation["relations"] = [{
            "type": "RIGHT_CELL", "source": "left", "target": "right",
        }]

        validate_annotation(annotation)

    def test_explicit_next_decisions_must_follow_scoped_reading_order(self) -> None:
        annotation = self._annotation()
        annotation["lines"].append({
            "id": "l2",
            "polygon_px": [[11, 1], [18, 1], [18, 5], [11, 5]],
            "text": "b",
            "role": "PARA_LINE",
        })
        annotation["reading_order"] = ["l1", "l2"]
        annotation["relations"] = [{"type": "NEXT", "source": "l2", "target": "l1"}]

        with self.assertRaisesRegex(AnnotationValidationError, "NEXT decision for 'l2' contradicts reading_order"):
            validate_annotation(annotation)

    def test_schema_rejects_null_contains_decisions(self) -> None:
        annotation = self._annotation()
        annotation["regions"] = [{
            "id": "parent",
            "polygon_px": [[1, 6], [18, 6], [18, 16], [1, 16]],
            "kind": "FORM_FIELD",
        }]
        annotation["relations"] = [{
            "type": "CONTAINS", "source": "parent", "target": None,
            "annotation_status": "explicit_null",
        }]
        with self.assertRaisesRegex(AnnotationValidationError, "CONTAINS cannot have a NULL target"):
            validate_annotation(annotation)

    def test_schema_checks_source_type_for_explicit_null_relations(self) -> None:
        annotation = self._annotation()
        annotation["relations"] = [{
            "type": "KV_VALUE",
            "source": "l1",
            "target": None,
            "annotation_status": "explicit_null",
        }]
        with self.assertRaises(AnnotationValidationError):
            validate_annotation(annotation)

    def test_dataset_padding_preserves_normalized_page_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.fromarray(np.zeros((17, 19, 3), dtype=np.uint8)).save(root / "page.png")
            annotation = self._annotation()
            annotation["image"]["sha256"] = hashlib.sha256((root / "page.png").read_bytes()).hexdigest()
            record = {"image_path": "page.png", "annotation": annotation}
            (root / "manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
            dataset = PrismPageDataset(root / "manifest.jsonl")
            image, target = dataset[0]
            images, targets = collate_pages([(image, target)])
            self.assertEqual(tuple(images.shape), (1, 3, 32, 32))
            box = targets[0]["lines"][0]["box"]
            self.assertAlmostEqual(box[0] * 32, 5.5)
            self.assertAlmostEqual(box[1] * 32, 3.0)

    def test_low_confidence_peaks_do_not_count_toward_capacity(self) -> None:
        result = TextLineHead.proposals(
            {
                "heatmap_logits": torch.full((1, 1, 4, 4), -10.0),
                "box_map": torch.zeros(1, 6, 4, 4),
                "role_map": torch.zeros(1, 2, 4, 4),
                "feature_map": torch.zeros(1, 3, 4, 4),
            },
            max_proposals=1,
            score_threshold=0.55,
        )
        self.assertEqual(int(result["candidate_count"][0]), 0)
        self.assertFalse(bool(result["capacity_overflow"][0]))
        self.assertFalse(bool(result["mask"].any()))

    def test_proposal_heads_can_predict_large_normalized_side_lengths(self) -> None:
        box_map = torch.zeros(1, 6, 1, 1)
        box_map[:, 2:4] = 10.0
        text = TextLineHead.proposals(
            {
                "heatmap_logits": torch.full((1, 1, 1, 1), 10.0),
                "box_map": box_map,
                "role_map": torch.zeros(1, 2, 1, 1),
                "feature_map": torch.zeros(1, 3, 1, 1),
            },
            max_proposals=1,
            score_threshold=0.5,
        )
        self.assertTrue(torch.all(text["boxes"][0, 0, 2:4] > 0.5))

        regions = RegionQueryHead(dim=5, num_queries=1, num_kinds=3)
        for parameter in regions.box.parameters():
            parameter.data.zero_()
        regions.box[-1].bias.data[2:4] = 10.0
        prediction = regions(torch.zeros(1, 5, 1, 1), score_threshold=0.0)
        self.assertTrue(torch.all(prediction["boxes"][0, 0, 2:4] > 0.5))

    def test_backbone_activation_checkpointing_preserves_gradients_and_batch_norm_state(self) -> None:
        torch.manual_seed(7)
        backbone = PrismBackbone(dim=16).train()
        backbone.activation_checkpointing = True
        images = torch.randn(2, 3, 32, 32, requires_grad=True)
        tracked_bn = backbone.stage4[0].dw.bn_3x3

        features = backbone(images)
        self.assertEqual(int(tracked_bn.num_batches_tracked), 1)
        sum(feature.square().mean() for feature in features).backward()

        self.assertEqual(int(tracked_bn.num_batches_tracked), 1)
        self.assertIsNotNone(images.grad)
        self.assertIsNotNone(backbone.stage4[0].dw.dw_3x3.weight.grad)

    def test_anisotropic_reparameterization_preserves_output(self) -> None:
        torch.manual_seed(7)
        block = AnisotropicRepDW(4).eval()
        image = torch.randn(2, 4, 12, 16)
        with torch.no_grad():
            expected = block(image)
        block.fuse_for_inference()
        with torch.no_grad():
            actual = block(image)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_shared_match_ctc_training_and_artifact_api(self) -> None:
        torch.manual_seed(11)
        vocabulary = Vocabulary.from_texts(["aa", "a" * 70], max_size=16)
        model = PrismOCRModel(
            dim=160,
            text_proposals=8,
            region_queries=4,
            vocab_size=len(vocabulary),
            vocabulary=vocabulary,
        )
        original_text_forward = model.text_head.forward

        def two_peak_text_forward(features: torch.Tensor) -> dict[str, torch.Tensor]:
            raw = original_text_forward(features)
            logits = torch.full_like(raw["heatmap_logits"], -10.0)
            logits[:, :, 1, 1] = 10.0
            logits[:, :, 1, 5] = 9.0
            raw["heatmap_logits"] = logits
            return raw

        model.text_head.forward = two_peak_text_forward
        images = torch.rand(1, 3, 64, 64)
        model.eval()
        with torch.no_grad():
            reference = model(images)
        target = [{
            "page_id": "p0",
            "lines": [
                {"id": "l0", "box": reference["text"]["boxes"][0, 0].tolist(), "text": "aa", "role": "TITLE"},
                {"id": "l1", "box": reference["text"]["boxes"][0, 1].tolist(), "text": "a" * 70, "role": "PARA_LINE"},
            ],
            "regions": [],
            "cells": [],
            "relations": [],
        }]
        outputs = model(images, target, teacher_force_ratio=1.0)
        self.assertEqual(set(outputs["ocr_groups"]), {64, 160})
        losses = PrismOCRLoss(vocabulary)(outputs, target)
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertTrue(any(param.grad is not None for param in model.recognizer.parameters()))

        with tempfile.TemporaryDirectory() as tmp:
            save_artifact(tmp, model, vocabulary)
            calibration_path = Path(tmp) / "calibration-v4.json"
            calibration_path.write_text(
                json.dumps({"scores_calibrated": True, "text_threshold": 0.99}),
                encoding="utf-8",
            )
            save_artifact(tmp, model, vocabulary)
            self.assertFalse(calibration_path.exists())
            artifact_manifest = json.loads((Path(tmp) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(artifact_manifest["model_tensor_bytes"], model.tensor_bytes())
            api = PrismOCR.load(tmp)
            self.assertFalse(api.calibration.get("scores_calibrated", False))
            result = api.predict(np.zeros((64, 64, 3), dtype=np.uint8)).to_dict()
            self.assertEqual(result["schema_version"], "4.0")
            self.assertEqual(result["image"], {"width": 64, "height": 64})
            self.assertIn("SCORES_UNCALIBRATED", {warning["code"] for warning in result["warnings"]})
            self.assertLess(api.model.tensor_bytes(), 10_000_000)
            artifact_manifest["model_tensor_bytes"] += 1
            (Path(tmp) / "manifest.json").write_text(
                json.dumps(artifact_manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "tensor inventory does not match"):
                PrismOCR.load(tmp)

    def test_manifest_free_load_still_enforces_fp32_tensor_inventory_ceiling(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        model = PrismOCRModel(vocab_size=len(vocabulary), vocabulary=vocabulary)
        with tempfile.TemporaryDirectory() as tmp:
            save_artifact(tmp, model, vocabulary)
            (Path(tmp) / "manifest.json").unlink()
            with patch("prismocr.api._MAX_FP32_TENSOR_BYTES", model.tensor_bytes() - 1):
                with self.assertRaisesRegex(ValueError, "exceeding the .*byte FP32 limit"):
                    PrismOCR.load(tmp)

    def test_artifact_loader_authenticates_configuration_and_manifest_schema(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        model = PrismOCRModel(vocab_size=len(vocabulary), vocabulary=vocabulary)
        with tempfile.TemporaryDirectory() as tmp:
            save_artifact(tmp, model, vocabulary)
            config_path = Path(tmp) / "model-config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["text_threshold"] = 0.91
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model_config_sha256.*does not match"):
                PrismOCR.load(tmp)

        with tempfile.TemporaryDirectory() as tmp:
            save_artifact(tmp, model, vocabulary)
            manifest_path = Path(tmp) / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = "99.0"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported artifact manifest schema_version"):
                PrismOCR.load(tmp)

    def test_manifest_free_artifact_is_rejected_after_size_inventory_check(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        model = PrismOCRModel(vocab_size=len(vocabulary), vocabulary=vocabulary)
        with tempfile.TemporaryDirectory() as tmp:
            save_artifact(tmp, model, vocabulary)
            (Path(tmp) / "manifest.json").unlink()
            with self.assertRaisesRegex(ValueError, "artifact manifest is required"):
                PrismOCR.load(tmp)

    def test_save_artifact_rejects_fused_loaded_model_without_replacing_bundle(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        model = PrismOCRModel(vocab_size=len(vocabulary), vocabulary=vocabulary)
        with tempfile.TemporaryDirectory() as tmp:
            save_artifact(tmp, model, vocabulary)
            loaded = PrismOCR.load(tmp)
            root = Path(tmp)
            before = {
                name: (root / name).read_bytes()
                for name in ("vocab-v4.json", "model-config.json", "model.pt", "manifest.json")
            }

            with self.assertRaisesRegex(ValueError, "requires an unfused training model"):
                save_artifact(root, loaded.model, vocabulary)

            self.assertEqual(
                {name: (root / name).read_bytes() for name in before}, before
            )
            PrismOCR.load(root)

    def test_failed_artifact_save_preserves_the_previous_calibrated_bundle(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        model = PrismOCRModel(vocab_size=len(vocabulary), vocabulary=vocabulary)
        with tempfile.TemporaryDirectory() as tmp:
            save_artifact(tmp, model, vocabulary)
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            calibration = {
                "model_sha256": manifest["model_sha256"],
                "model_config_sha256": manifest["model_config_sha256"],
                "vocabulary_sha256": manifest["vocabulary_sha256"],
                "study_marker": "last-known-good",
            }
            calibration_path = root / "calibration-v4.json"
            calibration_bytes = (json.dumps(calibration) + "\n").encode("utf-8")
            calibration_path.write_bytes(calibration_bytes)
            manifest["calibration_sha256"] = hashlib.sha256(calibration_bytes).hexdigest()
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            PrismOCR.load(root)
            before = {
                name: (root / name).read_bytes()
                for name in ("vocab-v4.json", "model-config.json", "model.pt", "calibration-v4.json", "manifest.json")
            }

            with patch("prismocr.api.torch.save", side_effect=OSError("simulated disk failure")):
                with self.assertRaisesRegex(OSError, "simulated disk failure"):
                    save_artifact(root, model, vocabulary)

            self.assertEqual(
                {name: (root / name).read_bytes() for name in before}, before
            )
            self.assertEqual(
                PrismOCR.load(root).calibration["study_marker"], "last-known-good"
            )

            from os import replace as original_replace

            def fail_manifest_publish(source: object, destination: object) -> None:
                if (
                    Path(destination) == manifest_path
                    and Path(source).parent.name != "previous"
                ):
                    raise OSError("simulated manifest publication failure")
                original_replace(source, destination)

            with patch("prismocr.api.os.replace", side_effect=fail_manifest_publish):
                with self.assertRaisesRegex(OSError, "manifest publication failure"):
                    save_artifact(root, model, vocabulary)

            self.assertEqual(
                {name: (root / name).read_bytes() for name in before}, before
            )
            self.assertEqual(
                PrismOCR.load(root).calibration["study_marker"], "last-known-good"
            )

    def test_save_artifact_enforces_fp32_tensor_inventory_ceiling(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        oversized = SimpleNamespace(vocab_size=len(vocabulary), tensor_bytes=lambda: 10_000_001)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "artifact"
            with self.assertRaisesRegex(ValueError, "exceeding the 10000000-byte FP32 limit"):
                save_artifact(output, oversized, vocabulary)  # type: ignore[arg-type]
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
