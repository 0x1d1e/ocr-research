from __future__ import annotations

import unittest

import torch

from prismocr.data.vocabulary import Vocabulary
from prismocr.losses import PrismOCRLoss
from prismocr.matching import match_proposals
from prismocr.model.network import (
    REGION_KIND_TO_ID,
    RELATION_NAMES,
    TEXT_ROLE_TO_ID,
    PrismOCRModel,
)


def _box(cx: float, cy: float, width: float, height: float) -> list[float]:
    return [cx, cy, width, height, 0.0, 1.0]


def _matched(boxes: list[list[float]], items: list[dict[str, object]], classes: torch.Tensor | None = None):
    predicted = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 6)
    truths = torch.tensor([item["box"] for item in items], dtype=torch.float32).reshape(-1, 6)
    gt_classes = None
    if classes is not None:
        gt_classes = torch.tensor([
            TEXT_ROLE_TO_ID[item["role"]] if "role" in item else REGION_KIND_TO_ID[item["kind"]]
            for item in items
        ])
    return match_proposals(
        predicted,
        truths,
        torch.ones(len(predicted), dtype=torch.bool),
        image_size=(100, 100),
        pred_class_logits=classes,
        gt_classes=gt_classes,
        iou_threshold=0.5,
    )


class TrainingContractTests(unittest.TestCase):
    def test_empty_page_backward_has_finite_gradients(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        model = PrismOCRModel(
            dim=10, text_proposals=4, region_queries=2, vocab_size=len(vocabulary),
            vocabulary=vocabulary,
        )
        target = {"page_id": "blank:page-0", "lines": [], "regions": [], "cells": [], "relations": []}

        outputs = model(torch.zeros((1, 3, 32, 32)), [target], teacher_force_ratio=0.0)
        losses = PrismOCRLoss(vocabulary)(outputs, [target])
        losses["total"].backward()

        non_finite = [
            name for name, parameter in model.named_parameters()
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
        ]
        self.assertEqual(non_finite, [])

    def test_heatmap_anchor_matches_decoded_offset_cell(self) -> None:
        target = {"lines": [{"box": _box(0.2, 0.2, 0.1, 0.1)}]}

        rendered, collisions, anchors = PrismOCRLoss._heatmap_targets(
            [target], (4, 4), torch.device("cpu"), torch.float32
        )

        self.assertEqual(tuple(torch.nonzero(rendered[0, 0] == 1.0)[0].tolist()), (0, 0))
        self.assertEqual(tuple(torch.nonzero(anchors[0, 0])[0].tolist()), (0, 0))
        self.assertEqual(collisions, 0)

    def test_heatmap_collision_is_counted_without_dropping_ground_truth(self) -> None:
        # Two ground-truth centers quantize into the same grid cell; the shared
        # anchor and its collision must both be reported rather than hidden.
        target = {"lines": [
            {"box": _box(0.20, 0.20, 0.1, 0.1)},
            {"box": _box(0.22, 0.22, 0.1, 0.1)},
        ]}

        rendered, collisions, anchors = PrismOCRLoss._heatmap_targets(
            [target], (4, 4), torch.device("cpu"), torch.float32
        )

        self.assertEqual(collisions, 1)
        self.assertEqual(tuple(torch.nonzero(anchors[0, 0])[0].tolist()), (0, 0))
        self.assertEqual(int((rendered[0, 0] == 1.0).sum()), 1)

    def test_heatmap_supervises_only_the_integer_anchor_not_gaussian_shoulders(self) -> None:
        target = {"lines": [{"box": _box(0.5, 0.5, 0.8, 0.6)}]}

        rendered, _, anchors = PrismOCRLoss._heatmap_targets(
            [target], (60, 60), torch.device("cpu"), torch.float32
        )

        self.assertEqual(int(anchors.sum()), 1)
        # Large lines spread a Gaussian well past 0.99, which the previous
        # value-threshold rule counted as additional positive detection centers.
        self.assertGreater(int((rendered[0, 0] >= 0.99).sum()), 1)
        logits = torch.zeros_like(rendered, requires_grad=True)
        loss = PrismOCRLoss._focal_heatmap(logits, rendered, anchors)
        loss.backward()
        gradient = logits.grad[0, 0]
        shoulder = (rendered[0, 0] >= 0.99) & ~anchors[0, 0]
        anchor_y, anchor_x = (int(value) for value in torch.nonzero(anchors[0, 0])[0])
        self.assertLess(
            float(gradient[shoulder].abs().max()),
            float(gradient[anchor_y, anchor_x].abs()),
        )

    def _outputs(
        self,
        vocabulary: Vocabulary,
        lines: list[dict[str, object]],
        regions: list[dict[str, object]],
        *,
        ctc_bucket: int | None = None,
        ctc_frames: int | None = None,
    ) -> dict[str, object]:
        text_boxes = [item["box"] for item in lines]
        region_boxes = [item["box"] for item in regions]
        text_class_logits = torch.zeros((len(text_boxes), len(TEXT_ROLE_TO_ID)))
        region_class_logits = torch.zeros((len(region_boxes), len(REGION_KIND_TO_ID)))
        text_match = _matched(text_boxes, lines, text_class_logits)
        region_match = _matched(region_boxes, regions, region_class_logits)
        node_count = len(text_boxes) + len(region_boxes)
        padded_node_count = node_count + 2
        graph_logits = torch.zeros(
            (1, len(RELATION_NAMES), padded_node_count, padded_node_count + 1),
            requires_grad=True,
        )
        ocr_groups: dict[int, dict[str, torch.Tensor]] = {}
        if ctc_bucket is not None:
            ocr_groups[ctc_bucket] = {
                "refs": torch.tensor([[0, 0]], dtype=torch.long),
                "logits": torch.zeros((1, ctc_frames or ctc_bucket, len(vocabulary)), requires_grad=True),
            }
        return {
            "text_heatmap_logits": torch.zeros((1, 1, 4, 4), requires_grad=True),
            "text": {
                "boxes": torch.tensor(text_boxes, dtype=torch.float32).reshape(1, len(text_boxes), 6),
                "role_logits": torch.zeros((1, len(text_boxes), len(TEXT_ROLE_TO_ID)), requires_grad=True),
            },
            "regions": {
                "boxes": torch.tensor(region_boxes, dtype=torch.float32).reshape(1, len(region_boxes), 6),
                "class_logits": torch.zeros((1, len(region_boxes), len(REGION_KIND_TO_ID)), requires_grad=True),
                "row_logits": torch.zeros((1, len(region_boxes), 128), requires_grad=True),
                "col_logits": torch.zeros((1, len(region_boxes), 128), requires_grad=True),
                "rowspan_logits": torch.zeros((1, len(region_boxes), 32), requires_grad=True),
                "colspan_logits": torch.zeros((1, len(region_boxes), 32), requires_grad=True),
                "header_logits": torch.zeros((1, len(region_boxes)), requires_grad=True),
            },
            "assignments": {"text": (text_match,), "regions": (region_match,)},
            "graph_logits": graph_logits,
            "ocr_groups": ocr_groups,
            "ocr_unsupported": (),
        }

    def test_reading_order_only_annotations_supervise_next_edges(self) -> None:
        vocabulary = Vocabulary.from_texts(["a", "b"], max_size=8)
        lines = [
            {"id": "line-1", "box": _box(0.25, 0.5, 0.2, 0.1), "text": "a", "role": "PARA_LINE"},
            {"id": "line-2", "box": _box(0.55, 0.5, 0.2, 0.1), "text": "b", "role": "PARA_LINE"},
        ]
        target = {
            "page_id": "reading-order-only",
            "lines": lines,
            "regions": [],
            "cells": [],
            "relations": [],
            "reading_order": ["line-1", "line-2"],
        }
        outputs = self._outputs(vocabulary, lines, [])

        losses = PrismOCRLoss(vocabulary)(outputs, [target])
        losses["graph"].backward()

        gradients = outputs["graph_logits"].grad[0, RELATION_NAMES.index("NEXT")]
        self.assertNotEqual(float(gradients[0, 1]), 0.0)
        self.assertNotEqual(float(gradients[1, -1]), 0.0)
        self.assertEqual(losses["unmatched_relation_count"], 0)

    def test_contains_annotations_supervise_the_decoded_parent_channel(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        line = {
            "id": "line-1", "box": _box(0.5, 0.5, 0.2, 0.1),
            "polygon_px": [[4, 4], [16, 4], [16, 8], [4, 8]],
            "text": "a", "role": "PARA_LINE",
        }
        table = {
            "id": "table-1", "box": _box(0.5, 0.5, 0.8, 0.6),
            "polygon_px": [[0, 0], [20, 0], [20, 20], [0, 20]], "kind": "TABLE",
        }
        target = {
            "page_id": "contains-page",
            "lines": [line],
            "regions": [table],
            "cells": [],
            "relations": [
                {"type": "PARENT", "source": "line-1", "target": "table-1"},
                {"type": "CONTAINS", "source": "table-1", "target": "line-1"},
            ],
        }
        outputs = self._outputs(vocabulary, [line], [table])

        losses = PrismOCRLoss(vocabulary)(outputs, [target])
        losses["graph"].backward()

        gradients = outputs["graph_logits"].grad
        self.assertNotEqual(float(gradients[0, RELATION_NAMES.index("PARENT"), 0, 1]), 0.0)
        self.assertEqual(float(gradients[0, RELATION_NAMES.index("CONTAINS")].abs().sum()), 0.0)
        self.assertEqual(losses["unmatched_relation_count"], 0)

        parent_only_target = {**target, "relations": [
            {"type": "PARENT", "source": "line-1", "target": "table-1"},
        ]}
        parent_only_outputs = self._outputs(vocabulary, [line], [table])
        parent_only_loss = PrismOCRLoss(vocabulary)(parent_only_outputs, [parent_only_target])
        parent_only_loss["graph"].backward()
        torch.testing.assert_close(losses["graph"], parent_only_loss["graph"])
        torch.testing.assert_close(
            outputs["graph_logits"].grad,
            parent_only_outputs["graph_logits"].grad,
        )

    def test_incompatible_parent_geometry_is_not_supervised(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        line = {
            "id": "line-1", "box": _box(0.2, 0.2, 0.2, 0.1),
            "polygon_px": [[2, 2], [10, 2], [10, 5], [2, 5]],
            "text": "a", "role": "PARA_LINE",
        }
        parent = {
            "id": "parent", "box": _box(0.8, 0.8, 0.2, 0.1),
            "polygon_px": [[15, 15], [19, 15], [19, 19], [15, 19]],
            "kind": "REGION",
        }
        target = {
            "page_id": "disjoint-parent",
            "lines": [line], "regions": [parent], "cells": [],
            "relations": [{"type": "PARENT", "source": "line-1", "target": "parent"}],
        }
        outputs = self._outputs(vocabulary, [line], [parent])

        losses = PrismOCRLoss(vocabulary)(outputs, [target])

        self.assertEqual(losses["unmatched_relation_count"], 1)
        self.assertEqual(float(losses["graph"].detach()), 0.0)

    def test_right_cell_supervision_requires_decoder_compatible_grid_metadata(self) -> None:
        vocabulary = Vocabulary.from_texts(["cell"], max_size=8)
        left = {
            "id": "left", "box": _box(0.3, 0.5, 0.2, 0.2), "kind": "CELL",
            "polygon_px": [[2, 2], [7, 2], [7, 8], [2, 8]],
            "table_id": None, "row": 0, "col": 0,
        }
        right = {
            "id": "right", "box": _box(0.7, 0.5, 0.2, 0.2), "kind": "CELL",
            "polygon_px": [[10, 2], [15, 2], [15, 8], [10, 8]],
            "table_id": None, "row": 0, "col": 1,
        }
        target = {
            "page_id": "unassigned-right-cell",
            "lines": [], "regions": [], "cells": [left, right],
            "relations": [{"type": "RIGHT_CELL", "source": "left", "target": "right"}],
        }
        outputs = self._outputs(vocabulary, [], [left, right])

        losses = PrismOCRLoss(vocabulary)(outputs, [target])

        self.assertEqual(losses["unmatched_relation_count"], 1)
        self.assertEqual(float(losses["graph"].detach()), 0.0)

    def test_overlength_ctc_sample_is_rejected_with_stable_identity_without_aborting_batch(self) -> None:
        vocabulary = Vocabulary.from_texts(["aa"], max_size=8)
        line = {"id": "line-7", "box": _box(0.5, 0.5, 0.4, 0.2), "text": "aa", "role": "TITLE"}
        target = {
            "page_id": "document:page-7",
            "lines": [line],
            "regions": [],
            "cells": [],
            "relations": [],
        }
        outputs = self._outputs(vocabulary, [line], [], ctc_bucket=1, ctc_frames=1)

        losses = PrismOCRLoss(vocabulary, strict_ctc_length=True)(outputs, [target])

        self.assertEqual(losses["ctc_rejected_sample_count"], 1)
        self.assertEqual(losses["rejected_ctc_samples"], ({
            "page_id": "document:page-7",
            "line_id": "line-7",
            "proposal_index": 0,
            "reason": "requires_3_frames_but_bucket_is_1",
        },))
        self.assertTrue(torch.isfinite(losses["total"]))
        self.assertEqual(float(losses["ctc"].detach()), 0.0)

    def test_cell_grid_loss_supervises_positions_spans_and_header(self) -> None:
        vocabulary = Vocabulary.from_texts(["cell"], max_size=8)
        cell = {
            "id": "cell-1",
            "box": _box(0.5, 0.5, 0.4, 0.3),
            "kind": "CELL",
            "row": 2,
            "col": 3,
            "rowspan": 2,
            "colspan": 4,
            "is_header": True,
        }
        target = {
            "page_id": "document:table-page",
            "lines": [],
            "regions": [],
            "cells": [cell],
            "relations": [],
        }
        outputs = self._outputs(vocabulary, [], [cell])

        losses = PrismOCRLoss(vocabulary)(outputs, [target])
        self.assertGreater(float(losses["grid"].detach()), 0.0)
        losses["total"].backward()
        for head in ("row_logits", "col_logits", "rowspan_logits", "colspan_logits", "header_logits"):
            gradient = outputs["regions"][head].grad
            self.assertIsNotNone(gradient, head)
            self.assertGreater(float(gradient.abs().sum()), 0.0, head)

    def test_large_fp16_empty_heatmap_loss_is_finite(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        target = {"page_id": "blank-page", "lines": [], "regions": [], "cells": [], "relations": []}
        outputs = self._outputs(vocabulary, [], [])
        outputs["text_heatmap_logits"] = torch.full(
            (1, 1, 256, 256), -2.0, dtype=torch.float16, requires_grad=True
        )

        losses = PrismOCRLoss(vocabulary)(outputs, [target])
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertTrue(torch.isfinite(outputs["text_heatmap_logits"].grad).all())

    def test_ambiguous_table_cells_are_excluded_from_grid_supervision(self) -> None:
        vocabulary = Vocabulary.from_texts(["cell"], max_size=8)
        table = {
            "id": "table-1",
            "box": _box(0.5, 0.5, 0.8, 0.7),
            "kind": "TABLE",
            "annotation_ambiguous": True,
        }
        cell = {
            "id": "cell-1",
            "table_id": "table-1",
            "box": _box(0.5, 0.5, 0.4, 0.3),
            "kind": "CELL",
            "row": 2,
            "col": 3,
            "rowspan": 2,
            "colspan": 2,
            "is_header": True,
        }
        target = {
            "page_id": "page-ambiguous-grid",
            "lines": [],
            "regions": [table],
            "cells": [cell],
            "relations": [],
        }
        outputs = self._outputs(vocabulary, [], [table, cell])

        losses = PrismOCRLoss(vocabulary)(outputs, [target])
        losses["total"].backward()

        self.assertEqual(losses["grid_supervision_excluded_cell_count"], 1)
        self.assertEqual(losses["grid_supervision_exclusions"][0]["cell_id"], "cell-1")
        self.assertEqual(float(losses["grid"].detach()), 0.0)
        for head in ("row_logits", "col_logits", "rowspan_logits", "colspan_logits", "header_logits"):
            gradient = outputs["regions"][head].grad
            self.assertTrue(gradient is None or float(gradient.abs().sum()) == 0.0, head)

    def test_ambiguous_table_out_of_range_grid_labels_are_excluded_not_validated(self) -> None:
        vocabulary = Vocabulary.from_texts(["cell"], max_size=8)
        table = {
            "id": "table-1",
            "box": _box(0.5, 0.5, 0.8, 0.7),
            "kind": "TABLE",
            "annotation_ambiguous": True,
        }
        cell = {
            "id": "cell-1",
            "table_id": "table-1",
            "box": _box(0.5, 0.5, 0.4, 0.3),
            "kind": "CELL",
            "row": 128,
            "col": 0,
            "rowspan": 1,
            "colspan": 1,
        }
        target = {
            "page_id": "page-ambiguous-overflow",
            "lines": [],
            "regions": [table],
            "cells": [cell],
            "relations": [],
        }
        outputs = self._outputs(vocabulary, [], [table, cell])

        losses = PrismOCRLoss(vocabulary)(outputs, [target])

        self.assertEqual(losses["grid_supervision_excluded_cell_count"], 1)
        self.assertEqual(losses["grid_supervision_exclusions"][0]["reason"], "annotation_ambiguous")
        self.assertEqual(float(losses["grid"].detach()), 0.0)

    def test_partial_cell_labels_and_default_spans_receive_supervision(self) -> None:
        vocabulary = Vocabulary.from_texts(["cell"], max_size=8)
        cell = {
            "id": "partial-cell",
            "box": _box(0.5, 0.5, 0.4, 0.3),
            "kind": "CELL",
            "row": 2,
        }
        target = {"page_id": "page-partial", "lines": [], "regions": [], "cells": [cell], "relations": []}
        outputs = self._outputs(vocabulary, [], [cell])

        losses = PrismOCRLoss(vocabulary)(outputs, [target])
        losses["total"].backward()

        self.assertGreater(float(outputs["regions"]["row_logits"].grad.abs().sum()), 0.0)
        self.assertGreater(float(outputs["regions"]["rowspan_logits"].grad.abs().sum()), 0.0)
        self.assertGreater(float(outputs["regions"]["colspan_logits"].grad.abs().sum()), 0.0)

    def test_unmatched_cell_grid_labels_are_still_validated(self) -> None:
        vocabulary = Vocabulary.from_texts(["cell"], max_size=8)
        cell = {
            "id": "unmatched-cell",
            "box": _box(0.5, 0.5, 0.4, 0.3),
            "kind": "CELL",
            "row": 128,
            "col": 0,
        }
        target = {"page_id": "page-unmatched", "lines": [], "regions": [], "cells": [cell], "relations": []}
        outputs = self._outputs(vocabulary, [], [])

        with self.assertRaisesRegex(ValueError, "unmatched-cell.*outside configured grid capacity 128"):
            PrismOCRLoss(vocabulary)(outputs, [target])

    def test_grid_labels_outside_configured_class_range_fail_explicitly(self) -> None:
        vocabulary = Vocabulary.from_texts(["cell"], max_size=8)
        cell = {
            "id": "cell-overflow",
            "box": _box(0.5, 0.5, 0.4, 0.3),
            "kind": "CELL",
            "row": 128,
            "col": 0,
            "rowspan": 1,
            "colspan": 1,
        }
        target = {"page_id": "page-overflow", "lines": [], "regions": [], "cells": [cell], "relations": []}
        outputs = self._outputs(vocabulary, [], [cell])
        with self.assertRaisesRegex(ValueError, "outside configured grid capacity 128"):
            PrismOCRLoss(vocabulary)(outputs, [target])

    def test_mixed_precision_region_loss_and_backward_are_supported(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required to verify the trainer's AMP path")
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        model = PrismOCRModel(
            dim=10,
            text_proposals=4,
            region_queries=2,
            vocab_size=len(vocabulary),
            vocabulary=vocabulary,
        ).cuda().train()
        criterion = PrismOCRLoss(vocabulary)
        target = {"page_id": "amp-empty", "lines": [], "regions": [], "cells": [], "relations": []}

        for dtype in (torch.float16, torch.bfloat16):
            for value in (0.0, 0.5, 1.0):
                with self.subTest(dtype=dtype, image_value=value):
                    image = torch.full((1, 3, 32, 32), value, device="cuda")
                    model.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        outputs = model(image, [target])
                    losses = criterion(outputs, [target])
                    self.assertTrue(torch.isfinite(losses["total"]))
                    losses["total"].backward()
                    self.assertTrue(all(
                        parameter.grad is None or torch.isfinite(parameter.grad).all()
                        for parameter in model.parameters()
                    ))

    def test_bfloat16_matching_inputs_are_supported(self) -> None:
        boxes = torch.tensor([_box(0.5, 0.5, 0.4, 0.2)], dtype=torch.bfloat16)
        logits = torch.zeros((1, len(TEXT_ROLE_TO_ID)), dtype=torch.bfloat16)
        result = match_proposals(
            boxes,
            torch.tensor([_box(0.5, 0.5, 0.4, 0.2)]),
            torch.ones(1, dtype=torch.bool),
            image_size=(100, 100),
            pred_class_logits=logits,
            gt_classes=[TEXT_ROLE_TO_ID["TITLE"]],
        )
        self.assertEqual(result.accepted_pairs, ((0, 0),))

    def test_graph_loss_masks_padded_self_and_type_invalid_endpoints(self) -> None:
        vocabulary = Vocabulary.from_texts(["a", "b"], max_size=8)
        lines = [
            {"id": "line-1", "box": _box(0.25, 0.5, 0.2, 0.1), "text": "a", "role": "KEY"},
            {"id": "line-2", "box": _box(0.55, 0.5, 0.2, 0.1), "text": "b", "role": "VALUE"},
        ]
        table = {"id": "table-1", "box": _box(0.5, 0.5, 0.9, 0.8), "kind": "TABLE"}
        target = {
            "page_id": "document:page-1",
            "lines": lines,
            "regions": [table],
            "cells": [],
            "relations": [{"type": "NEXT", "source": "line-1", "target": "line-2"}],
        }
        outputs = self._outputs(vocabulary, lines, [table])
        logits = outputs["graph_logits"]
        with torch.no_grad():
            # Node 2 is a valid matched TABLE, node 3 is padded, and the
            # final endpoint is NULL. Self-links are invalid NEXT endpoints.
            logits[0, 0, 0, 0] = 12.0
            logits[0, 0, 0, 2] = 15.0
            logits[0, 0, 0, 3] = 20.0
        loss = PrismOCRLoss(vocabulary)(outputs, [target])["graph"]
        loss.backward()

        gradients = logits.grad[0, 0, 0]
        self.assertNotEqual(float(gradients[1]), 0.0)
        self.assertNotEqual(float(gradients[-1]), 0.0)
        self.assertEqual(float(gradients[0]), 0.0)
        self.assertEqual(float(gradients[2]), 0.0)
        self.assertEqual(float(gradients[3]), 0.0)

    def test_box_loss_is_zero_for_identical_boxes(self) -> None:
        box = torch.tensor([[0.3, 0.4, 0.2, 0.03, 0.1, 0.994987]], dtype=torch.float32)
        self.assertAlmostEqual(float(PrismOCRLoss._box_loss(box, box)), 0.0, places=6)

    def test_box_loss_keeps_gradient_for_tiny_predicted_sizes(self) -> None:
        # The proposal head decodes size = softplus(raw - 2). A linear-size
        # smooth L1 let the head collapse small line heights to near zero
        # because d(size)/d(raw) = softplus'(raw - 2) vanishes. The log-size
        # term must keep a usable gradient when raw is far negative.
        raw_height = torch.tensor([[-7.0]], requires_grad=True)
        height = torch.nn.functional.softplus(raw_height - 2.0).clamp_min(1e-4)
        predicted = torch.cat(
            [torch.tensor([[0.5, 0.5, 0.2]], dtype=torch.float32), height, torch.tensor([[0.0, 1.0]])],
            dim=-1,
        )
        truth = torch.tensor([[0.5, 0.5, 0.2, 0.03, 0.0, 1.0]], dtype=torch.float32)
        loss = PrismOCRLoss._box_loss(predicted, truth)
        gradient = torch.autograd.grad(loss, raw_height)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(abs(float(gradient)), 0.1)

    def test_box_loss_penalizes_size_ratios_symmetrically(self) -> None:
        half = torch.tensor([[0.5, 0.5, 0.05, 0.03, 0.0, 1.0]], dtype=torch.float32)
        double = torch.tensor([[0.5, 0.5, 0.20, 0.03, 0.0, 1.0]], dtype=torch.float32)
        target = torch.tensor([[0.5, 0.5, 0.10, 0.03, 0.0, 1.0]], dtype=torch.float32)
        self.assertAlmostEqual(
            float(PrismOCRLoss._box_loss(half, target)),
            float(PrismOCRLoss._box_loss(double, target)),
            places=6,
        )

    def test_anchor_cell_supervision_produces_gradients_when_proposal_scores_are_zero(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        model = PrismOCRModel(
            dim=10, text_proposals=4, region_queries=2, vocab_size=len(vocabulary), vocabulary=vocabulary
        ).train()
        line = {"id": "line-1", "box": _box(0.5, 0.5, 0.2, 0.1), "text": "a", "role": "TITLE"}
        target = {"page_id": "doc:1", "lines": [line], "regions": [], "cells": [], "relations": []}
        outputs = model(torch.randn((1, 3, 32, 32)), [target], teacher_force_ratio=0.0)

        # Force all proposal scores to 0
        outputs["text"]["scores"].zero_()
        losses = PrismOCRLoss(vocabulary)(outputs, [target])
        self.assertGreater(float(losses["box"].detach()), 0.0)
        self.assertGreater(float(losses["role"].detach()), 0.0)

        model.zero_grad()
        losses["box"].backward(retain_graph=True)
        self.assertIsNotNone(model.text_head.box.weight.grad)
        self.assertGreater(float(model.text_head.box.weight.grad.abs().sum()), 0.0)

        model.zero_grad()
        losses["role"].backward()
        self.assertIsNotNone(model.text_head.role.weight.grad)
        self.assertGreater(float(model.text_head.role.weight.grad.abs().sum()), 0.0)

    def test_ctc_loss_bootstrap_backpropagates_to_recognizer_when_accepted_proposals_empty(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        model = PrismOCRModel(
            dim=10, text_proposals=4, region_queries=2, vocab_size=len(vocabulary), vocabulary=vocabulary
        ).train()
        line = {"id": "line-1", "box": _box(0.5, 0.5, 0.2, 0.1), "text": "a", "role": "TITLE"}
        target = {"page_id": "doc:1", "lines": [line], "regions": [], "cells": [], "relations": []}
        outputs = model(torch.randn((1, 3, 32, 32)), [target], teacher_force_ratio=0.0)

        # In a random initialization without trained proposals, accepted_pairs is empty,
        # but the bootstrap path populates ocr_groups using anchor-cell embeddings.
        self.assertEqual(len(outputs["assignments"]["text"][0].accepted_pairs), 0)
        self.assertGreater(len(outputs["ocr_groups"]), 0)

        losses = PrismOCRLoss(vocabulary)(outputs, [target])
        self.assertGreater(float(losses["ctc"].detach()), 0.0)

        model.zero_grad()
        losses["ctc"].backward()
        self.assertIsNotNone(model.recognizer.classifier.weight.grad)
        self.assertGreater(float(model.recognizer.classifier.weight.grad.abs().sum()), 0.0)

    def test_role_and_graph_loss_normalized_by_decisions_not_batch_size(self) -> None:
        vocabulary = Vocabulary.from_texts(["a", "b"], max_size=8)
        line_a = {"id": "line-a", "box": _box(0.2, 0.5, 0.2, 0.1), "text": "a", "role": "KEY"}
        line_b = {"id": "line-b", "box": _box(0.6, 0.5, 0.2, 0.1), "text": "b", "role": "VALUE"}

        # Target 0 has 2 lines and 1 relation decision; Target 1 has 0 lines and 0 relations.
        target_0 = {
            "page_id": "doc:page-0",
            "lines": [line_a, line_b],
            "regions": [],
            "cells": [],
            "relations": [{"type": "NEXT", "source": "line-a", "target": "line-b"}],
        }
        target_1 = {
            "page_id": "doc:page-1",
            "lines": [],
            "regions": [],
            "cells": [],
            "relations": [],
        }

        # Build a 2-page batch
        match_0 = _matched([line_a["box"], line_b["box"]], [line_a, line_b])
        match_1 = _matched([], [])
        match_regions_0 = _matched([], [])
        match_regions_1 = _matched([], [])

        role_logits = torch.randn((2, 2, len(TEXT_ROLE_TO_ID)), requires_grad=True)
        # Graph logits: 2 pages, 5 relations, 4 nodes, 5 target slots
        graph_logits = torch.randn((2, len(RELATION_NAMES), 4, 5), requires_grad=True)

        outputs = {
            "text_heatmap_logits": torch.zeros((2, 1, 4, 4), requires_grad=True),
            "text": {
                "boxes": torch.zeros((2, 2, 6)),
                "role_logits": role_logits,
            },
            "regions": {
                "boxes": torch.zeros((2, 0, 6)),
                "class_logits": torch.zeros((2, 0, len(REGION_KIND_TO_ID))),
                "row_logits": torch.zeros((2, 0, 128)),
                "col_logits": torch.zeros((2, 0, 128)),
                "rowspan_logits": torch.zeros((2, 0, 32)),
                "colspan_logits": torch.zeros((2, 0, 32)),
                "header_logits": torch.zeros((2, 0)),
            },
            "assignments": {
                "text": (match_0, match_1),
                "regions": (match_regions_0, match_regions_1),
            },
            "graph_logits": graph_logits,
            "ocr_groups": {},
            "ocr_unsupported": (),
        }

        criterion = PrismOCRLoss(vocabulary)
        losses = criterion(outputs, [target_0, target_1])

        # Expected role loss: mean over 2 supervised lines (not divided by batch_size 2 or available pages 2)
        expected_role_term_0 = torch.nn.functional.cross_entropy(
            role_logits[0, 0:1], torch.tensor([TEXT_ROLE_TO_ID["KEY"]])
        )
        expected_role_term_1 = torch.nn.functional.cross_entropy(
            role_logits[0, 1:2], torch.tensor([TEXT_ROLE_TO_ID["VALUE"]])
        )
        expected_role_loss = (expected_role_term_0 + expected_role_term_1) / 2.0
        self.assertAlmostEqual(float(losses["role"].detach()), float(expected_role_loss.detach()), places=5)

        # Expected graph loss: exactly 1 supervised relation decision on page 0.
        # It must normalize by 1 (the number of relation decisions), NOT by batch_size 2!
        rel_logits = graph_logits[0, RELATION_NAMES.index("NEXT"), 0].clone()
        valid = torch.zeros(5, dtype=torch.bool)
        valid[1] = True   # line-b
        valid[4] = True   # NULL
        rel_logits = rel_logits.masked_fill(~valid, float("-inf"))
        expected_graph_loss = torch.nn.functional.cross_entropy(
            rel_logits[None], torch.tensor([1])
        )
        self.assertAlmostEqual(float(losses["graph"].detach()), float(expected_graph_loss.detach()), places=5)

    def test_oov_accounting_tracks_rejected_lines_and_character_counts(self) -> None:
        vocabulary = Vocabulary.from_texts(["hello", "world"], max_size=16)
        # Vocabulary only contains characters from "hello", "world".
        # Characters '!' and '@' and '#' are OOV.
        lines = [
            {"id": "line-1", "box": _box(0.2, 0.5, 0.2, 0.1), "text": "hello!@", "role": "TITLE"},
            {"id": "line-2", "box": _box(0.5, 0.5, 0.2, 0.1), "text": "world#", "role": "PARA_LINE"},
        ]
        target = {
            "page_id": "doc:oov",
            "lines": lines,
            "regions": [],
            "cells": [],
            "relations": [],
        }
        outputs = self._outputs(vocabulary, lines, [])
        outputs["ocr_unsupported"] = (
            (0, 0, "ctc_oov_characters"),
            (0, 1, "requires_10_frames_but_bucket_is_8"),
            (0, 2, "ctc_oov_characters"),
        )

        losses = PrismOCRLoss(vocabulary)(outputs, [target])
        # Two lines were rejected specifically with reason "ctc_oov_characters"
        self.assertEqual(losses["rejected_oov_line_count"], 2)
        # Across line-1 ("hello!@" -> 2 OOV) and line-2 ("world#" -> 1 OOV)
        self.assertEqual(losses["oov_character_count"], 3)


if __name__ == "__main__":
    unittest.main()
