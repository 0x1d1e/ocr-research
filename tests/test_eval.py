from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from prismocr.eval import evaluate
from prismocr.evaluation import evaluate_page


class EvaluatorTests(unittest.TestCase):
    @staticmethod
    def _polygon(x1: int, y1: int, x2: int, y2: int) -> list[list[int]]:
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    def test_empty_evaluation_manifest_is_rejected_before_loading_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "empty.jsonl"
            manifest.write_text("", encoding="utf-8")
            with patch("prismocr.eval.PrismOCR.load") as load_artifact:
                with self.assertRaisesRegex(ValueError, "evaluation manifest must contain at least one page"):
                    evaluate(manifest, root / "artifact")
            load_artifact.assert_not_called()

    def test_exact_reading_order_penalizes_unmatched_extra_lines(self) -> None:
        annotation = {
            "image": {"width": 100, "height": 100},
            "lines": [{"id": "gt", "polygon_px": self._polygon(5, 5, 25, 15), "text": "a"}],
            "regions": [], "cells": [], "relations": [], "reading_order": ["gt"],
        }
        prediction = {
            "lines": [
                {"id": "p-gt", "polygon": self._polygon(5, 5, 25, 15), "text": "a", "reading_order": 0},
                {"id": "extra", "polygon": self._polygon(70, 70, 90, 80), "text": "x", "reading_order": 1},
            ],
            "regions": [], "cells": [], "relations": [], "reading_order": ["p-gt", "extra"],
        }

        metrics = evaluate_page(annotation, prediction)

        self.assertEqual(metrics["reading_order"]["exact_page"], 0)
        self.assertEqual(metrics["detections"]["lines"]["false_positive"], 1)

    def test_detection_matching_uses_source_polygon_iou(self) -> None:
        annotation = {
            "image": {"width": 100, "height": 100},
            "lines": [{
                "id": "gt",
                "polygon_px": [[10, 10], [70, 10], [10, 70]],
                "text": "a",
            }],
            "regions": [], "cells": [], "relations": [], "reading_order": ["gt"],
        }
        prediction = {
            "lines": [{
                "id": "pred",
                "polygon": [[70, 70], [70, 10], [10, 70]],
                "text": "a",
            }],
            "regions": [], "cells": [], "relations": [], "reading_order": ["pred"],
        }

        metrics = evaluate_page(annotation, prediction)

        self.assertEqual(metrics["detections"]["lines"]["true_positive"], 0)
        self.assertEqual(metrics["detections"]["lines"]["false_positive"], 1)
        self.assertEqual(metrics["detections"]["lines"]["false_negative"], 1)

    def test_grouped_evaluator_uses_polygon_iou_for_transcript_matching(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "page.png"
            Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(image_path)
            ground_truth_polygon = [[10, 10], [70, 10], [10, 70]]
            prediction_polygon = [[70, 70], [70, 10], [10, 70]]
            annotation = {
                "schema_version": "4.0",
                "document_id": "polygon-doc",
                "page_id": "polygon-doc:p0",
                "image": {
                    "width": 100,
                    "height": 100,
                    "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                },
                "provenance": {
                    "source_class": "synthetic",
                    "license": "generated",
                    "permission_reference": "not_applicable",
                    "reviewer": "unit_test",
                },
                "lines": [{
                    "id": "gt-line",
                    "polygon_px": ground_truth_polygon,
                    "text": "a",
                    "role": "PARA_LINE",
                }],
                "regions": [],
                "cells": [],
                "relations": [],
                "reading_order": ["gt-line"],
            }
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps({
                "image_path": image_path.name,
                "annotation": annotation,
            }) + "\n", encoding="utf-8")

            class FakeAPI:
                model_version = "fixture"

                def predict(self, _image: Image.Image, _options: object) -> SimpleNamespace:
                    return SimpleNamespace(to_dict=lambda: {
                        "lines": [{
                            "id": "pred-line",
                            "polygon": prediction_polygon,
                            "text": "a",
                        }],
                        "regions": [],
                        "cells": [],
                        "relations": [],
                        "reading_order": ["pred-line"],
                        "warnings": [],
                        "diagnostics": {"candidate_count": 1, "overflow": False},
                    })

            with patch("prismocr.eval.PrismOCR.load", return_value=FakeAPI()):
                metrics = evaluate(manifest, root / "artifact")

        self.assertEqual(metrics["line_detection_recall_at_iou"], 0.0)
        self.assertEqual(metrics["detection"]["0.5"]["lines"]["false_positive"], 1)
        self.assertEqual(metrics["detection"]["0.5"]["lines"]["false_negative"], 1)
        self.assertEqual(metrics["character_errors"], 2)

    def test_reading_order_pair_denominator_includes_missed_lines(self) -> None:
        annotation = {
            "image": {"width": 100, "height": 100},
            "lines": [
                {"id": "first", "polygon_px": self._polygon(5, 5, 25, 15), "text": "a"},
                {"id": "second", "polygon_px": self._polygon(5, 25, 25, 35), "text": "b"},
            ],
            "regions": [], "cells": [], "relations": [],
            "reading_order": ["first", "second"],
        }
        prediction = {
            "lines": [{
                "id": "p-first", "polygon": self._polygon(5, 5, 25, 15),
                "text": "a", "reading_order": 0,
            }],
            "regions": [], "cells": [], "relations": [],
            "reading_order": ["p-first"],
        }

        metrics = evaluate_page(annotation, prediction)

        self.assertEqual(metrics["reading_order"]["pair_count"], 1)
        self.assertEqual(metrics["reading_order"]["correct_pairs"], 0)
        self.assertEqual(metrics["reading_order"]["exact_page"], 0)

    def test_relation_to_unmatched_detection_counts_as_false_positive(self) -> None:
        annotation = {
            "image": {"width": 100, "height": 100},
            "lines": [
                {"id": "first", "polygon_px": self._polygon(5, 5, 25, 15), "text": "a"},
                {"id": "second", "polygon_px": self._polygon(5, 25, 25, 35), "text": "b"},
            ],
            "regions": [], "cells": [],
            "relations": [
                {"type": "NEXT", "source": "first", "target": None},
                {"type": "NEXT", "source": "second", "target": None},
            ],
            "reading_order": ["first", "second"],
        }
        prediction = {
            "lines": [
                {"id": "p-first", "polygon": self._polygon(5, 5, 25, 15), "text": "a"},
                {"id": "p-second", "polygon": self._polygon(5, 25, 25, 35), "text": "b"},
                {"id": "extra", "polygon": self._polygon(70, 70, 90, 80), "text": "x"},
            ],
            "regions": [], "cells": [],
            "relations": [
                {"type": "NEXT", "source": "p-first", "target": "extra"},
                {"type": "NEXT", "source": "p-second", "target": None},
            ],
            "reading_order": ["p-first", "p-second", "extra"],
        }

        metrics = evaluate_page(annotation, prediction)

        self.assertEqual(metrics["relations"]["NEXT"], {
            "true_positive": 1, "false_positive": 1, "false_negative": 1,
        })

    def test_contains_to_unmatched_parent_counts_as_parent_false_positive(self) -> None:
        annotation = {
            "image": {"width": 100, "height": 100},
            "lines": [{
                "id": "child", "polygon_px": self._polygon(5, 5, 25, 15), "text": "a",
            }],
            "regions": [{
                "id": "parent", "kind": "FORM_FIELD",
                "polygon_px": self._polygon(5, 25, 40, 45),
            }],
            "cells": [],
            "relations": [{"type": "PARENT", "source": "child", "target": "parent"}],
            "reading_order": ["child"],
        }
        prediction = {
            "lines": [{
                "id": "p-child", "polygon": self._polygon(5, 5, 25, 15), "text": "a",
            }],
            "regions": [{
                "id": "extra-parent", "kind": "FORM_FIELD",
                "polygon": self._polygon(70, 70, 90, 90),
            }],
            "cells": [],
            "relations": [{
                "type": "CONTAINS", "source": "extra-parent", "target": "p-child",
            }],
            "reading_order": ["p-child"],
        }

        metrics = evaluate_page(annotation, prediction)

        self.assertEqual(metrics["relations"]["PARENT"], {
            "true_positive": 0, "false_positive": 1, "false_negative": 1,
        })

    def test_language_slices_include_unmatched_predicted_lines(self) -> None:
        annotation = {
            "image": {"width": 100, "height": 100},
            "lines": [{
                "id": "gt", "polygon_px": self._polygon(5, 5, 25, 15),
                "text": "known", "language": "english",
            }],
            "regions": [], "cells": [], "relations": [], "reading_order": ["gt"],
        }
        prediction = {
            "lines": [
                {"id": "p-gt", "polygon": self._polygon(5, 5, 25, 15), "text": "known"},
                {"id": "extra", "polygon": self._polygon(70, 70, 90, 80), "text": "hallucination", "language": "english"},
            ],
            "regions": [], "cells": [], "relations": [], "reading_order": ["p-gt", "extra"],
        }

        metrics = evaluate_page(annotation, prediction)

        english = metrics["language_slices"]["english"]
        self.assertEqual(english["character_errors"], len("hallucination"))
        self.assertEqual(english["word_errors"], 1)
        self.assertEqual(english["lines"], 2)
        self.assertEqual(english["reference_characters"], len("known"))
        self.assertEqual(english["character_errors"], len("hallucination"))
        self.assertEqual(english["reference_words"], 1)
        self.assertEqual(english["word_errors"], 1)

    def test_khmer_mark_errors_preserve_word_and_base_position_context(self) -> None:
        annotation = {
            "image": {"width": 100, "height": 100},
            "lines": [{
                "id": "gt", "polygon_px": self._polygon(5, 5, 25, 15),
                "text": "កក្", "language": "khmer",
            }],
            "regions": [], "cells": [], "relations": [], "reading_order": ["gt"],
        }
        prediction = {
            "lines": [{
                "id": "pred", "polygon": self._polygon(5, 5, 25, 15),
                "text": "ក្ក", "language": "khmer",
            }],
            "regions": [], "cells": [], "relations": [], "reading_order": ["pred"],
        }

        metrics = evaluate_page(annotation, prediction)

        self.assertEqual(metrics["khmer_marks"]["reference_coeng"], 1)
        self.assertEqual(metrics["khmer_marks"]["coeng_errors"], 1)
        self.assertEqual(metrics["khmer_marks"]["character_errors"], 1)

    def test_unmatched_khmer_mark_lines_count_as_mark_insertions(self) -> None:
        text = "\u1780\u17d2\u1780"
        annotation = {
            "image": {"width": 100, "height": 100},
            "lines": [{
                "id": "gt", "polygon_px": self._polygon(5, 5, 25, 15),
                "text": text, "language": "khmer",
            }],
            "regions": [], "cells": [], "relations": [], "reading_order": ["gt"],
        }
        prediction = {
            "lines": [
                {"id": "matched", "polygon": self._polygon(5, 5, 25, 15), "text": text},
                {"id": "extra", "polygon": self._polygon(60, 60, 80, 70), "text": text},
            ],
            "regions": [], "cells": [], "relations": [],
            "reading_order": ["matched", "extra"],
        }

        marks = evaluate_page(annotation, prediction)["khmer_marks"]

        self.assertEqual(marks["reference_characters"], 1)
        self.assertEqual(marks["character_errors"], 1)
        self.assertEqual(marks["cer"], 1.0)
        self.assertEqual(marks["reference_coeng"], 1)
        self.assertEqual(marks["coeng_errors"], 1)
        self.assertEqual(marks["coeng_error_rate"], 1.0)

    def test_malformed_predicted_region_kinds_count_as_false_positives(self) -> None:
        annotation = {
            "image": {"width": 100, "height": 100},
            "lines": [], "regions": [], "cells": [], "relations": [],
            "reading_order": [],
        }
        polygon = self._polygon(5, 5, 25, 15)
        prediction = {
            "lines": [],
            "regions": [
                {"id": "missing-kind", "polygon": polygon},
                {"id": "null-kind", "kind": None, "polygon": polygon},
                {"id": "numeric-kind", "kind": 7, "polygon": polygon},
                {"id": "unknown-kind", "kind": "UNKNOWN", "polygon": polygon},
            ],
            "cells": [], "relations": [], "reading_order": [],
        }

        metrics = evaluate_page(annotation, prediction)

        self.assertEqual(metrics["detections"]["regions"]["INVALID"], {
            "true_positive": 0, "false_positive": 4, "false_negative": 0,
        })

    def test_exact_table_accepts_implicit_unit_spans(self) -> None:
        table_polygon = self._polygon(5, 5, 95, 95)
        cell_polygon = self._polygon(10, 10, 45, 45)
        annotation = {
            "image": {"width": 100, "height": 100},
            "lines": [],
            "regions": [{"id": "gt-table", "kind": "TABLE", "polygon_px": table_polygon}],
            "cells": [{
                "id": "gt-cell", "table_id": "gt-table", "row": 0, "col": 0,
                "polygon_px": cell_polygon, "text": "value",
            }],
            "relations": [], "reading_order": [],
        }
        prediction = {
            "lines": [],
            "regions": [{"id": "p-table", "kind": "TABLE", "polygon": table_polygon}],
            "cells": [{
                "id": "p-cell", "table_id": "p-table", "row": 0, "col": 0,
                "polygon": cell_polygon, "text": "value",
            }],
            "tables": [{"table_id": "p-table", "status": "complete", "rows": 1, "cols": 1}],
            "relations": [], "reading_order": [],
        }

        metrics = evaluate_page(annotation, prediction)

        self.assertEqual(metrics["table_structure"]["exact_tables"], 1)
        self.assertEqual(metrics["table_structure"]["teds_sum"], 1.0)
        self.assertEqual(metrics["table_structure"]["teds_structure_sum"], 1.0)

    def test_invalid_or_extra_table_cells_cannot_score_exact_or_perfect_teds(self) -> None:
        table_polygon = self._polygon(5, 5, 95, 95)
        cell_polygon = self._polygon(10, 10, 45, 45)
        annotation = {
            "image": {"width": 100, "height": 100},
            "lines": [],
            "regions": [{"id": "gt-table", "kind": "TABLE", "polygon_px": table_polygon}],
            "cells": [{
                "id": "gt-cell", "table_id": "gt-table", "row": 0, "col": 0,
                "polygon_px": cell_polygon, "text": "value",
            }],
            "relations": [], "reading_order": [],
        }
        prediction = {
            "lines": [],
            "regions": [{"id": "p-table", "kind": "TABLE", "polygon": table_polygon}],
            "cells": [
                {"id": "p-cell", "table_id": "p-table", "row": 0, "col": 0,
                 "polygon": cell_polygon, "text": "value"},
                {"id": "extra-cell", "table_id": "p-table", "row": 0, "col": 0,
                 "polygon": cell_polygon, "text": "extra"},
            ],
            "tables": [{"table_id": "p-table", "status": "conflict", "rows": 1, "cols": 1}],
            "relations": [], "reading_order": [],
        }

        metrics = evaluate_page(annotation, prediction)

        self.assertEqual(metrics["table_structure"]["exact_tables"], 0)
        self.assertEqual(metrics["table_structure"]["eligible_tables"], 1)
        self.assertEqual(metrics["table_structure"]["teds_sum"], 0.0)
        self.assertEqual(metrics["table_structure"]["teds_structure_sum"], 0.0)
        self.assertEqual(metrics["detections"]["cells"]["false_positive"], 1)

    def test_blank_manifest_reports_zero_denominator_metrics_as_undefined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "blank.png"
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image_path)
            annotation = {
                "schema_version": "4.0",
                "document_id": "doc-blank",
                "page_id": "doc-blank:p0",
                "image": {
                    "width": 32,
                    "height": 32,
                    "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                },
                "provenance": {
                    "source_class": "synthetic",
                    "license": "generated",
                    "permission_reference": "not_applicable",
                    "reviewer": "unit_test",
                },
                "lines": [],
                "regions": [],
                "cells": [],
                "relations": [],
                "reading_order": [],
            }
            manifest = root / "blank.jsonl"
            manifest.write_text(
                json.dumps({"image_path": image_path.name, "annotation": annotation}) + "\n",
                encoding="utf-8",
            )

            class FakeAPI:
                model_version = "fixture"

                def predict(self, _image: Image.Image, _options: object) -> SimpleNamespace:
                    return SimpleNamespace(to_dict=lambda: {
                        "lines": [],
                        "warnings": [],
                        "diagnostics": {"candidate_count": 0, "overflow": False},
                    })

            with patch("prismocr.eval.PrismOCR.load", return_value=FakeAPI()):
                metrics = evaluate(manifest, root / "artifact")

        self.assertFalse(metrics["metrics_are_defined"])
        self.assertEqual(
            set(metrics["zero_denominator_metrics"]),
            {
                "line_detection_recall_at_iou",
                "cer_on_matched_detections_with_misses_as_empty",
                "wer_with_misses_as_empty",
                "reading_order",
                "table_structure",
            },
        )
        # The raw rates are still reported; the flags mark them as vacuous.
        self.assertEqual(metrics["line_detection_recall_at_iou"], 1.0)
        self.assertEqual(metrics["cer_on_matched_detections_with_misses_as_empty"], 0.0)

    def test_inference_failure_counts_as_missed_ground_truth_and_reports_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for index, (page_id, document_id, text) in enumerate(
                (("doc-1:p0", "doc-1", "same"), ("doc-2:p0", "doc-2", "lost"))
            ):
                image_path = root / f"page-{index}.png"
                Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image_path)
                polygon = [[4, 8], [24, 8], [24, 14], [4, 14]]
                annotation = {
                    "schema_version": "4.0",
                    "document_id": document_id,
                    "page_id": page_id,
                    "image": {
                        "width": 32,
                        "height": 32,
                        "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                    },
                    "provenance": {
                        "source_class": "synthetic",
                        "license": "generated",
                        "permission_reference": "not_applicable",
                        "reviewer": "unit_test",
                    },
                    "lines": [{
                        "id": f"line-{index}",
                        "polygon_px": polygon,
                        "text": text,
                        "role": "PARA_LINE",
                    }],
                    "regions": [],
                    "cells": [],
                    "relations": [],
                    "reading_order": [f"line-{index}"],
                }
                rows.append({"image_path": image_path.name, "annotation": annotation})
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            class FakeAPI:
                model_version = "fixture"

                def __init__(self) -> None:
                    self.calls = 0

                def predict(self, image: Image.Image, _options: object) -> SimpleNamespace:
                    self.calls += 1
                    if self.calls == 2:
                        raise RuntimeError("injected inference failure")
                    return SimpleNamespace(to_dict=lambda: {
                        "lines": [{
                            "polygon": [[4, 8], [24, 8], [24, 14], [4, 14]],
                            "text": "same",
                        }],
                        "warnings": [
                            {"code": "TABLE_GRID_CONFLICT"},
                            {"code": "UNSUPPORTED_LINE_LENGTH"},
                            {"code": "TABLE_GRID_UNRESOLVED"},
                        ],
                        "diagnostics": {"candidate_count": 3, "overflow": True},
                    })

            api = FakeAPI()
            with patch("prismocr.eval.PrismOCR.load", return_value=api):
                metrics = evaluate(manifest, root / "artifact")

        self.assertEqual(metrics["page_count"], 2)
        self.assertEqual(metrics["line_detection_recall_at_iou"], 0.5)
        self.assertEqual(metrics["matched_lines"], 1)
        self.assertEqual(metrics["ground_truth_lines"], 2)
        self.assertEqual(metrics["cer_on_matched_detections_with_misses_as_empty"], 0.5)
        self.assertEqual(metrics["wer_with_misses_as_empty"], 0.5)
        self.assertEqual(metrics["inference_failure_count"], 1)
        self.assertEqual(metrics["inference_failures"][0]["error_type"], "RuntimeError")
        self.assertEqual(metrics["table_grid_conflict_warnings"], 1)
        self.assertEqual(metrics["unsupported_line_warnings"], 1)
        self.assertEqual(metrics["pages_with_unresolved_table_grid"], 1)
        self.assertEqual(metrics["pages_with_capacity_overflow"], 1)
        self.assertEqual(metrics["proposal_candidates"], 3)


if __name__ == "__main__":
    unittest.main()
