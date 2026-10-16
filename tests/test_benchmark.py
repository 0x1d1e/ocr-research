from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from prismocr.api import PrismOCR
from prismocr.benchmark import (
    _StageInstrumentation,
    _benchmark_workload,
    _manifest_workloads,
    benchmark,
    line_count_bucket,
)
from prismocr.data.vocabulary import Vocabulary
from prismocr.model.network import REGION_KIND_TO_ID, TEXT_ROLE_TO_ID, PrismOCRModel


class BenchmarkProtocolTests(unittest.TestCase):
    def test_line_count_buckets_cover_protocol_boundaries(self) -> None:
        cases = {
            0: "blank",
            1: "lines_1_10",
            10: "lines_1_10",
            11: "lines_11_30",
            30: "lines_11_30",
            31: "lines_31_64",
            64: "lines_31_64",
            65: "lines_65_96",
            96: "lines_65_96",
            97: "lines_over_96",
        }
        for count, expected in cases.items():
            with self.subTest(count=count):
                self.assertEqual(line_count_bucket(count), expected)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            line_count_bucket(-1)

    def test_manifest_workloads_classify_line_tables_and_high_resolution(self) -> None:
        counts_and_sizes = (
            (0, 128, 128, False),
            (10, 256, 256, False),
            (11, 512, 512, True),
            (31, 768, 768, False),
            (65, 1024, 1024, True),
            (97, 1280, 768, False),
        )
        records = []
        for index, (count, width, height, has_table) in enumerate(counts_and_sizes):
            annotation = {
                "page_id": f"page-{index}",
                "image": {"width": width, "height": height},
                "lines": [{} for _ in range(count)],
                "regions": ([{"kind": "TABLE"}] if has_table else []),
            }
            records.append({"_annotation": annotation})
        workloads = _manifest_workloads(types.SimpleNamespace(records=records))
        grouped = {(item["category"], item["bucket"]): item["records"] for item in workloads}
        self.assertEqual(len(grouped[("line_count", "blank")]), 1)
        self.assertEqual(len(grouped[("line_count", "lines_1_10")]), 1)
        self.assertEqual(len(grouped[("line_count", "lines_11_30")]), 1)
        self.assertEqual(len(grouped[("line_count", "lines_31_64")]), 1)
        self.assertEqual(len(grouped[("line_count", "lines_65_96")]), 1)
        self.assertEqual(len(grouped[("line_count", "lines_over_96")]), 1)
        self.assertEqual(len(grouped[("table_presence", "with_tables")]), 2)
        self.assertEqual(len(grouped[("table_presence", "without_tables")]), 4)
        self.assertEqual(len(grouped[("resolution", "square_768")]), 1)
        self.assertEqual(len(grouped[("resolution", "square_768_plus")]), 3)
        self.assertEqual(len(grouped[("resolution", "tiled_high_resolution")]), 1)

    def test_manifest_benchmark_enforces_minimum_protocol_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 30 warmups and 100 timed iterations"):
            benchmark("unused", manifest="unused.jsonl", warmup=29, iterations=100)

    def test_benchmark_reports_only_distinct_pages_measured(self) -> None:
        records = [
            {
                "_image_path": Path(f"page-{index}.png"),
                "_image_sha256": str(index),
                "_annotation": {"page_id": f"page-{index}", "lines": [], "regions": []},
            }
            for index in range(101)
        ]
        measured_paths: list[str] = []

        def run_page(_engine: object, image_path: Path, **_kwargs: object) -> dict[str, object]:
            measured_paths.append(image_path.name)
            return {
                "stage_ns": {"end_to_end": 1},
                "recognizer_bucket_counts": {},
                "selected_ocr_lines": 0,
                "query_count": 0,
                "warning_counts": {},
                "error": None,
                "result": {"lines": [], "regions": [], "cells": [], "tables": [],
                           "diagnostics": {}},
            }

        with patch("prismocr.benchmark._run_page", side_effect=run_page):
            result = _benchmark_workload(
                object(), list(enumerate(records)), warmup=0, iterations=100,
                page_resolution=None, instrumentation=object(),
                category="diagnostic", bucket="sampled",
            )

        self.assertEqual(len(measured_paths), 100)
        self.assertEqual(len(set(measured_paths)), 100)
        self.assertEqual(result["source_page_count"], 101)
        self.assertEqual(result["page_count"], 100)
        self.assertIn("page-100", result["sample_page_ids"])
        self.assertEqual(len(result["sample_page_ids"]), 100)

    def test_sparse_stage_latency_reports_conditional_execution_counts(self) -> None:
        record = {
            "_image_path": Path("page.png"),
            "_image_sha256": "hash",
            "_annotation": {"page_id": "page", "lines": [], "regions": []},
        }
        runs = iter((
            {"end_to_end": 10_000_000, "recognizer": 4_000_000},
            {"end_to_end": 10_000_000},
            {"end_to_end": 10_000_000, "recognizer": 4_000_000},
        ))

        def run_page(_engine: object, _image_path: Path, **_kwargs: object) -> dict[str, object]:
            return {
                "stage_ns": next(runs),
                "recognizer_bucket_counts": {},
                "selected_ocr_lines": 0,
                "query_count": 0,
                "warning_counts": {},
                "error": None,
                "result": {"lines": [], "regions": [], "cells": [], "tables": [],
                           "diagnostics": {}},
            }

        with patch("prismocr.benchmark._run_page", side_effect=run_page):
            result = _benchmark_workload(
                object(), [(0, record)], warmup=0, iterations=3,
                page_resolution=None, instrumentation=object(),
                category="diagnostic", bucket="sparse",
            )

        self.assertEqual(result["measured_runs"], 3)
        self.assertEqual(result["stage_execution_counts"]["recognizer"], 2)
        self.assertEqual(result["stage_execution_counts"]["end_to_end"], 3)
        self.assertEqual(result["latency_ms"]["recognizer"]["mean_ms"], 4.0)
        self.assertIn("conditional", result["stage_latency_scope"])

    def test_resolution_workload_requires_effective_tiling(self) -> None:
        record = {
            "_image_path": Path("large-page.png"),
            "_image_sha256": "hash",
            "_annotation": {"page_id": "large-page", "lines": [], "regions": []},
        }

        def run_page(_engine: object, _image_path: Path, **_kwargs: object) -> dict[str, object]:
            return {
                "stage_ns": {"end_to_end": 1},
                "recognizer_bucket_counts": {},
                "selected_ocr_lines": 0,
                "query_count": 0,
                "warning_counts": {},
                "error": None,
                "result": {"lines": [], "regions": [], "cells": [], "tables": [],
                           "diagnostics": {}},
                "effective_size_px": [32, 19],
                "executed_tile_count": 1,
            }

        with patch("prismocr.benchmark._run_page", side_effect=run_page):
            result = _benchmark_workload(
                object(), [(0, record)], warmup=0, iterations=1,
                page_resolution=32, instrumentation=object(),
                category="resolution", bucket="tiled_high_resolution",
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["execution_issues"][0]["code"], "TILED_HIGH_RESOLUTION_NOT_EXECUTED")

    def test_nonblank_line_workload_requires_recognizer_execution(self) -> None:
        record = {
            "_image_path": Path("text-page.png"),
            "_image_sha256": "hash",
            "_annotation": {
                "page_id": "text-page", "lines": [{}] * 11, "regions": [],
            },
        }

        def run_page(_engine: object, _image_path: Path, **_kwargs: object) -> dict[str, object]:
            return {
                "stage_ns": {"end_to_end": 1},
                "recognizer_bucket_counts": {},
                "selected_ocr_lines": 0,
                "query_count": 0,
                "warning_counts": {},
                "error": None,
                "result": {"lines": [], "regions": [], "cells": [], "tables": [],
                           "diagnostics": {}},
            }

        with patch("prismocr.benchmark._run_page", side_effect=run_page):
            result = _benchmark_workload(
                object(), [(0, record)], warmup=0, iterations=1,
                page_resolution=None, instrumentation=object(),
                category="line_count", bucket="lines_11_30",
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["execution_issues"][0]["code"], "NO_RECOGNIZER_EXECUTION")

    def test_recognizer_forward_bucket_is_timed(self) -> None:
        class BucketRecognizer(torch.nn.Module):
            def forward(self) -> None:
                return None

            def forward_bucket(self, marker: str) -> str:
                return marker

        class FakeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.recognizer = BucketRecognizer()
                self.backbone = torch.nn.Identity()
                self.text_head = torch.nn.Identity()
                self.region_head = torch.nn.Identity()
                self.relation_head = torch.nn.Identity()

            def forward(self) -> dict[str, object]:
                return {}

        model = FakeModel()
        engine = types.SimpleNamespace(
            model=model, _prepare_image=lambda *_a: None, _tile_output=lambda *_a: None
        )
        instrumentation = _StageInstrumentation(engine)
        capture = {"stage_ns": {"recognizer": 0}}

        with instrumentation:
            instrumentation.capture = capture
            self.assertEqual(model.recognizer.forward_bucket("executed"), "executed")
            instrumentation.capture = None

        self.assertGreater(capture["stage_ns"]["recognizer"], 0)

    def test_single_image_benchmark_reports_public_api_stages(self) -> None:
        vocabulary = Vocabulary.from_texts(["a"], max_size=8)
        model = PrismOCRModel(
            dim=10,
            text_proposals=1,
            region_queries=1,
            vocab_size=len(vocabulary),
            vocabulary=vocabulary,
        )

        def empty_forward(_self: PrismOCRModel, images: torch.Tensor) -> dict[str, object]:
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
                    "mask": torch.zeros((1, 1), dtype=torch.bool, device=device),
                    "boxes": torch.zeros((1, 1, 6), device=device),
                    "scores": torch.zeros((1, 1), device=device),
                    "kinds": torch.full((1, 1), REGION_KIND_TO_ID["REGION"], device=device),
                    "embeddings": torch.zeros((1, 1, model.dim), device=device),
                    "row_logits": torch.zeros((1, 1, 128), device=device),
                    "col_logits": torch.zeros((1, 1, 128), device=device),
                    "rowspan_logits": torch.zeros((1, 1, 32), device=device),
                    "colspan_logits": torch.zeros((1, 1, 32), device=device),
                    "header_logits": torch.zeros((1, 1), device=device),
                },
                "ocr_groups": {},
                "ocr_unsupported": (),
            }

        model.forward = types.MethodType(empty_forward, model)
        engine = PrismOCR(model, vocabulary)
        with tempfile.TemporaryDirectory() as temporary:
            image_path = Path(temporary) / "page.png"
            Image.new("RGB", (32, 32), "white").save(image_path)
            with patch.object(PrismOCR, "load", return_value=engine):
                result = benchmark(
                    temporary,
                    image_path,
                    warmup=0,
                    iterations=1,
                )
        self.assertEqual(result["measurement_kind"], "single_page_diagnostic")
        workload = result["workloads"][0]
        self.assertEqual(workload["status"], "measured")
        self.assertEqual(workload["measured_runs"], 1)
        self.assertIn("end_to_end", workload["latency_ms"])
        self.assertIn("preprocessing", workload["latency_ms"])
        self.assertIn("model_forward", workload["latency_ms"])
        self.assertIn("graph_decode", workload["latency_ms"])
        self.assertIn("table_decode", workload["latency_ms"])
        self.assertTrue(result["protocol"]["tiling_enabled"])
        self.assertEqual(result["status"], "incomplete_workload_or_failed_inference")


if __name__ == "__main__":
    unittest.main()
