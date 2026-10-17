"""Tests for the PrismOCR Gradio demo application."""

from __future__ import annotations

from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import gradio as gr
import numpy as np
from PIL import Image

from prismocr.api import DocumentResult
from prismocr.demo import (
    annotate_document,
    build_demo,
    clear_model_cache,
    discover_artifacts,
    find_free_port,
    get_cached_model,
    is_port_available,
    main,
    parse_args,
    process_document,
)


class TestFindFreePort(unittest.TestCase):
    def test_find_free_port_default(self) -> None:
        port = find_free_port(start_port=7860, host="127.0.0.1")
        self.assertIsInstance(port, int)
        self.assertGreaterEqual(port, 7860)
        self.assertTrue(is_port_available(port, host="127.0.0.1"))

    def test_find_free_port_skips_occupied(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        occupied_port = sock.getsockname()[1]
        try:
            self.assertFalse(is_port_available(occupied_port, host="127.0.0.1"))
            next_free = find_free_port(start_port=occupied_port, host="127.0.0.1")
            self.assertGreater(next_free, occupied_port)
        finally:
            sock.close()

    def test_is_port_available_invalid(self) -> None:
        self.assertFalse(is_port_available(0))
        self.assertFalse(is_port_available(70000))


class TestDiscoverArtifacts(unittest.TestCase):
    def test_discover_nonexistent_returns_empty(self) -> None:
        self.assertEqual(discover_artifacts("/path/that/does/not/exist"), {})

    def test_discover_artifacts_temp_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)

            # Valid run 1: run1/artifact
            run1_art = base / "run1" / "artifact"
            run1_art.mkdir(parents=True)
            for f in ("model.pt", "model-config.json", "vocab-v4.json", "manifest.json"):
                (run1_art / f).write_text("{}", encoding="utf-8")

            # Valid run 2: run2 direct
            run2 = base / "run2"
            run2.mkdir()
            for f in ("model.pt", "model-config.json", "vocab-v4.json", "manifest.json"):
                (run2 / f).write_text("{}", encoding="utf-8")

            # Incomplete run 3: missing manifest.json
            run3 = base / "run3" / "artifact"
            run3.mkdir(parents=True)
            for f in ("model.pt", "model-config.json", "vocab-v4.json"):
                (run3 / f).write_text("{}", encoding="utf-8")

            # Incomplete run 4: missing model.pt
            run4 = base / "run4"
            run4.mkdir()
            for f in ("model-config.json", "vocab-v4.json", "manifest.json"):
                (run4 / f).write_text("{}", encoding="utf-8")

            discovered = discover_artifacts(base)
            self.assertIn("run1", discovered)
            self.assertEqual(discovered["run1"], str(run1_art.resolve()))
            self.assertIn("run2", discovered)
            self.assertEqual(discovered["run2"], str(run2.resolve()))
            self.assertNotIn("run3", discovered)
            self.assertNotIn("run4", discovered)

    def test_discover_default_runs_dir(self) -> None:
        default_dir = Path("/home/kosal/data/prismocr-runs")
        if default_dir.is_dir():
            artifacts = discover_artifacts()
            self.assertIsInstance(artifacts, dict)
            self.assertIn("real-v4-ep4-checkpoint", artifacts)
            self.assertIn("real-v3", artifacts)


class TestAnnotateDocument(unittest.TestCase):
    def test_annotate_empty_leaves_copy_unaltered(self) -> None:
        img = Image.new("RGB", (60, 60), color=(255, 255, 255))
        result = DocumentResult(payload={"lines": [], "regions": []})
        annotated = annotate_document(img, result)

        self.assertIsNot(annotated, img)
        self.assertEqual(annotated.size, img.size)
        self.assertEqual(np.array(annotated).tolist(), np.array(img).tolist())

    def test_annotate_lines_and_regions(self) -> None:
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        result = DocumentResult(payload={
            "lines": [
                {
                    "polygon": [[10, 10], [50, 10], [50, 30], [10, 30]],
                    "text": "Header",
                }
            ],
            "regions": [
                {
                    "polygon": [[5, 5], [60, 5], [60, 40], [5, 40]],
                    "kind": "PARAGRAPH",
                },
                {
                    "polygon": [[10, 60], [80, 60], [80, 90], [10, 90]],
                    "kind": "TABLE",
                },
            ],
            "cells": [
                {
                    "polygon": [[12, 62], [40, 62], [40, 80], [12, 80]],
                }
            ],
        })

        annotated = annotate_document(img, result)
        self.assertIsNot(annotated, img)
        arr_orig = np.array(img)
        arr_annot = np.array(annotated)

        # Original is still all white
        self.assertTrue((arr_orig == 255).all())
        # Annotated has modified pixels (colored outlines)
        self.assertFalse((arr_annot == 255).all())

    def test_annotate_normalized_coordinates(self) -> None:
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        result = DocumentResult(payload={
            "coordinate_space": "normalized",
            "lines": [
                {
                    "polygon": [[0.1, 0.1], [0.5, 0.1], [0.5, 0.3], [0.1, 0.3]],
                    "text": "Normalized Line",
                }
            ],
            "regions": [],
        })
        annotated = annotate_document(img, result)
        arr = np.array(annotated)
        self.assertFalse((arr == 255).all())


class TestModelCaching(unittest.TestCase):
    def setUp(self) -> None:
        clear_model_cache()

    def tearDown(self) -> None:
        clear_model_cache()

    @patch("prismocr.demo.PrismOCR.load")
    def test_get_cached_model_avoids_repeated_disk_reads(self, mock_load: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_load.return_value = mock_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "checkpoint"
            model_path.mkdir()

            m1 = get_cached_model(model_path, device="cpu")
            self.assertIs(m1, mock_instance)
            self.assertEqual(mock_load.call_count, 1)

            # Second call with same path & device hits cache
            m2 = get_cached_model(model_path, device="cpu")
            self.assertIs(m2, mock_instance)
            self.assertEqual(mock_load.call_count, 1)

            # Different device calls load again
            get_cached_model(model_path, device="cuda")
            self.assertEqual(mock_load.call_count, 2)


class TestBuildDemo(unittest.TestCase):
    def setUp(self) -> None:
        clear_model_cache()

    def tearDown(self) -> None:
        clear_model_cache()

    def test_build_demo_creates_blocks_without_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            demo = build_demo(base_dir=tmpdir)
            self.assertIsInstance(demo, gr.Blocks)

    def test_process_document_success_and_reading_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            run = base / "run1" / "artifact"
            run.mkdir(parents=True)
            for f in ("model.pt", "model-config.json", "vocab-v4.json", "manifest.json"):
                (run / f).write_text("{}", encoding="utf-8")

            # Mock model output with lines in reverse reading order
            mock_payload = {
                "schema_version": "4.0",
                "model_version": "v4-test",
                "coordinate_space": "pixels",
                "lines": [
                    {
                        "id": "line2",
                        "reading_order": 1,
                        "text": "Second paragraph text",
                        "polygon": [[10, 40], [80, 40], [80, 50], [10, 50]],
                    },
                    {
                        "id": "line1",
                        "reading_order": 0,
                        "text": "First heading text",
                        "polygon": [[10, 10], [80, 10], [80, 20], [10, 20]],
                    },
                ],
                "regions": [
                    {
                        "id": "r1",
                        "kind": "PARAGRAPH",
                        "polygon": [[5, 5], [90, 5], [90, 60], [5, 60]],
                    }
                ],
                "cells": [],
                "tables": [],
                "diagnostics": {"tiles": 1, "tile_model_calls": 1},
                "warnings": [],
            }
            mock_model = MagicMock()
            mock_model.predict.return_value = DocumentResult(payload=mock_payload)

            with patch("prismocr.demo.get_cached_model", return_value=mock_model):
                img = Image.new("RGB", (100, 100), color=(255, 255, 255))
                annotated, text, payload, diag_md = process_document(
                    image=img,
                    model_name="run1",
                    device="cpu",
                    resolution="Original",
                    allow_tiling=True,
                    base_dir=base,
                )

                self.assertIsInstance(annotated, Image.Image)
                # Extracted text should be ordered by reading order: 0 first, then 1
                self.assertEqual(text, "First heading text\nSecond paragraph text")
                self.assertEqual(payload["model_version"], "v4-test")
                self.assertIn("Latency", diag_md)
                self.assertIn("v4-test", diag_md)

    def test_process_document_no_image(self) -> None:
        annotated, text, payload, diag = process_document(
            image=None,
            model_name="test",
            device="cpu",
        )
        self.assertIsNone(annotated)
        self.assertEqual(text, "")
        self.assertEqual(payload, {})
        self.assertIn("No Image Provided", diag)

    def test_process_document_missing_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Image.new("RGB", (50, 50), color=(255, 255, 255))
            annotated, text, payload, diag = process_document(
                image=img,
                model_name="nonexistent_checkpoint",
                device="cpu",
                base_dir=tmpdir,
            )
            self.assertIsNone(annotated)
            self.assertEqual(text, "")
            self.assertEqual(payload, {})
            self.assertIn("not found", diag)


class TestMainCLI(unittest.TestCase):
    def test_parse_args_defaults(self) -> None:
        args = parse_args([])
        self.assertEqual(args.host, "0.0.0.0")
        self.assertIsNone(args.port)
        self.assertIsNone(args.runs_dir)
        self.assertFalse(args.share)

    def test_parse_args_custom(self) -> None:
        args = parse_args([
            "--host", "127.0.0.1",
            "--port", "8080",
            "--runs-dir", "/tmp/runs",
            "--share",
        ])
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8080)
        self.assertEqual(args.runs_dir, "/tmp/runs")
        self.assertTrue(args.share)

    @patch("prismocr.demo.build_demo")
    @patch("prismocr.demo.is_port_available", return_value=True)
    def test_main_launches_with_specified_port(
        self, _mock_is_avail: MagicMock, mock_build: MagicMock
    ) -> None:
        mock_demo = MagicMock()
        mock_build.return_value = mock_demo

        main(["--host", "127.0.0.1", "--port", "7900", "--share"])
        mock_build.assert_called_once_with(base_dir=None)
        mock_demo.launch.assert_called_once_with(
            server_name="127.0.0.1", server_port=7900, share=True
        )

    @patch("prismocr.demo.build_demo")
    @patch("prismocr.demo.find_free_port", return_value=7862)
    @patch("prismocr.demo.is_port_available", return_value=False)
    def test_main_uses_free_port_when_unavailable(
        self, _mock_avail: MagicMock, mock_find: MagicMock, mock_build: MagicMock
    ) -> None:
        mock_demo = MagicMock()
        mock_build.return_value = mock_demo

        main(["--port", "7860"])
        mock_find.assert_called_once_with(start_port=7860, host="0.0.0.0")
        mock_demo.launch.assert_called_once_with(
            server_name="0.0.0.0", server_port=7862, share=False
        )


if __name__ == "__main__":
    unittest.main()
