"""Critical metric/selection invariants for the local diagnostic, not model quality tests."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch
from PIL import Image

from prismocr.geometry import rotated_iou
from scripts.diagnose_ocr import aggregate_public, aggregate_recognition, box_quality, decode_stats, public_probe, select_manifest
from prismocr.data.vocabulary import Vocabulary


class DiagnosisTests(unittest.TestCase):
    def test_failed_buckets_remain_in_cer_denominator_but_not_blank_rate(self):
        rows = [
            {"reference": "ab", "text": "ab", "frames": 4,
             "frame_blank_posterior_mean": 0.2, "frame_blank_top1_fraction": 0.25},
            {"reference": "cd", "text": "", "failure": "no_supported_bucket"},
            {"reference": "ef", "text": "", "frames": 8,
             "frame_blank_posterior_mean": 0.8, "frame_blank_top1_fraction": 1.0},
        ]
        result = aggregate_recognition(rows)
        self.assertAlmostEqual(result["cer_failures_as_empty"], 4 / 6)
        self.assertEqual(result["bucket_failures"], 1)
        self.assertEqual(result["blank_outputs"], 1)
        self.assertEqual(result["blank_output_rate_available"], 0.5)
        self.assertAlmostEqual(result["frame_blank_posterior_mean"], 0.6)

    def test_cached_geometry_matches_production_iou_with_padded_axes(self):
        gt = [[0.3, 0.4, 0.4, 0.03, 0.1, 0.994987], [0.7, 0.8, 0.1, 0.05, 0.0, 1.0]]
        pred = [gt[0], [0.31, 0.405, 0.35, 0.04, -0.1, 0.994987], gt[1]]
        for size in ((1024, 1408), (1600, 912)):
            actual = box_quality(gt, pred, size)
            expected = [[rotated_iou(g, p, size) for p in pred] for g in gt]
            np.testing.assert_allclose(actual, expected, atol=1e-9)
            self.assertAlmostEqual(actual[0, 0], 1.0)
            self.assertEqual(actual[0, 2], 0.0)

    def test_selection_is_manifest_order_per_stratum_and_no_sibling_writes(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            records = []
            for doc, lines in (("doclaynet:1", []), ("fintech-val:0", [{"text": "a"}]),
                               ("fintech-val:1", [{"text": "b"}]), ("khmerst:1", [{"text": "c"}])):
                records.append({"annotation": {"document_id": doc, "lines": lines}, "image_path": "image.jpg"})
            manifest = root / "input.jsonl"
            original = "\n".join(json.dumps(r) for r in records)
            manifest.write_text(original)
            selected = select_manifest(manifest, root / "selected.jsonl", 1, 12)
            self.assertEqual([r["annotation"]["document_id"] for r in selected],
                             ["fintech-val:0", "khmerst:1", "doclaynet:1"])
            self.assertEqual(manifest.read_text(), original)
            self.assertTrue(all(Path(r["image_path"]).is_absolute() for r in selected))
            self.assertEqual([r["diagnosis_selection"]["text_supervision"] for r in selected],
                             ["annotated", "annotated", "unlabelled"])

    def test_unlabelled_layout_emissions_excluded_but_supervised_blank_insertions_count(self):
        polygon = [[1, 1], [9, 1], [9, 5], [1, 5]]
        payload = {"lines": [{"text": "abc", "polygon": polygon}],
                   "diagnostics": {"candidate_count": 1}, "warnings": []}
        engine = SimpleNamespace(model=torch.nn.Identity(),
                                 predict=Mock(return_value=SimpleNamespace(to_dict=lambda: payload)))
        blank_annotation = {"image": {"width": 12, "height": 12}, "lines": []}
        text_annotation = {**blank_annotation, "lines": [{"text": "abc", "polygon_px": polygon}]}
        with tempfile.TemporaryDirectory() as root:
            image_path = Path(root) / "page.png"
            Image.new("RGB", (12, 12)).save(image_path)
            for allow_tiling in (True, False):
                with self.subTest(allow_tiling=allow_tiling):
                    unlabelled, result = public_probe(engine, image_path, blank_annotation, 1600,
                                                     allow_tiling=allow_tiling, text_supervised=False)
                    blank, _ = public_probe(engine, image_path, blank_annotation, 1600,
                                            allow_tiling=allow_tiling, text_supervised=True)
                    text, _ = public_probe(engine, image_path, text_annotation, 1600,
                                           allow_tiling=allow_tiling, text_supervised=True)
                    self.assertEqual(unlabelled["text_supervision"], "unlabelled")
                    self.assertIsNone(unlabelled["reference_characters"])
                    self.assertIsNone(unlabelled["character_errors_including_unmatched_insertions"])
                    self.assertIsNone(unlabelled["e2e_cer"])
                    self.assertEqual(unlabelled["detections"], 1)
                    self.assertEqual(unlabelled["diagnostics"], payload["diagnostics"])
                    self.assertEqual(result["lines"][0]["text"], "abc")
                    self.assertEqual(blank["text_supervision"], "annotated")
                    self.assertEqual(blank["reference_characters"], 0)
                    self.assertEqual(blank["character_errors_including_unmatched_insertions"], 3)
                    self.assertIsNone(blank["e2e_cer"])
                    excluded = aggregate_public([text, unlabelled])
                    penalized = aggregate_public([text, blank])
                    self.assertEqual(excluded["character_errors_including_unmatched_insertions"], 0)
                    self.assertEqual(excluded["reference_characters"], 3)
                    self.assertEqual(excluded["e2e_cer"], 0.0)
                    self.assertEqual(excluded["text_unlabelled_pages"], 1)
                    self.assertEqual(excluded["text_unlabelled_detections"], 1)
                    self.assertEqual(excluded["detections"], 2)
                    self.assertEqual(penalized["character_errors_including_unmatched_insertions"], 3)
                    self.assertEqual(penalized["reference_characters"], 3)
                    self.assertEqual(penalized["e2e_cer"], 1.0)
                    self.assertEqual(penalized["text_supervised_pages"], 2)

    def test_ctc_blank_posterior_separate_from_collapsed_text(self):
        vocabulary = Vocabulary.from_texts(["a"])
        logits = torch.full((1, 4, len(vocabulary)), -20.0)
        logits[0, 0, 2] = logits[0, 1, 2] = logits[0, 2, 0] = logits[0, 3, 2] = 20.0
        result = decode_stats(logits, vocabulary)[0]
        self.assertEqual(result["text"], "aa")
        self.assertAlmostEqual(result["frame_blank_posterior_mean"], 0.25)
        self.assertAlmostEqual(result["frame_blank_top1_fraction"], 0.25)


if __name__ == "__main__":
    unittest.main()
