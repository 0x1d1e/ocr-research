from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from prismocr.data.dataset import PrismPageDataset
from prismocr.data.tools import audit_manifest, build_document_splits, convert_icdar, write_split_report


class DataToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_page(
        self,
        directory: Path,
        page_id: str,
        pixels: np.ndarray,
        *,
        document_id: str | None = None,
        family_id: str | None = None,
        lines: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        directory.mkdir(parents=True, exist_ok=True)
        image_path = directory / f"{page_id}.png"
        Image.fromarray(pixels).save(image_path)
        document = document_id or f"doc-{page_id}"
        provenance: dict[str, str] = {
            "source_class": "synthetic",
            "license": "generated",
            "permission_reference": "not_applicable",
            "reviewer": "test",
        }
        if family_id is not None:
            provenance["template_family_id"] = family_id
        page_lines = lines if lines is not None else []
        annotation = {
            "schema_version": "4.0",
            "document_id": document,
            "page_id": f"{document}:page-0",
            "image": {
                "width": int(pixels.shape[1]),
                "height": int(pixels.shape[0]),
                "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            },
            "provenance": provenance,
            "lines": page_lines,
            "regions": [],
            "cells": [],
            "relations": [],
            "reading_order": [line["id"] for line in page_lines],
        }
        return {"image_path": str(image_path), "annotation": annotation}

    @staticmethod
    def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_split_report_describes_pre_frozen_manifests_without_reassigning(self) -> None:
        directory = self.root / "images"
        rng = np.random.default_rng(3)
        rows = [
            self._write_page(
                directory, name, rng.integers(0, 255, (20, 20, 3), dtype=np.uint8)
            )
            for name in ("train-page", "validation-page", "test-page")
        ]
        output = self.root / "frozen"
        output.mkdir()
        manifests = {}
        for name, row in zip(("train", "validation", "test"), rows, strict=True):
            manifests[name] = output / f"{name}.jsonl"
            self._write_manifest(manifests[name], [row])

        report = write_split_report(output, manifests=manifests, seed=17, near_duplicate_hamming=0)

        self.assertEqual(report["status"], "grouped_by_document_template_family_and_exact_image_sha256")
        self.assertEqual(report["page_counts"], {"train": 1, "validation": 1, "test": 1})
        self.assertEqual(report["independent_group_count"], 3)
        self.assertEqual(
            report["manifest_sha256"]["validation"],
            hashlib.sha256(manifests["validation"].read_bytes()).hexdigest(),
        )
        groups = [set(report["split_fingerprints"][name]["group_sha256"]) for name in ("train", "validation", "test")]
        self.assertTrue(all(len(values) == 1 for values in groups))
        self.assertFalse(groups[0] & groups[1] | groups[0] & groups[2] | groups[1] & groups[2])
        self.assertTrue((output / "split-report.json").is_file())
        # Calibration is the consumer; its provenance check must accept the report.
        from prismocr.calibration import _validation_split_provenance, _verified_split_datasets

        _, validation_hash, _ = _validation_split_provenance(manifests["validation"], output / "split-report.json")
        self.assertEqual(validation_hash, report["manifest_sha256"]["validation"])
        self.assertEqual(
            {name: len(dataset) for name, dataset in _verified_split_datasets(report).items()},
            report["page_counts"],
        )

        with self.assertRaisesRegex(FileExistsError, "frozen split outputs"):
            write_split_report(output, manifests=manifests, seed=17)
        self.assertEqual(
            write_split_report(output, manifests=manifests, seed=17, overwrite=True)["page_counts"],
            report["page_counts"],
        )

        leaking = self.root / "leaking"
        leaking.mkdir()
        leak_manifests = {name: leaking / f"{name}.jsonl" for name in ("train", "validation", "test")}
        for name, row in zip(("train", "validation", "test"), (rows[0], rows[0], rows[2]), strict=True):
            self._write_manifest(leak_manifests[name], [row])
        with self.assertRaisesRegex(ValueError, "spans more than one split"):
            write_split_report(leaking, manifests=leak_manifests, seed=17)

    def test_split_report_perceptual_grouping_is_opt_in(self) -> None:
        directory = self.root / "images"
        pixels = np.zeros((20, 20, 3), dtype=np.uint8)
        pixels[:10] = 255
        neighbour = np.clip(pixels.astype(np.int16) + 1, 0, 255).astype(np.uint8)
        rows = [
            self._write_page(directory, "plain-page", pixels),
            self._write_page(directory, "neighbour-page", neighbour),
            self._write_page(directory, "other-page", np.full((20, 20, 3), 90, dtype=np.uint8)),
        ]
        output = self.root / "frozen"
        output.mkdir()
        manifests = {name: output / f"{name}.jsonl" for name in ("train", "validation", "test")}
        for name, row in zip(("train", "validation", "test"), rows, strict=True):
            self._write_manifest(manifests[name], [row])

        # Near-identical pages in different splits are one perceptual group unless the
        # policy is disabled, which is what a corpus of unrelated near-blank pages needs.
        with self.assertRaisesRegex(ValueError, "spans more than one split"):
            write_split_report(output, manifests=manifests, seed=17, near_duplicate_hamming=2)
        report = write_split_report(output, manifests=manifests, seed=17)
        self.assertIsNone(report["near_duplicate_hamming"])
        self.assertEqual(report["independent_group_count"], 3)

    def test_split_report_rejects_incomplete_manifest_sets(self) -> None:
        row = self._write_page(self.root / "images", "only-page", np.zeros((8, 8, 3), dtype=np.uint8))
        output = self.root / "frozen"
        output.mkdir()
        manifests = {name: output / f"{name}.jsonl" for name in ("train", "validation", "test")}
        for path in manifests.values():
            self._write_manifest(path, [row])

        with self.assertRaisesRegex(ValueError, "must name exactly"):
            write_split_report(output, manifests={"train": manifests["train"]}, seed=17)
        with self.assertRaisesRegex(ValueError, "missing frozen split manifest"):
            write_split_report(output, manifests={**manifests, "test": output / "absent.jsonl"}, seed=17)
        with self.assertRaisesRegex(ValueError, "is empty"):
            self._write_manifest(manifests["test"], [])
            write_split_report(output, manifests=manifests, seed=17)

    def test_icdar_conversion_preserves_quadrilaterals_and_governance(self) -> None:
        image_dir = self.root / "images"
        gt_dir = self.root / "gt"
        image_dir.mkdir()
        gt_dir.mkdir()
        Image.fromarray(np.zeros((20, 30, 3), dtype=np.uint8)).save(image_dir / "page-1.png")
        (gt_dir / "gt_page-1.txt").write_text(
            '1,2,15,2,15,8,1,8, "Khmer, text"\n'
            "1,10,15,10,15,17,1,17,Khmer, text\n"
            "1,10,15,10,15,17,1,17,###\n",
            encoding="utf-8",
        )
        output = self.root / "converted.jsonl"

        result = convert_icdar(
            image_dir,
            gt_dir,
            output,
            document_prefix="icdar/train",
            license_name="CC-BY-4.0",
            permission_reference="dataset-license-123",
            reviewer="qa-person",
            template_family_id="form-v1",
        )

        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(result["pages"], 1)
        self.assertEqual(result["lines"], 2)
        self.assertEqual(result["ignored_dont_care_regions"], 1)
        self.assertEqual(result["status"], "converted_requires_human_label_qa")
        annotation = rows[0]["annotation"]
        self.assertEqual(annotation["lines"][0]["text"], "Khmer, text")
        self.assertEqual(annotation["lines"][0]["polygon_px"], [[1.0, 2.0], [15.0, 2.0], [15.0, 8.0], [1.0, 8.0]])
        self.assertEqual(annotation["lines"][1]["text"], "Khmer, text")
        self.assertEqual(
            annotation["reading_order"],
            [line["id"] for line in annotation["lines"]],
        )
        self.assertEqual(annotation["provenance"]["permission_reference"], "dataset-license-123")
        self.assertEqual(annotation["provenance"]["template_family_id"], "form-v1")
        dataset = PrismPageDataset(output)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(annotation["supervision"], {"lines": "partial"})
        self.assertEqual(dataset[0][1]["supervision"], {"lines": "partial"})

    def test_icdar_complete_blank_and_ignored_only_are_distinct(self) -> None:
        image_dir, gt_dir = self.root / "images", self.root / "gt"
        image_dir.mkdir()
        gt_dir.mkdir()
        for name, text in (("blank", ""), ("ignored", "1,2,15,2,15,8,1,8,###\n"),
                           ("positive", "1,2,15,2,15,8,1,8,abc\n")):
            Image.new("RGB", (30, 20)).save(image_dir / f"{name}.png")
            (gt_dir / f"gt_{name}.txt").write_text(text)
        output = self.root / "converted.jsonl"
        convert_icdar(image_dir, gt_dir, output, document_prefix="test", license_name="generated",
                      permission_reference="fixture", reviewer="test")
        dataset = PrismPageDataset(output)
        self.assertEqual([dataset[index][1]["supervision"]["lines"] for index in range(3)],
                         ["complete", "unavailable", "complete"])

    def test_manifest_qa_rejects_declared_dimension_mismatch(self) -> None:
        directory = self.root / "dimension-data"
        row = self._write_page(directory, "dimension-page", np.zeros((16, 32, 3), dtype=np.uint8))
        row["annotation"]["image"]["width"] = 64
        manifest = self.root / "dimension.jsonl"
        self._write_manifest(manifest, [row])

        with self.assertRaisesRegex(ValueError, "image dimensions.*annotation says"):
            audit_manifest(manifest)

    def test_manifest_qa_reports_exact_and_perceptual_duplicate_candidates(self) -> None:
        directory = self.root / "qa-data"
        solid = np.zeros((16, 16, 3), dtype=np.uint8)
        alternating = np.tile(np.array([0, 255] * 8, dtype=np.uint8), (16, 1))
        near_alternating = np.clip(alternating.astype(np.int16) + 1, 0, 255).astype(np.uint8)
        rows = [
            self._write_page(directory, "exact-a", solid),
            self._write_page(directory, "exact-b", solid),
            self._write_page(directory, "near-a", alternating),
            self._write_page(directory, "near-b", near_alternating),
        ]
        manifest = self.root / "qa.jsonl"
        self._write_manifest(manifest, rows)

        report = audit_manifest(manifest, near_duplicate_hamming=2)

        self.assertEqual(report["page_count"], 4)
        self.assertEqual(report["document_count"], 4)
        self.assertEqual(report["exact_duplicate_image_groups"], [[0, 1]])
        self.assertEqual(report["near_duplicate_same_hash_groups"][0]["page_count"], 2)
        self.assertEqual(report["status"], "manifest_content_and_schema_audited_not_annotation_quality")
        self.assertTrue(any("human review" in item for item in report["limitations"]))

    def test_split_builder_keeps_exact_near_duplicate_and_family_groups_together(self) -> None:
        directory = self.root / "split-data"
        solid = np.zeros((16, 16, 3), dtype=np.uint8)
        near_solid = np.ones((16, 16, 3), dtype=np.uint8)
        decreasing = np.tile(np.arange(255, -1, -16, dtype=np.uint8)[:16], (16, 1))
        alternating = np.tile(np.array([0, 255] * 8, dtype=np.uint8), (16, 1))
        patterned = np.tile(np.array([0, 0, 255, 0, 255, 255, 0, 0, 0, 0, 255, 0, 255, 255, 0, 0], dtype=np.uint8), (16, 1))
        independent = np.random.default_rng(918).integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
        rows = [
            self._write_page(directory, "exact-a", solid),
            self._write_page(directory, "exact-b", solid),
            self._write_page(directory, "near-a", alternating),
            self._write_page(directory, "near-b", np.clip(alternating.astype(np.int16) + 1, 0, 255).astype(np.uint8)),
            self._write_page(directory, "family-a", decreasing, family_id="shared-template"),
            self._write_page(directory, "family-b", patterned, family_id="shared-template"),
            self._write_page(directory, "independent", independent),
        ]
        manifest = self.root / "split-source.jsonl"
        self._write_manifest(manifest, rows)

        first = build_document_splits(manifest, self.root / "splits-a", seed=91, near_duplicate_hamming=2)
        second = build_document_splits(manifest, self.root / "splits-b", seed=91, near_duplicate_hamming=2)
        frozen_test_path = Path(first["manifests"]["test"])
        frozen_test_bytes = frozen_test_path.read_bytes()
        with self.assertRaisesRegex(FileExistsError, "frozen split outputs"):
            build_document_splits(manifest, self.root / "splits-a", seed=92, near_duplicate_hamming=2)
        self.assertEqual(frozen_test_path.read_bytes(), frozen_test_bytes)

        self.assertEqual(first["page_counts"], second["page_counts"])
        self.assertEqual(first["independent_group_count"], 4)
        for split, manifest_path in first["manifests"].items():
            self.assertEqual(
                first["manifest_sha256"][split],
                hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
            )
        split_groups = [
            set(first["split_fingerprints"][split]["group_sha256"])
            for split in ("train", "validation", "test")
        ]
        self.assertFalse(split_groups[0] & split_groups[1])
        self.assertFalse(split_groups[0] & split_groups[2])
        self.assertFalse(split_groups[1] & split_groups[2])
        split_pages: dict[str, set[str]] = {}
        for split, manifest_path in first["manifests"].items():
            split_pages[split] = {
                record["_annotation"]["page_id"]
                for record in PrismPageDataset(manifest_path).records
            }
            second_rows = [
                record["_annotation"]["page_id"]
                for record in PrismPageDataset(second["manifests"][split]).records
            ]
            self.assertEqual(split_pages[split], set(second_rows))
        for left, right in (("exact-a", "exact-b"), ("near-a", "near-b"), ("family-a", "family-b")):
            containing = [name for name, page_ids in split_pages.items() if f"doc-{left}:page-0" in page_ids]
            self.assertEqual(len(containing), 1, left)
            self.assertIn(f"doc-{right}:page-0", split_pages[containing[0]])
        self.assertTrue(all(split_pages.values()))

    def test_source_crop_grouping_and_split_disjointness(self) -> None:
        from prismocr.data.tools import _group_records, _split_membership

        directory = self.root / "crop-split-data"
        p1 = self._write_page(
            directory,
            "crop-page-1",
            np.zeros((10, 10, 3), dtype=np.uint8),
            lines=[
                {
                    "id": "doc-crop-page-1:l0",
                    "polygon_px": [[0, 0], [10, 0], [10, 5], [0, 5]],
                    "text": "test",
                    "role": "PARA_LINE",
                    "source_crop_id": "crops/shared_crop.png",
                    "source_crop_hash": "a" * 64,
                }
            ],
        )
        p2 = self._write_page(
            directory,
            "crop-page-2",
            np.ones((10, 10, 3), dtype=np.uint8) * 100,
            lines=[
                {
                    "id": "doc-crop-page-2:l0",
                    "polygon_px": [[0, 0], [10, 0], [10, 5], [0, 5]],
                    "text": "other",
                    "role": "PARA_LINE",
                    "source_crop_id": "crops/shared_crop.png",
                    "source_crop_hash": "a" * 64,
                }
            ],
        )
        p3 = self._write_page(
            directory,
            "crop-page-3",
            np.ones((10, 10, 3), dtype=np.uint8) * 200,
            lines=[
                {
                    "id": "doc-crop-page-3:l0",
                    "polygon_px": [[0, 0], [10, 0], [10, 5], [0, 5]],
                    "text": "third",
                    "role": "PARA_LINE",
                    "source_crop_id": "crops/different_crop.png",
                    "source_crop_hash": "b" * 64,
                }
            ],
        )
        dataset = object.__new__(PrismPageDataset)
        dataset.records = [p1, p2, p3]
        groups = _group_records(dataset, near_duplicate_hamming=None)
        self.assertEqual(len(groups), 2)
        group_with_p1 = [g for g in groups if 0 in g][0]
        self.assertIn(1, group_with_p1)

        leak_split = {"train": [p1], "validation": [p2], "test": [p3]}
        with self.assertRaisesRegex(ValueError, "a source crop spans more than one split"):
            _split_membership(leak_split, near_duplicate_hamming=None)

        clean_split = {
            "train": [p1, p2],
            "validation": [p3],
            "test": [self._write_page(directory, "p4", np.full((10, 10, 3), 255, dtype=np.uint8))],
        }
        membership = _split_membership(clean_split, near_duplicate_hamming=None)
        self.assertEqual(membership["source_crop_counts"]["train"], 1)
        self.assertEqual(membership["source_crop_counts"]["validation"], 1)
        self.assertEqual(membership["source_crop_counts"]["test"], 0)
        self.assertEqual(membership["split_fingerprints"]["train"]["source_crop_sha256"], ["a" * 64])
        self.assertEqual(membership["split_fingerprints"]["validation"]["source_crop_sha256"], ["b" * 64])
        self.assertEqual(membership["split_fingerprints"]["test"]["source_crop_sha256"], [])

    def test_split_report_transcript_and_ngram_overlap(self) -> None:
        directory = self.root / "overlap-images"
        train_row = self._write_page(
            directory,
            "train-p",
            np.zeros((10, 10, 3), dtype=np.uint8),
            lines=[
                {"id": "l0", "polygon_px": [[0, 0], [10, 0], [10, 5], [0, 5]], "text": "ជិះម៉ូតបុករថយន្ត", "role": "PARA_LINE"},
                {"id": "l1", "polygon_px": [[0, 5], [10, 5], [10, 10], [0, 10]], "text": "apple pie", "role": "PARA_LINE"},
            ],
        )
        val_row = self._write_page(
            directory,
            "val-p",
            np.ones((10, 10, 3), dtype=np.uint8) * 50,
            lines=[
                {"id": "l2", "polygon_px": [[0, 0], [10, 0], [10, 5], [0, 5]], "text": "apple tree", "role": "PARA_LINE"},
                {"id": "l3", "polygon_px": [[0, 5], [10, 5], [10, 10], [0, 10]], "text": "banana", "role": "PARA_LINE"},
            ],
        )
        test_row = self._write_page(
            directory,
            "test-p",
            np.ones((10, 10, 3), dtype=np.uint8) * 100,
            lines=[
                {"id": "l4", "polygon_px": [[0, 0], [10, 0], [10, 5], [0, 5]], "text": "apple pie", "role": "PARA_LINE"},
            ],
        )
        output = self.root / "overlap-report"
        manifests = {
            "train": output / "train.jsonl",
            "validation": output / "val.jsonl",
            "test": output / "test.jsonl",
        }
        output.mkdir(parents=True, exist_ok=True)
        self._write_manifest(manifests["train"], [train_row])
        self._write_manifest(manifests["validation"], [val_row])
        self._write_manifest(manifests["test"], [test_row])

        report = write_split_report(output, manifests=manifests, seed=42)
        self.assertIn("transcript_overlap", report)
        overlap = report["transcript_overlap"]
        self.assertIn("train_vs_val", overlap)
        self.assertIn("train_vs_test", overlap)
        self.assertIn("val_vs_test", overlap)

        tv = overlap["train_vs_val"]
        self.assertEqual(tv["exact_shared_count"], 0)
        self.assertEqual(tv["exact_jaccard"], 0.0)
        self.assertGreater(tv["ngram_shared_count"], 0)
        self.assertGreater(tv["ngram_jaccard"], 0.0)
        self.assertEqual(tv["ngram_jaccard"], tv["char_3gram_jaccard"])

        tt = overlap["train_vs_test"]
        self.assertEqual(tt["exact_shared_count"], 1)
        self.assertEqual(tt["exact_jaccard"], 0.5)
        self.assertGreater(tt["ngram_shared_count"], 0)
        self.assertGreater(tt["ngram_jaccard"], 0.0)


if __name__ == "__main__":
    unittest.main()
