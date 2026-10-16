from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from prismocr.data.dataset import PrismPageDataset
from prismocr.data.real_corpus import (
    _rotate_crop,
    assign_scene_splits,
    compose_recognition_pages,
    concat_manifests,
    convert_coco_layout,
    convert_khmerst_scene,
    read_khmerst_annotations,
    read_recognition_labels,
)


class RecognitionCompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.crops = self.root / "crops"
        self.crops.mkdir()
        self.texts = [
            "ជិះម៉ូតបុករថយន្ត",
            "Khmer OCR 2026",
            "សម្គាល់ឱ្យបានត្រឹមត្រូវ",
            "invoice total 1,250.00",
        ]
        for index, _ in enumerate(self.texts):
            image = Image.new("RGB", (60, 10), (255, 255, 255))
            ImageDraw.Draw(image).rectangle([4, 1, 55, 8], fill=(0, 0, 0))
            image.save(self.crops / f"line-{index}.png")
        self.labels = self.root / "labels.txt"
        self.labels.write_text(
            "".join(f"crops/line-{index}.png\t{text}\n" for index, text in enumerate(self.texts)),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _compose(self, output: Path, **overrides):
        options = {
            "pages": 3,
            "license_name": "CC-BY-4.0",
            "permission_reference": "local-corpus-permission-7",
            "reviewer": "qa-person",
            "template_family_id": "composed-lines-v1",
            "source_class": "derived",
            "seed": 17,
            "image_size": (256, 320),
            "columns": 1,
            "lines_per_page": (3, 3),
            "blank_page_fraction": 0.0,
            "rotation_degrees": 0.0,
            "scale_range": (1.0, 1.0),
            "margin": 8,
            "gutter": 4,
            "line_padding": 1,
        }
        options.update(overrides)
        return compose_recognition_pages(self.labels, output, **options)

    def test_composed_pages_keep_crop_geometry_transcripts_and_provenance(self) -> None:
        result = self._compose(self.root / "composed")
        self.assertEqual(result["pages"], 3)
        self.assertEqual(result["lines"], 9)
        self.assertEqual(result["regions"], 3)
        self.assertEqual(result["status"], "converted_requires_human_label_qa")

        dataset = PrismPageDataset(result["manifest"])
        self.assertEqual(len(dataset), 3)
        for record in dataset.records:
            annotation = record["_annotation"]
            self.assertEqual(annotation["supervision"], {"lines": "complete"})
            self.assertEqual(annotation["provenance"]["source_class"], "derived")
            self.assertEqual(annotation["provenance"]["license"], "CC-BY-4.0")
            self.assertEqual(annotation["provenance"]["template_family_id"], "composed-lines-v1")
            self.assertEqual(annotation["reading_order"], [line["id"] for line in annotation["lines"]])
            self.assertEqual(len(annotation["regions"]), 1)
            region = annotation["regions"][0]["polygon_px"]
            page = np.asarray(Image.open(record["_image_path"]).convert("L"))
            self.assertEqual(page.shape, (320, 256))
            for line in annotation["lines"]:
                self.assertIn(line["text"], self.texts)
                self.assertEqual(line["role"], "PARA_LINE")
                xs = [point[0] for point in line["polygon_px"]]
                ys = [point[1] for point in line["polygon_px"]]
                self.assertGreaterEqual(min(xs), 0.0)
                self.assertGreaterEqual(min(ys), 0.0)
                self.assertLessEqual(max(xs), 256.0)
                self.assertLessEqual(max(ys), 320.0)
                self.assertGreaterEqual(min(xs), min(point[0] for point in region) - 1e-6)
                self.assertLessEqual(max(xs), max(point[0] for point in region) + 1e-6)
                patch = page[
                    int(min(ys)) + 1 : int(max(ys)) - 1,
                    int(min(xs)) + 1 : int(max(xs)) - 1,
                ]
                self.assertLess(float(patch.mean()), 200.0, line["id"])

    def test_composition_is_deterministic_for_a_seed(self) -> None:
        output = self.root / "deterministic"
        first = self._compose(output)
        first_rows = [row["annotation"] for row in PrismPageDataset(first["manifest"]).records]
        second = self._compose(output)
        second_rows = [row["annotation"] for row in PrismPageDataset(second["manifest"]).records]
        self.assertEqual(json.dumps(first_rows), json.dumps(second_rows))
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])

    def test_rotated_crop_polygon_matches_the_pasted_pixels(self) -> None:
        crop = Image.new("RGB", (40, 12), (0, 0, 0))
        tile, polygon = _rotate_crop(crop, 25.0)
        self.assertGreater(tile.size[0], 40)
        self.assertGreater(tile.size[1], 12)
        center_x = sum(point[0] for point in polygon) / len(polygon)
        center_y = sum(point[1] for point in polygon) / len(polygon)
        pixels = np.asarray(tile.convert("L"))
        for point in polygon:
            inner_x = round(center_x + 0.8 * (point[0] - center_x))
            inner_y = round(center_y + 0.8 * (point[1] - center_y))
            self.assertLess(int(pixels[inner_y, inner_x]), 60, point)
        self.assertGreater(int(pixels[0, 0]), 200)

    def test_blank_pages_are_schema_valid_and_empty(self) -> None:
        result = self._compose(self.root / "blank", blank_page_fraction=0.999, pages=2)
        dataset = PrismPageDataset(result["manifest"])
        self.assertTrue(all(len(record["_annotation"]["lines"]) == 0 for record in dataset.records))
        self.assertTrue(all(record["_annotation"]["supervision"] == {"lines": "complete"} for record in dataset.records))
        self.assertEqual(dataset[0][1]["supervision"], {"lines": "complete"})

    def test_recognition_label_reader_counts_and_rejects(self) -> None:
        labels = self.root / "messy.txt"
        labels.write_text(
            "crops/line-0.png\tជិះម៉ូតបុករថយន្ត\n"
            "crops/missing.png\tmissing crop\n"
            "crops/line-1.png\t\n"
            f"crops/line-2.png\t{'x' * 200}\n",
            encoding="utf-8",
        )
        rows, skipped = read_recognition_labels(labels, max_characters=160)
        self.assertEqual(len(rows), 1)
        self.assertEqual(skipped["missing_file"], 1)
        self.assertEqual(skipped["empty_text"], 1)
        self.assertEqual(skipped["too_long"], 1)
        (self.root / "bad.txt").write_text("no-tab-here\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "expected 'image_path<TAB>text'"):
            read_recognition_labels(self.root / "bad.txt")

    def test_rotated_lines_stay_inside_the_page_bounds(self) -> None:
        result = self._compose(
            self.root / "rotated-bounds",
            pages=3,
            image_size=(160, 320),
            columns=2,
            lines_per_page=(6, 6),
            rotation_degrees=15.0,
            scale_range=(1.2, 1.2),
            margin=8,
            gutter=4,
        )
        dataset = PrismPageDataset(result["manifest"])
        placed = 0
        for record in dataset.records:
            for line in record["_annotation"]["lines"]:
                placed += 1
                for x, y in line["polygon_px"]:
                    self.assertGreaterEqual(x, 0.0, line["id"])
                    self.assertGreaterEqual(y, 0.0, line["id"])
                    self.assertLessEqual(x, 160.0, line["id"])
                    self.assertLessEqual(y, 320.0, line["id"])
        self.assertGreater(placed, 0)

    def test_two_column_pages_keep_unique_line_ids(self) -> None:
        result = self._compose(self.root / "two-column", columns=2, lines_per_page=(4, 4), pages=2)
        dataset = PrismPageDataset(result["manifest"])
        for record in dataset.records:
            annotation = record["_annotation"]
            ids = [line["id"] for line in annotation["lines"]]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(annotation["reading_order"], ids)
            self.assertEqual(len(annotation["regions"]), 2)
            scope_ids = {line["scope_id"] for line in annotation["lines"]}
            self.assertEqual(len(scope_ids), 2)

    def test_concat_manifests_merges_rows(self) -> None:
        first = self._compose(self.root / "concat-a", pages=1)
        second = self._compose(self.root / "concat-b", pages=2)
        merged = concat_manifests([first["manifest"], second["manifest"]], self.root / "merged.jsonl")
        self.assertEqual(merged["pages"], 3)
        self.assertEqual(len(PrismPageDataset(merged["manifest"])), 3)


class KhmerstSceneConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.images = self.root / "scene images"
        self.images.mkdir()
        for index, name in enumerate(("SCENE_0001.jpg", "SCENE_0002.jpg", "SCENE_0003.jpg", "SCENE_0004.jpg")):
            pixels = np.full((60, 80, 3), 255, dtype=np.uint8)
            # Distinct per page so content-dedup does not merge the fixtures.
            pixels[30 + index : 34 + index, 2 : 6 + index] = index * 20
            Image.fromarray(pixels).save(self.images / name)
        # Lower line listed first so conversion must sort it back to reading order.
        payload = {
            "SCENE_0001.jpg100": {
                "filename": "SCENE_0001.jpg",
                "regions": [
                    {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": [40, 70, 70, 40],
                            "all_points_y": [40, 40, 50, 50],
                        },
                        "region_attributes": {"Name": "បន្ទាត់ទីពីរ"},
                    },
                    {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": [5, 35, 35, 5],
                            "all_points_y": [10, 10, 20, 20],
                        },
                        "region_attributes": {"Name": "បន្ទាត់ទីមួយ"},
                    },
                    {
                        "shape_attributes": {
                            "name": "rectangle",
                            "all_points_x": [0, 1],
                            "all_points_y": [0, 1],
                        },
                        "region_attributes": {"Name": "ignored"},
                    },
                    {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": [0, 10, 10, 0],
                            "all_points_y": [55, 55, 59, 59],
                        },
                        "region_attributes": {"Name": "   "},
                    },
                ],
            },
            "SCENE_0002.jpg200": {
                "filename": "SCENE_0002.jpg",
                "regions": [
                    {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": [0, 80, 80, 0],
                            "all_points_y": [5, 5, 6, 6],
                        },
                        "region_attributes": {"Name": ""},
                    }
                ],
            },
            "SCENE_0003.jpg300": {
                "filename": "SCENE_0003.jpg",
                "regions": [
                    {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": [10, 60, 60, 10],
                            "all_points_y": [20, 20, 30, 30],
                        },
                        "region_attributes": {"Name": "អក្សរខ្មែរ"},
                    }
                ],
            },
            "SCENE_0004.jpg400": {
                "filename": "SCENE_0004.jpg",
                "regions": [
                    {
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": [12, 52, 52, 12],
                            "all_points_y": [8, 8, 16, 16],
                        },
                        "region_attributes": {"Name": "បន្ទាត់ទីបី"},
                    }
                ],
            },
        }
        self.annotations = self.root / "annotations"
        self.annotations.mkdir()
        for index, (key, entry) in enumerate(payload.items()):
            (self.annotations / f"SCENE_000{index + 1}.json").write_text(
                json.dumps({key: entry}, ensure_ascii=False), encoding="utf-8"
            )
        self.zip_path = self.root / "via.zip"
        with zipfile.ZipFile(self.zip_path, "w") as archive:
            for path in sorted(self.annotations.glob("*.json")):
                archive.writestr(f"json_v4/{path.name}", path.read_text(encoding="utf-8"))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _convert(self, output: Path, **overrides):
        options = {
            "license_name": "Apache-2.0",
            "permission_reference": "khmerst-public-release",
            "reviewer": "scene-qa",
            "document_prefix": "khmerst",
            "split_assignment": {
                "SCENE_0001.jpg": "train",
                "SCENE_0002.jpg": "val",
                "SCENE_0003.jpg": "test",
                "SCENE_0004.jpg": "val",
            },
        }
        options.update(overrides)
        return convert_khmerst_scene(self.zip_path, self.images, output, **options)

    def test_scene_annotations_convert_to_real_line_supervision(self) -> None:
        records, skipped = read_khmerst_annotations(self.zip_path)
        self.assertEqual(
            [record["filename"] for record in records],
            ["SCENE_0001.jpg", "SCENE_0003.jpg", "SCENE_0004.jpg"],
        )
        self.assertEqual(skipped["unsupported_shape"], 1)
        self.assertEqual(skipped["empty_text"], 2)
        self.assertEqual(skipped["empty_image_entry"], 1)

        result = self._convert(self.root / "scene.jsonl")
        self.assertEqual(result["pages"], 3)
        self.assertEqual(result["lines"], 4)
        self.assertEqual(result["status"], "converted_requires_human_label_qa")
        dataset = PrismPageDataset(result["manifest"])
        first = dataset.records[0]["_annotation"]
        self.assertEqual(first["supervision"], {"lines": "partial"})
        self.assertEqual(dataset[0][1]["supervision"], {"lines": "partial"})
        self.assertEqual(dataset.records[-1]["_annotation"]["supervision"], {"lines": "complete"})
        self.assertEqual(first["document_id"], "khmerst:SCENE_0001")
        self.assertEqual(first["provenance"]["source_class"], "real")
        self.assertEqual(first["provenance"]["license"], "Apache-2.0")
        self.assertEqual(first["provenance"]["template_family_id"], "SCENE_0001")
        self.assertEqual(first["regions"], [])
        self.assertEqual(first["reading_order"], [line["id"] for line in first["lines"]])
        self.assertEqual([line["text"] for line in first["lines"]], ["បន្ទាត់ទីមួយ", "បន្ទាត់ទីពីរ"])
        self.assertEqual({line["role"] for line in first["lines"]}, {"UNKNOWN"})
        self.assertEqual(len({line["scope_id"] for line in first["lines"]}), 2)
        self.assertGreater(_signed_area(first["lines"][0]["polygon_px"]), 0.0)
        for line in first["lines"]:
            for x, y in line["polygon_px"]:
                self.assertGreaterEqual(x, 0.0)
                self.assertGreaterEqual(y, 0.0)
                self.assertLessEqual(x, 80.0)
                self.assertLessEqual(y, 60.0)

    def test_mapping_region_container_preserves_positives_and_known_omissions(self) -> None:
        payload = json.loads((self.annotations / "SCENE_0001.json").read_text())
        entry = next(iter(payload.values()))
        entry["regions"] = {str(index): region for index, region in enumerate(entry["regions"])}
        # Merge another valid list payload for the same filename: neither stream
        # may silently disappear while certifying the remaining labels complete.
        second = json.loads((self.annotations / "SCENE_0004.json").read_text())
        next(iter(second.values()))["filename"] = "SCENE_0001.jpg"
        directory = self.root / "mapping-labels"
        directory.mkdir()
        (directory / "a.json").write_text(json.dumps(payload))
        (directory / "b.json").write_text(json.dumps(second))
        result = convert_khmerst_scene(
            directory, self.images, self.root / "mapping.jsonl", license_name="generated",
            permission_reference="test", reviewer="test", document_prefix="test")
        dataset = PrismPageDataset(result["manifest"])
        self.assertEqual(len(dataset), 1)
        self.assertEqual(len(dataset[0][1]["lines"]), 3)
        self.assertEqual(dataset[0][1]["supervision"], {"lines": "partial"})

    def test_split_assignment_is_disjoint_and_deterministic(self) -> None:
        names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg", "f.jpg"]
        first = assign_scene_splits(names, seed=20261016)
        self.assertEqual(sorted(first.values()), sorted(assign_scene_splits(names, seed=20261016).values()))
        self.assertEqual(set(first), set(names))
        self.assertTrue(set(first.values()) <= {"train", "val", "test"})
        with self.assertRaisesRegex(ValueError, "must sum to 1.0"):
            assign_scene_splits(names, ratios={"train": 0.5, "val": 0.2})

    def test_split_filter_emits_only_the_requested_split(self) -> None:
        result = self._convert(self.root / "val.jsonl", split="val")
        self.assertEqual(result["pages"], 1)
        self.assertEqual(result["split_pages"], {"val": 1})
        self.assertEqual(result["lines"], 1)

    def test_annotations_directory_and_zip_read_identically(self) -> None:
        zip_records, _ = read_khmerst_annotations(self.zip_path)
        dir_records, _ = read_khmerst_annotations(self.annotations)
        self.assertEqual(
            json.dumps([record["lines"] for record in zip_records], ensure_ascii=False),
            json.dumps([record["lines"] for record in dir_records], ensure_ascii=False),
        )

    def test_duplicate_scene_images_convert_only_their_first_filename(self) -> None:
        images = self.root / "duplicate images"
        images.mkdir()
        pixels = np.full((60, 80, 3), 200, dtype=np.uint8)
        pixels[10:20, 5:35] = 0
        for name in ("DUP_A.jpg", "DUP_B.jpg"):
            Image.fromarray(pixels).save(images / name)
        Image.fromarray(np.full((60, 80, 3), 30, dtype=np.uint8)).save(images / "DUP_C.jpg")
        annotations = self.root / "duplicate annotations"
        annotations.mkdir()
        for name in ("DUP_A.jpg", "DUP_B.jpg", "DUP_C.jpg"):
            (annotations / f"{name}.json").write_text(
                json.dumps(
                    {
                        f"{name}1": {
                            "filename": name,
                            "regions": [
                                {
                                    "shape_attributes": {
                                        "name": "polygon",
                                        "all_points_x": [5, 35, 35, 5],
                                        "all_points_y": [10, 10, 20, 20],
                                    },
                                    "region_attributes": {"Name": "អក្សរ"},
                                }
                            ],
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        # The duplicate pair is assigned to different splits, so only content-dedup
        # keeps identical bytes out of both.
        assignment = {"DUP_A.jpg": "train", "DUP_B.jpg": "val", "DUP_C.jpg": "val"}
        train = convert_khmerst_scene(
            annotations,
            images,
            self.root / "dup-train.jsonl",
            license_name="Apache-2.0",
            permission_reference="khmerst-public-release",
            reviewer="scene-qa",
            split="train",
            split_assignment=assignment,
        )
        val = convert_khmerst_scene(
            annotations,
            images,
            self.root / "dup-val.jsonl",
            license_name="Apache-2.0",
            permission_reference="khmerst-public-release",
            reviewer="scene-qa",
            split="val",
            split_assignment=assignment,
        )
        self.assertEqual(train["pages"], 1)
        self.assertEqual(val["pages"], 1)
        self.assertEqual(train["skipped"].get("duplicate_image_bytes", 0), 0)
        self.assertEqual(val["skipped"].get("duplicate_image_bytes", 0), 1)
        train_records = PrismPageDataset(train["manifest"]).records
        val_records = PrismPageDataset(val["manifest"]).records
        self.assertEqual(train_records[0]["_annotation"]["document_id"], "khmerst:DUP_A")
        self.assertEqual(val_records[0]["_annotation"]["document_id"], "khmerst:DUP_C")
        self.assertNotEqual(
            train_records[0]["_image_sha256"], val_records[0]["_image_sha256"]
        )

    def test_exif_rotated_scene_images_use_the_displayed_frame(self) -> None:
        images = self.root / "exif images"
        images.mkdir()
        raw = Image.new("RGB", (20, 30), (255, 255, 255))
        ImageDraw.Draw(raw).rectangle([4, 5, 14, 11], fill=(0, 0, 0))
        exif = Image.Exif()
        exif[0x0112] = 6
        image_path = images / "ROTATED.jpg"
        raw.save(image_path, exif=exif)
        annotations = self.root / "exif annotations"
        annotations.mkdir()
        (annotations / "ROTATED.json").write_text(
            json.dumps(
                {
                    "ROTATED.jpg1": {
                        "filename": "ROTATED.jpg",
                        "regions": [
                            {
                                "shape_attributes": {
                                    "name": "polygon",
                                    "all_points_x": [2, 12, 12, 2],
                                    "all_points_y": [3, 3, 9, 9],
                                },
                                "region_attributes": {"Name": "បន្ទាត់"},
                            }
                        ],
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        result = convert_khmerst_scene(
            annotations,
            images,
            self.root / "exif.jsonl",
            license_name="Apache-2.0",
            permission_reference="khmerst-public-release",
            reviewer="scene-qa",
        )
        dataset = PrismPageDataset(result["manifest"])
        annotation = dataset.records[0]["_annotation"]
        self.assertEqual((annotation["image"]["width"], annotation["image"]["height"]), (30, 20))
        tensor, _ = dataset[0]
        self.assertEqual(tuple(tensor.shape), (3, 20, 30))
        with Image.open(image_path) as opened:
            expected = np.asarray(ImageOps.exif_transpose(opened).convert("L"), dtype=np.float32) / 255.0
        loaded = tensor.mean(dim=0).numpy()
        np.testing.assert_allclose(loaded, expected, atol=1e-6)

    def test_max_page_side_downscales_pixels_without_changing_normalized_targets(self) -> None:
        images = self.root / "capped images"
        images.mkdir()
        pixels = np.full((600, 400, 3), 255, dtype=np.uint8)
        pixels[100:140, 50:300] = 0
        Image.fromarray(pixels).save(images / "BIG.png")
        annotations = self.root / "capped annotations"
        annotations.mkdir()
        (annotations / "BIG.json").write_text(
            json.dumps(
                {
                    "BIG.png1": {
                        "filename": "BIG.png",
                        "regions": [
                            {
                                "shape_attributes": {
                                    "name": "polygon",
                                    "all_points_x": [50, 300, 300, 50],
                                    "all_points_y": [100, 100, 140, 140],
                                },
                                "region_attributes": {"Name": "ជួរបន្ទាត់"},
                            }
                        ],
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        manifest = self.root / "capped.jsonl"
        convert_khmerst_scene(
            annotations,
            images,
            manifest,
            license_name="Apache-2.0",
            permission_reference="khmerst-public-release",
            reviewer="scene-qa",
        )

        native = PrismPageDataset(manifest)
        native_tensor, native_target = native[0]
        self.assertEqual(tuple(native_tensor.shape), (3, 600, 400))
        capped = PrismPageDataset(manifest, max_page_side=200)
        capped_tensor, capped_target = capped[0]
        self.assertEqual(tuple(capped_tensor.shape), (3, 200, 133))
        for field in ("lines", "regions"):
            native_boxes = [item["box"] for item in native_target[field]]
            capped_boxes = [item["box"] for item in capped_target[field]]
            self.assertEqual(len(native_boxes), len(capped_boxes))
            for native_box, capped_box in zip(native_boxes, capped_boxes, strict=True):
                np.testing.assert_allclose(capped_box, native_box, atol=1e-4)
        self.assertEqual(float(capped_tensor.min()), 0.0)
        np.testing.assert_allclose(
            float(capped_tensor.mean()), float(native_tensor.mean()), atol=5e-3
        )
        with self.assertRaisesRegex(ValueError, "max_page_side must be a positive integer"):
            PrismPageDataset(manifest, max_page_side=0)
        with self.assertRaisesRegex(ValueError, "max_page_side must be a positive integer"):
            PrismPageDataset(manifest, max_page_side=True)


class CocoLayoutConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        for name in ("page-a.png", "page-b.png"):
            Image.fromarray(np.full((20, 16, 3), 255, dtype=np.uint8)).save(self.images / name)
        payload = {
            "categories": [
                {"id": 1, "name": "Table"},
                {"id": 2, "name": "Text"},
                {"id": 3, "name": "List-item"},
            ],
            "images": [
                {"id": 0, "file_name": "page-a.png", "width": 16, "height": 20, "doc_name": "report.pdf", "page_no": 4},
                {"id": 1, "file_name": "page-b.png", "width": 16, "height": 20, "doc_name": "report.pdf", "page_no": 5},
            ],
            "annotations": [
                {
                    "id": 10,
                    "image_id": 0,
                    "category_id": 1,
                    "segmentation": [[2, 2, 2, 10, 12, 10, 12, 2]],
                },
                {
                    "id": 11,
                    "image_id": 0,
                    "category_id": 2,
                    "segmentation": [[[3, 12], [13, 12], [13, 17], [3, 17]]],
                },
                {"id": 12, "image_id": 0, "category_id": 3, "segmentation": {"counts": [1, 2], "size": [20, 16]}},
                {"id": 13, "image_id": 0, "category_id": 2, "segmentation": [[5, 5, 5, 5, 5, 5]]},
                {
                    "id": 14,
                    "image_id": 1,
                    "category_id": 3,
                    "segmentation": [[1, 1, 1, 8, 9, 8, 9, 1]],
                },
            ],
        }
        self.coco = self.root / "coco.json"
        self.coco.write_text(json.dumps(payload), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _convert(self, output: Path, **overrides):
        options = {
            "license_name": "CDLA-Permissive-1.0",
            "permission_reference": "doclaynet-public-release",
            "reviewer": "layout-qa",
            "document_prefix": "doclaynet",
        }
        options.update(overrides)
        return convert_coco_layout(self.coco, self.images, output, **options)

    def test_layout_conversion_maps_categories_and_normalizes_winding(self) -> None:
        result = self._convert(self.root / "layout.jsonl")
        self.assertEqual(result["pages"], 2)
        self.assertEqual(result["region_kinds"], {"LIST": 1, "REGION": 1, "TABLE": 1})
        self.assertEqual(result["skipped"]["unsupported_segmentation"], 1)
        self.assertEqual(result["skipped"]["degenerate_polygon"], 1)

        dataset = PrismPageDataset(result["manifest"])
        first = dataset.records[0]["_annotation"]
        self.assertEqual(first["provenance"]["source_class"], "real")
        self.assertEqual(first["provenance"]["license"], "CDLA-Permissive-1.0")
        self.assertEqual(first["supervision"], {"lines": "unavailable"})
        self.assertEqual(dataset[0][1]["supervision"], {"lines": "unavailable"})
        self.assertEqual(first["provenance"]["template_family_id"], "report.pdf")
        self.assertEqual(first["document_id"], "doclaynet:report.pdf:4")
        by_kind = {region["kind"]: region for region in first["regions"]}
        self.assertEqual(set(by_kind), {"TABLE", "REGION"})
        table_polygon = by_kind["TABLE"]["polygon_px"]
        self.assertEqual(
            sorted(tuple(point) for point in table_polygon),
            [(2.0, 2.0), (2.0, 10.0), (12.0, 2.0), (12.0, 10.0)],
        )
        self.assertGreater(_signed_area(table_polygon), 0.0)
        self.assertEqual(first["lines"], [])
        self.assertEqual(first["reading_order"], [])

    def test_limit_and_dimension_validation(self) -> None:
        limited = self._convert(self.root / "limited.jsonl", limit=1)
        self.assertEqual(limited["pages"], 1)
        payload = json.loads(self.coco.read_text(encoding="utf-8"))
        payload["images"][0]["width"] = 32
        mismatched = self.root / "mismatched.json"
        mismatched.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "do not match COCO annotation"):
            convert_coco_layout(
                mismatched,
                self.images,
                self.root / "mismatch.jsonl",
                license_name="CDLA-Permissive-1.0",
                permission_reference="doclaynet-public-release",
                reviewer="layout-qa",
                document_prefix="doclaynet",
            )


def _signed_area(points):
    total = 0.0
    for index, point in enumerate(points):
        following = points[(index + 1) % len(points)]
        total += point[0] * following[1] - following[0] * point[1]
    return total / 2.0


if __name__ == "__main__":
    unittest.main()
