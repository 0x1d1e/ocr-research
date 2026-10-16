from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import ImageFont

from prismocr.data.dataset import PrismPageDataset
from prismocr.data.schema import validate_annotation
from prismocr.data.synthetic import generate_synthetic_dataset


class SyntheticDatasetTests(unittest.TestCase):
    def test_seeded_generation_is_deterministic_and_loads_as_a_dataset(self) -> None:
        pages = [["Synthetic first line", "Café second line"], ["Another page"]]
        font = ImageFont.load_default()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = generate_synthetic_dataset(
                root / "first", pages, font=font, seed=37, image_size=(240, 120)
            )
            second = generate_synthetic_dataset(
                root / "second", pages, font=font, seed=37, image_size=(240, 120)
            )

            self.assertEqual(first.read_bytes(), second.read_bytes())
            for filename in ("page-0000.png", "page-0000.json", "page-0001.png", "page-0001.json"):
                self.assertEqual((first.parent / filename).read_bytes(), (second.parent / filename).read_bytes())

            dataset = PrismPageDataset(first)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.texts(), ["Synthetic first line", "Café second line", "Another page"])
            image, target = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 120, 240))
            self.assertEqual(target["reading_order"], [line["id"] for line in target["lines"]])

            manifest_row = json.loads(first.read_text(encoding="utf-8").splitlines()[0])
            annotation = json.loads((first.parent / manifest_row["annotation_path"]).read_text(encoding="utf-8"))
            validate_annotation(annotation)
            self.assertEqual(annotation["annotation"]["source"], "synthetic")
            self.assertEqual(annotation["supervision"], {"lines": "complete"})
            self.assertEqual(target["supervision"], {"lines": "complete"})
            self.assertEqual(annotation["reading_order"], [line["id"] for line in annotation["lines"]])
            self.assertEqual(
                annotation["image"]["sha256"],
                hashlib.sha256((first.parent / manifest_row["image_path"]).read_bytes()).hexdigest(),
            )
            for line in annotation["lines"]:
                polygon = line["polygon_px"]
                self.assertEqual(
                    polygon,
                    [
                        [polygon[0][0], polygon[0][1]],
                        [polygon[2][0], polygon[0][1]],
                        [polygon[2][0], polygon[2][1]],
                        [polygon[0][0], polygon[2][1]],
                    ],
                )
                self.assertEqual(annotation["document_id"], "synthetic-seed-37")

    def test_malformed_text_fails_with_page_and_line_context(self) -> None:
        font = ImageFont.load_default()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "page 0 line 0 must be a string"):
                generate_synthetic_dataset(
                    Path(tmp) / "bad-type", [[None]], font=font  # type: ignore[list-item]
                )
            with self.assertRaisesRegex(ValueError, "page 0 line 0.*one line"):
                generate_synthetic_dataset(
                    Path(tmp) / "multiline", [["first\nsecond"]], font=font
                )

    def test_horizontal_and_vertical_overflow_fail_clearly(self) -> None:
        font = ImageFont.load_default()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "page 0 line 0 overflows image size"):
                generate_synthetic_dataset(
                    Path(tmp) / "wide",
                    [["This synthetic line is too long for the page"]],
                    font=font,
                    image_size=(64, 100),
                )
            with self.assertRaisesRegex(ValueError, "page 0 line 1 overflows image size"):
                generate_synthetic_dataset(
                    Path(tmp) / "tall",
                    [["First", "Second"]],
                    font=font,
                    image_size=(100, 30),
                )


if __name__ == "__main__":
    unittest.main()
