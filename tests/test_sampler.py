from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from prismocr.data.dataset import (
    AspectRatioBatchSampler,
    PageBucketSampler,
    PrismPageDataset,
    collate_pages,
)
from prismocr.data.sampler import (
    assign_bucket,
    compute_scaled_dimensions,
    extract_aspect_ratios_from_dataset,
    extract_record_aspect_ratio,
    validate_bucket_boundaries,
)


class AspectRatioBatchSamplerTests(unittest.TestCase):
    def test_assign_bucket_and_boundaries(self) -> None:
        boundaries = (0.85, 1.15)
        # Portrait: ar < 0.85
        self.assertEqual(assign_bucket(0.5, boundaries), 0)
        self.assertEqual(assign_bucket(0.7, boundaries), 0)
        self.assertEqual(assign_bucket(0.8499, boundaries), 0)

        # Square: 0.85 <= ar < 1.15
        self.assertEqual(assign_bucket(0.85, boundaries), 1)
        self.assertEqual(assign_bucket(1.0, boundaries), 1)
        self.assertEqual(assign_bucket(1.1499, boundaries), 1)

        # Landscape: ar >= 1.15
        self.assertEqual(assign_bucket(1.15, boundaries), 2)
        self.assertEqual(assign_bucket(1.4, boundaries), 2)
        self.assertEqual(assign_bucket(2.5, boundaries), 2)

        # Custom boundaries
        custom = (0.5, 1.0, 1.5, 2.0)
        self.assertEqual(assign_bucket(0.3, custom), 0)
        self.assertEqual(assign_bucket(0.5, custom), 1)
        self.assertEqual(assign_bucket(0.8, custom), 1)
        self.assertEqual(assign_bucket(1.0, custom), 2)
        self.assertEqual(assign_bucket(1.2, custom), 2)
        self.assertEqual(assign_bucket(1.5, custom), 3)
        self.assertEqual(assign_bucket(1.8, custom), 3)
        self.assertEqual(assign_bucket(2.0, custom), 4)
        self.assertEqual(assign_bucket(3.0, custom), 4)

    def test_validate_bucket_boundaries(self) -> None:
        self.assertEqual(validate_bucket_boundaries([0.85, 1.15]), (0.85, 1.15))
        self.assertEqual(validate_bucket_boundaries((0.5, 1.0, 1.5)), (0.5, 1.0, 1.5))

        with self.assertRaisesRegex(ValueError, "non-empty"):
            validate_bucket_boundaries([])
        with self.assertRaisesRegex(ValueError, "positive finite"):
            validate_bucket_boundaries([-0.5, 1.0])
        with self.assertRaisesRegex(ValueError, "positive finite"):
            validate_bucket_boundaries([0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "positive finite"):
            validate_bucket_boundaries([float("nan"), 1.0])
        with self.assertRaisesRegex(ValueError, "positive finite"):
            validate_bucket_boundaries([True, 1.0])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_bucket_boundaries([1.15, 0.85])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_bucket_boundaries([0.85, 0.85])

    def test_compute_scaled_dimensions(self) -> None:
        # No max_page_side
        self.assertEqual(compute_scaled_dimensions(1024, 1400, None), (1024, 1400))
        # Within max_page_side
        self.assertEqual(compute_scaled_dimensions(1024, 1024, 1600), (1024, 1024))
        # Scaled down proportionally
        w, h = compute_scaled_dimensions(3500, 4300, 1600)
        self.assertEqual(h, 1600)
        self.assertEqual(w, round(3500 * (1600 / 4300)))
        # Invalid dimensions
        with self.assertRaisesRegex(ValueError, "must be positive"):
            compute_scaled_dimensions(0, 100)

    def test_extract_record_aspect_ratio(self) -> None:
        record = {
            "_annotation": {
                "image": {"width": 1000, "height": 2000, "sha256": "0" * 64}
            }
        }
        self.assertAlmostEqual(extract_record_aspect_ratio(record), 0.5)
        # With max_page_side capping longer side to 1000
        # height becomes 1000, width becomes 500 -> 500 / 1000 = 0.5
        self.assertAlmostEqual(extract_record_aspect_ratio(record, max_page_side=1000), 0.5)

    def test_extract_aspect_ratios_preserves_explicit_max_page_side_when_dataset_attribute_is_none(self) -> None:
        class DummyDataset:
            max_page_side = None
            records = [
                {"annotation": {"image": {"width": 300, "height": 700}}},
            ]

            def __len__(self) -> int:
                return len(self.records)

        dataset = DummyDataset()
        # Without explicit max_page_side, raw aspect ratio 300 / 700 is returned
        ratios_unconstrained = extract_aspect_ratios_from_dataset(dataset)
        self.assertAlmostEqual(ratios_unconstrained[0], 300 / 700)

        # With explicit max_page_side=2, width=round(300*2/700)=1, height=2 -> ratio 0.5 != 3/7
        ratios_constrained = extract_aspect_ratios_from_dataset(dataset, max_page_side=2)
        self.assertAlmostEqual(ratios_constrained[0], 0.5)
        self.assertNotAlmostEqual(ratios_constrained[0], 300 / 700)

        # Also verify via AspectRatioBatchSampler
        sampler = AspectRatioBatchSampler(dataset, batch_size=1, max_page_side=2)
        self.assertAlmostEqual(sampler.aspect_ratios[0], 0.5)

    def test_bucket_assignment_groups_indices(self) -> None:
        # 3 portrait (0.6, 0.7, 0.8), 2 square (0.9, 1.0), 3 landscape (1.2, 1.3, 1.5)
        aspect_ratios = [0.6, 1.0, 1.3, 0.7, 0.9, 1.2, 0.8, 1.5]
        sampler = AspectRatioBatchSampler(
            len(aspect_ratios),
            batch_size=2,
            aspect_ratios=aspect_ratios,
            shuffle=False,
        )
        self.assertEqual(sampler.bucket_counts, {0: 3, 1: 2, 2: 3})
        self.assertEqual(sampler.bucket_indices[0], [0, 3, 6])
        self.assertEqual(sampler.bucket_indices[1], [1, 4])
        self.assertEqual(sampler.bucket_indices[2], [2, 5, 7])

    def test_index_coverage_and_chunking_drop_last_false(self) -> None:
        aspect_ratios = [0.7, 0.7, 0.7, 1.0, 1.0, 1.3, 1.3, 1.3, 1.3]
        batch_size = 2
        sampler = AspectRatioBatchSampler(
            len(aspect_ratios),
            batch_size=batch_size,
            aspect_ratios=aspect_ratios,
            drop_last=False,
            shuffle=True,
        )
        # Bucket 0: 3 items -> 2 batches (sizes 2, 1)
        # Bucket 1: 2 items -> 1 batch (size 2)
        # Bucket 2: 4 items -> 2 batches (sizes 2, 2)
        # Total batches = 5
        self.assertEqual(len(sampler), 5)
        batches = list(sampler)
        self.assertEqual(len(batches), 5)

        # Every index 0..8 must appear exactly once
        flattened = [idx for batch in batches for idx in batch]
        self.assertEqual(sorted(flattened), list(range(len(aspect_ratios))))

        # Intra-batch homogeneity: all samples in a batch must share the same bucket
        for batch in batches:
            buckets_in_batch = {assign_bucket(aspect_ratios[i], (0.85, 1.15)) for i in batch}
            self.assertEqual(len(buckets_in_batch), 1)

    def test_index_coverage_and_chunking_drop_last_true(self) -> None:
        aspect_ratios = [0.7, 0.7, 0.7, 1.0, 1.0, 1.3, 1.3, 1.3, 1.3]
        batch_size = 2
        sampler = AspectRatioBatchSampler(
            len(aspect_ratios),
            batch_size=batch_size,
            aspect_ratios=aspect_ratios,
            drop_last=True,
            shuffle=True,
        )
        # Bucket 0: 3 items -> 1 batch (3 // 2 = 1)
        # Bucket 1: 2 items -> 1 batch (2 // 2 = 1)
        # Bucket 2: 4 items -> 2 batches (4 // 2 = 2)
        # Total batches = 4
        self.assertEqual(len(sampler), 4)
        batches = list(sampler)
        self.assertEqual(len(batches), 4)

        for batch in batches:
            self.assertEqual(len(batch), batch_size)
            buckets_in_batch = {assign_bucket(aspect_ratios[i], (0.85, 1.15)) for i in batch}
            self.assertEqual(len(buckets_in_batch), 1)

    def test_batch_interleaving_and_shuffling(self) -> None:
        # 10 portrait, 10 square, 10 landscape with batch_size=2
        aspect_ratios = [0.7] * 10 + [1.0] * 10 + [1.4] * 10
        g = torch.Generator().manual_seed(123)
        sampler = AspectRatioBatchSampler(
            len(aspect_ratios),
            batch_size=2,
            aspect_ratios=aspect_ratios,
            shuffle=True,
            generator=g,
        )
        batches = list(sampler)
        self.assertEqual(len(batches), 15)

        # Map each batch to its bucket id
        batch_buckets = [assign_bucket(aspect_ratios[batch[0]], (0.85, 1.15)) for batch in batches]
        # Verify that buckets are interleaved (not all portrait first, then all square, then all landscape)
        self.assertNotEqual(batch_buckets, sorted(batch_buckets))

    def test_shuffle_false_preserves_deterministic_order(self) -> None:
        aspect_ratios = [1.4, 0.7, 1.0, 0.7, 1.4, 1.0]
        sampler = AspectRatioBatchSampler(
            len(aspect_ratios),
            batch_size=2,
            aspect_ratios=aspect_ratios,
            shuffle=False,
        )
        batches = list(sampler)
        # Bucket 0 (portrait): indices [1, 3] -> [[1, 3]]
        # Bucket 1 (square): indices [2, 5] -> [[2, 5]]
        # Bucket 2 (landscape): indices [0, 4] -> [[0, 4]]
        self.assertEqual(batches, [[1, 3], [2, 5], [0, 4]])

    def test_deterministic_rng_reproduction(self) -> None:
        aspect_ratios = [0.7, 1.0, 1.4, 0.7, 1.0, 1.4, 0.7, 1.0, 1.4, 1.0]
        g1 = torch.Generator().manual_seed(42)
        sampler1 = AspectRatioBatchSampler(
            len(aspect_ratios),
            batch_size=2,
            aspect_ratios=aspect_ratios,
            shuffle=True,
            generator=g1,
        )
        batches1_epoch0 = list(sampler1)
        batches1_epoch1 = list(sampler1)

        # Same seed must produce identical batches across epochs
        g2 = torch.Generator().manual_seed(42)
        sampler2 = AspectRatioBatchSampler(
            len(aspect_ratios),
            batch_size=2,
            aspect_ratios=aspect_ratios,
            shuffle=True,
            generator=g2,
        )
        batches2_epoch0 = list(sampler2)
        batches2_epoch1 = list(sampler2)

        self.assertEqual(batches1_epoch0, batches2_epoch0)
        self.assertEqual(batches1_epoch1, batches2_epoch1)
        # Epoch 0 and Epoch 1 must differ due to shuffling
        self.assertNotEqual(batches1_epoch0, batches1_epoch1)

        # Different seed produces different order
        g3 = torch.Generator().manual_seed(99)
        sampler3 = AspectRatioBatchSampler(
            len(aspect_ratios),
            batch_size=2,
            aspect_ratios=aspect_ratios,
            shuffle=True,
            generator=g3,
        )
        self.assertNotEqual(batches1_epoch0, list(sampler3))

    def test_generator_state_save_and_restore_resume(self) -> None:
        aspect_ratios = [0.7, 1.0, 1.4, 0.7, 1.0, 1.4, 0.7, 1.0, 1.4, 1.0]
        # Uninterrupted 2-epoch run
        g_uninterrupted = torch.Generator().manual_seed(77)
        sampler_uninterrupted = AspectRatioBatchSampler(
            len(aspect_ratios),
            batch_size=2,
            aspect_ratios=aspect_ratios,
            shuffle=True,
            generator=g_uninterrupted,
        )
        _ = list(sampler_uninterrupted)  # epoch 0
        state_after_epoch0 = g_uninterrupted.get_state().clone()
        uninterrupted_epoch1 = list(sampler_uninterrupted)  # epoch 1

        # Resumed run starting at epoch 1 with restored generator state
        g_resumed = torch.Generator()
        g_resumed.set_state(state_after_epoch0)
        sampler_resumed = AspectRatioBatchSampler(
            len(aspect_ratios),
            batch_size=2,
            aspect_ratios=aspect_ratios,
            shuffle=True,
            generator=g_resumed,
        )
        resumed_epoch1 = list(sampler_resumed)

        self.assertEqual(resumed_epoch1, uninterrupted_epoch1)

    def test_edge_cases(self) -> None:
        # Empty dataset
        sampler_empty = AspectRatioBatchSampler(0, batch_size=2)
        self.assertEqual(len(sampler_empty), 0)
        self.assertEqual(list(sampler_empty), [])

        # Single item dataset, batch_size=2
        sampler_single = AspectRatioBatchSampler(1, batch_size=2, drop_last=False)
        self.assertEqual(len(sampler_single), 1)
        self.assertEqual(list(sampler_single), [[0]])

        sampler_single_drop = AspectRatioBatchSampler(1, batch_size=2, drop_last=True)
        self.assertEqual(len(sampler_single_drop), 0)
        self.assertEqual(list(sampler_single_drop), [])

        # Only one bucket populated
        sampler_one_bucket = AspectRatioBatchSampler(
            5, batch_size=2, aspect_ratios=[1.0] * 5, drop_last=False
        )
        self.assertEqual(len(sampler_one_bucket), 3)
        flattened = [idx for b in sampler_one_bucket for idx in b]
        self.assertEqual(sorted(flattened), list(range(5)))

        # Invalid batch_size
        with self.assertRaisesRegex(ValueError, "batch_size"):
            AspectRatioBatchSampler(5, batch_size=0)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            AspectRatioBatchSampler(5, batch_size=-1)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            AspectRatioBatchSampler(5, batch_size=True)

        # Invalid aspect ratio
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            AspectRatioBatchSampler(2, batch_size=1, aspect_ratios=[-1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            AspectRatioBatchSampler(2, batch_size=1, aspect_ratios=[0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            AspectRatioBatchSampler(2, batch_size=1, aspect_ratios=[float("inf"), 1.0])
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            AspectRatioBatchSampler(2, batch_size=1, aspect_ratios=[True, 1.0])
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            AspectRatioBatchSampler(2, batch_size=1, aspect_ratios=[False, 1.0])

        # Mismatched length
        with self.assertRaisesRegex(ValueError, "does not match"):
            AspectRatioBatchSampler(3, batch_size=1, aspect_ratios=[1.0, 1.0])

    def test_page_bucket_sampler_alias(self) -> None:
        self.assertIs(PageBucketSampler, AspectRatioBatchSampler)

    def test_dataset_and_subset_integration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            shapes = [(100, 200), (200, 200), (300, 150)]  # portrait, square, landscape
            rows = []
            for idx, (w, h) in enumerate(shapes):
                img_path = root / f"page_{idx}.png"
                Image.fromarray(np.full((h, w, 3), 128, dtype=np.uint8)).save(img_path)
                sha = hashlib.sha256(img_path.read_bytes()).hexdigest()
                row = {
                    "image_path": img_path.name,
                    "annotation": {
                        "schema_version": "4.0",
                        "document_id": f"doc_{idx}",
                        "page_id": f"doc_{idx}:p0",
                        "image": {"width": w, "height": h, "sha256": sha},
                        "provenance": {
                            "source_class": "synthetic",
                            "license": "generated",
                            "permission_reference": "test",
                            "reviewer": "test",
                        },
                        "lines": [],
                        "regions": [],
                        "cells": [],
                        "relations": [],
                        "reading_order": [],
                    },
                }
                rows.append(row)

            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

            dataset = PrismPageDataset(manifest_path)
            extracted = extract_aspect_ratios_from_dataset(dataset)
            self.assertEqual(len(extracted), 3)
            self.assertAlmostEqual(extracted[0], 100 / 200)
            self.assertAlmostEqual(extracted[1], 200 / 200)
            self.assertAlmostEqual(extracted[2], 300 / 150)

            # Test AspectRatioBatchSampler with PrismPageDataset directly
            sampler = AspectRatioBatchSampler(dataset, batch_size=1, shuffle=False)
            self.assertEqual(sampler.bucket_counts, {0: 1, 1: 1, 2: 1})
            batches = list(sampler)
            self.assertEqual(batches, [[0], [1], [2]])

            # Test Subset
            subset = Subset(dataset, [2, 0])
            subset_ratios = extract_aspect_ratios_from_dataset(subset)
            self.assertEqual(len(subset_ratios), 2)
            self.assertAlmostEqual(subset_ratios[0], 300 / 150)
            self.assertAlmostEqual(subset_ratios[1], 100 / 200)

            subset_sampler = AspectRatioBatchSampler(subset, batch_size=1, shuffle=False)
            self.assertEqual(subset_sampler.bucket_counts, {0: 1, 1: 0, 2: 1})
            self.assertEqual(list(subset_sampler), [[1], [0]])

            # Test DataLoader integration with collate_pages
            loader = DataLoader(
                dataset,
                batch_sampler=sampler,
                collate_fn=collate_pages,
            )
            collated = list(loader)
            self.assertEqual(len(collated), 3)
            # First batch is portrait: shape [1, 3, padded_h, padded_w]
            p_img, _ = collated[0]
            self.assertEqual(p_img.shape[0], 1)
            self.assertEqual(p_img.shape[1], 3)

    def test_train_settings_and_config_modes(self) -> None:
        from unittest.mock import patch
        from prismocr.train import _resolved_settings, train
        from prismocr.data.vocabulary import Vocabulary
        from tests.test_train_lifecycle import TinyTrainingLoss, TinyTrainingModel

        vocab = Vocabulary.from_texts(["test"])
        # Default bucket_batching is True
        settings_default = _resolved_settings({}, {}, vocab, batch_size=2, num_workers=0)
        self.assertTrue(settings_default["training"]["bucket_batching"])
        self.assertEqual(settings_default["training"]["bucket_boundaries"], [0.85, 1.15])

        # Explicit bucket_batching = False
        settings_disabled = _resolved_settings({}, {"bucket_batching": False}, vocab, batch_size=2, num_workers=0)
        self.assertFalse(settings_disabled["training"]["bucket_batching"])

        # Explicit custom bucket_boundaries
        settings_custom = _resolved_settings(
            {}, {"bucket_boundaries": [0.6, 1.0, 1.4]}, vocab, batch_size=2, num_workers=0
        )
        self.assertEqual(settings_custom["training"]["bucket_boundaries"], [0.6, 1.0, 1.4])

        # Test end-to-end training runs with both modes
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            img_path = root / "p.png"
            Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)).save(img_path)
            sha = hashlib.sha256(img_path.read_bytes()).hexdigest()
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps({
                    "image_path": img_path.name,
                    "annotation": {
                        "schema_version": "4.0",
                        "document_id": "doc",
                        "page_id": "doc:p0",
                        "image": {"width": 16, "height": 16, "sha256": sha},
                        "provenance": {
                            "source_class": "synthetic",
                            "license": "generated",
                            "permission_reference": "test",
                            "reviewer": "test",
                        },
                        "lines": [{"id": "l0", "polygon_px": [[1, 1], [10, 1], [10, 5], [1, 5]], "text": "test", "role": "PARA_LINE"}],
                        "regions": [],
                        "cells": [],
                        "relations": [],
                        "reading_order": ["l0"],
                    },
                }) + "\n",
                encoding="utf-8",
            )

            for bucket_mode in (True, False):
                config_path = root / f"config_{bucket_mode}.toml"
                config_path.write_text(
                    f"""[model]
dim = 10
text_proposals = 2
region_queries = 1
strip_height = 5
vocabulary_size = 16
recognizer = "transformer"
text_threshold = 0.55
region_threshold = 0.5

[training]
seed = 42
epochs = 1
batch_size = 1
num_workers = 0
bucket_batching = {str(bucket_mode).lower()}
""",
                    encoding="utf-8",
                )
                output_dir = root / f"output_{bucket_mode}"
                with patch("prismocr.train.PrismOCRModel", TinyTrainingModel), patch("prismocr.train.PrismOCRLoss", TinyTrainingLoss):
                    result = train(manifest, output_dir, config_path=config_path)
                self.assertEqual(result["steps"], 1)
                checkpoint = torch.load(output_dir / "training-state.pt", map_location="cpu", weights_only=True)
                self.assertEqual(checkpoint["compatibility"]["settings"]["training"]["bucket_batching"], bucket_mode)

