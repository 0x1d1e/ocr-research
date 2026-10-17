from __future__ import annotations

import hashlib
import io
import json
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch import nn

from prismocr.data.dataset import PrismPageDataset, dataset_content_sha256
from prismocr.train import SELECTION_POLICY, _atomic_checkpoint_save, _is_better, main, train


class TinyTrainingModel(nn.Module):
    """Small stochastic model for exercising lifecycle persistence without OCR cost."""

    def __init__(self, **settings: object) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(()))
        self.dim = int(settings["dim"])
        self.text_proposals = int(settings["text_proposals"])
        self.region_queries = int(settings["region_queries"])
        self.vocab_size = int(settings["vocab_size"])
        self.strip_buckets = tuple(int(value) for value in settings["strip_buckets"])
        self.max_table_rows = int(settings["max_table_rows"])
        self.max_table_cols = int(settings["max_table_cols"])
        self.max_cell_span = int(settings["max_cell_span"])
        self.recognizer = SimpleNamespace(
            strip_height=int(settings["strip_height"]),
            variant=str(settings["recognizer"]),
        )
        self.text_threshold = float(settings["text_threshold"])
        self.region_threshold = float(settings["region_threshold"])
        self.match_iou_threshold = 0.5

    def forward(self, images: torch.Tensor, targets: list[dict[str, object]], *, teacher_force_ratio: float) -> dict[str, torch.Tensor]:
        noise = torch.rand(()) + random.random() + float(np.random.random()) + teacher_force_ratio * 0.0
        return {"total": self.weight.square() + noise * self.weight}

    def tensor_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self.state_dict().values())


class TinyTrainingLoss:
    def __init__(
        self,
        vocabulary: object,
        *,
        strict_ctc_length: bool,
        loss_weights: dict[str, float] | None = None,
    ) -> None:
        self.strict_ctc_length = strict_ctc_length
        self.loss_weights = {"total": 1.0, **(loss_weights or {})}

    def __call__(self, outputs: dict[str, torch.Tensor], targets: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        return {"total": outputs["total"]}


class CheckpointSelectionTests(unittest.TestCase):
    def _metrics(self, *, recall: float, cer: float, failures: int = 0) -> dict[str, float]:
        return {
            "line_detection_recall_at_iou": recall,
            "cer_on_matched_detections_with_misses_as_empty": cer,
            "inference_failure_count": failures,
        }

    def test_selection_prefers_recall_then_cer_for_working_epochs(self) -> None:
        self.assertTrue(_is_better(self._metrics(recall=0.9, cer=0.2), self._metrics(recall=0.8, cer=0.1)))
        self.assertTrue(_is_better(self._metrics(recall=0.8, cer=0.1), self._metrics(recall=0.8, cer=0.2)))
        self.assertFalse(_is_better(self._metrics(recall=0.8, cer=0.2), self._metrics(recall=0.8, cer=0.2)))

    def test_degenerate_epoch_cannot_win_on_empty_recall_and_cer(self) -> None:
        # A validation manifest with no ground-truth lines reports recall 1.0 and
        # CER 0.0 from zero denominators, so a page-failing epoch would otherwise
        # displace a working one.
        degenerate = self._metrics(recall=1.0, cer=0.0, failures=4)
        working = self._metrics(recall=0.75, cer=0.3, failures=0)
        self.assertFalse(_is_better(degenerate, working))
        self.assertTrue(_is_better(working, degenerate))
        self.assertFalse(_is_better(self._metrics(recall=1.0, cer=0.0, failures=4), self._metrics(recall=1.0, cer=0.0, failures=2)))
        self.assertTrue(_is_better(self._metrics(recall=1.0, cer=0.0, failures=2), degenerate))

    def test_selection_policy_documents_failure_ordering(self) -> None:
        self.assertIn("inference_failure_count", SELECTION_POLICY)

    def test_threshold_free_proposal_metric_selects_best_checkpoint_when_peaks_below_threshold(self) -> None:
        # When detector peaks are below 0.55 operating threshold, line recall is 0.0 and CER is 1.0 (empty hyp).
        # Checkpoint selection should advance based on mean proposal IoU to ground truth.
        epoch1 = {**self._metrics(recall=0.0, cer=1.0), "mean_proposal_iou": 0.12}
        epoch2 = {**self._metrics(recall=0.0, cer=1.0), "mean_proposal_iou": 0.45}
        self.assertTrue(_is_better(epoch2, epoch1))
        self.assertFalse(_is_better(epoch1, epoch2))
        # Exact tie keeps earliest epoch
        self.assertFalse(_is_better(epoch2, epoch2))


class TrainingLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = self.root / "training.toml"
        self._write_config(self.config)
        self.patches = [
            patch("prismocr.train.PrismOCRModel", TinyTrainingModel),
            patch("prismocr.train.PrismOCRLoss", TinyTrainingLoss),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_config(path: Path, *, learning_rate: float = 0.01, lr_warmup_steps: int = 0) -> None:
        path.write_text(
            """[model]
            dim = 10
            text_proposals = 2
            region_queries = 1
            strip_height = 5
            vocabulary_size = 16
            recognizer = "transformer"
            text_threshold = 0.55
            region_threshold = 0.5

            [training]
            seed = 17
            teacher_forcing_start = 0.5
            teacher_forcing_end = 0.5
            strict_ctc_length = true
            epochs = 1
            batch_size = 1
            num_workers = 0
            learning_rate = LEARNING_RATE
            lr_warmup_steps = LR_WARMUP_STEPS
            weight_decay = 0.0
            max_grad_norm = 5.0
            """.replace("LEARNING_RATE", repr(learning_rate)).replace("LR_WARMUP_STEPS", repr(lr_warmup_steps)),
            encoding="utf-8",
        )

    def _manifest(self, name: str, texts: tuple[str, ...] = ("a", "b")) -> Path:
        directory = self.root / name
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, text in enumerate(texts):
            image_name = f"page-{index}.png"
            Image.fromarray(np.full((16, 16, 3), (index * 40 + sum(name.encode())) % 256, dtype=np.uint8)).save(directory / image_name)
            image_sha256 = hashlib.sha256((directory / image_name).read_bytes()).hexdigest()
            annotation = {
                "schema_version": "4.0",
                "document_id": f"{name}-document-{index}",
                "page_id": f"{name}-document-{index}:page-0",
                "image": {"width": 16, "height": 16, "sha256": image_sha256},
                "provenance": {
                    "source_class": "synthetic",
                    "license": "generated",
                    "permission_reference": "not_applicable",
                    "reviewer": "unit_test",
                },
                "lines": [{
                    "id": f"line-{index}",
                    "polygon_px": [[1, 1], [12, 1], [12, 5], [1, 5]],
                    "text": text,
                    "role": "TITLE",
                }],
                "regions": [],
                "cells": [],
                "relations": [],
                "reading_order": [f"line-{index}"],
            }
            rows.append({"image_path": image_name, "annotation": annotation})
        manifest = directory / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return manifest

    def _train(self, manifest: Path, output: Path, **kwargs: object) -> dict[str, object]:
        return train(manifest, output, config_path=self.config, **kwargs)

    def test_fresh_run_saves_atomic_epoch_state_and_labels_unvalidated(self) -> None:
        manifest = self._manifest("fresh-data")
        output = self.root / "fresh-output"

        result = self._train(manifest, output, epochs=1)

        self.assertEqual(result["validation_status"], "unvalidated")
        self.assertEqual(result["status"], "training_completed_without_validation_claim")
        self.assertEqual(result["steps"], 2)
        checkpoint = torch.load(output / "training-state.pt", map_location="cpu", weights_only=True)
        artifact = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertEqual(checkpoint["step"], 2)
        self.assertEqual(checkpoint["log_offset"], (output / "training-log.jsonl").stat().st_size)
        self.assertIn("optimizer_state", checkpoint)
        artifact_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(
            artifact_manifest["training_provenance"]["train_dataset_sha256"],
            dataset_content_sha256(PrismPageDataset(manifest)),
        )
        self.assertIsNone(artifact_manifest["training_provenance"]["validation_dataset_sha256"])
        self.assertEqual(
            set(checkpoint["rng_state"]),
            {"python", "numpy", "torch", "cuda", "data_loader"},
        )
        torch.testing.assert_close(artifact["model_state"]["weight"], checkpoint["model_state"]["weight"])

    def test_resume_restores_rng_optimizer_and_loader_order(self) -> None:
        manifest = self._manifest("resume-data")
        resumed_output = self.root / "resumed-output"
        uninterrupted_output = self.root / "uninterrupted-output"

        calls = 0
        original_forward = TinyTrainingModel.forward

        def fail_at_second_epoch(
            model: TinyTrainingModel,
            images: torch.Tensor,
            targets: list[dict[str, object]],
            *,
            teacher_force_ratio: float,
        ) -> dict[str, torch.Tensor]:
            nonlocal calls
            calls += 1
            if calls == 4:
                raise RuntimeError("synthetic interruption")
            return original_forward(
                model, images, targets, teacher_force_ratio=teacher_force_ratio
            )

        with patch.object(TinyTrainingModel, "forward", fail_at_second_epoch):
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                self._train(manifest, resumed_output, epochs=2)

        interrupted = torch.load(
            resumed_output / "training-state.pt", map_location="cpu", weights_only=True
        )
        self.assertEqual(interrupted["epoch"], 1)
        self.assertEqual(len((resumed_output / "training-log.jsonl").read_text().splitlines()), 3)
        resumed_page_ids: list[str] = []
        original_forward = TinyTrainingModel.forward

        def record_resumed_order(
            model: TinyTrainingModel,
            images: torch.Tensor,
            targets: list[dict[str, object]],
            *,
            teacher_force_ratio: float,
        ) -> dict[str, torch.Tensor]:
            resumed_page_ids.extend(str(target["page_id"]) for target in targets)
            return original_forward(
                model, images, targets, teacher_force_ratio=teacher_force_ratio
            )

        with patch.object(TinyTrainingModel, "forward", record_resumed_order):
            self._train(manifest, resumed_output, epochs=2, resume_from=resumed_output / "training-state.pt")

        uninterrupted_page_ids: list[str] = []

        def record_uninterrupted_order(
            model: TinyTrainingModel,
            images: torch.Tensor,
            targets: list[dict[str, object]],
            *,
            teacher_force_ratio: float,
        ) -> dict[str, torch.Tensor]:
            uninterrupted_page_ids.extend(str(target["page_id"]) for target in targets)
            return original_forward(
                model, images, targets, teacher_force_ratio=teacher_force_ratio
            )

        with patch.object(TinyTrainingModel, "forward", record_uninterrupted_order):
            self._train(manifest, uninterrupted_output, epochs=2)
        self.assertEqual(len(resumed_page_ids), 2)
        self.assertEqual(resumed_page_ids, uninterrupted_page_ids[-2:])

        resumed = torch.load(resumed_output / "training-state.pt", map_location="cpu", weights_only=True)
        uninterrupted = torch.load(uninterrupted_output / "training-state.pt", map_location="cpu", weights_only=True)
        self.assertEqual(resumed["epoch"], 2)
        self.assertEqual(resumed["step"], uninterrupted["step"])
        torch.testing.assert_close(resumed["model_state"]["weight"], uninterrupted["model_state"]["weight"])
        torch.testing.assert_close(
            resumed["optimizer_state"]["state"][0]["exp_avg"],
            uninterrupted["optimizer_state"]["state"][0]["exp_avg"],
        )
        torch.testing.assert_close(resumed["rng_state"]["torch"], uninterrupted["rng_state"]["torch"])
        torch.testing.assert_close(
            resumed["rng_state"]["data_loader"], uninterrupted["rng_state"]["data_loader"]
        )
        self.assertEqual(resumed["rng_state"]["python"], uninterrupted["rng_state"]["python"])
        self.assertEqual(resumed["rng_state"]["numpy"], uninterrupted["rng_state"]["numpy"])
        self.assertEqual(len((resumed_output / "training-log.jsonl").read_text().splitlines()), 4)

    def test_completed_resume_from_backup_materializes_canonical_checkpoint(self) -> None:
        manifest = self._manifest("backup-data", ("a",))
        output = self.root / "backup-output"
        self._train(manifest, output, epochs=1)
        canonical = output / "training-state.pt"
        backup = output / "backup-state.pt"
        canonical.rename(backup)

        result = self._train(
            manifest,
            output,
            epochs=1,
            resume_from=backup,
        )

        self.assertEqual(result["steps"], 1)
        self.assertEqual(Path(result["checkpoint_path"]), canonical.resolve())
        self.assertTrue(canonical.is_file())
        resumed = torch.load(canonical, map_location="cpu", weights_only=True)
        original = torch.load(backup, map_location="cpu", weights_only=True)
        self.assertEqual(resumed["epoch"], original["epoch"])
        self.assertEqual(resumed["step"], original["step"])
        torch.testing.assert_close(resumed["model_state"]["weight"], original["model_state"]["weight"])

    def test_failed_backup_promotion_preserves_canonical_checkpoint_and_log(self) -> None:
        manifest = self._manifest("promotion-data", ("a",))
        output = self.root / "promotion-output"
        canonical = output / "training-state.pt"
        backup = output / "backup-state.pt"
        save_calls = 0

        def save_with_epoch_one_backup(state: dict[str, object], path: Path) -> None:
            nonlocal save_calls
            _atomic_checkpoint_save(state, path)
            save_calls += 1
            if save_calls == 1:
                backup.write_bytes(path.read_bytes())

        with patch("prismocr.train._atomic_checkpoint_save", side_effect=save_with_epoch_one_backup):
            self._train(manifest, output, epochs=2)
        canonical_before = canonical.read_bytes()
        log_path = output / "training-log.jsonl"
        log_before = log_path.read_bytes()

        with patch(
            "prismocr.train._atomic_checkpoint_save",
            side_effect=OSError("injected checkpoint promotion failure"),
        ):
            with self.assertRaisesRegex(OSError, "injected checkpoint promotion failure"):
                self._train(manifest, output, epochs=2, resume_from=backup)

        self.assertEqual(canonical.read_bytes(), canonical_before)
        self.assertEqual(log_path.read_bytes(), log_before)
        result = self._train(manifest, output, epochs=2, resume_from=canonical)
        self.assertEqual(result["steps"], 2)

    def test_non_finite_loss_aborts_before_optimizer_or_checkpoint_update(self) -> None:
        manifest = self._manifest("nonfinite-loss-data", ("a",))
        output = self.root / "nonfinite-loss-output"

        def return_nan(
            model: TinyTrainingModel,
            images: torch.Tensor,
            targets: list[dict[str, object]],
            *,
            teacher_force_ratio: float,
        ) -> dict[str, torch.Tensor]:
            return {"total": model.weight * torch.tensor(float("nan"))}

        with patch.object(TinyTrainingModel, "forward", return_nan):
            with self.assertRaisesRegex(FloatingPointError, "non-finite training loss"):
                self._train(manifest, output, epochs=1)
        self.assertFalse((output / "training-state.pt").exists())
        self.assertFalse((output / "model.pt").exists())

    def test_non_finite_gradient_aborts_before_optimizer_or_checkpoint_update(self) -> None:
        manifest = self._manifest("nonfinite-gradient-data", ("a",))
        output = self.root / "nonfinite-gradient-output"

        def return_finite_loss_with_bad_gradient(
            model: TinyTrainingModel,
            images: torch.Tensor,
            targets: list[dict[str, object]],
            *,
            teacher_force_ratio: float,
        ) -> dict[str, torch.Tensor]:
            model.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
            return {"total": model.weight.square()}

        with patch.object(TinyTrainingModel, "forward", return_finite_loss_with_bad_gradient):
            with self.assertRaisesRegex(FloatingPointError, "non-finite gradients before optimizer step"):
                self._train(manifest, output, epochs=1)
        self.assertFalse((output / "training-state.pt").exists())
        self.assertFalse((output / "model.pt").exists())

    def test_amp_gradient_overflow_skips_update_and_lowers_scaler(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required to exercise GradScaler overflow recovery")
        self.config.write_text(
            self.config.read_text(encoding="utf-8").replace(
                "max_grad_norm = 5.0",
                'max_grad_norm = 5.0\namp = true\namp_dtype = "float16"',
            ),
            encoding="utf-8",
        )
        manifest = self._manifest("amp-overflow-data", ("a",))
        output = self.root / "amp-overflow-output"

        def return_finite_loss_with_overflowed_gradient(
            model: TinyTrainingModel,
            images: torch.Tensor,
            targets: list[dict[str, object]],
            *,
            teacher_force_ratio: float,
        ) -> dict[str, torch.Tensor]:
            model.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
            return {"total": model.weight.square()}

        with patch.object(TinyTrainingModel, "forward", return_finite_loss_with_overflowed_gradient):
            result = self._train(manifest, output, epochs=1, device="cuda")

        rows = [json.loads(line) for line in (output / "training-log.jsonl").read_text().splitlines()]
        checkpoint = torch.load(output / "training-state.pt", map_location="cpu", weights_only=True)
        self.assertEqual(result["steps"], 0)
        self.assertFalse(rows[0]["optimizer_updated"])
        self.assertTrue(rows[0]["scaler_overflow_skipped"])
        self.assertEqual(checkpoint["scaler_state"]["scale"], 32768.0)

    def test_gradient_accumulation_updates_at_full_and_partial_boundaries(self) -> None:
        self.config.write_text(
            self.config.read_text(encoding="utf-8").replace(
                "max_grad_norm = 5.0", "max_grad_norm = 5.0\ngradient_accumulation_steps = 2"
            ),
            encoding="utf-8",
        )
        manifest = self._manifest("accumulation-data", ("a", "b", "c"))
        output = self.root / "accumulation-output"

        result = self._train(manifest, output, epochs=1)

        rows = [json.loads(line) for line in (output / "training-log.jsonl").read_text().splitlines()]
        self.assertEqual(result["steps"], 2)
        self.assertEqual([row["optimizer_updated"] for row in rows], [False, True, True])
        self.assertEqual([row["step"] for row in rows], [0, 1, 2])

    def test_resume_rejects_incompatible_vocabulary_config_and_cardinality(self) -> None:
        original_manifest = self._manifest("compatibility-data")
        checkpoint_dir = self.root / "compatibility-output"
        self._train(original_manifest, checkpoint_dir, epochs=2)
        checkpoint_path = checkpoint_dir / "training-state.pt"

        cases = (
            ("vocabulary", self._manifest("different-vocab", ("c", "d")), self.config),
            ("cardinality", self._manifest("different-count", ("a",)), self.config),
            ("manifest", self._manifest("changed-manifest", ("b", "a")), self.config),
        )
        changed_config = self.root / "changed.toml"
        self._write_config(changed_config, learning_rate=0.02)
        for label, other_manifest, config in cases:
            with self.subTest(incompatibility=label):
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    train(
                        other_manifest,
                        checkpoint_dir,
                        config_path=config,
                        epochs=2,
                        resume_from=checkpoint_path,
                    )
        with self.assertRaisesRegex(ValueError, "configuration is incompatible"):
            train(
                original_manifest,
                checkpoint_dir,
                config_path=changed_config,
                epochs=2,
                resume_from=checkpoint_path,
            )
        with self.assertRaisesRegex(ValueError, "target epoch count is incompatible"):
            train(
                original_manifest,
                checkpoint_dir,
                config_path=self.config,
                epochs=1,
                resume_from=checkpoint_path,
            )
        different_output = self.root / "different-output"
        with self.assertRaisesRegex(ValueError, "output directory is incompatible"):
            train(
                original_manifest,
                different_output,
                config_path=self.config,
                epochs=2,
                resume_from=checkpoint_path,
            )
        self.assertFalse(different_output.exists())

    def test_model_contract_resume_rejected_before_log_or_checkpoint_mutation(self) -> None:
        manifest = self._manifest("contract-resume")
        output = self.root / "contract-output"
        self._train(manifest, output, epochs=2)
        checkpoint = output / "training-state.pt"
        log = output / "training-log.jsonl"
        before_log, before_checkpoint = log.read_bytes(), checkpoint.read_bytes()
        changed = self.root / "contract2.toml"
        changed.write_text(self.config.read_text().replace("[model]", "[model]\nmodel_contract_version = 2"))
        with self.assertRaisesRegex(ValueError, "configuration is incompatible"):
            train(manifest, output, config_path=changed, epochs=2, resume_from=checkpoint)
        self.assertEqual(log.read_bytes(), before_log)
        self.assertEqual(checkpoint.read_bytes(), before_checkpoint)
        saved = torch.load(checkpoint, weights_only=True)
        saved["format_version"] -= 1
        torch.save(saved, checkpoint)
        old_format_checkpoint = checkpoint.read_bytes()
        with self.assertRaisesRegex(ValueError, "unsupported training checkpoint format"):
            train(manifest, output, config_path=self.config, epochs=2, resume_from=checkpoint)
        self.assertEqual(log.read_bytes(), before_log)
        self.assertEqual(checkpoint.read_bytes(), old_format_checkpoint)

    def test_resume_rejects_changed_images_and_external_annotations(self) -> None:
        manifest = self._manifest("external-content")
        directory = manifest.parent
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        annotation_paths = []
        for index, row in enumerate(rows):
            annotation_path = directory / f"annotation-{index}.json"
            annotation_path.write_text(
                json.dumps(row.pop("annotation"), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            row["annotation_path"] = annotation_path.name
            annotation_paths.append(annotation_path)
        manifest.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        output = self.root / "external-content-output"
        self._train(manifest, output, epochs=2)
        checkpoint_path = output / "training-state.pt"

        image_path = directory / rows[0]["image_path"]
        original_image = image_path.read_bytes()
        original_annotation = json.loads(annotation_paths[0].read_text(encoding="utf-8"))
        Image.fromarray(np.full((16, 16, 3), 255, dtype=np.uint8)).save(image_path)
        changed_annotation = dict(original_annotation)
        changed_annotation["image"] = {
            **original_annotation["image"],
            "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        }
        annotation_paths[0].write_text(
            json.dumps(changed_annotation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "training data contents are incompatible"):
            train(
                manifest,
                output,
                config_path=self.config,
                epochs=2,
                resume_from=checkpoint_path,
            )
        image_path.write_bytes(original_image)

        annotation = dict(original_annotation)
        annotation["lines"][0]["polygon_px"] = [[2, 1], [12, 1], [12, 5], [2, 5]]
        annotation_paths[0].write_text(
            json.dumps(annotation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "training data contents are incompatible"):
            train(
                manifest,
                output,
                config_path=self.config,
                epochs=2,
                resume_from=checkpoint_path,
            )

    def test_validation_selects_recall_then_cer_and_keeps_earliest_exact_tie(self) -> None:
        manifest = self._manifest("validation-data", ("a",))
        output = self.root / "validated-output"
        scores = iter((
            {"line_detection_recall_at_iou": 0.5, "cer_on_matched_detections_with_misses_as_empty": 0.1,
             "inference_failure_count": 0, "inference_failure_rate": 0.0, "ground_truth_lines": 4},
            {"line_detection_recall_at_iou": 0.7, "cer_on_matched_detections_with_misses_as_empty": 0.8,
             "inference_failure_count": 0, "inference_failure_rate": 0.0, "ground_truth_lines": 4},
            {"line_detection_recall_at_iou": 0.7, "cer_on_matched_detections_with_misses_as_empty": 0.6,
             "inference_failure_count": 0, "inference_failure_rate": 0.0, "ground_truth_lines": 4},
            {"line_detection_recall_at_iou": 0.7, "cer_on_matched_detections_with_misses_as_empty": 0.6,
             "inference_failure_count": 0, "inference_failure_rate": 0.0, "ground_truth_lines": 4},
        ))
        with patch("prismocr.train.evaluate", side_effect=lambda *args, **kwargs: next(scores)) as evaluator:
            result = self._train(
                manifest,
                output,
                epochs=4,
                validation_manifest=self._manifest("validation-data-heldout", ("a",)),
            )

        self.assertEqual(evaluator.call_count, 4)
        self.assertEqual(result["validation_status"], "measured_on_supplied_manifest")
        self.assertEqual(result["best_epoch"], 3)
        self.assertEqual(result["best_metrics"]["line_detection_recall_at_iou"], 0.7)
        self.assertEqual(result["best_metrics"]["cer_on_matched_detections_with_misses_as_empty"], 0.6)
        self.assertEqual(result["best_metrics"]["inference_failure_count"], 0)
        self.assertEqual(result["selection_policy"], SELECTION_POLICY)
        checkpoint = torch.load(output / "training-state.pt", map_location="cpu", weights_only=True)
        artifact = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["best_epoch"], 3)
        torch.testing.assert_close(
            artifact["model_state"]["weight"],
            checkpoint["best_model_state"]["weight"],
        )
        self.assertFalse(torch.equal(artifact["model_state"]["weight"], checkpoint["model_state"]["weight"]))

    def test_learning_rate_warmup_and_cosine_decay_in_training_log(self) -> None:
        manifest = self._manifest("warmup-data", ("a", "b"))
        output = self.root / "warmup-output"
        self._write_config(self.config, learning_rate=0.01, lr_warmup_steps=2)
        result = self._train(manifest, output, epochs=4)
        self.assertEqual(result["epochs"], 4)
        log_lines = (output / "training-log.jsonl").read_text().splitlines()
        rows = [json.loads(line) for line in log_lines if "learning_rate" in json.loads(line)]
        self.assertGreaterEqual(len(rows), 4)
        lrs = [row["learning_rate"] for row in rows]
        # Step 1 -> step 2 is warming up
        self.assertLess(lrs[0], lrs[1])
        # Later steps decay
        self.assertGreater(lrs[1], lrs[-1])

    def test_init_weights_initializes_training_from_checkpoint_and_raw_state_dict(self) -> None:
        manifest = self._manifest("init-weights-data")
        original_forward = TinyTrainingModel.forward

        # 1. Verify initialization from file with "model_state" key (e.g. checkpoint / model.pt)
        target_weight_checkpoint = torch.tensor(12.345)
        checkpoint_file = self.root / "checkpoint-weights.pt"
        torch.save({"model_state": {"weight": target_weight_checkpoint}}, checkpoint_file)

        observed_weights_ckpt: list[torch.Tensor] = []

        def spy_forward_ckpt(
            model: TinyTrainingModel,
            images: torch.Tensor,
            targets: list[dict[str, object]],
            *,
            teacher_force_ratio: float,
        ) -> dict[str, torch.Tensor]:
            observed_weights_ckpt.append(model.weight.detach().clone())
            return original_forward(model, images, targets, teacher_force_ratio=teacher_force_ratio)

        with patch.object(TinyTrainingModel, "forward", spy_forward_ckpt):
            self._train(manifest, self.root / "out-ckpt", epochs=1, init_weights=checkpoint_file)

        self.assertGreaterEqual(len(observed_weights_ckpt), 1)
        torch.testing.assert_close(observed_weights_ckpt[0], target_weight_checkpoint)

        # 2. Verify initialization from raw model state dict
        target_weight_raw = torch.tensor(98.765)
        raw_file = self.root / "raw-weights.pt"
        torch.save({"weight": target_weight_raw}, raw_file)

        observed_weights_raw: list[torch.Tensor] = []

        def spy_forward_raw(
            model: TinyTrainingModel,
            images: torch.Tensor,
            targets: list[dict[str, object]],
            *,
            teacher_force_ratio: float,
        ) -> dict[str, torch.Tensor]:
            observed_weights_raw.append(model.weight.detach().clone())
            return original_forward(model, images, targets, teacher_force_ratio=teacher_force_ratio)

        with patch.object(TinyTrainingModel, "forward", spy_forward_raw):
            self._train(manifest, self.root / "out-raw", epochs=1, init_weights=raw_file)

        self.assertGreaterEqual(len(observed_weights_raw), 1)
        torch.testing.assert_close(observed_weights_raw[0], target_weight_raw)

        # 3. Verify CLI argument --init-weights via main()
        observed_weights_cli: list[torch.Tensor] = []

        def spy_forward_cli(
            model: TinyTrainingModel,
            images: torch.Tensor,
            targets: list[dict[str, object]],
            *,
            teacher_force_ratio: float,
        ) -> dict[str, torch.Tensor]:
            observed_weights_cli.append(model.weight.detach().clone())
            return original_forward(model, images, targets, teacher_force_ratio=teacher_force_ratio)

        cli_out = self.root / "out-cli"
        with patch.object(TinyTrainingModel, "forward", spy_forward_cli):
            with redirect_stdout(io.StringIO()):
                exit_code = main([
                    "--manifest", str(manifest),
                    "--output", str(cli_out),
                    "--config", str(self.config),
                    "--init-weights", str(checkpoint_file),
                    "--epochs", "1",
                ])

        self.assertEqual(exit_code, 0)
        self.assertGreaterEqual(len(observed_weights_cli), 1)
        torch.testing.assert_close(observed_weights_cli[0], target_weight_checkpoint)

        # 4. Verify that when resume_from is given, resume_from takes precedence over init_weights
        resumed_out = self.root / "out-resume"
        calls = 0

        def fail_at_second_epoch(
            model: TinyTrainingModel,
            images: torch.Tensor,
            targets: list[dict[str, object]],
            *,
            teacher_force_ratio: float,
        ) -> dict[str, torch.Tensor]:
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("interrupted")
            return original_forward(model, images, targets, teacher_force_ratio=teacher_force_ratio)

        with patch.object(TinyTrainingModel, "forward", fail_at_second_epoch):
            with self.assertRaises(RuntimeError):
                self._train(manifest, resumed_out, epochs=2)

        existing_checkpoint = resumed_out / "training-state.pt"
        saved_checkpoint = torch.load(existing_checkpoint, map_location="cpu", weights_only=True)
        resumed_checkpoint_weight = saved_checkpoint["model_state"]["weight"]

        observed_weights_resume: list[torch.Tensor] = []

        def spy_forward_resume(
            model: TinyTrainingModel,
            images: torch.Tensor,
            targets: list[dict[str, object]],
            *,
            teacher_force_ratio: float,
        ) -> dict[str, torch.Tensor]:
            observed_weights_resume.append(model.weight.detach().clone())
            return original_forward(model, images, targets, teacher_force_ratio=teacher_force_ratio)

        with patch.object(TinyTrainingModel, "forward", spy_forward_resume):
            self._train(
                manifest,
                resumed_out,
                epochs=2,
                resume_from=existing_checkpoint,
                init_weights=raw_file,
            )

        self.assertGreaterEqual(len(observed_weights_resume), 1)
        torch.testing.assert_close(observed_weights_resume[0], resumed_checkpoint_weight)
        self.assertFalse(torch.equal(observed_weights_resume[0], target_weight_raw))


if __name__ == "__main__":
    unittest.main()
