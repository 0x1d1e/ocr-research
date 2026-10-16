# Verification log

Append-only record of deterministic checks run against this checkout. A passing entry is software-path evidence for the listed behavior only; it is not release or real-corpus acceptance. Update `docs/implementation-status.md` for current state, not this file.

## 2026-10-16 — `python -m prismocr.verify`

`Ran 207 tests ... OK` (python 3.11, CPU, torch pinned by `pyproject.toml`). Covers geometry, schema, graph decode, table decode, losses/training contracts, training lifecycle, data tools, synthetic fixtures, evaluation/benchmark, tiling, calibration, public API, and ONNX export/parity.

Defects fixed in this pass, each with the regression that now covers it:

| Defect | Fix | Regression |
| --- | --- | --- |
| `_focal_heatmap` treated every target value ≥ 0.99 as foreground, so Gaussian shoulders of large lines trained extra detection centers. | `_heatmap_targets` returns an anchor mask; `_focal_heatmap` uses it as the positive mask. | `TrainingContractTests.test_heatmap_supervises_only_the_integer_anchor_not_gaussian_shoulders`, `...test_heatmap_anchor_matches_decoded_offset_cell` |
| Ambiguous table grids were capacity-validated before ambiguity exclusion, so a schema-valid ambiguous grid could raise instead of being skipped. | `_validate_grid_annotations` skips cells whose table is `annotation_ambiguous`. | `TrainingContractTests.test_ambiguous_table_out_of_range_grid_labels_are_excluded_not_validated` |
| Schema accepted self-intersecting polygons and unbounded vertex counts, which `source_polygon_iou` later rejects or cannot bound. | `geometry.polygon_is_simple` plus a vertex-limit rejection in `schema._valid_polygon`. | `DataAndModelTests.test_schema_rejects_self_intersecting_polygons`, `...test_schema_rejects_excessive_polygon_vertex_counts` |
| Manifest QA and dataset loading trusted declared image dimensions. | `PrismPageDataset._load_records` decodes each image and compares size to the annotation. | `DataToolsTests.test_manifest_qa_rejects_declared_dimension_mismatch` |
| Checkpoint selection could promote an epoch whose pages all failed inference, because zero-denominator recall reads as 1.0 and CER as 0.0. | `_is_better` ranks `inference_failure_count` first and records failure count/rate plus `ground_truth_lines` in `best_metrics`; `CHECKPOINT_FORMAT_VERSION` bumped to 4 so checkpoints holding the old policy are rejected explicitly. | `CheckpointSelectionTests.test_degenerate_epoch_cannot_win_on_empty_recall_and_cer`, `...test_selection_prefers_recall_then_cer_for_working_epochs`, `TrainingLifecycleTests.test_validation_selects_recall_then_cer_and_keeps_earliest_exact_tie` |
| Aggregate evaluation reported perfect recall/CER/reading-order/table metrics from zero denominators. | `evaluate()` adds `metrics_are_defined` and `zero_denominator_metrics`. | `EvaluatorTests.test_blank_manifest_reports_zero_denominator_metrics_as_undefined` |

## 2026-10-16 — end-to-end CLI smoke (synthetic, CPU)

Ran the installed entry points end to end on a generated synthetic corpus (`prismocr.data.synthetic`, DejaVu font, tiny config: `dim=20`, 8 text proposals, buckets `[8,16]`), two training documents plus one held-out document:

- `python -m prismocr.train` wrote `model.pt`, `vocab-v4.json`, `model-config.json`, `manifest.json`, `training-state.pt`, `training-log.jsonl`, `training-run.json`.
- `python -m prismocr.onnx_export` wrote `core.onnx`, `recognizer-8.onnx`, `recognizer-16.onnx`, `model-weights.bin`, `vocab-v4.json`, and reported `parity.status = "passed"`.
- `python -m prismocr.eval` reported `status = measured_on_supplied_manifest_not_release_validation`, `page_count = 1`, `inference_failure_count = 0`, `zero_denominator_metrics = ["table_structure"]` (the synthetic pages contain no tables), and `reading_order.pair_count = 1` with accuracy 0.0 because the one-epoch model predicted no lines.
- `python -m prismocr.benchmark` exited 2 with `status = incomplete_workload_or_failed_inference` and `missing_required_workloads` listing the buckets the one-page manifest cannot cover; it still measured the `lines_1_10` and `without_tables` workloads and reported `inference_parameter_bytes` and `max_rss_bytes_process_wide`. This is the intended refusal to call an incomplete workload complete.

This confirms the entry points interoperate and that reports stay honest about undefined metrics, untrained quality, and incomplete workloads. It is not OCR quality or release evidence.

## 2026-10-16 — local real corpus, frozen split report, full suite

`PYTHONPATH=/home/kosal/ocr-research` is required for these commands (the package is not installed in the sandbox interpreter).

- `python -m prismocr.verify` — `Ran 228 tests ... OK` (python 3.12, CPU). Software-path evidence only.
- `scripts/build_real_corpus.py --splits train val test --jobs 14` rewrote `~/data/prismocr-corpus/splits/{train,val,val-select,test}.jsonl`. Page counts: train 41227, val 8043, validation selection subset 843, test 6546. KhmerST scene pages contribute train 1227 / val 163 / test 152 after content dedup of the two byte-identical source image pairs.
- Byte-disjointness check across the manifests: train 41205 unique `_image_sha256`, val 8043, test 6544, val-select 843, and zero identical image bytes shared by train/val, train/test, val/test, or train/val-select. The 22 train and 2 test intra-split repeats are within-document DocLayNet near-blank pages.
- The corpus builder now writes `splits/split-report.json` (sha256 `ddfa86b3aabfd418ccf7da4b58b4d3591ff967efbe5543b0a6f6f144f3f89d0b`) through `prismocr.data.tools.write_split_report` with `near_duplicate_hamming = null`, so split membership is proven by `document_id`, `template_family_id`, and exact image bytes only. Perceptual dHash grouping is opt-in because uniform pages from unrelated DocLayNet documents collapse to an identical hash and produced false cross-split groups.
- `prismocr.calibration._validation_split_provenance` and `_verified_split_datasets` accept that report and return the frozen datasets (train 41227 / validation 843 / test 6546), so the earlier calibration block on a missing or leaking split report is cleared. `tests.test_data_tools` and `tests.test_calibration` run 15 tests OK, including `...test_split_report_perceptual_grouping_is_opt_in`.

This is corpus construction and provenance, not human label QA, convergence, or OCR quality evidence.

## 2026-10-16 — box-regression gradient defect

Diagnosing the real-v1/v2 zero-proposal result on the built corpus (checkpoints emitted no 
lines at the configured 0.55 text threshold) traced to `PrismOCRLoss._box_loss`. It regressed 
normalized side lengths linearly with `F.smooth_l1_loss`, while `TextLineHead` decodes
`size = softplus(raw - 2.0)`. Once the size head drifts negative, `d(size)/d(raw) = softplus'(raw - 2)`
vanishes, so a predicted line height of 0.0005 against a 0.027 target cost about 3.7e-4 and
recovered only ~0.0004 of raw per 400 steps. A single-page overfit at 1024x1408 reached
`p_at_anchor` 0.84 with box loss 0.0087 yet `quality_ge_0.5 = 0/39` (IoU 0.02-0.12).

Fix: `_box_loss` keeps the smooth L1 center term and the periodic orientation term but regresses
`log(size)`, so the chain-rule factor is `sigmoid(x)/softplus(x) -> 1` for negative `raw` and the
term is a scale-invariant side ratio. After the change the same overfit reached
`quality_mean` 0.45 with `quality_ge_0.5 = 15/39` and finite non-zero gradient at `raw = -7`.

Regressions added in `tests/test_training_contracts.py`:
`test_box_loss_is_zero_for_identical_boxes`, `test_box_loss_keeps_gradient_for_tiny_predicted_sizes`,
`test_box_loss_penalizes_size_ratios_symmetrically`. `python -m prismocr.verify` then runs
`Ran 231 tests ... OK`. This is a supervision/optimization fix, not evidence of OCR quality.

## Standing blockers

Real-corpus calibration reliability, convergence, CPU/GPU latency, and PyTorch/ONNX parity on a trained artifact all require a permissioned Khmer/English manifest and pinned target hardware. None were available in this project or standard local data roots (`/mnt/data`, `/data`, `/home/kosal/datasets`). Do not mark the corresponding spec gates satisfied from the tests above.
