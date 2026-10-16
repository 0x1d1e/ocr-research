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

## 2026-10-16 — review findings verification & test suite expansion (281 tests, synthetic CLI smoke)

Ran full deterministic test suite with OpenMP thread bounding:
`OMP_NUM_THREADS=12 python -m prismocr.verify` — `Ran 281 tests in 12.542s ... OK` (Python 3.12, CPU).

### Fixed review findings and regression coverage

| Finding | Description & Fix | Regression / Test |
| --- | --- | --- |
| **Finding 3.4** | EXIF orientation in `PrismOCR._tensor_image`: images loaded via PIL are transposed to the displayed frame via `exif_oriented(image)` so orientation tags (e.g. tag 6, 90° CCW) do not transpose inference dimensions or bounding boxes. | `PublicInferenceIntegrationTests.test_exif_orientation_tag_6_transposes_to_displayed_frame` |
| **Finding 3.16** | Model Contract v2 alignment for diagnostic scripts: `scripts/diagnose_heatmap.py` passes `model_contract_version` from config and validates model state via `validated_model_state`; `scripts/diagnose_ocr.py` uses `TextLineHead.decode_sizes` with `model_contract_version` and image dimensions. | Diagnostic scripts (`scripts/diagnose_heatmap.py`, `scripts/diagnose_ocr.py`) aligned with contract v2 |
| **Finding 3.15** | Safe `_is_autocast_enabled` fallback across PyTorch versions: catches `TypeError` when `torch.is_autocast_enabled(device_type)` is not supported and falls back to zero-argument `torch.is_autocast_enabled()`. | Safe fallback in `prismocr/model/backbone.py` exercised across AMP paths |
| **Finding 3.2** | Direct GT anchor-cell box and role supervision: when proposal scores are zero or during early training, target box regression and role classification loss backpropagate directly through feature map anchor cells rather than dropping supervision. | `TrainingContractTests.test_anchor_cell_supervision_produces_gradients_when_proposal_scores_are_zero` |
| **Finding 3.3** | CTC recognition bootstrap on untrained proposals using anchor-cell embeddings: when accepted proposals are empty (or teacher forcing is active), ground-truth lines are assigned anchor-cell embeddings so the recognizer receives gradient and trains CTC weights before detector convergence. | `TrainingContractTests.test_ctc_loss_bootstrap_backpropagates_to_recognizer_when_accepted_proposals_empty` |
| **Finding 3.6** | Learning rate warmup & cosine decay scheduler, plus threshold-free validation detection metrics / tie-breaking: added linear warmup and cosine annealing via `LambdaLR` logged to `training-log.jsonl`; checkpoint selection uses threshold-free proposal metrics (`mean_proposal_iou` / `proposal_ap_50`) to advance and break ties when detector peaks remain below operating threshold. | `TrainingLifecycleTests.test_learning_rate_warmup_and_cosine_decay_in_training_log`, `CheckpointSelectionTests.test_threshold_free_proposal_metric_selects_best_checkpoint_when_peaks_below_threshold` |

### End-to-end synthetic CLI smoke run (CPU)

Ran all CLI entry points end to end on generated synthetic manifests (`prismocr.data.synthetic.generate_synthetic_dataset`, DejaVu font, tiny config: `dim=20`, 8 text proposals, buckets `[8,16]`), two training pages plus one held-out validation page:

- `python -m prismocr.train --manifest <train> --output <dir> --config <cfg> --validation-manifest <val>`: trained 1 epoch, verified warmup and cosine decay scheduling, computed threshold-free validation metrics, and wrote `model.pt`, `vocab-v4.json`, `model-config.json`, `manifest.json`, `training-state.pt`, `training-log.jsonl`, `training-run.json`.
- `python -m prismocr.onnx_export --artifact <dir> --output <onnx_dir>`: exported `core.onnx`, `recognizer-8.onnx`, `recognizer-16.onnx`, `model-weights.bin`, `vocab-v4.json`, and verified numeric parity (`parity.status = "passed"`).
- `python -m prismocr.eval --manifest <val> --artifact <dir>`: successfully evaluated held-out synthetic page with zero inference failures (`inference_failure_count = 0`), reporting honest unvalidated metrics (`status = "measured_on_supplied_manifest_not_release_validation"`).
- `python -m prismocr.benchmark --manifest <val> --artifact <dir>`: ran representative workloads, reported parameter sizes and process memory, and exited 2 with `status = "incomplete_workload_or_failed_inference"` due to expected unrepresented workload buckets on single-page manifest.

## 2026-10-16 — Phase 2 & Phase 3 verification: fused text head, corpus caching, lineage & supervision (287 tests)

Ran full deterministic verification suite under 12-core CPU bounding:
`OMP_NUM_THREADS=12 python -m prismocr.verify` — `Ran 287 tests in ~18.5s ... OK` (Python 3.12, CPU).

### Verified Phase 2 / Phase 3 implementations and review findings

| Finding / Topic | Implementation details & Verification | Regression / Test |
| --- | --- | --- |
| **Finding 3.1** | **Fused text head**: Added configurable top-down multiscale lateral feature fusion (`fused_text_head: bool = False`) in `PrismOCRModel`. Lateral 1x1 convolutions and bilinear upsampling fuse deep semantic features from `f16` (stride 16) and `f8` (stride 8) into the high-resolution `f4` (stride 4) text feature map before the line head. Empirically expands text-head effective receptive field from 23x55 px to 239x559 px (>100x receptive area expansion), providing broad horizontal context for long text lines. FP32 parameter byte inventory strictly maintains the <10MB ceiling (9,885,052 bytes / 9.88MB vs 10,000,000-byte cap). Full ONNX export and runtime numeric parity pass across all recognizer sequence buckets (`recognizer-8`, `recognizer-16`, etc.) and variable image sizes. | `ModelContractTests.test_fused_text_head_receptive_field_and_inventory`, `ModelContractTests.test_fused_text_head_onnx_export_and_parity` |
| **Finding 3.9** | **Stale corpus cache elimination**: In `scripts/build_real_corpus.py`, `_completed_shard` validates existing shards against `shard_fingerprint.json` tracking 12 configuration facets (`labels`, `labels_sha256`, `pages`, `license`, `permission`, `template_family`, `name`, `source_class`, `source_dataset`, `seed`, `prefix`, `page_offset`). Upstream KhmerST and DocLayNet conversion manifests similarly record `.meta.json` sidecars (`input_file_sha256`, `seed`, `split`, `limit`), preventing stale cached shards from surviving changes to source annotations, seeds, or build limits. | `RecognitionCompositionTests.test_shard_fingerprint_creation_and_invalidation` |
| **Finding 3.8** | **Source crop lineage & split disjointness**: In `compose_recognition_pages` (`prismocr/data/real_corpus.py`), composed lines record explicit crop lineage via `source_crop_id` and `source_crop_hash`. `_group_records` performs transitive union-find grouping over document IDs, template families, image SHA-256 digests, and source crop IDs/hashes. `_split_membership` asserts strict cross-split disjointness, raising `ValueError` if any source crop or group spans multiple splits. `write_split_report` audits and records exact transcript and character 3-gram Jaccard overlap between splits. | `DataToolsTests.test_source_crop_grouping_and_split_disjointness`, `DataToolsTests.test_split_report_transcript_and_ngram_overlap` |
| **Finding 3.18** | **SPDX compliance & dataset provenance**: Governed metadata compliance standardized across corpus tooling: SPDX-compliant license naming (`NOASSERTION` for proprietary/unasserted local synthetic generators, `Apache-2.0`, `CDLA-Permissive-1.0`), hierarchical source dataset provenance paths in `source_dataset` (e.g. `paddlex_rec_dataset/<shard>`), and explicit `automated:` reviewer prefixes (`automated:prismocr-corpus-build`, `automated:KhmerST_annotation_release`, `automated:DocLayNet_annotation_release`). | Manifest schema and provenance validation in `scripts/build_real_corpus.py` and `prismocr/data/real_corpus.py` |
| **Finding 3.5** | **GT PARENT relation supervision**: `compose_recognition_pages` groups placed text lines into paragraph/block regions when vertical line spacing exceeds threshold (12px), creating schema-valid container `REGION` annotations with verified geometric containment. Emits ground-truth `PARENT` relations (`line -> region`), providing dense, geometrically verified supervision for the structural graph head. | `RecognitionCompositionTests.test_composed_pages_keep_crop_geometry_transcripts_and_provenance` |
| **Finding 3.16 follow-up** | **Diagnostic text feature routing**: In `scripts/diagnose_ocr.py`, `_probe_direct` and `_probe_canvas` route backbone features through `model.forward_text_features(f4, f8, f16)` prior to calling `model.text_head`, ensuring seamless diagnostic evaluation for both fused (`fused_text_head=True`) and unfused (`fused_text_head=False`) text head configurations. | `DiagnosisTests.test_direct_and_canvas_probe_support_both_fused_and_unfused_text_head` |

## 2026-10-16 — Phase 3 follow-up verification: decision-normalized losses, OOV accounting, table composition, CI & pinned dependencies (290 tests)

Ran full deterministic verification suite under 12-core CPU bounding:
`OMP_NUM_THREADS=12 python -m prismocr.verify` — `Ran 290 tests in 20.2s ... OK` (Python 3.12, CPU).

### Verified review findings and regression coverage

| Finding / Topic | Implementation details & Verification | Regression / Test |
| --- | --- | --- |
| **Finding 3.12** | **Role and graph loss normalized by instance decisions**: `PrismOCRLoss` normalizes role and graph loss by instance decisions rather than page count or batch size. Role loss collects individual cross-entropy terms per supervised line role into `role_terms` and calculates the mean across supervised line roles (`torch.stack(role_terms).mean() if role_terms else zero`). Graph loss collects cross-entropy terms across valid candidate relations into `graph_terms` and calculates the mean across supervised candidate relations (`torch.stack(graph_terms).mean() if graph_terms else zero`). This prevents loss dilution in sparse batches or batches with variable line and relation counts. | `TrainingContractTests.test_role_and_graph_loss_normalized_by_decisions_not_batch_size` |
| **Finding 3.11** | **Separate OOV accounting**: `PrismOCRLoss` separates line-level drop reporting from codepoint counts: `rejected_oov_line_count` tracks lines dropped from CTC training due to out-of-vocabulary codepoints (`ctc_oov_characters` in `outputs["ocr_unsupported"]`), while `oov_character_count` tracks total OOV codepoint occurrences across all ground-truth line texts. This keeps dropped text line counts distinct from character-level vocabulary mismatches. | `TrainingContractTests.test_oov_accounting_tracks_rejected_lines_and_character_counts` |
| **Finding 3.5** | **Synthetic table generator (`compose_table_pages`)**: Implemented `compose_table_pages` in `prismocr/data/real_corpus.py` to synthesize table layouts providing exact table regions (`kind: TABLE`), grid cells (`row_index`, `col_index`, `row_span`, `col_span`), line-to-cell and cell-to-table `PARENT` relations, and `RIGHT_CELL` affinities, fully validated against schema v4.0 (`validate_annotation`). | `RecognitionCompositionTests.test_compose_table_pages_generates_valid_table_grid_and_relations` |
| **Findings 3.15 & 3.19** | **Pinned dependencies & automated CI workflow**: Added `requirements-lock.txt` with pinned deployment dependencies (`torch==2.4.1`, `numpy==1.26.4`, `Pillow==10.4.0`, `onnx==1.17.0`, `onnxruntime==1.19.2`). Added `.github/workflows/ci.yml` configuring automated CI unit testing on Python 3.11 under bounded threads (`TORCH_NUM_THREADS=2`, `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=2`) to prevent CPU thread oversubscription. | Pinned environment files: `requirements-lock.txt`, `.github/workflows/ci.yml` |

## 2026-10-16 — matching optimization, project milestone delineation & test suite re-verification (290 tests)

Ran full deterministic verification suite under 12-core CPU bounding:
`OMP_NUM_THREADS=12 python -m prismocr.verify` — `Ran 290 tests in ~20.6s ... OK` (Python 3.12, CPU).

### Verified review findings and regression coverage

| Finding / Topic | Implementation details & Verification | Regression / Test |
| --- | --- | --- |
| **Finding 3.20** | **Proposal matching AABB precomputation & zero-overlap bypass**: Proposal matching (`prismocr/matching.py`) optimization precomputes rotated box polygon vertices, areas, and axis-aligned bounding boxes (AABBs) for ground-truth and candidate proposals. An early zero-overlap bypass checks disjoint AABB extents (`gt_max_x < pred_min_x or pred_max_x < gt_min_x or gt_max_y < pred_min_y or pred_max_y < gt_min_y`) to assign 0.0 IoU immediately, avoiding redundant Sutherland-Hodgman polygon clipping while preserving exact mathematical IoU equivalence. | `GeometryAndMatchingTests.test_aspect_aware_box_and_iou`, `GeometryAndMatchingTests.test_assignment_coverage_is_not_detection_recall` |
| **Finding 3.21** | **Explicit project milestone delineation across documentation**: Standardized explicit project milestone delineation across documentation (`docs/implementation-status.md`, `README.md`): Milestone 1: `Corpus Built`, Milestone 2: `Labels QA'd`, Milestone 3: `Model Trained`, and Milestone 4: `Model Validated`. Formally decouples software-path verification from certified training data and physical deployment benchmarking. | Documentation alignment in `docs/implementation-status.md` and `README.md` |

## 2026-10-17 — Phase 4 verification: real corpus build, manifest QA audit, default fused text head & FP16 fusion parity (297 tests)

Ran full deterministic verification suite under 12-core CPU bounding:
`OMP_NUM_THREADS=12 python -m prismocr.verify` — `Ran 297 tests in 20.052s ... OK` (Python 3.12, CPU).

### Verified Phase 4 implementations and milestone progress

| Finding / Topic | Implementation details & Verification | Regression / Test |
| --- | --- | --- |
| **Milestone 1: Corpus Built** | **Real corpus build completed**: Executed `scripts/build_real_corpus.py` across train (41,227 pages), val (8,043 pages), val-select (843 pages), and test (6,546 pages) from `paddlex_rec_dataset`, `khmerst-full`, and `DocLayNet`. Enforces transitive group closure (`_group_records`), a strict split DAG hierarchy (`train -> val -> test`), and sequential crop hash exclusion (`exclude_crop_hashes`). Shard fingerprints track `exclude_crop_hashes_digest` and `exclude_crop_hashes_count` alongside the 12 existing configuration facets in `shard_fingerprint.json`, ensuring cached shards invalidate if prior-split crop contents change. | `RecognitionCompositionTests.test_compose_recognition_pages_excludes_crops`, `RecognitionCompositionTests.test_build_real_corpus_shard_fingerprint_and_crop_hash_collection`, `RecognitionCompositionTests.test_collect_crop_hashes_handles_none_lines_and_missing`, `RecognitionCompositionTests.test_build_real_corpus_split_hierarchy` |
| **Milestone 2: Labels QA'd** | **Manifest QA audit completed**: Completed manifest structural and disjointness audit via `prismocr.data.tools qa` across `val-select.jsonl` (843 pages, 11,011 lines, 0 duplicate image hashes) and `train.jsonl` (41,227 pages, 387,134 lines). Verified image dimension alignment, annotation schema validity, and strict split disjointness across documents, template families, image SHA-256 digests, and source crop hashes. `split-report.json` records 2,517 independent groups with zero cross-split byte duplicates. | `DataToolsTests.test_manifest_qa_reports_exact_and_perceptual_duplicate_candidates`, `DataToolsTests.test_manifest_qa_rejects_declared_dimension_mismatch`, `DataToolsTests.test_source_crop_grouping_and_split_disjointness` |
| **Configuration alignment** | **Default `fused_text_head` in v4 configs**: Enabled `fused_text_head = true` by default in `configs/v4_detection.toml` and `configs/v4_real_corpus.toml`. Top-down multiscale lateral feature fusion (f16 and f8 into f4) expands text-head receptive field to 239x559 px (>100x area expansion) while holding parameter inventory to 9.88MB (<10MB ceiling). | `ModelContractTests.test_fused_text_head_receptive_field_and_inventory`, `ModelContractTests.test_fused_text_head_onnx_export_and_parity` |
| **ONNX export & runtime** | **FP16 fusion parity proposal alignment & boundary divergence**: Added `_align_fusion_parity_outputs` in `prismocr/onnx_export.py` to canonicalize proposal-indexed tensors (`text_boxes`, `text_scores`, `text_role_logits`, `text_embeddings`, `text_mask`, `text_candidate_mask`, `text_proposal_indices`) and relation logits/masks by sorted proposal index prior to parity comparisons. Extended `_fusion_parity_report` to tolerate marginal float16 score drift at the top-K capacity boundary within `atol=1e-3` / `rtol=1e-2` while strictly rejecting non-marginal discrepancies. Validated parity across FP32 and FP16 modes. | `ExportableHeadTests.test_fusion_parity_proposal_alignment_canonicalizes_swapped_slots`, `ExportableHeadTests.test_fusion_parity_report_tolerates_marginal_boundary_proposals`, `ExportableHeadTests.test_fusion_parity_report_rejects_non_marginal_boundary_proposal_discrepancy` |

## Standing blockers

Real corpus construction (Milestone 1) and manifest QA audit (Milestone 2) are completed. Full-scale training convergence (Milestone 3), real-corpus calibration reliability, CPU/GPU target latency benchmarking, and PyTorch/ONNX parity on a trained checkpoint (Milestone 4) require executing multi-epoch GPU training and evaluating the resulting artifact on pinned target hardware. Do not mark the corresponding spec gates satisfied from software tests alone.

