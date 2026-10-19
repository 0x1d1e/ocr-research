# OCR Validation and Architecture Review

**Status date:** 2026-10-19
**Scope:** Real-v4 training outcome, PaddleOCR comparison, PrismOCR inference and data claims, dependency/CI/demo risks, and proposed next work.
**Review type:** Source inspection plus bounded GPU diagnostics on the existing validation split. No code changes to application/model behavior and no new training run were performed as part of this review.

## Executive assessment

The real-v4 run was stopped at the user's direction during epoch 14. Its selected checkpoint was epoch 4. Validation results showed very poor detection precision and high text error. Follow-up comparison on all 643 complete-line validation pages found substantially stronger results from the installed PaddleOCR detector and local fine-tuned Khmer/English recognizer. PrismOCR also had very high blank CTC output rates when given ground-truth line geometry, so detector false positives alone do not explain the text failure.

The results support treating the current PrismOCR artifact as **not release validated**. They do not establish that PaddleOCR architecture is categorically superior: the local Paddle recognizer is fine-tuned, and PrismOCR and Paddle do not consume identical recognition inputs. The validation split was used for checkpoint selection and subsequent diagnosis, so these measurements are diagnostic and must not be represented as an independent final test result.

The source audit found a mixture of confirmed defects, partly true claims, and claims not established by the available code review. The most actionable confirmed concerns are the ONNX Runtime lock mismatch, undeclared Gradio dependency, demo's unauthenticated all-interface default and direct path acceptance, unbounded model cache, error detail exposure, inert `strict_ctc_length` behavior, incomplete training-level leakage guard, and lack of a general inference image pixel cap.

## 1. Training run and OCR evidence

### Training state

- Run: `real-v4`; artifact used for the comparisons: `/home/kosal/data/prismocr-runs/real-v4-ep4-checkpoint/artifact`.
- The training process was terminated with SIGTERM after explicit user authorization. It stopped during epoch 14; the latest completed validation was epoch 13.
- The best checkpoint remained epoch 4. Validation CER worsened from 17.43 at epoch 4 to 27.22 at epoch 13.
- Epoch 13 reported 11,011 ground-truth lines and 1,644,489 predictions: recall 71.91%, precision 0.623%, F1 1.236% at IoU 0.5. Unmatched detections count as text insertions, confounding aggregate CER/WER.
- These are the recorded run metrics, not release validation.

### Full PaddleOCR comparison

The validation-only comparison selected every page with complete line supervision: 643 pages and 11,011 lines. The remaining 200 manifest pages lacked complete line labels and were excluded. Source counts were fintech 120, KhmerST 163, KhmerSynth 120, and WildKhmer 240. No test split was opened.

| Measure | PaddleOCR | PrismOCR |
|---|---:|---:|
| Detection F1, IoU 0.50 | 0.8161 | 0.1331 |
| Detection F1, IoU 0.75 | 0.5158 | 0.0931 |
| GT-geometry recognition CER, overall | 0.1679 | 0.8420 |
| GT-geometry recognition CER, Khmer | 0.1867 | 0.8795 |

At IoU 0.50, PrismOCR had 6,173 true matches, 75,577 false positives, and 4,838 false negatives. It emitted 81,750 detections over 643 pages. The model returned exactly 128 detections on 636 pages, consistent with the configured per-call proposal cap; tiling can multiply such outputs across tiles.

Recognition summaries cover the 11,011 ground-truth lines and reported zero inference exceptions for both systems. PrismOCR's oracle CER is high even when using ground-truth geometry, so detector false positives are not the only source of poor text metrics.

#### Source breakdown

| Source | Pages | Lines | Paddle CER | PrismOCR CER | Paddle detection F1 @ .50 | PrismOCR detection F1 @ .50 |
|---|---:|---:|---:|---:|---:|---:|
| fintech | 120 | 3,157 | 0.0749 | 0.8039 | 0.8059 | 0.2088 |
| KhmerST | 163 | 372 | 0.2059 | 0.9938 | 0.2929 | 0.0102 |
| KhmerSynth | 120 | 2,902 | 0.1794 | 0.8117 | 0.9473 | 0.1925 |
| WildKhmer | 240 | 4,580 | 0.2896 | 0.9680 | 0.8561 | 0.1347 |

The small KhmerST line count makes its detection statistics especially sensitive to a small number of errors.

#### Language breakdown

| Language bucket | Lines | Paddle CER | PrismOCR CER |
|---|---:|---:|---:|
| Khmer | 8,155 | 0.1867 | 0.8795 |
| English | 2,010 | 0.1258 | 0.7569 |
| Other or empty | 846 | 0.2244 | 0.9710 |

Paddle's recognizer was a **local PP-OCRv6 tiny Khmer/English fine-tune**, not an official PaddleOCR Khmer model. The stock recognizer available in the environment did not contain Khmer characters. OOV counts were computed by code point against model dictionaries; they are not a full measure of model coverage.

### Controlled 48-page recognition follow-up

A stratified 48-page, 688-line validation subset used ground-truth quadrilateral geometry for both recognizers. It capped recognition at 20 lines per page. The comparison still cannot provide pixel-identical model inputs: Paddle consumes a perspective-warped RGB crop resized by its inference preprocessing to `3×48×320`; PrismOCR consumes full-page-derived feature strips and a text-head node embedding. The trained PrismOCR architecture has no existing RGB crop encoder, so a faithful same-crop route is unavailable without changing/training the model.

| Measure | PaddleOCR | PrismOCR |
|---|---:|---:|
| CER | 0.1405 | 0.8251 |
| Exact-line rate | 33.3% | 0% |
| Blank output rate | 0.15% | 54.4% |

PrismOCR's 256-frame greedy paths had 93.8% blank frames and a mean blank posterior of 0.825 on this sample. This is evidence of a serious recognition-path/training problem, but it does not isolate architecture, training data, page-feature extraction, or preprocessing as the cause.

### Detector score and proposal-cap diagnostic

A separate 12-page validation diagnostic covered 187 ground-truth lines across six strata. Raw text proposal confidence is available as the sigmoid heatmap score in `outputs["text"]["scores"]`; public serialization removes the internal `_tile_score` field. The proposal head was saturated at 128 candidates per page at every tested threshold from 0.00 through 0.70.

| Raw score threshold | Predictions | Precision @ .50 | Recall @ .50 | F1 @ .50 | Pages over 128 cap |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 1,536 | 0.060 | 0.492 | 0.107 | 12/12 |
| 0.30 | 1,536 | 0.060 | 0.492 | 0.107 | 12/12 |
| 0.55 | 1,536 | 0.060 | 0.492 | 0.107 | 12/12 |
| 0.70 | 1,536 | 0.060 | 0.492 | 0.107 | 12/12 |
| 0.85 | 1,275 | 0.072 | 0.492 | 0.126 | 9/12 |

On the same small diagnostic, top-k F1 at IoU .50 was 0.206 for top-16, 0.186 for top-32, 0.155 for top-64, and 0.107 for top-128. These are validation-derived diagnostics, not recommended thresholds or held-out operating points. Lowering the threshold cannot expand the candidate set while the top-128 cap is saturated. The threshold-0 pool contained 208 to 88,470 qualifying candidates per page.

The existing bounded inference diagnostic also observed public tiled outputs of roughly 1,200–3,300 lines on its 12 selected pages, while untiled outputs were at the 128-per-page cap. This suggests an inference/tiling aggregation issue or multiplication of per-tile proposals and merits a direct trace. It should not be treated as evidence that a simple confidence threshold fixes the problem.

### Experiment limitations

- `val-select.jsonl` was used for checkpoint selection and the follow-up diagnostics. This is diagnostic evidence, not an unbiased final quality estimate.
- The corpus report documents split controls, but repeated transcript n-grams remain a generalization consideration.
- Source annotation completeness and human label QA are not established for every label.
- Paddle and Prism recognition inputs differ, despite using the same GT line geometry.
- No official Paddle Khmer baseline was evaluated.
- No new training, production fix, test-set evaluation, or CPU deployment benchmark was performed.

## 2. PaddleOCR and FintechOCR adoption assessment

The useful PaddleOCR lesson is modular measurement: score detection separately and evaluate recognition on fixed ground-truth lines before relying on end-to-end CER/WER. The validation comparison supports that approach. Paddle's numerical DB detector thresholds are not transferable to PrismOCR proposal scores.

The available Khmer-capable Paddle recognizer is a local fine-tune. Its observed performance is a useful experimental reference, not an official PaddleOCR Khmer quality claim. A future controlled model comparison should document model provenance, vocabulary, preprocessing, and training corpus.

“FintechOCR” was ambiguous. The likely public match found was FIN-OCR, a FINOS project for MICR/check recognition. Its transferable idea is to inspect intermediate stages and isolate translators; its check-specific OCR and templates are not general Khmer OCR replacements. No evidence identified a different FintechOCR project in this repository.

## 3. Claim-by-claim source audit

Classifications below refer to source review of the current workspace, not to fresh runtime verification. No tests were run for this audit.

### Most urgent claims

1. **ONNX Runtime pin conflict: confirmed mismatch; CI failure claim overstated.** The exporter requires ONNX Runtime 1.24.3 at `prismocr/onnx_export.py:324`; `requirements-lock.txt:5` pins 1.19.2. The optional `export` extra in `pyproject.toml:13` specifies 1.24.3. This can block export under the lock, but current CI does not run export, so CI failure is not established.
2. **Gradio undeclared: confirmed.** `prismocr/demo.py:13` and `tests/test_demo.py:11` import Gradio at module import time. Gradio is absent from declared dependencies in `pyproject.toml:10` and `requirements-lock.txt:1`. A clean install can fail test collection or demo import; no clean-install CI reproduction was run.
3. **Demo binds unauthenticated to all interfaces: confirmed.** `prismocr/demo.py:377` defaults to `0.0.0.0`; `main` launches with no authentication configuration at `prismocr/demo.py:411`. Public sharing is optional via `--share`.
4. **Client can pass any server path as `model_name`: confirmed.** `process_document` accepts any existing path after checking discovered artifact labels in `prismocr/demo.py:216`. This bypasses the artifact list. Impact depends on server filesystem permissions and whether the path is a valid model artifact.
5. **Model cache is unbounded: confirmed.** `_MODEL_CACHE` is a plain dictionary at `prismocr/demo.py:26`; `get_cached_model` inserts each distinct resolved path/device pair at `prismocr/demo.py:93`; there is no eviction policy.
6. **Demo error messages leak paths/details: confirmed.** `process_document` returns `str(exc)` in JSON and renders the exception in Markdown at `prismocr/demo.py:225`. The exception may contain filesystem paths or model-loading details.
7. **`torch.load(weights_only=True)` means no pickle risk: overstated.** API/checkpoint paths use `weights_only=True`, which reduces arbitrary pickle loading risk. It is not a general guarantee that loading an untrusted checkpoint is harmless. The demo itself does not call `torch.load` directly.

### Training and data claims

8. **`strict_ctc_length` does nothing but changes resume fingerprint: substantially confirmed.** It appears in resolved settings and resume compatibility at `prismocr/train.py:114` and `prismocr/train.py:251`; the loss stores the value, but overlength examples are rejected unconditionally at `prismocr/losses.py:607`. The flag therefore appears behaviorally inert while still affecting resume compatibility.
9. **No unannotated region/cell state; KhmerST and ICDAR pages become region negatives: not established as stated.** The code audit found explicit unavailable/partial/complete supervision semantics for text and noted calibration excludes noncomplete text pages. The stronger claim about region/cell supervision and named sources needs a targeted trace from manifest generation through region/cell loss masks before confirmation. It remains a priority question, not a verified finding here.
10. **One oversize table label aborts whole training: not confirmed.** The public API table path checks a 100,000 slot bound, but the exact training failure path was not established.
11. **Training leakage guard is narrower than split tooling: confirmed.** `prismocr/train.py:63` checks train/validation overlap by document ID, template family, and exact image bytes. Broader source crop ID/hash checks exist in `prismocr/data/tools.py:318` and split report checks in `prismocr/data/calibration.py:430`; tests exist in `tests/test_data_tools.py` and `tests/test_calibration.py`. The training guard itself does not perform those broader crop-ID/hash checks.
12. **`compose_table_pages` cannot exclude held-out crops and split builder cannot split composed runs: partly investigated, unresolved.** Composition and grouped split assignment exist in `prismocr/data/real_corpus.py:468` and `prismocr/data/tools.py:592`. Generated composed data is marked as requiring human label QA. The claimed exact limitation was not confirmed in this audit.
13. **Calibration fails if no regions/relation types; thresholds fit in-sample: partly true.** Platt fitting rejects empty and one-class samples at `prismocr/calibration.py:54`. Calibration is based on validation, and provenance/checkpoint dataset hashes are checked at `prismocr/calibration.py:510`. It is not test-fitted, but validation fitting does not establish out-of-sample reliability. Whether all empty region/relation streams are handled at every call site was not established.
14. **Fresh run overwrites existing output without asking: partly true.** Split report generation refuses overwrite unless requested (`prismocr/data/tools.py:531`). Corpus composition writes into destination paths (`prismocr/data/real_corpus.py:448`). Checkpoint writes and artifact publication have safeguards/rollback mechanisms. Behavior must be reviewed per command; the blanket claim is too broad.
15. **Threshold-free metric compares coordinate frames, ~15 px offset; only affects ties: partly confirmed.** The metric uses collated page dimensions and boxes in model/padded frame at `prismocr/train.py:364`, and is a threshold-zero proposal-overlap selection metric rather than deployment detection quality. The exact maximum offset estimate and whether only ties can be affected were not established in the audit.

### Robustness claims

16. **No input pixel cap: confirmed at inference/demo boundary.** Training has optional `max_page_side` downscaling in `prismocr/data/dataset.py:175`, but that is not a general decoded-pixel limit. No equivalent demo upload cap was found.
17. **`validate_grid` and `render_table` have no span bounds: partly true.** The API enforces a slot cap in `prismocr/table_decode.py:388` and checks dimensions at `prismocr/table_decode.py:767`; direct `render_table` iteration in `prismocr/decode.py:139` is unbounded. Span-specific validation was not exhaustively traced.
18. **`PrismOCR.load` parses checkpoint before rejecting missing manifest: partly true.** Checkpoint loading uses `weights_only=True` before strict state loading in `prismocr/api.py:327`, and exporter/training resume have analogous safe-loading calls. The exact missing-manifest ordering claim was not verified here. Safe deserialization and artifact contract validation are distinct stages.
19. **`save_artifact` is not atomic: partly true.** Training checkpoint save uses same-directory temporary file, `fsync`, and `os.replace` (`prismocr/train.py:185`). Artifact publication stages files and rolls back failures (`prismocr/api.py:1641`, `1697`), but does not switch the entire artifact directory in one atomic operation for concurrent readers.

### Docs, CI, and reproducibility claims

20. **Docs say 297 tests and “CI implemented”; count predates demo changes: historical count confirmed, current count not verified.** `docs/implementation-status.md:3` records 297 tests on Python 3.12 CPU. This is not a current test run or current count.
21. **CI runs only Python 3.11: confirmed.** `.github/workflows/ci.yml:9` defines one Ubuntu/Python 3.11 unittest discovery job (`ci.yml:14`), with no Python matrix.
22. **CI has no lint, build, script smoke, CUDA tests: mostly confirmed, with qualification.** No lint, package-build, or CUDA jobs are configured. The verification log records historical synthetic CLI smoke runs (`docs/verification-log.md:20`), so it is inaccurate to say no script smoke has ever occurred.
23. **README and status doc disagree on corpus and milestone 2: not checked conclusively in the source audit.** The implementation status records milestone and corpus status, but a complete semantic comparison against README is still needed before calling this a confirmed contradiction.
24. **`torch>=2.2` conflicts with 2.4.1 lock and code needs: partly confirmed.** `pyproject.toml:10` permits `torch>=2.2`; `requirements-lock.txt:2` pins 2.4.1. The lock/package constraint difference is confirmed; the exact minimum version actually needed by every code path was not established.
25. **Hardcoded `/home/...` paths and real corpus absent from repository limit reproduction: confirmed.** Examples include `prismocr/demo.py:19`, `tests/test_demo.py:94`, and `docs/verification-log.md:31`. The corpus is external to this repository, so exact corpus reproduction requires external data and local path configuration.

### Prior “verified correct” and “not reviewed” lists

The workspace does not contain a separate review checklist with the supplied “verified correct” or “not reviewed” headings. Those labels cannot be authenticated as a previous review artifact. The source audit did not independently execute the following claims:

- Polygon-to-box conversion.
- Vocabulary construction and CTC decoding.
- Sampler and collate behavior.
- EXIF orientation handling.
- Training loop accumulation, scheduler, resume, and RNG behavior.
- Hungarian matching.
- Platt fitting correctness.

The audit did find related implementation and tests, but source presence is not proof of correctness. No test suite was run for this document.

The listed areas still requiring a dedicated review remain:

- Model network, heads, and recognizer.
- `graph_decode.py`.
- `table_decode.py` beyond the bounds noted above.
- `geometry.py`.
- `tiling.py`.
- `benchmark.py`.
- ONNX export numerics.

The earlier implementation status also identifies real-corpus quality, calibration reliability, hardware performance, and trained-checkpoint ONNX acceptance as unproven release evidence.

## 4. Recommended work plan

This sequence addresses immediate exposure and reproducibility risks before another expensive training run. Each step should retain the current artifact and reports as the baseline.

### Phase A: Close environment and demo risks

1. Align `requirements-lock.txt` and the export extra with the ONNX Runtime version checked by `prismocr/onnx_export.py`; add a dependency-consistency check to CI or install instructions.
2. Declare Gradio in an appropriate optional demo dependency group and ensure test collection does not import optional demo dependencies when they are unavailable, or install the demo extra explicitly in the relevant CI job.
3. Change the demo default host to loopback. Require an explicit opt-in for non-loopback binding and configure authentication for shared/network use.
4. Remove arbitrary existing-path loading from `model_name`; resolve only against discovered artifacts under an explicitly configured artifact root.
5. Add model-cache eviction or a fixed allowlist/lifecycle, and return user-safe error codes while keeping detailed exceptions in local server logs.
6. Add an image pixel/dimension cap before expensive decode/inference where feasible, and bound direct table-grid/render helpers, not only the API wrapper.

### Phase B: Resolve training/data contracts

7. Decide whether `strict_ctc_length` is a supported behavior. Either implement its intended difference and test it, or remove it from configuration and resume fingerprints.
8. Trace region and cell supervision from source manifests through data collation to loss masks. Define and test explicit complete, partial, and unavailable states; verify that datasets without such annotations do not train as all-negative pages.
9. Reproduce and test oversize table-label handling. A single bad sample should be surfaced with page/label identity and a deliberate skip/fail policy, not an opaque run abort.
10. Extend the training overlap guard to the same source crop identities/hashes as split tooling, or require and validate a signed/frozen split report at training entry. Add direct tests for all leakage keys.
11. Review composed-page identity and held-out crop exclusion. Ensure all derived/composed samples inherit source identities so group splitting cannot put related crops across splits.
12. Exercise calibration with empty region/relation classes and define a clear unavailable-calibrator result. Keep final quality assessment on an untouched test set; validation-fit thresholds are not independent evidence.
13. Clarify run directory policy: refuse existing output by default or require an explicit overwrite/resume mode, with behavior tested per command.
14. Validate threshold-free metric coordinates against source-frame ground truth with small, explicit geometry cases. Document that it is a checkpoint-selection proxy and quantify any padding/resize effects.

### Phase C: Diagnose model failures before retraining

15. Trace tiled inference end-to-end: enumerate per-tile proposals, coordinate transforms, deduplication, and final limits. Explain why public inference returns thousands of detections while untiled tile calls cap at 128.
16. Build a detector diagnostic that preserves raw heatmap scores, all qualifying candidates before top-k truncation, polygon, source tile, and GT matching. Use a dedicated tuning partition for thresholds and preserve a separate final evaluation set.
17. Investigate PrismOCR recognition's 93.8% blank-frame rate and high blank-line rate. Log target lengths, input frame counts, CTC minimum-frame violations, valid loss counts, per-source supervision, vocabulary coverage, logits, and decoded outputs on a fixed sample.
18. Compare current page-feature recognition and any crop route only if the route faithfully uses trained components. The current model has no trained RGB crop encoder; do not claim an architecture comparison from non-identical inputs.
19. Run a small text-only experiment only after confirming label/CTC construction and a learnable baseline. Stop early if training loss and blank-frame/output rates fail predefined milestones. Do not restart the previous 30-epoch multi-task run yet.
20. After diagnosis, compare against the local PaddleOCR fine-tune and a documented official model where Khmer support exists. Record provenance and training corpus. Do not use validation tuning data as the final test.

### Phase D: Release evidence, docs, and remaining review

21. Update CI to the intended supported Python versions. Add dependency install/import checks, package build, focused script smokes, and CPU tests; add CUDA validation only where a supported runner exists.
22. Reconcile README, implementation status, verification logs, and current test counts. Label historical runs with date, environment, command, and scope; keep source code, runtime evidence, and release claims distinct.
23. Parameterize local corpus paths and document external data acquisition, licenses, checksums, and split construction so another machine can reproduce the real-corpus pipeline.
24. Review network and head implementations, graph decoding, table decoding, geometry, tiling, benchmark methodology, and ONNX numerical parity. Keep trained-artifact ONNX parity separate from synthetic export smoke.
25. Run final evaluation only on the untouched test split after all model and threshold choices are frozen. Report detection, matched-line recognition, end-to-end CER/WER, source/language breakdowns, failure rates, and target CPU latency.

## 5. Evidence files and code references

### Local diagnostic reports

These reports include page identifiers, paths, or transcripts. Keep them local and review dataset license/permission restrictions before sharing.

- Full 643-page PaddleOCR/PrismOCR report: `/home/kosal/data/paddleocr-prismocr-validation.json`
- Controlled 48-page recognition comparison: `/home/kosal/data/paddleocr-prismocr-controlled-48.json`
- 12-page raw-score threshold report: `/home/kosal/data/prismocr-threshold-diagnostic-v4-final2/report.json`
- Candidate-level threshold details: `/home/kosal/data/prismocr-threshold-diagnostic-v4-final2/candidates.jsonl`
- Bounded inference/CTC diagnostic summary: `/tmp/prismocr-next-diagnosis/summary.json`
- Earlier 12-page stratified smoke: `/tmp/paddleocr-prismocr-stratified-smoke.json`

### Added comparison utilities

- `scripts/compare_paddleocr.py` supports bounded smoke and full validation selection.
- `scripts/compare_gt_crops_controlled.py` runs the bounded 48-page recognition follow-up.
- `/tmp/prismocr_threshold_diagnostic.py` generated the raw-score/proposal-cap report; it is outside the repository.

The comparison scripts are uncommitted workspace changes. Syntax compilation and `git diff --check` passed during the work. This document adds no application behavior. No general test suite was run for this report.
