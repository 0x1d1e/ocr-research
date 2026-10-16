# PrismOCR v4 research implementation

This repository implements a research-stage PyTorch version of the v4 model and data contracts in [`PrismOCR_unified_v4_specification.html`](PrismOCR_unified_v4_specification.html). It has no permissioned real OCR corpus or validated release checkpoint. Synthetic training and smoke tests do not establish Khmer/English OCR quality or performance. Historical v3/v3.1 material in the specification does not establish v4 results.

## Install and verify

Python 3.11+, PyTorch 2.2+, NumPy, and Pillow are required.

```sh
python -m pip install -e .
python -m prismocr.verify
```

The tests cover geometry, training and inference contracts, synthetic annotations, table safety, checkpoint resume, and the pinned opset-18 ONNX Runtime CPU path. ONNX checks use synthetic weights and inputs; they do not establish document accuracy, real-corpus convergence, target-CPU latency, or release acceptance.

## Train and evaluate

Prepare a JSONL manifest with one record per page. Each record has `image_path` plus either an inline v4 annotation under `annotation` or an `annotation_path` relative to the manifest. The annotation schema and polygon coordinate rules are implemented in `prismocr.data.schema`.

```sh
prismocr-train --manifest data/train.jsonl --validation-manifest data/validation.jsonl \
  --output artifacts/run-001 --config configs/v4_detection.toml --device cuda --epochs 100
prismocr-train --manifest data/train.jsonl --validation-manifest data/validation.jsonl \
  --output artifacts/run-001 --config configs/v4_detection.toml --device cuda --epochs 100 \
  --resume artifacts/run-001/training-state.pt
prismocr-eval --manifest data/heldout.jsonl --artifact artifacts/run-001 \
  --output artifacts/run-001/metrics.json
```

Long training and corpus builds are CPU-parallel. Cap them to the cores you own by launching under an affinity mask, for example `taskset -c 0-11 python -m prismocr.train ...` or `taskset -c 0-11 python -m prismocr.data.tools ...`; the mask bounds the parent process, its data-loader workers, and the shard pool together. Keep `training.num_workers` below the mask width so the training process keeps CPU headroom.

Use `prismocr.data.tools` to convert supported ICDAR quadrilateral annotations, audit schema/provenance/checksums and duplicate candidates, and create deterministic document/template-family/image-grouped train/validation/test manifests:

```sh
python -m prismocr.data.tools convert-icdar --images data/images --ground-truth data/gt \
  --output data/converted.jsonl --document-prefix icdar/train --license CC-BY-4.0 \
  --permission-reference dataset-license --reviewer qa-person
python -m prismocr.data.tools qa --manifest data/converted.jsonl --output data/qa.json
python -m prismocr.data.tools split --manifest data/converted.jsonl --output-dir data/splits --seed 42
```

Conversion marks output as requiring human label QA; perceptual-hash matches are review candidates, not proof of duplicates. Keep train, validation, and held-out splits document-grouped and review split reports before training. Validation after each epoch selects the checkpoint by inference-failure count, then line-detection recall, then CER, with exact ties favoring the earlier epoch; failure count is ranked first so a page-failing epoch cannot win on the zero-denominator recall/CER that a manifest without ground-truth lines reports. `training-state.pt` is an atomic epoch-boundary resume checkpoint. Resume requires the same training/validation image and annotation contents, vocabulary, effective model/training settings, batch size, target epoch count, and output directory. The startup compatibility check hashes all input image bytes, which adds one sequential read of each dataset. The training command derives vocabulary from the training manifest unless a frozen vocabulary is supplied with `--vocabulary`. Training supports configurable loss weights, optional CUDA AMP, gradient accumulation, and backbone activation checkpointing. Run the release-shaped CPU workload protocol against a representative manifest with `prismocr-benchmark --artifact artifacts/run-001 --manifest data/benchmark.jsonl`; it requires at least 30 warmups and 100 timed iterations per workload. `--image` remains a single-page diagnostic, not release evidence.

A deterministic synthetic fixture generator is available as `prismocr.data.synthetic.generate_synthetic_dataset`. It requires caller-supplied text and a Pillow font; no font or corpus is bundled. The utility creates simple black-on-white pages and axis-aligned annotations for pipeline tests only. It does not simulate realistic Khmer rendering or provide meaningful OCR training data.

## Prediction

```python
from prismocr import InferenceOptions, PrismOCR

engine = PrismOCR.load("artifacts/run-001", device="cpu")
result = engine.predict(image, InferenceOptions(return_debug=True)).to_dict()
```

Scores remain `null` unless a model-bound calibration artifact is supplied. The API handles one page per call and supports overlapping high-resolution tiling with overflow retries and cross-tile stitching. It always decodes typed `NEXT`, `PARENT`, `KV_VALUE`, `RIGHT_CELL`, and `PARENT`-derived `CONTAINS` relations, applying validation-fitted calibrators to relation and detector scores when a bound calibration artifact is present, and falls back to geometry only when the relation head supplies no usable scores. Independent per-column reading-order scopes and `RIGHT_CELL`-informed table placement are implemented. `emit_character_confidence=True` reports uncalibrated greedy-CTC frame posteriors, not calibrated character probabilities. `page_resolution` can resize a page and remove small text detail.

## Project milestones

The engineering delivery is delineated into four progressive milestones:

- **Milestone 1: Corpus Built** (conversion, composition, shard caching, and split tools implemented and tested; local raw files staged)
- **Milestone 2: Labels QA'd** (human verification of real scene annotations and upstream licenses; pending full human audit)
- **Milestone 3: Model Trained** (architecture with fused text head, anchor supervision, CTC warmup, and cosine schedule implemented; awaiting full-scale training run)
- **Milestone 4: Model Validated** (release checkpoint evaluated against human-audited real test set and hardware latency profile)

## Scope and release status

The code includes oriented geometry, proposal and region heads, sparse compact CTC recognition, typed graph decoding, learned row/column/span/header heads, `RIGHT_CELL`-guided table reconstruction, high-resolution tiling, corpus conversion and grouped-split tools, validation-only Platt calibration fitting with reliability reports, structured evaluation and benchmarking, and training/evaluation/export entry points. `python -m prismocr.verify` runs the full software suite.

There is no permissioned Khmer/English corpus, trained v3 control, held-out result, target-CPU latency report, or release-qualified ONNX artifact, so no release gate is claimed. Validation-only calibration fitting, structured evaluation, and canonical decoded-document PyTorch/ONNX parity are implemented and covered by synthetic tests, but synthetic evidence does not establish real-corpus quality, calibration reliability, or deployment acceptance. The pinned CPU exporter checks tensor inventory, fusion parity, recognizer buckets, and decoded-document parity on synthetic weights only. Do not treat example or synthetic artifacts as deployable OCR. See [`docs/implementation-status.md`](docs/implementation-status.md) for the current evidence boundary, [`docs/spec-review.md`](docs/spec-review.md) for unresolved specification choices, and [`docs/verification-log.md`](docs/verification-log.md) for the checks that were actually run.
