# PrismOCR v4 specification review

## Verdict

The specification is a useful architecture and release contract, but it is not a complete build-ready implementation recipe. It clearly separates v4 decisions from the v3.1 historical annex, corrects assignment coverage versus geometric detection recall, and makes sparse OCR, Unicode reporting, graph validity, and renderer safety explicit. Several important choices still require a pinned convention or corpus evidence. The implementation makes conservative choices where possible and records unavailable evidence rather than presenting targets as achieved results.

## Findings that affect implementation

1. **Rotated-box coordinate convention is incomplete.** The six-value box is normalized center/size plus `sinθ, cosθ`, but the relationship between normalized x/y coordinates, pixel-space angle, page aspect ratio, and oriented IoU is not fixed. The implementation treats angle as a pixel-space angle, uses page width/height during geometry and sampling, and canonicalizes antipodal orientation to the dataset's `[-π/2, π/2)` scan axis so geometrically equivalent boxes cannot reverse a transcript. Dataset converters must preserve this convention.
2. **Assignment and accepted detection are different states.** The spec requires one assignment reused by downstream consumers, while also forbidding distant assignments from counting as detections or receiving unrelated OCR text. A MatchSet therefore retains all Hungarian assignments for auxiliary box/role learning, but separately records IoU-accepted pairs. CTC, detection recall, and graph endpoint mapping use only accepted pairs, and detection quality uses source-polygon IoU rather than boxes derived from the polygons.
3. **Proposal validity needs two masks.** Training requires low-confidence candidates to participate in assignment, but inference only runs OCR for threshold-accepted text. The model exposes candidate and accepted masks separately. A flat 0-logit map has score 0.5 and produces no accepted text at the default threshold.
4. **NULL and unannotated relations need distinct labels.** An explicit NULL relation trains the NULL candidate; an unannotated edge is ignored. Inference resolves each source to at most one target per relation, including the NULL candidate, and reports suppression reason codes.
5. **Data-dependent defaults are not available.** Kt/Kr, score thresholds, sequence buckets, vocabulary contents, and quality deltas depend on corpus percentiles or validation experiments. Shipped values are configurable candidates, not locked release values. Scores remain uncalibrated until a validation calibration artifact is bound to the artifact.
6. **The historical baseline is not reproducible from this directory.** Annex A contains v3.1 source and reported toy assertions, but the v4 specification explicitly says those are not a validated v4 baseline. There is no dataset, trained checkpoint, split manifest, or reference runtime result here. M0 and all comparative release gates remain blocked on those inputs.
7. **Grid inference is under-specified for raw cell boxes.** The contract requires a constrained grid solver and structured conflict states, but leaves row/column track thresholds and learned-versus-geometric evidence to experiments. The implementation prefers learned row/column indices and learned `RIGHT_CELL` affinities, uses a gap-tolerant geometric solver only as a fallback, validates occupancy, and refuses conflicts; inferred grid quality is not claimed.
8. **Export acceptance cannot be established from an untrained model.** The contract allows a staged ONNX encoder plus deterministic host decoder when ragged inference is unsupported. A complete, version-pinned operator matrix and PyTorch/ONNX parity test need trained artifacts and target-runtime execution; the synthetic parity run is a code check only.
9. **"Table-only; ignore missing table labels" is ambiguous.** The specification's `L_grid` row says grid supervision should ignore missing table labels, while the canonical JSON always carries `row`/`col`/`rowspan`/`colspan`. The implementation supervises any labelled cell, defaults an absent `rowspan`/`colspan` to the mandatory value `1`, and excludes cells whose table is flagged `annotation_ambiguous`. Treat this reading as an implementation choice to revisit if a locked label policy is published.

## Conservative implementation choices

- Preserve the original HTML specification unchanged; treat its v4 body as normative and Annex A as historical.
- Use pixel-space rotated geometry with aspect-aware conversion and canonical orientation normalization.
- Keep assignment pairs, quality-accepted pairs, and unassigned/missed GT counts separately visible.
- Do not fabricate confidence calibration. Public prediction scores are `null` unless a validation-fitted calibration artifact is bound to the loaded model, vocabulary, split, and IoU threshold.
- Preserve Unicode verbatim by default. OOV characters map to an explicit UNK token and decode visibly as U+FFFD with positions.
- Treat table span conflicts as errors; escape OCR text before HTML rendering.
- Rank valid no-sample rates as undefined in reports (`metrics_are_defined`, `zero_denominator_metrics`) and rank inference failures ahead of recall when selecting a checkpoint, so vacuous zero-denominator metrics cannot be read as success.
- Validate annotations against the same relation rules the decoder applies, so accepted supervision is never something inference would reject.
- Do not label a synthetic smoke test as document accuracy, training success, or a release benchmark.

## Remaining evidence needed

A permissioned, document-grouped Khmer/English corpus; converter and label QA; a reproducible v3 control; validation-fitted vocabulary and thresholds; repeated training runs; quality and latency ablations; a frozen final split; 6 GB GPU measurements; FP32 tensor-byte inventory on the selected fused artifact; and CPU PyTorch/ONNX parity on a pinned runtime matrix.
