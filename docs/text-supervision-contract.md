# Text supervision availability — contract v1

Stage 1 only: annotation availability, valid image extent, loss, evaluation and
calibration. This does not change model geometry, bucket selection, saved-model
configuration, public inference, or historical artifact loading.

## Domain and shared seam

`supervision.lines` is one of:

- `complete`: the provided `lines` array is the complete text annotation. An
  explicit empty array certifies a text-blank page; background negatives are valid.
- `partial`: nonempty positive annotations are provided, but omitted text may
  exist. Positive heatmap anchors, roles, geometry and CTC remain usable; no
  background negatives or inferred terminal/adjacency NEXT decisions are valid.
  Explicit relation decisions remain usable.
- `unavailable`: no text supervision; `lines` must be empty or absent. Region
  supervision does not establish text absence.

`prismocr.data.schema.resolve_line_supervision` is the canonical resolver used by
schema, dataset/collate, loss, evaluator and calibration. Invalid states, boolean
states, unavailable-with-lines, partial-without-positives, and complete-without-an
explicit `lines` array fail validation. There is no document-ID heuristic.

Compatibility: unmarked **nonempty** legacy lines retain the historical
`complete` interpretation. This is an explicit compatibility assumption, **not
proof of annotation exhaustiveness**. Unmarked empty/missing legacy lines resolve
to `unavailable`, never certified negative. Existing inline/path JSONL files are
read without mutation. Raw `_annotation` and `dataset_content_sha256` are unchanged;
resolved state is materialized on target copies only. To certify a legacy blank,
produce a separately reviewed/new manifest with explicit complete/empty labels.

New generated/composed pages (including genuine generated blanks) carry complete;
COCO layout carries unavailable. ICDAR with omitted `###` regions carries partial
(or unavailable if no usable positives). Scene pages with dropped unsupported,
empty-transcript, malformed or degenerate labels carry partial; pages with no
omissions carry complete. These claims still depend on source annotation QA.

## Valid extent and exact loss

Collation retains `_valid_image_extent = (valid_width / padded_width,
valid_height / padded_height)` after resizing. Grid cells are eligible when their
top-left grid coordinate is inside this extent (`x/W < extent_x`, `y/H < extent_y`).
Thus a boundary cell intersecting the page is retained, while cells wholly in
right/bottom stride/batch padding are excluded. No extra user-defined spatial
supervision mask is introduced; invalid extents/masks fail closed.

For each eligible page i, A is the integer-anchor set and N the valid non-anchor
cells **only on complete pages**. p = clamp(sigmoid(logit), 1e-6, 1-1e-6); g is the
existing Gaussian target. The heatmap loss is:

```
P_i = sum[a in A] -log(p_a) * (1-p_a)^2 / max(1, |A|)
N_i = sum[n in N] -log(1-p_n) * p_n^2 * (1-g_n)^4 / max(1, |N|)
L_heatmap = sum[i eligible] (P_i + N_i) / max(1, eligible_page_count)
```

Complete and partial pages are eligible; unavailable pages are not. Partial has
N empty. Certified blank has A empty, N valid image cells. All-unavailable gives
exact differentiable zero with zero heatmap gradient. Adding unavailable pages
does not dilute the text heatmap or role loss; region loss remains independent.
Role normalization uses the count of complete/partial pages, boxes/CTC retain
existing positive-sample means. No unavailable page supplies text geometry,
role or CTC labels.

**Tradeoff:** this changes known-complete positive-page weighting too: the old
CenterNet-like objective divided the entire negative sum by the anchor count;
this objective separately averages positive and negative classes per page.
Negative weight is therefore reduced by the old valid-negative/anchor ratio,
and pages rather than all batch anchors have equal weight. This deliberately
avoids the grid-size cliff for blank batches and padding-dependent scaling, but
is not a causal-mask-only intervention. Later experiments must check both misses
and excess emissions. Overemission warrants revisiting normalization, not hiding
it by threshold tuning. No quality/convergence claim is made here.

## Evaluation and calibration

Only complete pages enter official line detection, CER/WER, reading-order,
language/Khmer slices, or text detector calibration labels. Partial pages are
conservatively excluded, not assigned false complete-set accuracy. Complete blank
emissions are false-positive detections and transcript insertions. Excluded
pages retain emitted-line/character counts and annotated-positive counts in
per-page diagnostics, and all public emitted lines remain in aggregate emission
counts. Reports expose complete/partial/unavailable page counts and contract ID.
`text_metrics_defined` is false when no complete pages exist; existing raw empty
rates remain compatibility sentinels with undefined/zero-denominator flags, not
measured success. Blank-only CER has no reference denominator and is flagged even
when insertion counts are nonzero.

Text-inclusive TEDS requires complete lines or independently provided direct cell
transcripts; structural TEDS remains independently evaluable. KV text metrics are
excluded on noncomplete text pages. Existing closed-world region/cell detection,
region background loss, missing-KV-family policy, and ambiguous-table metric policy
are **not** generalized into new task masks by this repair; they remain separate
pre-existing limitations. Direct cell text completeness is likewise an existing
annotation assumption.

Calibration fitting excludes noncomplete text emissions and reports their counts;
certified text blanks count even on pages with region labels. Fits still require
both positive and negative proposals and retain all existing provenance checks.
Historical calibration sidecars/artifacts are not rewritten or silently refit;
newly fitted reports identify this availability contract. Old reports must not be
compared as though they had identical eligibility/normalization.

## Resume and scope

Training resolved settings include
`objective_contract = lines-availability-v1-page-balanced-focal`. Existing checkpoint
compatibility equality therefore rejects optimizer resume from old loss semantics,
even though raw dataset fingerprints intentionally remain identical. Model weight
loading/inference/configuration is unchanged. This stage includes no GPU training,
model geometry/bucket repairs, experiment configuration or artifact writes.
