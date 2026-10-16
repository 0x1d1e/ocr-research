"""Run the v4 CPU inference benchmark protocol on a local image or manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image

import prismocr.api as api_module
from prismocr.api import InferenceOptions, PrismOCR
from prismocr.data.dataset import PrismPageDataset

MINIMUM_WARMUP = 30
MINIMUM_ITERATIONS = 100
LINE_COUNT_BUCKETS = (
    "blank",
    "lines_1_10",
    "lines_11_30",
    "lines_31_64",
    "lines_65_96",
    "lines_over_96",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def line_count_bucket(count: int) -> str:
    if count < 0:
        raise ValueError("line count must be nonnegative")
    if count == 0:
        return "blank"
    if count <= 10:
        return "lines_1_10"
    if count <= 30:
        return "lines_11_30"
    if count <= 64:
        return "lines_31_64"
    if count <= 96:
        return "lines_65_96"
    return "lines_over_96"


def _manifest_workloads(dataset: PrismPageDataset) -> list[dict[str, Any]]:
    indexed_records = list(enumerate(dataset.records))
    line_groups: dict[str, list[tuple[int, dict[str, Any]]]] = {
        bucket: [] for bucket in LINE_COUNT_BUCKETS
    }
    table_groups: dict[str, list[tuple[int, dict[str, Any]]]] = {
        "with_tables": [], "without_tables": []
    }
    resolution_groups: dict[str, list[tuple[int, dict[str, Any]]]] = {
        "square_768": [],
        "square_768_plus": [],
        "tiled_high_resolution": [],
    }

    for index, record in indexed_records:
        annotation = record["_annotation"]
        image = annotation["image"]
        width, height = int(image["width"]), int(image["height"])
        lines = annotation.get("lines", ())
        line_groups[line_count_bucket(len(lines))].append((index, record))
        has_table = any(
            region.get("kind") == "TABLE"
            for region in annotation.get("regions", ())
        )
        table_groups["with_tables" if has_table else "without_tables"].append((index, record))
        if width == 768 and height == 768:
            resolution_groups["square_768"].append((index, record))
        if width >= 768 and height >= 768:
            resolution_groups["square_768_plus"].append((index, record))
        if max(width, height) > 1024:
            resolution_groups["tiled_high_resolution"].append((index, record))

    workloads: list[dict[str, Any]] = []
    for bucket in LINE_COUNT_BUCKETS:
        workloads.append({"category": "line_count", "bucket": bucket, "records": line_groups[bucket]})
    for bucket, records in table_groups.items():
        workloads.append({"category": "table_presence", "bucket": bucket, "records": records})
    for bucket, records in resolution_groups.items():
        workloads.append({"category": "resolution", "bucket": bucket, "records": records})
    return workloads


def _latency_summary(values_ms: list[float]) -> dict[str, float]:
    if not values_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "minimum_ms": 0.0, "maximum_ms": 0.0}
    ordered = sorted(values_ms)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[p95_index],
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
    }


def _cpu_metadata(threads: int) -> dict[str, Any]:
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    governor_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        governor = governor_path.read_text(encoding="utf-8").strip()
    except OSError:
        governor = None
    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "pillow": Image.__version__,
        "device": "cpu",
        "precision": "float32",
        "batch_size": 1,
        "threads": threads,
        "cpu_affinity": affinity,
        "cpu_governor": governor,
        "power_profile": governor,
    }


class _StageInstrumentation:
    """Collect inclusive stage durations during sequential public API calls."""

    def __init__(self, engine: PrismOCR) -> None:
        self.engine = engine
        self.capture: dict[str, Any] | None = None
        self._module_stacks: dict[int, list[int]] = defaultdict(list)
        self._hooks: list[Any] = []
        self._instance_attributes: dict[str, Any] = {}
        self._module_attributes: dict[str, Any] = {}
        self._recognizer_bucket_method: Any = None
        self._recognizer_bucket_was_instance_attribute = False

    def _record(self, stage: str, elapsed_ns: int) -> None:
        if self.capture is not None:
            self.capture["stage_ns"][stage] += elapsed_ns

    def _wrap_instance(self, name: str, stage: str) -> None:
        original = getattr(self.engine, name)
        self._instance_attributes[name] = original

        def timed(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                self._record(stage, time.perf_counter_ns() - started)

        setattr(self.engine, name, timed)

    def _wrap_api_function(self, name: str, stage: str) -> None:
        original = getattr(api_module, name)
        self._module_attributes[name] = original

        def timed(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                self._record(stage, time.perf_counter_ns() - started)

        setattr(api_module, name, timed)

    def _module_hook(self, name: str, *, model_output: bool = False):
        def before(module: torch.nn.Module, _inputs: tuple[Any, ...]) -> None:
            if self.capture is not None:
                self._module_stacks[id(module)].append(time.perf_counter_ns())

        def after(
            module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            if self.capture is None:
                return
            stack = self._module_stacks[id(module)]
            if stack:
                self._record(name, time.perf_counter_ns() - stack.pop())
            if model_output and isinstance(output, dict):
                text = output.get("text", {})
                regions = output.get("regions", {})
                text_mask = text.get("mask") if isinstance(text, dict) else None
                region_mask = regions.get("mask") if isinstance(regions, dict) else None
                if isinstance(text_mask, torch.Tensor):
                    self.capture["query_count"] += int(text_mask.sum().item())
                if isinstance(region_mask, torch.Tensor):
                    self.capture["query_count"] += int(region_mask.sum().item())
                groups = output.get("ocr_groups", {})
                if isinstance(groups, dict):
                    for bucket, group in groups.items():
                        refs = group.get("refs") if isinstance(group, dict) else None
                        count = int(refs.shape[0]) if isinstance(refs, torch.Tensor) else 0
                        self.capture["selected_ocr_lines"] += count
                        self.capture["recognizer_bucket_counts"][str(bucket)] += count

        return before, after

    def __enter__(self) -> "_StageInstrumentation":
        self._wrap_instance("_prepare_image", "preprocessing")
        self._wrap_instance("_tile_output", "tile_processing")
        for name, stage in (
            ("stitch_tile_results", "tile_stitching"),
            ("decode_graph", "graph_decode"),
            ("reconstruct_tables", "table_decode"),
        ):
            self._wrap_api_function(name, stage)

        recognizer = self.engine.model.recognizer
        self._recognizer_bucket_method = recognizer.forward_bucket
        self._recognizer_bucket_was_instance_attribute = (
            "forward_bucket" in vars(recognizer)
        )

        def timed_forward_bucket(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter_ns()
            try:
                return self._recognizer_bucket_method(*args, **kwargs)
            finally:
                self._record("recognizer", time.perf_counter_ns() - started)

        recognizer.forward_bucket = timed_forward_bucket
        modules = (
            ("model_forward", self.engine.model, True),
            ("backbone", self.engine.model.backbone, False),
            ("text_detector", self.engine.model.text_head, False),
            ("region_detector", self.engine.model.region_head, False),
            ("relation_head", self.engine.model.relation_head, False),
        )
        for name, module, model_output in modules:
            before, after = self._module_hook(name, model_output=model_output)
            self._hooks.append(module.register_forward_pre_hook(before))
            self._hooks.append(module.register_forward_hook(after))
        return self

    def __exit__(self, *_exc: Any) -> None:
        for hook in self._hooks:
            hook.remove()
        for name in self._instance_attributes:
            delattr(self.engine, name)
        recognizer = self.engine.model.recognizer
        if self._recognizer_bucket_was_instance_attribute:
            recognizer.forward_bucket = self._recognizer_bucket_method
        else:
            delattr(recognizer, "forward_bucket")
        for name, original in self._module_attributes.items():
            setattr(api_module, name, original)


def _run_page(
    engine: PrismOCR,
    image_path: Path,
    *,
    page_resolution: int | None,
    instrumentation: _StageInstrumentation,
    capture: bool,
) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "stage_ns": defaultdict(int),
        "recognizer_bucket_counts": Counter(),
        "selected_ocr_lines": 0,
        "query_count": 0,
        "warning_counts": Counter(),
        "error": None,
        "result": None,
    }
    instrumentation.capture = sample if capture else None
    total_started = time.perf_counter_ns()
    try:
        load_started = time.perf_counter_ns()
        with Image.open(image_path) as source:
            source.load()
            image = source.convert("RGB").copy()
        original_width, original_height = image.size
        scale = (
            min(1.0, page_resolution / max(original_width, original_height))
            if page_resolution is not None else 1.0
        )
        sample["input_size_px"] = [original_width, original_height]
        sample["effective_size_px"] = [
            max(1, round(original_width * scale)),
            max(1, round(original_height * scale)),
        ]
        sample["stage_ns"]["image_loading"] += time.perf_counter_ns() - load_started
        result = engine.predict(image, InferenceOptions(page_resolution=page_resolution)).to_dict()
        sample["result"] = result
        diagnostics = result.get("diagnostics", {})
        sample["executed_tile_count"] = int(diagnostics.get("tiles", 0))
        sample["executed_tile_model_calls"] = int(diagnostics.get("tile_model_calls", 0))
        for warning in result.get("warnings", ()):
            code = warning.get("code")
            if isinstance(code, str):
                sample["warning_counts"][code] += 1
    except Exception as error:
        sample["error"] = {
            "stage": getattr(error, "stage", "benchmark_inference"),
            "type": getattr(error, "cause_type", type(error).__name__),
        }
    finally:
        sample["stage_ns"]["end_to_end"] += time.perf_counter_ns() - total_started
        instrumentation.capture = None
    return sample


def _benchmark_workload(
    engine: PrismOCR,
    records: list[tuple[int, dict[str, Any]]],
    *,
    warmup: int,
    iterations: int,
    page_resolution: int | None,
    instrumentation: _StageInstrumentation,
    category: str,
    bucket: str,
) -> dict[str, Any]:
    samples = [record for _, record in records]
    warmup_failures = 0
    for index in range(warmup):
        run = _run_page(
            engine,
            samples[index % len(samples)]["_image_path"],
            page_resolution=page_resolution,
            instrumentation=instrumentation,
            capture=False,
        )
        warmup_failures += run["error"] is not None

    measured_indices = [
        min(len(samples) - 1, int((index + 0.5) * len(samples) / iterations))
        for index in range(iterations)
    ]
    measured: list[dict[str, Any]] = []
    for sample_index in measured_indices:
        record = samples[sample_index]
        run = _run_page(
            engine,
            record["_image_path"],
            page_resolution=page_resolution,
            instrumentation=instrumentation,
            capture=True,
        )
        run["page_id"] = record["_annotation"]["page_id"]
        measured.append(run)
    measured_sample_indices = sorted(set(measured_indices))
    measured_samples = [samples[index] for index in measured_sample_indices]

    stage_values: dict[str, list[float]] = defaultdict(list)
    stage_execution_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    selected_ocr_lines = query_count = predicted_lines = predicted_regions = 0
    candidate_count = tile_count = model_calls = 0
    error_records = []
    for index, run in enumerate(measured):
        for stage, duration_ns in run["stage_ns"].items():
            stage_values[stage].append(duration_ns / 1_000_000)
            stage_execution_counts[stage] += 1
        warning_counts.update(run["warning_counts"])
        bucket_counts.update(run["recognizer_bucket_counts"])
        selected_ocr_lines += run["selected_ocr_lines"]
        query_count += run["query_count"]
        if run["error"] is not None:
            error_records.append({"iteration": index, **run["error"]})
            continue
        result = run["result"]
        predicted_lines += len(result.get("lines", ()))
        predicted_regions += len(result.get("regions", ())) + len(result.get("cells", ()))
        diagnostics = result.get("diagnostics", {})
        candidate_count += int(diagnostics.get("candidate_count", 0))
        tile_count += int(diagnostics.get("tiles", 0))
        model_calls += int(diagnostics.get("tile_model_calls", 0))

    durations = stage_values.get("end_to_end", [])
    total_seconds = sum(durations) / 1000.0
    annotation_line_count = sum(
        len(record["_annotation"].get("lines", ())) for record in measured_samples
    )
    annotated_tables = sum(
        any(region.get("kind") == "TABLE" for region in record["_annotation"].get("regions", ()))
        for record in measured_samples
    )
    execution_issues: list[dict[str, Any]] = []
    successful_runs = [run for run in measured if run["error"] is None]
    if category == "line_count" and bucket != "blank" and selected_ocr_lines == 0:
        execution_issues.append({
            "code": "NO_RECOGNIZER_EXECUTION",
            "message": "nonblank line-count workload produced no recognizer inputs",
        })
    if category == "table_presence" and bucket == "with_tables":
        if not any(run["result"].get("tables") for run in successful_runs):
            execution_issues.append({
                "code": "NO_TABLE_DECODE_OUTPUT",
                "message": "table-presence workload produced no decoded tables",
            })
    if category == "resolution":
        for run in successful_runs:
            width, height = run.get("effective_size_px", (0, 0))
            page_id = run.get("page_id")
            if bucket == "square_768" and (width, height) != (768, 768):
                execution_issues.append({
                    "code": "RESOLUTION_WORKLOAD_NOT_EXECUTED",
                    "page_id": page_id,
                    "expected": "768x768 effective inference image",
                    "actual": [width, height],
                })
            elif bucket == "square_768_plus" and (width < 768 or height < 768):
                execution_issues.append({
                    "code": "RESOLUTION_WORKLOAD_NOT_EXECUTED",
                    "page_id": page_id,
                    "expected": "both effective dimensions at least 768",
                    "actual": [width, height],
                })
            elif bucket == "tiled_high_resolution" and (
                max(width, height) <= 1024 or run.get("executed_tile_count", 0) <= 1
            ):
                execution_issues.append({
                    "code": "TILED_HIGH_RESOLUTION_NOT_EXECUTED",
                    "page_id": page_id,
                    "effective_size_px": [width, height],
                    "tile_count": run.get("executed_tile_count", 0),
                })
    if category == "resolution" and not successful_runs:
        execution_issues.append({
            "code": "RESOLUTION_WORKLOAD_NOT_EXECUTED",
            "message": "no successful timed run is available to verify effective resolution",
        })
    return {
        "category": category,
        "bucket": bucket,
        "page_count": len(measured_samples),
        "source_page_count": len(samples),
        "sample_page_ids": [record["_annotation"]["page_id"] for record in measured_samples],
        "sample_image_sha256": [record["_image_sha256"] for record in measured_samples],
        "ground_truth_line_count_min": min(
            len(record["_annotation"].get("lines", ())) for record in measured_samples
        ),
        "ground_truth_line_count_max": max(
            len(record["_annotation"].get("lines", ())) for record in measured_samples
        ),
        "pages_with_tables": annotated_tables,
        "warmup_runs": warmup,
        "measured_runs": iterations,
        "warmup_failures": warmup_failures,
        "inference_failures": error_records,
        "execution_issues": execution_issues,
        "status": "failed" if warmup_failures or error_records or execution_issues else "measured",
        "latency_ms": {
            stage: _latency_summary(values)
            for stage, values in sorted(stage_values.items())
        },
        "stage_execution_counts": dict(sorted(stage_execution_counts.items())),
        "stage_latency_scope": (
            "Each stage summary is conditional on attempts that executed that stage. "
            "Unexecuted stages are excluded, not counted as zero; compare the execution "
            "count with measured_runs."
        ),
        "throughput_pages_per_second": iterations / total_seconds if total_seconds > 0 else 0.0,
        "mean_selected_ocr_lines_per_page": selected_ocr_lines / iterations,
        "recognizer_bucket_counts": dict(sorted(bucket_counts.items())),
        "mean_query_count_per_page": query_count / iterations,
        "mean_candidate_count_per_page": candidate_count / iterations,
        "mean_predicted_lines_per_page": predicted_lines / iterations,
        "mean_predicted_regions_and_cells_per_page": predicted_regions / iterations,
        "mean_tile_count_per_page": tile_count / iterations,
        "mean_tile_model_calls_per_page": model_calls / iterations,
        "warning_counts": dict(sorted(warning_counts.items())),
        "ground_truth_line_count_across_samples": annotation_line_count,
    }


def _single_image_workload(
    engine: PrismOCR,
    image_path: Path,
    *,
    warmup: int,
    iterations: int,
    page_resolution: int | None,
    instrumentation: _StageInstrumentation,
) -> dict[str, Any]:
    record = {
        "_image_path": image_path,
        "_image_sha256": _sha256_file(image_path),
        "_annotation": {"page_id": image_path.name, "lines": [], "regions": []},
    }
    return _benchmark_workload(
        engine,
        [(0, record)],
        warmup=warmup,
        iterations=iterations,
        page_resolution=page_resolution,
        instrumentation=instrumentation,
        category="single_image_diagnostic",
        bucket="unspecified_workload",
    )


def benchmark(
    artifact: str | Path,
    image_path: str | Path | None = None,
    *,
    manifest: str | Path | None = None,
    warmup: int = MINIMUM_WARMUP,
    iterations: int = MINIMUM_ITERATIONS,
    page_resolution: int | None = None,
    threads: int = 1,
) -> dict[str, Any]:
    if (image_path is None) == (manifest is None):
        raise ValueError("provide exactly one of image_path or manifest")
    if warmup < 0 or iterations <= 0 or threads <= 0:
        raise ValueError("warmup must be nonnegative; iterations and threads must be positive")
    if manifest is not None and (warmup < MINIMUM_WARMUP or iterations < MINIMUM_ITERATIONS):
        raise ValueError(
            f"manifest benchmarking requires at least {MINIMUM_WARMUP} warmups and "
            f"{MINIMUM_ITERATIONS} timed iterations per workload"
        )
    if page_resolution is not None and page_resolution <= 0:
        raise ValueError("page_resolution must be positive")

    torch.set_num_threads(threads)
    engine = PrismOCR.load(artifact, device="cpu")
    root = Path(artifact)
    model_path = root / "model.pt"
    model_sha256 = _sha256_file(model_path) if model_path.is_file() else None
    manifest_sha256 = _sha256_file(Path(manifest).resolve()) if manifest is not None else None
    with _StageInstrumentation(engine) as instrumentation:
        if manifest is not None:
            dataset = PrismPageDataset(manifest)
            if not dataset.records:
                raise ValueError("benchmark manifest contains no pages")
            workloads = _manifest_workloads(dataset)
            workload_results = []
            missing_workloads = []
            for workload in workloads:
                records = workload["records"]
                if not records:
                    missing_workloads.append({
                        "category": workload["category"], "bucket": workload["bucket"]
                    })
                    continue
                workload_results.append(_benchmark_workload(
                    engine,
                    records,
                    warmup=warmup,
                    iterations=iterations,
                    page_resolution=page_resolution,
                    instrumentation=instrumentation,
                    category=workload["category"],
                    bucket=workload["bucket"],
                ))
        else:
            dataset = None
            workload_results = [_single_image_workload(
                engine,
                Path(image_path).resolve(),
                warmup=warmup,
                iterations=iterations,
                page_resolution=page_resolution,
                instrumentation=instrumentation,
            )]
            missing_workloads = []

    release_cases = [workload for workload in workload_results if workload["category"] != "single_image_diagnostic"]
    all_ran = bool(release_cases) and not missing_workloads and all(
        workload["status"] == "measured" for workload in release_cases
    )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != "Darwin":
        rss *= 1024
    return {
        "schema_version": "4.0",
        "measurement_kind": "representative_manifest_workloads" if manifest is not None else "single_page_diagnostic",
        "status": "protocol_measured_not_release_acceptance" if all_ran else "incomplete_workload_or_failed_inference",
        "artifact": str(root.resolve()),
        "model_version": engine.model_version,
        "model_sha256": model_sha256,
        "source_manifest": None if manifest is None else str(Path(manifest).resolve()),
        "source_manifest_sha256": manifest_sha256,
        "runtime": _cpu_metadata(threads),
        "protocol": {
            "minimum_warmup_runs": MINIMUM_WARMUP,
            "minimum_measured_runs": MINIMUM_ITERATIONS,
            "minimums_met": warmup >= MINIMUM_WARMUP and iterations >= MINIMUM_ITERATIONS,
            "page_resolution": page_resolution,
            "tiling_enabled": True,
            "stage_timing_scope": "stages are inclusive and may overlap; do not sum nested model and tile timings",
        },
        "recognizer_variant": engine.model.recognizer.variant,
        "inference_parameter_bytes": engine.model.tensor_bytes(),
        "max_rss_bytes_process_wide": int(rss),
        "workloads": workload_results,
        "missing_required_workloads": missing_workloads,
        "release_baseline_comparison": {
            "status": "not_measured",
            "required_variants": ["v3", "v4_transformer_sparse", "v4_conv1d"],
        },
        "note": (
            "A complete workload protocol is not release acceptance. Compare against a reproducibly trained v3 baseline, "
            "use permissioned frozen test data and the named target CPU, and lock quality and latency gates before final-set evaluation."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    workload = parser.add_mutually_exclusive_group(required=True)
    workload.add_argument("--image", help="single-image diagnostic workload")
    workload.add_argument("--manifest", help="v4 JSONL manifest for representative workload buckets")
    parser.add_argument("--output")
    parser.add_argument("--warmup", type=int, default=MINIMUM_WARMUP)
    parser.add_argument("--iterations", type=int, default=MINIMUM_ITERATIONS)
    parser.add_argument("--page-resolution", type=int)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    result = benchmark(
        args.artifact,
        args.image,
        manifest=args.manifest,
        warmup=args.warmup,
        iterations=args.iterations,
        page_resolution=args.page_resolution,
        threads=args.threads,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["status"] == "protocol_measured_not_release_acceptance" else 2


if __name__ == "__main__":
    raise SystemExit(main())
