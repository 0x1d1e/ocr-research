"""Build the local PrismOCR training corpus from permissioned host corpora.

Composes page images from text-line crops (``prismocr.data.real_corpus``) and
converts DocLayNet COCO layout and KhmerST page-level VIA line annotations,
then writes concatenated train/validation/test manifests, the grouped split
report used by calibration, and a build report.

The split report covers the manifests training and calibration actually use: the
train and test splits plus the frozen validation selection subset, because
``prismocr.calibration`` binds a checkpoint to the validation manifest it was
validated on.

Every page carries provenance (source class, dataset, license, permission
reference, reviewer, template family). Pages are independent per index, so a
shard of page ranges can run in a worker process and the concatenated result is
identical to a single-process build.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import sys
from pathlib import Path

from prismocr.data.dataset import PrismPageDataset
from prismocr.data.real_corpus import (
    assign_scene_splits,
    compose_recognition_pages,
    concat_manifests,
    convert_coco_layout,
    convert_khmerst_scene,
    read_khmerst_annotations,
)
from prismocr.data.tools import write_split_report

DATASET_ROOT = Path("/home/kosal/data")
LABELS = DATASET_ROOT / "paddlex_rec_dataset"
KHMERST_ROOT = DATASET_ROOT / "khmerst-full"
KHMERST_ANNOTATIONS = KHMERST_ROOT / "scen_json_data_v1.zip"
KHMERST_IMAGES = KHMERST_ROOT / "scene_img_data"

# (name, prefix, pages, source_class, dataset label, license, permission, template family)
COMPOSED_POOLS: dict[str, tuple[tuple[str, str, int, str, str, str, str, str], ...]] = {
    "train": (
        ("fintech-train", "images/fintech_train/", 6000, "synthetic", "fintech-ocr generated lines",
         "generated", "fintech-ocr synthetic generator (local build)", "fintech-lines-train-v1"),
        ("khmersynth-train", "images/khmersynth1m/", 6000, "synthetic", "KhmerSynthetic1M",
         "Apache-2.0", "KhmerSynthetic1M public release", "khmersynth-lines-train-v1"),
        ("wildkhmer-cropped-train", "images/wildkhmer_cropped/", 2000, "derived", "WildKhmerST cropped lines",
         "CC-BY-NC-4.0", "WildKhmerST public release (non-commercial)", "wildkhmer-cropped-train-v1"),
        ("wildkhmer-warped-train", "images/wildkhmer_warped/", 2000, "derived", "WildKhmerST warped lines",
         "CC-BY-NC-4.0", "WildKhmerST public release (non-commercial)", "wildkhmer-warped-train-v1"),
    ),
    "val": (
        ("fintech-val", "images/fintech_val/", 400, "synthetic", "fintech-ocr generated lines",
         "generated", "fintech-ocr synthetic generator (local build)", "fintech-lines-val-v1"),
        ("khmersynth-val", "images/khmersynth1m/", 400, "synthetic", "KhmerSynthetic1M",
         "Apache-2.0", "KhmerSynthetic1M public release", "khmersynth-lines-val-v1"),
        ("wildkhmer-cropped-val", "images/wildkhmer_cropped/", 300, "derived", "WildKhmerST cropped lines",
         "CC-BY-NC-4.0", "WildKhmerST public release (non-commercial)", "wildkhmer-cropped-val-v1"),
        ("wildkhmer-warped-val", "images/wildkhmer_warped/", 300, "derived", "WildKhmerST warped lines",
         "CC-BY-NC-4.0", "WildKhmerST public release (non-commercial)", "wildkhmer-warped-val-v1"),
    ),
    "test": (
        ("fintech-test", "images/fintech_test/", 400, "synthetic", "fintech-ocr generated lines",
         "generated", "fintech-ocr synthetic generator (local build)", "fintech-lines-test-v1"),
        ("khmersynth-test", "images/khmersynth1m/", 400, "synthetic", "KhmerSynthetic1M",
         "Apache-2.0", "KhmerSynthetic1M public release", "khmersynth-lines-test-v1"),
        ("wildkhmer-cropped-test", "images/wildkhmer_cropped/", 300, "derived", "WildKhmerST cropped lines",
         "CC-BY-NC-4.0", "WildKhmerST public release (non-commercial)", "wildkhmer-cropped-test-v1"),
        ("wildkhmer-warped-test", "images/wildkhmer_warped/", 300, "derived", "WildKhmerST warped lines",
         "CC-BY-NC-4.0", "WildKhmerST public release (non-commercial)", "wildkhmer-warped-test-v1"),
    ),
}

DOCLAYNET_LIMITS = {"train": 24000, "val": None, "test": None}
SHARD_PAGES = 500


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compose_shard(arguments: dict) -> dict:
    return compose_recognition_pages(
        arguments["labels"],
        arguments["output_dir"],
        pages=arguments["pages"],
        license_name=arguments["license"],
        permission_reference=arguments["permission"],
        reviewer="prismocr-corpus-build",
        template_family_id=arguments["template_family"],
        document_prefix=arguments["name"],
        source_class=arguments["source_class"],
        seed=arguments["seed"],
        prefix_filter=arguments["prefix"],
        page_offset=arguments["page_offset"],
        blank_page_fraction=0.0,
        image_format="JPEG",
        max_characters=160,
    )


def _completed_shard(directory: Path, expected_pages: int) -> dict | None:
    """Return a summary for an already-composed shard, or None to rebuild it."""
    manifest = directory / "manifest.jsonl"
    if not manifest.is_file():
        return None
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected_pages or any(not Path(row["image_path"]).is_file() for row in rows):
        return None
    return {
        "manifest": str(manifest),
        "pages": len(rows),
        "lines": sum(len(row["annotation"]["lines"]) for row in rows),
    }


def _compose_or_reuse(arguments: dict) -> dict:
    """Module-level worker: reuse a complete shard manifest or compose the shard."""
    cached = _completed_shard(Path(arguments["output_dir"]), arguments["pages"])
    return cached if cached is not None else _compose_shard(arguments)


def _run_shards(shards: list[dict], jobs: int) -> list[dict]:
    """Compose shards in worker processes, falling back to serial on failure."""
    if jobs <= 1 or len(shards) <= 1:
        return [_compose_or_reuse(arguments) for arguments in shards]
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=jobs, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            return list(executor.map(_compose_or_reuse, shards))
    except Exception as error:  # pragma: no cover - environment-dependent fallback
        print(f"  parallel composition failed ({error!r}); retrying serially", flush=True)
        return [_compose_or_reuse(arguments) for arguments in shards]


def build_split(root: Path, split: str, jobs: int, seed: int) -> dict:
    split_file = LABELS / f"{'val' if split == 'val' else split}.txt"
    manifests: list[Path] = []
    pools: dict[str, dict] = {}
    for pool_index, (name, prefix, pages, source_class, dataset, license_name, permission, family) in enumerate(
        COMPOSED_POOLS[split]
    ):
        pool_dir = root / "composed" / split / name
        shards = [
            {
                "labels": str(split_file),
                "output_dir": str(pool_dir / f"shard-{offset // SHARD_PAGES:04d}"),
                "pages": min(SHARD_PAGES, pages - offset),
                "license": license_name,
                "permission": permission,
                "template_family": family,
                "name": f"{name}-s{offset // SHARD_PAGES:04d}",
                "source_class": source_class,
                "seed": seed + 1000 * (pool_index + 1),
                "prefix": prefix,
                "page_offset": offset,
            }
            for offset in range(0, pages, SHARD_PAGES)
        ]
        results = _run_shards(shards, jobs)
        total_lines = sum(result["lines"] for result in results)
        shard_manifests = [Path(result["manifest"]) for result in results]
        pool_manifest = pool_dir / "manifest.jsonl"
        merged = concat_manifests(shard_manifests, pool_manifest)
        manifests.append(pool_manifest)
        pools[name] = {
            "pages": merged["pages"],
            "lines": total_lines,
            "source_class": source_class,
            "source_dataset": dataset,
            "license": license_name,
            "manifest_sha256": merged["manifest_sha256"],
            "manifests": [str(path) for path in shard_manifests],
        }
        print(f"  composed {split}/{name}: {merged['pages']} pages, {total_lines} lines", flush=True)

    khmerst_manifest = root / "khmerst" / f"{split}.jsonl"
    if not KHMERST_ANNOTATIONS.is_file() or not KHMERST_IMAGES.is_dir():
        print(
            f"  khmerst {split}: skipped; no annotations at {KHMERST_ANNOTATIONS} or images at {KHMERST_IMAGES}",
            flush=True,
        )
    elif not khmerst_manifest.is_file():
        records, skipped = read_khmerst_annotations(KHMERST_ANNOTATIONS)
        assignment = assign_scene_splits([record["filename"] for record in records], seed=seed)
        result = convert_khmerst_scene(
            KHMERST_ANNOTATIONS,
            KHMERST_IMAGES,
            khmerst_manifest,
            license_name="Apache-2.0",
            permission_reference=(
                "KhmerST public research benchmark (arXiv:2410.18277); cropped derivative release "
                "SoyVitou/KhmerST-Dataset-OCR-Cropped declares Apache-2.0"
            ),
            reviewer="KhmerST_annotation_release",
            document_prefix="khmerst",
            split=split,
            split_assignment=assignment,
        )
        print(
            f"  khmerst {split}: {result['pages']} pages, {result['lines']} lines, skipped={skipped}",
            flush=True,
        )
    if khmerst_manifest.is_file():
        manifests.append(khmerst_manifest)

    doclaynet_dir = root / "doclaynet"
    doclaynet_manifest = doclaynet_dir / f"{split}.jsonl"
    if not doclaynet_manifest.is_file():
        result = convert_coco_layout(
            DATASET_ROOT / "DocLayNet" / "extracted" / "COCO" / f"{split}.json",
            DATASET_ROOT / "DocLayNet" / "extracted" / "PNG",
            doclaynet_manifest,
            license_name="CDLA-Permissive-1.0",
            permission_reference="DocLayNet v1.0 public release (CDLA-Permissive-1.0)",
            reviewer="DocLayNet_annotation_release",
            document_prefix="doclaynet",
            limit=DOCLAYNET_LIMITS[split],
        )
        print(f"  doclaynet {split}: {result['pages']} pages, {result['regions']} regions", flush=True)
    manifests.append(doclaynet_manifest)
    combined = concat_manifests(manifests, root / "splits" / f"{split}.jsonl")
    dataset = PrismPageDataset(combined["manifest"])
    result = {
        "manifest": combined["manifest"],
        "manifest_sha256": combined["manifest_sha256"],
        "pages": combined["pages"],
        "pools": pools,
        "dataset_loadable_pages": len(dataset),
    }
    if split == "val":
        result["selection_subset"] = _write_selection_subset(root, Path(combined["manifest"]))
    return result


def _write_selection_subset(root: Path, manifest: Path, *, composed_per_pool: int = 120, doclaynet_pages: int = 200) -> dict:
    """Write a deterministic stratified subset for per-epoch checkpoint selection.

    Selecting on the full validation manifest after every epoch is expensive;
    the frozen full manifest is still used for final reporting and calibration.
    """
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    composed_by_family: dict[str, list[dict]] = {}
    real_by_dataset: dict[str, list[dict]] = {}
    for row in rows:
        provenance = row["annotation"]["provenance"]
        if provenance["source_class"] == "real":
            real_by_dataset.setdefault(provenance.get("source_dataset") or "real", []).append(row)
        else:
            composed_by_family.setdefault(provenance.get("template_family_id") or "unfamilied", []).append(row)
    selected: list[dict] = []
    for family in sorted(composed_by_family):
        selected.extend(composed_by_family[family][:composed_per_pool])
    for dataset in sorted(real_by_dataset):
        group = real_by_dataset[dataset]
        step = max(1, len(group) // doclaynet_pages)
        selected.extend(group[::step][:doclaynet_pages])
    output = root / "splits" / "val-select.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected), encoding="utf-8")
    return {"manifest": str(output), "manifest_sha256": _sha256(output), "pages": len(PrismPageDataset(output))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/kosal/data/prismocr-corpus")
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "build-report.json"
    report: dict = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    report.setdefault("schema_version", "4.0")
    report.setdefault("label_source", str(LABELS / "train.txt"))
    report.setdefault("seed", args.seed)
    report["splits"] = report.get("splits", {})
    for split in args.splits:
        print(f"building {split}", flush=True)
        report["splits"][split] = build_split(root, split, args.jobs, args.seed)
    if {"train", "val", "test"} <= set(args.splits):
        validation = report["splits"]["val"]["selection_subset"]["manifest"]
        report["split_report"] = write_split_report(
            root / "splits",
            manifests={
                "train": root / "splits" / "train.jsonl",
                "validation": validation,
                "test": root / "splits" / "test.jsonl",
            },
            seed=args.seed,
            overwrite=True,
        )
        print(
            "split report: "
            + json.dumps(report["split_report"]["page_counts"])
            + f" groups={report['split_report']['independent_group_count']}",
            flush=True,
        )
    report["status"] = "built_from_local_permissioned_corpora_not_label_reviewed"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({split: values["pages"] for split, values in report["splits"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
