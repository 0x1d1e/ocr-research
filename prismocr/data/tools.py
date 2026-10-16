"""Corpus conversion, manifest QA, and leakage-aware document splitting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageOps

from prismocr.data.dataset import PrismPageDataset
from prismocr.data.real_corpus import (
    compose_recognition_pages,
    concat_manifests,
    convert_coco_layout,
    convert_khmerst_scene,
)
from prismocr.data.schema import validate_annotation
from prismocr.geometry import signed_polygon_area

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

_SPLIT_NAMES = ("train", "validation", "test")

# Recognized by prismocr.calibration._validation_split_provenance.
_GROUPED_SPLIT_STATUS = "grouped_by_document_template_family_and_exact_image_sha256"
_GROUPED_SPLIT_REPORT = "split-report.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(path: str | Path) -> str:
    """Return a stable 64-bit dHash for manual near-duplicate leakage audits."""
    with Image.open(path) as image:
        pixels = ImageOps.grayscale(image).resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | int(pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return f"{value:016x}"


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def _near_hash_pairs(hashes: set[str], threshold: int) -> list[tuple[str, str, int]]:
    """Find nearby distinct hashes with a BK-tree rather than a quadratic scan."""
    if threshold < 0 or threshold > 64:
        raise ValueError("Hamming threshold must be between 0 and 64")
    if not hashes:
        return []
    values = sorted(hashes)
    root: dict[str, Any] = {"value": values[0], "children": {}}
    for value in values[1:]:
        node = root
        while True:
            distance = _hamming(value, node["value"])
            child = node["children"].get(distance)
            if child is None:
                node["children"][distance] = {"value": value, "children": {}}
                break
            node = child

    pairs: list[tuple[str, str, int]] = []
    for value in values:
        pending = [root]
        while pending:
            node = pending.pop()
            distance = _hamming(value, node["value"])
            if 0 < distance <= threshold and value < node["value"]:
                pairs.append((value, node["value"], distance))
            low, high = distance - threshold, distance + threshold
            pending.extend(child for edge, child in node["children"].items() if low <= edge <= high)
    return pairs


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def convert_icdar(
    image_dir: str | Path,
    ground_truth_dir: str | Path,
    output_manifest: str | Path,
    *,
    document_prefix: str,
    license_name: str,
    permission_reference: str,
    reviewer: str,
    template_family_id: str | None = None,
) -> dict[str, Any]:
    """Convert ICDAR-style 8-coordinate quadrilateral text annotations to v4 JSONL.

    Ground-truth files are named ``gt_<image-stem>.txt``. Each non-empty line
    contains eight comma-separated coordinates followed by the transcript.
    Ignored ``###`` transcripts are counted and deliberately omitted.
    """
    image_root = Path(image_dir).resolve()
    annotation_root = Path(ground_truth_dir).resolve()
    images = sorted(
        (path for path in image_root.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES),
        key=lambda path: path.name,
    )
    if not images:
        raise ValueError(f"no supported image files in {image_root}")
    if not all((document_prefix, license_name, permission_reference, reviewer)):
        raise ValueError("document prefix, license, permission reference and reviewer are required")

    rows: list[dict[str, Any]] = []
    ignored_count = 0
    line_count = 0
    for page_index, image_path in enumerate(images):
        gt_path = annotation_root / f"gt_{image_path.stem}.txt"
        if not gt_path.is_file():
            raise FileNotFoundError(f"missing ICDAR ground truth file: {gt_path}")
        with Image.open(image_path) as source_image:
            width, height = source_image.size
        image_sha = _sha256_file(image_path)
        document_id = f"{document_prefix}:{image_path.stem}"
        lines: list[dict[str, Any]] = []
        page_ignored = False
        with gt_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for source_line, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                fields = next(csv.reader([raw], skipinitialspace=True))
                if len(fields) < 9:
                    raise ValueError(f"{gt_path}:{source_line}: expected 8 coordinates and text")
                try:
                    coordinates = [float(value.strip()) for value in fields[:8]]
                except ValueError as exc:
                    raise ValueError(f"{gt_path}:{source_line}: invalid quadrilateral coordinate") from exc
                if len(coordinates) != 8:
                    raise ValueError(f"{gt_path}:{source_line}: expected exactly 8 coordinates")
                raw_fields = raw.split(",", maxsplit=8)
                if len(raw_fields) < 9:
                    raise ValueError(f"{gt_path}:{source_line}: expected 8 coordinates and text")
                transcript_field = raw_fields[8].strip()
                transcript_parts = next(csv.reader([transcript_field], skipinitialspace=False))
                text = ",".join(transcript_parts).strip()
                if text == "###":
                    ignored_count += 1
                    page_ignored = True
                    continue
                points = [[coordinates[i], coordinates[i + 1]] for i in range(0, 8, 2)]
                if signed_polygon_area(points) < 0:
                    points.reverse()
                line_id = f"{document_id}:line:{source_line}"
                lines.append({
                    "id": line_id,
                    "polygon_px": points,
                    "text": text,
                    "role": "UNKNOWN",
                })
        line_count += len(lines)
        provenance: dict[str, str] = {
            "source_class": "real",
            "source_dataset": "ICDAR quadrilateral text",
            "license": license_name,
            "permission_reference": permission_reference,
            "reviewer": reviewer,
        }
        if template_family_id:
            provenance["template_family_id"] = template_family_id
        annotation = {
            "schema_version": "4.0",
            "document_id": document_id,
            "page_id": f"{document_id}:page:0",
            "image": {"width": width, "height": height, "sha256": image_sha},
            "provenance": provenance,
            "supervision": {"lines": ("partial" if lines else "unavailable") if page_ignored else "complete"},
            "lines": lines,
            "regions": [],
            "cells": [],
            "relations": [],
            "reading_order": [line["id"] for line in lines],
        }
        validate_annotation(annotation)
        rows.append({"image_path": str(image_path), "annotation": annotation})

    output = Path(output_manifest).resolve()
    _atomic_jsonl(output, rows)
    return {
        "manifest": str(output),
        "pages": len(rows),
        "lines": line_count,
        "ignored_dont_care_regions": ignored_count,
        "status": "converted_requires_human_label_qa",
    }


def audit_manifest(manifest: str | Path, *, near_duplicate_hamming: int = 4) -> dict[str, Any]:
    """Validate content checksums and summarize provenance and duplicate risks."""
    if isinstance(near_duplicate_hamming, bool) or not isinstance(near_duplicate_hamming, int) or not 0 <= near_duplicate_hamming <= 64:
        raise ValueError("near_duplicate_hamming must be between 0 and 64")
    dataset = PrismPageDataset(manifest)
    documents: set[str] = set()
    source_classes: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    region_counts: Counter[str] = Counter()
    exact: dict[str, list[int]] = defaultdict(list)
    fingerprints: list[dict[str, Any]] = []
    pages_by_dhash: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(dataset.records):
        annotation = record["_annotation"]
        provenance = annotation["provenance"]
        documents.add(annotation["document_id"])
        source_classes[provenance["source_class"]] += 1
        digest = record["_image_sha256"]
        exact[digest].append(index)
        image_path = record["_image_path"]
        dhash = difference_hash(image_path)
        fingerprints.append({
            "page_id": annotation["page_id"],
            "document_id": annotation["document_id"],
            "image_sha256": digest,
            "dhash64": dhash,
        })
        pages_by_dhash[dhash].append(index)
        for line in annotation.get("lines", ()):
            role_counts[line["role"]] += 1
        for region in annotation.get("regions", ()):
            region_counts[region["kind"]] += 1
        for _cell in annotation.get("cells", ()):
            region_counts["CELL"] += 1

    hash_pairs = _near_hash_pairs(set(pages_by_dhash), near_duplicate_hamming)
    near_pairs = [
        {
            "left_dhash64": left_hash,
            "right_dhash64": right_hash,
            "left_page_count": len(pages_by_dhash[left_hash]),
            "right_page_count": len(pages_by_dhash[right_hash]),
            "dhash_hamming_distance": distance,
            "sample_page_ids": [
                fingerprints[index]["page_id"]
                for index in (pages_by_dhash[left_hash][:2] + pages_by_dhash[right_hash][:2])
            ],
        }
        for left_hash, right_hash, distance in hash_pairs[:512]
    ]
    exact_hash_near_groups = [
        {"dhash64": value, "page_count": len(indices), "page_ids": [fingerprints[i]["page_id"] for i in indices[:4]]}
        for value, indices in pages_by_dhash.items()
        if len(indices) > 1 and len({fingerprints[i]["image_sha256"] for i in indices}) > 1
    ]
    return {
        "schema_version": "4.0",
        "status": "manifest_content_and_schema_audited_not_annotation_quality",
        "manifest": str(Path(manifest).resolve()),
        "page_count": len(dataset),
        "document_count": len(documents),
        "source_page_counts": dict(sorted(source_classes.items())),
        "line_count": sum(role_counts.values()),
        "line_role_counts": dict(sorted(role_counts.items())),
        "region_and_cell_counts": dict(sorted(region_counts.items())),
        "exact_duplicate_image_groups": [indices for indices in exact.values() if len(indices) > 1],
        "near_duplicate_hash_groups": near_pairs,
        "near_duplicate_same_hash_groups": exact_hash_near_groups,
        "near_duplicate_group_count": len(hash_pairs) + len(exact_hash_near_groups),
        "near_duplicate_details_truncated": len(hash_pairs) > 512,
        "near_duplicate_hamming_threshold": near_duplicate_hamming,
        "fingerprints": fingerprints,
        "limitations": [
            "Image and annotation structure/checksums were validated; semantic label correctness still requires human review.",
            "Perceptual-hash matches are review candidates, not proof of duplicate documents.",
        ],
    }


def _group_records(
    dataset: PrismPageDataset, near_duplicate_hamming: int | None
) -> list[list[int]]:
    """Union records that must stay in one split.

    Document identity, template family, and exact image bytes are always enforced.
    ``near_duplicate_hamming`` additionally unions perceptual dHash neighbours; pass
    ``None`` to disable perceptual grouping, which is required for corpora that hold
    near-blank pages from different documents (perceptually identical, unrelated
    content) and is recorded as ``null`` in the split report.
    """
    parent = list(range(len(dataset)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_for_key: dict[tuple[str, str], int] = {}
    first_for_digest: dict[str, int] = {}
    for index, record in enumerate(dataset.records):
        annotation = record["_annotation"]
        provenance = annotation["provenance"]
        document = annotation["document_id"]
        family = provenance.get("template_family_id")
        keys = [("document", document)]
        if family:
            keys.append(("family", family))
        for key in keys:
            previous = first_for_key.setdefault(key, index)
            union(index, previous)
        digest = record["_image_sha256"]
        previous_image = first_for_digest.setdefault(digest, index)
        union(index, previous_image)

    if near_duplicate_hamming is not None:
        indices_by_dhash: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(dataset.records):
            indices_by_dhash[difference_hash(record["_image_path"])].append(index)
        for indices in indices_by_dhash.values():
            for index in indices[1:]:
                union(index, indices[0])
        for left_hash, right_hash, _distance in _near_hash_pairs(
            set(indices_by_dhash), near_duplicate_hamming
        ):
            for index in indices_by_dhash[right_hash]:
                union(index, indices_by_dhash[left_hash][0])

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(dataset)):
        groups[find(index)].append(index)
    return list(groups.values())


def _split_membership(
    records_by_split: Mapping[str, Sequence[dict[str, Any]]], near_duplicate_hamming: int | None
) -> dict[str, Any]:
    """Hash document, image, and near-duplicate group membership per split.

    ``prismocr.calibration`` recomputes exactly these memberships from the frozen
    manifests, so splitting and corpus building share one implementation.
    """
    if near_duplicate_hamming is not None and (
        isinstance(near_duplicate_hamming, bool)
        or not isinstance(near_duplicate_hamming, int)
        or not 0 <= near_duplicate_hamming <= 64
    ):
        raise ValueError("near_duplicate_hamming must be between 0 and 64 or None")
    records = [record for name in _SPLIT_NAMES for record in records_by_split[name]]
    combined = object.__new__(PrismPageDataset)
    combined.records = records
    groups = _group_records(combined, near_duplicate_hamming)
    owner: dict[int, str] = {}
    index = 0
    for name in _SPLIT_NAMES:
        for _record in records_by_split[name]:
            owner[index] = name
            index += 1
    membership = {
        name: {"document_sha256": set(), "image_sha256": set(), "group_sha256": set()}
        for name in _SPLIT_NAMES
    }
    for name in _SPLIT_NAMES:
        for record in records_by_split[name]:
            membership[name]["document_sha256"].add(
                hashlib.sha256(record["_annotation"]["document_id"].encode("utf-8")).hexdigest()
            )
            membership[name]["image_sha256"].add(record["_image_sha256"])
    for group in groups:
        owners = {owner[item] for item in group}
        if len(owners) != 1:
            raise ValueError("a document, template family, or near-duplicate group spans more than one split")
        name = next(iter(owners))
        membership[name]["group_sha256"].add(
            hashlib.sha256(
                "\n".join(sorted(records[item]["_image_sha256"] for item in group)).encode("utf-8")
            ).hexdigest()
        )
    return {
        "page_counts": {name: len(records_by_split[name]) for name in _SPLIT_NAMES},
        "independent_group_count": len(groups),
        "split_fingerprints": {
            name: {
                "document_sha256": sorted(membership[name]["document_sha256"]),
                "image_sha256": sorted(membership[name]["image_sha256"]),
                "group_sha256": sorted(membership[name]["group_sha256"]),
            }
            for name in _SPLIT_NAMES
        },
    }


def write_split_report(
    output_dir: str | Path,
    *,
    manifests: Mapping[str, str | Path],
    seed: int,
    near_duplicate_hamming: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Describe already-frozen train/validation/test manifests in a split report.

    Use this when an upstream process owns split assignment and the manifests must
    keep their membership (for example the source-disjoint real-corpus build)
    instead of being reassigned by :func:`build_document_splits`. The default
    ``near_duplicate_hamming=None`` records that perceptual grouping is disabled:
    membership is then proven by document, template family, and exact image bytes.
    """
    if set(manifests) != set(_SPLIT_NAMES):
        raise ValueError("manifests must name exactly train, validation, and test")
    paths = {name: Path(manifests[name]).resolve() for name in _SPLIT_NAMES}
    missing = sorted(str(path) for path in paths.values() if not path.is_file())
    if missing:
        raise ValueError(f"missing frozen split manifest: {', '.join(missing)}")
    output = Path(output_dir).resolve()
    report_path = output / _GROUPED_SPLIT_REPORT
    if (report_path.exists() or report_path.is_symlink()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing frozen split outputs: {report_path}")

    datasets = {name: PrismPageDataset(path) for name, path in paths.items()}
    empty = sorted(name for name, dataset in datasets.items() if not len(dataset))
    if empty:
        raise ValueError(f"frozen split manifest is empty: {', '.join(empty)}")
    membership = _split_membership(
        {name: dataset.records for name, dataset in datasets.items()}, near_duplicate_hamming
    )
    total_pages = sum(membership["page_counts"].values())
    result: dict[str, Any] = {
        "schema_version": "4.0",
        "seed": seed,
        "near_duplicate_hamming": near_duplicate_hamming,
        "ratios": {
            name: membership["page_counts"][name] / total_pages for name in _SPLIT_NAMES
        },
        **membership,
        "manifests": {name: str(paths[name]) for name in _SPLIT_NAMES},
        "manifest_sha256": {name: _sha256_file(paths[name]) for name in _SPLIT_NAMES},
        "status": _GROUPED_SPLIT_STATUS,
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def build_document_splits(
    manifest: str | Path,
    output_dir: str | Path,
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    near_duplicate_hamming: int | None = 2,
) -> dict[str, Any]:
    """Create deterministic train/validation/test manifests without group leakage."""
    if len(ratios) != 3 or any(value <= 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("split ratios must be three positive values summing to 1")
    dataset = PrismPageDataset(manifest)
    if near_duplicate_hamming is not None and (
        isinstance(near_duplicate_hamming, bool)
        or not isinstance(near_duplicate_hamming, int)
        or not 0 <= near_duplicate_hamming <= 64
    ):
        raise ValueError("near_duplicate_hamming must be between 0 and 64 or None")
    groups = _group_records(dataset, near_duplicate_hamming)
    if len(groups) < 3:
        raise ValueError("at least three independent document/family/image groups are required")
    names = _SPLIT_NAMES
    total_pages = len(dataset)
    targets = [ratio * total_pages for ratio in ratios]
    assigned: list[list[int]] = [[], [], []]
    group_order = list(groups)
    rng = random.Random(seed)
    rng.shuffle(group_order)
    group_order.sort(key=len, reverse=True)
    assigned_pages = [0, 0, 0]
    for group in group_order:
        eligible = [index for index, ratio in enumerate(ratios) if ratio > 0]
        split_index = max(
            eligible,
            key=lambda index: ((targets[index] - assigned_pages[index]) / targets[index], -index),
        )
        assigned[split_index].extend(group)
        assigned_pages[split_index] += len(group)

    output = Path(output_dir).resolve()
    protected_outputs = [
        *(output / f"{name}.jsonl" for name in names),
        output / _GROUPED_SPLIT_REPORT,
    ]
    existing_outputs = [path for path in protected_outputs if path.exists() or path.is_symlink()]
    if existing_outputs:
        listed = ", ".join(str(path) for path in existing_outputs)
        raise FileExistsError(
            f"refusing to overwrite existing frozen split outputs: {listed}; use a new versioned output directory"
        )
    split_paths: dict[str, str] = {}
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    for name, indices in zip(names, assigned, strict=True):
        ordered = sorted(indices)
        rows = [
            {"image_path": str(dataset.records[index]["_image_path"]), "annotation": dataset.records[index]["_annotation"]}
            for index in ordered
        ]
        path = output / f"{name}.jsonl"
        _atomic_jsonl(path, rows)
        split_paths[name] = str(path)
        records_by_split[name] = [dataset.records[index] for index in ordered]
    result = {
        "schema_version": "4.0",
        "seed": seed,
        "near_duplicate_hamming": near_duplicate_hamming,
        "ratios": dict(zip(names, ratios, strict=True)),
        **_split_membership(records_by_split, near_duplicate_hamming),
        "manifests": split_paths,
        "manifest_sha256": {
            name: _sha256_file(Path(path)) for name, path in split_paths.items()
        },
        "fingerprints": audit_manifest(manifest)["fingerprints"],
        "status": _GROUPED_SPLIT_STATUS,
    }
    (output / "split-report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert-icdar", help="convert ICDAR quadrilateral text annotations")
    convert.add_argument("--images", required=True)
    convert.add_argument("--ground-truth", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--document-prefix", required=True)
    convert.add_argument("--license", required=True)
    convert.add_argument("--permission-reference", required=True)
    convert.add_argument("--reviewer", required=True)
    convert.add_argument("--template-family-id")
    qa = commands.add_parser("qa", help="validate a manifest and report label/data fingerprints")
    qa.add_argument("--manifest", required=True)
    qa.add_argument("--output")
    qa.add_argument("--near-duplicate-hamming", type=int, default=4)
    split = commands.add_parser("split", help="create document/family-grouped train/validation/test manifests")
    split.add_argument("--manifest", required=True)
    split.add_argument("--output-dir", required=True)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1))
    split.add_argument("--near-duplicate-hamming", type=int, default=2)
    compose = commands.add_parser(
        "compose-rec-pages", help="compose page canvases from text-line recognition crops"
    )
    compose.add_argument("--labels", required=True, help="PaddleOCR-style path<TAB>text file")
    compose.add_argument("--output-dir", required=True)
    compose.add_argument("--pages", type=int, required=True)
    compose.add_argument("--license", required=True)
    compose.add_argument("--permission-reference", required=True)
    compose.add_argument("--reviewer", required=True)
    compose.add_argument("--template-family-id", required=True)
    compose.add_argument("--document-prefix")
    compose.add_argument("--source-class", default="derived", choices=("real", "synthetic", "derived"))
    compose.add_argument("--seed", type=int, default=42)
    compose.add_argument("--image-size", type=int, nargs=2, default=(1024, 1400))
    compose.add_argument("--columns", type=int, default=2)
    compose.add_argument("--lines-per-page", type=int, nargs=2, default=(16, 44))
    compose.add_argument("--blank-page-fraction", type=float, default=0.02)
    compose.add_argument("--rotation-degrees", type=float, default=2.0)
    compose.add_argument("--scale-range", type=float, nargs=2, default=(0.85, 1.15))
    compose.add_argument("--margin", type=int, default=28)
    compose.add_argument("--gutter", type=int, default=24)
    compose.add_argument("--max-characters", type=int, default=160)
    compose.add_argument("--prefix-filter")
    convert_coco = commands.add_parser(
        "convert-coco-layout", help="convert COCO region polygons into region-only page manifests"
    )
    convert_coco.add_argument("--coco", required=True)
    convert_coco.add_argument("--images", required=True)
    convert_coco.add_argument("--output", required=True)
    convert_coco.add_argument("--license", required=True)
    convert_coco.add_argument("--permission-reference", required=True)
    convert_coco.add_argument("--reviewer", required=True)
    convert_coco.add_argument("--document-prefix", required=True)
    convert_coco.add_argument("--limit", type=int)
    convert_khmerst = commands.add_parser(
        "convert-khmerst-scene", help="convert KhmerST page-level VIA line annotations"
    )
    convert_khmerst.add_argument("--annotations", required=True, help="VIA annotation zip or directory of JSON")
    convert_khmerst.add_argument("--images", required=True)
    convert_khmerst.add_argument("--output", required=True)
    convert_khmerst.add_argument("--license", required=True)
    convert_khmerst.add_argument("--permission-reference", required=True)
    convert_khmerst.add_argument("--reviewer", required=True)
    convert_khmerst.add_argument("--document-prefix", default="khmerst")
    convert_khmerst.add_argument("--split", choices=("train", "val", "test"))
    convert_khmerst.add_argument("--seed", type=int, default=0)
    convert_khmerst.add_argument("--line-role", default="UNKNOWN")
    concat = commands.add_parser("concat-manifests", help="concatenate JSONL manifests")
    concat.add_argument("--manifest", action="append", required=True)
    concat.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "convert-icdar":
        result = convert_icdar(
            args.images,
            args.ground_truth,
            args.output,
            document_prefix=args.document_prefix,
            license_name=args.license,
            permission_reference=args.permission_reference,
            reviewer=args.reviewer,
            template_family_id=args.template_family_id,
        )
    elif args.command == "qa":
        result = audit_manifest(args.manifest, near_duplicate_hamming=args.near_duplicate_hamming)
        if args.output:
            Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif args.command == "split":
        result = build_document_splits(
            args.manifest,
            args.output_dir,
            ratios=tuple(args.ratios),
            seed=args.seed,
            near_duplicate_hamming=args.near_duplicate_hamming,
        )
    elif args.command == "compose-rec-pages":
        result = compose_recognition_pages(
            args.labels,
            args.output_dir,
            pages=args.pages,
            license_name=args.license,
            permission_reference=args.permission_reference,
            reviewer=args.reviewer,
            template_family_id=args.template_family_id,
            document_prefix=args.document_prefix,
            source_class=args.source_class,
            seed=args.seed,
            image_size=tuple(args.image_size),
            columns=args.columns,
            lines_per_page=tuple(args.lines_per_page),
            blank_page_fraction=args.blank_page_fraction,
            rotation_degrees=args.rotation_degrees,
            scale_range=tuple(args.scale_range),
            margin=args.margin,
            gutter=args.gutter,
            max_characters=args.max_characters,
            prefix_filter=args.prefix_filter,
        )
    elif args.command == "convert-khmerst-scene":
        from prismocr.data.real_corpus import assign_scene_splits, read_khmerst_annotations

        records, _ = read_khmerst_annotations(args.annotations)
        assignment = assign_scene_splits([record["filename"] for record in records], seed=args.seed)
        result = convert_khmerst_scene(
            args.annotations,
            args.images,
            args.output,
            license_name=args.license,
            permission_reference=args.permission_reference,
            reviewer=args.reviewer,
            document_prefix=args.document_prefix,
            split=args.split,
            split_assignment=assignment,
            line_role=args.line_role,
        )
    elif args.command == "convert-coco-layout":
        result = convert_coco_layout(
            args.coco,
            args.images,
            args.output,
            license_name=args.license,
            permission_reference=args.permission_reference,
            reviewer=args.reviewer,
            document_prefix=args.document_prefix,
            limit=args.limit,
        )
    else:
        result = concat_manifests(args.manifest, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
