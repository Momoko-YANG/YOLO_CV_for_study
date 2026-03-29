"""
数据集质量验证脚本 (Dataset Quality Validation)

Usage:
    # Validate the RPS hand gesture dataset
    python training/validate_dataset.py --data-root datasets/RPS

    # Strict mode: fail on warnings (e.g. class imbalance)
    python training/validate_dataset.py --data-root datasets/RPS --strict
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Validate YOLO-format dataset quality")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory of the dataset (e.g. datasets/RPS)")
    parser.add_argument("--splits", type=str, default="train,valid,test",
                        help="Comma-separated splits to check")
    parser.add_argument("--num-classes", type=int, default=3,
                        help="Expected number of classes")
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as errors")
    parser.add_argument("--sample-images", type=int, default=100,
                        help="Number of images to sample for integrity check")
    parser.add_argument("--output", type=str, default="",
                        help="Path to save JSON report")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def check_directory_structure(data_root: Path, splits: list) -> list:
    """Verify images/ and labels/ directories exist for each split."""
    issues = []
    for split in splits:
        img_dir = data_root / split / "images"
        lbl_dir = data_root / split / "labels"
        if not img_dir.is_dir():
            issues.append({"level": "error", "check": "structure", "msg": f"Missing: {img_dir}"})
        if not lbl_dir.is_dir():
            issues.append({"level": "error", "check": "structure", "msg": f"Missing: {lbl_dir}"})
    return issues


def check_image_label_pairing(data_root: Path, split: str) -> tuple:
    """Check that every image has a label and vice versa."""
    img_dir = data_root / split / "images"
    lbl_dir = data_root / split / "labels"
    issues = []

    if not img_dir.is_dir() or not lbl_dir.is_dir():
        return issues, 0, 0

    img_stems = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")}
    lbl_stems = {p.stem for p in lbl_dir.iterdir() if p.suffix == ".txt"}

    orphan_images = img_stems - lbl_stems
    orphan_labels = lbl_stems - img_stems

    for name in sorted(orphan_images)[:10]:
        issues.append({"level": "warning", "check": "pairing", "msg": f"[{split}] Image without label: {name}"})
    if len(orphan_images) > 10:
        issues.append({"level": "warning", "check": "pairing", "msg": f"[{split}] ... and {len(orphan_images)-10} more orphan images"})

    for name in sorted(orphan_labels)[:10]:
        issues.append({"level": "warning", "check": "pairing", "msg": f"[{split}] Label without image: {name}"})
    if len(orphan_labels) > 10:
        issues.append({"level": "warning", "check": "pairing", "msg": f"[{split}] ... and {len(orphan_labels)-10} more orphan labels"})

    return issues, len(img_stems), len(lbl_stems)


def check_annotation_format(data_root: Path, split: str, num_classes: int) -> tuple:
    """Validate YOLO annotation format: class_id cx cy w h (normalized)."""
    lbl_dir = data_root / split / "labels"
    issues = []
    total_annotations = 0
    class_counts = Counter()

    if not lbl_dir.is_dir():
        return issues, total_annotations, class_counts

    for lbl_file in sorted(lbl_dir.glob("*.txt")):
        with open(lbl_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    issues.append({"level": "error", "check": "format",
                                   "msg": f"[{split}] {lbl_file.name}:{line_num} expected 5 values, got {len(parts)}"})
                    continue

                try:
                    cls_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    issues.append({"level": "error", "check": "format",
                                   "msg": f"[{split}] {lbl_file.name}:{line_num} non-numeric values"})
                    continue

                total_annotations += 1
                class_counts[cls_id] += 1

                if cls_id < 0 or cls_id >= num_classes:
                    issues.append({"level": "error", "check": "format",
                                   "msg": f"[{split}] {lbl_file.name}:{line_num} class_id {cls_id} out of range [0, {num_classes-1}]"})

                for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
                    if val < 0.0 or val > 1.0:
                        issues.append({"level": "error", "check": "format",
                                       "msg": f"[{split}] {lbl_file.name}:{line_num} {name}={val:.4f} out of [0, 1]"})

                # Bbox boundary check
                if cx - w / 2 < -0.01 or cx + w / 2 > 1.01 or cy - h / 2 < -0.01 or cy + h / 2 > 1.01:
                    issues.append({"level": "warning", "check": "bounds",
                                   "msg": f"[{split}] {lbl_file.name}:{line_num} bbox may extend outside image"})

                area = w * h
                if area < 0.001:
                    issues.append({"level": "warning", "check": "size",
                                   "msg": f"[{split}] {lbl_file.name}:{line_num} very small bbox (area={area:.5f})"})
                elif area > 0.9:
                    issues.append({"level": "warning", "check": "size",
                                   "msg": f"[{split}] {lbl_file.name}:{line_num} very large bbox (area={area:.3f})"})

    return issues, total_annotations, class_counts


def check_class_distribution(class_counts: Counter, split: str, num_classes: int) -> list:
    """Analyze class balance."""
    issues = []
    total = sum(class_counts.values())
    if total == 0:
        issues.append({"level": "error", "check": "distribution", "msg": f"[{split}] No annotations found"})
        return issues

    for cls_id in range(num_classes):
        count = class_counts.get(cls_id, 0)
        pct = count / total * 100
        if pct < 10:
            issues.append({"level": "warning", "check": "distribution",
                           "msg": f"[{split}] Class {cls_id} underrepresented: {count} ({pct:.1f}%)"})
        elif pct > 70:
            issues.append({"level": "warning", "check": "distribution",
                           "msg": f"[{split}] Class {cls_id} overrepresented: {count} ({pct:.1f}%)"})

    if class_counts:
        max_c = max(class_counts.values())
        min_c = min(class_counts.values()) if min(class_counts.values()) > 0 else 1
        if max_c / min_c > 3:
            issues.append({"level": "warning", "check": "distribution",
                           "msg": f"[{split}] Class imbalance ratio {max_c/min_c:.1f}:1 exceeds 3:1 threshold"})

    return issues


def check_image_integrity(data_root: Path, split: str, sample_n: int) -> list:
    """Sample images and verify they can be decoded."""
    img_dir = data_root / split / "images"
    issues = []

    if not img_dir.is_dir():
        return issues

    all_images = list(img_dir.glob("*"))
    all_images = [p for p in all_images if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    sample = random.sample(all_images, min(sample_n, len(all_images)))

    corrupt_count = 0
    for img_path in sample:
        img = cv2.imread(str(img_path))
        if img is None:
            issues.append({"level": "error", "check": "integrity",
                           "msg": f"[{split}] Corrupt or unreadable: {img_path.name}"})
            corrupt_count += 1

    return issues


def check_duplicate_images(data_root: Path, split: str, sample_n: int = 200) -> list:
    """Hash a sample of images to detect duplicates."""
    img_dir = data_root / split / "images"
    issues = []

    if not img_dir.is_dir():
        return issues

    all_images = list(img_dir.glob("*"))
    all_images = [p for p in all_images if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    sample = all_images[:min(sample_n, len(all_images))]

    hashes = {}
    for img_path in sample:
        with open(img_path, "rb") as f:
            h = hashlib.md5(f.read(65536)).hexdigest()
        if h in hashes:
            issues.append({"level": "warning", "check": "duplicates",
                           "msg": f"[{split}] Possible duplicate: {img_path.name} == {hashes[h]}"})
        else:
            hashes[h] = img_path.name

    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    data_root = Path(args.data_root)
    splits = [s.strip() for s in args.splits.split(",")]

    if not data_root.is_dir():
        print(f"Error: Dataset root not found: {data_root}")
        sys.exit(1)

    all_issues = []
    report = {"data_root": str(data_root), "splits": {}}

    print(f"\n{'='*60}")
    print(f"  Dataset Quality Validation")
    print(f"  Root: {data_root}")
    print(f"{'='*60}")

    # 1. Directory structure
    all_issues.extend(check_directory_structure(data_root, splits))

    for split in splits:
        split_report = {}

        # 2. Image-label pairing
        pairing_issues, n_images, n_labels = check_image_label_pairing(data_root, split)
        all_issues.extend(pairing_issues)
        split_report["images"] = n_images
        split_report["labels"] = n_labels

        # 3. Annotation format
        format_issues, n_annotations, class_counts = check_annotation_format(data_root, split, args.num_classes)
        all_issues.extend(format_issues)
        split_report["annotations"] = n_annotations
        split_report["class_counts"] = dict(class_counts)

        # 4. Class distribution
        dist_issues = check_class_distribution(class_counts, split, args.num_classes)
        all_issues.extend(dist_issues)

        # 5. Image integrity
        integrity_issues = check_image_integrity(data_root, split, args.sample_images)
        all_issues.extend(integrity_issues)

        # 6. Duplicates
        dup_issues = check_duplicate_images(data_root, split)
        all_issues.extend(dup_issues)

        # Print split summary
        print(f"\n  [{split}]")
        print(f"    Images:      {n_images}")
        print(f"    Labels:      {n_labels}")
        print(f"    Annotations: {n_annotations}")
        if class_counts:
            total = sum(class_counts.values())
            for cls_id in sorted(class_counts.keys()):
                count = class_counts[cls_id]
                pct = count / total * 100
                print(f"    Class {cls_id}:     {count:>6} ({pct:.1f}%)")

        report["splits"][split] = split_report

    # Summary
    errors = [i for i in all_issues if i["level"] == "error"]
    warnings = [i for i in all_issues if i["level"] == "warning"]

    print(f"\n{'-'*60}")
    print(f"  Results: {len(errors)} errors, {len(warnings)} warnings")
    print(f"{'-'*60}")

    for issue in errors:
        print(f"  ERROR   {issue['msg']}")
    for issue in warnings[:20]:
        print(f"  WARN    {issue['msg']}")
    if len(warnings) > 20:
        print(f"  ... and {len(warnings)-20} more warnings")

    if not errors and not warnings:
        print("  All checks passed!")

    print(f"{'='*60}\n")

    # Save report
    if args.output:
        report["issues"] = all_issues
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to: {args.output}")

    # Exit code
    if errors:
        sys.exit(1)
    if warnings and args.strict:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
