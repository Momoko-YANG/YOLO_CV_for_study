"""
Experiment Comparison Script

Scan all evaluation result JSONs and generate a comparison table.

Usage:
    # Compare all experiments
    python training/compare.py

    # Compare specific results directory
    python training/compare.py --results-dir runs/evaluate

    # Sort by specific metric
    python training/compare.py --sort-by mAP50
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("--results-dir", type=str, default="runs/evaluate",
                        help="Directory containing evaluation JSON files")
    parser.add_argument("--sort-by", type=str, default="mAP50",
                        choices=["mAP50", "mAP50-95", "precision", "recall"],
                        help="Metric to sort by")
    parser.add_argument("--output", type=str, default="",
                        help="Output CSV path (optional)")
    return parser.parse_args()


def load_results(results_dir: str) -> list:
    """Load all evaluation JSON files."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run evaluate.py first to generate results.")
        sys.exit(1)

    results = []
    for json_file in sorted(results_path.glob("eval_*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            data["_file"] = json_file.name
            results.append(data)

    return results


def print_comparison_table(results: list, sort_by: str):
    """Print a formatted comparison table."""
    if not results:
        print("No evaluation results found.")
        return

    # Sort by metric (descending)
    results.sort(key=lambda x: x.get(sort_by, 0), reverse=True)

    # Header
    print(f"\n{'='*90}")
    print(f"  Experiment Comparison (sorted by {sort_by})")
    print(f"{'='*90}")
    print(f"  {'Model':<30} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8} {'Time':>14}")
    print(f"  {'-'*80}")

    # Rows
    best_map50 = results[0].get("mAP50", 0)
    for r in results:
        model_name = Path(r.get("model", "unknown")).stem[:28]
        map50 = r.get("mAP50", 0)
        map50_95 = r.get("mAP50-95", 0)
        precision = r.get("precision", 0)
        recall = r.get("recall", 0)
        timestamp = r.get("timestamp", "N/A")

        marker = " *" if map50 == best_map50 else "  "
        print(f"{marker}{model_name:<30} {map50:>8.4f} {map50_95:>10.4f} {precision:>10.4f} {recall:>8.4f} {timestamp:>14}")

    print(f"\n  * = best {sort_by}")

    # Per-class breakdown for top model
    top = results[0]
    if top.get("per_class"):
        print(f"\n  Per-class detail (best model: {Path(top['model']).stem}):")
        for cls_name, cls_metrics in top["per_class"].items():
            print(f"    {cls_name:<15} AP50={cls_metrics['AP50']:.4f}  AP50-95={cls_metrics['AP50-95']:.4f}")

    print(f"{'='*90}\n")


def export_csv(results: list, output_path: str):
    """Export comparison to CSV."""
    import csv
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "mAP50", "mAP50-95", "precision", "recall", "split", "timestamp"])
        for r in results:
            writer.writerow([
                Path(r.get("model", "")).stem,
                r.get("mAP50", ""),
                r.get("mAP50-95", ""),
                r.get("precision", ""),
                r.get("recall", ""),
                r.get("split", ""),
                r.get("timestamp", ""),
            ])
    print(f"CSV exported to: {output_path}")


def main():
    args = parse_args()

    results = load_results(args.results_dir)
    print_comparison_table(results, args.sort_by)

    if args.output:
        export_csv(results, args.output)


if __name__ == "__main__":
    main()
