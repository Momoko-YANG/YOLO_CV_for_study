"""
Model Evaluation Script

Evaluate a trained YOLO model and output detailed metrics.

Usage:
    # Evaluate a trained model on validation set
    python training/evaluate.py --model runs/gesture/exp1/weights/best.pt

    # Evaluate on test set
    python training/evaluate.py --model weights/best.pt --split test

    # Compare with baseline
    python training/evaluate.py --model weights/best.pt --baseline yolov8s.pt
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model weights (.pt)")
    parser.add_argument("--data", type=str, default="training/dataset.yaml",
                        help="Dataset config yaml")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Confidence threshold for evaluation")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="IoU threshold for NMS")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--baseline", type=str, default="",
                        help="Baseline model to compare against")
    parser.add_argument("--save-dir", type=str, default="runs/evaluate",
                        help="Directory to save evaluation results")
    return parser.parse_args()


def evaluate_model(model_path: str, args) -> dict:
    """Run validation and return metrics dict."""
    print(f"\nEvaluating: {model_path}")
    print("-" * 50)

    model = YOLO(model_path)

    val_kwargs = dict(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        plots=True,
        save_json=True,
        verbose=True,
    )
    if args.device:
        val_kwargs["device"] = args.device

    metrics = model.val(**val_kwargs)

    # Extract key metrics
    results = {
        "model": model_path,
        "dataset": args.data,
        "split": args.split,
        "mAP50": round(metrics.box.map50, 4),
        "mAP50-95": round(metrics.box.map, 4),
        "precision": round(metrics.box.mp, 4),
        "recall": round(metrics.box.mr, 4),
        "per_class": {},
    }

    # Per-class metrics
    class_names = model.names
    if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None:
        for i, name in class_names.items():
            if i < len(metrics.box.ap50):
                results["per_class"][name] = {
                    "AP50": round(float(metrics.box.ap50[i]), 4),
                    "AP50-95": round(float(metrics.box.ap[i]), 4),
                }

    return results


def print_results(results: dict):
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"  Model:      {results['model']}")
    print(f"  Split:      {results['split']}")
    print(f"{'='*60}")
    print(f"  mAP@50:     {results['mAP50']}")
    print(f"  mAP@50-95:  {results['mAP50-95']}")
    print(f"  Precision:  {results['precision']}")
    print(f"  Recall:     {results['recall']}")

    if results["per_class"]:
        print(f"\n  Per-class AP@50:")
        for cls_name, cls_metrics in results["per_class"].items():
            print(f"    {cls_name:15s}  AP50={cls_metrics['AP50']:.4f}  AP50-95={cls_metrics['AP50-95']:.4f}")

    print(f"{'='*60}\n")


def print_comparison(current: dict, baseline: dict):
    """Print side-by-side comparison."""
    print(f"\n{'='*60}")
    print(f"  Comparison: {Path(current['model']).name} vs {Path(baseline['model']).name}")
    print(f"{'='*60}")
    print(f"  {'Metric':<15} {'Current':>10} {'Baseline':>10} {'Delta':>10}")
    print(f"  {'-'*45}")

    for key in ["mAP50", "mAP50-95", "precision", "recall"]:
        cur_val = current[key]
        base_val = baseline[key]
        delta = cur_val - base_val
        sign = "+" if delta >= 0 else ""
        print(f"  {key:<15} {cur_val:>10.4f} {base_val:>10.4f} {sign}{delta:>9.4f}")

    # Per-class comparison
    if current["per_class"] and baseline["per_class"]:
        print(f"\n  Per-class AP@50:")
        all_classes = set(list(current["per_class"].keys()) + list(baseline["per_class"].keys()))
        for cls_name in sorted(all_classes):
            cur_ap = current["per_class"].get(cls_name, {}).get("AP50", 0)
            base_ap = baseline["per_class"].get(cls_name, {}).get("AP50", 0)
            delta = cur_ap - base_ap
            sign = "+" if delta >= 0 else ""
            print(f"    {cls_name:<15} {cur_ap:>8.4f}  {base_ap:>8.4f}  {sign}{delta:>8.4f}")

    print(f"{'='*60}\n")


def save_results(results: dict, save_dir: str):
    """Save evaluation results to JSON."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model_name = Path(results["model"]).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = save_path / f"eval_{model_name}_{timestamp}.json"

    results["timestamp"] = timestamp
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {filename}")


def main():
    args = parse_args()

    # Evaluate main model
    results = evaluate_model(args.model, args)
    print_results(results)
    save_results(results, args.save_dir)

    # Compare with baseline if provided
    if args.baseline:
        baseline_results = evaluate_model(args.baseline, args)
        print_results(baseline_results)
        save_results(baseline_results, args.save_dir)
        print_comparison(results, baseline_results)


if __name__ == "__main__":
    main()
