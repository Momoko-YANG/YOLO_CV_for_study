"""
模型推理基准测试 (Inference Benchmark)

Compare inference speed and accuracy across model formats.

Usage:
    # Speed benchmark: PyTorch vs ONNX
    python training/benchmark.py --models backend/weights/best-yolov8n.pt runs/export/best-yolov8n.onnx

    # Speed + accuracy benchmark
    python training/benchmark.py --models backend/weights/best-yolov8n.pt runs/export/best-yolov8n.onnx --data training/dataset.yaml
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark YOLO model inference")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="Model paths to compare (.pt, .onnx)")
    parser.add_argument("--data", type=str, default="",
                        help="Dataset config yaml for accuracy benchmark")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of timed iterations")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=str, default="runs/benchmark",
                        help="Directory to save results")
    return parser.parse_args()


def benchmark_latency(model_path: str, imgsz: int, warmup: int, iterations: int) -> dict:
    """Benchmark inference latency."""
    model = YOLO(model_path)
    img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(warmup):
        model(img, verbose=False)

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.time()
        model(img, verbose=False)
        times.append(time.time() - start)

    times_ms = np.array(times) * 1000
    return {
        "mean_ms": round(float(np.mean(times_ms)), 2),
        "p50_ms": round(float(np.percentile(times_ms, 50)), 2),
        "p95_ms": round(float(np.percentile(times_ms, 95)), 2),
        "p99_ms": round(float(np.percentile(times_ms, 99)), 2),
        "fps": round(float(1000 / np.mean(times_ms)), 1),
    }


def benchmark_accuracy(model_path: str, data: str, imgsz: int) -> dict:
    """Benchmark model accuracy on validation set."""
    model = YOLO(model_path)
    metrics = model.val(data=data, imgsz=imgsz, verbose=False)
    return {
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }


def main():
    args = parse_args()
    results = []

    print(f"\n{'='*70}")
    print(f"  Inference Benchmark")
    print(f"  Models:     {len(args.models)}")
    print(f"  Iterations: {args.warmup} warmup + {args.iterations} timed")
    print(f"  Image size: {args.imgsz}")
    print(f"{'='*70}\n")

    for model_path in args.models:
        path = Path(model_path)
        if not path.exists():
            print(f"  SKIP: {model_path} (not found)")
            continue

        print(f"  Benchmarking: {path.name} ...")
        entry = {
            "model": path.name,
            "format": path.suffix,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        }

        # Latency
        entry.update(benchmark_latency(model_path, args.imgsz, args.warmup, args.iterations))

        # Accuracy (optional)
        if args.data:
            entry.update(benchmark_accuracy(model_path, args.data, args.imgsz))

        results.append(entry)

    # --- Print comparison table ---
    print(f"\n{'='*70}")
    print(f"  Results")
    print(f"{'='*70}")

    header_cols = ["Model", "Format", "Size(MB)", "Mean(ms)", "P95(ms)", "FPS"]
    if args.data:
        header_cols += ["mAP50", "mAP50-95"]

    header = "  ".join(f"{h:>10}" for h in header_cols)
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    for r in results:
        row = [
            f"{r['model']:>10}",
            f"{r['format']:>10}",
            f"{r['size_mb']:>10.1f}",
            f"{r['mean_ms']:>10.1f}",
            f"{r['p95_ms']:>10.1f}",
            f"{r['fps']:>10.1f}",
        ]
        if args.data:
            row += [
                f"{r.get('mAP50', 'N/A'):>10}",
                f"{r.get('mAP50-95', 'N/A'):>10}",
            ]
        print(f"  {'  '.join(row)}")

    print(f"{'='*70}\n")

    # Save results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = save_dir / f"benchmark_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
