"""
ONNX 模型导出脚本 (Model Export to ONNX)

Usage:
    # Export to ONNX
    python training/export_onnx.py --model backend/weights/best-yolov8n.pt

    # Export with INT8 quantization
    python training/export_onnx.py --model backend/weights/best-yolov8n.pt --quantize int8

    # Export and benchmark
    python training/export_onnx.py --model backend/weights/best-yolov8n.pt --benchmark
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX format")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to .pt model weights")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--output-dir", type=str, default="runs/export",
                        help="Output directory")
    parser.add_argument("--opset", type=int, default=12,
                        help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true", default=True,
                        help="Simplify ONNX graph with onnxsim")
    parser.add_argument("--dynamic", action="store_true",
                        help="Enable dynamic batch size")
    parser.add_argument("--quantize", type=str, choices=["none", "fp16", "int8"], default="none",
                        help="Post-training quantization type")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark comparison after export")
    parser.add_argument("--benchmark-iterations", type=int, default=50,
                        help="Number of inference iterations for benchmarking")
    return parser.parse_args()


def export_onnx(args) -> Path:
    """Export model to ONNX using ultralytics API."""
    print(f"\n{'='*60}")
    print(f"  ONNX Export")
    print(f"  Model:    {args.model}")
    print(f"  Size:     {args.imgsz}")
    print(f"  Opset:    {args.opset}")
    print(f"  Simplify: {args.simplify}")
    print(f"  Dynamic:  {args.dynamic}")
    print(f"{'='*60}\n")

    model = YOLO(args.model)
    export_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=args.dynamic,
    )

    export_path = Path(export_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dest = output_dir / export_path.name
    if export_path != dest:
        shutil.copy2(export_path, dest)

    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  Exported: {dest} ({size_mb:.1f} MB)")

    return dest


def quantize_model(onnx_path: Path, quantize_type: str) -> Path:
    """Apply post-training quantization."""
    if quantize_type == "none":
        return onnx_path

    from onnxruntime.quantization import QuantType, quantize_dynamic

    stem = onnx_path.stem
    parent = onnx_path.parent

    if quantize_type == "int8":
        out_path = parent / f"{stem}_int8.onnx"
        quantize_dynamic(str(onnx_path), str(out_path), weight_type=QuantType.QInt8)
    elif quantize_type == "fp16":
        out_path = parent / f"{stem}_fp16.onnx"
        quantize_dynamic(str(onnx_path), str(out_path), weight_type=QuantType.QUInt8)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Quantized ({quantize_type}): {out_path} ({size_mb:.1f} MB)")
    return out_path


def benchmark_comparison(pt_model_path: str, onnx_path: Path, imgsz: int, iterations: int):
    """Compare PyTorch vs ONNX inference speed."""
    import numpy as np

    print(f"\n{'='*60}")
    print(f"  Benchmark: PyTorch vs ONNX ({iterations} iterations)")
    print(f"{'='*60}\n")

    img = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # --- PyTorch ---
    pt_model = YOLO(pt_model_path)
    pt_model(img, verbose=False)  # warmup

    pt_times = []
    for _ in range(iterations):
        start = time.time()
        pt_model(img, verbose=False)
        pt_times.append(time.time() - start)

    # --- ONNX ---
    onnx_model = YOLO(str(onnx_path))
    onnx_model(img, verbose=False)  # warmup

    onnx_times = []
    for _ in range(iterations):
        start = time.time()
        onnx_model(img, verbose=False)
        onnx_times.append(time.time() - start)

    # --- Report ---
    pt_times_ms = np.array(pt_times) * 1000
    onnx_times_ms = np.array(onnx_times) * 1000

    pt_size = Path(pt_model_path).stat().st_size / (1024 * 1024)
    onnx_size = onnx_path.stat().st_size / (1024 * 1024)

    print(f"  {'Metric':<20} {'PyTorch':>12} {'ONNX':>12}")
    print(f"  {'-'*44}")
    print(f"  {'Mean latency (ms)':<20} {np.mean(pt_times_ms):>12.1f} {np.mean(onnx_times_ms):>12.1f}")
    print(f"  {'P50 (ms)':<20} {np.percentile(pt_times_ms, 50):>12.1f} {np.percentile(onnx_times_ms, 50):>12.1f}")
    print(f"  {'P95 (ms)':<20} {np.percentile(pt_times_ms, 95):>12.1f} {np.percentile(onnx_times_ms, 95):>12.1f}")
    print(f"  {'P99 (ms)':<20} {np.percentile(pt_times_ms, 99):>12.1f} {np.percentile(onnx_times_ms, 99):>12.1f}")
    print(f"  {'FPS':<20} {1000/np.mean(pt_times_ms):>12.1f} {1000/np.mean(onnx_times_ms):>12.1f}")
    print(f"  {'Model size (MB)':<20} {pt_size:>12.1f} {onnx_size:>12.1f}")

    speedup = np.mean(pt_times_ms) / np.mean(onnx_times_ms)
    compression = pt_size / onnx_size if onnx_size > 0 else 0
    print(f"\n  Speedup: {speedup:.2f}x")
    print(f"  Size ratio: {compression:.2f}x")
    print(f"{'='*60}\n")


def deploy_to_backend(onnx_path: Path):
    """Copy ONNX model to backend weights directory."""
    deploy_dir = PROJECT_ROOT / "backend" / "weights"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    dest = deploy_dir / onnx_path.name
    shutil.copy2(onnx_path, dest)
    print(f"  Deployed to: {dest}")


def main():
    args = parse_args()

    # Export
    onnx_path = export_onnx(args)

    # Quantize
    if args.quantize != "none":
        onnx_path = quantize_model(onnx_path, args.quantize)

    # Deploy
    deploy_to_backend(onnx_path)

    # Benchmark
    if args.benchmark:
        benchmark_comparison(args.model, onnx_path, args.imgsz, args.benchmark_iterations)

    print("  Done!")


if __name__ == "__main__":
    main()
