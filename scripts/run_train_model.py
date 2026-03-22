"""
模型训练脚本 — 在 RPS 手势数据集上微调 YOLOv8n 和 YOLOv5n。

用法:
    source venv/bin/activate
    python scripts/run_train_model.py                    # 默认参数
    python scripts/run_train_model.py --epochs 50        # 自定义轮数
    python scripts/run_train_model.py --model v8         # 只训练 YOLOv8
    python scripts/run_train_model.py --batch 16         # 调整 batch size
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def abs_path(relative: str) -> str:
    return str(PROJECT_ROOT / relative)


def fix_dataset_path(data_path: str) -> None:
    """将数据集 yaml 中的 path 更新为当前机器上的实际路径。"""
    with open(data_path, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    if 'path' in data:
        data['path'] = os.path.dirname(data_path.replace(os.sep, '/'))
        with open(data_path, 'w') as f:
            yaml.safe_dump(data, file=f, sort_keys=False)


def train_model(weights: str, data_path: str, device: str,
                epochs: int, batch: int, workers: int, name: str):
    print(f"\n{'='*50}")
    print(f"  训练: {name}")
    print(f"  权重: {weights}")
    print(f"  设备: {device}")
    print(f"  轮数: {epochs}, batch: {batch}")
    print(f"{'='*50}\n")

    model = YOLO(weights, task='detect')
    model.train(
        data=data_path,
        device=device,
        workers=workers,
        imgsz=640,
        epochs=epochs,
        batch=batch,
        name=name,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练 YOLO 手势识别模型')
    parser.add_argument('--epochs', type=int, default=120, help='训练轮数 (默认: 120)')
    parser.add_argument('--batch', type=int, default=8, help='batch size (默认: 8)')
    parser.add_argument('--workers', type=int, default=1, help='数据加载线程数 (默认: 1)')
    parser.add_argument('--model', choices=['v8', 'v5', 'both'], default='both',
                        help='训练哪个模型 (默认: both)')
    parser.add_argument('--device', type=str, default=None,
                        help='训练设备, 如 cpu/0/mps (默认: 自动检测)')
    args = parser.parse_args()

    device = args.device or get_device()
    data_name = "RPS"
    data_path = abs_path(f'datasets/{data_name}/{data_name}.yaml')

    fix_dataset_path(data_path)

    print(f"数据集: {data_path}")
    print(f"设备: {device}")

    if args.model in ('v8', 'both'):
        train_model(
            weights=abs_path('weights/yolov8n.pt'),
            data_path=data_path, device=device,
            epochs=args.epochs, batch=args.batch, workers=args.workers,
            name=f'train_v8_{data_name}',
        )

    if args.model in ('v5', 'both'):
        train_model(
            weights=abs_path('weights/yolov5nu.pt'),
            data_path=data_path, device=device,
            epochs=args.epochs, batch=args.batch, workers=args.workers,
            name=f'train_v5_{data_name}',
        )
