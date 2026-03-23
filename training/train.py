"""
Hand Gesture YOLO Fine-tuning Script

Usage:
    # Default: fine-tune yolov8s on hand gesture dataset
    python training/train.py

    # Specify model size and epochs
    python training/train.py --model yolov8m.pt --epochs 200

    # Use a custom config
    python training/train.py --config training/configs/finetune_v8s.yaml

    # Resume interrupted training
    python training/train.py --resume
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO for hand gesture recognition")

    # Model
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                        help="Pretrained model to fine-tune (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--data", type=str, default="training/dataset.yaml",
                        help="Path to dataset config yaml")

    # Training params
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="",
                        help="cuda device (0, 0,1, cpu)")

    # Fine-tuning strategy
    parser.add_argument("--freeze", type=int, default=10,
                        help="Freeze first N layers (0=train all, 10=freeze backbone)")
    parser.add_argument("--lr0", type=float, default=0.001,
                        help="Initial learning rate (lower than default 0.01 for fine-tuning)")
    parser.add_argument("--optimizer", type=str, default="AdamW",
                        choices=["SGD", "Adam", "AdamW", "NAdam", "RAdam", "auto"])
    parser.add_argument("--cos-lr", action="store_true", default=True,
                        help="Use cosine learning rate scheduler")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience")

    # Augmentation
    parser.add_argument("--degrees", type=float, default=15.0,
                        help="Rotation augmentation degrees")
    parser.add_argument("--mixup", type=float, default=0.1,
                        help="Mixup augmentation probability")
    parser.add_argument("--close-mosaic", type=int, default=20,
                        help="Disable mosaic for last N epochs")

    # Experiment management
    parser.add_argument("--project", type=str, default="runs/gesture",
                        help="Project directory for saving results")
    parser.add_argument("--name", type=str, default="",
                        help="Experiment name (auto-generated if empty)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")

    # Config file (overrides all above if provided)
    parser.add_argument("--config", type=str, default="",
                        help="Path to experiment config yaml (overrides CLI args)")

    return parser.parse_args()


def build_experiment_name(args) -> str:
    """Generate a descriptive experiment name."""
    if args.name:
        return args.name
    model_tag = Path(args.model).stem  # e.g. yolov8s
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{model_tag}_freeze{args.freeze}_lr{args.lr0}_{timestamp}"


def load_config_overrides(config_path: str) -> dict:
    """Load training overrides from a yaml config file."""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def train(args):
    # Load model
    print(f"\n{'='*60}")
    print(f"  Fine-tuning: {args.model}")
    print(f"  Dataset:     {args.data}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Freeze:      first {args.freeze} layers")
    print(f"  LR:          {args.lr0}")
    print(f"{'='*60}\n")

    model = YOLO(args.model)

    # Build training kwargs
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        freeze=args.freeze,
        optimizer=args.optimizer,
        lr0=args.lr0,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        degrees=args.degrees,
        mixup=args.mixup,
        flipud=0.0,          # Hand gestures should not be flipped vertically
        fliplr=0.5,
        project=args.project,
        name=build_experiment_name(args),
        pretrained=True,
        resume=args.resume,
        plots=True,          # Save training curves and confusion matrix
        save=True,
        save_period=10,      # Save checkpoint every 10 epochs
        verbose=True,
    )

    if args.device:
        train_kwargs["device"] = args.device

    # Override with config file if provided
    if args.config:
        overrides = load_config_overrides(args.config)
        train_kwargs.update(overrides)
        print(f"  Config overrides from: {args.config}")
        for k, v in overrides.items():
            print(f"    {k}: {v}")
        print()

    # Train
    results = model.train(**train_kwargs)

    # Print results summary
    best_weight = Path(args.project) / build_experiment_name(args) / "weights" / "best.pt"
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best weights: {best_weight}")
    print(f"{'='*60}\n")

    # Copy best weights to backend for deployment
    deploy_path = PROJECT_ROOT / "backend" / "weights"
    deploy_path.mkdir(parents=True, exist_ok=True)
    if best_weight.exists():
        import shutil
        dest = deploy_path / f"best-{Path(args.model).stem}-finetuned.pt"
        shutil.copy2(best_weight, dest)
        print(f"  Deployed to: {dest}")
        print(f"  Update backend/core/config.py model_path to use this weight.\n")

    return results


if __name__ == "__main__":
    args = parse_args()
    train(args)
