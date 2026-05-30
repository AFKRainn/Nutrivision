"""Production training entrypoint for NutriVision.

Run from PowerShell:
    python "F:/Final Project/Nutrivision/Code/Training model/train.py"
"""
from pathlib import Path

import torch
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Data"
MERGED_TRAIN_YAML = DATA_DIR / "merged" / "data.merged.yaml"


def main() -> None:
    if not MERGED_TRAIN_YAML.is_file():
        raise SystemExit(f"missing data yaml: {MERGED_TRAIN_YAML}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available — install torch with cu130 wheel")

    model = YOLO("yolo12n.pt")
    model.train(
        data=str(MERGED_TRAIN_YAML),
        epochs=35,
        imgsz=640,
        batch=16,
        patience=15,
        workers=8,
        seed=0,
        cos_lr=True,
        close_mosaic=10,
        cache="disk",
        project=str(SCRIPT_DIR / "runs"),
        name="nutrivision_merged_final",
        exist_ok=True,
        device=0,
        val=True,
        plots=True,
        save=True,
    )


if __name__ == "__main__":
    main()
