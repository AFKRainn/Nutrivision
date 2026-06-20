"""YOLOv12 ingredient detector for NutriVision (loads best.pt once)."""
from __future__ import annotations

from pathlib import Path

import cv2
from ultralytics import YOLO

CODE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = (
    CODE_DIR
    / "Training model"
    / "runs"
    / "nutrivision_merged_final"
    / "weights"
    / "best.pt"
)

DEFAULT_CONF = 0.25

_model: YOLO | None = None


def weights_path() -> Path:
    return WEIGHTS_PATH


def weights_available() -> bool:
    return WEIGHTS_PATH.is_file()


def get_model() -> YOLO:
    global _model
    if _model is None:
        if not WEIGHTS_PATH.is_file():
            raise FileNotFoundError(
                f"trained weights not found: {WEIGHTS_PATH}\n"
                "Run train.py first or check the path."
            )
        _model = YOLO(str(WEIGHTS_PATH))
    return _model


def detect_image(
    image_path: Path,
    conf: float = DEFAULT_CONF,
    save_annotated: bool = True,
) -> tuple[list[dict], Path | None]:
    """Run detection on one image. Returns (detections, annotated_image_path)."""
    image_path = Path(image_path)
    model = get_model()
    results = model.predict(source=str(image_path), conf=conf, verbose=False)
    result = results[0]

    detections: list[dict] = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            detections.append(
                {
                    "name": result.names[cls_id],
                    "confidence": float(box.conf.item()),
                    "class_id": cls_id,
                }
            )

    annotated_path: Path | None = None
    if save_annotated:
        annotated_path = image_path.with_name(f"{image_path.stem}_detected.jpg")
        cv2.imwrite(str(annotated_path), result.plot())

    return detections, annotated_path


def merge_detections(detection_lists: list[list[dict]]) -> dict[str, float]:
    """Merge detections from multiple images: one entry per name, highest confidence wins."""
    merged: dict[str, float] = {}
    for dets in detection_lists:
        for d in dets:
            name = d["name"]
            conf = d["confidence"]
            if name not in merged or conf > merged[name]:
                merged[name] = conf
    return dict(sorted(merged.items(), key=lambda x: (-x[1], x[0])))
