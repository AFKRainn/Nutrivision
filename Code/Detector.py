"""YOLOv12 ingredient detector for NutriVision (loads best.pt once)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
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

DEFAULT_CONF = 0.20

# Garbled/duplicate class names from the 5-dataset merge → canonical name
CANONICAL_NAMES: dict[str, str] = {
    "tomatoes": "tomato",
    "totomat": "tomato",
    "tomattomato": "tomato",
    "tomatoe": "tomato",
    "cherry tomato": "tomato",
    "eggs": "egg",
    "egg white": "egg",
    "peppers": "pepper",
    "bell pepper": "pepper",
    "green pepper": "pepper",
    "red pepper": "pepper",
    "carrots": "carrot",
    "apples": "apple",
    "onions": "onion",
    "milks": "milk",
    "butters": "butter",
    "cucumbers": "cucumber",
    "lemons": "lemon",
    "oranges": "orange",
    "potatoes": "potato",
    "garlic clove": "garlic",
    "garlics": "garlic",
}

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


def _normalize_name(name: str) -> str:
    return CANONICAL_NAMES.get(name.lower(), name.lower())


def _predict(source: str | np.ndarray, conf: float) -> list[dict]:
    """Run YOLO prediction with TTA on a path or numpy array."""
    model = get_model()
    results = model.predict(source=source, conf=conf, verbose=False, augment=True)
    result = results[0]
    dets: list[dict] = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            dets.append(
                {
                    "name": _normalize_name(result.names[cls_id]),
                    "confidence": float(box.conf.item()),
                    "class_id": cls_id,
                }
            )
    return dets


def _zoom_crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Crop a detected region with 25% padding and scale to 640×640."""
    h, w = img.shape[:2]
    pw = int((x2 - x1) * 0.25)
    ph = int((y2 - y1) * 0.25)
    x1 = max(0, x1 - pw)
    y1 = max(0, y1 - ph)
    x2 = min(w, x2 + pw)
    y2 = min(h, y2 + ph)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return crop
    return cv2.resize(crop, (640, 640), interpolation=cv2.INTER_LINEAR)


def _quadrant_tiles(img: np.ndarray, overlap: float = 0.15) -> list[np.ndarray]:
    """Split image into 4 overlapping quadrants."""
    h, w = img.shape[:2]
    mh, mw = h // 2, w // 2
    ph, pw = int(h * overlap), int(w * overlap)
    return [
        img[0 : mh + ph, 0 : mw + pw],
        img[0 : mh + ph, max(0, mw - pw) : w],
        img[max(0, mh - ph) : h, 0 : mw + pw],
        img[max(0, mh - ph) : h, max(0, mw - pw) : w],
    ]


def detect_image(
    image_path: Path,
    conf: float = DEFAULT_CONF,
    save_annotated: bool = True,
) -> tuple[list[dict], Path | None]:
    """Three-pass detection: full frame → zoom into each detection → quadrant tiles.

    Pass 1 (full frame + TTA): standard detection with test-time augmentation.
    Pass 2 (zoom): each detected bounding box is cropped, padded, scaled to
      640×640 and re-detected — catches details missed at full scale.
    Pass 3 (tiles): 4 overlapping quadrants catch items near edges or in
      corners that the full frame under-represents.
    All passes use TTA. Results are merged by merge_detections (best conf per name).
    """
    image_path = Path(image_path)
    model = get_model()
    img = cv2.imread(str(image_path))

    # --- Pass 1: full frame with TTA ---
    results = model.predict(source=str(image_path), conf=conf, verbose=False, augment=True)
    result = results[0]

    all_detections: list[dict] = []
    boxes_xyxy: list[tuple[int, int, int, int]] = []

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            all_detections.append(
                {
                    "name": _normalize_name(result.names[cls_id]),
                    "confidence": float(box.conf.item()),
                    "class_id": cls_id,
                }
            )
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            boxes_xyxy.append((x1, y1, x2, y2))

    annotated_path: Path | None = None
    if save_annotated:
        annotated_path = image_path.with_name(f"{image_path.stem}_detected.jpg")
        cv2.imwrite(str(annotated_path), result.plot())

    # --- Pass 2: zoom into every detected region ---
    for (x1, y1, x2, y2) in boxes_xyxy:
        zoomed = _zoom_crop(img, x1, y1, x2, y2)
        if zoomed.size == 0:
            continue
        all_detections.extend(_predict(zoomed, conf))

    # --- Pass 3: quadrant tiles ---
    for tile in _quadrant_tiles(img):
        if tile.size == 0:
            continue
        all_detections.extend(_predict(tile, conf))

    return all_detections, annotated_path


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
