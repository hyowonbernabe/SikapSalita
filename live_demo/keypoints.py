"""
Per-frame MediaPipe keypoint extraction for the live demo.

Initialises MediaPipe Holistic once at module level and exposes
`extract_from_base64(b64_string) -> np.ndarray` which returns
a (178,) float32 vector of normalised keypoint coordinates.
"""

import base64
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from preprocessing.extractors.keypoints_features import (
    MPModels,
    create_models,
    close_models,
    extract_keypoints_from_frame,
)
from preprocessing.core.preprocess import resize_with_aspect_ratio_and_pad

_models: MPModels | None = None


def get_models() -> MPModels:
    global _models
    if _models is None:
        _models = create_models(seg_model=1, detection_conf=0.35, tracking_conf=0.25)
    return _models


def shutdown_models() -> None:
    global _models
    if _models is not None:
        close_models(_models)
        _models = None


def extract_from_jpeg_bytes(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode a raw JPEG bytes object and return (vec178, mask89).

    vec178: (178,) float32 normalised keypoint coordinates
    mask89: (89,) bool — True where a keypoint was actually detected

    Skips background segmentation (used in training pipeline) for demo speed.
    """
    buf = np.frombuffer(data, dtype=np.uint8)
    frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return np.zeros(178, dtype=np.float32), np.zeros(89, dtype=bool)

    frame_bgr_resized, _ = resize_with_aspect_ratio_and_pad(frame_bgr, target_size=256)
    frame_rgb = cv2.cvtColor(frame_bgr_resized, cv2.COLOR_BGR2RGB)

    vec178, mask89 = extract_keypoints_from_frame(frame_rgb, get_models(), conf_thresh=0.35)
    return np.clip(vec178, 0.0, 1.0).astype(np.float32), mask89


def extract_from_base64(b64_string: str) -> tuple[np.ndarray, np.ndarray]:
    """Accept a base64-encoded JPEG string (with or without data-URI prefix).

    Returns (vec178, mask89) — see extract_from_jpeg_bytes.
    """
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    return extract_from_jpeg_bytes(base64.b64decode(b64_string))
