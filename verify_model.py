"""
Quick check: does the trained model predict HELLO on a training clip?

Feeds data/raw/clips/3/0.MOV through the exact streamlit-style preprocess
(seg → mediapipe → 30fps resample → 178D vec → SignTransformer) and prints
top-5.

If top-1 == HELLO with high confidence → model + checkpoint are fine.
If top-1 != HELLO → model itself is the bug, not the live pipeline.
"""

import sys
from pathlib import Path

import cv2
import math
import numpy as np
import torch

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from preprocessing.extractors.keypoints_features import (
    create_models, close_models, extract_keypoints_from_frame
)
from preprocessing.core.preprocess import resize_with_aspect_ratio_and_pad
from live_demo.labels import load_labels
from live_demo.model import load_model


def predict_clip(video_path: str, target_fps: int = 30, conf_thresh: float = 0.35,
                 do_segment: bool = True, flip_h: bool = False) -> list[tuple[str, float]]:
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step_s = 1.0 / target_fps
    next_t = 0.0

    models = create_models(seg_model=1, detection_conf=conf_thresh, tracking_conf=conf_thresh)
    X = []
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if ms < next_t * 1000.0:
                continue

            frame_bgr_resized, _ = resize_with_aspect_ratio_and_pad(frame_bgr, 256)
            if flip_h:
                frame_bgr_resized = cv2.flip(frame_bgr_resized, 1)
            frame_rgb = cv2.cvtColor(frame_bgr_resized, cv2.COLOR_BGR2RGB)

            if do_segment:
                seg_mask = models.seg.process(frame_rgb).segmentation_mask
                fg = (seg_mask > 0.5).astype(np.float32)[..., None]
                frame_rgb = (frame_rgb * fg).astype(np.uint8)

            vec178, _ = extract_keypoints_from_frame(frame_rgb, models, conf_thresh=conf_thresh)
            X.append(np.clip(vec178, 0.0, 1.0).astype(np.float32))
            next_t += step_s
    finally:
        cap.release()
        close_models(models)

    if not X:
        print(f"[ERR] No frames extracted from {video_path}")
        return []

    arr = np.stack(X, axis=0)  # (T, 178)
    print(f"[{Path(video_path).name}] segment={do_segment} frames={len(arr)}")

    model, device = load_model()
    labels = load_labels()
    x = torch.from_numpy(arr).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(x)
        probs = torch.softmax(logits, dim=-1)[0]
    top_v, top_i = torch.topk(probs, k=5)
    return [(labels[int(i)]["label"], float(v)) for v, i in zip(top_v, top_i)]


if __name__ == "__main__":
    path = str(_ROOT / "data/test/hello.mp4")
    if not Path(path).exists():
        print(f"[skip] {path} not found")
    else:
        for flip in (False, True):
            top5 = predict_clip(path, do_segment=False, flip_h=flip)
            tag = "FLIPPED" if flip else "NORMAL "
            print(f"  {tag} hello.mp4 (whole-clip): " + ", ".join(f"{l.strip()}({c:.3f})" for l, c in top5))
