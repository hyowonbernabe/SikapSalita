"""
Sliding-window inference. No segmentation. Just slide a window over the
video and predict on each chunk. Bypasses the hand-drop requirement.
"""

import sys
from pathlib import Path

import cv2
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


WINDOW_SEC = 2.0
STRIDE_SEC = 0.5


def slide_predict(video_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / src_fps
    print(f"video: {Path(video_path).name}  fps={src_fps:.1f}  frames={n_frames}  dur={duration:.1f}s")

    models = create_models(seg_model=1, detection_conf=0.35, tracking_conf=0.25)
    model, device = load_model()
    labels = load_labels()

    # Extract keypoints for every frame first
    keypoints_per_frame: list[np.ndarray] = []
    timestamps: list[float] = []
    idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_bgr_resized, _ = resize_with_aspect_ratio_and_pad(frame_bgr, target_size=256)
        frame_rgb = cv2.cvtColor(frame_bgr_resized, cv2.COLOR_BGR2RGB)
        vec178, _ = extract_keypoints_from_frame(frame_rgb, models, conf_thresh=0.35)
        keypoints_per_frame.append(np.clip(vec178, 0.0, 1.0).astype(np.float32))
        timestamps.append(idx / src_fps)
        idx += 1
    cap.release()
    close_models(models)

    print(f"extracted {len(keypoints_per_frame)} keypoint frames\n")

    window_frames = int(WINDOW_SEC * src_fps)
    stride_frames = int(STRIDE_SEC * src_fps)

    t = 0
    while t + window_frames <= len(keypoints_per_frame):
        chunk = np.stack(keypoints_per_frame[t:t + window_frames], axis=0)
        x = torch.from_numpy(chunk).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=-1)[0]
        top3_v, top3_i = torch.topk(probs, k=3)
        top3 = [(labels[int(i)]["label"].strip(), float(v)) for v, i in zip(top3_v, top3_i)]
        line = ", ".join(f"{l}({c:.3f})" for l, c in top3)
        print(f"  win {timestamps[t]:.2f}-{timestamps[t+window_frames-1]:.2f}s : {line}")
        t += stride_frames


if __name__ == "__main__":
    for vid in ["yes_no_test.mp4", "number_test.mp4"]:
        path = str(_ROOT / "data/test" / vid)
        print("="*80)
        slide_predict(path)
        print()
