"""
Run the v3 live-demo pipeline on a real-world video (stranger signing 1-10).

This is the closest we can get to "what the live demo would see" without
actually pointing the webcam at the video. We replicate exactly what
live_demo/keypoints.py + live_demo/inference_v3.py do, frame by frame.
"""

import sys
import time
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
from live_demo.inference_v3 import _resample_to_target_fps


STILLNESS_MS   = 500
MIN_SEGMENT_MS = 600
MAX_SEGMENT_MS = 6000


def run_pipeline_on_video(video_path: str, flip_h: bool = False, print_per_frame: bool = False) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERR] cannot open {video_path}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / src_fps
    print(f"video: {Path(video_path).name}  fps={src_fps:.1f}  frames={n_frames}  dur={duration:.1f}s  flip_h={flip_h}")

    models = create_models(seg_model=1, detection_conf=0.35, tracking_conf=0.25)
    model, device = load_model()
    labels = load_labels()

    # State (mirrors live_demo/inference_v3.py)
    buffer: list[tuple[float, np.ndarray]] = []
    active = False
    segment_start_t = 0.0
    last_hand_t = 0.0

    segments = []

    frame_idx = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            t_sec = frame_idx / src_fps  # use video timestamp, not wall clock

            frame_bgr_resized, _ = resize_with_aspect_ratio_and_pad(frame_bgr, target_size=256)
            if flip_h:
                frame_bgr_resized = cv2.flip(frame_bgr_resized, 1)
            frame_rgb = cv2.cvtColor(frame_bgr_resized, cv2.COLOR_BGR2RGB)
            vec178, mask89 = extract_keypoints_from_frame(frame_rgb, models, conf_thresh=0.35)
            vec178 = np.clip(vec178, 0.0, 1.0).astype(np.float32)
            hands = bool(mask89[25:67].any())

            if hands:
                if not active:
                    active = True
                    segment_start_t = t_sec
                    buffer = []
                last_hand_t = t_sec
                buffer.append((t_sec, vec178.copy()))
                if (t_sec - segment_start_t) * 1000 >= MAX_SEGMENT_MS:
                    segments.append(list(buffer)); buffer = []; active = False
            else:
                if active and (t_sec - last_hand_t) * 1000 >= STILLNESS_MS:
                    if buffer:
                        segments.append(list(buffer))
                    buffer = []; active = False

            frame_idx += 1
            if print_per_frame and frame_idx % 30 == 0:
                print(f"  frame={frame_idx} hands={hands} active={active} buf={len(buffer)}")
    finally:
        # close the trailing segment if still active
        if buffer:
            segments.append(list(buffer))
        cap.release()
        close_models(models)

    print(f"detected {len(segments)} segment(s)")
    print()

    for i, seg in enumerate(segments):
        duration_ms = int((seg[-1][0] - seg[0][0]) * 1000)
        if duration_ms < MIN_SEGMENT_MS:
            print(f"  seg#{i+1:02d} dur={duration_ms}ms  frames={len(seg)}  [SKIPPED — too short]")
            continue

        arr = _resample_to_target_fps(seg)
        x = torch.from_numpy(arr).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=-1)[0]
        top5_v, top5_i = torch.topk(probs, k=5)
        top5 = [(labels[int(idx)]["label"].strip(), float(v)) for v, idx in zip(top5_v, top5_i)]
        line = ", ".join(f"{l}({c:.3f})" for l, c in top5)
        print(f"  seg#{i+1:02d} t={seg[0][0]:.2f}-{seg[-1][0]:.2f}s frames={len(seg)}->{int(arr.shape[0])} : {line}")


if __name__ == "__main__":
    video = str(_ROOT / "data/test/hello.mp4")
    for flip in (False, True):
        print("="*80)
        run_pipeline_on_video(video, flip_h=flip)
        print()
