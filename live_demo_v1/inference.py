"""
Rolling frame buffer and inference engine for the live FSL demo.
"""

from __future__ import annotations

import collections
from collections import defaultdict
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from live_demo.labels import load_labels
from live_demo.model import load_model

_LABELS: dict[int, dict] | None = None


def get_labels() -> dict[int, dict]:
    global _LABELS
    if _LABELS is None:
        _LABELS = load_labels()
    return _LABELS


class FrameBuffer:
    """Rolling buffer of keypoint vectors for a single user session."""

    def __init__(self, max_frames: int = 120, min_frames: int = 30):
        self._deque: collections.deque[np.ndarray] = collections.deque(maxlen=max_frames)
        self._min_frames = min_frames

    def push(self, vec178: np.ndarray) -> None:
        self._deque.append(vec178)

    def ready(self) -> bool:
        return len(self._deque) >= self._min_frames

    def __len__(self) -> int:
        return len(self._deque)

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        """Return (1, T, 178) float32 tensor from current buffer contents."""
        arr = np.stack(list(self._deque), axis=0)        # (T, 178)
        return torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, T, 178)


def smooth_predictions(
    current: list[dict],
    history: list[list[dict]],
    window: int = 3,
) -> list[dict]:
    """Average confidence scores over `window` consecutive prediction calls.

    Returns a new list sorted by smoothed confidence descending.
    """
    recent = (history + [current])[-window:]
    if len(recent) == 1:
        return current

    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    meta: dict[str, str] = {}

    for frame_preds in recent:
        for p in frame_preds:
            sums[p["label"]] += p["confidence"]
            counts[p["label"]] += 1
            meta[p["label"]] = p["category"]

    smoothed = [
        {"label": lbl, "category": meta[lbl], "confidence": sums[lbl] / counts[lbl]}
        for lbl in sums
    ]
    smoothed.sort(key=lambda x: x["confidence"], reverse=True)
    return smoothed[: len(current)]


# Global buffer and prediction history (single-user demo)
_buffer = FrameBuffer(max_frames=120, min_frames=30)
_pred_history: list[list[dict]] = []


def push_frame(vec178: np.ndarray) -> None:
    _buffer.push(vec178)


def predict_top3() -> list[dict[str, Any]]:
    """Run inference on the current buffer. Returns [] during warmup."""
    if not _buffer.ready():
        return []

    model, device = load_model()
    labels = get_labels()

    with torch.no_grad():
        x = _buffer.to_tensor(device)                      # (1, T, 178)
        gloss_logits, _ = model(x)                         # (1, 105), (1, 10)
        probs = torch.softmax(gloss_logits, dim=-1)[0]     # (105,)

    top3_vals, top3_ids = torch.topk(probs, k=3)
    current = [
        {
            "label": labels[idx.item()]["label"],
            "category": labels[idx.item()]["category"],
            "confidence": round(val.item(), 4),
        }
        for val, idx in zip(top3_vals, top3_ids)
    ]

    global _pred_history
    smoothed = smooth_predictions(current, _pred_history, window=3)
    _pred_history = (_pred_history + [current])[-3:]
    return smoothed


def buffered_frame_count() -> int:
    return len(_buffer)
