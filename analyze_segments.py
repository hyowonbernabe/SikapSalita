"""
Analyze v3 saved segments. Lists every .npy in data/test/live_segments/
with model's top-5 prediction.

Use this AFTER signing into v3 a few times — each segment captured live is
re-run through the model so you can see exactly what the model thought.

Compares your live keypoints distribution against a target training clip
(default: HELLO clip 3/0.MOV).
"""

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from live_demo.labels import load_labels
from live_demo.model import load_model

SEG_DIR = _ROOT / "data" / "test" / "live_segments"


def predict(arr: np.ndarray, model, device, labels) -> list[tuple[str, float]]:
    x = torch.from_numpy(arr).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(x)
        probs = torch.softmax(logits, dim=-1)[0]
    top_v, top_i = torch.topk(probs, k=5)
    return [(labels[int(i)]["label"].strip(), float(v)) for v, i in zip(top_v, top_i)]


if __name__ == "__main__":
    if not SEG_DIR.exists():
        print(f"no segments saved yet — sign into v3 first to populate {SEG_DIR}")
        sys.exit(0)

    files = sorted(SEG_DIR.glob("seg_*.npy"))
    if not files:
        print(f"no .npy segments in {SEG_DIR}")
        sys.exit(0)

    model, device = load_model()
    labels = load_labels()

    print(f"{len(files)} saved segment(s)\n")
    for f in files:
        arr = np.load(f)
        top5 = predict(arr, model, device, labels)
        line = ", ".join(f"{l}({c:.3f})" for l, c in top5)
        print(f"  {f.name}  T={arr.shape[0]}  : {line}")
