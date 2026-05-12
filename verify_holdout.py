"""
Quick held-out accuracy check.

Reads data/raw/test.csv (clips the model never saw during training) and
reports top-1 accuracy on a random sample. This tells us how well the
model generalizes within the FSL-105 distribution.
"""

import csv
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from verify_model import predict_clip


SAMPLE_N = 30  # how many random test clips to evaluate


def main():
    test_csv = _ROOT / "data" / "raw" / "test.csv"
    rows = []
    with open(test_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    random.seed(42)
    sample = random.sample(rows, min(SAMPLE_N, len(rows)))

    correct = 0
    print(f"Sampling {len(sample)} held-out clips from data/raw/test.csv\n")
    for r in sample:
        vid_rel = r["vid_path"].replace("\\", "/")
        expected = r["label"].strip()
        path = _ROOT / "data" / "raw" / vid_rel
        if not path.exists():
            print(f"  [miss] {vid_rel} (file missing)")
            continue
        top5 = predict_clip(str(path), do_segment=False, flip_h=False)
        pred = top5[0][0].strip()
        ok = pred == expected
        if ok:
            correct += 1
        mark = "OK" if ok else "X"
        print(f"  [{mark}] {vid_rel:30s} expected={expected:20s} got={pred:20s} ({top5[0][1]*100:.0f}%)")

    print(f"\nTop-1 accuracy: {correct}/{len(sample)} = {correct/len(sample)*100:.1f}%")


if __name__ == "__main__":
    main()
