import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from live_demo.inference import FrameBuffer, smooth_predictions


def test_framebuffer_warmup():
    buf = FrameBuffer(max_frames=120, min_frames=30)
    buf.push(np.zeros(178, dtype=np.float32))
    assert not buf.ready()


def test_framebuffer_ready_after_min_frames():
    buf = FrameBuffer(max_frames=120, min_frames=30)
    for _ in range(30):
        buf.push(np.random.rand(178).astype(np.float32))
    assert buf.ready()


def test_framebuffer_respects_max():
    buf = FrameBuffer(max_frames=10, min_frames=3)
    for _ in range(20):
        buf.push(np.ones(178, dtype=np.float32))
    assert len(buf) == 10


def test_smooth_predictions_single():
    preds = [{"label": "HELLO", "category": "GREETING", "confidence": 0.9}]
    history = []
    result = smooth_predictions(preds, history, window=3)
    assert result[0]["label"] == "HELLO"
    assert abs(result[0]["confidence"] - 0.9) < 1e-5


def test_smooth_predictions_averages():
    history = [
        [{"label": "HELLO", "category": "GREETING", "confidence": 0.8},
         {"label": "THANK YOU", "category": "GREETING", "confidence": 0.2}],
        [{"label": "HELLO", "category": "GREETING", "confidence": 0.6},
         {"label": "THANK YOU", "category": "GREETING", "confidence": 0.4}],
    ]
    current = [
        {"label": "HELLO", "category": "GREETING", "confidence": 1.0},
        {"label": "THANK YOU", "category": "GREETING", "confidence": 0.0},
    ]
    result = smooth_predictions(current, history, window=3)
    # avg of 0.8, 0.6, 1.0 = 0.8
    assert abs(result[0]["confidence"] - 0.8) < 1e-4
