import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from live_demo.labels import load_labels


def test_load_labels_count():
    labels = load_labels()
    assert len(labels) == 105


def test_load_labels_structure():
    labels = load_labels()
    entry = labels[0]
    assert "label" in entry
    assert "category" in entry
    assert entry["label"] == "GOOD MORNING"
    assert entry["category"] == "GREETING"


def test_load_labels_lookup():
    labels = load_labels()
    assert labels[3]["label"] == "HELLO"
    assert labels[3]["category"] == "GREETING"
