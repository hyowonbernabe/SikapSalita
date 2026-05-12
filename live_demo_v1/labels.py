from pathlib import Path
import csv

_CSV_PATH = Path(__file__).parent.parent / "data" / "labels_reference.csv"


def load_labels() -> dict[int, dict]:
    """Return {gloss_id: {"label": str, "category": str}} from labels_reference.csv."""
    result = {}
    with open(_CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[int(row["gloss_id"])] = {
                "label": row["label"],
                "category": row["category"],
            }
    return result
