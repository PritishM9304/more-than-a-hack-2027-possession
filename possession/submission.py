"""Submission CSV writer and structural validator."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

VALID = {"H", "A", "D", "X"}
EXPECTED_FRAMES = 199_319


def write_submission(path: str | Path, labels: np.ndarray) -> Path:
    labels = np.asarray(labels).astype(str)
    if labels.ndim != 1 or len(labels) != EXPECTED_FRAMES:
        raise ValueError(f"Expected exactly {EXPECTED_FRAMES} labels, got shape {labels.shape}")
    bad = sorted(set(labels) - VALID)
    if bad:
        raise ValueError(f"Invalid labels: {bad}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["frame", "label"])
        writer.writerows((i, label) for i, label in enumerate(labels))
    return path


def validate_submission(path: str | Path, expected_frames: int = EXPECTED_FRAMES) -> dict:
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["frame", "label"]:
            raise ValueError(f"Header must be exactly frame,label; got {reader.fieldnames}")
        count = 0
        first = last = None
        counts = {k: 0 for k in sorted(VALID)}
        for count, row in enumerate(reader, start=1):
            frame = int(row["frame"])
            label = (row["label"] or "").strip().upper()
            expected_frame = count - 1
            if frame != expected_frame:
                raise ValueError(f"Frame order/index error at row {count}: expected {expected_frame}, got {frame}")
            if label not in VALID:
                raise ValueError(f"Invalid label {label!r} at frame {frame}")
            counts[label] += 1
            if first is None:
                first = frame
            last = frame
    if count != expected_frames:
        raise ValueError(f"Expected {expected_frames} data rows; found {count}")
    return {"rows": count, "first_frame": first, "last_frame": last, "label_counts": counts}
