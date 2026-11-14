"""Simple stateful decoder with switch confirmation and explicit abstention."""
from __future__ import annotations

import numpy as np


def decode_probabilities(probabilities: np.ndarray, *, confidence_threshold: float = 0.50,
                         switch_margin: float = 0.15, confirm_frames: int = 3,
                         dead_ball_threshold: float = 0.70) -> np.ndarray:
    """Decode [H,A,D] probabilities to H/A/D/X.

    The decoder is a transparent starter, not a tuned solution. It uses hysteresis for H<->A
    switches, handles D as a distinct state, and outputs X where evidence is insufficient.
    For H/A switching, the alternate team must beat the current state by switch_margin for
    confirm_frames consecutive observations. X never means D.
    """
    p = np.asarray(probabilities, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("probabilities must have shape (N, 3) in H,A,D order")
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("probabilities must be finite and non-negative")
    row_sums = p.sum(axis=1)
    if np.any(row_sums > 1.0001):
        p = p / np.maximum(row_sums[:, None], 1e-8)

    labels = np.full(len(p), "X", dtype="<U1")
    state = -1  # 0=H, 1=A, 2=D, -1=unknown
    candidate = -1
    run = 0
    for i, row in enumerate(p):
        if row[2] >= dead_ball_threshold:
            state, candidate, run = 2, -1, 0
            labels[i] = "D"
            continue
        team = int(np.argmax(row[:2]))
        best = float(row[team])
        second = float(row[1 - team])
        if best < confidence_threshold:
            labels[i] = "X" if state < 0 else ("H" if state == 0 else "A" if state == 1 else "D")
            candidate, run = -1, 0
            continue
        if state in (-1, 2):
            state, candidate, run = team, -1, 0
        elif team != state and best >= second + switch_margin:
            if candidate == team:
                run += 1
            else:
                candidate, run = team, 1
            if run >= max(1, confirm_frames):
                state, candidate, run = team, -1, 0
        else:
            candidate, run = -1, 0
        labels[i] = "H" if state == 0 else "A"
    return labels
