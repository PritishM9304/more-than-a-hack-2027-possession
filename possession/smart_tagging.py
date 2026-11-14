"""Parse Smart Tagging SportsCode XML into frame-aligned evidence features."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


KNOWN_CODES = (
    "SET PIECES", "BUILD UP", "PROGRESSION", "ATTACKING TRANSITION",
    "DEFENSIVE TRANSITION", "LONG BALL", "BALL IN FINAL THIRD", "BALL IN THE BOX",
    "PLAYERS IN THE BOX", "CREATING CHANCES", "GOALS",
    "DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN ATTACKING THIRD",
)
RESTART_TYPES = ("Throw-in", "Corner Kick", "Goal Kick", "Free Kick",
                 "Kick Off (After Goal)", "Kick Off (Start)")


def load_smart_tagging(path: str | Path, n_frames: int, fps: int = 25) -> dict[str, np.ndarray]:
    root = ET.parse(path).getroot()
    out: dict[str, np.ndarray] = {
        "tag_active": np.zeros(n_frames, dtype=np.uint8),
        "tag_team_H": np.zeros(n_frames, dtype=np.uint8),
        "tag_team_A": np.zeros(n_frames, dtype=np.uint8),
        "tag_team_NA": np.zeros(n_frames, dtype=np.uint8),
        "tag_set_pieces": np.zeros(n_frames, dtype=np.uint8),
        "tag_restart": np.zeros(n_frames, dtype=np.uint8),
        "tag_kickoff": np.zeros(n_frames, dtype=np.uint8),
        "tag_throw_in": np.zeros(n_frames, dtype=np.uint8),
        "tag_corner": np.zeros(n_frames, dtype=np.uint8),
        "tag_goal_kick": np.zeros(n_frames, dtype=np.uint8),
        "tag_free_kick": np.zeros(n_frames, dtype=np.uint8),
    }
    for code in KNOWN_CODES:
        key = "tag_code_" + code.lower().replace(" ", "_")
        out[key] = np.zeros(n_frames, dtype=np.uint8)

    for inst in root.iter("instance"):
        start_s = float(inst.findtext("start", "nan"))
        end_s = float(inst.findtext("end", "nan"))
        if not np.isfinite(start_s) or not np.isfinite(end_s):
            continue
        # Official loader convention: round seconds*25 and interpret as [start,end).
        start = max(0, int(round(start_s * fps)))
        end = min(n_frames, int(round(end_s * fps)))
        if end <= start:
            continue
        code = (inst.findtext("code") or "").strip().upper()
        tags = {label.findtext("group"): label.findtext("text") for label in inst.findall("label")}
        team = (tags.get("Team") or "").strip().lower()
        kind = (tags.get("Type") or "").strip().lower()
        sl = slice(start, end)
        out["tag_active"][sl] = 1
        if team == "barcelona":
            out["tag_team_H"][sl] = 1
        elif team in {"atletic club", "athletic club"}:
            out["tag_team_A"][sl] = 1
        elif team in {"n/a", "na", ""}:
            out["tag_team_NA"][sl] = 1
        if code == "SET PIECES":
            out["tag_set_pieces"][sl] = 1
        key = "tag_code_" + code.lower().replace(" ", "_")
        if key in out:
            out[key][sl] = 1
        if any(kind == t.lower() for t in RESTART_TYPES):
            out["tag_restart"][sl] = 1
        if "kick off" in kind:
            out["tag_kickoff"][sl] = 1
        if kind == "throw-in":
            out["tag_throw_in"][sl] = 1
        elif kind == "corner kick":
            out["tag_corner"][sl] = 1
        elif kind == "goal kick":
            out["tag_goal_kick"][sl] = 1
        elif kind == "free kick":
            out["tag_free_kick"][sl] = 1
    return out
