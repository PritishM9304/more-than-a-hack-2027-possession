"""Read the Metrica FIFA/EPTS ATD raw text and metadata files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass
class TrackingData:
    frame: np.ndarray                 # zero-based tactical video frame, shape (N,)
    raw_frame_id: np.ndarray          # one-based ATD counter, shape (N,)
    player_xy_m: np.ndarray           # shape (N, slots, 2), metres
    player_speed_mps: np.ndarray      # shape (N, slots), m/s
    ball_xy_m: np.ndarray             # shape (N, 2), metres
    ball_speed_mps: np.ndarray        # shape (N,), m/s
    team_of_slot: np.ndarray          # 0=home, 1=away, shape (slots,)
    team_names: tuple[str, str]       # (home, away)
    pitch_length_m: float
    pitch_width_m: float


def read_team_mapping(metadata_xml: str | Path) -> tuple[np.ndarray, tuple[str, str], float, float]:
    root = ET.parse(metadata_xml).getroot()
    teams = list(root.iter("Team"))
    players = list(root.iter("Player"))
    if len(teams) != 2:
        raise ValueError(f"Expected exactly 2 teams in {metadata_xml}, found {len(teams)}")
    if not players:
        raise ValueError(f"No Player slots found in {metadata_xml}")

    team_ids = [team.get("id") for team in teams]
    home_id = team_ids[0]  # Official baseline convention: first XML team is home.
    if any(player.get("teamId") not in team_ids for player in players):
        raise ValueError("At least one player slot refers to an unknown teamId")
    team_of_slot = np.asarray([0 if p.get("teamId") == home_id else 1 for p in players], dtype=np.int8)
    names = (teams[0].findtext("Name", "HOME"), teams[1].findtext("Name", "AWAY"))

    field = root.find(".//FieldSize")
    if field is None:
        raise ValueError("Missing FieldSize in ATD metadata")
    length = float(field.findtext("Width", "105"))
    width = float(field.findtext("Height", "68"))
    return team_of_slot, names, length, width


def read_atd(metadata_xml: str | Path, raw_txt: str | Path,
             pitch_override: tuple[float, float] = (105.3, 68.0),
             expected_frames: int | None = 199_319) -> TrackingData:
    """Load tracking into fixed arrays.

    Raw line contract (verified on supplied archive):
        <frame_id>:<slot1>;<slot2>;...;<slotN>:<ball>
    Each slot/ball group is x,y,speed; x and y are normalised to 0..1; speed is m/s.
    The raw ATD counter is 1-based and maps directly to tactical video_frame=frame_id-1.
    """
    team_of_slot, team_names, meta_length, meta_width = read_team_mapping(metadata_xml)
    n_slots = len(team_of_slot)
    length, width = pitch_override if pitch_override else (meta_length, meta_width)

    if expected_frames is None:
        # Determine the size without storing data; used only for non-standard exports.
        with open(raw_txt, "r", encoding="utf-8") as handle:
            expected_frames = sum(1 for _ in handle)

    raw_ids = np.empty(expected_frames, dtype=np.int32)
    player_xy = np.full((expected_frames, n_slots, 2), np.nan, dtype=np.float32)
    player_speed = np.full((expected_frames, n_slots), np.nan, dtype=np.float32)
    ball_xy = np.full((expected_frames, 2), np.nan, dtype=np.float32)
    ball_speed = np.full(expected_frames, np.nan, dtype=np.float32)

    count = 0
    with open(raw_txt, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split(":")
            if len(parts) != 3:
                raise ValueError(f"ATD line {line_no}: expected 3 colon-separated fields, found {len(parts)}")
            frame_id = int(parts[0])
            if count >= expected_frames:
                raise ValueError(f"ATD has more than expected {expected_frames} records")
            expected_id = count + 1
            if frame_id != expected_id:
                raise ValueError(f"ATD frame counter discontinuity at line {line_no}: expected {expected_id}, got {frame_id}")

            groups = parts[1].split(";")
            if len(groups) != n_slots:
                raise ValueError(f"ATD line {line_no}: metadata has {n_slots} slots but raw row has {len(groups)}")
            flat_players = np.fromstring(parts[1].replace(";", ","), sep=",", dtype=np.float32)
            ball = np.fromstring(parts[2], sep=",", dtype=np.float32)
            if flat_players.size != n_slots * 3 or ball.size != 3:
                raise ValueError(f"ATD line {line_no}: malformed player/ball coordinate group")
            player_groups = flat_players.reshape(n_slots, 3)
            player_xy[count, :, 0] = player_groups[:, 0] * length
            player_xy[count, :, 1] = player_groups[:, 1] * width
            player_speed[count, :] = player_groups[:, 2]
            ball_xy[count, 0] = ball[0] * length
            ball_xy[count, 1] = ball[1] * width
            ball_speed[count] = ball[2]
            raw_ids[count] = frame_id
            count += 1

    if count != expected_frames:
        raise ValueError(f"ATD record count mismatch: expected {expected_frames}, read {count}")

    frame = raw_ids.astype(np.int64) - 1
    if not np.array_equal(frame, np.arange(expected_frames)):
        raise ValueError("ATD-to-video frame mapping is not contiguous 0..N-1")

    return TrackingData(
        frame=frame,
        raw_frame_id=raw_ids,
        player_xy_m=player_xy,
        player_speed_mps=player_speed,
        ball_xy_m=ball_xy,
        ball_speed_mps=ball_speed,
        team_of_slot=team_of_slot,
        team_names=team_names,
        pitch_length_m=float(length),
        pitch_width_m=float(width),
    )
