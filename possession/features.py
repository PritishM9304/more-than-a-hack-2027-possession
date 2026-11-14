"""Frame-aligned tracking feature extraction; no labels or ground truth are read here."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .atd import TrackingData
from .smart_tagging import load_smart_tagging


def _adjacent_velocity(xy: np.ndarray, fps: float) -> np.ndarray:
    """Finite difference only across adjacent valid samples; gaps remain NaN."""
    vel = np.full_like(xy, np.nan, dtype=np.float32)
    if len(xy) < 2:
        return vel
    valid = np.isfinite(xy[1:]).all(axis=-1) & np.isfinite(xy[:-1]).all(axis=-1)
    delta = (xy[1:] - xy[:-1]) * fps
    vel[1:][valid] = delta[valid]
    # First frame can use a forward difference, if adjacent positions are both present.
    first_ok = np.isfinite(xy[0]).all() and np.isfinite(xy[1]).all()
    if first_ok:
        vel[0] = (xy[1] - xy[0]) * fps
    return vel


def _team_features(data: TrackingData, ball_v: np.ndarray, team: int, prefix: str) -> dict[str, np.ndarray]:
    slots = np.flatnonzero(data.team_of_slot == team)
    p = data.player_xy_m[:, slots, :]
    ps = data.player_speed_mps[:, slots]
    b = data.ball_xy_m
    b_ok = np.isfinite(b).all(axis=1)
    p_ok = np.isfinite(p).all(axis=2)
    delta = p - b[:, None, :]
    dist = np.sqrt(np.nansum(delta * delta, axis=2))
    dist[~(p_ok & b_ok[:, None])] = np.nan
    # Keep absent observations NaN, not zero. The count features explicitly handle them.
    sort_dist = np.sort(np.where(np.isfinite(dist), dist, np.inf), axis=1)
    n_valid = p_ok.sum(axis=1).astype(np.float32)
    nearest = sort_dist[:, 0]
    second = sort_dist[:, 1] if len(slots) > 1 else np.full(len(b), np.nan)
    third = sort_dist[:, 2] if len(slots) > 2 else np.full(len(b), np.nan)
    nearest[~b_ok] = np.nan
    second[~b_ok] = np.nan
    third[~b_ok] = np.nan
    # Identify the closest slot per row to calculate its speed and relative radial motion.
    safe_dist = np.where(np.isfinite(dist), dist, np.inf)
    nearest_idx = np.argmin(safe_dist, axis=1)
    rows = np.arange(len(b))
    nearest_speed = ps[rows, nearest_idx].astype(np.float32)
    nearest_speed[~np.isfinite(nearest)] = np.nan
    player_v = _adjacent_velocity(p, 25.0)
    nearest_pv = player_v[rows, nearest_idx, :]
    rel_v = ball_v - nearest_pv
    nearest_delta = -delta[rows, nearest_idx, :]  # ball - player
    denom = np.maximum(nearest, 1e-6)
    radial_rate = np.sum(nearest_delta * rel_v, axis=1) / denom
    radial_rate[~np.isfinite(nearest)] = np.nan
    top3 = sort_dist[:, :min(3, sort_dist.shape[1])]
    top3[top3 == np.inf] = np.nan
    top3_finite = np.isfinite(top3)
    top3_count = top3_finite.sum(axis=1)
    top3_sum = np.where(top3_finite, top3, 0.0).sum(axis=1)
    mean_top3 = np.full(len(b), np.nan, dtype=np.float32)
    has_top3 = top3_count > 0
    mean_top3[has_top3] = (top3_sum[has_top3] / top3_count[has_top3]).astype(np.float32)
    mean_top3[~b_ok] = np.nan
    feats = {
        f"{prefix}_nearest_dist_m": nearest.astype(np.float32),
        f"{prefix}_second_nearest_dist_m": second.astype(np.float32),
        f"{prefix}_third_nearest_dist_m": third.astype(np.float32),
        f"{prefix}_nearest_speed_mps": nearest_speed,
        f"{prefix}_nearest_range_rate_mps": radial_rate.astype(np.float32),
        f"{prefix}_visible_player_count": n_valid,
        f"{prefix}_players_within_2m": np.nansum(dist < 2.0, axis=1).astype(np.float32),
        f"{prefix}_players_within_5m": np.nansum(dist < 5.0, axis=1).astype(np.float32),
        f"{prefix}_players_within_10m": np.nansum(dist < 10.0, axis=1).astype(np.float32),
        f"{prefix}_mean_top3_dist_m": mean_top3,
    }
    # Distances/support metrics are undefined when ball is not observed.
    for key in list(feats):
        if "dist" in key or "within_" in key or "range_rate" in key:
            feats[key][~b_ok] = np.nan
    feats[f"{prefix}_players_within_2m"][~b_ok] = np.nan
    feats[f"{prefix}_players_within_5m"][~b_ok] = np.nan
    feats[f"{prefix}_players_within_10m"][~b_ok] = np.nan
    return feats


def load_homography_features(path: str | Path, n_frames: int) -> dict[str, np.ndarray]:
    """Load only useful reliability metadata; matrices are retained for future video fusion."""
    df = pd.read_csv(path)
    if len(df) != n_frames:
        raise ValueError(f"Homography rows={len(df)} but ATD frames={n_frames}")
    if not np.array_equal(df["frame"].to_numpy(dtype=np.int64), np.arange(n_frames)):
        raise ValueError("Homography frame index is not exactly 0..N-1")
    return {
        "homography_valid": df["valid"].to_numpy(dtype=np.float32),
        "homography_reprojection_error": df["reprojection_error"].to_numpy(dtype=np.float32),
    }


def build_features(data: TrackingData, smart_tagging_xml: str | Path | None = None,
                   homography_csv: str | Path | None = None) -> pd.DataFrame:
    n = len(data.frame)
    ball_ok = np.isfinite(data.ball_xy_m).all(axis=1)
    ball_v = _adjacent_velocity(data.ball_xy_m, 25.0)
    features: dict[str, np.ndarray] = {
        "frame": data.frame.astype(np.int64),
        "ball_visible": ball_ok.astype(np.uint8),
        "ball_x_m": data.ball_xy_m[:, 0].astype(np.float32),
        "ball_y_m": data.ball_xy_m[:, 1].astype(np.float32),
        "ball_speed_mps": data.ball_speed_mps.astype(np.float32),
        "ball_vx_mps": ball_v[:, 0],
        "ball_vy_mps": ball_v[:, 1],
        "ball_outside_pitch": (ball_ok & ((data.ball_xy_m[:, 0] < 0) | (data.ball_xy_m[:, 0] > data.pitch_length_m) |
                                            (data.ball_xy_m[:, 1] < 0) | (data.ball_xy_m[:, 1] > data.pitch_width_m))).astype(np.uint8),
    }
    features.update(_team_features(data, ball_v, 0, "home"))
    features.update(_team_features(data, ball_v, 1, "away"))
    features["nearest_dist_margin_away_minus_home_m"] = (
        features["away_nearest_dist_m"] - features["home_nearest_dist_m"]
    ).astype(np.float32)
    features["player_visible_count"] = np.isfinite(data.player_xy_m).all(axis=2).sum(axis=1).astype(np.float32)
    if smart_tagging_xml:
        features.update(load_smart_tagging(smart_tagging_xml, n_frames=n, fps=25))
    if homography_csv:
        features.update(load_homography_features(homography_csv, n))
    frame_df = pd.DataFrame(features)
    if len(frame_df) != n or not np.array_equal(frame_df["frame"].to_numpy(), np.arange(n)):
        raise AssertionError("Feature table lost tactical frame alignment")
    return frame_df
