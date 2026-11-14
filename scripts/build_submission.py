#!/usr/bin/env python3
"""Run the tracking-only nearest-player prototype and write a valid-format CSV."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from possession.atd import read_atd
from possession.features import build_features
from possession.model import load_model
from possession.decoder import decode_probabilities
from possession.submission import write_submission, validate_submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Root containing tactical/ and broadcast/ folders")
    ap.add_argument("--out", default="submission_tracking_baseline.csv")
    ap.add_argument("--model", default=None, help="Optional joblib classifier with predict_proba(DataFrame)")
    ap.add_argument("--confidence", type=float, default=0.50)
    ap.add_argument("--switch-margin", type=float, default=0.15)
    ap.add_argument("--confirm-frames", type=int, default=3)
    args = ap.parse_args()

    data_root = Path(args.data_dir)
    tactical = data_root / "tactical"
    meta = tactical / "BAR-ATH_tactical_atd.xml"
    raw = tactical / "BAR-ATH_tactical_atd.txt"
    tags = tactical / "BAR-ATH_tactical_smart_tagging.xml"
    homography = tactical / "BAR-ATH_tactical_homography.csv"
    for path in (meta, raw, tags, homography):
        if not path.exists():
            raise FileNotFoundError(path)

    tracking = read_atd(meta, raw, pitch_override=(105.3, 68.0), expected_frames=199_319)
    features = build_features(tracking, smart_tagging_xml=tags, homography_csv=homography)
    model = load_model(args.model)
    probabilities = model.predict_proba(features)
    labels = decode_probabilities(probabilities, confidence_threshold=args.confidence,
                                  switch_margin=args.switch_margin, confirm_frames=args.confirm_frames)
    out = write_submission(args.out, labels)
    report = validate_submission(out)
    print(f"teams: H={tracking.team_names[0]!r}, A={tracking.team_names[1]!r}")
    print(f"frames: {len(features)}; slots: {tracking.player_xy_m.shape[1]}; feature columns: {features.shape[1]}")
    print(f"ball visible: {int(features['ball_visible'].sum())}/{len(features)} ({features['ball_visible'].mean():.1%})")
    print(f"output: {out.resolve()}")
    print(f"validation: {report}")
    print("NOTE: This baseline does not infer D from missing ball observations; missing/low-evidence frames may be X.")


if __name__ == "__main__":
    main()
