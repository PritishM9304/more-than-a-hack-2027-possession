#!/usr/bin/env python3
"""Audit the supplied archive layout and verify timeline/data conventions."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import csv
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession.atd import read_team_mapping


def count_atd(path: Path, n_slots: int):
    n = first = last = 0
    ball_missing = 0
    player_missing = [0] * n_slots
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split(":")
            if len(parts) != 3:
                raise ValueError(f"{path}:{line_no}: expected 3 colon-separated fields")
            fid = int(parts[0])
            n += 1
            if first == 0:
                first = fid
            last = fid
            groups = parts[1].split(";")
            if len(groups) != n_slots:
                raise ValueError(f"{path}:{line_no}: expected {n_slots} player groups, got {len(groups)}")
            for i, group in enumerate(groups):
                xy = group.split(",")[:2]
                if len(xy) < 2 or xy[0].lower() == "nan" or xy[1].lower() == "nan":
                    player_missing[i] += 1
            ball_xy = parts[2].split(",")[:2]
            if len(ball_xy) < 2 or ball_xy[0].lower() == "nan" or ball_xy[1].lower() == "nan":
                ball_missing += 1
    return {"records": n, "first_id": first, "last_id": last,
            "ball_missing": ball_missing, "ball_visible": n-ball_missing,
            "ball_visible_rate": (n-ball_missing)/n if n else 0,
            "player_missing": player_missing}


def audit_homography(path: Path, expected: int):
    n = valid = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["frame"]) != n:
                raise ValueError(f"Homography index discontinuity at row {n}")
            n += 1
            valid += int(row["valid"])
    if n != expected:
        raise ValueError(f"Homography row count {n} != expected {expected}")
    return {"rows": n, "valid": valid, "valid_rate": valid/n if n else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()
    root = Path(args.data_dir)
    for source, expected_frames in (("tactical", 199319), ("broadcast", 164488)):
        folder = root / source
        prefix = f"BAR-ATH_{source}"
        meta, raw = folder / f"{prefix}_atd.xml", folder / f"{prefix}_atd.txt"
        team_slots, team_names, plen, pwidth = read_team_mapping(meta)
        summary = count_atd(raw, len(team_slots))
        print(f"\n[{source}]")
        print(f"teams={team_names}; slots={len(team_slots)}; slot counts H={(team_slots==0).sum()} A={(team_slots==1).sum()}")
        print(f"metadata pitch={plen}x{pwidth}m; row pitch used by the starter=105.3x68.0m")
        print(f"ATD rows={summary['records']} expected={expected_frames}; id={summary['first_id']}..{summary['last_id']}; frame=index-1")
        print(f"ball visible={summary['ball_visible']} ({summary['ball_visible_rate']:.1%}), missing={summary['ball_missing']}")
        h = audit_homography(folder / f"{prefix}_homography.csv", expected_frames)
        print(f"homography={h}")
    tags = root / "tactical/BAR-ATH_tactical_smart_tagging.xml"
    tag_count = sum(1 for _ in ET.parse(tags).getroot().iter("instance"))
    print(f"\nSmart Tagging instances={tag_count}; time base=seconds from tactical video start; frame=round(seconds*25)")
    print("No MP4 files present in supplied archive set; video branch is not available here.")


if __name__ == "__main__":
    main()
