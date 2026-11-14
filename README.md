# More Than A Hack 2027 — Phase 1 Possession

Tracking-only starter implementation for the Metrica Sports Phase 1 football-possession challenge.

## Repository contents

- `possession/atd.py`: tactical ATD parser and team-slot mapping from XML metadata.
- `possession/features.py`: frame-aligned tracking features, ball/player motion, team-wise proximity/support and optional homography/Smart Tagging inputs.
- `possession/smart_tagging.py`: Smart Tagging XML intervals converted to frame-aligned evidence features.
- `possession/model.py`: nearest-player baseline and interface for a learned `predict_proba(features)` model.
- `possession/decoder.py`: stateful H/A/D decoding with switch confirmation and X abstention.
- `possession/submission.py`: CSV writer and structural validator.
- `scripts/audit_data.py`: data layout, ATD and homography audit.
- `scripts/build_submission.py`: full tracking-only pipeline and CSV output.
- `tests/test_core.py`: core unit tests.

## Install

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

## Expected data layout

Keep the supplied match data outside this public repository. Extract the organiser-provided archives into:

```text
DATA_ROOT/
  tactical/
    BAR-ATH_tactical_atd.xml
    BAR-ATH_tactical_atd.txt
    BAR-ATH_tactical_homography.csv
    BAR-ATH_tactical_smart_tagging.xml
  broadcast/
    BAR-ATH_broadcast_atd.xml
    BAR-ATH_broadcast_atd.txt
    BAR-ATH_broadcast_homography.csv
```

Audit the files:

```bash
python scripts/audit_data.py --data-dir "/path/to/DATA_ROOT"
```

Generate the tracking-only prototype:

```bash
python scripts/build_submission.py --data-dir "/path/to/DATA_ROOT" --out submission.csv
```

The script accepts an optional joblib model implementing `predict_proba(features)` with probability columns ordered `H,A,D`:

```bash
python -m pip install joblib
python scripts/build_submission.py --data-dir DATA_ROOT --model model.joblib --out submission.csv
```

## Data alignment and challenge constraints

The supplied tactical ATD has 199,319 records with raw frame IDs 1–199,319. This parser maps them to zero-based submission frames 0–199,318 using `video_frame = frame_id - 1`. Tactical slot ownership is read from the XML; slots are tracklets, not permanent player identities. The starter preserves NaN observations and does not treat a missing ball position as a dead ball.

Smart Tagging is used only as supporting contextual features; its tactical intervals are not copied into possession predictions. Broadcast tracking runs on a separate timeline and must not be joined to the tactical stream by frame number.

**Competition rules:** do not use the public five-minute ground-truth annotations for training, model selection or parameter tuning. This project does not load those labels. Keep match data, videos and frame-level derived exports out of this public repository.

## Status and limitations

This is a reproducible tracking-only starter, not a trained or competition-ready final model. The nearest-player baseline is intentionally weak and does not infer dead-ball labels; low-evidence frames may become `X`. A trained controlled-possession/dead-ball classifier and stronger temporal decoding are future work. No official competition score is claimed.

Official challenge repository: https://github.com/metrica-sports/more-than-a-hack-2027
