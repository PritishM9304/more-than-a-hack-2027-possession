# More Than A Hack 2027 — Phase 1 Possession

Tracking-only starter implementation for the Metrica Sports Phase 1 football-possession challenge.

## Contents

- `source_code_for_github.zip`: complete Python starter project, including the ATD parser, Smart Tagging reader, tracking feature builder, baseline classifier, temporal decoder, submission writer, data-audit script, and unit tests.
- The archive contains code and documentation only. Match tracking data, videos, and generated frame-level CSV exports are intentionally excluded.

## Quick start

1. Download and extract `source_code_for_github.zip`.
2. Enter the extracted `mtah_possession_starter` directory.
3. Create a Python 3.10+ virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

4. Place the supplied dataset in the documented `tactical/` and `broadcast/` directory structure, then follow the detailed README in the archive to audit data and generate a CSV.

## Important limitations

This is a reproducible tracking-only starter, not a trained or competitive final model. The current nearest-player prototype does not infer dead-ball labels from missing ball observations; a trained possession/dead-ball classifier and improved temporal decoding remain future work. No official competition score is claimed.

The public five-minute ground-truth annotations must not be used for model training, model selection, or parameter tuning. The starter does not load them.

## Challenge

Official challenge repository: https://github.com/metrica-sports/more-than-a-hack-2027
