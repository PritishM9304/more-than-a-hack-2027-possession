import numpy as np
import pandas as pd

from possession.decoder import decode_probabilities
from possession.model import NearestPlayerBaseline
from possession.submission import write_submission, validate_submission
from possession.smart_tagging import load_smart_tagging


def test_decoder_confirmed_switch_and_unknown():
    p = np.array([
        [0.9, 0.05, 0.05],
        [0.2, 0.75, 0.05],
        [0.2, 0.75, 0.05],
        [0.2, 0.75, 0.05],
        [0.1, 0.1, 0.8],
    ], dtype=np.float32)
    out = decode_probabilities(p, confirm_frames=3)
    assert list(out) == ["H", "H", "H", "A", "D"]


def test_nearest_model_missing_ball_is_not_dead_ball():
    frame = pd.DataFrame({
        "ball_visible": [1, 0, 1],
        "home_nearest_dist_m": [1.0, np.nan, np.nan],
        "away_nearest_dist_m": [2.0, np.nan, 3.0],
    })
    p = NearestPlayerBaseline().predict_proba(frame)
    assert np.argmax(p[0]) == 0
    assert p[1].sum() == 0.0  # unresolved; decoder can abstain as X
    assert np.argmax(p[2]) == 1


def test_smart_tagging_frame_alignment(tmp_path):
    xml = tmp_path / "tag.xml"
    xml.write_text('''<file><ALL_INSTANCES><instance><start>1.00</start><end>1.20</end><code>SET PIECES</code><label><group>Team</group><text>Barcelona</text></label><label><group>Type</group><text>Corner Kick</text></label></instance></ALL_INSTANCES></file>''')
    out = load_smart_tagging(xml, n_frames=100)
    assert out["tag_active"][25:30].all()
    assert not out["tag_active"][30]
    assert out["tag_team_H"][25:30].all()
    assert out["tag_corner"][25:30].all()


def test_submission_schema(tmp_path):
    path = tmp_path / "out.csv"
    labels = np.full(199319, "X", dtype="<U1")
    write_submission(path, labels)
    report = validate_submission(path)
    assert report["rows"] == 199319
    assert report["first_frame"] == 0
    assert report["last_frame"] == 199318
