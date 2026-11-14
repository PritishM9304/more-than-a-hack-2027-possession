"""Classifier interface and deliberately simple tracking-only baseline."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

LABELS = ("H", "A", "D")


class PossessionModel:
    """Interface for a trained classifier. Implement predict_proba(features) -> (N,3)."""
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


@dataclass
class NearestPlayerBaseline(PossessionModel):
    """Closest observed player wins; if the ball is missing or no player is tracked, abstain.

    This is an intentionally weak baseline. It does NOT claim that proximity means control,
    and does NOT turn missing ball observations into dead-ball labels.
    """
    max_nearest_distance_m: float = 12.0

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        n = len(features)
        probs = np.zeros((n, 3), dtype=np.float32)
        dh = features["home_nearest_dist_m"].to_numpy(dtype=np.float32)
        da = features["away_nearest_dist_m"].to_numpy(dtype=np.float32)
        vis = features["ball_visible"].to_numpy(dtype=bool)
        valid = vis & (np.isfinite(dh) | np.isfinite(da))
        home_wins = np.isfinite(dh) & (~np.isfinite(da) | (dh <= da))
        away_wins = np.isfinite(da) & (~np.isfinite(dh) | (da < dh))
        close = np.minimum(dh, da) <= self.max_nearest_distance_m
        h = valid & home_wins & close
        a = valid & away_wins & close
        probs[h, 0] = 1.0
        probs[a, 1] = 1.0
        # D is intentionally not inferred by the nearest-player baseline.
        # Unresolved rows remain all-zero and will become X in the decoder.
        return probs


def load_model(path: str | None) -> PossessionModel:
    """Load an optional joblib model implementing predict_proba(DataFrame)."""
    if path is None:
        return NearestPlayerBaseline()
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("Install joblib to load a model: pip install joblib") from exc
    model = joblib.load(path)
    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError("Loaded model must implement predict_proba(features)")
    return model
