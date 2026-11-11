"""Phase 4 meta-learner at inference time: LightGBM + isotonic calibrator, score bands,
and the equal-weight baseline that doubles as the fallback before training."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from src.fusion.features import CHANNEL_SCORE, FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
# NEWSCRED_META_DIR: load / save the meta-learner from another folder (a candidate model is scored on the
# India sets and the benchmark next to the deployed one before it replaces it). Default: models/.
META_DIR = Path(os.environ.get("NEWSCRED_META_DIR") or ROOT / "models")
META_MODEL = META_DIR / "meta_lgbm.txt"
META_CALIBRATOR = META_DIR / "meta_calibrator.joblib"
META_INFO = META_DIR / "meta_info.json"

# Score bands. Upper bounds are inclusive: 30 is Low, 31 is Questionable.
BANDS = [(30, "Low"), (60, "Questionable"), (80, "Mostly credible"), (100, "High")]


def band(score: int | float | None) -> str:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "Unknown"
    for upper, name in BANDS:
        if score <= upper:
            return name
    return BANDS[-1][1]


def to_score(p: float) -> int | None:
    """Calibrated probability -> 0-100 integer score."""
    return None if p is None or math.isnan(p) else int(round(100 * float(np.clip(p, 0.0, 1.0))))


def average_channel_scores(rows: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Equal-weight average of the channel headline scores that are present (NaN
    ignored). Baseline (b) of the evaluation, and the fallback scorer until
    the LightGBM model is trained. NaN when no channel produced a score."""
    X = np.atleast_2d(np.asarray(rows, dtype=float))
    idx = [FEATURE_COLUMNS.index(c) for c in CHANNEL_SCORE.values()]
    sub = X[:, idx]
    out = np.full(len(X), np.nan)
    has = ~np.all(np.isnan(sub), axis=1)
    out[has] = np.nanmean(sub[has], axis=1)
    return out


class MetaModel:
    """LightGBM booster (cross-entropy objective, outputs probabilities) + isotonic calibrator."""

    def __init__(self, booster, calibrator, feature_names: Sequence[str] = FEATURE_COLUMNS, info: dict | None = None):
        self.booster = booster
        self.calibrator = calibrator
        self.feature_names = list(feature_names)
        self.info = info or {}
        if self.feature_names != FEATURE_COLUMNS:
            raise ValueError(
                "saved meta-model was trained on a different feature list; retrain with python -m src.fusion.train fit"
            )

    def predict_raw(self, X) -> np.ndarray:
        """Uncalibrated probability. For the cross_entropy objective LightGBM applies
        the sigmoid itself, so outputs are already in [0, 1]."""
        return np.asarray(self.booster.predict(np.atleast_2d(np.asarray(X, dtype=float))), dtype=float)

    def predict(self, X) -> np.ndarray:
        raw = self.predict_raw(X)
        if self.calibrator is None:
            return raw
        return np.clip(self.calibrator.predict(raw), 0.0, 1.0)

    def save(self, model_path: Path = META_MODEL, calibrator_path: Path = META_CALIBRATOR, info_path: Path = META_INFO):
        import joblib

        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(model_path))
        joblib.dump(self.calibrator, calibrator_path)
        info_path.write_text(json.dumps({**self.info, "feature_names": self.feature_names}, indent=2))

    @classmethod
    def load(cls, model_path: Path = META_MODEL, calibrator_path: Path = META_CALIBRATOR, info_path: Path = META_INFO):
        import joblib
        import lightgbm as lgb

        booster = lgb.Booster(model_file=str(model_path))
        calibrator = joblib.load(calibrator_path) if Path(calibrator_path).exists() else None
        info = json.loads(Path(info_path).read_text()) if Path(info_path).exists() else {}
        return cls(booster, calibrator, booster.feature_name(), info)
