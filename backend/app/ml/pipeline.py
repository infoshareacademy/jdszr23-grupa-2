from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.app.ml.data_loader import load_dataset
from backend.app.ml.feature_engineering import build_features


def prepare_dataset(path: Path) -> pd.DataFrame:
    raw = load_dataset(path)
    return build_features(raw)
