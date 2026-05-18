from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def train_score_model(feature_df: pd.DataFrame) -> RandomForestRegressor:
    # placeholder: training logic and model tracking with MLflow
    random_forest = RandomForestRegressor(n_estimators=100)
    return random_forest
