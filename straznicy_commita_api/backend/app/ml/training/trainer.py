from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


def train_score_model(feature_df: pd.DataFrame) -> tuple[RandomForestRegressor, XGBRegressor]:
    # placeholder: training logic and model tracking with MLflow
    random_forest = RandomForestRegressor(n_estimators=100)
    xgb = XGBRegressor(n_estimators=100)
    return random_forest, xgb
