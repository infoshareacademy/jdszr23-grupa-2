from __future__ import annotations

import pandas as pd


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    features = data.copy()
    features["price_avg"] = features[["price_min", "price_max"]].mean(axis=1)
    features["price_range"] = features.apply(
        lambda row: f"{row['price_min']}-{row['price_max']}", axis=1
    )
    features["sales_velocity"] = features["sales"] / (features["days_listed"] + 1)
    return features
