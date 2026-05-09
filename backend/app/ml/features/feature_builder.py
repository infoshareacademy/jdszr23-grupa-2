from __future__ import annotations

import pandas as pd


def build_product_features(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Build unified feature set for product trend scoring and forecasting."""
    features = raw_data.copy()
    features["price_range"] = features[["price_min", "price_max"]].apply(lambda row: f"{row[0]}-{row[1]}", axis=1)
    features["price_avg"] = features[["price_min", "price_max"]].mean(axis=1)
    features["popularity_ratio"] = features["sales"] / (features["views"] + 1)
    return features
