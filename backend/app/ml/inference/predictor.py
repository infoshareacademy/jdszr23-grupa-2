from __future__ import annotations

import pandas as pd


def predict_trend(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Run inference on prepared features and return scoring output."""
    predictions = feature_df.copy()
    predictions["trend_score"] = 0.0
    predictions["forecast_growth_90d"] = 0.0
    predictions["suggested_selling_price"] = predictions["price_avg"] * 1.15
    predictions["estimated_margin"] = predictions["suggested_selling_price"] - predictions["price_avg"]
    return predictions
