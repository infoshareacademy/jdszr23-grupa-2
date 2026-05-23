from __future__ import annotations

from pathlib import Path
import pandas as pd

from backend.app.ml.model import (
    PRICE_RANGE_BY_SEGMENT,
    FIT_OPTIONS,
    SLEEVE_OPTIONS,
    normalize_fit,
    normalize_sleeve,
    ensure_model_exists,
)

_model = None


def _load_model():
    global _model
    if _model is None:
        _model = ensure_model_exists()
    return _model


from backend.app.ml.model import normalize_brand


def predict_product(fit: str, sleeve: str, marka: str) -> dict[str, object]:
    canonical_fit = normalize_fit(fit)
    canonical_sleeve = normalize_sleeve(sleeve)
    canonical_brand = normalize_brand(marka)
    model = _load_model()

    sample = pd.DataFrame(
        [{"fit": canonical_fit, "sleeve": canonical_sleeve, "brand": canonical_brand}]
    )
    segment = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]
    confidence = float(round(float(probabilities.max()), 2))

    return {
        "segment": segment,
        "predicted_price_range": PRICE_RANGE_BY_SEGMENT.get(segment, "100–300 zł"),
        "confidence": confidence,
        "model_source": Path(__file__).resolve().stem,
    }
