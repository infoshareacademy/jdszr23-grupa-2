from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.ml.inference.predictor import predict_product as run_prediction
from backend.app.ml.model import FIT_OPTIONS, SLEEVE_OPTIONS, get_brand_options, normalize_brand

router = APIRouter()


class ProductPredictionRequest(BaseModel):
    fit: str = Field(..., title="Fit produktu", example="koszulka slim fit")
    sleeve: str = Field(..., title="Typ rękawa", example="Długi rękaw")
    marka: str = Field(..., title="Marka produktu", example="nike")


class ProductPredictionResponse(BaseModel):
    segment: str
    predicted_price_range: str
    confidence: float


@router.get("/options", response_model=dict[str, list[str]])
async def get_product_options() -> dict[str, list[str]]:
    brands = get_brand_options()
    return {"fits": FIT_OPTIONS, "sleeves": SLEEVE_OPTIONS, "brands": brands}


@router.post("/predict", response_model=ProductPredictionResponse)
async def predict_product(payload: ProductPredictionRequest) -> ProductPredictionResponse:
    if payload.fit not in FIT_OPTIONS:
        raise HTTPException(status_code=422, detail="Nieprawidłowa wartość fit produktu.")
    if payload.sleeve not in SLEEVE_OPTIONS:
        raise HTTPException(status_code=422, detail="Nieprawidłowa wartość typu rękawa.")
    try:
        # validate/normalize brand
        normalized_brand = normalize_brand(payload.marka)
    except ValueError:
        raise HTTPException(status_code=422, detail="Nieprawidłowa wartość marki produktu.")
    try:
        result = run_prediction(payload.fit, payload.sleeve, normalized_brand)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return ProductPredictionResponse(
        segment=result["segment"],
        predicted_price_range=result["predicted_price_range"],
        confidence=result["confidence"],
    )
