from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.ml.inference.predictor import predict_product as run_prediction
from backend.app.ml.model import FIT_OPTIONS, SLEEVE_OPTIONS

router = APIRouter()


class ProductPredictionRequest(BaseModel):
    fit: str = Field(..., title="Fit produktu", example="koszulka slim fit")
    sleeve: str = Field(..., title="Typ rękawa", example="Długi rękaw")


class ProductPredictionResponse(BaseModel):
    segment: str
    predicted_price_range: str
    confidence: float


@router.get("/options", response_model=dict[str, list[str]])
async def get_product_options() -> dict[str, list[str]]:
    return {"fits": FIT_OPTIONS, "sleeves": SLEEVE_OPTIONS}


@router.post("/predict", response_model=ProductPredictionResponse)
async def predict_product(payload: ProductPredictionRequest) -> ProductPredictionResponse:
    if payload.fit not in FIT_OPTIONS:
        raise HTTPException(status_code=422, detail="Nieprawidłowa wartość fit produktu.")
    if payload.sleeve not in SLEEVE_OPTIONS:
        raise HTTPException(status_code=422, detail="Nieprawidłowa wartość typu rękawa.")
    try:
        result = run_prediction(payload.fit, payload.sleeve)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return ProductPredictionResponse(
        segment=result["segment"],
        predicted_price_range=result["predicted_price_range"],
        confidence=result["confidence"],
    )
