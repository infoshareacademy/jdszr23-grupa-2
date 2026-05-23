from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_FILE = MODEL_DIR / "product_segment_model.pkl"
DATA_PATH = BASE_DIR / "data" / "dane_do_modelu.csv"

FIT_OPTIONS = [
    "koszulka regular fit",
    "koszulka loose fit",
    "koszulka slim fit",
    "koszulka oversize",
    "koszulka skinny fit",
    "koszulka other",
    "koszulka fit",
]

SLEEVE_OPTIONS = [
    "Krótki rękaw",
    "Półrękawek",
    "Długi rękaw",
    "Bez rękawów",
]

PRICE_RANGE_BY_SEGMENT = {
    "niski": "100–250 zł",
    "średni": "150–300 zł",
    "premium": "250–400 zł",
}

_MODEL_COLUMNS = ["fit", "sleeve", "brand"]


def _clean_brand_raw(value: object) -> str | None:
    import re
    import html

    if value is None:
        return None
    try:
        text = html.unescape(str(value)).lower().strip()
    except Exception:
        text = str(value).lower().strip()
    # remove pack indicators like '3 pack', '2pk', etc.
    text = re.sub(r"\b\d+\s*-?\s*(pack|pk|er-?pack)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text == "" or text == "nan":
        return None
    return text


def normalize_brand(value: str) -> str:
    cleaned = _clean_brand_raw(value)
    if not cleaned:
        raise ValueError(f"Nieznana marka: {value}")
    return cleaned


def normalize_fit(value: str) -> str:
    text = value.strip().lower()
    for option in FIT_OPTIONS:
        if option == text:
            return option
    raise ValueError(f"Nieznany fit: {value}")


def normalize_sleeve(value: str) -> str:
    text = value.strip().lower()
    if "bez" in text and "rękaw" in text:
        return "Bez rękawów"
    if "pół" in text or "p�r" in text or "pó" in text:
        return "Półrękawek"
    if "dług" in text:
        return "Długi rękaw"
    if "krót" in text:
        return "Krótki rękaw"
    if "rękaw" in text and "bez" not in text:
        return "Długi rękaw"
    raise ValueError(f"Nieznany typ rękawa: {value}")


def price_to_segment(price: float) -> str:
    if price <= 150:
        return "niski"
    if price <= 300:
        return "średni"
    return "premium"


def load_training_data(path: Path = DATA_PATH) -> pd.DataFrame:
    from backend.app.ml.data_loader import load_dataset

    raw = load_dataset(path)
    raw = raw.copy()
    raw["fit"] = raw["fason_clean"].astype(str).str.strip().str.lower()
    raw["sleeve"] = raw["dlugosc_rekawa"].astype(str).apply(normalize_sleeve).str.strip()
    raw["segment"] = raw["cena_aktualna"].astype(float).apply(price_to_segment)
    # brand / marka cleaning
    raw["brand"] = raw.get("marka")
    raw["brand"] = raw["brand"].apply(_clean_brand_raw)

    # filter rows with known categories
    raw = raw[raw["fit"].isin(FIT_OPTIONS) & raw["sleeve"].isin(SLEEVE_OPTIONS) & raw["brand"].notna()]
    return raw[["fit", "sleeve", "brand", "segment"]]


def get_brand_options(path: Path = DATA_PATH) -> list[str]:
    from backend.app.ml.data_loader import load_dataset

    raw = load_dataset(path)
    if "marka" not in raw.columns:
        return []
    brands = raw["marka"].apply(_clean_brand_raw).dropna().unique().tolist()
    brands = sorted([b for b in brands if b])
    return brands


def build_model() -> Pipeline:
    encoder = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                _MODEL_COLUMNS,
            )
        ],
        remainder="drop",
    )
    model = Pipeline(
        steps=[
            ("encoder", encoder),
            (
                "classifier",
                LogisticRegression(max_iter=500, class_weight="balanced", random_state=42),
            ),
        ]
    )

    training_data = load_training_data()
    if training_data.empty:
        raise RuntimeError("Brak danych treningowych do zbudowania modelu.")

    X = training_data[[_MODEL_COLUMNS[0], _MODEL_COLUMNS[1], _MODEL_COLUMNS[2]]]
    y = training_data["segment"]
    model.fit(X, y)
    return model


def ensure_model_exists() -> Pipeline:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_FILE.exists():
        with MODEL_FILE.open("rb") as handle:
            return pickle.load(handle)
    model = build_model()
    with MODEL_FILE.open("wb") as handle:
        pickle.dump(model, handle)
    return model
