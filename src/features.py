from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, PRODUCT_COLS


def clean_age(series: pd.Series) -> pd.Series:
    age = pd.to_numeric(series, errors="coerce")
    age = age.where((age >= 10) & (age <= 100))
    return age.fillna(age.median())


def clean_antiguedad(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.fillna(0.0)


def clean_renta(series: pd.Series) -> pd.Series:
    renta = pd.to_numeric(series, errors="coerce")
    return renta.fillna(renta.median())


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age"] = clean_age(out["age"])
    out["antiguedad"] = clean_antiguedad(out["antiguedad"])
    out["renta"] = clean_renta(out["renta"])
    out["n_products"] = out[PRODUCT_COLS].sum(axis=1)

    for col in BINARY_FEATURES + PRODUCT_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].fillna("missing").astype(str)

    return out


def encode_features(df: pd.DataFrame, encoders: dict | None = None) -> tuple[pd.DataFrame, dict]:
    encoded = df.copy()
    encoders = encoders or {}

    for col in CATEGORICAL_FEATURES:
        if col not in encoders:
            categories = sorted(encoded[col].astype(str).unique().tolist())
            encoders[col] = {value: idx for idx, value in enumerate(categories)}
        encoded[col] = encoded[col].astype(str).map(encoders[col]).fillna(-1).astype(int)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES + PRODUCT_COLS
    return encoded[feature_cols], encoders
