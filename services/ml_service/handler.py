from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from src.config import PRODUCT_COLS
from src.features import encode_features, prepare_features


class RecommenderHandler:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, "rb") as file:
            payload = pickle.load(file)

        self.models = payload["models"]
        self.encoders = payload["encoders"]
        self.feature_cols = payload["feature_cols"]
        self.product_cols = payload["product_cols"]
        self.product_labels = payload["product_labels"]
        self.top_k = payload["top_k"]

    def _build_frame(self, data: dict) -> pd.DataFrame:
        row = {col: data.get(col) for col in self.feature_cols if col in data}
        for product in self.product_cols:
            row[product] = int(data.get("products", {}).get(product, 0))

        frame = pd.DataFrame([row])
        cleaned = prepare_features(frame)
        encoded, _ = encode_features(cleaned, self.encoders)
        return encoded

    def recommend(self, data: dict) -> list[dict]:
        features = self._build_frame(data)
        scores = np.array([self.models[product].predict(features)[0] for product in self.product_cols])

        for idx, product in enumerate(self.product_cols):
            if int(data.get("products", {}).get(product, 0)) == 1:
                scores[idx] = -1.0

        top_idx = np.argsort(-scores)[: self.top_k]
        return [
            {
                "product_code": self.product_cols[i],
                "product_name": self.product_labels[self.product_cols[i]],
                "score": float(scores[i]),
            }
            for i in top_idx
        ]

    def feature_snapshot(self, data: dict) -> dict[str, float]:
        frame = self._build_frame(data)
        return {col: float(frame.iloc[0][col]) for col in ["age", "renta", "n_products"]}
