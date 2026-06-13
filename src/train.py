from __future__ import annotations

import gc
import json
import pickle

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from src.config import (
    ARTIFACTS_DIR,
    FEATURE_COLS,
    LGBM_PARAMS,
    MLFLOW_EXPERIMENT,
    MODELS_DIR,
    PRODUCT_COLS,
    RANDOM_STATE,
    ROOT,
    TOP_K,
    TRAIN_PAIRS,
    TRAIN_SAMPLE_SIZE,
    VAL_PAIR,
    VAL_SAMPLE_SIZE,
)
from src.data import build_dataset, prepare_matrix, product_labels, save_encoders
from src.metrics import coverage_at_k, map_at_k, precision_at_k, recall_at_k


def popularity_baseline(y_train: np.ndarray, n_rows: int) -> np.ndarray:
    popularity = y_train.mean(axis=0)
    return np.tile(popularity, (n_rows, 1))


def subsample(
    X: pd.DataFrame,
    y: pd.DataFrame,
    size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(X) <= size:
        return X, y
    idx = np.random.default_rng(seed).choice(len(X), size=size, replace=False)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def train_models(X_train: pd.DataFrame, y_train: pd.DataFrame) -> dict[str, lgb.Booster]:
    models = {}
    for idx, product in enumerate(PRODUCT_COLS, start=1):
        print(f"  training {idx}/{len(PRODUCT_COLS)}: {product}", flush=True)
        train_set = lgb.Dataset(X_train, label=y_train[product])
        models[product] = lgb.train(LGBM_PARAMS, train_set, num_boost_round=100)
    return models


def predict_scores(models: dict[str, lgb.Booster], X: pd.DataFrame) -> np.ndarray:
    scores = np.column_stack([models[product].predict(X) for product in PRODUCT_COLS])
    for idx, product in enumerate(PRODUCT_COLS):
        scores[X[product].values == 1, idx] = -1.0
    return scores


def evaluate(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        f"map@{TOP_K}": map_at_k(y_true, scores, TOP_K),
        f"precision@{TOP_K}": precision_at_k(y_true, scores, TOP_K),
        f"recall@{TOP_K}": recall_at_k(y_true, scores, TOP_K),
        f"coverage@{TOP_K}": coverage_at_k(scores, TOP_K),
    }


def mlflow_metric_name(name: str) -> str:
    return name.replace("@", "_at_")


def log_metrics(prefix: str, metrics: dict[str, float]) -> None:
    for name, value in metrics.items():
        mlflow.log_metric(f"{prefix}{mlflow_metric_name(name)}", value)


def save_artifacts(models: dict, encoders: dict, metrics: dict[str, float]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODELS_DIR / "recommender.pkl", "wb") as file:
        pickle.dump(
            {
                "models": models,
                "encoders": encoders,
                "feature_cols": FEATURE_COLS,
                "product_cols": PRODUCT_COLS,
                "product_labels": product_labels(),
                "top_k": TOP_K,
            },
            file,
        )

    save_encoders(encoders, ARTIFACTS_DIR / "encoders.json")
    (ARTIFACTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_training() -> dict[str, float]:
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print("Building train dataset...", flush=True)
    X_train_raw, y_train = build_dataset(TRAIN_PAIRS)
    X_train_raw, y_train = subsample(X_train_raw, y_train, TRAIN_SAMPLE_SIZE, RANDOM_STATE)
    print(f"Train rows: {len(X_train_raw)}", flush=True)

    print("Building validation dataset...", flush=True)
    X_val_raw, y_val = build_dataset([VAL_PAIR])
    X_val_raw, y_val = subsample(X_val_raw, y_val, VAL_SAMPLE_SIZE, RANDOM_STATE + 1)
    print(f"Validation rows: {len(X_val_raw)}", flush=True)

    X_train, encoders = prepare_matrix(X_train_raw)
    del X_train_raw
    gc.collect()

    X_val, _ = prepare_matrix(X_val_raw, encoders)
    del X_val_raw
    gc.collect()

    y_train_np = y_train.values
    y_val_np = y_val.values

    with mlflow.start_run(run_name="lightgbm_24_products"):
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("train_pairs", str(TRAIN_PAIRS))
        mlflow.log_param("train_sample_size", TRAIN_SAMPLE_SIZE)
        mlflow.log_param("val_sample_size", VAL_SAMPLE_SIZE)
        mlflow.log_param("val_pair", str(VAL_PAIR))
        mlflow.log_params({f"lgbm_{k}": v for k, v in LGBM_PARAMS.items() if k != "verbose"})

        baseline_metrics = evaluate(y_val_np, popularity_baseline(y_train_np, len(y_val_np)))
        log_metrics("baseline_", baseline_metrics)
        print("Baseline:", baseline_metrics, flush=True)

        models = train_models(X_train, y_train)
        del y_train
        gc.collect()

        model_metrics = evaluate(y_val_np, predict_scores(models, X_val))
        log_metrics("", model_metrics)
        print("Model:", model_metrics, flush=True)

        save_artifacts(models, encoders, model_metrics)
        mlflow.log_artifacts(str(MODELS_DIR), artifact_path="models")
        mlflow.log_artifacts(str(ARTIFACTS_DIR), artifact_path="artifacts")

    return model_metrics


if __name__ == "__main__":
    print(run_training(), flush=True)
