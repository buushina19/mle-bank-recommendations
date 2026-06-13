from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import DATA_PATH, PRODUCT_COLS, ROOT
from src.features import encode_features, prepare_features

CACHE_DIR = ROOT / "data" / "cache"


USECOLS = [
    "fecha_dato",
    "ncodpers",
    "ind_empleado",
    "sexo",
    "age",
    "ind_nuevo",
    "antiguedad",
    "indrel_1mes",
    "tiprel_1mes",
    "indresi",
    "indext",
    "conyuemp",
    "canal_entrada",
    "ind_actividad_cliente",
    "renta",
    "segmento",
    *PRODUCT_COLS,
]


STRING_COLS = [
    "ind_empleado",
    "sexo",
    "indrel_1mes",
    "tiprel_1mes",
    "canal_entrada",
    "segmento",
]
INT_COLS = [
    "ncodpers",
    "ind_nuevo",
    "indresi",
    "indext",
    "conyuemp",
    "ind_actividad_cliente",
    *PRODUCT_COLS,
]
FLOAT_COLS = ["age", "antiguedad", "renta"]


def normalize_for_cache(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in STRING_COLS:
        if col in out.columns:
            out[col] = out[col].astype(str).replace({"nan": "missing", "None": "missing"})
    for col in INT_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("int32")
    for col in FLOAT_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def cache_months() -> None:
    """Один проход по CSV → parquet-кэш по месяцам."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if (CACHE_DIR / ".complete").exists():
        return

    print("Caching months to parquet (one pass)...", flush=True)
    for chunk in pd.read_csv(
        DATA_PATH,
        usecols=USECOLS,
        parse_dates=["fecha_dato"],
        chunksize=500_000,
        low_memory=False,
    ):
        for date, group in chunk.groupby("fecha_dato"):
            key = date.strftime("%Y-%m-%d")
            path = CACHE_DIR / f"{key}.parquet"
            if path.exists():
                group = pd.concat([pd.read_parquet(path), group], ignore_index=True)
            group = normalize_for_cache(group)
            group.to_parquet(path, index=False)
    (CACHE_DIR / ".complete").touch()
    print(f"  cached {len(list(CACHE_DIR.glob('*.parquet')))} months", flush=True)


def load_month(date: str) -> pd.DataFrame:
    cache_months()
    cache_path = CACHE_DIR / f"{date}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    target = pd.to_datetime(date)
    chunks = []
    for chunk in pd.read_csv(
        DATA_PATH,
        usecols=USECOLS,
        parse_dates=["fecha_dato"],
        chunksize=500_000,
        low_memory=False,
    ):
        part = chunk[chunk["fecha_dato"] == target]
        if not part.empty:
            chunks.append(part)
    if not chunks:
        raise ValueError(f"No rows found for date {date}")
    return pd.concat(chunks, ignore_index=True)


def build_month_pair(prev_date: str, curr_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prev = load_month(prev_date).set_index("ncodpers")
    curr = load_month(curr_date).set_index("ncodpers")
    common = prev.index.intersection(curr.index)
    prev = prev.loc[common].reset_index()
    curr = curr.loc[common].reset_index()

    features = prepare_features(prev)
    target = ((curr[PRODUCT_COLS].values - prev[PRODUCT_COLS].values) > 0).astype(int)
    target_df = pd.DataFrame(target, columns=PRODUCT_COLS)
    return features, target_df


def build_dataset(pairs: list[tuple[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_parts = []
    target_parts = []
    for prev_date, curr_date in pairs:
        features, target = build_month_pair(prev_date, curr_date)
        feature_parts.append(features)
        target_parts.append(target)
    return pd.concat(feature_parts, ignore_index=True), pd.concat(target_parts, ignore_index=True)


def save_encoders(encoders: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(encoders, ensure_ascii=False, indent=2), encoding="utf-8")


def load_encoders(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_matrix(
    features: pd.DataFrame,
    encoders: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    cleaned = prepare_features(features)
    return encode_features(cleaned, encoders)


def product_labels() -> dict[str, str]:
    labels = {
        "ind_ahor_fin_ult1": "Сберегательный счёт",
        "ind_aval_fin_ult1": "Банковская гарантия",
        "ind_cco_fin_ult1": "Текущий счёт",
        "ind_cder_fin_ult1": "Деривативный счёт",
        "ind_cno_fin_ult1": "Зарплатный проект",
        "ind_ctju_fin_ult1": "Детский счёт",
        "ind_ctma_fin_ult1": "Особый счёт 3",
        "ind_ctop_fin_ult1": "Особый счёт",
        "ind_ctpp_fin_ult1": "Особый счёт 2",
        "ind_deco_fin_ult1": "Краткосрочный депозит",
        "ind_deme_fin_ult1": "Среднесрочный депозит",
        "ind_dela_fin_ult1": "Долгосрочный депозит",
        "ind_ecue_fin_ult1": "Цифровой счёт",
        "ind_fond_fin_ult1": "Денежные средства",
        "ind_hip_fin_ult1": "Ипотека",
        "ind_plan_fin_ult1": "Пенсионный план",
        "ind_pres_fin_ult1": "Кредит",
        "ind_reca_fin_ult1": "Налоговый счёт",
        "ind_tjcr_fin_ult1": "Кредитная карта",
        "ind_valo_fin_ult1": "Ценные бумаги",
        "ind_viv_fin_ult1": "Домашний счёт",
        "ind_nomina_ult1": "Счёт для зарплаты",
        "ind_nom_pens_ult1": "Пенсионный счёт",
        "ind_recibo_ult1": "Дебетовый счёт",
    }
    return labels
