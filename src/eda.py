from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import ARTIFACTS_DIR, DATA_PATH, PRODUCT_COLS
from src.data import load_month


def run_eda(sample_date: str = "2015-06-28") -> dict:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = ARTIFACTS_DIR / "eda"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_month(sample_date)
    summary = {
        "sample_date": sample_date,
        "rows": int(len(df)),
        "unique_clients": int(df["ncodpers"].nunique()),
        "missing_renta_share": float(df["renta"].isna().mean()),
        "active_clients_share": float(df["ind_actividad_cliente"].mean()),
    }

    product_rate = df[PRODUCT_COLS].mean().sort_values(ascending=False)
    summary["top_products"] = product_rate.head(5).to_dict()
    summary["product_penetration_mean"] = float(product_rate.mean())

    monthly_rows = []
    for chunk in pd.read_csv(DATA_PATH, usecols=["fecha_dato", "ncodpers"], chunksize=500_000):
        chunk["fecha_dato"] = pd.to_datetime(chunk["fecha_dato"])
        monthly_rows.append(chunk.groupby("fecha_dato").size())
    monthly = pd.concat(monthly_rows).groupby(level=0).sum()

    sns.set_theme(style="whitegrid")

    product_rate.plot(kind="barh", figsize=(10, 8), title=f"Product penetration ({sample_date})")
    plt.tight_layout()
    plt.savefig(figures_dir / "product_penetration.png", dpi=120)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    df["segmento"].value_counts().plot(kind="bar", ax=axes[0], title="segmento")
    df["ind_actividad_cliente"].value_counts().plot(kind="bar", ax=axes[1], title="activity")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age"].dropna().hist(bins=30, ax=axes[2])
    axes[2].set_title("age")
    plt.tight_layout()
    plt.savefig(figures_dir / "client_segments.png", dpi=120)
    plt.close()

    monthly.plot(figsize=(10, 4), title="Rows by month")
    plt.ylabel("rows")
    plt.tight_layout()
    plt.savefig(figures_dir / "monthly_rows.png", dpi=120)
    plt.close()

    corr = df[PRODUCT_COLS].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Product correlation")
    plt.tight_layout()
    plt.savefig(figures_dir / "product_correlation.png", dpi=120)
    plt.close()

    summary_path = ARTIFACTS_DIR / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    run_eda()
