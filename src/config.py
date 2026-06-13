from pathlib import Path

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "train_ver2.csv"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"

MLFLOW_EXPERIMENT = "bank_product_recommendations"
TOP_K = 7

PRODUCT_COLS = [
    "ind_ahor_fin_ult1",
    "ind_aval_fin_ult1",
    "ind_cco_fin_ult1",
    "ind_cder_fin_ult1",
    "ind_cno_fin_ult1",
    "ind_ctju_fin_ult1",
    "ind_ctma_fin_ult1",
    "ind_ctop_fin_ult1",
    "ind_ctpp_fin_ult1",
    "ind_deco_fin_ult1",
    "ind_deme_fin_ult1",
    "ind_dela_fin_ult1",
    "ind_ecue_fin_ult1",
    "ind_fond_fin_ult1",
    "ind_hip_fin_ult1",
    "ind_plan_fin_ult1",
    "ind_pres_fin_ult1",
    "ind_reca_fin_ult1",
    "ind_tjcr_fin_ult1",
    "ind_valo_fin_ult1",
    "ind_viv_fin_ult1",
    "ind_nomina_ult1",
    "ind_nom_pens_ult1",
    "ind_recibo_ult1",
]

NUMERIC_FEATURES = ["age", "antiguedad", "renta", "n_products"]
CATEGORICAL_FEATURES = [
    "sexo",
    "segmento",
    "ind_empleado",
    "canal_entrada",
    "indrel_1mes",
    "tiprel_1mes",
]
BINARY_FEATURES = [
    "ind_actividad_cliente",
    "ind_nuevo",
    "indresi",
    "indext",
    "conyuemp",
]

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES + PRODUCT_COLS

TRAIN_PAIRS = [
    ("2015-10-28", "2015-11-28"),
    ("2015-11-28", "2015-12-28"),
]
VAL_PAIR = ("2015-12-28", "2016-01-28")

# Ограничение для VM с 8 GB RAM (без swap)
TRAIN_SAMPLE_SIZE = 400_000
VAL_SAMPLE_SIZE = 200_000

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": RANDOM_STATE,
}
