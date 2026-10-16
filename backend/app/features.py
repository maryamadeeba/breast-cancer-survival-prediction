"""Feature definitions shared by training (train_model.py) and serving (app/main.py).

Keeping them in one module guarantees that the model sees exactly the same
columns, encodings and engineered features at training time and at prediction time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COL = "Patient's Vital Status"
FOLLOWUP_COL = "Overall Survival (Months)"
HORIZON_YEARS = 10  # headline estimate: chance of being alive 10 years after diagnosis
MILESTONE_YEARS = [1, 3, 5, 10]
CURVE_MONTHS = list(range(0, 181, 6))  # survival curve drawn from 0 to 15 years in 6-month steps

# The 15 clinical inputs the model uses. (The form also asks for "Sex", but every
# patient in the METABRIC file is female, so sex carries no information and is not a
# model input. The API accepts it and returns a warning if "male" is chosen.)
NUMERIC_INPUTS = [
    "Age at Diagnosis",
    "Tumor Size",
    "Tumor Stage",
    "Neoplasm Histologic Grade",
    "Lymph nodes examined positive",
    "Mutation Count",
]
CATEGORICAL_INPUTS = [
    "Inferred Menopausal State",
    "Cancer Type Detailed",
    "ER Status",
    "PR Status",
    "HER2 Status",
    "Chemotherapy",
    "Hormone Therapy",
    "Radio Therapy",
    "Type of Breast Surgery",
]
MODEL_INPUTS = NUMERIC_INPUTS + CATEGORICAL_INPUTS

# The explanation treats ER, PR and HER2 as one "player" (they are read together clinically);
# this also keeps the exact Shapley computation fast (2^13 coalitions instead of 2^15).
RECEPTOR_GROUP = "Receptor status (ER/PR/HER2)"
PLAYERS: list[tuple[str, list[str]]] = [
    (col, [col]) for col in MODEL_INPUTS if col not in ("ER Status", "PR Status", "HER2 Status")
]
PLAYERS.append((RECEPTOR_GROUP, ["ER Status", "PR Status", "HER2 Status"]))

# Engineered columns (all computed from the inputs above, so no extra form fields)
ENGINEERED = ["log_size", "log_nodes", "log_mutations", "npi", "age_x_nodes", "age_centered_sq", "triple_negative"]
NUMERIC_ALL = NUMERIC_INPUTS + ENGINEERED

CANCER_TYPE_GROUPS = {
    "Breast Invasive Ductal Carcinoma": "Invasive Ductal",
    "Breast Invasive Lobular Carcinoma": "Invasive Lobular",
    "Breast Mixed Ductal and Lobular Carcinoma": "Mixed Ductal and Lobular",
}
CANCER_TYPE_OTHER = "Other"


def group_cancer_type(value) -> object:
    """Map the 8 raw METABRIC cancer subtypes onto the 4 options the form offers."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    return CANCER_TYPE_GROUPS.get(str(value), CANCER_TYPE_OTHER)


def add_engineered(X: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns. Used inside the sklearn pipelines (FunctionTransformer)."""
    X = X.copy()
    for col in NUMERIC_INPUTS:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    size = X["Tumor Size"]
    nodes = X["Lymph nodes examined positive"]
    X["log_size"] = np.log1p(size)
    X["log_nodes"] = np.log1p(nodes)
    X["log_mutations"] = np.log1p(X["Mutation Count"])
    # Nottingham Prognostic Index = 0.2 * size(cm) + lymph-node score (1/2/3) + grade
    node_score = np.where(nodes.isna(), np.nan, np.where(nodes == 0, 1.0, np.where(nodes <= 3, 2.0, 3.0)))
    X["npi"] = 0.2 * size / 10.0 + node_score + X["Neoplasm Histologic Grade"]
    X["age_x_nodes"] = X["Age at Diagnosis"] * X["log_nodes"] / 100.0
    X["age_centered_sq"] = ((X["Age at Diagnosis"] - 60.0) / 15.0) ** 2
    # Triple-negative breast cancer (ER-, PR-, HER2-) is a well-known high-risk subtype
    X["triple_negative"] = (
        (X["ER Status"] == "Negative") & (X["PR Status"] == "Negative") & (X["HER2 Status"] == "Negative")
    ).astype(float)
    return X


def load_survival_frame(csv_path: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """Load the METABRIC csv, remove duplicated patients and return (X, months, died, info).

    months = follow-up time in whole months (rounded up) since diagnosis
    died   = True if the patient died (of disease or other causes), False if still alive at last
             follow-up ("censored": we know they survived at least that long, but not beyond).
    """
    raw = pd.read_csv(csv_path)
    n_raw = len(raw)
    df = raw.dropna(subset=[TARGET_COL, FOLLOWUP_COL]).copy()
    # In the supplied file every patient appears 6 times (identical clinical values).
    # Duplicates must be removed, otherwise the same patient lands in both the training and
    # the test split and every metric is inflated (Random Forest = 100%).
    dedup_cols = MODEL_INPUTS + ["Sex", TARGET_COL, FOLLOWUP_COL]
    df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

    df["Cancer Type Detailed"] = df["Cancer Type Detailed"].map(group_cancer_type)
    X = df[MODEL_INPUTS].copy()
    for col in CATEGORICAL_INPUTS:
        X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
    months = np.maximum(np.ceil(df[FOLLOWUP_COL].astype(float).to_numpy()), 1.0)  # whole months, >= 1
    died = (df[TARGET_COL] != "Living").to_numpy()
    info = {
        "rows_in_file": int(n_raw),
        "unique_patients": int(len(df)),
        "duplicate_rows_removed": int(n_raw - len(df)),
        "deaths": int(died.sum()),
        "censored_alive": int((~died).sum()),
        "median_follow_up_months": float(np.median(months)),
        "max_follow_up_months": float(months.max()),
    }
    return X, months, died, info


def recalibrate(S, s10, slope: float, centre: float):
    """Stretch/shrink survival curves so predicted risks line up with what really happened.

    Each patient's cumulative hazard H(t) is multiplied by g = exp((slope - 1) * (log H(10y) - centre)).
    The ranking of patients never changes; only the probabilities do. `slope` and `centre` are learned
    from out-of-fold predictions in train_model.py (slope > 1 means the raw model was not extreme enough).
    """
    import numpy as np

    log_h10 = np.log(-np.log(np.clip(np.asarray(s10, dtype=float), 1e-4, 1 - 1e-4)))
    g = np.exp((slope - 1.0) * (log_h10 - centre))
    H = -np.log(np.clip(np.asarray(S, dtype=float), 1e-9, 1.0))
    return np.exp(-H * g.reshape(-1, *([1] * (H.ndim - 1))))
