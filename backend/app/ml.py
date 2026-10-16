"""Loads the trained Extra Survival Trees and turns a patient into a survival curve + explanation."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import CATEGORICAL_INPUTS, MODEL_INPUTS, NUMERIC_INPUTS, PLAYERS, recalibrate

log = logging.getLogger("survival")

BACKEND_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BACKEND_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"

# Exact Shapley values need every coalition of "players" (features; ER/PR/HER2 count as one player).
N = len(PLAYERS)  # 13 players -> 8192 coalitions
_MASKS = np.arange(2 ** N)
_BITS = ((_MASKS[:, None] >> np.arange(N)[None, :]) & 1).astype(bool)  # (8192, 13)
_POPCOUNT = _BITS.sum(axis=1)
_FACT = [math.factorial(i) for i in range(N + 1)]
_WEIGHT = np.array([_FACT[k] * _FACT[N - k - 1] / _FACT[N] for k in range(N)])  # weight for |S| = k

RISK_LABELS = ("lower", "intermediate", "higher")


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


class SurvivalModel:
    def __init__(self, artifact: dict):
        self.pipeline = artifact["model"]
        self.baseline: dict = artifact["baseline"]
        self.ranges: dict = artifact["ranges"]
        self.metrics: dict = artifact["metrics"]
        self.risk_cutoffs: list[float] = artifact["risk_cutoffs"]
        self.risk_distribution = np.asarray(artifact["risk_distribution"])
        self.horizon_years: int = artifact["horizon_years"]
        self.milestone_years: list[int] = artifact["milestone_years"]
        self.curve_months: list[int] = artifact["curve_months"]
        self.recal: dict | None = artifact.get("recalibration")  # absent in models saved by v1
        self._forest = self.pipeline.named_steps["model"]
        self._times = np.asarray(self._forest.unique_times_, dtype=float)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "SurvivalModel":
        try:
            return cls(joblib.load(path))
        except Exception as exc:  # missing/outdated file or library-version mismatch
            log.warning("Could not load %s (%s). Re-training the model now (about 40 s)...", path, exc)
            from train_model import DEFAULT_DATA, train  # noqa: WPS433 (local import on purpose)

            train(DEFAULT_DATA, path.parent, run_cv=False)
            return cls(joblib.load(path))

    # --------------------------------------------------------------- prediction
    def survival(self, X: pd.DataFrame, months) -> np.ndarray:
        """P(alive at each of `months`), shape (len(X), len(months))."""
        S = self._forest.predict_survival_function(self.pipeline[:-1].transform(X), return_array=True)
        idx = np.searchsorted(self._times, np.asarray(months, dtype=float), side="right") - 1
        out = S[:, np.clip(idx, 0, None)]
        out[:, idx < 0] = 1.0  # before the first event everybody is alive
        if self.recal:
            i10 = int(np.searchsorted(self._times, 12.0 * self.horizon_years, side="right")) - 1
            out = recalibrate(out, S[:, max(i10, 0)], self.recal["slope"], self.recal["centre"])
        return out

    def _frame(self, row: dict) -> pd.DataFrame:
        df = pd.DataFrame([row], columns=MODEL_INPUTS)
        for c in NUMERIC_INPUTS:
            df[c] = df[c].astype(float)
        for c in CATEGORICAL_INPUTS:
            df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
        return df

    def predict(self, row: dict) -> dict:
        months = np.array(self.curve_months, dtype=float)
        curve = self.survival(self._frame(row), months)[0]
        curve = np.minimum.accumulate(np.clip(curve, 0.0, 1.0))  # a survival curve can only go down
        at = lambda yr: float(np.interp(12 * yr, months, curve))  # noqa: E731

        horizon = float(np.interp(12 * self.horizon_years, months, curve))
        risk = 1.0 - horizon
        t1, t2 = self.risk_cutoffs
        level = RISK_LABELS[0] if risk < t1 else RISK_LABELS[1] if risk < t2 else RISK_LABELS[2]
        percentile = float(np.searchsorted(self.risk_distribution, risk) / len(self.risk_distribution))

        below = np.nonzero(curve <= 0.5)[0]
        median_years = float(months[below[0]] / 12) if len(below) else None  # None: stays above 50% for 15 years
        return {
            "survival_probability": horizon,
            "death_probability": risk,
            "risk_level": level,
            "risk_percentile": percentile,
            "milestones": [{"years": yr, "survival": at(yr)} for yr in self.milestone_years],
            "curve": {"months": [int(m) for m in months], "survival": [float(s) for s in curve]},
            "median_survival_years": median_years,
            "contributions": self.explain(row),
        }

    # ------------------------------------------------------------- explanation
    def explain(self, row: dict) -> list[dict]:
        """Exact Shapley values (in log-odds) of each input versus a typical reference patient.

        Value function: v(S) = logit(P(death within 10 years)) when only the inputs in S take the
        patient's values and all the others are set to the reference patient's values.
        Values > 0 push the risk up, values < 0 push it down.
        """
        data = {}
        for col in MODEL_INPUTS:
            player = next(i for i, (_, cols) in enumerate(PLAYERS) if col in cols)
            use_patient = _BITS[:, player]
            patient_value, base_value = row[col], self.baseline[col]
            if col in NUMERIC_INPUTS:
                pv = np.nan if patient_value is None else float(patient_value)
                data[col] = np.where(use_patient, pv, float(base_value))
            else:
                pv = np.nan if patient_value is None else patient_value
                arr = np.empty(len(_MASKS), dtype=object)
                arr[:] = base_value
                arr[use_patient] = pv
                data[col] = arr
        X = pd.DataFrame(data, columns=MODEL_INPUTS)
        v = _logit(1.0 - self.survival(X, [12 * self.horizon_years])[:, 0])  # indexed by coalition mask

        phi = np.zeros(N)
        for j in range(N):
            without = _MASKS[~_BITS[:, j]]
            with_j = without | (1 << j)
            phi[j] = float(np.sum(_WEIGHT[_POPCOUNT[without]] * (v[with_j] - v[without])))
        return [{"feature": PLAYERS[j][0], "value": float(phi[j])} for j in range(N)]

    # -------------------------------------------------------------------- info
    def info(self) -> dict:
        h = self.metrics.get("headline") or {}
        data = self.metrics.get("data", {})
        return {
            "name": "Extra Survival Trees",
            "cIndex": h.get("c_index"),
            "auc5Year": h.get("auc_5y"),
            "auc10Year": h.get("auc_10y"),
            "brier5Year": h.get("brier_5y"),
            "brier5YearNoSkill": h.get("brier_noskill_5y"),
            "patients": data.get("unique_patients"),
            "deaths": data.get("deaths"),
            "horizonYears": self.horizon_years,
        }


def read_metrics_json() -> dict:
    p = ARTIFACT_DIR / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}
