"""Request / response models. Field names match src/types/PatientData.ts in the frontend."""
from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator

from .features import MODEL_INPUTS

Number = Union[str, int, float]


def _to_float(v: Number, name: str, lo: float, hi: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number")
    if not np.isfinite(x) or x < lo or x > hi:
        raise ValueError(f"{name} must be between {lo:g} and {hi:g}")
    return x


class PatientData(BaseModel):
    ageAtDiagnosis: Number
    sex: Literal["female", "male"]
    menopausalState: Literal["pre", "post"]
    tumorSize: Number
    tumorStage: Literal["1", "2", "3", "4"]
    histologicGrade: Literal["1", "2", "3"]
    cancerType: Literal["invasive_ductal", "invasive_lobular", "mixed", "other"]
    erStatus: Literal["positive", "negative"]
    prStatus: Literal["positive", "negative"]
    her2Status: Literal["positive", "negative"]
    lymphNodesPositive: Number
    mutationCount: Number
    chemotherapy: Literal["yes", "no"]
    hormoneTherapy: Literal["yes", "no"]
    radiotherapy: Literal["yes", "no"]
    surgeryType: Literal["mastectomy", "lumpectomy", "none"]

    @field_validator("ageAtDiagnosis")
    @classmethod
    def _age(cls, v):
        return _to_float(v, "Age at diagnosis", 15, 110)

    @field_validator("tumorSize")
    @classmethod
    def _size(cls, v):
        return _to_float(v, "Tumor size (mm)", 0, 300)

    @field_validator("lymphNodesPositive")
    @classmethod
    def _nodes(cls, v):
        return _to_float(v, "Positive lymph nodes", 0, 100)

    @field_validator("mutationCount")
    @classmethod
    def _mut(cls, v):
        return _to_float(v, "Mutation count", 0, 500)


class FeatureContribution(BaseModel):
    feature: str
    contribution: float = Field(description="absolute impact on the log-odds of dying within 10 years")
    direction: Literal["increases", "decreases"]


class ModelInfo(BaseModel):
    name: str
    cIndex: Optional[float] = None
    auc5Year: Optional[float] = None
    auc10Year: Optional[float] = None
    brier5Year: Optional[float] = None
    brier5YearNoSkill: Optional[float] = None
    patients: Optional[int] = None
    deaths: Optional[int] = None
    horizonYears: int = 10


class Milestone(BaseModel):
    years: int
    survival: float = Field(description="estimated chance of being alive this many years after diagnosis, 0-1")


class SurvivalCurve(BaseModel):
    months: list[int]
    survival: list[float]


class PredictionResponse(BaseModel):
    survivalProbability: float = Field(description="estimated chance of being alive 10 years after diagnosis, 0-1")
    deathProbability: float = Field(description="estimated chance of dying within 10 years, 0-1")
    riskLevel: Literal["lower", "intermediate", "higher"] = Field(
        description="risk group compared with the patients the model was trained on"
    )
    riskPercentile: float = Field(description="share (0-1) of training patients with a lower risk than this patient")
    horizonYears: int
    milestones: list[Milestone]
    survivalCurve: SurvivalCurve
    medianSurvivalYears: Optional[float] = Field(
        default=None, description="time at which the survival curve falls to 50%; null if it stays above 50% for 15 years"
    )
    featureContributions: list[FeatureContribution]
    modelInfo: ModelInfo
    warnings: list[str] = []


# --------------------------------------------------------------------------- mapping
_CANCER = {
    "invasive_ductal": "Invasive Ductal",
    "invasive_lobular": "Invasive Lobular",
    "mixed": "Mixed Ductal and Lobular",
    "other": "Other",
}


def to_model_row(p: PatientData, ranges: dict) -> tuple[dict, list[str]]:
    """Convert the form values into the model's columns; also collect warnings."""
    warnings: list[str] = []
    surgery = {"mastectomy": "Mastectomy", "lumpectomy": "Breast Conserving", "none": None}[p.surgeryType]
    if surgery is None:
        warnings.append(
            "'No surgery' does not occur in the training data, so surgery type was treated as unknown."
        )
    if p.sex == "male":
        warnings.append(
            "The training data contains only female patients, so sex is not used by the model."
        )
    row = {
        "Age at Diagnosis": float(p.ageAtDiagnosis),
        "Tumor Size": float(p.tumorSize),
        "Tumor Stage": float(p.tumorStage),
        "Neoplasm Histologic Grade": float(p.histologicGrade),
        "Lymph nodes examined positive": float(p.lymphNodesPositive),
        "Mutation Count": float(p.mutationCount),
        "Inferred Menopausal State": "Pre" if p.menopausalState == "pre" else "Post",
        "Cancer Type Detailed": _CANCER[p.cancerType],
        "ER Status": p.erStatus.capitalize(),
        "PR Status": p.prStatus.capitalize(),
        "HER2 Status": p.her2Status.capitalize(),
        "Chemotherapy": "Yes" if p.chemotherapy == "yes" else "No",
        "Hormone Therapy": "Yes" if p.hormoneTherapy == "yes" else "No",
        "Radio Therapy": "Yes" if p.radiotherapy == "yes" else "No",
        "Type of Breast Surgery": surgery,
    }
    assert list(row) == MODEL_INPUTS
    labels = {
        "Age at Diagnosis": "Age",
        "Tumor Size": "Tumor size",
        "Lymph nodes examined positive": "Positive lymph nodes",
        "Mutation Count": "Mutation count",
    }
    for col, label in labels.items():
        lo, hi = ranges[col]
        if not lo <= row[col] <= hi:
            warnings.append(
                f"{label} ({row[col]:g}) is outside the range seen in training ({lo:g}-{hi:g}); "
                "treat this prediction with extra caution."
            )
    return row, warnings
