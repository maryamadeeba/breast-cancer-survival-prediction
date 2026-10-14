"""Backend tests. Run from the backend/ folder:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import STATE, app  # noqa: E402
from app.ml import _logit  # noqa: E402

PATIENT = {
    "ageAtDiagnosis": "55", "sex": "female", "menopausalState": "post", "tumorSize": "25",
    "tumorStage": "2", "histologicGrade": "2", "cancerType": "invasive_ductal",
    "erStatus": "positive", "prStatus": "positive", "her2Status": "negative",
    "lymphNodesPositive": "3", "mutationCount": "4", "chemotherapy": "no",
    "hormoneTherapy": "yes", "radiotherapy": "yes", "surgeryType": "lumpectomy",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs the lifespan -> loads the model
        yield c


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok", "modelLoaded": True}


def test_predict_shape(client):
    r = client.post("/api/predict", json=PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["survivalProbability"] <= 1
    assert body["survivalProbability"] + body["deathProbability"] == pytest.approx(1.0)
    assert body["riskLevel"] in ("lower", "intermediate", "higher")
    assert 0 <= body["riskPercentile"] <= 1
    assert body["horizonYears"] == 10
    assert len(body["featureContributions"]) == 6
    assert all(c["direction"] in ("increases", "decreases") for c in body["featureContributions"])
    assert body["modelInfo"]["patients"] == 2507
    assert body["warnings"] == []


def test_survival_curve_and_milestones(client):
    body = client.post("/api/predict", json=PATIENT).json()
    curve = body["survivalCurve"]
    assert curve["months"][0] == 0 and curve["months"][-1] == 180 and len(curve["months"]) == len(curve["survival"])
    assert curve["survival"][0] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(curve["survival"], curve["survival"][1:]))  # never increases
    years = [m["years"] for m in body["milestones"]]
    surv = [m["survival"] for m in body["milestones"]]
    assert years == [1, 3, 5, 10]
    assert surv == sorted(surv, reverse=True)
    assert surv[-1] == pytest.approx(body["survivalProbability"])  # headline = 10-year milestone


def test_higher_risk_profile_gets_higher_probability(client):
    low = client.post("/api/predict", json={**PATIENT, "ageAtDiagnosis": "45", "menopausalState": "pre", "lymphNodesPositive": "0", "tumorStage": "1", "tumorSize": "10", "histologicGrade": "1"}).json()
    high = client.post("/api/predict", json={**PATIENT, "ageAtDiagnosis": "80", "lymphNodesPositive": "15", "tumorStage": "3", "tumorSize": "70", "histologicGrade": "3"}).json()
    assert low["riskLevel"] == "lower" and high["riskLevel"] == "higher"
    assert low["survivalProbability"] > high["survivalProbability"] + 0.4
    assert high["medianSurvivalYears"] is not None  # a high-risk curve crosses 50% within 15 years


def test_validation_errors(client):
    assert client.post("/api/predict", json={**PATIENT, "ageAtDiagnosis": "abc"}).status_code == 422
    assert client.post("/api/predict", json={**PATIENT, "ageAtDiagnosis": "-5"}).status_code == 422
    assert client.post("/api/predict", json={**PATIENT, "tumorStage": "9"}).status_code == 422
    assert client.post("/api/predict", json={**PATIENT, "erStatus": "maybe"}).status_code == 422
    bad = {k: v for k, v in PATIENT.items() if k != "tumorSize"}
    assert client.post("/api/predict", json=bad).status_code == 422


def test_warnings_for_unsupported_inputs(client):
    r = client.post("/api/predict", json={**PATIENT, "sex": "male", "surgeryType": "none", "tumorSize": "250"}).json()
    text = " ".join(r["warnings"])
    assert "female" in text and "surgery" in text.lower() and "outside the range" in text


def test_numbers_may_be_sent_as_numbers(client):
    r = client.post("/api/predict", json={**PATIENT, "ageAtDiagnosis": 55, "tumorSize": 25.5})
    assert r.status_code == 200


def test_shapley_values_add_up_to_prediction_minus_baseline():
    """Exact Shapley values must satisfy: sum(phi) = logit(P(patient)) - logit(P(reference patient))."""
    from app.schemas import PatientData, to_model_row
    model = STATE.get("model") or __import__("app.ml", fromlist=["SurvivalModel"]).SurvivalModel.load()
    row, _ = to_model_row(PatientData(**PATIENT), model.ranges)
    phi = sum(c["value"] for c in model.explain(row))  # includes the grouped ER/PR/HER2 player
    month = 12 * model.horizon_years
    risk_patient = 1.0 - model.survival(model._frame(row), [month])[0, 0]
    risk_base = 1.0 - model.survival(model._frame(model.baseline), [month])[0, 0]
    assert phi == pytest.approx(float(_logit(risk_patient) - _logit(risk_base)), abs=1e-6)


def test_receptor_status_is_used(client):
    a = client.post("/api/predict", json=PATIENT).json()
    b = client.post("/api/predict", json={**PATIENT, "erStatus": "negative", "prStatus": "negative", "her2Status": "negative"}).json()
    assert a["deathProbability"] != b["deathProbability"]


def test_metrics_endpoint(client):
    m = client.get("/api/metrics").json()
    rsf, cox = m["cross_validation"]["random_survival_forest"], m["cross_validation"]["cox_proportional_hazards"]
    assert rsf["c_index"] > 0.68 and rsf["c_index"] > cox["c_index"]
    assert rsf["brier_5y"] < rsf["brier_noskill_5y"]
    rates = m["calibration"]["risk_groups_10y"]
    assert (rates["lower"]["observed_10y_risk_kaplan_meier"]
            < rates["intermediate"]["observed_10y_risk_kaplan_meier"]
            < rates["higher"]["observed_10y_risk_kaplan_meier"])
