# Breast Cancer Survival Prediction

A web app that estimates a breast cancer patient's **survival curve**: the chance of being alive 1, 3, 5 and 10 years after diagnosis (and out to 15 years). It also shows a **lower / intermediate / higher risk** group and explains which inputs pushed the estimate up or down.

- **Frontend:** React + TypeScript + Vite + Tailwind + shadcn-ui + Recharts
- **Backend:** Python FastAPI that serves the trained model
- **Model:** Random Survival Forest (scikit-survival), compared against a Cox proportional-hazards baseline, trained on the METABRIC dataset
- **Explainability:** exact Shapley (SHAP-style) values for every prediction

> Research and education project. It is **not** a medical device and must not be used for clinical decisions.

## What the model does

It is a **time-to-event (survival analysis)** model. For every patient it uses the follow-up time and whether the patient died; patients still alive at the last follow-up are "censored" and are used correctly instead of being thrown away. For a new patient it predicts a full survival curve S(t), the probability of still being alive at time t.

Inputs (15): age at diagnosis, menopausal state, tumor size, tumor stage, histologic grade, cancer type, ER, PR and HER2 status, positive lymph nodes, mutation count, chemotherapy, hormone therapy, radiotherapy and type of surgery. (The form also asks for sex, but all METABRIC patients are female, so sex is not used.)

## Results (honest, cross-validated)

5-fold cross-validation repeated 3 times on 2,507 de-duplicated patients (1,669 deaths). Every number is measured on patients the model had not seen.

| Metric | Random Survival Forest | Cox regression | No-skill reference |
|---|---|---|---|
| C-index (0.5 = chance, 1 = perfect) | **0.712** | 0.699 | 0.500 |
| Time-dependent AUC at 5 years | 0.789 | 0.789 | 0.500 |
| Time-dependent AUC at 10 years | **0.781** | 0.768 | 0.500 |
| Brier score at 5 years (lower is better) | **0.144** | 0.176 | 0.227 |
| Brier score at 10 years | **0.182** | 0.196 | 0.249 |

The forest ranks patients better than Cox and, more importantly, gives better calibrated probabilities (lower Brier score). Its 3-year AUC is weaker (0.684), because early deaths are harder to predict from these inputs.

**Calibration** (out-of-fold, checked against Kaplan-Meier curves of the actual patients): the predicted 10-year death risk matches the observed risk closely in the middle of the range. For the sickest fifth of patients the model slightly underestimates risk (predicted 87%, observed 93%).

| Risk group (thirds of predicted 10-year risk) | Observed risk of death by 5 years | by 10 years |
|---|---|---|
| Lower | 10% | 24% |
| Intermediate | 25% | 51% |
| Higher | 69% | 84% |

The full report, including the calibration tables, is in `backend/artifacts/metrics.json`.

## Things worth knowing

- The supplied CSV contains **every patient 6 times** (15,054 rows = 2,507 unique patients). With duplicates, the same patient lands in both the training and the test split, which is why an early script showed Random Forest at 100%. The pipeline removes duplicates before splitting.
- The event is death from **any cause** (cancer or other), so age matters a lot. The file's own "overall survival status" column disagrees with the vital-status column for 116 patients who died of other causes; the vital-status column is used.
- Beyond about 10-15 years few patients remain under observation, so long-term estimates are less certain.
- Sex = Male and "No surgery" are accepted by the form but flagged with a note, because they never occur in the training data.
- Feature effects are statistical associations, not causes. For example, patients who received hormone therapy were not healthier, because treatment is given to more serious cases.
- With about 2,500 patients and 15 inputs, discrimination is limited (C-index around 0.71). Larger data (for example SEER) and external validation on a second dataset would make the model more trustworthy.

## Project structure

```
backend/
  app/features.py      feature definitions, survival-data loader, engineered features (e.g. Nottingham Prognostic Index)
  app/ml.py            loads the model, predicts survival curves, computes Shapley explanations
  app/schemas.py       request validation and form-to-model mapping
  app/main.py          FastAPI app: /api/predict, /api/metrics, /api/health
  train_model.py       trains + cross-validates + checks calibration + saves artifacts/
  artifacts/           model.joblib (trained model) and metrics.json (results)
  data/                Breast_Cancer_METABRIC_Epic_Hospital.csv
  tests/test_api.py    backend tests
src/                   React frontend (src/lib/mlModel.ts calls the API; src/components/PredictionResult.tsx draws the curve)
```

## Run it locally

You need **Python 3.10 to 3.13** and **Node.js 18+**. Open two terminals in the project folder.

**Terminal 1: backend**

```
cd backend
python -m venv .venv
.venv\Scripts\activate          (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

**Terminal 2: frontend**

```
npm install
npm run dev
```

Open http://localhost:8080, fill in the form and click **Estimate Survival Chance**.

If Vite prints `EAFNOSUPPORT` or cannot bind the address, run `npx vite --host 127.0.0.1 --port 8080` instead.

### Check that it works

1. http://127.0.0.1:8000/api/health should show `{"status":"ok","modelLoaded":true}`.
2. http://127.0.0.1:8000/docs lets you try `/api/predict` interactively.
3. In the app, the result page shows the 10-year survival percentage, the risk group, a survival curve with 1/3/5/10-year milestones, and the model's real metrics (C-index 0.712, 2,507 patients).
4. Try a young, low-stage, ER-positive patient (a high, flat curve, "Lower risk") and an older patient with many positive nodes and triple-negative receptors (a steep curve, "Higher risk").
5. Run the tests: `cd backend && python -m pytest -q` (10 tests) and, in the project root, `npm test` (3 tests).
6. Stop the backend and press the button again: the app shows "Cannot reach the prediction server".

### Optional

- **Retrain the model** (about 1-2 minutes, prints cross-validation results): `cd backend && python train_model.py`
- **One-server mode:** run `npm run build`, then start the backend; the built site is served at http://127.0.0.1:8000.
- If the saved model cannot be loaded (for example, a different scikit-learn version), the backend retrains automatically on startup (about 40 seconds).
- To use a deployed backend, set `VITE_API_URL=https://your-server/api` before `npm run build`.

## API

`POST /api/predict` with the same fields as the form, for example:

```json
{
  "ageAtDiagnosis": "55", "sex": "female", "menopausalState": "post",
  "tumorSize": "25", "tumorStage": "2", "histologicGrade": "2",
  "cancerType": "invasive_ductal", "erStatus": "positive", "prStatus": "positive", "her2Status": "negative",
  "lymphNodesPositive": "3", "mutationCount": "4",
  "chemotherapy": "no", "hormoneTherapy": "yes", "radiotherapy": "yes",
  "surgeryType": "lumpectomy"
}
```

It returns `survivalProbability` (10 years), `deathProbability`, `riskLevel` (`lower` / `intermediate` / `higher`), `riskPercentile`, `milestones` (1, 3, 5 and 10 years), `survivalCurve` (every 6 months up to 15 years), `medianSurvivalYears` (null if the curve stays above 50%), the top 6 feature contributions, the model's metrics, and any warnings.

## Roadmap

1. Train on a larger dataset such as SEER and validate externally on METABRIC or TCGA.
2. Add confidence intervals around the survival curve (for example by bootstrapping).
3. Separate cancer-specific death from deaths of other causes (competing-risks model).
