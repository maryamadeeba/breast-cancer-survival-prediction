# Breast Cancer Survival Prediction

A web app that estimates a breast cancer patient's **survival curve**: the chance of being alive 1, 3, 5 and 10 years after diagnosis (and out to 15 years). It also shows a **lower / intermediate / higher risk** group and explains which inputs pushed the estimate up or down.

- **Frontend:** React + TypeScript + Vite + Tailwind + shadcn-ui + Recharts
- **Backend:** Python FastAPI that serves the trained model
- **Model (v2):** Extra Survival Trees (scikit-survival) with a calibration step, compared against a Cox proportional-hazards baseline, trained on the METABRIC dataset
- **Explainability:** exact Shapley (SHAP-style) values for every prediction

> Research and education project. It is **not** a medical device and must not be used for clinical decisions.

## What the model does

It is a **time-to-event (survival analysis)** model. For every patient it uses the follow-up time and whether the patient died; patients still alive at the last follow-up are "censored" and are used correctly instead of being thrown away. For a new patient it predicts a full survival curve S(t), the probability of still being alive at time t.

Inputs (15): age at diagnosis, menopausal state, tumor size, tumor stage, histologic grade, cancer type, ER, PR and HER2 status, positive lymph nodes, mutation count, chemotherapy, hormone therapy, radiotherapy and type of surgery. (The form also asks for sex, but all METABRIC patients are female, so sex is not used.)

## Results (honest, cross-validated)

5-fold cross-validation repeated 3 times on 2,507 de-duplicated patients (1,669 deaths). Every number is measured on patients the model had not seen.

| Metric | Extra Survival Trees (v2) | Random Survival Forest (v1) | Cox regression | No-skill reference |
|---|---|---|---|---|
| C-index (0.5 = chance, 1 = perfect) | **0.720** | 0.712 | 0.699 | 0.500 |
| Time-dependent AUC at 5 years | **0.809** | 0.789 | 0.789 | 0.500 |
| Time-dependent AUC at 10 years | **0.798** | 0.781 | 0.768 | 0.500 |
| Brier score at 5 years (lower is better) | 0.144 | 0.144 | 0.176 | 0.227 |
| Brier score at 10 years | 0.182 | 0.182 | 0.196 | 0.249 |

**What changed in v2.** A model audit showed that adding more patients would not help (the learning curve is flat), so the limit is the information in the 15 inputs. Tuning and comparing 13 model settings in repeated cross-validation, Extra Survival Trees ranked patients best (v1 forest 0.714, gradient boosting 0.713, Cox 0.699 in the same splits), and blending models added nothing. The model also under-spread its risks (low-risk patients looked a bit riskier than they were, and the sickest patients looked a bit safer), so a one-parameter calibration step (slope 1.24, fitted on out-of-fold predictions) now stretches the curves. It does not change who ranks above whom; it only fixes the probabilities. The 3-year AUC is weaker (0.673), because early deaths are harder to predict from these inputs. The C-index gain is small compared with its fold-to-fold spread (about 0.01), so treat it as a modest, consistent improvement rather than a leap.

**Calibration** (out-of-fold, checked against Kaplan-Meier curves of the actual patients): the predicted 10-year death risk matches the observed risk closely in the middle of the range (50.2% predicted vs 50.1% observed). For the sickest fifth the model still slightly underestimates risk (predicted 89%, observed 94%).

| Risk group (thirds of predicted 10-year risk) | Observed risk of death by 5 years | by 10 years |
|---|---|---|
| Lower | 11% | 26% |
| Intermediate | 25% | 48% |
| Higher | 68% | 85% |

The full report, including the calibration tables, is in `backend/artifacts/metrics.json`.

## External validation on TCGA-BRCA (a different study)

To check that the model is not just memorising METABRIC, it was tested on **TCGA breast cancer** (1,071 women, 150 deaths, median follow-up only 29 months), a separate study with different hospitals and a different era. Run it yourself: `cd backend && python external_validation.py`.

TCGA has no tumor grade, mutation count, chemotherapy or hormone-therapy field, so a version of the model using only the inputs both studies share (age, menopausal state, stage, tumor size, positive nodes, cancer type, ER/PR/HER2, surgery) was trained on METABRIC and tested on TCGA. Tumor size and node counts are approximated from T and N stage, so treat this as a good-faith check, not a perfect replication.

| Test on TCGA (never seen in training) | C-index (95% interval) |
|---|---|
| Extra Survival Trees, shared inputs | **0.709** (0.661-0.756) |
| Deployed model, missing inputs filled with typical values | 0.708 (0.655-0.759) |
| Cox regression, shared inputs | 0.696 (0.637-0.743) |
| Tumor stage alone | 0.682 (0.633-0.729) |
| Same model, internal METABRIC cross-validation | 0.708 |

Time-dependent AUC on TCGA: 0.72 at 3 years and 0.74 at 5 years.

**What this shows:** the model's ability to rank who is at higher risk carries over to a new study (0.709 vs 0.708 inside METABRIC). It is only slightly above tumor stage alone, and the intervals overlap, so the model adds a modest amount of information over a single staging number.

**What it does not show:** the absolute percentages do not transfer. On TCGA the model predicted a 5-year death risk of 48% for its highest-risk third, but about 27% actually died (15% vs 9% and 26% vs 18% in the other thirds). METABRIC patients were treated in an earlier era and deaths from any cause are counted, so survival was lower than in TCGA. TCGA also has only 253 patients followed for 5 years, so these figures are uncertain. Use the risk groups and curves to compare patients, not as exact forecasts, and recalibrate on local data before any real use.

## Things worth knowing

- The supplied CSV contains **every patient 6 times** (15,054 rows = 2,507 unique patients). With duplicates, the same patient lands in both the training and the test split, which is why an early script showed Random Forest at 100%. The pipeline removes duplicates before splitting.
- The event is death from **any cause** (cancer or other), so age matters a lot. The file's own "overall survival status" column disagrees with the vital-status column for 116 patients who died of other causes; the vital-status column is used.
- Beyond about 10-15 years few patients remain under observation, so long-term estimates are less certain.
- Sex = Male and "No surgery" are accepted by the form but flagged with a note, because they never occur in the training data.
- Feature effects are statistical associations, not causes. For example, patients who received hormone therapy were not healthier, because treatment is given to more serious cases.
- With about 2,500 patients and 15 inputs, discrimination is limited (C-index around 0.72). The model has been externally validated on TCGA-BRCA (see above); a larger cohort such as SEER, with longer follow-up, would be the next step toward trustworthy absolute probabilities.

## Project structure

```
backend/
  app/features.py      feature definitions, survival-data loader, engineered features (e.g. Nottingham Prognostic Index)
  app/ml.py            loads the model, predicts + recalibrates survival curves, computes Shapley explanations
  app/schemas.py       request validation and form-to-model mapping
  app/main.py          FastAPI app: /api/predict, /api/metrics, /api/health
  train_model.py       trains + cross-validates + checks calibration + saves artifacts/
  artifacts/           model.joblib (trained model) and metrics.json (results)
  data/                Breast_Cancer_METABRIC_Epic_Hospital.csv
  external_validation.py  checks the model on TCGA-BRCA and writes artifacts/external_validation.json
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
3. In the app, the result page shows the 10-year survival percentage, the risk group, a survival curve with 1/3/5/10-year milestones, and the model's real metrics (C-index 0.720, 2,507 patients).
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

0. Known limits found in the audit: accuracy is lower for patients 65+ (C-index 0.66), stage 3-4 (0.60) and 4+ positive nodes (0.65), and drops to about 0.63-0.68 when a whole hospital cohort is held out. Optional molecular inputs (PAM50 subtype, Integrative Cluster, 3-gene classifier) would add about +0.017 C-index but need lab tests that the form does not ask for.
1. Add a second external check on SEER (needs a signed data-use agreement) and recalibrate the probabilities for modern patients.
2. Add confidence intervals around the survival curve (for example by bootstrapping).
3. Separate cancer-specific death from deaths of other causes (competing-risks model).
