import { PatientData, RiskLevel } from "@/types/PatientData";

// The model now runs in the Python backend (backend/app/main.py).
// In development Vite proxies /api -> http://127.0.0.1:8000 (see vite.config.ts).
// To point at a deployed backend, set VITE_API_URL, e.g. VITE_API_URL=https://my-api.example.com/api
const API_BASE = ((import.meta.env.VITE_API_URL as string | undefined) ?? "/api").replace(/\/$/, "");

export interface ModelInfo {
  name: string;
  cIndex: number | null;
  auc5Year: number | null;
  auc10Year: number | null;
  brier5Year: number | null;
  brier5YearNoSkill: number | null;
  patients: number | null;
  deaths: number | null;
  horizonYears: number;
}

export interface Milestone {
  years: number;
  survival: number; // chance of being alive this many years after diagnosis (0-1)
}

export interface SurvivalCurve {
  months: number[];
  survival: number[];
}

export interface PredictionOutput {
  survivalProbability: number; // estimated chance of being alive 10 years after diagnosis (0-1)
  deathProbability: number; // estimated chance of dying within 10 years (0-1)
  riskLevel: RiskLevel; // compared with the patients the model was trained on
  riskPercentile: number; // share (0-1) of training patients with a lower risk
  horizonYears: number;
  milestones: Milestone[];
  survivalCurve: SurvivalCurve;
  medianSurvivalYears: number | null; // null = the curve stays above 50% for the whole 15 years
  featureContributions: { feature: string; contribution: number; direction: "increases" | "decreases" }[];
  modelInfo?: ModelInfo;
  warnings?: string[];
}

export class PredictionError extends Error {}

export async function predictSurvival(data: PatientData): Promise<PredictionOutput> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
  } catch {
    throw new PredictionError(
      "Cannot reach the prediction server. Make sure the backend is running (see README) and try again."
    );
  }

  if (!response.ok) {
    let message = `The prediction server returned an error (${response.status}).`;
    try {
      const body = await response.json();
      if (Array.isArray(body.detail) && body.detail.length > 0) {
        message = body.detail
          .map((d: { msg?: string }) => (d.msg ?? "").replace(/^Value error, /, ""))
          .filter(Boolean)
          .join(" ");
      }
    } catch {
      /* keep default message */
    }
    throw new PredictionError(message);
  }
  return (await response.json()) as PredictionOutput;
}

export async function getModelMetrics() {
  const response = await fetch(`${API_BASE}/metrics`);
  if (!response.ok) throw new PredictionError("Could not load model metrics.");
  return response.json();
}
