import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { PatientData, RiskLevel } from "@/types/PatientData";
import type { Milestone, ModelInfo, SurvivalCurve } from "@/lib/mlModel";
import { Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ArrowLeft, Heart, HeartPulse, Activity, CheckCircle2, AlertCircle, TrendingUp, TrendingDown, BarChart3 } from "lucide-react";

interface PredictionResultProps {
  survivalProbability: number;
  deathProbability: number;
  riskLevel: RiskLevel;
  riskPercentile: number;
  horizonYears: number;
  milestones: Milestone[];
  survivalCurve: SurvivalCurve;
  medianSurvivalYears: number | null;
  featureContributions: { feature: string; contribution: number; direction: 'increases' | 'decreases' }[];
  modelInfo?: ModelInfo;
  warnings?: string[];
  patientData: PatientData;
  onBack: () => void;
}

const riskLabels: Record<RiskLevel, string> = {
  lower: "LOWER RISK",
  intermediate: "INTERMEDIATE RISK",
  higher: "HIGHER RISK",
};

export function PredictionResultScreen({
  survivalProbability,
  deathProbability,
  riskLevel,
  riskPercentile,
  horizonYears,
  milestones,
  survivalCurve,
  medianSurvivalYears,
  featureContributions,
  modelInfo,
  warnings = [],
  patientData,
  onBack,
}: PredictionResultProps) {
  const isLower = riskLevel === "lower";
  const survivalPct = survivalProbability * 100;
  const deathPct = deathProbability * 100;
  const higherThanPct = Math.round(riskPercentile * 100);

  const num = (v: number | null | undefined) => (v == null ? "n/a" : v.toFixed(3));
  const curveData = survivalCurve.months.map((m, i) => ({
    year: m / 12,
    survival: Math.round(survivalCurve.survival[i] * 1000) / 10,
  }));
  const patients = modelInfo?.patients != null ? modelInfo.patients.toLocaleString() : "n/a";
  const receptor = (v: string) => v.charAt(0).toUpperCase() + v.slice(1);

  const keyFactors = [
    { label: "Age", value: `${patientData.ageAtDiagnosis} years` },
    { label: "Tumor Stage", value: `Stage ${patientData.tumorStage}` },
    { label: "Histologic Grade", value: `Grade ${patientData.histologicGrade}` },
    { label: "Lymph Nodes +", value: patientData.lymphNodesPositive },
    { label: "Chemotherapy", value: patientData.chemotherapy === "yes" ? "Yes" : "No" },
    { label: "Surgery", value: patientData.surgeryType.charAt(0).toUpperCase() + patientData.surgeryType.slice(1) },
    { label: "ER Status", value: receptor(patientData.erStatus) },
    { label: "PR Status", value: receptor(patientData.prStatus) },
    { label: "HER2 Status", value: receptor(patientData.her2Status) },
  ];

  return (
    <div className="min-h-screen gradient-hero py-8 px-4">
      <div className="max-w-2xl mx-auto">
        {/* Back Button */}
        <Button
          variant="ghost"
          onClick={onBack}
          className="mb-6 text-muted-foreground hover:text-foreground animate-slide-in"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          New Prediction
        </Button>

        {/* Result Card */}
        <Card className="overflow-hidden shadow-card-hover border-border/50 animate-scale-in">
          {/* Result Header */}
          <div
            className={`p-8 text-center ${
              isLower ? "gradient-success" : "gradient-warning"
            }`}
          >
            <div
              className={`inline-flex items-center justify-center w-24 h-24 rounded-full mb-6 ${
                isLower
                  ? "bg-success-foreground/20"
                  : "bg-warning-foreground/20"
              } ${isLower ? "animate-heartbeat" : ""}`}
            >
              {isLower ? (
                <Heart className="w-12 h-12 text-success-foreground" />
              ) : (
                <HeartPulse className="w-12 h-12 text-warning-foreground" />
              )}
            </div>
            <h1 className="text-2xl md:text-3xl font-display font-bold text-success-foreground mb-1">
              {horizonYears}-Year Survival Estimate
            </h1>
            <p className="text-5xl md:text-6xl font-display font-bold text-success-foreground mt-3">
              {survivalPct.toFixed(0)}%
            </p>
            <p className="text-sm text-success-foreground/80 mt-1">
              estimated chance of being alive {horizonYears} years after diagnosis
            </p>
            <div
              className={`inline-flex items-center gap-2 px-6 py-3 rounded-full mt-5 ${
                isLower
                  ? "bg-success-foreground/20 text-success-foreground"
                  : "bg-warning-foreground/20 text-warning-foreground"
              }`}
            >
              {isLower ? (
                <CheckCircle2 className="w-6 h-6" />
              ) : (
                <AlertCircle className="w-6 h-6" />
              )}
              <span className="text-xl font-bold">{riskLabels[riskLevel]}</span>
            </div>

            {/* Risk context */}
            <div className="mt-4 text-success-foreground/80">
              <p className="text-sm">
                Estimated {horizonYears}-year risk of death: <span className="font-bold">{deathPct.toFixed(0)}%</span>
              </p>
              <p className="text-xs mt-1">
                Higher risk than {higherThanPct}% of the patients the model was trained on
              </p>
            </div>
          </div>

          {/* Details Section */}
          <div className="p-6 md:p-8 bg-card">
            {/* Survival curve */}
            <div className="mb-8 animate-fade-in">
              <div className="flex items-center gap-2 mb-4">
                <TrendingDown className="w-5 h-5 text-primary" />
                <h2 className="font-display font-semibold text-lg text-foreground">
                  Estimated Survival Over Time
                </h2>
              </div>
              <div className="grid grid-cols-4 gap-2 mb-4 text-center">
                {milestones.map((m) => (
                  <div key={m.years} className="p-3 rounded-xl bg-primary/5 border border-primary/20">
                    <p className="text-lg font-bold text-primary">{(m.survival * 100).toFixed(0)}%</p>
                    <p className="text-xs text-muted-foreground">
                      alive at {m.years} {m.years === 1 ? "year" : "years"}
                    </p>
                  </div>
                ))}
              </div>
              <div className="h-56 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={curveData} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                    <defs>
                      <linearGradient id="survivalFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.35} />
                        <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0.03} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis
                      dataKey="year"
                      type="number"
                      domain={[0, 15]}
                      ticks={[0, 3, 5, 10, 15]}
                      tickFormatter={(v) => `${v}y`}
                      tick={{ fontSize: 12 }}
                    />
                    <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 12 }} width={44} />
                    <Tooltip
                      formatter={(value: number) => [`${value}%`, "Chance of being alive"]}
                      labelFormatter={(label: number) => `${label} years after diagnosis`}
                    />
                    <ReferenceLine y={50} stroke="hsl(var(--muted-foreground))" strokeDasharray="4 4" />
                    <ReferenceLine x={horizonYears} stroke="hsl(var(--primary))" strokeDasharray="4 4" />
                    <Area
                      type="stepAfter"
                      dataKey="survival"
                      stroke="hsl(var(--primary))"
                      strokeWidth={2.5}
                      fill="url(#survivalFill)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <p className="text-xs text-muted-foreground mt-2">
                {medianSurvivalYears != null
                  ? `The estimated survival curve falls to 50% at about ${medianSurvivalYears.toFixed(1)} years.`
                  : "The estimated survival curve stays above 50% for the whole 15 years shown."}{" "}
                The further out, the less data the model has, so long-term estimates are less certain.
              </p>
            </div>

            <div className="flex items-center gap-2 mb-4">
              <Activity className="w-5 h-5 text-primary" />
              <h2 className="font-display font-semibold text-lg text-foreground">
                Key Factors Analyzed
              </h2>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
              {keyFactors.map((factor, index) => (
                <div
                  key={factor.label}
                  className="p-4 rounded-xl bg-secondary/50 border border-border/30 animate-fade-in"
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  <p className="text-xs text-muted-foreground mb-1">
                    {factor.label}
                  </p>
                  <p className="font-semibold text-foreground">{factor.value}</p>
                </div>
              ))}
            </div>

            {/* ML Model Metrics */}
            <div
              className="p-4 rounded-xl bg-primary/5 border border-primary/20 mb-6 animate-fade-in"
              style={{ animationDelay: "500ms" }}
            >
              <div className="flex items-center gap-2 mb-3">
                <BarChart3 className="w-4 h-4 text-primary" />
                <h3 className="font-semibold text-foreground text-sm">ML Model Performance</h3>
              </div>
              <div className="grid grid-cols-4 gap-2 text-center">
                <div>
                  <p className="text-lg font-bold text-primary">{num(modelInfo?.cIndex)}</p>
                  <p className="text-xs text-muted-foreground">C-index</p>
                </div>
                <div>
                  <p className="text-lg font-bold text-primary">{num(modelInfo?.auc5Year)}</p>
                  <p className="text-xs text-muted-foreground">AUC (5-year)</p>
                </div>
                <div>
                  <p className="text-lg font-bold text-primary">{num(modelInfo?.auc10Year)}</p>
                  <p className="text-xs text-muted-foreground">AUC (10-year)</p>
                </div>
                <div>
                  <p className="text-lg font-bold text-primary">{patients}</p>
                  <p className="text-xs text-muted-foreground">Patients</p>
                </div>
              </div>
              <p className="text-xs text-muted-foreground mt-3 text-center">
                Cross-validated on held-out patients (C-index: 0.5 = chance, 1 = perfect).
                {modelInfo?.brier5Year != null && modelInfo?.brier5YearNoSkill != null &&
                  ` 5-year Brier score ${modelInfo.brier5Year.toFixed(3)} vs ${modelInfo.brier5YearNoSkill.toFixed(3)} for a no-skill model (lower is better).`}
              </p>
            </div>

            {/* XAI Feature Contributions */}
            <div
              className="p-4 rounded-xl bg-muted/50 border border-border/30 animate-fade-in"
              style={{ animationDelay: "600ms" }}
            >
              <h3 className="font-semibold text-foreground mb-3 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-primary" />
                SHAP Feature Contributions
              </h3>
              <div className="space-y-2">
                {featureContributions.map((fc, index) => (
                  <div
                    key={fc.feature}
                    className="flex items-center justify-between p-2 rounded-lg bg-background/50 animate-fade-in"
                    style={{ animationDelay: `${650 + index * 50}ms` }}
                  >
                    <div className="flex items-center gap-2">
                      {fc.direction === 'increases' ? (
                        <TrendingUp className="w-4 h-4 text-destructive" />
                      ) : (
                        <TrendingDown className="w-4 h-4 text-success" />
                      )}
                      <span className="text-sm text-foreground">{fc.feature}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div
                        className={`h-2 rounded-full ${fc.direction === 'increases' ? 'bg-destructive/60' : 'bg-success/60'}`}
                        style={{ width: `${Math.min(fc.contribution * 200, 80)}px` }}
                      />
                      <span className={`text-xs font-medium ${fc.direction === 'increases' ? 'text-destructive' : 'text-success'}`}>
                        {fc.direction === 'increases' ? '↑ Risk' : '↓ Risk'}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-xs text-muted-foreground mt-3">
                Factors that increase risk lower the estimated survival chance; factors that decrease risk raise it.
              </p>
            </div>

            {warnings.length > 0 && (
              <div className="p-4 rounded-xl border border-border/50 bg-muted/50 mt-6 animate-fade-in">
                <p className="text-sm font-medium text-foreground mb-1">Please note</p>
                <ul className="list-disc pl-5 space-y-1 text-sm text-muted-foreground">
                  {warnings.map((w) => (
                    <li key={w}>{w}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Confidence Note */}
            <div
              className={`p-4 rounded-xl border mt-6 animate-fade-in ${
                isLower
                  ? "bg-success-soft border-success/30"
                  : "bg-warning-soft border-warning/30"
              }`}
              style={{ animationDelay: "800ms" }}
            >
              <p className="text-sm text-muted-foreground">
                <span className="font-medium text-foreground">Note:</span> This is a statistical estimate for
                patients with similar characteristics, not a certainty for any individual. It comes from a
                Extra Survival Trees model trained on the METABRIC dataset ({patients} patients, including those
                still alive at last follow-up). Feature effects show associations,
                not causes. For clinical decisions, always consult with healthcare professionals.
              </p>
            </div>
          </div>
        </Card>

        {/* Footer */}
        <p
          className="text-center text-xs text-muted-foreground mt-6 animate-fade-in"
          style={{ animationDelay: "900ms" }}
        >
          Breast Cancer Survival Prediction • Machine Learning + Explainable AI (SHAP)
        </p>
      </div>
    </div>
  );
}
