import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { FormField } from "./FormField";
import { PatientData } from "@/types/PatientData";
import { Activity, Stethoscope, ArrowRight, HeartPulse } from "lucide-react";

interface PredictionFormProps {
  onPredict: (data: PatientData) => void;
  isLoading?: boolean;
}

const initialFormData: PatientData = {
  ageAtDiagnosis: "",
  sex: "",
  menopausalState: "",
  tumorSize: "",
  tumorStage: "",
  histologicGrade: "",
  cancerType: "",
  erStatus: "",
  prStatus: "",
  her2Status: "",
  lymphNodesPositive: "",
  mutationCount: "",
  chemotherapy: "",
  hormoneTherapy: "",
  radiotherapy: "",
  surgeryType: "",
};

const sexOptions = [
  { value: "female", label: "Female" },
  { value: "male", label: "Male" },
];

const menopausalOptions = [
  { value: "pre", label: "Pre-Menopausal" },
  { value: "post", label: "Post-Menopausal" },
];

const tumorStageOptions = [
  { value: "1", label: "Stage I" },
  { value: "2", label: "Stage II" },
  { value: "3", label: "Stage III" },
  { value: "4", label: "Stage IV" },
];

const gradeOptions = [
  { value: "1", label: "Grade 1 (Low)" },
  { value: "2", label: "Grade 2 (Intermediate)" },
  { value: "3", label: "Grade 3 (High)" },
];

const cancerTypeOptions = [
  { value: "invasive_ductal", label: "Invasive Ductal Carcinoma" },
  { value: "invasive_lobular", label: "Invasive Lobular Carcinoma" },
  { value: "mixed", label: "Mixed Type" },
  { value: "other", label: "Other" },
];

const receptorOptions = [
  { value: "positive", label: "Positive" },
  { value: "negative", label: "Negative" },
];

const yesNoOptions = [
  { value: "yes", label: "Yes" },
  { value: "no", label: "No" },
];

const surgeryOptions = [
  { value: "mastectomy", label: "Mastectomy" },
  { value: "lumpectomy", label: "Lumpectomy" },
  { value: "none", label: "No Surgery" },
];

export function PredictionForm({ onPredict, isLoading = false }: PredictionFormProps) {
  const [formData, setFormData] = useState<PatientData>(initialFormData);

  const handleChange = (field: keyof PatientData) => (value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onPredict(formData);
  };

  const isFormValid = Object.values(formData).every((value) => value !== "");

  return (
    <div className="min-h-screen gradient-hero py-8 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-10 animate-fade-in">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl gradient-primary shadow-primary mb-6">
            <HeartPulse className="w-8 h-8 text-primary-foreground" />
          </div>
          <h1 className="text-3xl md:text-4xl font-display font-bold text-foreground mb-3">
            Breast Cancer Survival Prediction
          </h1>
          <p className="text-muted-foreground max-w-xl mx-auto">
            Enter patient clinical and treatment data to estimate the chance of
            10-year survival using machine learning
          </p>
        </div>

        {/* Form Card */}
        <Card className="p-6 md:p-8 shadow-card gradient-card border-border/50 animate-scale-in">
          <form onSubmit={handleSubmit} className="space-y-8">
            {/* Patient Information Section */}
            <div className="space-y-4">
              <div className="flex items-center gap-2 pb-2 border-b border-border/50">
                <Activity className="w-5 h-5 text-primary" />
                <h2 className="font-display font-semibold text-lg text-foreground">
                  Patient Information
                </h2>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <FormField
                  label="Age at Diagnosis"
                  name="ageAtDiagnosis"
                  type="input"
                  inputType="number"
                  placeholder="e.g., 55"
                  value={formData.ageAtDiagnosis}
                  onChange={handleChange("ageAtDiagnosis")}
                  delay={50}
                />
                <FormField
                  label="Sex"
                  name="sex"
                  type="select"
                  placeholder="Select sex"
                  options={sexOptions}
                  value={formData.sex}
                  onChange={handleChange("sex")}
                  delay={100}
                />
                <FormField
                  label="Menopausal State"
                  name="menopausalState"
                  type="select"
                  placeholder="Select state"
                  options={menopausalOptions}
                  value={formData.menopausalState}
                  onChange={handleChange("menopausalState")}
                  delay={150}
                />
              </div>
            </div>

            {/* Cancer Characteristics Section */}
            <div className="space-y-4">
              <div className="flex items-center gap-2 pb-2 border-b border-border/50">
                <Stethoscope className="w-5 h-5 text-primary" />
                <h2 className="font-display font-semibold text-lg text-foreground">
                  Cancer Characteristics
                </h2>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <FormField
                  label="Tumor Size (mm)"
                  name="tumorSize"
                  type="input"
                  inputType="number"
                  placeholder="e.g., 25"
                  value={formData.tumorSize}
                  onChange={handleChange("tumorSize")}
                  delay={200}
                />
                <FormField
                  label="Tumor Stage"
                  name="tumorStage"
                  type="select"
                  placeholder="Select stage"
                  options={tumorStageOptions}
                  value={formData.tumorStage}
                  onChange={handleChange("tumorStage")}
                  delay={250}
                />
                <FormField
                  label="Histologic Grade"
                  name="histologicGrade"
                  type="select"
                  placeholder="Select grade"
                  options={gradeOptions}
                  value={formData.histologicGrade}
                  onChange={handleChange("histologicGrade")}
                  delay={300}
                />
                <FormField
                  label="Cancer Type"
                  name="cancerType"
                  type="select"
                  placeholder="Select type"
                  options={cancerTypeOptions}
                  value={formData.cancerType}
                  onChange={handleChange("cancerType")}
                  delay={350}
                />
                <FormField
                  label="ER Status"
                  name="erStatus"
                  type="select"
                  placeholder="Select ER status"
                  options={receptorOptions}
                  value={formData.erStatus}
                  onChange={handleChange("erStatus")}
                  delay={375}
                />
                <FormField
                  label="PR Status"
                  name="prStatus"
                  type="select"
                  placeholder="Select PR status"
                  options={receptorOptions}
                  value={formData.prStatus}
                  onChange={handleChange("prStatus")}
                  delay={385}
                />
                <FormField
                  label="HER2 Status"
                  name="her2Status"
                  type="select"
                  placeholder="Select HER2 status"
                  options={receptorOptions}
                  value={formData.her2Status}
                  onChange={handleChange("her2Status")}
                  delay={395}
                />
                <FormField
                  label="Lymph Nodes Positive"
                  name="lymphNodesPositive"
                  type="input"
                  inputType="number"
                  placeholder="e.g., 3"
                  value={formData.lymphNodesPositive}
                  onChange={handleChange("lymphNodesPositive")}
                  delay={400}
                />
                <FormField
                  label="Mutation Count"
                  name="mutationCount"
                  type="input"
                  inputType="number"
                  placeholder="e.g., 5"
                  value={formData.mutationCount}
                  onChange={handleChange("mutationCount")}
                  delay={450}
                />
              </div>
            </div>

            {/* Treatment Details Section */}
            <div className="space-y-4">
              <div className="flex items-center gap-2 pb-2 border-b border-border/50">
                <Activity className="w-5 h-5 text-primary" />
                <h2 className="font-display font-semibold text-lg text-foreground">
                  Treatment Details
                </h2>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <FormField
                  label="Chemotherapy"
                  name="chemotherapy"
                  type="select"
                  placeholder="Select"
                  options={yesNoOptions}
                  value={formData.chemotherapy}
                  onChange={handleChange("chemotherapy")}
                  delay={500}
                />
                <FormField
                  label="Hormone Therapy"
                  name="hormoneTherapy"
                  type="select"
                  placeholder="Select"
                  options={yesNoOptions}
                  value={formData.hormoneTherapy}
                  onChange={handleChange("hormoneTherapy")}
                  delay={550}
                />
                <FormField
                  label="Radiotherapy"
                  name="radiotherapy"
                  type="select"
                  placeholder="Select"
                  options={yesNoOptions}
                  value={formData.radiotherapy}
                  onChange={handleChange("radiotherapy")}
                  delay={600}
                />
                <FormField
                  label="Type of Surgery"
                  name="surgeryType"
                  type="select"
                  placeholder="Select surgery"
                  options={surgeryOptions}
                  value={formData.surgeryType}
                  onChange={handleChange("surgeryType")}
                  delay={650}
                />
              </div>
            </div>

            {/* Submit Button */}
            <div className="pt-4 animate-fade-in" style={{ animationDelay: "700ms" }}>
              <Button
                type="submit"
                variant="predict"
                size="xl"
                className="w-full md:w-auto md:min-w-[280px] mx-auto flex"
                disabled={!isFormValid || isLoading}
              >
                {isLoading ? "Estimating..." : "Estimate Survival Chance"}
                {!isLoading && <ArrowRight className="w-5 h-5 ml-1" />}
              </Button>
              {!isFormValid && (
                <p className="text-sm text-muted-foreground text-center mt-3">
                  Please fill in all fields to enable prediction
                </p>
              )}
            </div>
          </form>
        </Card>

        {/* Footer Note */}
        <p className="text-center text-xs text-muted-foreground mt-6 animate-fade-in" style={{ animationDelay: "800ms" }}>
          Powered by Machine Learning & Explainable AI (SHAP) • University Project
        </p>
      </div>
    </div>
  );
}
