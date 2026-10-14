import { useState } from "react";
import { PredictionForm } from "@/components/PredictionForm";
import { PredictionResultScreen } from "@/components/PredictionResult";
import { PatientData } from "@/types/PatientData";
import { predictSurvival, PredictionOutput } from "@/lib/mlModel";
import { useToast } from "@/hooks/use-toast";

const Index = () => {
  const [step, setStep] = useState<"form" | "result">("form");
  const [patientData, setPatientData] = useState<PatientData | null>(null);
  const [predictionOutput, setPredictionOutput] = useState<PredictionOutput | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  const handlePredict = async (data: PatientData) => {
    setPatientData(data);
    setIsLoading(true);

    try {
      // The trained model lives in the Python backend (FastAPI)
      const output = await predictSurvival(data);
      setPredictionOutput(output);
      setStep("result");
    } catch (error) {
      toast({
        title: "Prediction failed",
        description: error instanceof Error ? error.message : "Something went wrong.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleBack = () => {
    setStep("form");
    setPredictionOutput(null);
  };

  return (
    <>
      {step === "form" && <PredictionForm onPredict={handlePredict} isLoading={isLoading} />}
      {step === "result" && patientData && predictionOutput && (
        <PredictionResultScreen
          survivalProbability={predictionOutput.survivalProbability}
          deathProbability={predictionOutput.deathProbability}
          riskLevel={predictionOutput.riskLevel}
          riskPercentile={predictionOutput.riskPercentile}
          horizonYears={predictionOutput.horizonYears}
          milestones={predictionOutput.milestones}
          survivalCurve={predictionOutput.survivalCurve}
          medianSurvivalYears={predictionOutput.medianSurvivalYears}
          featureContributions={predictionOutput.featureContributions}
          modelInfo={predictionOutput.modelInfo}
          warnings={predictionOutput.warnings}
          patientData={patientData}
          onBack={handleBack}
        />
      )}
    </>
  );
};

export default Index;
