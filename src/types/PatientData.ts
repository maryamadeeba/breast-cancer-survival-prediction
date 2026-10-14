export interface PatientData {
  ageAtDiagnosis: string;
  sex: string;
  menopausalState: string;
  tumorSize: string;
  tumorStage: string;
  histologicGrade: string;
  cancerType: string;
  erStatus: string;
  prStatus: string;
  her2Status: string;
  lymphNodesPositive: string;
  mutationCount: string;
  chemotherapy: string;
  hormoneTherapy: string;
  radiotherapy: string;
  surgeryType: string;
}

export type RiskLevel = "lower" | "intermediate" | "higher";
