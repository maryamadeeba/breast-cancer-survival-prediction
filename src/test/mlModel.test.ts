import { describe, it, expect, vi, afterEach } from "vitest";
import { predictSurvival, PredictionError } from "@/lib/mlModel";
import type { PatientData } from "@/types/PatientData";

const patient: PatientData = {
  ageAtDiagnosis: "55",
  sex: "female",
  menopausalState: "post",
  tumorSize: "25",
  tumorStage: "2",
  histologicGrade: "2",
  cancerType: "invasive_ductal",
  erStatus: "positive",
  prStatus: "positive",
  her2Status: "negative",
  lymphNodesPositive: "3",
  mutationCount: "4",
  chemotherapy: "no",
  hormoneTherapy: "yes",
  radiotherapy: "yes",
  surgeryType: "lumpectomy",
};

afterEach(() => vi.unstubAllGlobals());

describe("predictSurvival (backend client)", () => {
  it("POSTs the form data to /api/predict and returns the backend result", async () => {
    const payload = {
      survivalProbability: 0.72,
      deathProbability: 0.28,
      riskLevel: "lower",
      riskPercentile: 0.17,
      horizonYears: 10,
      milestones: [{ years: 10, survival: 0.72 }],
      survivalCurve: { months: [0, 6], survival: [1, 0.99] },
      medianSurvivalYears: null,
      featureContributions: [],
    };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => payload });
    vi.stubGlobal("fetch", fetchMock);

    const out = await predictSurvival(patient);

    expect(out.survivalProbability).toBe(0.72);
    expect(out.riskLevel).toBe("lower");
    expect(out.survivalCurve.months).toHaveLength(2);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/predict$/);
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual(patient);
  });

  it("explains clearly when the backend is not running", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(predictSurvival(patient)).rejects.toThrow(/backend is running/);
  });

  it("shows validation messages returned by the API", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 422,
        json: async () => ({ detail: [{ msg: "Value error, Age at diagnosis must be between 15 and 110" }] }),
      })
    );
    const err = await predictSurvival(patient).catch((e) => e);
    expect(err).toBeInstanceOf(PredictionError);
    expect(err.message).toBe("Age at diagnosis must be between 15 and 110");
  });
});
