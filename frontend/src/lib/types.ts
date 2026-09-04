export interface User {
  id: string;
  email: string;
  full_name: string;
  role: string;
}

export interface Study {
  id: string;
  patient_id: string;
  image_path: string;
  eye: string | null;
  status: "uploaded" | "processing" | "completed" | "failed";
  created_at: string;
}

export interface Diagnosis {
  code: string;
  label: string;
  probability: number;
}

export interface PathologyFinding {
  key: string;
  label_ru: string;
  color: [number, number, number];
  detected: boolean;
  zone_count: number;
}

export interface AnalysisResult {
  id: string;
  study_id: string;
  quality_score: number;
  quality_issues: string[];
  segmentation_map_path: string | null;
  layer_thickness: Record<string, number>;
  pathology_map_path: string | null;
  pathology_findings: PathologyFinding[];
  diagnoses: Diagnosis[];
  report_text: string;
  created_at: string;
}

export interface AnalysisSummary {
  study_id: string;
  created_at: string;
  quality_score: number;
  top_diagnosis: Diagnosis | null;
  layer_thickness: Record<string, number>;
  eye: string | null;
}
