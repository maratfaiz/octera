from datetime import datetime

from pydantic import BaseModel


class Diagnosis(BaseModel):
    code: str
    label: str
    probability: float


class AnalysisResultRead(BaseModel):
    id: str
    study_id: str
    quality_score: float
    quality_issues: list[str]
    segmentation_map_path: str | None
    layer_thickness: dict[str, float]
    pathology_map_path: str | None
    pathology_zone_count: int
    diagnoses: list[Diagnosis]
    report_text: str
    created_at: datetime

    model_config = {"from_attributes": True}


class AnalysisSummary(BaseModel):
    study_id: str
    created_at: datetime
    quality_score: float
    top_diagnosis: Diagnosis | None
    layer_thickness: dict[str, float]

    model_config = {"from_attributes": True}
