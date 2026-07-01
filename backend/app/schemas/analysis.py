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
    diagnoses: list[Diagnosis]
    report_text: str
    created_at: datetime

    model_config = {"from_attributes": True}
