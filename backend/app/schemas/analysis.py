from datetime import datetime

from pydantic import BaseModel


class Diagnosis(BaseModel):
    code: str
    label: str
    probability: float


class PathologyFinding(BaseModel):
    key: str
    label_ru: str
    color: tuple[int, int, int]
    detected: bool
    zone_count: int


class AnalysisResultRead(BaseModel):
    id: str
    study_id: str
    quality_score: float
    quality_issues: list[str]
    segmentation_map_path: str | None
    layer_thickness: dict[str, float]
    pathology_map_path: str | None
    pathology_findings: list[PathologyFinding]
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
    # From the joined Study, not AnalysisResult itself -- lets clients split
    # trend charts by eye (OD/OS) instead of mixing two different eyes'
    # measurements into one line.
    eye: str | None

    model_config = {"from_attributes": True}
