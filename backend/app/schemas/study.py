from datetime import datetime

from pydantic import BaseModel


class StudyRead(BaseModel):
    id: str
    patient_id: str
    image_path: str
    eye: str | None
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}
