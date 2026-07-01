from datetime import date, datetime

from pydantic import BaseModel


class PatientCreate(BaseModel):
    full_name: str
    birth_date: date | None = None
    sex: str | None = None
    mrn: str | None = None


class PatientRead(BaseModel):
    id: str
    full_name: str
    birth_date: date | None
    sex: str | None
    mrn: str | None
    created_at: datetime

    model_config = {"from_attributes": True}
