from fastapi import APIRouter, Depends

from app.api.deps import get_current_patient
from app.models.patient import Patient
from app.schemas.patient import PatientRead

router = APIRouter(prefix="/patients", tags=["patients"])


@router.get("/me", response_model=PatientRead)
def get_my_patient(patient: Patient = Depends(get_current_patient)) -> Patient:
    return patient
