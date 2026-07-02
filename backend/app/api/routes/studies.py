import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_patient
from app.core.config import settings
from app.db.session import get_db
from app.models.patient import Patient
from app.models.study import Study
from app.schemas.study import StudyRead

router = APIRouter(prefix="/studies", tags=["studies"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/tiff"}


def _get_own_study(study_id: str, patient: Patient, db: Session) -> Study:
    study = db.get(Study, study_id)
    if not study or study.patient_id != patient.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Исследование не найдено")
    return study


@router.post("", response_model=StudyRead, status_code=status.HTTP_201_CREATED)
async def upload_study(
    file: UploadFile,
    eye: str | None = None,
    db: Session = Depends(get_db),
    patient: Patient = Depends(get_current_patient),
) -> Study:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Поддерживаются только изображения JPEG, PNG или TIFF",
        )

    storage_dir = Path(settings.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    extension = Path(file.filename or "").suffix or ".jpg"
    filename = f"{uuid.uuid4()}{extension}"
    destination = storage_dir / filename

    contents = await file.read()
    destination.write_bytes(contents)

    study = Study(patient_id=patient.id, image_path=str(destination), eye=eye)
    db.add(study)
    db.commit()
    db.refresh(study)
    return study


@router.get("", response_model=list[StudyRead])
def list_studies(
    db: Session = Depends(get_db),
    patient: Patient = Depends(get_current_patient),
) -> list[Study]:
    return db.query(Study).filter(Study.patient_id == patient.id).order_by(Study.created_at.desc()).all()


@router.get("/{study_id}", response_model=StudyRead)
def get_study(
    study_id: str,
    db: Session = Depends(get_db),
    patient: Patient = Depends(get_current_patient),
) -> Study:
    return _get_own_study(study_id, patient, db)


@router.get("/{study_id}/image")
def get_study_image(
    study_id: str,
    db: Session = Depends(get_db),
    patient: Patient = Depends(get_current_patient),
) -> FileResponse:
    study = _get_own_study(study_id, patient, db)
    if not Path(study.image_path).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Изображение не найдено")
    return FileResponse(study.image_path)
