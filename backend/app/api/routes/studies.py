import io
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from app.api.deps import get_current_patient
from app.core.config import settings
from app.db.session import get_db
from app.models.patient import Patient
from app.models.study import Study
from app.schemas.study import StudyRead

router = APIRouter(prefix="/studies", tags=["studies"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/tiff"}

# Maps a PIL-verified image format to the extension used for on-disk storage
# -- deliberately not derived from the client-supplied filename or
# Content-Type header, both trivially spoofable. Before this validation, an
# authenticated user could upload arbitrary bytes (a script, an executable)
# with a spoofed image Content-Type and an arbitrary filename extension
# (e.g. "shell.php"), and the server stored + later served them back
# verbatim: `Image.open(io.BytesIO(contents))` never ran until the ML
# pipeline itself, so a corrupt/non-image upload only failed much later at
# analysis time, not at upload time, and the stored extension controlled
# what Content-Type FileResponse would later serve it back with. Confirmed
# by uploading non-image bytes as "shell.php" with Content-Type: image/jpeg
# -- accepted with 201, stored on disk with a .php extension, before this
# fix. Validating the actual decoded format closes both the spoofable-header
# gap and the attacker-controlled-extension gap in one place.
_EXTENSION_BY_FORMAT = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tiff"}


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

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    contents = await file.read(max_bytes + 1)
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Файл слишком большой. Максимальный размер: {settings.max_upload_size_mb} МБ",
        )

    try:
        with Image.open(io.BytesIO(contents)) as img:
            image_format = img.format
            img.verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Файл повреждён или не является изображением",
        )
    except Image.DecompressionBombError:
        # Pillow raises this from Image.open() itself (before img.verify()) when the
        # declared pixel count exceeds 2x Image.MAX_IMAGE_PIXELS -- independent of the
        # file's byte size, so the 20MB cap above doesn't guard against it. It's a plain
        # Exception subclass, not OSError, so it falls through the except above uncaught
        # and used to surface as an unhandled 500 instead of a clean validation error
        # (reproduced with a ~246KB flat 15000x15000 PNG). Same 415 as any other
        # unusable upload -- an oversized-pixel-dimension image is as unusable to the
        # ML pipeline as a corrupt one.
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Файл повреждён или не является изображением",
        )
    if image_format not in _EXTENSION_BY_FORMAT:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Поддерживаются только изображения JPEG, PNG или TIFF",
        )

    storage_dir = Path(settings.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4()}{_EXTENSION_BY_FORMAT[image_format]}"
    destination = storage_dir / filename
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
