from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import get_current_patient
from app.db.session import get_db
from app.models.analysis import AnalysisResult
from app.models.patient import Patient
from app.models.study import Study
from app.schemas.analysis import AnalysisResultRead
from app.services.pipeline import run_analysis_pipeline

router = APIRouter(prefix="/analysis", tags=["analysis"])


def _get_own_study(study_id: str, patient: Patient, db: Session) -> Study:
    study = db.get(Study, study_id)
    if not study or study.patient_id != patient.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Исследование не найдено")
    return study


@router.post("/{study_id}/run", response_model=AnalysisResultRead, status_code=status.HTTP_201_CREATED)
def run_analysis(
    study_id: str,
    db: Session = Depends(get_db),
    patient: Patient = Depends(get_current_patient),
) -> AnalysisResult:
    study = _get_own_study(study_id, patient, db)

    study.status = "processing"
    db.commit()

    try:
        pipeline_output = run_analysis_pipeline(study.image_path)
    except Exception as exc:  # noqa: BLE001
        study.status = "failed"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при анализе исследования",
        ) from exc

    result = AnalysisResult(study_id=study.id, **pipeline_output)
    db.add(result)
    study.status = "completed"
    db.commit()
    db.refresh(result)
    return result


@router.get("/{study_id}", response_model=AnalysisResultRead)
def get_analysis_result(
    study_id: str,
    db: Session = Depends(get_db),
    patient: Patient = Depends(get_current_patient),
) -> AnalysisResult:
    study = _get_own_study(study_id, patient, db)
    result = db.query(AnalysisResult).filter(AnalysisResult.study_id == study.id).first()
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Результат анализа не найден")
    return result


@router.get("/{study_id}/segmentation-map")
def get_segmentation_map(
    study_id: str,
    db: Session = Depends(get_db),
    patient: Patient = Depends(get_current_patient),
) -> FileResponse:
    study = _get_own_study(study_id, patient, db)
    result = db.query(AnalysisResult).filter(AnalysisResult.study_id == study.id).first()
    if not result or not result.segmentation_map_path or not Path(result.segmentation_map_path).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Карта сегментации не найдена")
    return FileResponse(result.segmentation_map_path)
