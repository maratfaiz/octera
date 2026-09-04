from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import get_current_patient
from app.db.session import get_db
from app.models.analysis import AnalysisResult
from app.models.patient import Patient
from app.models.study import Study
from app.schemas.analysis import AnalysisResultRead, AnalysisSummary
from app.services.pipeline import run_analysis_pipeline
from app.services.report_generator import resolve_flagged_diagnosis

router = APIRouter(prefix="/analysis", tags=["analysis"])


def _get_own_study(study_id: str, patient: Patient, db: Session) -> Study:
    study = db.get(Study, study_id)
    if not study or study.patient_id != patient.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Исследование не найдено")
    return study


@router.get("", response_model=list[AnalysisSummary])
def list_analysis_history(
    db: Session = Depends(get_db),
    patient: Patient = Depends(get_current_patient),
) -> list[AnalysisSummary]:
    results = (
        db.query(AnalysisResult, Study.eye)
        .join(Study, Study.id == AnalysisResult.study_id)
        .filter(Study.patient_id == patient.id)
        .order_by(AnalysisResult.created_at.asc())
        .all()
    )
    return [
        AnalysisSummary(
            study_id=r.study_id,
            created_at=r.created_at,
            quality_score=r.quality_score,
            # Matches generate_report's DME-screening logic (see report_generator.py)
            # so a study flagged in its detail report isn't shown as unremarkable here.
            top_diagnosis=resolve_flagged_diagnosis(r.diagnoses),
            layer_thickness=r.layer_thickness,
            eye=eye,
        )
        for r, eye in results
    ]


@router.post("/{study_id}/run", response_model=AnalysisResultRead, status_code=status.HTTP_201_CREATED)
def run_analysis(
    study_id: str,
    db: Session = Depends(get_db),
    patient: Patient = Depends(get_current_patient),
) -> AnalysisResult:
    study = _get_own_study(study_id, patient, db)

    # AnalysisResult.study_id is unique -- without this check, a double-submit
    # (double-click, browser retry, a stray duplicate request) would run the
    # whole pipeline a second time and then fail at the DB insert with an
    # unhandled IntegrityError, leaving `study.status` stuck at "processing"
    # forever (the study page just shows an indefinite "waiting" state, no
    # retry or timeout) instead of a clean, actionable error.
    if db.query(AnalysisResult).filter(AnalysisResult.study_id == study.id).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Анализ уже выполнен для этого исследования",
        )

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
    try:
        study.status = "completed"
        db.commit()
    except IntegrityError:
        # Narrows the window the check above can't close on its own: two
        # truly concurrent requests can both pass the check before either
        # commits. Same clean 409 instead of an unhandled 500 + a study
        # stuck in "processing".
        db.rollback()
        study.status = "failed"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Анализ уже выполнен для этого исследования",
        )
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


@router.get("/{study_id}/pathology-map")
def get_pathology_map(
    study_id: str,
    db: Session = Depends(get_db),
    patient: Patient = Depends(get_current_patient),
) -> FileResponse:
    study = _get_own_study(study_id, patient, db)
    result = db.query(AnalysisResult).filter(AnalysisResult.study_id == study.id).first()
    if not result or not result.pathology_map_path or not Path(result.pathology_map_path).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Карта патологий не найдена")
    return FileResponse(result.pathology_map_path)
