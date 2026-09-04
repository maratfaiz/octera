from app.services.diagnosis import Diagnosis
from app.services.report_generator import generate_report


def _diagnoses(dme_probability: float) -> list[Diagnosis]:
    return [
        Diagnosis(code="DME", label="Диабетический макулярный отек", probability=dme_probability),
        Diagnosis(code="NORMAL", label="Без признаков патологии", probability=1 - dme_probability),
    ]


def test_high_confidence_pathological_finding_has_no_low_confidence_warning():
    report = generate_report(
        quality_score=0.9,
        quality_issues=[],
        layer_thickness_um={"rpe": 40.0},
        diagnoses=_diagnoses(dme_probability=0.85),
    )

    assert "Диабетический макулярный отек" in report
    assert "низкая" not in report


def test_low_confidence_finding_adds_warning():
    report = generate_report(
        quality_score=0.9,
        quality_issues=[],
        layer_thickness_um={"rpe": 40.0},
        diagnoses=_diagnoses(dme_probability=0.45),
    )

    assert "уверенность модели в этой находке низкая" in report
    assert "ручной просмотр" in report


def test_normal_case_recommends_routine_followup():
    diagnoses = [
        Diagnosis(code="NORMAL", label="Без признаков патологии", probability=0.995),
        Diagnosis(code="DME", label="Диабетический макулярный отек", probability=0.005),
    ]
    report = generate_report(
        quality_score=0.9, quality_issues=[], layer_thickness_um={"rpe": 40.0}, diagnoses=diagnoses
    )

    assert "Признаков патологии с высокой вероятностью не обнаружено" in report


def test_dme_below_argmax_but_above_screening_threshold_is_flagged():
    diagnoses = [
        Diagnosis(code="NORMAL", label="Без признаков патологии", probability=0.6),
        Diagnosis(code="DME", label="Диабетический макулярный отек", probability=0.4),
    ]
    report = generate_report(
        quality_score=0.9, quality_issues=[], layer_thickness_um={"rpe": 40.0}, diagnoses=diagnoses
    )

    assert "скрининговый порог" in report
    assert "офтальмолога" in report
    assert "уверенность модели в этой находке низкая" in report
