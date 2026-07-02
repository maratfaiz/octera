from app.services.diagnosis import Diagnosis
from app.services.report_generator import generate_report


def _diagnoses(top_probability: float) -> list[Diagnosis]:
    remainder = (1 - top_probability) / 3
    return [
        Diagnosis(code="DME", label="Диабетический макулярный отек", probability=top_probability),
        Diagnosis(code="CNV", label="Хориоидальная неоваскуляризация", probability=remainder),
        Diagnosis(code="DRUSEN", label="Друзы", probability=remainder),
        Diagnosis(code="NORMAL", label="Без признаков патологии", probability=remainder),
    ]


def test_high_confidence_pathological_finding_has_no_low_confidence_warning():
    report = generate_report(
        quality_score=0.9,
        quality_issues=[],
        layer_thickness_um={"rpe": 40.0},
        diagnoses=_diagnoses(top_probability=0.85),
    )

    assert "Диабетический макулярный отек" in report
    assert "низкая" not in report


def test_low_confidence_finding_adds_warning():
    report = generate_report(
        quality_score=0.9,
        quality_issues=[],
        layer_thickness_um={"rpe": 40.0},
        diagnoses=_diagnoses(top_probability=0.35),
    )

    assert "уверенность модели в наиболее вероятном диагнозе низкая" in report
    assert "ручной просмотр" in report


def test_normal_case_recommends_routine_followup():
    diagnoses = [
        Diagnosis(code="NORMAL", label="Без признаков патологии", probability=0.9),
        Diagnosis(code="CNV", label="Хориоидальная неоваскуляризация", probability=0.05),
        Diagnosis(code="DME", label="Диабетический макулярный отек", probability=0.03),
        Diagnosis(code="DRUSEN", label="Друзы", probability=0.02),
    ]
    report = generate_report(
        quality_score=0.9, quality_issues=[], layer_thickness_um={"rpe": 40.0}, diagnoses=diagnoses
    )

    assert "Признаков патологии с высокой вероятностью не обнаружено" in report
