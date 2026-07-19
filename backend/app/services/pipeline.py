"""Orchestrates the full OCT analysis pipeline.

quality_check -> segmentation -> diagnosis -> report_generator

If image quality is too low, segmentation/diagnosis are skipped and the
pipeline returns early with the quality issues only.
"""

import dataclasses
from typing import Any

from app.services.diagnosis import predict_diagnoses
from app.services.quality_check import check_quality
from app.services.report_generator import generate_report
from app.services.segmentation import segment_layers

MIN_ACCEPTABLE_QUALITY = 0.5


def run_analysis_pipeline(image_path: str) -> dict[str, Any]:
    quality = check_quality(image_path)

    if quality.score < MIN_ACCEPTABLE_QUALITY:
        return {
            "quality_score": quality.score,
            "quality_issues": quality.issues,
            "segmentation_map_path": None,
            "layer_thickness": {},
            "pathology_map_path": None,
            "pathology_findings": [],
            "diagnoses": [],
            "report_text": (
                "Качество снимка недостаточно для автоматического анализа. "
                "Рекомендуется повторить исследование. Проблемы: " + ", ".join(quality.issues)
            ),
        }

    segmentation = segment_layers(image_path)
    diagnoses = predict_diagnoses(image_path)
    pathology_findings = [dataclasses.asdict(f) for f in segmentation.pathology_findings]
    report_text = generate_report(
        quality_score=quality.score,
        quality_issues=quality.issues,
        layer_thickness_um=segmentation.layer_thickness_um,
        pathology_findings=pathology_findings,
        diagnoses=diagnoses,
    )

    return {
        "quality_score": quality.score,
        "quality_issues": quality.issues,
        "segmentation_map_path": segmentation.map_path,
        "layer_thickness": segmentation.layer_thickness_um,
        "pathology_map_path": segmentation.pathology_map_path,
        "pathology_findings": pathology_findings,
        "diagnoses": [d.__dict__ for d in diagnoses],
        "report_text": report_text,
    }
