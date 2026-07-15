"""Medical report generation module.

Template-based stub that turns structured findings into Russian free text.
Swap `generate_report` with a call to an LLM (e.g. a locally hosted model via
Ollama) that is fed the same structured findings — never let the model
introduce findings that are not present in the input.
"""

from app.services.diagnosis import Diagnosis

LOW_CONFIDENCE_THRESHOLD = 0.5

# High-recall operating point from ML round 5's precision/recall analysis
# (app/ml/artifacts/metrics.json -> dme_threshold_analysis.high_recall_operating_point),
# chosen deliberately over the raw 50% argmax cutoff: on the held-out real-patient
# test set it catches 91% of DME cases instead of 67% (misses 3/33 instead of 11/33),
# at the cost of more false alarms (precision 0.62 vs 0.92). For a screening
# assistant where a doctor reviews every flagged case, a missed edema is worse than
# an extra review, so DME is flagged as soon as its probability crosses this bar --
# even when NORMAL is still the nominally more likely class. Revisit this number if
# app/ml/train.py is rerun and the model card's high-recall cutoff moves.
DME_SCREENING_THRESHOLD = 0.018


def generate_report(
    quality_score: float,
    quality_issues: list[str],
    layer_thickness_um: dict[str, float],
    diagnoses: list[Diagnosis],
    pathology_zone_count: int = 0,
) -> str:
    lines = ["Автоматическое заключение по результатам ОКТ-исследования (предварительное, требует подтверждения врачом).", ""]

    lines.append(f"Качество изображения: {quality_score * 100:.0f}%.")
    if quality_issues:
        lines.append("Обнаруженные проблемы качества: " + ", ".join(quality_issues) + ".")
    else:
        lines.append("Значимых проблем качества не выявлено.")
    lines.append("")

    lines.append("Толщина слоев сетчатки (мкм):")
    for layer, value in layer_thickness_um.items():
        lines.append(f"  - {layer}: {value}")
    lines.append("")

    if pathology_zone_count > 0:
        lines.append(
            f"Алгоритм выделил {pathology_zone_count} гипорефлективных (тёмных) зон на снимке — "
            "это не диагноз, а автоматически найденные участки, требующие визуальной проверки врачом "
            "(могут быть жидкостью, кистой, отслойкой или артефактом снимка)."
        )
        lines.append("")

    top = diagnoses[0] if diagnoses else None
    dme = next((d for d in diagnoses if d.code == "DME"), None)
    dme_flagged = dme is not None and dme.probability >= DME_SCREENING_THRESHOLD

    lines.append("Вероятные диагнозы:")
    for d in diagnoses[:3]:
        lines.append(f"  - {d.label}: {d.probability * 100:.1f}%")
    lines.append("")

    if dme_flagged and top is dme:
        lines.append(
            f"Наиболее вероятная находка: {top.label} (уверенность модели {top.probability * 100:.1f}%). "
            "Рекомендуется очная консультация офтальмолога для подтверждения диагноза."
        )
    elif dme_flagged:
        lines.append(
            f"Вероятность диабетического макулярного отёка по модели ({dme.probability * 100:.1f}%) превышает "
            f"чувствительный скрининговый порог ({DME_SCREENING_THRESHOLD * 100:.1f}%, а не 50%) — порог намеренно "
            "снижен, чтобы реже пропускать отёк ценой больших ложных срабатываний. Рекомендуется очная консультация "
            "офтальмолога для исключения диагноза."
        )
    else:
        lines.append("Признаков патологии с высокой вероятностью не обнаружено. Рекомендовано плановое наблюдение.")

    flagged = dme if dme_flagged else top
    if flagged and flagged.probability < LOW_CONFIDENCE_THRESHOLD:
        lines.append("")
        lines.append(
            "Внимание: уверенность модели в этой находке низкая "
            f"({flagged.probability * 100:.1f}%). Результат недостаточно надежен для самостоятельных "
            "клинических выводов — необходим приоритетный ручной просмотр врачом."
        )

    return "\n".join(lines)
