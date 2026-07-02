"""Medical report generation module.

Template-based stub that turns structured findings into Russian free text.
Swap `generate_report` with a call to an LLM (e.g. a locally hosted model via
Ollama) that is fed the same structured findings — never let the model
introduce findings that are not present in the input.
"""

from app.services.diagnosis import Diagnosis

LOW_CONFIDENCE_THRESHOLD = 0.5


def generate_report(
    quality_score: float,
    quality_issues: list[str],
    layer_thickness_um: dict[str, float],
    diagnoses: list[Diagnosis],
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

    top = diagnoses[0] if diagnoses else None
    lines.append("Вероятные диагнозы:")
    for d in diagnoses[:3]:
        lines.append(f"  - {d.label}: {d.probability * 100:.1f}%")
    lines.append("")

    if top and top.code != "NORMAL":
        lines.append(
            f"Наиболее вероятная находка: {top.label} (уверенность модели {top.probability * 100:.1f}%). "
            "Рекомендуется очная консультация офтальмолога для подтверждения диагноза."
        )
    else:
        lines.append("Признаков патологии с высокой вероятностью не обнаружено. Рекомендовано плановое наблюдение.")

    if top and top.probability < LOW_CONFIDENCE_THRESHOLD:
        lines.append("")
        lines.append(
            "Внимание: уверенность модели в наиболее вероятном диагнозе низкая "
            f"({top.probability * 100:.1f}%). Результат недостаточно надежен для самостоятельных "
            "клинических выводов — необходим приоритетный ручной просмотр врачом."
        )

    return "\n".join(lines)
