"""Medical report generation module.

Template-based stub that turns structured findings into Russian free text.
Swap `generate_report` with a call to an LLM (e.g. a locally hosted model via
Ollama) that is fed the same structured findings — never let the model
introduce findings that are not present in the input.
"""

from app.services.diagnosis import Diagnosis

LOW_CONFIDENCE_THRESHOLD = 0.5

# High-recall operating point from ML round 5's precision/recall analysis, chosen
# deliberately over the raw 50% argmax cutoff: for a screening assistant where a
# doctor reviews every flagged case, a missed edema is worse than an extra review,
# so DME is flagged as soon as its probability crosses this bar -- even when NORMAL
# is still the nominally more likely class.
#
# Round 5 originally shipped 0.018, chosen by running the precision/recall curve
# directly against the held-out test set -- which is test-set leakage (the
# threshold that looks best on those specific 223 images isn't necessarily one
# that generalizes). Round 6 fixed the selection method (out-of-fold CV on the
# training split only, see app/ml/train.py's _cv_threshold_analysis) and got
# 0.042 (test recall 0.79 / precision 0.68). Round 10 found a second, bigger
# leak upstream of the threshold entirely: ~25% of dataset images share a
# patient with another image (both eyes / repeat visits), and the plain
# stratified train/test split let 79 of 831 patients leak across both splits.
# A patient-grouped split (_group_aware_split) fixed that and produced more
# modest, but still real, honest numbers: at 0.037, test recall is 0.88 with
# precision 0.57 (the un-thresholded 50% cutoff's honest recall dropped to
# 0.64 from the previously-reported, leaky 0.67).
#
# Round 11 swapped the shipped model itself: PCA(30)+SVM scored notably higher
# on patient-grouped CV (balanced accuracy 0.879 vs 0.817 for the previous
# MLP) and, at the plain 50% argmax cutoff, already gets test recall 0.88 /
# precision 0.71 -- no threshold trick needed to reach round 10's recall
# target. The recall>=0.85-on-CV cutoff for *this* model is 0.260 (test
# recall 0.88 / precision 0.66, about the same operating point argmax already
# gives, just confirmed by the same CV-selection method as before for
# consistency). See app/ml/artifacts/metrics.json and the README for the full
# history across rounds 5/6/10/11.
#
# Round 14 found a third leak: 12 pairs of byte-identical images filed under
# two different nominal patient IDs (likely a duplicate-entry artifact in the
# source dataset), 2 of which still crossed train/test under round 10's
# patient-only grouping. Merging those into shared groups changed the
# train/test split; recall>=0.85-on-CV for *this* split is 0.350 (test recall
# 0.84 / precision 0.50 -- this specific reproducible split happens to be a
# harder one for precision specifically: other split seeds tried in the README
# give precision 0.61-0.72 at similar recall, so treat 0.50 as a real but
# somewhat pessimistic draw rather than a genuine regression from round 11).
# Revisit this number if app/ml/train.py is rerun and the model card's
# high-recall cutoff moves.
DME_SCREENING_THRESHOLD = 0.350


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
