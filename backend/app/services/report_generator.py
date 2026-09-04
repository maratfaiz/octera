"""Medical report generation module.

Template-based stub that turns structured findings into Russian free text.
Swap `generate_report` with a call to an LLM (e.g. a locally hosted model via
Ollama) that is fed the same structured findings — never let the model
introduce findings that are not present in the input.
"""

import json
from typing import Any

from app.ml.model import DEFAULT_CHECKPOINT_PATH
from app.services.diagnosis import Diagnosis
from app.services.segmentation import LAYER_LABELS_RU

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
# This used to be a hand-copied literal that a human had to remember to update
# every time app/ml/train.py was rerun -- it had already drifted from the
# shipped checkpoint's actual metrics.json (0.350 vs. the real 0.3497...) by
# the time this was caught in review. It's now loaded from metrics.json
# (written alongside the checkpoint by train.py) at import time, falling back
# to this historical value only if metrics.json is missing or malformed (e.g.
# a checkpoint-less dev environment).
_FALLBACK_DME_SCREENING_THRESHOLD = 0.350


def _load_dme_screening_threshold() -> float:
    metrics_path = DEFAULT_CHECKPOINT_PATH.parent / "metrics.json"
    try:
        metrics: dict[str, Any] = json.loads(metrics_path.read_text())
        threshold = metrics["dme_threshold_analysis"]["high_recall_operating_point"]["threshold"]
        return float(threshold)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return _FALLBACK_DME_SCREENING_THRESHOLD


DME_SCREENING_THRESHOLD = _load_dme_screening_threshold()


def _field(diagnosis: Any, name: str) -> Any:
    # diagnoses arrive as Diagnosis dataclass instances fresh from the model
    # (app.services.diagnosis) but as plain {code, label, probability} dicts
    # once round-tripped through AnalysisResult's JSON column -- accept both.
    return diagnosis[name] if isinstance(diagnosis, dict) else getattr(diagnosis, name)


def resolve_flagged_diagnosis(diagnoses: list) -> Any | None:
    """Returns whichever diagnosis is clinically operative for this study.

    That's DME if it clears the screening threshold -- even when NORMAL is
    still the nominally more likely class -- otherwise the plain top
    (highest-probability) diagnosis. Used both for the free-text report and
    for the analysis-history list, so the two views of the same study can't
    disagree about whether a case was flagged.
    """
    if not diagnoses:
        return None
    dme = next((d for d in diagnoses if _field(d, "code") == "DME"), None)
    if dme is not None and _field(dme, "probability") >= DME_SCREENING_THRESHOLD:
        return dme
    return diagnoses[0]


def generate_report(
    quality_score: float,
    quality_issues: list[str],
    layer_thickness_um: dict[str, float],
    diagnoses: list[Diagnosis],
    pathology_findings: list[dict[str, Any]] | None = None,
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
        lines.append(f"  - {LAYER_LABELS_RU.get(layer, layer)}: {value}")
    lines.append("")

    detected = [f for f in (pathology_findings or []) if f.get("detected")]
    if detected:
        lines.append("Алгоритм отметил возможные патологические зоны (не диагноз, требует проверки врачом):")
        for f in detected:
            lines.append(f"  - {f['label_ru']}: {f['zone_count']} зон(ы)")
        lines.append(
            "  Проверены только эти категории — метод пока не различает эпиретинальную мембрану "
            "и отслойку пигментного эпителия."
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

    flagged = resolve_flagged_diagnosis(diagnoses)
    if flagged and flagged.probability < LOW_CONFIDENCE_THRESHOLD:
        lines.append("")
        lines.append(
            "Внимание: уверенность модели в этой находке низкая "
            f"({flagged.probability * 100:.1f}%). Результат недостаточно надежен для самостоятельных "
            "клинических выводов — необходим приоритетный ручной просмотр врачом."
        )

    return "\n".join(lines)
