"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ApiError, analysis, fetchImageObjectUrl, getToken, studies } from "@/lib/api";
import type { AnalysisSummary, Study } from "@/lib/types";
import { Sidebar } from "@/components/Sidebar";
import { TrendChart } from "@/components/TrendChart";
import { AlertZoneIcon, RulerIcon } from "@/components/icons";
import { LAYER_LABELS_RU } from "@/lib/labels";

const STATUS_LABELS: Record<Study["status"], string> = {
  uploaded: "Загружено",
  processing: "Анализ…",
  completed: "Готово",
  failed: "Ошибка",
};

function shortDate(iso: string): string {
  return new Date(iso).toLocaleDateString("ru-RU", { day: "2-digit", month: "2-digit" });
}

function HistoryThumb({ studyId }: { studyId: string }) {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    let objectUrl: string | null = null;
    let cancelled = false;
    fetchImageObjectUrl(`/api/v1/studies/${studyId}/image`)
      .then((u) => {
        if (cancelled) {
          URL.revokeObjectURL(u);
          return;
        }
        objectUrl = u;
        setUrl(u);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [studyId]);

  return (
    <div className="history-thumb">
      {url && (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={url} alt="" />
      )}
    </div>
  );
}

const EYE_ORDER: Record<string, number> = { OD: 0, OS: 1 };

export default function HistoryPage() {
  const router = useRouter();
  const [items, setItems] = useState<Study[]>([]);
  const [history, setHistory] = useState<AnalysisSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedEye, setSelectedEye] = useState<string | null>(null);

  useEffect(() => {
    if (!getToken()) {
      router.replace("/login");
      return;
    }
    Promise.all([studies.list(), analysis.history()])
      .then(([studyList, historyList]) => {
        setItems(studyList);
        setHistory(historyList);
        // Default to the most recent study's eye so trends open on "how is my
        // latest scan trending" rather than a merged line across both eyes.
        setSelectedEye(historyList[historyList.length - 1]?.eye ?? null);
      })
      .catch((err) => setError(err instanceof ApiError ? err.message : "Не удалось загрузить историю"))
      .finally(() => setLoading(false));
  }, [router]);

  // Trend charts plot one value per study against a shared x-axis by index, not
  // by eye -- OD and OS are two different biological measurements, so a history
  // mixing both would show swings that are really just alternating eyes, not
  // real progression. Only offer the OD/OS split once a patient actually has
  // both on file; single-eye patients (the common case) see the trends
  // unfiltered, exactly as before this feature existed.
  //
  // "Only one distinct eye" must count null (untagged upload, no eye
  // specified) as its own bucket alongside any real eye code, not silently
  // drop it -- otherwise a patient with e.g. some OD-tagged studies and some
  // untagged ones (which could easily be the other eye) hits availableEyes.length
  // === 1 and falls into the unfiltered branch, mixing the same "alternating
  // eyes" problem this filter exists to prevent.
  const distinctEyeBuckets = new Set(history.map((h) => h.eye));
  const availableEyes = Array.from(distinctEyeBuckets)
    .filter((e): e is string => e !== null)
    .sort((a, b) => (EYE_ORDER[a] ?? 99) - (EYE_ORDER[b] ?? 99));
  // Untagged uploads (eye === null) are their own bucket in the toggle too --
  // shown last, after the real eye codes -- so they stay selectable instead of
  // being invisibly absorbed into whichever real eye happens to be selected.
  const eyeToggleOptions: (string | null)[] = distinctEyeBuckets.has(null) ? [...availableEyes, null] : availableEyes;
  const showEyeSplit = distinctEyeBuckets.size > 1;
  const trendHistory = showEyeSplit ? history.filter((h) => h.eye === selectedEye) : history;

  // Each layer gets its own min/max-normalized 0..1 series -- raw thickness values
  // live on very different absolute scales per layer (a few microns vs. over a
  // hundred), so a shared scale would flatten the thinner layers' trends to
  // invisible. A flat history (min === max, including a single data point) maps
  // to a flat mid-line rather than dividing by zero.
  //
  // A low-quality scan makes the pipeline skip segmentation entirely (empty
  // layer_thickness, see run_analysis_pipeline) -- if that happens to be a
  // patient's most recent study, "latest" must not silently fall back to an
  // older study's value with no indication it isn't current.
  const mostRecentStudy = trendHistory[trendHistory.length - 1];
  const layerTrends = Object.keys(LAYER_LABELS_RU).flatMap((layerKey) => {
    const raw = trendHistory
      .filter((h) => layerKey in h.layer_thickness)
      .map((h) => ({ date: shortDate(h.created_at), value: h.layer_thickness[layerKey] }));
    if (raw.length === 0) return [];
    const values = raw.map((p) => p.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;
    const points = raw.map((p) => ({ date: p.date, value: range > 0 ? (p.value - min) / range : 0.5 }));
    const latest = mostRecentStudy && layerKey in mostRecentStudy.layer_thickness ? mostRecentStudy.layer_thickness[layerKey] : null;
    return [{ layerKey, label: LAYER_LABELS_RU[layerKey], latest, points }];
  });

  return (
    <div className="app-shell">
      <Sidebar />
      <div className="main-content">
        <div className="container" style={{ maxWidth: 1080 }}>
          <h1 className="page-heading">История исследований</h1>
          <p className="page-subtitle">Динамика по всем ОКТ-снимкам пациента.</p>
          {error && (
            <div className="error">
              <AlertZoneIcon />
              {error}
            </div>
          )}

          {!loading && showEyeSplit && (
            <div className="eye-toggle" style={{ maxWidth: 280 }}>
              {eyeToggleOptions.map((eyeOption) => (
                <button
                  key={eyeOption ?? "unspecified"}
                  type="button"
                  className={`eye-toggle-option ${selectedEye === eyeOption ? "active" : ""}`}
                  onClick={() => setSelectedEye(eyeOption)}
                >
                  {eyeOption === "OD"
                    ? "OD · правый"
                    : eyeOption === "OS"
                      ? "OS · левый"
                      : eyeOption === null
                        ? "Без указания глаза"
                        : eyeOption}
                </button>
              ))}
            </div>
          )}

          {!loading && showEyeSplit && trendHistory.length < 2 && (
            <p style={{ color: "var(--ink-soft)", fontSize: 13, marginBottom: 20 }}>
              Недостаточно снимков этого глаза для графика динамики — нужно хотя бы два.
            </p>
          )}

          {!loading && trendHistory.length >= 2 && layerTrends.length > 0 && (
            <div className="card">
              <p className="card-title">
                <span className="card-title-icon">
                  <RulerIcon />
                </span>
                Динамика толщины слоёв сетчатки (мкм)
              </p>
              {layerTrends.map((t) => (
                <div key={t.layerKey} className="bar-row">
                  <div className="bar-row-labels">
                    <span>{t.label}</span>
                    <span>{t.latest !== null ? t.latest : "—"}</span>
                  </div>
                  <TrendChart points={t.points} color="var(--accent)" height={32} showGrid={false} />
                </div>
              ))}
            </div>
          )}

          <div className="history-table">
            <div className="history-table-head">
              <div />
              <div>Снимок</div>
              <div>Дата / время</div>
              <div>Качество</div>
              <div>Статус</div>
              <div />
            </div>
            {loading && (
              <div className="loading-row" style={{ padding: "16px 20px" }}>
                <span className="spinner" /> Загрузка…
              </div>
            )}
            {!loading && items.length === 0 && (
              <p style={{ color: "var(--ink-soft)", padding: "16px 20px" }}>
                Исследований пока нет — загрузите первый снимок на странице «Новый ОКТ».
              </p>
            )}
            {items.map((s) => {
              const summary = history.find((h) => h.study_id === s.id);
              return (
                <div key={s.id} className="history-table-row">
                  <HistoryThumb studyId={s.id} />
                  <div style={{ fontSize: 13.5, fontWeight: 700 }}>{s.eye ?? "—"}</div>
                  <div className="mono" style={{ fontSize: 13, color: "var(--ink-soft)" }}>
                    {new Date(s.created_at).toLocaleString("ru-RU")}
                  </div>
                  <div className="mono" style={{ fontSize: 13, fontWeight: 600 }}>
                    {summary ? `${(summary.quality_score * 100).toFixed(0)}%` : "—"}
                  </div>
                  <div>
                    <span className={`badge ${s.status}`}>{STATUS_LABELS[s.status]}</span>
                  </div>
                  <Link href={`/studies/${s.id}`} className="history-open-link">
                    Открыть →
                  </Link>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
