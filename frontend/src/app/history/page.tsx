"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ApiError, analysis, fetchImageObjectUrl, getToken, studies } from "@/lib/api";
import type { AnalysisSummary, Study } from "@/lib/types";
import { Sidebar } from "@/components/Sidebar";
import { TrendChart } from "@/components/TrendChart";
import { AlertZoneIcon, FolderClockIcon, GaugeIcon } from "@/components/icons";

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

export default function HistoryPage() {
  const router = useRouter();
  const [items, setItems] = useState<Study[]>([]);
  const [history, setHistory] = useState<AnalysisSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!getToken()) {
      router.replace("/login");
      return;
    }
    Promise.all([studies.list(), analysis.history()])
      .then(([studyList, historyList]) => {
        setItems(studyList);
        setHistory(historyList);
      })
      .catch((err) => setError(err instanceof ApiError ? err.message : "Не удалось загрузить историю"))
      .finally(() => setLoading(false));
  }, [router]);

  const qualityPoints = history.map((h) => ({ date: shortDate(h.created_at), value: h.quality_score }));
  const confidencePoints = history
    .filter((h) => h.top_diagnosis)
    .map((h) => ({ date: shortDate(h.created_at), value: h.top_diagnosis!.probability }));

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

          {!loading && history.length >= 2 && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
              <div className="card" style={{ marginBottom: 0 }}>
                <p className="card-title">
                  <span className="card-title-icon">
                    <GaugeIcon />
                  </span>
                  Качество снимков
                </p>
                <p className="mono" style={{ fontSize: 28, fontWeight: 800, margin: "4px 0 12px" }}>
                  {(qualityPoints[qualityPoints.length - 1].value * 100).toFixed(0)}
                  <span style={{ fontSize: 16, color: "var(--ink-soft)" }}>%</span>
                </p>
                <TrendChart points={qualityPoints} color="var(--accent)" />
              </div>
              <div className="card" style={{ marginBottom: 0 }}>
                <p className="card-title">
                  <span className="card-title-icon">
                    <AlertZoneIcon />
                  </span>
                  Уверенность диагностики
                </p>
                <p className="mono" style={{ fontSize: 28, fontWeight: 800, margin: "4px 0 12px" }}>
                  {confidencePoints.length > 0
                    ? confidencePoints[confidencePoints.length - 1].value * 100 < 100
                      ? (confidencePoints[confidencePoints.length - 1].value * 100).toFixed(1)
                      : "100"
                    : "—"}
                  {confidencePoints.length > 0 && <span style={{ fontSize: 16, color: "var(--ink-soft)" }}>%</span>}
                </p>
                <TrendChart points={confidencePoints} color="var(--success)" />
              </div>
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
