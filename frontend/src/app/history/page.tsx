"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ApiError, analysis, getToken, studies } from "@/lib/api";
import type { AnalysisSummary, Study } from "@/lib/types";
import { Sidebar } from "@/components/Sidebar";
import { TrendChart } from "@/components/TrendChart";

const STATUS_LABELS: Record<Study["status"], string> = {
  uploaded: "Загружено",
  processing: "Анализ…",
  completed: "Готово",
  failed: "Ошибка",
};

function shortDate(iso: string): string {
  return new Date(iso).toLocaleDateString("ru-RU", { day: "2-digit", month: "2-digit" });
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
        <div className="container">
          <h1>История</h1>
          {error && <div className="error">{error}</div>}

          {!loading && history.length >= 2 && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              <div className="card">
                <p className="card-title">Динамика: качество снимков</p>
                <p style={{ fontSize: 22, fontWeight: 700, margin: "0 0 8px" }}>
                  {(qualityPoints[qualityPoints.length - 1].value * 100).toFixed(0)}%
                </p>
                <TrendChart points={qualityPoints} color="var(--accent)" />
              </div>
              <div className="card">
                <p className="card-title">Динамика: уверенность диагностики</p>
                <p style={{ fontSize: 22, fontWeight: 700, margin: "0 0 8px" }}>
                  {confidencePoints.length > 0
                    ? `${(confidencePoints[confidencePoints.length - 1].value * 100).toFixed(0)}%`
                    : "—"}
                </p>
                <TrendChart points={confidencePoints} color="var(--success)" />
              </div>
            </div>
          )}

          <div className="card">
            {loading && (
              <div className="loading-row">
                <span className="spinner" /> Загрузка…
              </div>
            )}
            {!loading && items.length === 0 && (
              <p style={{ color: "var(--text-muted)" }}>
                Исследований пока нет — загрузите первый снимок на странице «Новый ОКТ».
              </p>
            )}
            {items.map((s) => (
              <div key={s.id} className="list-item">
                <div>
                  <div>
                    {s.eye ?? "—"} · {new Date(s.created_at).toLocaleString("ru-RU")}
                  </div>
                  <span className={`badge ${s.status}`}>{STATUS_LABELS[s.status]}</span>
                </div>
                <Link href={`/studies/${s.id}`}>Открыть →</Link>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
