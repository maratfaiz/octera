"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { ApiError, analysis, fetchImageObjectUrl, getToken, studies } from "@/lib/api";
import type { AnalysisResult, Study } from "@/lib/types";
import { Sidebar } from "@/components/Sidebar";
import {
  ActivityIcon,
  AlertZoneIcon,
  ArrowLeftIcon,
  GaugeIcon,
  ImageIcon,
  LayersMapIcon,
  ReportIcon,
  RulerIcon,
} from "@/components/icons";

// Keep in sync with backend/app/services/segmentation.py's LAYER_LABELS_RU --
// the raw codes are an internal shorthand, not something to show a doctor.
const LAYER_LABELS_RU: Record<string, string> = {
  nfl_gcl: "Слой нервных волокон / ганглиозных клеток (NFL/GCL)",
  ipl_inl: "Внутренний плексиформный / внутренний ядерный слой (IPL/INL)",
  opl_onl: "Наружный плексиформный / наружный ядерный слой (OPL/ONL)",
  photoreceptor: "Слой фоторецепторов",
  rpe: "Пигментный эпителий сетчатки (RPE)",
};

export default function StudyPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();

  const [study, setStudy] = useState<Study | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [segmentationUrl, setSegmentationUrl] = useState<string | null>(null);
  const [pathologyUrl, setPathologyUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!getToken()) {
      router.replace("/login");
      return;
    }
    load();
  }, [id, router]);

  async function load() {
    try {
      const s = await studies.get(id);
      setStudy(s);
      setImageUrl(await fetchImageObjectUrl(`/api/v1/studies/${id}/image`));

      if (s.status === "completed") {
        const r = await analysis.get(id);
        setResult(r);
        if (r.segmentation_map_path) {
          setSegmentationUrl(await fetchImageObjectUrl(`/api/v1/analysis/${id}/segmentation-map`));
        }
        if (r.pathology_map_path) {
          setPathologyUrl(await fetchImageObjectUrl(`/api/v1/analysis/${id}/pathology-map`));
        }
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось загрузить исследование");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <Sidebar />
      <div className="main-content">
        <div className="container">
          <Link href="/history" style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <ArrowLeftIcon /> История
          </Link>
          <h2 className="page-heading">Результаты исследования</h2>
          {error && (
            <div className="error">
              <AlertZoneIcon />
              {error}
            </div>
          )}

          {loading && (
            <div className="card loading-row">
              <span className="spinner" /> Загрузка исследования…
            </div>
          )}

          {!loading && <div className="results-grid">
            <div>
              <div className="image-panels">
                <div className="card">
                  <p className="card-title">
                    <span className="card-title-icon">
                      <ImageIcon />
                    </span>
                    Исходное изображение
                  </p>
                  {imageUrl ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={imageUrl} alt="ОКТ-снимок" style={{ width: "100%", borderRadius: 8 }} />
                  ) : (
                    <p style={{ color: "var(--text-muted)" }}>Загрузка…</p>
                  )}
                </div>
                <div className="card">
                  <p className="card-title">
                    <span className="card-title-icon">
                      <LayersMapIcon />
                    </span>
                    Карта внимания AI
                  </p>
                  {segmentationUrl ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={segmentationUrl} alt="Карта сегментации" style={{ width: "100%", borderRadius: 8 }} />
                  ) : (
                    <p style={{ color: "var(--text-muted)" }}>
                      {study?.status === "completed" ? "Недоступна" : "Ожидает завершения анализа"}
                    </p>
                  )}
                </div>
                <div className="card">
                  <p className="card-title">
                    <span className="card-title-icon">
                      <AlertZoneIcon />
                    </span>
                    Карта патологий
                  </p>
                  {pathologyUrl ? (
                    <>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={pathologyUrl} alt="Карта патологий" style={{ width: "100%", borderRadius: 8 }} />
                      <p style={{ color: "var(--text-muted)", fontSize: 12, marginTop: 8, marginBottom: 0 }}>
                        {result && result.pathology_zone_count > 0
                          ? `Алгоритм отметил ${result.pathology_zone_count} зон(ы) — требуют проверки врачом, это не диагноз.`
                          : "Выраженных гипорефлективных зон не найдено."}
                      </p>
                    </>
                  ) : (
                    <p style={{ color: "var(--text-muted)" }}>
                      {study?.status === "completed" ? "Недоступна" : "Ожидает завершения анализа"}
                    </p>
                  )}
                </div>
              </div>

              {result && (
                <>
                  <div className="card">
                    <p className="card-title">
                      <span className="card-title-icon">
                        <RulerIcon />
                      </span>
                      Толщина слоев сетчатки (мкм)
                    </p>
                    {Object.entries(result.layer_thickness).map(([layer, value]) => (
                      <div key={layer} className="list-item">
                        <span>{LAYER_LABELS_RU[layer] ?? layer}</span>
                        <span>{value}</span>
                      </div>
                    ))}
                  </div>

                  <div className="card">
                    <p className="card-title">
                      <span className="card-title-icon">
                        <ReportIcon />
                      </span>
                      AI-заключение
                    </p>
                    <p className="report-text">{result.report_text}</p>
                  </div>
                </>
              )}

              {study && study.status !== "completed" && !error && (
                <div className="card">
                  <p style={{ color: "var(--text-muted)" }}>
                    Статус исследования: {study.status}. Результаты анализа появятся после завершения обработки.
                  </p>
                </div>
              )}
            </div>

            <div>
              {result && (
                <div className="card">
                  <p className="card-title">
                    <span className="card-title-icon">
                      <GaugeIcon />
                    </span>
                    Качество изображения
                  </p>
                  <p style={{ fontSize: 28, fontWeight: 700, margin: "0 0 4px" }}>
                    {(result.quality_score * 100).toFixed(0)}%
                  </p>
                  {result.quality_issues.length > 0 && (
                    <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
                      Проблемы: {result.quality_issues.join(", ")}
                    </p>
                  )}
                </div>
              )}

              {result && (
                <div className="card">
                  <p className="card-title">
                    <span className="card-title-icon">
                      <ActivityIcon />
                    </span>
                    Вероятность заболеваний
                  </p>
                  {result.diagnoses.map((d) => (
                    <div key={d.code} style={{ marginBottom: 12 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13 }}>
                        <span>{d.label}</span>
                        <span>{(d.probability * 100).toFixed(1)}%</span>
                      </div>
                      <div className="diagnosis-bar">
                        <div className="diagnosis-bar-fill" style={{ width: `${d.probability * 100}%` }} />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>}
        </div>
      </div>
    </div>
  );
}
