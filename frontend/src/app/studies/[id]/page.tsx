"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { ApiError, analysis, fetchImageObjectUrl, getToken, studies } from "@/lib/api";
import type { AnalysisResult, Study } from "@/lib/types";
import { Sidebar } from "@/components/Sidebar";
import { APP_VERSION } from "@/lib/version";
import { LAYER_LABELS_RU } from "@/lib/labels";
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

  const maxLayerValue = result ? Math.max(...Object.values(result.layer_thickness), 1) : 1;
  const sortedDiagnoses = result ? [...result.diagnoses].sort((a, b) => b.probability - a.probability) : [];

  return (
    <div className="app-shell">
      <Sidebar />
      <div className="main-content">
        <div className="container" style={{ maxWidth: 1160 }}>
          <Link
            href="/history"
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
              fontSize: 13,
              fontWeight: 600,
              color: "var(--ink-soft)",
              marginBottom: 10,
              textDecoration: "none",
            }}
          >
            <ArrowLeftIcon /> История
          </Link>
          <div className="page-header-row">
            <div>
              <h1 className="page-heading">Результаты исследования</h1>
              {study && (
                <div className="mono" style={{ fontSize: 13, color: "var(--ink-soft)" }}>
                  {study.eye ?? "—"} · {new Date(study.created_at).toLocaleString("ru-RU")}
                </div>
              )}
            </div>
            {result && <div className="pill pill-amber">⬤ Предварительное заключение</div>}
          </div>
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
                    <p style={{ color: "var(--ink-soft)" }}>Загрузка…</p>
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
                    <p style={{ color: "var(--ink-soft)" }}>
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
                      <p style={{ color: "var(--ink-soft)", fontSize: 12, marginTop: 8, marginBottom: 0 }}>
                        {result && result.pathology_zone_count > 0
                          ? `Алгоритм отметил ${result.pathology_zone_count} зон(ы) — требуют проверки врачом, это не диагноз.`
                          : "Выраженных гипорефлективных зон не найдено."}
                      </p>
                    </>
                  ) : (
                    <p style={{ color: "var(--ink-soft)" }}>
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
                      Толщина слоёв сетчатки (мкм)
                    </p>
                    {Object.entries(result.layer_thickness).map(([layer, value]) => (
                      <div key={layer} className="bar-row">
                        <div className="bar-row-labels">
                          <span>{LAYER_LABELS_RU[layer] ?? layer}</span>
                          <span>{value}</span>
                        </div>
                        <div className="diagnosis-bar">
                          <div
                            className="diagnosis-bar-fill"
                            style={{ width: `${Math.round((value / maxLayerValue) * 100)}%` }}
                          />
                        </div>
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
                    <div className="report-banner">
                      <AlertZoneIcon />
                      Автоматическое заключение носит предварительный характер и требует подтверждения врачом.
                    </div>
                    <p className="report-text">{result.report_text}</p>
                  </div>
                </>
              )}

              {study && study.status !== "completed" && !error && (
                <div className="card">
                  <p style={{ color: "var(--ink-soft)" }}>
                    Статус исследования: {study.status}. Результаты анализа появятся после завершения обработки.
                  </p>
                </div>
              )}
            </div>

            <div>
              {result && (
                <div className="card quality-ring-card">
                  <p className="card-title" style={{ justifyContent: "center" }}>
                    <span className="card-title-icon">
                      <GaugeIcon />
                    </span>
                    Качество изображения
                  </p>
                  <div
                    className="quality-ring"
                    style={{
                      background: `conic-gradient(var(--accent) ${result.quality_score * 100}%, var(--track) 0)`,
                    }}
                  >
                    <div className="quality-ring-inner">{(result.quality_score * 100).toFixed(0)}%</div>
                  </div>
                  {result.quality_issues.length > 0 && (
                    <p style={{ color: "var(--ink-soft)", fontSize: 12, marginTop: 12, marginBottom: 0 }}>
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
                    Вероятность диагноза
                  </p>
                  {sortedDiagnoses.map((d, i) => (
                    <div key={d.code} className="bar-row">
                      <div className="bar-row-labels">
                        <span>{d.label}</span>
                        <span>{(d.probability * 100).toFixed(1)}%</span>
                      </div>
                      <div className="diagnosis-bar">
                        <div
                          className="diagnosis-bar-fill"
                          style={{
                            width: `${d.probability * 100}%`,
                            background: i === 0 ? undefined : "#c7d2e0",
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {result && (
                <div className="inference-card">
                  <div className="inference-card-title">Инференс модели</div>
                  <div className="inference-row">
                    <span>Модель</span>
                    <span>OCTera Segmentation Engine</span>
                  </div>
                  <div className="inference-row">
                    <span>Версия движка</span>
                    <span>v{APP_VERSION}</span>
                  </div>
                  <div className="inference-row">
                    <span>Дата анализа</span>
                    <span>{new Date(result.created_at).toLocaleString("ru-RU")}</span>
                  </div>
                </div>
              )}
            </div>
          </div>}
        </div>
      </div>
    </div>
  );
}
