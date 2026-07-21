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
  ShieldCheckIcon,
  WaveLogoIcon,
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
    // `cancelled` guards against a slower fetch for a previous `id` resolving
    // after navigation to a new study and overwriting this page with the
    // wrong study's data -- Next.js reuses this component instance across
    // dynamic [id] navigations, so a fresh `load()` call alone doesn't stop
    // an in-flight one from a prior id.
    let cancelled = false;
    const objectUrls: string[] = [];

    async function load() {
      setLoading(true);
      setError(null);
      setStudy(null);
      setResult(null);
      setImageUrl(null);
      setSegmentationUrl(null);
      setPathologyUrl(null);
      try {
        const s = await studies.get(id);
        if (cancelled) return;
        setStudy(s);
        const img = await fetchImageObjectUrl(`/api/v1/studies/${id}/image`);
        if (cancelled) {
          URL.revokeObjectURL(img);
          return;
        }
        objectUrls.push(img);
        setImageUrl(img);

        if (s.status === "completed") {
          const r = await analysis.get(id);
          if (cancelled) return;
          setResult(r);
          if (r.segmentation_map_path) {
            const seg = await fetchImageObjectUrl(`/api/v1/analysis/${id}/segmentation-map`);
            if (cancelled) {
              URL.revokeObjectURL(seg);
              return;
            }
            objectUrls.push(seg);
            setSegmentationUrl(seg);
          }
          if (r.pathology_map_path) {
            const path = await fetchImageObjectUrl(`/api/v1/analysis/${id}/pathology-map`);
            if (cancelled) {
              URL.revokeObjectURL(path);
              return;
            }
            objectUrls.push(path);
            setPathologyUrl(path);
          }
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof ApiError ? err.message : "Не удалось загрузить исследование");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();

    return () => {
      cancelled = true;
      objectUrls.forEach((u) => URL.revokeObjectURL(u));
    };
  }, [id, router]);

  const maxLayerValue = result ? Math.max(...Object.values(result.layer_thickness), 1) : 1;
  const sortedDiagnoses = result ? [...result.diagnoses].sort((a, b) => b.probability - a.probability) : [];

  return (
    <div className="app-shell">
      <Sidebar />
      <div className="main-content">
        <div className="container" style={{ maxWidth: 1160 }}>
          <div className="print-header">
            <WaveLogoIcon />
            <strong>OCTera</strong>
          </div>
          <Link
            href="/history"
            className="no-print"
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
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              {result && (
                <div className="status-badge">
                  <ShieldCheckIcon />
                  Предварительное заключение
                </div>
              )}
              {result && (
                <button type="button" className="secondary no-print" onClick={() => window.print()}>
                  Скачать PDF
                </button>
              )}
            </div>
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
                <div className="card image-panel-wide">
                  <p className="card-title">
                    <span className="card-title-icon">
                      <AlertZoneIcon />
                    </span>
                    Карта патологий
                  </p>
                  {pathologyUrl ? (
                    <div className="pathology-body">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={pathologyUrl} alt="Карта патологий" style={{ width: "100%", borderRadius: 8 }} />
                      {result && (
                        <div className="pathology-checklist">
                          {result.pathology_findings.map((f) => (
                            <div key={f.key} className="pathology-checklist-row">
                              <span
                                className="pathology-checklist-dot"
                                style={{ background: `rgb(${f.color.join(",")})` }}
                              />
                              <span className="pathology-checklist-label">{f.label_ru}</span>
                              <span
                                className={
                                  f.detected
                                    ? "pathology-checklist-status positive"
                                    : "pathology-checklist-status"
                                }
                              >
                                {f.detected ? `Выявлены (${f.zone_count})` : "Не выявлены"}
                              </span>
                            </div>
                          ))}
                          <p className="pathology-checklist-note">
                            Это не диагноз — автоматически найденные участки для проверки врачом. Проверены пока
                            только эти категории; эпиретинальная мембрана, отслойка пигментного эпителия, друзы и
                            субретинальный гиперрефлективный материал — в разработке.
                          </p>
                        </div>
                      )}
                    </div>
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

                  <div className="card report-card">
                    <p className="card-title">
                      <span className="card-title-icon">
                        <ReportIcon />
                      </span>
                      AI-заключение
                    </p>
                    <div className="report-banner">
                      <ShieldCheckIcon />
                      Автоматическое заключение носит предварительный характер и требует подтверждения врачом.
                    </div>
                    <div className="report-body">
                      <p className="report-text">{result.report_text}</p>
                      <svg className="report-wave" viewBox="0 0 160 90" fill="none" aria-hidden="true">
                        <path
                          d="M0 60 C 20 40, 35 75, 55 55 S 90 30, 110 50 S 145 70, 160 45"
                          stroke="currentColor"
                          strokeWidth={10}
                          strokeLinecap="round"
                        />
                      </svg>
                    </div>
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
                <div className="card">
                  <p className="card-title">
                    <span className="card-title-icon">
                      <GaugeIcon />
                    </span>
                    Качество изображения
                  </p>
                  <div className="bar-row">
                    <div className="bar-row-labels">
                      <span>Оценка качества</span>
                      <span>{(result.quality_score * 100).toFixed(0)}%</span>
                    </div>
                    <div className="diagnosis-bar">
                      <div className="diagnosis-bar-fill" style={{ width: `${result.quality_score * 100}%` }} />
                    </div>
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
