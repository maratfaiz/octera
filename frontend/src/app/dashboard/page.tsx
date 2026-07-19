"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ApiError, MAX_UPLOAD_SIZE_MB, analysis, getToken, studies, validateUploadFile } from "@/lib/api";
import { Sidebar } from "@/components/Sidebar";
import {
  ActivityIcon,
  AlertZoneIcon,
  ArrowRightIcon,
  CloudUploadIcon,
  EyeIcon,
  InfoIcon,
  LayersMapIcon,
  SparkleIcon,
  UploadIcon,
} from "@/components/icons";

export default function DashboardPage() {
  const router = useRouter();
  const [eye, setEye] = useState("OD");
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!getToken()) {
      router.replace("/login");
    }
  }, [router]);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function selectFile(candidate: File | null): boolean {
    if (candidate) {
      const validationError = validateUploadFile(candidate);
      if (validationError) {
        setError(validationError);
        setFile(null);
        return false;
      }
    }
    setError(null);
    setFile(candidate);
    return true;
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    if (!selectFile(e.target.files?.[0] ?? null)) {
      e.target.value = "";
    }
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    selectFile(e.dataTransfer.files?.[0] ?? null);
  }

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const study = await studies.upload(file, eye);
      await analysis.run(study.id);
      router.push(`/studies/${study.id}`);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось загрузить снимок или выполнить анализ");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="app-shell">
      <Sidebar />
      <div className="main-content">
        <div className="container" style={{ maxWidth: 920 }}>
          <div className="page-header-row">
            <div>
              <h1 className="page-heading">Новое исследование</h1>
              <p className="page-subtitle" style={{ margin: 0 }}>
                Загрузите ОКТ-снимок сетчатки — модель проанализирует слои, качество и вероятные патологии.
              </p>
            </div>
            <div className="pill pill-success" style={{ flexShrink: 0 }}>
              <span className="pill-dot" />
              Модель активна
            </div>
          </div>

          <form onSubmit={handleUpload} className="upload-card">
            <div className="upload-card-col">
              <p className="card-title">
                <span className="card-title-icon">
                  <EyeIcon />
                </span>
                Параметры исследования
              </p>
              <label style={{ fontSize: 12, fontWeight: 600, color: "var(--ink-soft)", display: "block", marginBottom: 6 }}>
                Выберите глаз для анализа
              </label>
              <div className="eye-toggle">
                <button
                  type="button"
                  className={`eye-toggle-option ${eye === "OD" ? "active" : ""}`}
                  onClick={() => setEye("OD")}
                >
                  OD · правый
                </button>
                <button
                  type="button"
                  className={`eye-toggle-option ${eye === "OS" ? "active" : ""}`}
                  onClick={() => setEye("OS")}
                >
                  OS · левый
                </button>
              </div>
              <div className="report-banner" style={{ marginBottom: 0 }}>
                <InfoIcon />
                Убедитесь, что выбран правильный глаз перед загрузкой.
              </div>
            </div>

            <div className="upload-card-col" style={{ display: "flex", flexDirection: "column" }}>
              <p className="card-title">
                <span className="card-title-icon">
                  <CloudUploadIcon />
                </span>
                Загрузите ОКТ-снимок
              </p>
              <div className="upload-dropzone" onDrop={handleDrop} onDragOver={(e) => e.preventDefault()}>
                {previewUrl ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={previewUrl} alt="Предпросмотр снимка" style={{ maxWidth: "100%", maxHeight: 100, borderRadius: 8 }} />
                ) : (
                  <>
                    <span className="upload-dropzone-icon">
                      <CloudUploadIcon />
                    </span>
                    <span className="upload-dropzone-hint">Перетащите файл сюда или</span>
                  </>
                )}
                <div className="file-input-trigger">
                  <label htmlFor="oct-file-upload">
                    <UploadIcon />
                    Выбрать файл
                  </label>
                  <input
                    id="oct-file-upload"
                    type="file"
                    accept="image/jpeg,image/png,image/tiff"
                    onChange={handleFileChange}
                    required
                  />
                </div>
                <span className="file-input-name">{file ? file.name : "Файл не выбран"}</span>
                <div style={{ fontSize: 12.5, color: "var(--ink-soft)", textAlign: "center" }}>
                  JPEG, PNG, TIFF · до {MAX_UPLOAD_SIZE_MB} МБ
                </div>
              </div>
              {error && (
                <div className="error">
                  <AlertZoneIcon />
                  {error}
                </div>
              )}
              <button type="submit" disabled={uploading || !file} style={{ marginTop: 16, width: "100%" }}>
                {uploading ? (
                  <span className="loading-row">
                    <span className="spinner" /> Анализ…
                  </span>
                ) : (
                  "Запустить анализ →"
                )}
              </button>
            </div>
          </form>

          <div className="feature-grid">
            <div className="feature-card">
              <div className="feature-card-icon">
                <LayersMapIcon />
              </div>
              <div className="feature-card-text">
                <div className="feature-card-title">Сегментация слоёв</div>
                <div className="feature-card-body">Автоматическое измерение толщины слоёв сетчатки на снимке.</div>
              </div>
              <span className="feature-card-link">
                <ArrowRightIcon />
              </span>
            </div>
            <div className="feature-card">
              <div className="feature-card-icon">
                <AlertZoneIcon />
              </div>
              <div className="feature-card-text">
                <div className="feature-card-title">Карта аномалий</div>
                <div className="feature-card-body">Выделение гипорефлективных зон для визуальной проверки врачом.</div>
              </div>
              <span className="feature-card-link">
                <ArrowRightIcon />
              </span>
            </div>
            <div className="feature-card">
              <div className="feature-card-icon">
                <ActivityIcon />
              </div>
              <div className="feature-card-text">
                <div className="feature-card-title">Вероятность диагноза</div>
                <div className="feature-card-body">
                  Ранжированный список вероятных патологий с уровнем уверенности модели.
                </div>
              </div>
              <span className="feature-card-link">
                <ArrowRightIcon />
              </span>
            </div>
          </div>

          <div className="ai-banner">
            <span className="ai-banner-icon">
              <SparkleIcon />
            </span>
            <div>
              <div className="ai-banner-title">Искусственный интеллект на службе офтальмологии</div>
              <p className="ai-banner-body">
                OCTera помогает врачам быстрее и точнее интерпретировать ОКТ-исследования и принимать обоснованные
                клинические решения.
              </p>
            </div>
            <svg className="ai-banner-wave" viewBox="0 0 220 90" fill="none" aria-hidden="true">
              <path
                d="M0 60 C 25 35, 45 80, 70 55 S 115 25, 140 50 S 195 75, 220 45"
                stroke="currentColor"
                strokeWidth={12}
                strokeLinecap="round"
              />
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}
