"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ApiError, MAX_UPLOAD_SIZE_MB, analysis, getToken, studies, validateUploadFile } from "@/lib/api";
import { Sidebar } from "@/components/Sidebar";

export default function DashboardPage() {
  const router = useRouter();
  const [eye, setEye] = useState("OD");
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!getToken()) {
      router.replace("/login");
    }
  }, [router]);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const selected = e.target.files?.[0] ?? null;
    if (selected) {
      const validationError = validateUploadFile(selected);
      if (validationError) {
        setError(validationError);
        setFile(null);
        e.target.value = "";
        return;
      }
    }
    setError(null);
    setFile(selected);
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
        <div className="container">
          <h1 style={{ marginBottom: 4 }}>Добро пожаловать в OCTera</h1>
          <p style={{ color: "var(--text-muted)", marginTop: 0 }}>
            Загрузите ОКТ-снимок для анализа с помощью искусственного интеллекта
          </p>

          <form onSubmit={handleUpload} className="upload-card">
            <div className="upload-plus">+</div>
            <h2 style={{ margin: "0 0 4px" }}>Новый ОКТ-снимок</h2>
            <p style={{ color: "var(--text-muted)", marginTop: 0 }}>
              Выберите файл для загрузки. Поддерживаются JPEG, PNG, TIFF. Максимальный размер: {MAX_UPLOAD_SIZE_MB}{" "}
              МБ.
            </p>

            <div className="form-row">
              <label>Глаз</label>
              <select value={eye} onChange={(e) => setEye(e.target.value)}>
                <option value="OD">OD (правый)</option>
                <option value="OS">OS (левый)</option>
              </select>
            </div>
            <div className="form-row">
              <input type="file" accept="image/jpeg,image/png,image/tiff" onChange={handleFileChange} required />
            </div>
            {error && <div className="error">{error}</div>}
            <button type="submit" disabled={uploading || !file}>
              {uploading ? (
                <span className="loading-row">
                  <span className="spinner" /> Анализ…
                </span>
              ) : (
                "Загрузить и запустить анализ"
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
