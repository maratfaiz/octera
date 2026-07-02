"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ApiError, analysis, getToken, studies } from "@/lib/api";
import type { Study } from "@/lib/types";
import { Topbar } from "@/components/Topbar";

const STATUS_LABELS: Record<Study["status"], string> = {
  uploaded: "Загружено",
  processing: "Анализ…",
  completed: "Готово",
  failed: "Ошибка",
};

export default function DashboardPage() {
  const router = useRouter();
  const [items, setItems] = useState<Study[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [eye, setEye] = useState("OD");
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    if (!getToken()) {
      router.replace("/login");
      return;
    }
    loadStudies();
  }, [router]);

  async function loadStudies() {
    setLoading(true);
    try {
      setItems(await studies.list());
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось загрузить исследования");
    } finally {
      setLoading(false);
    }
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
    <>
      <Topbar />
      <div className="container">
        <h2>Загрузить ОКТ-снимок</h2>
        <form onSubmit={handleUpload} className="card">
          <div className="form-row">
            <label>Глаз</label>
            <select value={eye} onChange={(e) => setEye(e.target.value)}>
              <option value="OD">OD (правый)</option>
              <option value="OS">OS (левый)</option>
            </select>
          </div>
          <div className="form-row">
            <label>Файл (JPEG/PNG/TIFF)</label>
            <input
              type="file"
              accept="image/jpeg,image/png,image/tiff"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              required
            />
          </div>
          {error && <div className="error">{error}</div>}
          <button type="submit" disabled={uploading || !file}>
            {uploading ? "Загрузка…" : "Загрузить и запустить анализ"}
          </button>
        </form>

        <h2>Мои исследования</h2>
        <div className="card">
          {loading && <p style={{ color: "var(--text-muted)" }}>Загрузка…</p>}
          {!loading && items.length === 0 && (
            <p style={{ color: "var(--text-muted)" }}>Исследований пока нет — загрузите первый снимок выше.</p>
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
    </>
  );
}
