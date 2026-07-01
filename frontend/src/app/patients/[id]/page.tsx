"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { ApiError, analysis, getToken, patients, studies } from "@/lib/api";
import type { Patient, Study } from "@/lib/types";
import { Topbar } from "@/components/Topbar";

const STATUS_LABELS: Record<Study["status"], string> = {
  uploaded: "Загружено",
  processing: "Анализ…",
  completed: "Готово",
  failed: "Ошибка",
};

export default function PatientPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();

  const [patient, setPatient] = useState<Patient | null>(null);
  const [items, setItems] = useState<Study[]>([]);
  const [eye, setEye] = useState("OD");
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
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
      const [p, s] = await Promise.all([patients.get(id), studies.listByPatient(id)]);
      setPatient(p);
      setItems(s);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось загрузить данные пациента");
    }
  }

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const study = await studies.upload(id, file, eye);
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
        <Link href="/dashboard">← К списку пациентов</Link>
        <h2>{patient?.full_name ?? "…"}</h2>

        <div className="card">
          <p className="card-title">Загрузить ОКТ-снимок</p>
          <form onSubmit={handleUpload}>
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
        </div>

        <h2>Исследования</h2>
        <div className="card">
          {items.length === 0 && <p style={{ color: "var(--text-muted)" }}>Исследований пока нет.</p>}
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
