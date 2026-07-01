"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ApiError, getToken, patients } from "@/lib/api";
import type { Patient } from "@/lib/types";
import { Topbar } from "@/components/Topbar";

export default function DashboardPage() {
  const router = useRouter();
  const [items, setItems] = useState<Patient[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [fullName, setFullName] = useState("");
  const [birthDate, setBirthDate] = useState("");
  const [sex, setSex] = useState("");
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    if (!getToken()) {
      router.replace("/login");
      return;
    }
    loadPatients();
  }, [router]);

  async function loadPatients() {
    setLoading(true);
    try {
      setItems(await patients.list());
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось загрузить пациентов");
    } finally {
      setLoading(false);
    }
  }

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    setCreating(true);
    setError(null);
    try {
      await patients.create({
        full_name: fullName,
        birth_date: birthDate || undefined,
        sex: sex || undefined,
      });
      setFullName("");
      setBirthDate("");
      setSex("");
      await loadPatients();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось создать пациента");
    } finally {
      setCreating(false);
    }
  }

  return (
    <>
      <Topbar />
      <div className="container">
        <h2>Новый пациент</h2>
        <form onSubmit={handleCreate} className="card">
          <div className="form-row">
            <label>ФИО пациента</label>
            <input value={fullName} onChange={(e) => setFullName(e.target.value)} required />
          </div>
          <div className="form-row">
            <label>Дата рождения</label>
            <input type="date" value={birthDate} onChange={(e) => setBirthDate(e.target.value)} />
          </div>
          <div className="form-row">
            <label>Пол</label>
            <select value={sex} onChange={(e) => setSex(e.target.value)}>
              <option value="">Не указан</option>
              <option value="M">Мужской</option>
              <option value="F">Женский</option>
            </select>
          </div>
          {error && <div className="error">{error}</div>}
          <button type="submit" disabled={creating}>
            {creating ? "Создание…" : "Добавить пациента"}
          </button>
        </form>

        <h2>Пациенты</h2>
        <div className="card">
          {loading && <p style={{ color: "var(--text-muted)" }}>Загрузка…</p>}
          {!loading && items.length === 0 && (
            <p style={{ color: "var(--text-muted)" }}>Пациентов пока нет.</p>
          )}
          {items.map((p) => (
            <div key={p.id} className="list-item">
              <div>
                <div>{p.full_name}</div>
                <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                  {p.birth_date ?? "дата рождения не указана"}
                </div>
              </div>
              <Link href={`/patients/${p.id}`}>Открыть →</Link>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
