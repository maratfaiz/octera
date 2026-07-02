"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { ApiError, getToken, studies } from "@/lib/api";
import type { Study } from "@/lib/types";
import { Sidebar } from "@/components/Sidebar";

const STATUS_LABELS: Record<Study["status"], string> = {
  uploaded: "Загружено",
  processing: "Анализ…",
  completed: "Готово",
  failed: "Ошибка",
};

export default function HistoryPage() {
  const router = useRouter();
  const [items, setItems] = useState<Study[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!getToken()) {
      router.replace("/login");
      return;
    }
    studies
      .list()
      .then(setItems)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Не удалось загрузить историю"))
      .finally(() => setLoading(false));
  }, [router]);

  return (
    <div className="app-shell">
      <Sidebar />
      <div className="main-content">
        <div className="container">
          <h1>История</h1>
          {error && <div className="error">{error}</div>}
          <div className="card">
            {loading && <p style={{ color: "var(--text-muted)" }}>Загрузка…</p>}
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
