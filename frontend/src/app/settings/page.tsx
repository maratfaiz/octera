"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ApiError, auth, getToken } from "@/lib/api";
import { Sidebar } from "@/components/Sidebar";
import { AlertZoneIcon, LockIcon, ShieldCheckIcon } from "@/components/icons";

export default function SettingsPage() {
  const router = useRouter();
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!getToken()) {
      router.replace("/login");
    }
  }, [router]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSuccess(false);

    if (newPassword !== confirmPassword) {
      setError("Новый пароль и подтверждение не совпадают");
      return;
    }

    setLoading(true);
    try {
      await auth.changePassword(currentPassword, newPassword);
      setSuccess(true);
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось сменить пароль");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <Sidebar />
      <div className="main-content">
        <div className="container" style={{ maxWidth: 520 }}>
          <div className="page-header-row">
            <div>
              <h1 className="page-heading">Настройки</h1>
              <p className="page-subtitle" style={{ margin: 0 }}>
                Управление аккаунтом.
              </p>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="card">
            <p className="card-title">
              <span className="card-title-icon">
                <LockIcon />
              </span>
              Смена пароля
            </p>

            <div className="form-row">
              <label>Текущий пароль</label>
              <div className="input-icon-row">
                <LockIcon />
                <input
                  type="password"
                  value={currentPassword}
                  onChange={(e) => setCurrentPassword(e.target.value)}
                  required
                  autoComplete="current-password"
                />
              </div>
            </div>

            <div className="form-row">
              <label>Новый пароль</label>
              <div className="input-icon-row">
                <LockIcon />
                <input
                  type="password"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  minLength={8}
                  maxLength={72}
                  required
                  autoComplete="new-password"
                />
              </div>
            </div>

            <div className="form-row">
              <label>Повторите новый пароль</label>
              <div className="input-icon-row">
                <LockIcon />
                <input
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  minLength={8}
                  maxLength={72}
                  required
                  autoComplete="new-password"
                />
              </div>
            </div>

            {error && (
              <div className="error">
                <AlertZoneIcon />
                {error}
              </div>
            )}
            {success && (
              <div className="success">
                <ShieldCheckIcon />
                Пароль успешно изменён
              </div>
            )}

            <button type="submit" disabled={loading} style={{ width: "100%", marginTop: 8 }}>
              {loading ? "Подождите…" : "Сменить пароль"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
