"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ApiError, auth, getToken } from "@/lib/api";
import { Sidebar } from "@/components/Sidebar";
import type { User } from "@/lib/types";
import { AlertZoneIcon, CloseIcon, LockIcon, ShieldCheckIcon, UserIcon } from "@/components/icons";

export default function SettingsPage() {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [passwordModalOpen, setPasswordModalOpen] = useState(false);
  const [passwordChanged, setPasswordChanged] = useState(false);

  useEffect(() => {
    if (!getToken()) {
      router.replace("/login");
    }
  }, [router]);

  useEffect(() => {
    auth.me().then(setUser).catch(() => setUser(null));
  }, []);

  return (
    <div className="app-shell">
      <Sidebar />
      <div className="main-content">
        <div className="container" style={{ maxWidth: 640 }}>
          <div className="page-header-row">
            <div>
              <h1 className="page-heading">Настройки</h1>
              <p className="page-subtitle" style={{ margin: 0 }}>
                Управление аккаунтом и безопасностью.
              </p>
            </div>
          </div>

          <div className="card">
            <p className="card-title">
              <span className="card-title-icon">
                <UserIcon />
              </span>
              Профиль
            </p>
            <div className="list-item">
              <span style={{ color: "var(--ink-soft)" }}>ФИО</span>
              <span style={{ fontWeight: 600 }}>{user?.full_name ?? "…"}</span>
            </div>
            <div className="list-item">
              <span style={{ color: "var(--ink-soft)" }}>Email</span>
              <span style={{ fontWeight: 600 }}>{user?.email ?? "…"}</span>
            </div>
          </div>

          <div className="card">
            <p className="card-title">
              <span className="card-title-icon">
                <LockIcon />
              </span>
              Безопасность
            </p>
            <div className="list-item">
              <div>
                <div style={{ fontWeight: 600 }}>Пароль</div>
                <div style={{ fontSize: 12, color: "var(--ink-soft)" }}>••••••••</div>
              </div>
              <button
                className="secondary"
                onClick={() => {
                  setPasswordChanged(false);
                  setPasswordModalOpen(true);
                }}
              >
                Изменить
              </button>
            </div>
            {passwordChanged && (
              <div className="success" style={{ marginTop: 4 }}>
                <ShieldCheckIcon />
                Пароль успешно изменён
              </div>
            )}
          </div>
        </div>
      </div>

      {passwordModalOpen && (
        <PasswordChangeModal
          onClose={() => setPasswordModalOpen(false)}
          onSuccess={() => {
            setPasswordModalOpen(false);
            setPasswordChanged(true);
          }}
        />
      )}
    </div>
  );
}

function PasswordChangeModal({ onClose, onSuccess }: { onClose: () => void; onSuccess: () => void }) {
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    if (newPassword !== confirmPassword) {
      setError("Новый пароль и подтверждение не совпадают");
      return;
    }

    setLoading(true);
    try {
      await auth.changePassword(currentPassword, newPassword);
      onSuccess();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось сменить пароль");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2 className="modal-title">Смена пароля</h2>
          <button type="button" className="modal-close" aria-label="Закрыть" onClick={onClose}>
            <CloseIcon />
          </button>
        </div>

        <form onSubmit={handleSubmit}>
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
                autoFocus
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

          <button type="submit" disabled={loading} style={{ width: "100%", marginTop: 8 }}>
            {loading ? "Подождите…" : "Сменить пароль"}
          </button>
        </form>
      </div>
    </div>
  );
}
