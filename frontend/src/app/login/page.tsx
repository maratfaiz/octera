"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { ApiError, auth } from "@/lib/api";
import { AlertZoneIcon, LockIcon, LogoIcon, MailIcon, UserIcon } from "@/components/icons";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [fullName, setFullName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      if (mode === "register") {
        await auth.register(email, fullName, password);
      }
      await auth.login(email, password);
      router.push("/dashboard");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Не удалось выполнить вход");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="auth-shell">
      <div className="auth-card">
        <div className="auth-logo">
          <span className="sidebar-logo-mark">
            <LogoIcon />
          </span>
          <span className="gradient-text">OCTera</span>
        </div>
        <p style={{ color: "var(--text-muted)", marginTop: 0, textAlign: "center" }}>
          {mode === "login" ? "Вход в личный кабинет" : "Регистрация"}
        </p>

        <form onSubmit={handleSubmit} className="card">
          {mode === "register" && (
            <div className="form-row">
              <label>ФИО</label>
              <div className="input-icon-row">
                <UserIcon />
                <input value={fullName} onChange={(e) => setFullName(e.target.value)} required />
              </div>
            </div>
          )}
          <div className="form-row">
            <label>Email</label>
            <div className="input-icon-row">
              <MailIcon />
              <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
            </div>
          </div>
          <div className="form-row">
            <label>Пароль</label>
            <div className="input-icon-row">
              <LockIcon />
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                minLength={8}
                required
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
            {loading ? "Подождите…" : mode === "login" ? "Войти" : "Зарегистрироваться"}
          </button>
        </form>

        <div className="auth-toggle">
          <button
            className="secondary"
            style={{ width: "100%" }}
            onClick={() => setMode(mode === "login" ? "register" : "login")}
          >
            {mode === "login" ? "Нет аккаунта? Зарегистрироваться" : "Уже есть аккаунт? Войти"}
          </button>
        </div>
      </div>
    </div>
  );
}
