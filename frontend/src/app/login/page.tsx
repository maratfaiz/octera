"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { ApiError, auth } from "@/lib/api";
import { AlertZoneIcon, LockIcon, MailIcon, UserIcon, WaveLogoIcon } from "@/components/icons";

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
    let registered = false;
    try {
      if (mode === "register") {
        await auth.register(email, fullName, password);
        registered = true;
      }
      await auth.login(email, password);
      router.push("/dashboard");
    } catch (err) {
      const message = err instanceof ApiError ? err.message : "Не удалось выполнить вход";
      if (registered) {
        // The account was created; only the follow-up login call failed
        // (network blip, etc). Resubmitting "Зарегистрироваться" now would
        // hit a 409 "already exists" that reads as if registration itself
        // failed -- switch to login mode instead so the user's next
        // submission actually retries the thing that failed.
        setMode("login");
        setError(`Аккаунт создан, но не удалось выполнить вход автоматически (${message}). Попробуйте войти.`);
      } else {
        setError(message);
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="auth-shell">
      <div className="auth-card">
        <div className="auth-logo">
          <span className="sidebar-logo-mark">
            <WaveLogoIcon />
          </span>
          <span className="gradient-text">OCTera</span>
        </div>
        <p style={{ color: "var(--ink-soft)", marginTop: 0, textAlign: "center" }}>
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
            onClick={() => {
              setMode(mode === "login" ? "register" : "login");
              setError(null);
            }}
          >
            {mode === "login" ? "Нет аккаунта? Зарегистрироваться" : "Уже есть аккаунт? Войти"}
          </button>
        </div>
      </div>
    </div>
  );
}
