"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { auth, clearToken } from "@/lib/api";
import type { User } from "@/lib/types";
import { APP_VERSION } from "@/lib/version";

const NAV_ITEMS = [
  { href: "/dashboard", label: "Новый ОКТ", icon: "+" },
  { href: "/history", label: "История", icon: "⏱" },
];

export function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    auth.me().then(setUser).catch(() => setUser(null));
  }, []);

  function handleLogout() {
    clearToken();
    router.replace("/login");
  }

  const initial = (user?.full_name || user?.email || "?").charAt(0).toUpperCase();

  return (
    <aside className="sidebar">
      <Link href="/dashboard" className="sidebar-logo">
        OCTera
      </Link>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className={`sidebar-link ${pathname === item.href ? "active" : ""}`}
          >
            <span className="sidebar-link-icon">{item.icon}</span>
            {item.label}
          </Link>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="sidebar-account">
          <span className="sidebar-avatar">{initial}</span>
          <div>
            <div>{user?.full_name ?? "…"}</div>
            <button className="sidebar-logout" onClick={handleLogout}>
              Выйти
            </button>
          </div>
        </div>
        <div className="sidebar-version">OCTera v{APP_VERSION}</div>
      </div>
    </aside>
  );
}
