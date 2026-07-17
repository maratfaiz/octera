"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { auth, clearToken } from "@/lib/api";
import type { User } from "@/lib/types";
import { APP_VERSION } from "@/lib/version";
import { ClockIcon, LogOutIcon, LogoIcon, PlusCircleIcon } from "@/components/icons";

const NAV_ITEMS = [
  { href: "/dashboard", label: "Новый ОКТ", icon: PlusCircleIcon },
  { href: "/history", label: "История", icon: ClockIcon },
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
        <span className="sidebar-logo-mark">
          <LogoIcon />
        </span>
        OCTera
      </Link>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`sidebar-link ${pathname === item.href ? "active" : ""}`}
            >
              <span className="sidebar-link-icon">
                <Icon />
              </span>
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <div className="sidebar-account">
          <span className="sidebar-avatar">{initial}</span>
          <div>
            <div>{user?.full_name ?? "…"}</div>
            <button className="sidebar-logout" onClick={handleLogout}>
              <LogOutIcon />
              Выйти
            </button>
          </div>
        </div>
        <div className="sidebar-version">OCTera v{APP_VERSION}</div>
      </div>
    </aside>
  );
}
