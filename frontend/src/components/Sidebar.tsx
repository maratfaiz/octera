"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { auth, clearToken } from "@/lib/api";
import type { User } from "@/lib/types";
import { APP_VERSION } from "@/lib/version";
import { ClockIcon, HomeIcon, LogOutIcon, MenuIcon, PlusIcon, UserIcon, WaveLogoIcon } from "@/components/icons";

// Settings isn't a peer of these content tabs -- it's reached through the
// account row in the footer below (see .sidebar-account), not listed here.
const NAV_ITEMS = [
  { href: "/dashboard", label: "Новый ОКТ", icon: PlusIcon, isActive: (path: string) => path === "/dashboard" },
  { href: "/history", label: "История", icon: ClockIcon, isActive: (path: string) => path === "/history" || path.startsWith("/studies") },
];

export function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    auth.me().then(setUser).catch(() => setUser(null));
  }, []);

  // Route changes (tapping a nav link) should close the mobile menu panel;
  // otherwise it stays open over the new page underneath it.
  useEffect(() => {
    setMenuOpen(false);
  }, [pathname]);

  function handleLogout() {
    clearToken();
    router.replace("/login");
  }

  const initial = (user?.full_name || user?.email || "?").charAt(0).toUpperCase();
  const isHistoryActive = pathname === "/history" || pathname.startsWith("/studies");

  return (
    <>
      <aside className="sidebar">
        <Link href="/dashboard" className="sidebar-logo">
          <span className="sidebar-logo-mark">
            <WaveLogoIcon />
          </span>
          <span>
            <div className="sidebar-logo-name">OCTera</div>
            <div className="sidebar-logo-tag">Retina AI Platform</div>
          </span>
        </Link>

        <nav className="sidebar-nav">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`sidebar-link ${item.isActive(pathname) ? "active" : ""}`}
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
          <Link href="/settings" className={`sidebar-account ${pathname === "/settings" ? "active" : ""}`}>
            <span className="sidebar-avatar">{initial}</span>
            <div>{user?.full_name ?? "…"}</div>
          </Link>
          <button className="sidebar-logout" onClick={handleLogout}>
            <LogOutIcon />
            Выйти
          </button>
          <div className="sidebar-version">OCTera v{APP_VERSION}</div>
        </div>
      </aside>

      <header className="mobile-topbar">
        <Link href="/dashboard" className="sidebar-logo">
          <span className="sidebar-logo-mark">
            <WaveLogoIcon />
          </span>
          <span>
            <div className="sidebar-logo-name">OCTera</div>
            <div className="sidebar-logo-tag">Retina AI Platform</div>
          </span>
        </Link>
        <button
          type="button"
          className="mobile-menu-toggle"
          aria-label="Меню"
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen((open) => !open)}
        >
          <MenuIcon />
        </button>
      </header>

      {menuOpen && (
        <div className="mobile-menu-panel">
          <nav className="mobile-menu-nav">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`sidebar-link ${item.isActive(pathname) ? "active" : ""}`}
                >
                  <span className="sidebar-link-icon">
                    <Icon />
                  </span>
                  {item.label}
                </Link>
              );
            })}
          </nav>
          <div className="sidebar-footer" style={{ border: "none", padding: 0 }}>
            <Link href="/settings" className={`sidebar-account ${pathname === "/settings" ? "active" : ""}`}>
              <span className="sidebar-avatar">{initial}</span>
              <div>{user?.full_name ?? "…"}</div>
            </Link>
            <button className="sidebar-logout" onClick={handleLogout}>
              <LogOutIcon />
              Выйти
            </button>
            <div className="sidebar-version">OCTera v{APP_VERSION}</div>
          </div>
        </div>
      )}

      <nav className="mobile-tabbar">
        <Link href="/dashboard" className={`mobile-tab ${pathname === "/dashboard" ? "active" : ""}`}>
          <HomeIcon />
          Главная
        </Link>
        <Link href="/history" className={`mobile-tab ${isHistoryActive ? "active" : ""}`}>
          <ClockIcon />
          История
        </Link>
        <button type="button" className={`mobile-tab ${menuOpen ? "active" : ""}`} onClick={() => setMenuOpen((open) => !open)}>
          <UserIcon />
          Профиль
        </button>
      </nav>
    </>
  );
}
