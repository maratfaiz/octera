"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { clearToken } from "@/lib/api";

export function Topbar() {
  const router = useRouter();

  function handleLogout() {
    clearToken();
    router.replace("/login");
  }

  return (
    <div className="topbar">
      <Link href="/dashboard">OCTera</Link>
      <button className="secondary" onClick={handleLogout}>
        Выйти
      </button>
    </div>
  );
}
