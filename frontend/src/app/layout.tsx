import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OCTera",
  description: "AI-платформа для анализа ОКТ сетчатки",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ru">
      <body>{children}</body>
    </html>
  );
}
