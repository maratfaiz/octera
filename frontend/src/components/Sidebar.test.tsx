import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("next/navigation", () => ({
  usePathname: () => "/history",
  useRouter: () => ({ replace: vi.fn() }),
}));

vi.mock("@/lib/api", () => ({
  auth: {
    me: vi.fn().mockResolvedValue({ id: "1", email: "anna@example.com", full_name: "Anna Test", role: "user" }),
  },
  clearToken: vi.fn(),
}));

import { Sidebar } from "./Sidebar";

describe("Sidebar", () => {
  it("highlights the nav link matching the current path and not the other one", async () => {
    render(<Sidebar />);
    await waitFor(() => expect(screen.getByText("Anna Test")).toBeInTheDocument());

    const historyLink = screen.getByText("История").closest("a");
    const newOctLink = screen.getByText("Новый ОКТ").closest("a");

    expect(historyLink).toHaveClass("active");
    expect(newOctLink).not.toHaveClass("active");
  });

  it("shows the account name once the current user has loaded", async () => {
    render(<Sidebar />);

    await waitFor(() => expect(screen.getByText("Anna Test")).toBeInTheDocument());
  });
});
