import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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
    const { container } = render(<Sidebar />);
    await waitFor(() => expect(screen.getByText("Anna Test")).toBeInTheDocument());

    // "История" also appears in the always-rendered mobile bottom tab bar
    // (shown/hidden by CSS breakpoint, not conditional rendering, so both
    // copies exist in the DOM regardless of viewport) -- scope to the
    // desktop <aside> to test one concrete instance.
    const desktopNav = within(container.querySelector(".sidebar") as HTMLElement);
    const historyLink = desktopNav.getByText("История").closest("a");
    const newOctLink = desktopNav.getByText("Новый ОКТ").closest("a");

    expect(historyLink).toHaveClass("active");
    expect(newOctLink).not.toHaveClass("active");
  });

  it("shows the account name once the current user has loaded", async () => {
    render(<Sidebar />);

    await waitFor(() => expect(screen.getByText("Anna Test")).toBeInTheDocument());
  });

  it("opens the mobile menu panel from the hamburger button and closes it from the Профиль tab", async () => {
    render(<Sidebar />);
    await waitFor(() => expect(screen.getByText("Anna Test")).toBeInTheDocument());

    // "Anna Test" starts out appearing once (the desktop <aside>, present in
    // the DOM regardless of viewport -- CSS decides visibility). The mobile
    // menu panel is conditionally rendered, so opening it adds a second copy.
    expect(screen.getAllByText("Anna Test")).toHaveLength(1);

    fireEvent.click(screen.getByRole("button", { name: "Меню" }));
    expect(screen.getAllByText("Anna Test")).toHaveLength(2);

    fireEvent.click(screen.getByRole("button", { name: "Профиль" }));
    expect(screen.getAllByText("Anna Test")).toHaveLength(1);
  });
});
