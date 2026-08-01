import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

beforeEach(() => {
  vi.clearAllMocks();
});

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn() }),
}));

const meMock = vi.fn();
const changePasswordMock = vi.fn();
vi.mock("@/lib/api", () => ({
  ApiError: class ApiError extends Error {
    status: number;
    constructor(status: number, message: string) {
      super(message);
      this.status = status;
    }
  },
  getToken: () => "test-token",
  auth: {
    me: (...args: unknown[]) => meMock(...args),
    changePassword: (...args: unknown[]) => changePasswordMock(...args),
  },
}));

vi.mock("@/components/Sidebar", () => ({ Sidebar: () => null }));

import { ApiError } from "@/lib/api";
import SettingsPage from "./page";

function openModalAndFill(current: string, next: string, confirm: string) {
  fireEvent.click(screen.getByRole("button", { name: "Изменить" }));
  const inputs = document.querySelectorAll(".modal-card .form-row input");
  fireEvent.change(inputs[0], { target: { value: current } });
  fireEvent.change(inputs[1], { target: { value: next } });
  fireEvent.change(inputs[2], { target: { value: confirm } });
}

describe("SettingsPage profile section", () => {
  it("shows the current user's name and email once loaded", async () => {
    meMock.mockResolvedValue({ id: "1", email: "anna@example.com", full_name: "Anna Test", role: "user" });

    render(<SettingsPage />);

    await waitFor(() => expect(screen.getByText("Anna Test")).toBeInTheDocument());
    expect(screen.getByText("anna@example.com")).toBeInTheDocument();
  });
});

describe("SettingsPage change-password modal", () => {
  beforeEach(() => {
    meMock.mockResolvedValue({ id: "1", email: "anna@example.com", full_name: "Anna Test", role: "user" });
  });

  it("is closed until the Изменить button is clicked", async () => {
    render(<SettingsPage />);
    await waitFor(() => expect(screen.getByText("Anna Test")).toBeInTheDocument());
    expect(document.querySelector(".modal-overlay")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Изменить" }));
    expect(document.querySelector(".modal-overlay")).toBeInTheDocument();
  });

  it("closes without submitting when the close button is clicked", async () => {
    render(<SettingsPage />);
    await waitFor(() => expect(screen.getByText("Anna Test")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Изменить" }));
    fireEvent.click(screen.getByRole("button", { name: "Закрыть" }));

    expect(document.querySelector(".modal-overlay")).not.toBeInTheDocument();
    expect(changePasswordMock).not.toHaveBeenCalled();
  });

  it("closes the modal and shows a success message on a valid submit", async () => {
    changePasswordMock.mockResolvedValue(undefined);

    render(<SettingsPage />);
    openModalAndFill("old-password", "new-password", "new-password");
    fireEvent.click(screen.getByRole("button", { name: "Сменить пароль" }));

    await waitFor(() => expect(screen.getByText("Пароль успешно изменён")).toBeInTheDocument());
    expect(changePasswordMock).toHaveBeenCalledWith("old-password", "new-password");
    expect(document.querySelector(".modal-overlay")).not.toBeInTheDocument();
  });

  it("shows a mismatch error inside the modal and does not call the API when confirmation doesn't match", async () => {
    render(<SettingsPage />);
    await waitFor(() => expect(screen.getByText("Anna Test")).toBeInTheDocument());
    openModalAndFill("old-password", "new-password", "different-password");
    fireEvent.click(screen.getByRole("button", { name: "Сменить пароль" }));

    expect(screen.getByText("Новый пароль и подтверждение не совпадают")).toBeInTheDocument();
    expect(changePasswordMock).not.toHaveBeenCalled();
    // Still open -- a failed validation must not close the modal out from under the user.
    expect(document.querySelector(".modal-overlay")).toBeInTheDocument();
  });

  it("shows the server's error message inside the modal when the current password is wrong", async () => {
    changePasswordMock.mockRejectedValue(new ApiError(400, "Неверный текущий пароль"));

    render(<SettingsPage />);
    openModalAndFill("wrong-password", "new-password", "new-password");
    fireEvent.click(screen.getByRole("button", { name: "Сменить пароль" }));

    await waitFor(() => expect(screen.getByText("Неверный текущий пароль")).toBeInTheDocument());
    expect(document.querySelector(".modal-overlay")).toBeInTheDocument();
  });
});
