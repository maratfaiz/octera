import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

beforeEach(() => {
  vi.clearAllMocks();
});

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn() }),
}));

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
  auth: { changePassword: (...args: unknown[]) => changePasswordMock(...args) },
}));

vi.mock("@/components/Sidebar", () => ({ Sidebar: () => null }));

import { ApiError } from "@/lib/api";
import SettingsPage from "./page";

function fillForm(current: string, next: string, confirm: string) {
  const inputs = document.querySelectorAll(".form-row input");
  fireEvent.change(inputs[0], { target: { value: current } });
  fireEvent.change(inputs[1], { target: { value: next } });
  fireEvent.change(inputs[2], { target: { value: confirm } });
}

describe("SettingsPage change-password form", () => {
  it("calls auth.changePassword and shows a success message on a valid submit", async () => {
    changePasswordMock.mockResolvedValue(undefined);

    render(<SettingsPage />);
    fillForm("old-password", "new-password", "new-password");
    fireEvent.click(screen.getByRole("button", { name: "Сменить пароль" }));

    await waitFor(() => expect(screen.getByText("Пароль успешно изменён")).toBeInTheDocument());
    expect(changePasswordMock).toHaveBeenCalledWith("old-password", "new-password");
  });

  it("shows a mismatch error and does not call the API when confirmation doesn't match", async () => {
    render(<SettingsPage />);
    fillForm("old-password", "new-password", "different-password");
    fireEvent.click(screen.getByRole("button", { name: "Сменить пароль" }));

    expect(screen.getByText("Новый пароль и подтверждение не совпадают")).toBeInTheDocument();
    expect(changePasswordMock).not.toHaveBeenCalled();
  });

  it("shows the server's error message when the current password is wrong", async () => {
    changePasswordMock.mockRejectedValue(new ApiError(400, "Неверный текущий пароль"));

    render(<SettingsPage />);
    fillForm("wrong-password", "new-password", "new-password");
    fireEvent.click(screen.getByRole("button", { name: "Сменить пароль" }));

    await waitFor(() => expect(screen.getByText("Неверный текущий пароль")).toBeInTheDocument());
  });
});
