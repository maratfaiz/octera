import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const pushMock = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock, replace: vi.fn() }),
}));

const registerMock = vi.fn();
const loginMock = vi.fn();
vi.mock("@/lib/api", () => ({
  auth: {
    register: (...args: unknown[]) => registerMock(...args),
    login: (...args: unknown[]) => loginMock(...args),
  },
  ApiError: class ApiError extends Error {
    status: number;
    constructor(status: number, message: string) {
      super(message);
      this.status = status;
    }
  },
}));

import { ApiError } from "@/lib/api";
import LoginPage from "./page";

describe("LoginPage register-then-login flow", () => {
  it("switches to login mode (not staying stuck in register) if registration succeeds but the follow-up login fails", async () => {
    registerMock.mockResolvedValue({ id: "1", email: "a@example.com" });
    loginMock.mockRejectedValue(new Error("network blip"));

    render(<LoginPage />);
    fireEvent.click(screen.getByText("Нет аккаунта? Зарегистрироваться"));

    const inputs = document.querySelectorAll(".form-row input");
    fireEvent.change(inputs[0], { target: { value: "Test User" } }); // ФИО
    fireEvent.change(inputs[1], { target: { value: "a@example.com" } }); // Email
    fireEvent.change(inputs[2], { target: { value: "password123" } }); // Пароль
    fireEvent.click(screen.getByRole("button", { name: /Зарегистрироваться/ }));

    await waitFor(() => expect(screen.getByText(/Аккаунт создан/)).toBeInTheDocument());

    // Now in login mode: resubmitting must call login again, NOT register
    // again (which would 409 "already exists" and read as registration
    // itself having failed).
    expect(screen.getByRole("button", { name: "Войти" })).toBeInTheDocument();
    registerMock.mockClear();
    loginMock.mockResolvedValue({ access_token: "abc" });
    fireEvent.click(screen.getByRole("button", { name: "Войти" }));

    await waitFor(() => expect(pushMock).toHaveBeenCalledWith("/dashboard"));
    expect(registerMock).not.toHaveBeenCalled();
  });

  it("clears a stale error when the user manually switches modes", async () => {
    registerMock.mockRejectedValue(new ApiError(409, "boom"));

    render(<LoginPage />);
    fireEvent.click(screen.getByText("Нет аккаунта? Зарегистрироваться"));
    const inputs = document.querySelectorAll(".form-row input");
    fireEvent.change(inputs[0], { target: { value: "Test User" } });
    fireEvent.change(inputs[1], { target: { value: "a@example.com" } });
    fireEvent.change(inputs[2], { target: { value: "password123" } });
    fireEvent.click(screen.getByRole("button", { name: /Зарегистрироваться/ }));

    await waitFor(() => expect(screen.getByText("boom")).toBeInTheDocument());

    fireEvent.click(screen.getByText("Уже есть аккаунт? Войти"));

    expect(screen.queryByText("boom")).not.toBeInTheDocument();
  });
});
