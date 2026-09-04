import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

// jsdom doesn't implement these; the dashboard's file-preview effect calls
// them as soon as a file is selected.
beforeAll(() => {
  URL.createObjectURL = vi.fn(() => "blob:mock-url");
  URL.revokeObjectURL = vi.fn();
});

beforeEach(() => {
  vi.clearAllMocks();
});

const pushMock = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock, replace: vi.fn() }),
}));

const uploadMock = vi.fn();
const runMock = vi.fn();
vi.mock("@/lib/api", () => ({
  ApiError: class ApiError extends Error {
    status: number;
    constructor(status: number, message: string) {
      super(message);
      this.status = status;
    }
  },
  MAX_UPLOAD_SIZE_MB: 20,
  getToken: () => "test-token",
  studies: { upload: (...args: unknown[]) => uploadMock(...args) },
  analysis: { run: (...args: unknown[]) => runMock(...args) },
  validateUploadFile: () => null,
}));

vi.mock("@/components/Sidebar", () => ({ Sidebar: () => null }));

import DashboardPage from "./page";

function makeFile(): File {
  return new File([new Uint8Array(10)], "scan.png", { type: "image/png" });
}

describe("DashboardPage upload flow", () => {
  it("navigates to the new study even when the follow-up analysis run fails", async () => {
    // The study record is created by studies.upload() before analysis.run()
    // is ever called -- if only the analysis step fails (network blip,
    // backend exception), the user must still land on that study (which
    // shows its own failed/processing status) instead of being stuck on the
    // dashboard with no link back to a study that already exists, which
    // used to invite re-uploading the same file as a duplicate.
    uploadMock.mockResolvedValue({ id: "study-123" });
    runMock.mockRejectedValue(new Error("pipeline exploded"));

    const { container } = render(<DashboardPage />);
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [makeFile()] } });
    fireEvent.submit(container.querySelector("form") as HTMLFormElement);

    await waitFor(() => expect(pushMock).toHaveBeenCalledWith("/studies/study-123"));
  });

  it("shows an error and does not navigate when the upload itself fails", async () => {
    uploadMock.mockRejectedValue(new Error("upload failed"));

    const { container } = render(<DashboardPage />);
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [makeFile()] } });
    fireEvent.submit(container.querySelector("form") as HTMLFormElement);

    await waitFor(() => expect(screen.getByText("Не удалось загрузить снимок")).toBeInTheDocument());
    expect(pushMock).not.toHaveBeenCalled();
    expect(runMock).not.toHaveBeenCalled();
  });
});
