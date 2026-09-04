import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ALLOWED_UPLOAD_TYPES, MAX_UPLOAD_SIZE_MB, auth, clearToken, getToken, setToken, studies, validateUploadFile } from "./api";

function makeFile(type: string, sizeBytes: number): File {
  return new File([new Uint8Array(sizeBytes)], "scan.png", { type });
}

describe("validateUploadFile", () => {
  it("accepts an allowed type under the size limit", () => {
    const file = makeFile(ALLOWED_UPLOAD_TYPES[0], 1024);
    expect(validateUploadFile(file)).toBeNull();
  });

  it("rejects a disallowed content type", () => {
    const file = makeFile("application/pdf", 1024);
    expect(validateUploadFile(file)).toMatch(/JPEG, PNG/);
  });

  it("rejects a file over the max size", () => {
    const file = makeFile("image/png", (MAX_UPLOAD_SIZE_MB + 1) * 1024 * 1024);
    expect(validateUploadFile(file)).toMatch(/Максимальный размер/);
  });
});

describe("auth.login", () => {
  beforeEach(() => {
    clearToken();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("stores the access token on success", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(JSON.stringify({ access_token: "abc123" }), { status: 200 })),
    );

    await auth.login("test@example.com", "password123");

    expect(getToken()).toBe("abc123");
  });

  it("throws an ApiError carrying the backend's detail message on failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(JSON.stringify({ detail: "Неверный email или пароль" }), { status: 401 }),
        ),
    );

    await expect(auth.login("test@example.com", "wrong")).rejects.toThrow("Неверный email или пароль");
    expect(getToken()).toBeNull();
  });

  it("extracts a readable message from FastAPI's array-shaped 422 validation detail", async () => {
    // FastAPI's own pydantic validation errors send `detail` as an array of
    // {loc, msg, type} objects, not a string -- passed through unchecked, the
    // Error base class's string coercion turns that into "[object Object]".
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            detail: [{ loc: ["body", "email"], msg: "field required", type: "value_error.missing" }],
          }),
          { status: 422 },
        ),
      ),
    );

    await expect(auth.login("test@example.com", "wrong")).rejects.toThrow("field required");
  });
});

describe("session-expiry handling", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    clearToken();
  });

  it("clears the stored token when an authenticated request comes back 401", async () => {
    // A 401 while a token WAS sent means the session expired/was revoked --
    // distinct from auth.login's own 401 (wrong password, sent with no
    // token at all, see the test above), which must not clear anything.
    setToken("stale-token");
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "Not authenticated" }), { status: 401 })),
    );

    await expect(studies.list()).rejects.toThrow();

    expect(getToken()).toBeNull();
  });

  it("does not clear anything on a 401 with no token present (e.g. a failed login attempt)", async () => {
    clearToken();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "Неверный email или пароль" }), { status: 401 })),
    );

    await expect(auth.login("test@example.com", "wrong")).rejects.toThrow();

    expect(getToken()).toBeNull();
  });
});
