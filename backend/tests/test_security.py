import io

import pytest
from PIL import Image

from app.core import config
from app.core.config import Settings, assert_production_config_is_safe


def test_login_is_rate_limited(client):
    client.post(
        "/api/v1/auth/register",
        json={"email": "ratelimit@example.com", "full_name": "Rate Limit", "password": "password123"},
    )

    last_status = None
    for _ in range(10):
        resp = client.post(
            "/api/v1/auth/login", json={"email": "ratelimit@example.com", "password": "wrong-password"}
        )
        last_status = resp.status_code

    assert last_status == 429


def test_rate_limit_resets_between_tests(client):
    # If reset_rate_limits() weren't wired into the test fixtures, the previous
    # test's failed login attempts would still count against this one.
    resp = client.post(
        "/api/v1/auth/register",
        json={"email": "fresh@example.com", "full_name": "Fresh User", "password": "password123"},
    )
    assert resp.status_code == 201


def test_rate_limit_prunes_stale_ip_entries(monkeypatch):
    """Regression test: `_attempts` is a module-level dict keyed by
    `key_prefix:client_ip`. An earlier version only ever wrote back the
    *current* key's filtered attempts, so a client IP's entry, once
    created, stayed in the dict forever -- even after every one of its
    attempts had aged out of the window -- since nothing else ever revisited
    it. Over the life of a long-running process handling many distinct
    client IPs, this is an unbounded memory leak. Confirmed directly against
    the module's internal state: two IPs make one attempt each, time is
    advanced past the window, and a third IP's request should sweep both
    stale entries away rather than leaving them to accumulate forever.
    """
    from app.core import rate_limit as rate_limit_module

    rate_limit_module.reset_rate_limits()
    dependency = rate_limit_module.rate_limit("probe", max_attempts=5, window_seconds=60)

    class _FakeClient:
        def __init__(self, host: str) -> None:
            self.host = host

    class _FakeRequest:
        def __init__(self, host: str) -> None:
            self.client = _FakeClient(host)

    fake_now = 1_000_000.0
    monkeypatch.setattr(rate_limit_module.time, "time", lambda: fake_now)

    dependency(_FakeRequest("1.1.1.1"))
    dependency(_FakeRequest("2.2.2.2"))
    assert set(rate_limit_module._attempts.keys()) == {"probe:1.1.1.1", "probe:2.2.2.2"}

    fake_now += 61  # past the 60s window -- both entries are now stale
    dependency(_FakeRequest("3.3.3.3"))

    assert set(rate_limit_module._attempts.keys()) == {"probe:3.3.3.3"}


def test_upload_rejects_oversized_file(client, monkeypatch):
    monkeypatch.setattr("app.api.routes.studies.settings.max_upload_size_mb", 0)

    client.post(
        "/api/v1/auth/register",
        json={"email": "bigfile@example.com", "full_name": "Big File", "password": "password123"},
    )
    login = client.post("/api/v1/auth/login", json={"email": "bigfile@example.com", "password": "password123"})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    img = Image.new("L", (512, 512), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    resp = client.post(
        "/api/v1/studies?eye=OD",
        files={"file": ("scan.png", buf, "image/png")},
        headers=headers,
    )
    assert resp.status_code == 413


def test_upload_rejects_non_image_bytes_with_spoofed_content_type(client):
    """Regression test: an earlier version trusted the client-supplied
    Content-Type header (accept/reject gate) and the client-supplied
    filename's extension (used verbatim for on-disk storage), but never
    checked that the uploaded bytes actually decoded as an image. Both are
    attacker-controlled -- confirmed exploitable by uploading a PHP payload
    as "shell.php" with Content-Type: image/jpeg, which the old code
    accepted (201) and stored on disk with a .php extension. The fix
    decodes and verifies the actual bytes with PIL and derives the stored
    extension from the verified format, never from client input.
    """
    client.post(
        "/api/v1/auth/register",
        json={"email": "spoofed@example.com", "full_name": "Spoofed Upload", "password": "password123"},
    )
    login = client.post("/api/v1/auth/login", json={"email": "spoofed@example.com", "password": "password123"})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    malicious_bytes = b'<?php echo shell_exec($_GET["cmd"]); ?>'
    resp = client.post(
        "/api/v1/studies?eye=OD",
        files={"file": ("shell.php", io.BytesIO(malicious_bytes), "image/jpeg")},
        headers=headers,
    )

    assert resp.status_code == 415


def test_upload_rejects_decompression_bomb(client):
    """Regression test: Pillow raises Image.DecompressionBombError from
    Image.open() itself once the declared pixel count exceeds
    2 * Image.MAX_IMAGE_PIXELS -- independent of the file's byte size, so the
    max_upload_size_mb check above doesn't catch it. DecompressionBombError
    is a plain Exception subclass, not OSError, so an earlier version of the
    except clause (UnidentifiedImageError, OSError) let it fall through
    uncaught and surface as an unhandled 500 instead of the intended 415. A
    flat, highly-compressible image is enough to trigger it well under the
    20MB size cap.
    """
    client.post(
        "/api/v1/auth/register",
        json={"email": "bomb@example.com", "full_name": "Decompression Bomb", "password": "password123"},
    )
    login = client.post("/api/v1/auth/login", json={"email": "bomb@example.com", "password": "password123"})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    img = Image.new("L", (15000, 15000), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    resp = client.post(
        "/api/v1/studies?eye=OD",
        files={"file": ("scan.png", buf, "image/png")},
        headers=headers,
    )

    assert resp.status_code == 415


def test_upload_derives_extension_from_actual_content_not_filename(client):
    """A real JPEG uploaded under a misleading .png filename must be stored
    with a .jpg extension (derived from the verified decoded format), not
    the client-supplied .png -- proving the fix validates actual bytes
    rather than trusting client-supplied metadata.
    """
    client.post(
        "/api/v1/auth/register",
        json={"email": "mismatch@example.com", "full_name": "Mismatched Extension", "password": "password123"},
    )
    login = client.post("/api/v1/auth/login", json={"email": "mismatch@example.com", "password": "password123"})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    img = Image.new("L", (256, 256), color=100)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    resp = client.post(
        "/api/v1/studies?eye=OD",
        files={"file": ("scan.png", buf, "image/jpeg")},
        headers=headers,
    )

    assert resp.status_code == 201
    assert resp.json()["image_path"].endswith(".jpg")


def test_register_rejects_password_shorter_than_8_chars(client):
    """Regression test: an earlier version had no server-side password
    length constraint at all -- the registration form's own `minLength={8}`
    is client-side only. Confirmed exploitable by calling the API directly:
    registering with password="a" was accepted (201) and created a real
    account before this fix.
    """
    resp = client.post(
        "/api/v1/auth/register",
        json={"email": "weakpass@example.com", "full_name": "Weak Pass", "password": "a"},
    )

    assert resp.status_code == 422


def test_register_accepts_password_of_exactly_8_chars(client):
    resp = client.post(
        "/api/v1/auth/register",
        json={"email": "eightchars@example.com", "full_name": "Eight Chars", "password": "abcdefgh"},
    )

    assert resp.status_code == 201


def test_production_config_rejects_default_secret(monkeypatch):
    insecure = Settings(environment="production", secret_key="change-me-in-production")
    monkeypatch.setattr(config, "settings", insecure)

    with pytest.raises(RuntimeError):
        assert_production_config_is_safe()


def test_production_config_accepts_custom_secret(monkeypatch):
    secure = Settings(environment="production", secret_key="a-real-random-secret")
    monkeypatch.setattr(config, "settings", secure)

    assert_production_config_is_safe()  # should not raise
