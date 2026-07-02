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


def test_production_config_rejects_default_secret(monkeypatch):
    insecure = Settings(environment="production", secret_key="change-me-in-production")
    monkeypatch.setattr(config, "settings", insecure)

    with pytest.raises(RuntimeError):
        assert_production_config_is_safe()


def test_production_config_accepts_custom_secret(monkeypatch):
    secure = Settings(environment="production", secret_key="a-real-random-secret")
    monkeypatch.setattr(config, "settings", secure)

    assert_production_config_is_safe()  # should not raise
