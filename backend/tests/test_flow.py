import io

from PIL import Image


def _register_and_login(client, email="user@example.com"):
    client.post(
        "/api/v1/auth/register",
        json={"email": email, "full_name": "Ivan Petrov", "password": "password123"},
    )
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": "password123"})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _upload_test_study(client, headers):
    img = Image.new("L", (512, 512), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return client.post(
        "/api/v1/studies?eye=OD",
        files={"file": ("scan.png", buf, "image/png")},
        headers=headers,
    )


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert "version" in resp.json()


def test_me_returns_current_user(client):
    headers = _register_and_login(client, email="me_test@example.com")

    resp = client.get("/api/v1/auth/me", headers=headers)

    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == "me_test@example.com"
    assert body["full_name"] == "Ivan Petrov"


def test_registration_auto_creates_own_patient_profile(client):
    headers = _register_and_login(client)

    resp = client.get("/api/v1/patients/me", headers=headers)

    assert resp.status_code == 200
    assert resp.json()["full_name"] == "Ivan Petrov"


def test_full_flow(client):
    headers = _register_and_login(client)

    study_resp = _upload_test_study(client, headers)
    assert study_resp.status_code == 201
    study_id = study_resp.json()["id"]

    analysis_resp = client.post(f"/api/v1/analysis/{study_id}/run", headers=headers)
    assert analysis_resp.status_code == 201
    body = analysis_resp.json()
    assert "diagnoses" in body
    assert body["report_text"]

    fetched = client.get(f"/api/v1/analysis/{study_id}", headers=headers)
    assert fetched.status_code == 200


def test_analysis_history_lists_own_results_only(client):
    headers_a = _register_and_login(client, email="history_a@example.com")
    headers_b = _register_and_login(client, email="history_b@example.com")

    study_resp = _upload_test_study(client, headers_a)
    study_id = study_resp.json()["id"]
    client.post(f"/api/v1/analysis/{study_id}/run", headers=headers_a)

    resp_a = client.get("/api/v1/analysis", headers=headers_a)
    assert resp_a.status_code == 200
    history = resp_a.json()
    assert len(history) == 1
    assert history[0]["study_id"] == study_id
    assert "top_diagnosis" in history[0]

    resp_b = client.get("/api/v1/analysis", headers=headers_b)
    assert resp_b.status_code == 200
    assert resp_b.json() == []


def test_users_cannot_access_each_others_studies(client):
    headers_a = _register_and_login(client, email="user_a@example.com")
    headers_b = _register_and_login(client, email="user_b@example.com")

    study_resp = _upload_test_study(client, headers_a)
    study_id = study_resp.json()["id"]

    resp = client.get(f"/api/v1/studies/{study_id}", headers=headers_b)
    assert resp.status_code == 404

    resp = client.post(f"/api/v1/analysis/{study_id}/run", headers=headers_b)
    assert resp.status_code == 404
