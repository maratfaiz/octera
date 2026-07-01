import io

from PIL import Image


def _register_and_login(client):
    client.post(
        "/api/v1/auth/register",
        json={"email": "doc@example.com", "full_name": "Dr. House", "password": "password123"},
    )
    resp = client.post("/api/v1/auth/login", json={"email": "doc@example.com", "password": "password123"})
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_full_flow(client, tmp_path):
    headers = _register_and_login(client)

    patient_resp = client.post(
        "/api/v1/patients",
        json={"full_name": "Ivan Petrov", "sex": "M"},
        headers=headers,
    )
    assert patient_resp.status_code == 201
    patient_id = patient_resp.json()["id"]

    img = Image.new("L", (512, 512), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    study_resp = client.post(
        f"/api/v1/studies?patient_id={patient_id}&eye=OD",
        files={"file": ("scan.png", buf, "image/png")},
        headers=headers,
    )
    assert study_resp.status_code == 201
    study_id = study_resp.json()["id"]

    analysis_resp = client.post(f"/api/v1/analysis/{study_id}/run", headers=headers)
    assert analysis_resp.status_code == 201
    body = analysis_resp.json()
    assert "diagnoses" in body
    assert body["report_text"]

    fetched = client.get(f"/api/v1/analysis/{study_id}", headers=headers)
    assert fetched.status_code == 200
