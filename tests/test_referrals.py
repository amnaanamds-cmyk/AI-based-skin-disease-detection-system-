import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from dermaai.api import main
from dermaai.referrals import CaseStore

AUTH = {"Authorization": "Bearer doc-secret"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DERMAAI_DB", str(tmp_path / "cases.db"))
    monkeypatch.setenv("DERMAAI_CLINICIAN_TOKEN", "doc-secret")
    main.get_store.cache_clear()
    yield TestClient(main.app)
    main.get_store.cache_clear()


def png(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def submit(client, lesion_rgb, **data):
    return client.post("/api/referrals", files={"image": ("l.png", png(lesion_rgb), "image/png")},
                       data={"consent": "true", "age": "58", "sex": "female", "localization": "back", **data})


def test_full_referral_lifecycle(client, lesion_rgb):
    r = submit(client, lesion_rgb, note="Has grown recently")
    assert r.status_code == 201, r.text
    case_id, token = r.json()["case_id"], r.json()["access_token"]
    me = {"X-Case-Token": token}

    assert client.get(f"/api/referrals/{case_id}", headers=me).json()["status"] == "new"
    queue = client.get("/api/clinician/cases", headers=AUTH).json()
    assert queue["stats"]["open"] == 1 and queue["cases"][0]["id"] == case_id

    case = client.get(f"/api/clinician/cases/{case_id}", headers=AUTH).json()
    assert case["image"].startswith("data:image/jpeg;base64,") and case["patient"]["note"] == "Has grown recently"
    assert "outline" not in (case["analysis"]["abcde"] or {})

    r = client.post(f"/api/clinician/cases/{case_id}/review", headers=AUTH,
                    json={"status": "closed", "response": "Benign-looking. Re-check in 3 months.",
                          "clinical_impression": "Compound naevus", "reviewer": "Dr A"})
    assert r.status_code == 200 and r.json()["status"] == "closed"
    mine = client.get(f"/api/referrals/{case_id}", headers=me).json()
    assert mine["response"].startswith("Benign-looking") and mine["reviewed_at"]
    assert "clinical_impression" not in mine and "reviewer" not in mine  # clinician-only fields

    fhir = client.get(f"/api/clinician/cases/{case_id}/fhir", headers=AUTH).json()
    types = [e["resource"]["resourceType"] for e in fhir["entry"]]
    assert fhir["resourceType"] == "Bundle" and types.count("Observation") == 7
    assert {"Patient", "DiagnosticReport", "Media"} <= set(types)
    report = next(e["resource"] for e in fhir["entry"] if e["resource"]["resourceType"] == "DiagnosticReport")
    assert report["status"] == "final" and "Compound naevus" in report["conclusion"]
    full_urls = {e["fullUrl"] for e in fhir["entry"]}
    assert all(any(u.endswith(ref["reference"]) for u in full_urls) for ref in report["result"])

    assert client.delete(f"/api/referrals/{case_id}", headers=me).json()["deleted"]
    assert client.get("/api/clinician/cases", headers=AUTH).json()["stats"]["total"] == 0


def test_access_control(client, lesion_rgb):
    case_id = submit(client, lesion_rgb).json()["case_id"]
    assert client.get(f"/api/referrals/{case_id}", headers={"X-Case-Token": "guess"}).status_code == 404
    assert client.delete(f"/api/referrals/{case_id}", headers={"X-Case-Token": "guess"}).status_code == 404
    assert client.get("/api/clinician/cases").status_code == 401
    assert client.get("/api/clinician/cases", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert submit(client, lesion_rgb, consent="false").status_code == 400


def test_clinician_disabled_without_token(client, monkeypatch):
    monkeypatch.delenv("DERMAAI_CLINICIAN_TOKEN")
    assert client.get("/api/clinician/cases", headers=AUTH).status_code == 503


def test_queue_orders_by_urgency(tmp_path):
    store = CaseStore(tmp_path / "q.db")
    img = Image.new("RGB", (64, 64))

    def analysis(level, mal):
        return {"triage": {"level": level, "malignancy_probability": mal},
                "probabilities": [{"name": "x", "probability": 1.0}], "model": {}}

    for level, mal in [("low", 0.05), ("high", 0.6), ("moderate", 0.2), ("high", 0.9)]:
        store.create(img, analysis(level, mal), {})
    assert [(c["urgency"], c["malignancy"]) for c in store.queue()] == [("high", 0.9), ("high", 0.6), ("moderate", 0.2), ("low", 0.05)]
