import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from dermaai.api.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def png(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def test_health_and_metadata_endpoints(client):
    assert client.get("/api/health").json()["status"] == "ok"
    assert len(client.get("/api/conditions").json()["conditions"]) == 7
    assert "back" in client.get("/api/options").json()["localization"]
    assert client.get("/").status_code == 200


def test_analyze_returns_full_report(client, lesion_rgb):
    r = client.post("/api/analyze", files={"image": ("l.png", png(lesion_rgb), "image/png")},
                    data={"age": "60", "sex": "male", "localization": "back"})
    assert r.status_code == 200, r.text
    body = r.json()
    probs = [p["probability"] for p in body["prediction"]["probabilities"]]
    assert abs(sum(probs) - 1) < 1e-3 and probs == sorted(probs, reverse=True)
    assert body["triage"]["level"] in {"high", "moderate", "low"}
    assert body["explanation"]["heatmap"].startswith("data:image/jpeg;base64,")
    assert body["abcde"]["asymmetry"] < 0.2
    assert body["model"]["demo_mode"] is True


def test_bad_quality_triggers_retake(client, lesion_rgb):
    dark = (lesion_rgb * 0.08).astype(np.uint8)
    body = client.post("/api/analyze", files={"image": ("d.png", png(dark), "image/png")}, data={"explain": "false"}).json()
    assert body["triage"]["level"] == "retake" and body["explanation"] is None


def test_rejects_invalid_input(client, lesion_rgb):
    assert client.post("/api/analyze", files={"image": ("x.png", b"not an image", "image/png")}).status_code == 415
    r = client.post("/api/analyze", files={"image": ("l.png", png(lesion_rgb), "image/png")}, data={"localization": "moon"})
    assert r.status_code == 422
