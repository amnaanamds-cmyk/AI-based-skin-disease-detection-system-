import io
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from dermaai import assistant
from dermaai.api.main import app
from dermaai.progress import compare_lesions
from dermaai.skin_analysis import analyze_skin
from dermaai.skincare import Profile, build_routine


def skin(color=(215, 165, 140), seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((600, 600, 3), color, np.float32) + rng.normal(0, 2, (600, 600, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def png(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ---- skin analysis ----
def test_clear_skin_scores_better_than_blemished():
    clear, spotty = skin(), skin()
    rng = np.random.default_rng(1)
    for _ in range(25):
        cv2.circle(spotty, tuple(int(v) for v in rng.integers(50, 550, 2)), int(rng.integers(4, 9)), (200, 95, 95), -1)
    a, b = analyze_skin(clear), analyze_skin(spotty)
    assert a["ok"] and b["ok"]
    assert b["scores"]["blemishes"] > a["scores"]["blemishes"] + 30
    assert b["skin_health_score"] < a["skin_health_score"]
    assert a["visualization"].startswith("data:image/jpeg")


def test_dark_skin_tone_is_not_penalised():
    light, dark = analyze_skin(skin()), analyze_skin(skin((110, 70, 50)))
    assert dark["ok"]
    assert abs(dark["skin_health_score"] - light["skin_health_score"]) <= 5


def test_non_skin_image_rejected():
    r = analyze_skin(np.full((400, 400, 3), (40, 90, 200), np.uint8))
    assert not r["ok"] and "skin" in r["message"].lower()


# ---- routine & products ----
def test_routine_for_oily_acne_skin():
    r = build_routine(Profile(skin_type="oily", concerns=["acne"], budget="low"), {"blemishes": 70, "shine": 60})
    codes = {c["code"] for c in r["concerns"]}
    assert {"acne", "oiliness"} <= codes
    assert r["am"][-1]["step"] == "sunscreen"
    assert r["actives"]["pm"] in {"salicylic_acid", "retinoid", "azelaic_acid"}
    assert all(s["products"] for s in r["am"] + r["pm"])


def test_pregnancy_excludes_retinoids():
    r = build_routine(Profile(skin_type="normal", concerns=["aging", "texture"], age=35, pregnant=True))
    assert r["actives"]["pm"] != "retinoid"
    names = [p["name"].lower() for s in r["am"] + r["pm"] for p in s["products"]]
    assert not any("retinol" in n or "adapalene" in n for n in names)


def test_sensitive_skin_gets_fragrance_free():
    r = build_routine(Profile(skin_type="sensitive", concerns=["redness"], sensitive=True))
    assert all(p["fragrance_free"] for s in r["am"] + r["pm"] for p in s["products"])


# ---- progress tracking ----
def lesion(radius, color=(95, 60, 45), extra=None):
    img = skin()
    cv2.circle(img, (300, 300), radius, color, -1)
    if extra:
        cv2.circle(img, (300 + radius // 3, 300), radius // 3, extra, -1)
    return img


def test_stable_lesion():
    r = compare_lesions(lesion(90), lesion(91), days_between=30)
    assert r["ok"] and r["change_level"] == "stable"


def test_growing_lesion_with_new_colour_flagged():
    r = compare_lesions(lesion(70), lesion(110, extra=(20, 15, 15)), days_between=60)
    assert r["change_level"] == "significant"
    assert r["deltas"]["area_change"] > 0.5 and "black" in r["deltas"]["new_colors"]
    assert r["monthly_area_growth"] > 0


# ---- assistant ----
def test_red_flags_escalate(monkeypatch):
    monkeypatch.setenv("DERMAAI_ASSISTANT_LLM", "0")
    assert assistant.chat("my mole is bleeding and getting bigger")["escalation"]["level"] == "urgent"
    r = assistant.chat("my lips are swelling and I can't breathe")
    assert r["escalation"]["level"] == "emergency" and "emergency" in r["reply"].lower()
    assert assistant.chat("how much sunscreen do I need")["escalation"] is None


def test_knowledge_base_retrieval(monkeypatch):
    monkeypatch.setenv("DERMAAI_ASSISTANT_LLM", "0")
    for q, title in [("how much sunscreen should I use", "Sunscreen"), ("is retinol safe when pregnant", "Pregnancy-safe skincare"),
                     ("ring shaped itchy rash", "Fungal skin infections"), ("dark spots on my cheeks", "Hyperpigmentation")]:
        r = assistant.chat(q)
        assert r["engine"] == "knowledge-base" and r["sources"][0]["title"] == title


def test_claude_path_uses_grounding_and_falls_back(monkeypatch):
    calls = []

    class FakeMessages:
        def create(self, **kw):
            calls.append(kw)
            return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="Use SPF 30+ daily.")])

    monkeypatch.setenv("DERMAAI_ASSISTANT_LLM", "1")
    monkeypatch.setattr(assistant, "_client", lambda: SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages())))
    r = assistant.chat("how much sunscreen?", history=[{"role": "assistant", "content": "hi"}], context={"skin_health_score": 80})
    assert r["engine"] == "claude" and r["reply"] == "Use SPF 30+ daily."
    kw = calls[0]
    assert kw["model"] == assistant.CLAUDE_MODEL and kw["messages"][0]["role"] == "user"
    assert "<reference_notes>" in kw["messages"][-1]["content"] and "skin_health_score: 80" in kw["messages"][-1]["content"]

    class Refusing:
        def create(self, **kw):
            return SimpleNamespace(stop_reason="refusal", content=[])

    monkeypatch.setattr(assistant, "_client", lambda: SimpleNamespace(beta=SimpleNamespace(messages=Refusing())))
    r = assistant.chat("how much sunscreen?")
    assert r["engine"] == "knowledge-base" and "SPF" in r["reply"]


# ---- API ----
def test_skin_analyze_endpoint(client):
    r = client.post("/api/skin/analyze", files={"image": ("f.png", png(skin()), "image/png")},
                    data={"skin_type": "dry", "concerns": "pigmentation,dryness", "age": "40", "budget": "high"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["analysis"]["ok"] and body["routine"]["profile"]["skin_type"] == "dry"
    assert client.post("/api/skin/analyze", files={"image": ("f.png", png(skin()), "image/png")},
                       data={"concerns": "warts"}).status_code == 422


def test_routine_and_chat_endpoints(client, monkeypatch):
    monkeypatch.setenv("DERMAAI_ASSISTANT_LLM", "0")
    r = client.post("/api/skin/routine", json={"skin_type": "combination", "concerns": ["dullness"], "photo_scores": {"redness": 70}})
    assert r.status_code == 200 and "redness" in {c["code"] for c in r.json()["concerns"]}
    c = client.post("/api/assistant/chat", json={"message": "what is the ABCDE rule?", "history": []})
    assert c.status_code == 200 and "Asymmetry" in c.json()["reply"]
    assert client.post("/api/assistant/chat", json={"message": ""}).status_code == 422


def test_compare_endpoint(client):
    r = client.post("/api/lesion/compare", files={"before": ("a.png", png(lesion(70)), "image/png"),
                                                  "after": ("b.png", png(lesion(110, extra=(20, 15, 15))), "image/png")},
                    data={"days_between": "30"})
    assert r.status_code == 200 and r.json()["change_level"] == "significant"


def test_offline_assistant_explains_latest_results(monkeypatch):
    monkeypatch.setenv("DERMAAI_ASSISTANT_LLM", "0")
    ctx = {"lesion_triage": "moderate", "lesion_top_prediction": "Melanocytic nevus (mole) (61%)",
           "lesion_malignancy_probability": "22%", "skin_health_score": 74, "routine_actives": "AM niacinamide, PM azelaic_acid"}
    r = assistant.chat("Explain my last result", context=ctx)
    assert "moderate priority" in r["reply"] and "74/100" in r["reply"] and "niacinamide" in r["reply"]
