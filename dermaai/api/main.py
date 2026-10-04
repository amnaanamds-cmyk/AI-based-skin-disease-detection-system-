"""FastAPI service. Run with: uvicorn dermaai.api.main:app --host 0.0.0.0 --port 8000"""

from __future__ import annotations

import asyncio
import io
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import hmac
import os
import threading

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from .. import __version__
from .. import model_store
from ..config import LOCALIZATIONS, SEXES, condition_info, get_settings
from ..assistant import chat
from ..inference import DISCLAIMER, Predictor
from ..progress import compare_lesions
from ..referrals import CaseStore, summarize_analysis, to_fhir
from ..skin_analysis import analyze_skin
from ..skincare import BUDGETS, CONCERN_LIST, SKIN_TYPES, Profile, build_routine

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(
    title="DermaAI API",
    version=__version__,
    description="Explainable AI triage for skin lesions. Images are processed in memory and never stored.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

MAX_PIXELS = 40_000_000

# Inference is CPU/GPU bound; serialise it so concurrent requests don't thrash memory.
_lock = asyncio.Lock()


_predictor: Predictor | None = None
_predictor_key: tuple | None = None
_predictor_lock = threading.Lock()


def _model_key() -> tuple:
    """Identifies the model the app should serve; changes when training promotes a new version."""
    p = get_settings().checkpoint or model_store.model_path()
    return (str(p), p.stat().st_mtime) if p and p.exists() else (None, None)


def get_predictor() -> Predictor:
    """Load the current model once, and reload it automatically after a retraining run
    switches models/trained/CURRENT (no restart, no code change)."""
    global _predictor, _predictor_key
    key = _model_key()
    if _predictor is None or key != _predictor_key:
        with _predictor_lock:
            if _predictor is None or key != _predictor_key:
                _predictor, _predictor_key = Predictor(), key
    return _predictor


get_predictor.cache_clear = lambda: globals().update(_predictor=None, _predictor_key=None)  # used by tests


@app.get("/api/health")
def health() -> dict:
    p = get_predictor()
    return {"status": "ok", "version": __version__, "demo_mode": p.demo_mode, "model_version": p.version}


@app.get("/api/model")
def model_info() -> dict:
    return get_predictor().info


@app.get("/api/conditions")
def conditions() -> dict:
    p = get_predictor()
    return {"conditions": [{"code": c, **condition_info(c, p.class_info)} for c in p.classes], "disclaimer": DISCLAIMER}


@app.get("/api/options")
def options() -> dict:
    return {"sex": SEXES, "localization": LOCALIZATIONS, "skin_types": SKIN_TYPES,
            "concerns": CONCERN_LIST, "budgets": BUDGETS}


async def _read_image(upload: UploadFile) -> Image.Image:
    limit = int(get_predictor().settings.max_upload_mb * 1024 * 1024)
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"Image larger than {limit // (1024 * 1024)} MB")
    if not data:
        raise HTTPException(400, "Empty upload")
    try:
        img = Image.open(io.BytesIO(data))  # reads the header only
        # Check dimensions before decoding: a tiny compressed file can expand to gigabytes.
        if img.width * img.height > MAX_PIXELS:
            raise HTTPException(413, "Image resolution too large")
        img.load()
    except Image.DecompressionBombError:
        raise HTTPException(413, "Image resolution too large")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "Unsupported or corrupt image file")
    return img


def _rgb(img: Image.Image) -> np.ndarray:
    return np.asarray(ImageOps.exif_transpose(img).convert("RGB"))


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(..., description="Close-up or dermoscopic photo of the lesion"),
    age: float | None = Form(None, ge=0, le=120),
    sex: Literal["male", "female", "unknown"] | None = Form(None),
    localization: str | None = Form(None),
    explain: bool = Form(True),
) -> dict:
    predictor = get_predictor()
    if localization and localization.lower() not in LOCALIZATIONS + ["unknown"]:
        raise HTTPException(422, f"localization must be one of {LOCALIZATIONS}")
    img = await _read_image(image)
    async with _lock:
        return await asyncio.to_thread(predictor.analyze, img, age, sex, localization, explain)



@app.post("/api/lesion/compare")
async def lesion_compare(
    before: UploadFile = File(..., description="Earlier photo of the lesion"),
    after: UploadFile = File(..., description="Recent photo of the same lesion"),
    days_between: int | None = Form(None, ge=1, le=3650),
) -> dict:
    a, b = _rgb(await _read_image(before)), _rgb(await _read_image(after))
    return await asyncio.to_thread(compare_lesions, a, b, days_between)


def _profile(skin_type, concerns, age, sensitive, pregnant, sun_exposure, budget) -> Profile:
    if skin_type and skin_type not in SKIN_TYPES:
        raise HTTPException(422, f"skin_type must be one of {SKIN_TYPES}")
    items = [c.strip() for c in (concerns or "").split(",") if c.strip()] if isinstance(concerns, str) else list(concerns or [])
    bad = sorted(set(items) - set(CONCERN_LIST))
    if bad:
        raise HTTPException(422, f"unknown concerns {bad}; expected {CONCERN_LIST}")
    return Profile(skin_type=skin_type or "normal", concerns=items, age=age, sensitive=bool(sensitive),
                   pregnant=bool(pregnant), sun_exposure=sun_exposure or "moderate", budget=budget or "mid")


@app.post("/api/skin/analyze")
async def skin_analyze(
    image: UploadFile = File(..., description="Front-facing face photo or close-up of a skin area"),
    skin_type: str | None = Form(None),
    concerns: str | None = Form(None, description="comma-separated"),
    age: int | None = Form(None, ge=10, le=120),
    sensitive: bool = Form(False),
    pregnant: bool = Form(False),
    sun_exposure: Literal["low", "moderate", "high"] = Form("moderate"),
    budget: Literal["low", "mid", "high"] = Form("mid"),
) -> dict:
    profile = _profile(skin_type, concerns, age, sensitive, pregnant, sun_exposure, budget)
    rgb = _rgb(await _read_image(image))
    analysis = await asyncio.to_thread(analyze_skin, rgb)
    routine = build_routine(profile, analysis.get("scores") if analysis.get("ok") else None)
    return {"analysis": analysis, "routine": routine}


class RoutineRequest(BaseModel):
    skin_type: str = "normal"
    concerns: list[str] = Field(default_factory=list)
    age: int | None = Field(None, ge=10, le=120)
    sensitive: bool = False
    pregnant: bool = False
    sun_exposure: Literal["low", "moderate", "high"] = "moderate"
    budget: Literal["low", "mid", "high"] = "mid"
    photo_scores: dict[str, int] | None = None


@app.post("/api/skin/routine")
def skin_routine(req: RoutineRequest) -> dict:
    p = _profile(req.skin_type, req.concerns, req.age, req.sensitive, req.pregnant, req.sun_exposure, req.budget)
    return build_routine(p, req.photo_scores)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=40)
    context: dict[str, str | int | float | None] | None = None


@app.post("/api/assistant/chat")
async def assistant_chat(req: ChatRequest) -> dict:
    return await asyncio.to_thread(chat, req.message, [t.model_dump() for t in req.history], req.context)



# ---------------- teledermatology referrals ----------------
# The patient's case secret travels in the X-Case-Token header, never the URL, so it stays out of access logs.

@lru_cache(maxsize=1)
def get_store() -> CaseStore:
    return CaseStore()


@app.post("/api/referrals", status_code=201)
async def create_referral(
    image: UploadFile = File(...),
    consent: bool = Form(..., description="Patient agrees to store the photo and share it with a clinician"),
    age: float | None = Form(None, ge=0, le=120),
    sex: Literal["male", "female", "unknown"] | None = Form(None),
    localization: str | None = Form(None),
    note: str | None = Form(None, max_length=1000),
    contact: str | None = Form(None, max_length=200),
) -> dict:
    if not consent:
        raise HTTPException(400, "Consent is required to send a case to a clinician")
    if localization and localization.lower() not in LOCALIZATIONS + ["unknown"]:
        raise HTTPException(422, f"localization must be one of {LOCALIZATIONS}")
    img = await _read_image(image)
    # Re-run the analysis server-side: the clinician queue must not trust client-supplied results.
    async with _lock:
        result = await asyncio.to_thread(get_predictor().analyze, img, age, sex, localization, False)
    patient = {"age": age, "sex": sex, "localization": localization, "note": note, "contact": contact}
    case_id, token = await asyncio.to_thread(get_store().create, img, summarize_analysis(result), patient)
    return {"case_id": case_id, "access_token": token, "status": "new", "urgency": result["triage"]["level"],
            "message": "Your case was sent. Keep the access code to check the reply or delete your case."}


def _patient_case(case_id: str, token: str) -> None:
    if not get_store().verify(case_id, token):
        raise HTTPException(404, "Case not found")  # same answer for wrong token: don't reveal which ids exist


@app.get("/api/referrals/{case_id}")
def referral_status(case_id: str, token: str = Header(..., alias="X-Case-Token")) -> dict:
    _patient_case(case_id, token)
    return get_store().patient_view(case_id)


@app.delete("/api/referrals/{case_id}")
def delete_referral(case_id: str, token: str = Header(..., alias="X-Case-Token")) -> dict:
    _patient_case(case_id, token)
    get_store().delete(case_id)
    return {"deleted": True}


def clinician(authorization: str | None = Header(None)) -> None:
    expected = os.getenv("DERMAAI_CLINICIAN_TOKEN")
    if not expected:
        raise HTTPException(503, "Clinician access is not configured (set DERMAAI_CLINICIAN_TOKEN)")
    given = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(given.encode(), expected.encode()):
        raise HTTPException(401, "Invalid clinician token", headers={"WWW-Authenticate": "Bearer"})


@app.get("/api/clinician/cases", dependencies=[Depends(clinician)])
def clinician_queue(status: Literal["new", "in_review", "closed"] | None = None) -> dict:
    store = get_store()
    return {"cases": store.queue(status), "stats": store.stats()}


def _case_or_404(case_id: str, with_image: bool = False) -> dict:
    case = get_store().get(case_id, with_image=with_image)
    if not case:
        raise HTTPException(404, "Case not found")
    return case


@app.get("/api/clinician/cases/{case_id}", dependencies=[Depends(clinician)])
def clinician_case(case_id: str) -> dict:
    return _case_or_404(case_id, with_image=True)


class Review(BaseModel):
    status: Literal["new", "in_review", "closed"] = "closed"
    response: str | None = Field(None, max_length=4000, description="Message shown to the patient")
    clinical_impression: str | None = Field(None, max_length=500)
    reviewer: str | None = Field(None, max_length=120)


@app.post("/api/clinician/cases/{case_id}/review", dependencies=[Depends(clinician)])
def clinician_review(case_id: str, review: Review) -> dict:
    _case_or_404(case_id)
    get_store().review(case_id, review.status, review.response, review.clinical_impression, review.reviewer)
    return _case_or_404(case_id)


@app.get("/api/clinician/cases/{case_id}/fhir", dependencies=[Depends(clinician)])
def clinician_fhir(case_id: str) -> dict:
    case = _case_or_404(case_id, with_image=True)
    return to_fhir(case, case.pop("image"))


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/clinician", include_in_schema=False)
    def clinician_page() -> FileResponse:
        return FileResponse(WEB_DIR / "clinician.html")
