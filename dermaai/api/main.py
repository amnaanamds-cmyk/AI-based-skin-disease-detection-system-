"""FastAPI service. Run with: uvicorn dermaai.api.main:app --host 0.0.0.0 --port 8000"""

from __future__ import annotations

import asyncio
import io
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from .. import __version__
from ..config import CLASSES, CONDITIONS, LOCALIZATIONS, SEXES
from ..assistant import chat
from ..inference import DISCLAIMER, Predictor
from ..progress import compare_lesions
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


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()


@app.get("/api/health")
def health() -> dict:
    p = get_predictor()
    return {"status": "ok", "version": __version__, "demo_mode": p.demo_mode}


@app.get("/api/model")
def model_info() -> dict:
    return get_predictor().info


@app.get("/api/conditions")
def conditions() -> dict:
    return {"conditions": [{"code": c, **CONDITIONS[c]} for c in CLASSES], "disclaimer": DISCLAIMER}


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


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")
