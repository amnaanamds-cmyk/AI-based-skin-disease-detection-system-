"""FastAPI service. Run with: uvicorn dermaai.api.main:app --host 0.0.0.0 --port 8000"""

from __future__ import annotations

import asyncio
import io
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from .. import __version__
from ..config import CLASSES, CONDITIONS, LOCALIZATIONS, SEXES
from ..inference import DISCLAIMER, Predictor

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(
    title="DermaAI API",
    version=__version__,
    description="Explainable AI triage for skin lesions. Images are processed in memory and never stored.",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

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
    return {"sex": SEXES, "localization": LOCALIZATIONS}


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(..., description="Close-up or dermoscopic photo of the lesion"),
    age: float | None = Form(None, ge=0, le=120),
    sex: Literal["male", "female", "unknown"] | None = Form(None),
    localization: str | None = Form(None),
    explain: bool = Form(True),
) -> dict:
    predictor = get_predictor()
    limit = int(predictor.settings.max_upload_mb * 1024 * 1024)
    data = await image.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"Image larger than {predictor.settings.max_upload_mb:g} MB")
    if not data:
        raise HTTPException(400, "Empty upload")
    if localization and localization.lower() not in LOCALIZATIONS + ["unknown"]:
        raise HTTPException(422, f"localization must be one of {LOCALIZATIONS}")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "Unsupported or corrupt image file")
    if img.width * img.height > 40_000_000:
        raise HTTPException(413, "Image resolution too large")
    async with _lock:
        return await asyncio.to_thread(predictor.analyze, img, age, sex, localization, explain)


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")
