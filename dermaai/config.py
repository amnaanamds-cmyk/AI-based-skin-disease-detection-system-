"""Class taxonomy, clinical knowledge base and runtime settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ISIC 2019 diagnostic categories: the HAM10000 / ISIC 2018 seven plus squamous cell carcinoma.
# Order matters (index = model output); "scc" is appended so 7-class checkpoints keep their indices.
CLASSES: list[str] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc", "scc"]

MALIGNANT: frozenset[str] = frozenset({"mel", "bcc", "akiec", "scc"})

CONDITIONS: dict[str, dict] = {
    "akiec": {
        "name": "Actinic keratosis / Intraepithelial carcinoma",
        "malignancy": "pre-malignant",
        "risk": "moderate",
        "summary": "Rough, scaly patch caused by sun damage. Can progress to squamous cell carcinoma.",
        "action": "Book a dermatologist appointment within 2-4 weeks for assessment and treatment.",
    },
    "bcc": {
        "name": "Basal cell carcinoma",
        "malignancy": "malignant",
        "risk": "high",
        "summary": "The most common skin cancer. Slow growing and rarely spreads, but locally destructive.",
        "action": "See a dermatologist within 2 weeks. Treatment is highly effective when started early.",
    },
    "bkl": {
        "name": "Benign keratosis",
        "malignancy": "benign",
        "risk": "low",
        "summary": "Seborrheic keratosis, solar lentigo or lichen-planus-like keratosis. Harmless, common with age.",
        "action": "No urgent action. Monitor for changes and mention it at your next routine check-up.",
    },
    "df": {
        "name": "Dermatofibroma",
        "malignancy": "benign",
        "risk": "low",
        "summary": "Firm, benign nodule of fibrous tissue, often on the legs.",
        "action": "No urgent action. See a doctor if it grows, bleeds or becomes painful.",
    },
    "mel": {
        "name": "Melanoma",
        "malignancy": "malignant",
        "risk": "high",
        "summary": "The most dangerous skin cancer. Survival exceeds 99% when caught early.",
        "action": "Seek a dermatologist urgently (within days). Do not wait for the lesion to change.",
    },
    "nv": {
        "name": "Melanocytic nevus (mole)",
        "malignancy": "benign",
        "risk": "low",
        "summary": "A common mole. Most people have 10-40.",
        "action": "No urgent action. Re-check monthly using the ABCDE rule and photograph for comparison.",
    },
    "scc": {
        "name": "Squamous cell carcinoma",
        "malignancy": "malignant",
        "risk": "high",
        "summary": "The second most common skin cancer: a scaly, crusted or wart-like growth, often on sun-exposed skin. Can spread if untreated.",
        "action": "See a dermatologist within 2 weeks. Early treatment is usually curative.",
    },
    "vasc": {
        "name": "Vascular lesion",
        "malignancy": "benign",
        "risk": "low",
        "summary": "Cherry angioma, angiokeratoma, pyogenic granuloma or haemorrhage.",
        "action": "Usually harmless. See a doctor if it bleeds repeatedly or grows quickly.",
    },
}

def condition_info(code: str, custom: dict | None = None) -> dict:
    """Display info for a class. Custom classes (added by training on new data) carry their own
    info inside the model file; unknown ones get a neutral description instead of crashing."""
    if custom and code in custom:
        base = {"name": code.replace("_", " ").title(), "malignancy": "unknown", "risk": "moderate",
                "summary": "", "action": "Ask a dermatologist to review this lesion."}
        return {**base, **CONDITIONS.get(code, {}), **custom[code]}
    if code in CONDITIONS:
        return CONDITIONS[code]
    return {"name": code.replace("_", " ").title(), "malignancy": "unknown", "risk": "moderate",
            "summary": "A condition added by a custom-trained model.",
            "action": "Ask a dermatologist to review this lesion."}


SEXES: list[str] = ["male", "female"]

LOCALIZATIONS: list[str] = [
    "abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital",
    "hand", "lower extremity", "neck", "scalp", "trunk", "upper extremity",
]

# age (value, missing flag) + sex one-hot (+unknown) + localization one-hot (+unknown)
META_DIM: int = 2 + (len(SEXES) + 1) + (len(LOCALIZATIONS) + 1)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Settings:
    # Explicit model file. Leave unset to serve models/trained/<CURRENT> (see dermaai.model_store).
    checkpoint: Path | None = field(
        default_factory=lambda: Path(p) if (p := os.getenv("DERMAAI_CHECKPOINT")) else None
    )
    demo_arch: str = os.getenv("DERMAAI_DEMO_ARCH", "efficientnet_b0")
    tta: bool = os.getenv("DERMAAI_TTA", "1") == "1"
    max_upload_mb: float = float(os.getenv("DERMAAI_MAX_UPLOAD_MB", "10"))
    device: str = os.getenv("DERMAAI_DEVICE", "auto")
    # Triage thresholds, tuned for high melanoma sensitivity (screening setting).
    mel_urgent_threshold: float = float(os.getenv("DERMAAI_MEL_THRESHOLD", "0.15"))
    malignant_high_threshold: float = float(os.getenv("DERMAAI_MALIGNANT_HIGH", "0.40"))
    malignant_moderate_threshold: float = float(os.getenv("DERMAAI_MALIGNANT_MODERATE", "0.15"))
    uncertainty_threshold: float = float(os.getenv("DERMAAI_UNCERTAINTY", "0.75"))


def get_settings() -> Settings:
    return Settings()
