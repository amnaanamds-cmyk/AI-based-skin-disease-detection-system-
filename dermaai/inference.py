"""End-to-end analysis: quality gate → calibrated TTA prediction → triage → explanation."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

import numpy as np
import torch
from PIL import Image, ImageOps

from . import __version__
from .abcde import compute_abcde
from .config import CLASSES, CONDITIONS, IMAGENET_MEAN, IMAGENET_STD, MALIGNANT, Settings, get_settings
from .explain import grad_cam, overlay, to_data_url
from .metadata import encode_metadata
from .metrics import mahalanobis_score
from .model import DermNet, load_checkpoint
from .quality import assess_quality
from .train import pick_device
from .transforms import eval_transform

DISCLAIMER = (
    "DermaAI is a decision-support and education tool, not a medical diagnosis. "
    "It can miss cancers. Always consult a qualified dermatologist about any lesion that "
    "is new, changing, bleeding, itching or worrying you."
)

TRIAGE_TEXT = {
    "high": ("See a dermatologist urgently",
             "Features associated with skin cancer were detected. Book an appointment within days, not weeks."),
    "moderate": ("Get it checked soon",
                 "Some concerning features were found. Arrange a dermatologist review within 2-4 weeks."),
    "low": ("Low concern — keep monitoring",
            "Findings are consistent with a benign lesion. Re-photograph monthly and watch for changes."),
    "retake": ("Please retake the photo",
               "The image quality is too poor for a reliable analysis."),
    "unfamiliar": ("This doesn't look like a typical skin-lesion photo",
                   "The image is unlike the photos the AI was trained on, so its answer would not be reliable. "
                   "Photograph a single spot close-up, in focus and in daylight. If you are worried about it, see a doctor."),
}


class Predictor:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.device = pick_device(self.settings.device)
        ckpt = self.settings.checkpoint
        if ckpt and ckpt.exists():
            self.model, info = load_checkpoint(ckpt, map_location="cpu")
            self.demo_mode = False
        else:
            # Without trained weights the API still runs so the product can be demoed,
            # but every response is clearly flagged as coming from an untrained model.
            torch.manual_seed(0)
            self.model = DermNet(self.settings.demo_arch, len(CLASSES)).eval()
            info = {"arch": self.settings.demo_arch, "img_size": 224, "temperature": 1.0}
            self.demo_mode = True
        self.model.to(self.device)
        self.classes: list[str] = list(info.get("classes", CLASSES))
        self.img_size = int(info.get("img_size", 224))
        self.temperature = float(info.get("temperature") or 1.0)
        self.ood = info.get("ood")  # {"means", "precision", "threshold"} or None
        self.info = {
            "name": "DermaAI",
            "version": info.get("version", __version__),
            "arch": info["arch"],
            "img_size": self.img_size,
            "temperature": round(self.temperature, 4),
            "ood_detection": self.ood is not None,
            "classes": self.classes,
            "demo_mode": self.demo_mode,
            "trained_at": info.get("trained_at"),
            "metrics": {k: v for k, v in (info.get("metrics") or {}).items() if k != "confusion_matrix"},
        }
        self.transform = eval_transform(self.img_size)

    @torch.no_grad()
    def _predict(self, x: torch.Tensor, meta: torch.Tensor) -> tuple[np.ndarray, float]:
        views = [x]
        if self.settings.tta:
            views += [x.flip(3), x.flip(2), x.flip(2).flip(3), x.transpose(2, 3)]
        batch = torch.cat(views)
        logits = self.model(batch, meta.expand(len(views), -1))
        probs = torch.softmax(logits / self.temperature, dim=1).cpu().numpy()
        mean = probs.mean(0)
        return mean, float(probs[:, mean.argmax()].std())

    @torch.no_grad()
    def _ood_check(self, x: torch.Tensor) -> dict | None:
        if not self.ood:
            return None
        score = float(mahalanobis_score(self.model.embed(x).float(), self.ood).item())
        thr = float(self.ood["threshold"])
        # "far": so unlike any lesion (noise, objects, screenshots) that even an urgent answer is meaningless.
        return {"score": round(score, 2), "threshold": round(thr, 2), "unfamiliar": score > thr, "far": score > 3 * thr}

    def _triage(self, probs: dict[str, float], uncertainty: float, quality_ok: bool, abcde: dict | None,
                unfamiliar: bool | str = False) -> dict:
        s = self.settings
        p_mel = probs.get("mel", 0.0)
        p_mal = sum(probs.get(c, 0.0) for c in MALIGNANT)
        reasons: list[str] = []
        if not quality_ok:
            level = "retake"
        elif p_mel >= s.mel_urgent_threshold or p_mal >= s.malignant_high_threshold:
            level = "high"
            if p_mel >= s.mel_urgent_threshold:
                reasons.append(f"Melanoma probability {p_mel:.0%} exceeds the screening threshold of {s.mel_urgent_threshold:.0%}.")
            if p_mal >= s.malignant_high_threshold:
                reasons.append(f"Combined probability of a malignant or pre-malignant lesion is {p_mal:.0%}.")
        elif p_mal >= s.malignant_moderate_threshold:
            level = "moderate"
            reasons.append(f"Combined probability of a malignant or pre-malignant lesion is {p_mal:.0%}.")
        else:
            level = "low"
        if level == "low" and uncertainty >= s.uncertainty_threshold:
            level = "moderate"
            reasons.append("The model is uncertain about this image, so a professional review is recommended.")
        if level == "low" and abcde and abcde["suspicion_score"] >= 0.6:
            level = "moderate"
            reasons.append("The ABCDE analysis found marked asymmetry, irregular border or multiple colours.")
        text_key = level
        if (unfamiliar and level in {"low", "moderate"}) or unfamiliar == "far":
            # A merely unusual image keeps an urgent result (an atypical cancer must still be flagged);
            # only images far outside the training distribution are sent back for a retake.
            level = "retake"
            text_key = "unfamiliar"
            reasons = ["Out-of-distribution check: the image's features are far from every lesion type seen in training."]
        elif unfamiliar:
            reasons.append("The image is unusual for the model; the urgent recommendation stands as a precaution.")
        if self.demo_mode:
            reasons.insert(0, "DEMO MODE: no trained model is loaded, so this result is not meaningful.")
        title, message = TRIAGE_TEXT[text_key]
        return {"level": level, "title": title, "message": message, "reasons": reasons,
                "malignancy_probability": round(p_mal, 4), "melanoma_probability": round(p_mel, 4)}

    def analyze(self, image: Image.Image, age=None, sex=None, localization=None, explain: bool = True) -> dict:
        t0 = time.perf_counter()
        image = ImageOps.exif_transpose(image).convert("RGB")
        rgb = np.asarray(image)
        quality = assess_quality(rgb)

        x = self.transform(image).unsqueeze(0).to(self.device)
        meta = encode_metadata(age, sex, localization).unsqueeze(0).to(self.device)
        probs, tta_std = self._predict(x, meta)
        ood = self._ood_check(x)
        unfamiliar = "far" if ood and ood["far"] else bool(ood and ood["unfamiliar"])
        entropy = float(-(probs * np.log(probs + 1e-12)).sum() / np.log(len(probs)))
        uncertainty = float(np.clip(0.8 * entropy + 2.0 * tta_std, 0, 1))
        prob_map = {c: float(p) for c, p in zip(self.classes, probs)}
        ranked = sorted(prob_map.items(), key=lambda kv: kv[1], reverse=True)

        abcde = compute_abcde(rgb)
        triage = self._triage(prob_map, uncertainty, quality["acceptable"], abcde, unfamiliar)

        explanation = None
        if explain:
            top_idx = self.classes.index(ranked[0][0])
            cam = grad_cam(self.model, x, meta, top_idx)
            view = _denormalize(x[0])
            explanation = {
                "target_class": ranked[0][0],
                "heatmap": to_data_url(overlay(view, cam)),
                "analyzed_view": to_data_url(view),
                "attention_focus": round(float((cam > 0.5).mean()), 3),
            }

        def entry(code: str, p: float) -> dict:
            c = CONDITIONS[code]
            return {"code": code, "name": c["name"], "probability": round(p, 4),
                    "malignancy": c["malignancy"], "risk": c["risk"]}

        top = ranked[0][0]
        return {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": {k: self.info[k] for k in ("name", "version", "arch", "demo_mode")},
            "quality": quality,
            "prediction": {
                "top": {**entry(top, prob_map[top]), "summary": CONDITIONS[top]["summary"],
                        "action": CONDITIONS[top]["action"]},
                "probabilities": [entry(c, p) for c, p in ranked],
            },
            "uncertainty": {
                "score": round(uncertainty, 3),
                "entropy": round(entropy, 3),
                "tta_std": round(tta_std, 4),
                "level": "high" if uncertainty >= self.settings.uncertainty_threshold
                         else "medium" if uncertainty >= 0.45 else "low",
            },
            "triage": triage,
            "ood": ood,
            "abcde": abcde,
            "explanation": explanation,
            "disclaimer": DISCLAIMER,
            "processing_ms": int((time.perf_counter() - t0) * 1000),
        }


def _denormalize(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, device=t.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=t.device).view(3, 1, 1)
    img = (t * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)
