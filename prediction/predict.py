"""Predict with the trained model — no training code involved.

Command line:
    python prediction/predict.py photo.jpg
    python prediction/predict.py photo.jpg --age 54 --sex female --site back
    python prediction/predict.py photo.jpg --model model_v1 --json

From Python:
    from prediction.predict import load_model, predict
    model = load_model()                 # the app's current model (models/trained/CURRENT)
    result = predict(model, "photo.jpg")
    print(result["top_class"], result["probabilities"])

Pipeline: load model -> load image -> same preprocessing as training -> predict -> result.
The web app (app.py) uses this same engine (dermaai.inference.Predictor).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dermaai import model_store
from dermaai.config import Settings
from dermaai.inference import Predictor
from dermaai.preprocessing import load_image


def load_model(version: str | None = None, models_dir: str | None = None) -> Predictor:
    """Load a trained version (default: the one the app uses). Never trains anything."""
    path = model_store.model_path(version, models_dir)
    if path is None:
        raise SystemExit(f"No trained model found in {model_store.models_dir(models_dir)}. "
                         "Train one first: python training/train.py")
    return Predictor(Settings(checkpoint=path))


def predict(model: Predictor, image, age=None, sex=None, site=None, explain: bool = False) -> dict:
    """Full analysis of one image (path, file object or PIL image), plus a compact summary on top."""
    result = model.analyze(load_image(image), age=age, sex=sex, localization=site, explain=explain)
    probs = {p["code"]: p["probability"] for p in result["prediction"]["probabilities"]}
    return {
        "top_class": result["prediction"]["top"]["code"],
        "top_name": result["prediction"]["top"]["name"],
        "confidence": result["prediction"]["top"]["probability"],
        "probabilities": probs,
        "triage": result["triage"]["level"],
        "model_version": model.version,
        "details": result,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Classify a skin-lesion photo with the trained model.")
    p.add_argument("images", nargs="+", help="one or more image files")
    p.add_argument("--model", default=None, help="version to use (default: the app's current model)")
    p.add_argument("--age", type=float)
    p.add_argument("--sex", choices=["male", "female"])
    p.add_argument("--site", help="body site, e.g. back, face, lower extremity")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    a = p.parse_args(argv)

    model = load_model(a.model)
    print(f"model: {model.version or 'explicit checkpoint'} ({model.info['arch']}, classes: {', '.join(model.classes)})\n")
    for img in a.images:
        r = predict(model, img, a.age, a.sex, a.site)
        if a.json:
            print(json.dumps({k: v for k, v in r.items() if k != "details"} | {"triage_detail": r["details"]["triage"]}, indent=2))
            continue
        print(f"{img}")
        print(f"  prediction: {r['top_name']} ({r['confidence']:.1%})   triage: {r['triage']}")
        for code, prob in list(r["probabilities"].items())[:4]:
            print(f"    {code:<8} {prob:6.1%}")
        if r["details"]["quality"]["issues"]:
            print("  photo issues: " + "; ".join(i["message"] for i in r["details"]["quality"]["issues"]))


if __name__ == "__main__":
    main()
