# DermaAI — Explainable AI Skin Lesion Triage

DermaAI checks a photo of a mole or skin lesion and answers the question patients actually have:
**"How soon should I see a dermatologist?"** It does not stop at a label. Each check runs a quality gate,
gives calibrated probabilities for 7 conditions, reports how uncertain the model is, shows where the network
looked (Grad-CAM), runs an independent ABCDE analysis, and ends with a triage level biased toward sensitivity.

> ⚠️ DermaAI supports decisions and education. It is **not** a medical device and does not give a diagnosis.

![pipeline](docs/pipeline.svg)

## Why it stands out

| Most hackathon skin-AI demos | DermaAI |
|---|---|
| Single softmax label | Calibrated probabilities (temperature scaling) + an uncertainty score from TTA and entropy |
| Accepts any photo | **Quality gate** for blur, exposure, glare, resolution and framing. Bad photos get "retake" with instructions |
| Black box | **Grad-CAM attention map** plus a **classical ABCDE analysis** that clinicians already use |
| Accuracy on a leaky random split | **Lesion-grouped splits** (HAM10000 has several photos of the same lesion) and model selection on balanced accuracy + melanoma AUC |
| Image only | **Fusion of image and metadata** (age, sex, body site) |
| Argmax output | **Safety-first triage** that flags high melanoma risk from a 15% melanoma probability |
| Server keeps photos | Photos are processed **in memory only**. History stays **on the device** |
| Notebook | Production REST API, responsive web app, Docker, ONNX export for mobile/edge, CI tests |

## Conditions (HAM10000 / ISIC 2018 taxonomy)

| Code | Condition | Class |
|---|---|---|
| `mel` | Melanoma | malignant |
| `bcc` | Basal cell carcinoma | malignant |
| `akiec` | Actinic keratosis / intraepithelial carcinoma | pre-malignant |
| `nv` | Melanocytic nevus | benign |
| `bkl` | Benign keratosis | benign |
| `df` | Dermatofibroma | benign |
| `vasc` | Vascular lesion | benign |

## Quick start

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision   # or a CUDA build
pip install -r requirements-dev.txt

# Run the app. With no checkpoint it starts in clearly-labelled DEMO mode.
uvicorn dermaai.api.main:app --reload
# open http://localhost:8000   ·   API docs at http://localhost:8000/docs
```

Docker:

```bash
docker compose up --build        # serves ./models/dermaai.pt if present
```

## Training a real model

1. Download **HAM10000** (Harvard Dataverse, doi:10.7910/DVN/DBW86T) into `data/`:
   `HAM10000_metadata.csv`, `HAM10000_images_part_1/`, `HAM10000_images_part_2/`.
2. Train (a GPU is strongly recommended):

```bash
python -m dermaai.train \
  --csv data/HAM10000_metadata.csv \
  --images data/HAM10000_images_part_1 data/HAM10000_images_part_2 \
  --arch efficientnet_b3 --img-size 300 --epochs 30 --batch-size 32 --pretrained \
  --out models
```

This produces `models/dermaai.pt` (weights, calibration temperature, test metrics) and `models/metrics.json`.

3. Serve it with `DERMAAI_CHECKPOINT=models/dermaai.pt uvicorn dermaai.api.main:app`.
4. Optional: `python -m dermaai.evaluate --checkpoint models/dermaai.pt --csv external.csv --images ext/`
   to validate on external data (ISIC 2019, PAD-UFES-20, Derm7pt), and
   `python -m dermaai.export --checkpoint models/dermaai.pt --out models/dermaai.onnx` for mobile.

Any CSV with columns `image`/`image_id`, `dx` and optionally `lesion_id`, `age`, `sex`, `localization` works.

**What the training pipeline does:** stratified, lesion-grouped train/val/test split · square-root class-balanced
sampling plus a mild class-weighted loss · heavy dermoscopy augmentation (rotation, flips, colour jitter, blur,
random erasing) · AdamW with a 10× learning rate on the head, warmup and cosine decay · AMP on GPU · early stopping
on `balanced_accuracy + 0.5 × melanoma_AUC` · flip-TTA evaluation · temperature calibration on the validation set.

**Reported metrics:** accuracy, balanced accuracy, macro-F1, macro AUC, per-class recall, confusion matrix,
ECE (calibration), malignant-vs-benign AUC, melanoma AUC, and **specificity at 90% melanoma sensitivity**,
the operating point that matters for screening.

> No figures are claimed here: the bundled code has not been trained on real data in this repository.
> For reference, ISIC 2018 Task 3 leaders reached ≈0.88 balanced accuracy with large ensembles and external
> data. Report your own numbers from `models/metrics.json` on the lesion-grouped test split.

## API

`POST /api/analyze` (multipart): `image` (required), `age`, `sex` (`male|female|unknown`), `localization`, `explain`.

```jsonc
{
  "triage":      { "level": "high|moderate|low|retake", "title": "...", "reasons": ["..."],
                   "malignancy_probability": 0.88, "melanoma_probability": 0.87 },
  "prediction":  { "top": { "code": "mel", "name": "Melanoma", "probability": 0.87, ... },
                   "probabilities": [ ... sorted ... ] },
  "uncertainty": { "score": 0.27, "entropy": 0.3, "tta_std": 0.02, "level": "low" },
  "quality":     { "score": 100, "acceptable": true, "issues": [], "measurements": { ... } },
  "abcde":       { "asymmetry": 0.24, "border_irregularity": 0.3, "color_variegation": 0.25,
                   "colors_detected": ["blue-gray", "black"], "suspicion_score": 0.27, "outline": [[x, y], ...] },
  "explanation": { "heatmap": "data:image/jpeg;base64,...", "analyzed_view": "...", "target_class": "mel" },
  "model": { "arch": "efficientnet_b3", "version": "1.0.0", "demo_mode": false },
  "disclaimer": "..."
}
```

Other endpoints: `GET /api/health`, `/api/model` (model card + test metrics), `/api/conditions`, `/api/options`.

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `DERMAAI_CHECKPOINT` | none | Path to trained `.pt`. Unset or missing means demo mode |
| `DERMAAI_DEVICE` | `auto` | `cuda`, `mps` or `cpu` |
| `DERMAAI_TTA` | `1` | 5-view test-time augmentation |
| `DERMAAI_MEL_THRESHOLD` | `0.15` | Melanoma probability that triggers urgent referral |
| `DERMAAI_MALIGNANT_HIGH` / `_MODERATE` | `0.40` / `0.15` | Combined malignant probability thresholds |
| `DERMAAI_UNCERTAINTY` | `0.75` | Uncertainty above which a "low" result is upgraded to "moderate" |
| `DERMAAI_MAX_UPLOAD_MB` | `10` | Upload size limit |

Tune the thresholds on your validation set to reach the sensitivity you need.

## Project layout

```
dermaai/
  config.py      taxonomy, clinical knowledge base, settings
  model.py       DermNet: timm CNN backbone fused with metadata MLP
  data.py        CSV loading, lesion-grouped splits, balanced sampling
  transforms.py  augmentation
  train.py       training CLI          evaluate.py  evaluation CLI       export.py  ONNX export
  metrics.py     clinical metrics, ECE, temperature scaling
  quality.py     image quality gate    abcde.py     segmentation + ABCDE  explain.py Grad-CAM
  inference.py   end-to-end analysis and triage
  api/main.py    FastAPI service
web/             responsive single-page app (upload/camera, results, history, print report)
scripts/         synthetic dataset generator (pipeline smoke tests only)
tests/           unit, API and end-to-end train→serve tests
docs/PITCH.md    competition pitch, impact and business model
```

## Testing

```bash
pytest -q        # 15 tests: vision, metrics, API, and a full synthetic train → calibrate → serve run
```

## Responsible AI

- **Skin-tone bias.** HAM10000 is mostly lighter skin. Before deployment, evaluate per Fitzpatrick type
  (Fitzpatrick17k, DDI) and fine-tune on diverse data such as PAD-UFES-20. The quality gate's skin detector
  uses a broad colour range so darker skin is not rejected.
- **Out-of-distribution input.** Non-lesion photos are flagged by the quality gate and by high uncertainty,
  but this is not a guarantee. Clinical photos differ from dermoscopy, so fine-tune for smartphone use.
- **Over-referral by design.** Thresholds trade specificity for melanoma sensitivity.
- **Privacy.** Images are never written to disk server-side. History lives in the browser's localStorage.
- **Regulation.** Clinical use needs regulatory clearance (e.g. EU MDR class IIa, FDA 510(k)/De Novo) and
  prospective validation.
