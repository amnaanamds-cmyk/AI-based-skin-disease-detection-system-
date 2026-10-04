# DermaAI — Explainable AI Skin Health Platform

DermaAI combines these services in one app:

| Service | What it does |
|---|---|
| **AI lesion check** | Skin-cancer triage for a mole or spot: calibrated probabilities, Grad-CAM, ABCDE and how soon to see a doctor |
| **AI skin analysis** | A face or skin photo is measured for redness, blemishes, uneven tone, texture and shine, giving a skin health score |
| **Personalised skincare** | A rules engine builds an AM/PM routine from skin type, concerns, age, pregnancy, sensitivity and the photo |
| **Smart product recommendations** | Each step is matched to product types by key ingredients, skin type, concerns and budget, with the reasons shown |
| **Progress tracking** | Before/after comparison of a spot (size, shape, border, new colours, growth per month) and a skin score trend |
| **Virtual dermatology assistant** | Chat with red-flag escalation, a built-in knowledge base, and optional Claude-powered answers grounded in your results |
| **Live camera photo coach** | Real-time focus, light, glare and framing feedback in the browser, with auto-capture once the frame is good |
| **Body map (mole mapping)** | Pin spots on a front/back body diagram; each spot keeps a timeline of checks and can compare its last two photos |
| **Teledermatology referral** | With consent, send a case to a dermatologist; clinicians work an urgency-sorted queue at `/clinician` and reply to the patient |
| **EHR export** | Each case exports as an HL7 FHIR R4 bundle (DiagnosticReport, Observations, Media) |
| **Out-of-distribution detection** | Photos unlike the training data (not a lesion, wrong body part) are recognised instead of being guessed at |

The lesion check is the core. It answers the question patients actually have:
**"How soon should I see a dermatologist?"** It does not stop at a label. Each check runs a quality gate,
gives calibrated probabilities for 7 conditions, reports how uncertain the model is, shows where the network
looked (Grad-CAM), runs an independent ABCDE analysis, and ends with a triage level biased toward sensitivity.

> ⚠️ DermaAI supports decisions and education. It is **not** a medical device and does not give a diagnosis.

![pipeline](docs/pipeline.svg)

## Why it stands out

| Most hackathon skin-AI demos | DermaAI |
|---|---|
| Single softmax label | Calibrated probabilities (temperature scaling) + an uncertainty score from TTA and entropy |
| Accepts any photo | **Quality gate** for blur, exposure, glare, resolution and framing, a **live camera coach**, and **out-of-distribution detection** on deep features |
| Black box | **Grad-CAM attention map** plus a **classical ABCDE analysis** that clinicians already use |
| Accuracy on a leaky random split | **Lesion-grouped splits** (HAM10000 has several photos of the same lesion) and model selection on balanced accuracy + melanoma AUC |
| Image only | **Fusion of image and metadata** (age, sex, body site) |
| Argmax output | **Safety-first triage** that flags high melanoma risk from a 15% melanoma probability |
| Server keeps photos | Photos are processed **in memory only**; they are stored only when the patient consents to a referral, and the patient can delete the case |
| Stops at a prediction | **Closes the loop**: body-map follow-up, a clinician queue sorted by AI urgency, and FHIR export to the hospital record |
| Notebook | Production REST API, responsive web app, Docker, ONNX export for mobile/edge, CI tests |

## Conditions (ISIC 2019 taxonomy)

| Code | Condition | Class |
|---|---|---|
| `mel` | Melanoma | malignant |
| `bcc` | Basal cell carcinoma | malignant |
| `akiec` | Actinic keratosis / intraepithelial carcinoma | pre-malignant |
| `nv` | Melanocytic nevus | benign |
| `bkl` | Benign keratosis | benign |
| `df` | Dermatofibroma | benign |
| `vasc` | Vascular lesion | benign |
| `scc` | Squamous cell carcinoma | malignant |

## Quick start

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision   # or a CUDA build
pip install -r requirements.txt

python app.py                    # open http://localhost:8000  (API docs at /docs)
```

The app loads the trained model named in `models/trained/CURRENT`. If there is no trained model, it runs in a
clearly labelled demo mode.

Docker: `docker compose up --build` (mounts `./models` read-only).

## How the AI is organised

```
data/raw/  ──►  training/train.py  ──►  models/trained/model_vN/  ──►  app.py / prediction/predict.py
(your data)     (the only place         (versioned; CURRENT names       (load the model, preprocess the
                 that trains)             the one the app serves)          same way, predict; never train)
```

| Folder | Purpose | Guide |
|---|---|---|
| `data/` | datasets: `raw/` (yours), `processed/` (automatic cache), `splits.csv` (stable train/val/test) | [data/README.md](data/README.md) |
| `training/` | training module: `config.py` (all settings), `train.py`, `dataset.py`, `preprocessing.py`, `calibration.py`, `evaluate.py`, `manage_models.py` | [training/README.md](training/README.md) |
| `models/` | `trained/model_v1`, `model_v2`, … plus `CURRENT` | [models/README.md](models/README.md) |
| `prediction/` | `predict.py`: load the current model and classify images (CLI or Python) | below |
| `dermaai/` | the app (API, web UI, skin care, assistant, referrals) and the code shared with training: model architecture (`model.py`), image preprocessing (`preprocessing.py`), model store (`model_store.py`) | |
| `app.py` | starts the web app | |

### Train or retrain

```bash
python scripts/prepare_isic2019.py              # optional: download ISIC 2019 into data/raw (≈550 MB, ≈20 min)
python training/train.py                        # train on data/raw -> models/trained/model_vN
python training/train.py --init-from current    # retrain by fine-tuning the current model on more data
python training/manage_models.py list           # versions and scores; * = used by the app
python training/manage_models.py use model_v1   # roll back
```

To train on your own data, put it in `data/raw/` as described in [data/README.md](data/README.md) (one folder per
class, or `labels.csv`) and run `python training/train.py`. Preprocessing, splitting, training, calibration,
evaluation, versioned saving and switching the app to the new model all happen automatically. `app.py` never needs
editing.

### Predict without the web app

```bash
python prediction/predict.py photo.jpg --age 54 --sex female --site back
```

```python
from prediction.predict import load_model, predict
model = load_model()                         # current version
print(predict(model, "photo.jpg")["top_class"])
```

### Bundled model

Two versions were trained in this repository on **ISIC 2019**: 25,327 usable dermoscopy images, 8 classes, from
Vienna, Barcelona and New York. Both use EfficientNet-B0 at 192 px on 4 CPU cores and are scored on the same 3,619
held-out test images, split by lesion so no lesion appears in both training and test:

| Metric | model_v1 | model_v2 (current) |
|---|---|---|
| How it was trained | ImageNet weights, 7 epochs (best: 3) | fine-tuned from v1 with `--init-from current`, 3 epochs, LR 5e-5 |
| Balanced accuracy (8 classes; chance = 0.125) | 0.616 | **0.634** |
| Accuracy | 0.745 | 0.744 |
| Macro F1 | 0.584 | **0.605** |
| Macro AUC / malignant AUC / melanoma AUC | 0.927 / 0.906 / 0.883 | 0.926 / **0.908** / **0.896** |
| Calibration error (ECE) | 0.021 | 0.021 |
| Recall: mel / bcc / scc / akiec | 0.57 / 0.74 / 0.53 / 0.28 | **0.66** / 0.74 / 0.39 / **0.53** |
| Recall: nv / bkl / df / vasc | 0.89 / 0.57 / 0.59 / 0.76 | 0.83 / **0.62** / 0.56 / 0.73 |

v2 was promoted automatically because it beat v1 on the same test images. SCC recall dropped, partly perhaps
because this run's fine-tuning started with the SCC and vascular outputs swapped (a class-order bug, since fixed and
covered by a test; see `notes` in `model_v2/model.json`). If SCC matters most for your use, roll back with
`python training/manage_models.py use model_v1`.

These are honest but **modest** numbers, limited by CPU-only training at low resolution (192 px, one small model).
For context, the top ISIC 2019 challenge entries scored roughly 0.6 balanced accuracy on the official test set (which
also contains an "unknown" class), using GPU-trained ensembles at much higher resolution. These numbers are not directly
comparable, but a GPU retrain with `--model-type efficientnet_b2 --img-size 260` should improve results. Weak spots are actinic keratosis (often confused with BCC and SCC) and melanoma vs. nevus. Open
`models/trained/model_v1/report.html` for the full confusion matrix.

**Licence:** ISIC 2019 is CC BY-NC 4.0, so this model is for **non-commercial use**. A commercial product needs
training data licensed for commercial use. Retrain with `python training/train.py` once you have it.

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

### Skin care, tracking and assistant

| Endpoint | Input | Output |
|---|---|---|
| `POST /api/skin/analyze` | multipart `image` + `skin_type`, `concerns` (comma-separated), `age`, `sensitive`, `pregnant`, `sun_exposure`, `budget` | `analysis` (health score, 5 concern scores, visualisation) + `routine` |
| `POST /api/skin/routine` | JSON profile, optional `photo_scores` | AM/PM routine with matched products and tips (no photo needed) |
| `POST /api/lesion/compare` | multipart `before`, `after`, optional `days_between` | change level, findings, deltas, monthly growth, outlines |
| `POST /api/assistant/chat` | JSON `message`, `history`, optional `context` (latest results) | `reply`, `escalation` (urgent/emergency), `sources`, `engine` |

**Assistant engines.** Every message first passes a deterministic red-flag check (allergic reaction, a rash with
fever, cellulitis, a changing or bleeding mole, a sore that won't heal). The built-in knowledge base always works
offline. When Claude API credentials are present (`ANTHROPIC_API_KEY`, or `DERMAAI_ASSISTANT_LLM=1` with an
`ant auth login` profile), replies are written by Claude (`claude-opus-5-5`, low effort, server-side refusal
fallback), grounded in the retrieved notes and the user's latest results. If the API fails, it falls back to the
knowledge base. Set `DERMAAI_ASSISTANT_LLM=0` to force offline mode.

**Skin analysis limits.** These are classical image measurements on detected skin pixels, compared against the
person's own skin tone so darker skin is not penalised. They are cosmetic estimates that depend on lighting and
camera, so they are best used for tracking change in photos taken in the same conditions. No face-landmark model is
bundled, so remove glasses and keep hair off the face.

### Teledermatology referrals and clinician queue

| Endpoint | Auth | Purpose |
|---|---|---|
| `POST /api/referrals` | none (requires `consent=true`) | Store a case. The server re-runs the analysis and returns `case_id` + a one-time `access_token` |
| `GET /api/referrals/{id}` | `X-Case-Token` header | Patient checks status and the clinician's reply |
| `DELETE /api/referrals/{id}` | `X-Case-Token` header | Patient deletes the case and photo (right to erasure) |
| `GET /api/clinician/cases?status=` | `Authorization: Bearer $DERMAAI_CLINICIAN_TOKEN` | Queue sorted by urgency, malignancy probability, then age, plus counts |
| `GET /api/clinician/cases/{id}` | clinician | Full case with photo, patient details and AI analysis |
| `POST /api/clinician/cases/{id}/review` | clinician | Set status, private clinical impression, and the message shown to the patient |
| `GET /api/clinician/cases/{id}/fhir` | clinician | HL7 FHIR R4 `Bundle` for import into an EHR |

The dashboard is at `/clinician`. Clinician endpoints return 503 until `DERMAAI_CLINICIAN_TOKEN` is set. Only a
SHA-256 hash of each patient token is stored, tokens travel in headers (not URLs, so they stay out of access logs),
and a wrong token returns the same 404 as a missing case. Cases live in SQLite (`DERMAAI_DB`). For production,
put the service behind HTTPS and replace the shared clinician token with your identity provider.

### Out-of-distribution detection

After training, `training/train.py` fits class-conditional Gaussians with a shared Ledoit-Wolf covariance to the
backbone features of the training images. It then sets a threshold at the 95th percentile of Mahalanobis distances
on the validation images, which are held out from that fit. At inference, an image beyond the threshold is flagged
`ood.unfamiliar`. A low or moderate result is replaced by a "doesn't look like a typical lesion photo" retake
message, but a **high result is kept** so an atypical cancer is still referred. Images more than 3× past the
A second, "far" threshold is set 10% above the most unusual real validation image. Beyond it, even an urgent result
becomes a retake request, which never affects a genuine lesion photo from the validation set. With model_v1, a
flat-colour graphic and random noise are both flagged as unusual (about 2× the threshold). Whether the urgent result
is kept then depends on what the model predicts: v1 sends the graphic back for a retake, while v2 predicts "malignant"
for it and keeps the urgent result with an "unusual image" warning. Both images fall inside the range of real outliers,
so the far threshold deliberately does not override them. Energy scores were tried first and caught
nothing. To reject non-lesion photos reliably, add a `not_lesion/` class folder with everyday photos to `data/raw`
and retrain.

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `DERMAAI_MODELS_DIR` | `models/trained` | Where model versions and `CURRENT` live |
| `DERMAAI_CHECKPOINT` | none | Force a specific model file instead of `CURRENT` |
| `DERMAAI_DEVICE` | `auto` | `cuda`, `mps` or `cpu` |
| `DERMAAI_TTA` | `1` | 5-view test-time augmentation |
| `DERMAAI_MEL_THRESHOLD` | `0.15` | Melanoma probability that triggers urgent referral |
| `DERMAAI_MALIGNANT_HIGH` / `_MODERATE` | `0.40` / `0.15` | Combined malignant probability thresholds |
| `DERMAAI_UNCERTAINTY` | `0.75` | Uncertainty above which a "low" result is upgraded to "moderate" |
| `DERMAAI_MAX_UPLOAD_MB` | `10` | Upload size limit |
| `DERMAAI_DB` | `dermaai_cases.db` | SQLite file for referral cases |
| `DERMAAI_CLINICIAN_TOKEN` | none | Enables the clinician API and dashboard |
| `DERMAAI_FHIR_BASE` | `https://dermaai.local/fhir` | Base URL used for `fullUrl`s in FHIR bundles |
| `DERMAAI_ASSISTANT_LLM` | `auto` | `1` / `0` to force Claude on or off for the assistant |

Tune the thresholds on your validation set to reach the sensitivity you need.

## Project layout

```
app.py              start the web app (never trains)
data/               datasets — see data/README.md
training/           training module — see training/README.md
  config.py         ALL training settings (paths, model type, image size, epochs, batch size, LR, splits, seed)
  train.py          train / retrain -> models/trained/model_vN, evaluate, promote
  dataset.py        read data/raw (class folders or labels.csv), class names, stable lesion-grouped splits
  preprocessing.py  check, auto-rotate and cache images
  calibration.py    temperature scaling + out-of-distribution detector
  evaluate.py       metrics and reports (also usable on its own)
  manage_models.py  list versions, switch / roll back, import
models/trained/     model_v1, model_v2, ... and CURRENT — see models/README.md
prediction/
  predict.py        load the current model and classify images (CLI + Python API)
dermaai/            the application and the code it shares with training
  preprocessing.py  image -> model input (identical in training and prediction; parameters saved per model)
  model.py          DermNet: timm CNN backbone fused with metadata MLP
  model_store.py    where versions live and which one is current
  inference.py      quality gate -> calibrated prediction -> triage -> explanation
  config.py         class descriptions, app settings
  quality.py  abcde.py  explain.py  skin_analysis.py  skincare.py  progress.py  assistant.py  referrals.py
  api/main.py       FastAPI service (reloads the model automatically when CURRENT changes)
web/                responsive web app and clinician dashboard
scripts/            prepare_isic2019.py (download ISIC 2019), make_synthetic_dataset.py (tiny test data)
tests/              unit, API and end-to-end training tests
docs/PITCH.md       competition pitch
```

## Testing

```bash
pytest -q        # 35 tests: vision, metrics, skin care, tracking, assistant (incl. mocked Claude), referrals and access
                 # control, FHIR, API hardening, and a synthetic train → serve run with out-of-distribution checks
```

## Responsible AI

- **Skin-tone bias.** HAM10000 is mostly lighter skin. Before deployment, evaluate per Fitzpatrick type
  (Fitzpatrick17k, DDI) and fine-tune on diverse data such as PAD-UFES-20. The quality gate's skin detector
  uses a broad colour range so darker skin is not rejected.
- **Out-of-distribution input.** Non-lesion photos are caught by the quality gate, the feature-space OOD detector
  and high uncertainty, but this is not a guarantee. Clinical photos differ from dermoscopy, so fine-tune for smartphone
  use and recalibrate the OOD threshold on smartphone validation images.
- **Over-referral by design.** Thresholds trade specificity for melanoma sensitivity.
- **Privacy.** Images are never written to disk server-side, except for referral cases the patient explicitly
  consents to (deletable with their access code). History, body map, skin scores and chat live in the browser's
  localStorage. In Claude mode, chat text and a short summary of results (no images) are sent to the Claude API.
- **Product neutrality.** Recommendations are generic product types defined by ingredients, not brands. Add a
  partner catalogue in `skincare.PRODUCTS` with the same fields if you commercialise.
- **Regulation.** Clinical use needs regulatory clearance (e.g. EU MDR class IIa, FDA 510(k)/De Novo) and
  prospective validation.
