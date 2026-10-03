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

1. Prepare **ISIC 2019**: 25,331 dermoscopy images in 8 classes, combining HAM10000 (Vienna), BCN20000
   (Barcelona) and MSK (New York). The script streams the official 9.8 GB archive and keeps resized copies only
   (about 550 MB, about 20 minutes, resumable):

```bash
python scripts/prepare_isic2019.py --out data/isic2019 --size 256
```

2. Train. `--pretrained` loads ImageNet weights, from GitHub if the Hugging Face Hub is unreachable. On a GPU use a
   bigger model and resolution. The command below is the CPU recipe that produced the bundled results:

```bash
python -m dermaai.train --csv data/isic2019/metadata.csv --images data/isic2019/images \
  --arch efficientnet_b0 --pretrained --img-size 192 --epochs 10 --batch-size 32 --lr 2e-4 \
  --workers 2 --threads 4 --out models
# GPU: --arch efficientnet_b2 --img-size 260 --epochs 25 --batch-size 64
# interrupted? add --resume; cap wall-clock time with --max-hours
```

This produces `models/dermaai.pt` (weights, calibration temperature, OOD detector, test metrics) and
`models/metrics.json`. HAM10000 alone also works: pass its `HAM10000_metadata.csv` and image folders.

3. Serve it with `DERMAAI_CHECKPOINT=models/dermaai.pt uvicorn dermaai.api.main:app`.
4. Optional: `python -m dermaai.evaluate --checkpoint models/dermaai.pt --csv external.csv --images ext/`
   to validate on external data (ISIC 2019, PAD-UFES-20, Derm7pt), and
   `python -m dermaai.export --checkpoint models/dermaai.pt --out models/dermaai.onnx` for mobile.

Any CSV with columns `image`/`image_id`, `dx` and optionally `lesion_id`, `age`, `sex`, `localization` works.

**What the training pipeline does:** stratified, lesion-grouped train/val/test split · square-root class-balanced
sampling plus a mild class-weighted loss · heavy dermoscopy augmentation (rotation, flips, colour jitter, blur,
random erasing) · per-field metadata dropout so the model works without age/sex/site · AdamW with a 10× learning
rate on the head, warmup and cosine decay · AMP on GPU · per-epoch resumable checkpoints · early stopping on
`balanced_accuracy + 0.5 × melanoma_AUC` · 5-view TTA evaluation · temperature calibration and OOD detector fitting
on held-out data · per-source (hospital) test metrics when the CSV has a `source` column.

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

After training, `dermaai.train` fits class-conditional Gaussians with a shared Ledoit-Wolf covariance to the
backbone features of the training images. It then sets a threshold at the 95th percentile of Mahalanobis distances
on the validation images, which are held out from that fit. At inference, an image beyond the threshold is flagged
`ood.unfamiliar`. A low or moderate result is replaced by a "doesn't look like a typical lesion photo" retake
message, but a **high result is kept** so an atypical cancer is still referred. Images more than 3× past the
threshold (noise, objects, screenshots) are always sent back for a retake. With the synthetic smoke-test model,
this flagged 20/20 non-lesion images (noise, flat colour, gradients, patterns) and 2/50 real-distribution images.
Energy scores were tried first and caught none, because weak models stay confident on garbage.

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
| `DERMAAI_DB` | `dermaai_cases.db` | SQLite file for referral cases |
| `DERMAAI_CLINICIAN_TOKEN` | none | Enables the clinician API and dashboard |
| `DERMAAI_FHIR_BASE` | `https://dermaai.local/fhir` | Base URL used for `fullUrl`s in FHIR bundles |
| `DERMAAI_ASSISTANT_LLM` | `auto` | `1` / `0` to force Claude on or off for the assistant |

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
  skin_analysis.py  facial/skin cosmetic analysis       skincare.py  routine engine + product catalogue
  progress.py    before/after lesion change tracking    assistant.py red flags, knowledge base, Claude
  referrals.py   consented case store, clinician queue, FHIR R4 export
  api/main.py    FastAPI service
web/             responsive app: lesion check, live camera coach, skin care, tracking, body map, assistant,
                 history + referrals, print report; clinician.html is the clinician dashboard
scripts/         synthetic dataset generator (pipeline smoke tests only)
tests/           unit, API and end-to-end train→serve tests
docs/PITCH.md    competition pitch, impact and business model
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
