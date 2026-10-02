# DermaAI — Competition Pitch

## The problem
- Skin cancer is the most common cancer worldwide. Melanoma causes most skin-cancer deaths, but
  **5-year survival is above 99% when caught early** and falls to roughly 35% once it has spread.
- Dermatologists are scarce. Waits of several months are common, and many rural and low-income regions
  have almost none. Most suspicious spots are never examined in time, while benign ones crowd the clinics.

## Our solution
DermaAI turns any phone or dermatoscope into a **triage assistant** that answers one question:
*how urgently should this spot be seen?*

1. **Quality gate.** It refuses to guess on bad photos and tells the user how to retake them.
2. **Fusion deep learning.** An EfficientNet CNN reads the image, and patient age, sex and body site are fused in.
3. **Honest confidence.** Probabilities are calibrated, an uncertainty score is shown, and uncertain cases are escalated.
4. **Explainable.** A Grad-CAM heatmap shows where the model looked, next to an independent ABCDE analysis that doctors already trust.
5. **Safety-first triage.** It is tuned for melanoma sensitivity, so it would rather over-refer than miss a cancer.
6. **Private by design.** Images are processed in memory only and history stays on the user's phone.

## Live demo script (3 minutes)
1. Upload a blurry photo. DermaAI asks for a retake and says why. *(Trust: it knows its limits.)*
2. Upload a nevus. Result: low priority, the outline is drawn and ABCDE is low. Save to history.
3. Upload a melanoma. Result: **HIGH priority**, the heatmap sits on the irregular, multi-coloured region, with a clear next step.
4. Open the model card: lesion-grouped test metrics, melanoma AUC, and specificity at 90% sensitivity.
5. Show the ONNX export. It runs offline on a phone for clinics with no connectivity.

## Who pays (business model)
| Segment | Offer | Model |
|---|---|---|
| Consumers | Free monthly mole check + history | Freemium; premium tracking reminders |
| GPs / primary care | Referral prioritisation, printable reports | SaaS per seat |
| Teledermatology platforms | Pre-screening API to sort queues by urgency | Per-call API pricing |
| Insurers / employers | Population skin-cancer screening programmes | B2B licence |
| NGOs / public health | Offline ONNX app for community health workers | Grant-funded |

## Why we win
- **Clinically grounded**: lesion-grouped evaluation, calibration, and a sensitivity-first operating point, with no inflated accuracy.
- **Trustworthy**: a quality gate, uncertainty, two kinds of explanation, and prominent disclaimers.
- **Shippable**: a REST API, a responsive web app, Docker, CI tests, and ONNX for edge. It is a product, not a notebook.

## Roadmap
1. Train on ISIC 2019/2020 + HAM10000 and validate externally on PAD-UFES-20 (smartphone images).
2. Fairness audit per Fitzpatrick type on Fitzpatrick17k / DDI; add diverse fine-tuning data.
3. Lesion change tracking: register photos over time and quantify growth (the "E" in ABCDE).
4. Native mobile app with on-device inference and a guided capture overlay.
5. Clinical pilot with a dermatology partner, then a regulatory path (EU MDR IIa / FDA De Novo).
