# Models

```
models/
├── trained/
│   ├── CURRENT          <- the version the app serves (one line, e.g. "model_v1")
│   ├── model_v1/        <- created by training/train.py, never overwritten
│   │   ├── model.pt     <- weights + classes + preprocessing + calibration: all the app needs
│   │   ├── model.json   <- readable summary: classes, scores, settings, dataset
│   │   └── report.html  <- open in a browser to see how good this version is
│   └── model_v2/ ...
└── checkpoints/         <- in-progress training state for --resume (temporary, not committed)
```

- **Train a new version:** `python training/train.py`. It is promoted to `CURRENT` automatically if it scores at
  least as well as the current model on the same test images.
- **See versions:** `python training/manage_models.py list`
- **Switch or roll back:** `python training/manage_models.py use model_v1`. A running app switches on its next
  request; no restart is needed.
- **Use a model file from elsewhere:** `python training/manage_models.py import path/to/model.pt`

The app finds the model by itself (`models/trained/CURRENT`). To force a specific file instead, set the
environment variable `DERMAAI_CHECKPOINT=path/to/model.pt`. If no model exists, the app runs in a clearly
labelled demo mode with an untrained network.

Keep at least the last working version when cleaning up. Deleting a `model_vN` folder is safe as long as it is
not the one named in `CURRENT`.
