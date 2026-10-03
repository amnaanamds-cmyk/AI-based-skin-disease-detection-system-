"""End-to-end smoke test: synthetic data → train → calibrate → serve."""

import subprocess
import sys
from pathlib import Path

from PIL import Image

from dermaai.config import Settings
from dermaai.inference import Predictor
from dermaai.train import build_parser, train

ROOT = Path(__file__).resolve().parents[1]


def test_train_and_serve(tmp_path):
    data = tmp_path / "data"
    subprocess.run([sys.executable, str(ROOT / "scripts/make_synthetic_dataset.py"), "--out", str(data), "--per-class", "12"], check=True)
    args = build_parser().parse_args([
        "--csv", str(data / "metadata.csv"), "--images", str(data / "images"), "--out", str(tmp_path / "models"),
        "--arch", "resnet18", "--img-size", "64", "--epochs", "2", "--batch-size", "16", "--workers", "0",
        "--device", "cpu",
    ])
    metrics = train(args)
    assert 0 <= metrics["balanced_accuracy"] <= 1
    ckpt = tmp_path / "models" / "dermaai.pt"
    assert ckpt.exists() and (tmp_path / "models" / "metrics.json").exists()

    p = Predictor(Settings(checkpoint=ckpt, device="cpu"))
    assert not p.demo_mode and p.img_size == 64
    img = Image.open(next((data / "images").glob("SYN_mel_*.jpg")))
    out = p.analyze(img, age=70, sex="male", localization="back")
    assert out["model"]["demo_mode"] is False
    assert not any("DEMO" in r for r in out["triage"]["reasons"])
    assert p.info["ood_detection"] and out["ood"]["threshold"] > 0

    # Out-of-distribution: random noise is nothing like a lesion photo.
    import numpy as np
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8))
    r = p.analyze(noise, explain=False)
    assert r["ood"]["unfamiliar"] and r["triage"]["level"] == "retake"


def test_resume_continues_from_last_epoch(tmp_path, capsys):
    data = tmp_path / "data"
    subprocess.run([sys.executable, str(ROOT / "scripts/make_synthetic_dataset.py"), "--out", str(data), "--per-class", "8"], check=True)
    base = ["--csv", str(data / "metadata.csv"), "--images", str(data / "images"), "--out", str(tmp_path / "m"),
            "--arch", "resnet18", "--img-size", "48", "--batch-size", "16", "--workers", "0", "--device", "cpu",
            "--log-every", "0", "--patience", "99"]
    train(build_parser().parse_args(base + ["--epochs", "1"]))
    assert (tmp_path / "m" / "last.pt").exists()
    capsys.readouterr()
    metrics = train(build_parser().parse_args(base + ["--epochs", "2", "--resume"]))
    out = capsys.readouterr().out
    assert "resumed after epoch 1" in out and "[epoch 02]" in out and "[epoch 01]" not in out
    assert metrics["splits"]["train"] > 0
