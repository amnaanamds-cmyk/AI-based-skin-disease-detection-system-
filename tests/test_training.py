"""End-to-end tests of the training module: data folder -> train -> versioned model -> app."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dermaai import model_store
from dermaai.config import Settings
from dermaai.inference import Predictor
from training.config import parse_config
from training.train import train

ROOT = Path(__file__).resolve().parents[1]


def make_data(path: Path, per_class: int, seed: int = 0) -> Path:
    subprocess.run([sys.executable, str(ROOT / "scripts/make_synthetic_dataset.py"), "--out", str(path),
                    "--per-class", str(per_class), "--seed", str(seed)], check=True, capture_output=True)
    return path


def config(tmp: Path, data: Path, *extra: str):
    return parse_config([
        "--data", str(data), "--processed-dir", str(tmp / "processed"), "--models-dir", str(tmp / "models"),
        "--checkpoint-dir", str(tmp / "ckpt"), "--split-file", str(tmp / "splits.csv"),
        "--model-type", "resnet18", "--no-pretrained", "--img-size", "48", "--processed-size", "64",
        "--batch-size", "16", "--num-workers", "0", "--device", "cpu", "--log-every", "0", "--min-images-per-class", "5",
        *extra,
    ])


def test_train_save_promote_and_serve(tmp_path):
    data = make_data(tmp_path / "raw", 14)
    result = train(config(tmp_path, data, "--epochs", "2"))
    models = tmp_path / "models"
    v1 = models / "model_v1"
    assert result["version"] == "model_v1" and result["promoted"]
    for f in ("model.pt", "model.json", "report.html", "metrics.json", "evaluation.txt", "confusion_matrix.csv",
              "training_history.csv", "dataset_used.csv"):
        assert (v1 / f).exists(), f
    info = json.loads((v1 / "model.json").read_text())
    assert info["classes"] == ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    assert info["preprocessing"]["img_size"] == 48 and info["dataset"]["images"] == 98
    assert model_store.current_version(models) == "model_v1"

    # The app side: loads the current version on its own, with the saved preprocessing and classes.
    p = Predictor(Settings(checkpoint=model_store.model_path(None, models), device="cpu"))
    assert not p.demo_mode and p.version == "model_v1" and p.img_size == 48
    out = p.analyze(Image.open(next((data / "mel").glob("*.jpg"))), age=70, sex="male", localization="back")
    assert out["model"]["model_version"] == "model_v1" and len(out["prediction"]["probabilities"]) == 7
    assert p.info["ood_detection"] and out["ood"]["threshold"] > 0
    noise = Image.fromarray(np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8))
    r = p.analyze(noise, explain=False)
    assert r["ood"]["unfamiliar"] and r["triage"]["level"] == "retake"


def test_retrain_with_more_data_keeps_splits_and_versions(tmp_path):
    data = make_data(tmp_path / "raw", 12)
    train(config(tmp_path, data, "--epochs", "1"))
    v1_used = pd.read_csv(tmp_path / "models/model_v1/dataset_used.csv")

    extra = make_data(tmp_path / "more", 6, seed=9)  # new photos of the same classes
    for img in extra.rglob("*.jpg"):
        shutil.copy(img, data / img.parent.name / f"new_{img.name}")
    (data / "nv" / "broken.jpg").write_text("not an image")
    result = train(config(tmp_path, data, "--epochs", "1", "--init-from", "current", "--promote", "always"))

    assert result["version"] == "model_v2" and model_store.list_versions(tmp_path / "models") == ["model_v1", "model_v2"]
    assert (tmp_path / "models/model_v1/model.pt").exists()  # old version untouched
    info = model_store.read_info("model_v2", tmp_path / "models")
    assert info["started_from"] == "model_v1" and info["dataset"]["images"] == 84 + 42
    v2_used = pd.read_csv(tmp_path / "models/model_v2/dataset_used.csv")
    both = v1_used.merge(v2_used, on="path", suffixes=("_1", "_2"))
    assert len(both) == len(v1_used) and (both.split_1 == both.split_2).all()  # test images stay test images
    assert not v2_used["path"].str.endswith("broken.jpg").any()


def test_new_class_and_rollback(tmp_path):
    data = make_data(tmp_path / "raw", 10)
    train(config(tmp_path, data, "--epochs", "1"))
    (data / "vasc").rename(data / "Eczema Patch")  # a class the app has never seen
    train(config(tmp_path, data, "--epochs", "1", "--init-from", "current"))
    models = tmp_path / "models"
    assert model_store.current_version(models) == "model_v2"  # classes changed -> promoted
    p = Predictor(Settings(checkpoint=model_store.model_path(None, models), device="cpu"))
    assert "eczema_patch" in p.classes and "vasc" not in p.classes
    out = p.analyze(Image.open(next((data / "Eczema Patch").glob("*.jpg"))), explain=False)
    assert any(e["code"] == "eczema_patch" and e["name"] == "Eczema Patch" for e in out["prediction"]["probabilities"])

    subprocess.run([sys.executable, str(ROOT / "training/manage_models.py"), "use", "model_v1",
                    "--models-dir", str(models)], check=True, capture_output=True)
    assert model_store.current_version(models) == "model_v1"


def test_resume_continues_from_last_epoch(tmp_path, capsys):
    data = make_data(tmp_path / "raw", 8)
    train(config(tmp_path, data, "--epochs", "1", "--patience", "99", "--promote", "never"))
    assert (tmp_path / "ckpt" / "last.pt").exists()
    capsys.readouterr()
    train(config(tmp_path, data, "--epochs", "2", "--patience", "99", "--resume"))
    out = capsys.readouterr().out
    assert "resuming after epoch 1" in out and "[epoch 2]" in out and "[epoch 1]" not in out


def test_helpful_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        train(config(tmp_path, tmp_path / "missing", "--epochs", "1"))
    data = make_data(tmp_path / "raw", 3)
    with pytest.raises(ValueError, match="fewer than"):
        train(parse_config(["--data", str(data), "--processed-dir", str(tmp_path / "p"), "--split-file",
                            str(tmp_path / "s.csv"), "--models-dir", str(tmp_path / "m"), "--checkpoint-dir", str(tmp_path / "c")]))


def test_label_csv_layout_and_aliases(tmp_path):
    from training.dataset import load_dataset
    raw = tmp_path / "raw"
    (raw / "images").mkdir(parents=True)
    for i in range(4):
        Image.new("RGB", (80, 80), (120, 80, 60)).save(raw / "images" / f"x{i}.jpg")
    pd.DataFrame({"image_id": ["x0", "x1", "x2.jpg", "missing"], "diagnosis": ["Melanoma", "nevus", "BCC", "mel"],
                  "age": [40, None, 60, 1]}).to_csv(raw / "labels.csv", index=False)
    (raw / "Basal Cell Carcinoma").mkdir()
    Image.new("RGB", (80, 80)).save(raw / "Basal Cell Carcinoma" / "y.png")
    df = load_dataset(raw)
    assert sorted(df["label"]) == ["bcc", "bcc", "mel", "nv"]
    assert df["lesion_id"].notna().all()
