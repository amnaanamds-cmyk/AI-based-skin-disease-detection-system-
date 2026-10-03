import numpy as np
import torch

from dermaai.config import CLASSES, META_DIM
from dermaai.data import class_weights
from dermaai.metadata import encode_metadata
from dermaai.metrics import compute_metrics, expected_calibration_error, fit_temperature
from dermaai.model import DermNet


def test_metadata_encoding():
    v = encode_metadata(50, "female", "back")
    assert v.shape == (META_DIM,)
    assert v[0] == 0.5 and v[1] == 0
    missing = encode_metadata(None, None, "nowhere")
    assert missing[1] == 1 and missing.sum() == 3  # missing-age flag + unknown sex + unknown site


def test_model_forward_with_and_without_meta():
    m = DermNet("resnet18", len(CLASSES)).eval()
    x = torch.randn(2, 3, 64, 64)
    assert m(x).shape == (2, len(CLASSES))
    logits, fmap = m(x, torch.zeros(2, META_DIM), return_features=True)
    assert logits.shape == (2, len(CLASSES)) and fmap.dim() == 4


def test_backbones_with_conv_head():
    for arch in ("mobilenetv3_large_100", "efficientnet_b0"):
        m = DermNet(arch, len(CLASSES)).eval()
        assert m(torch.randn(1, 3, 64, 64)).shape == (1, len(CLASSES))
        assert m.embed(torch.randn(1, 3, 64, 64)).shape[1] == m.head.in_features - 64


def test_class_weights_favour_rare_classes():
    w = class_weights(np.array([5] * 90 + [4] * 10))
    assert w[4] > w[5] and w[0] == 0


def test_metrics_perfect_classifier():
    labels = np.repeat(np.arange(len(CLASSES)), 10)
    probs = np.eye(len(CLASSES))[labels] * 0.9 + 0.1 / len(CLASSES)
    m = compute_metrics(probs, labels)
    assert m["balanced_accuracy"] == 1.0
    assert m["melanoma_auc"] == 1.0 and m["malignant_auc"] == 1.0
    assert m["melanoma_at_90_sensitivity"]["specificity"] == 1.0


def test_temperature_reduces_overconfidence():
    torch.manual_seed(0)
    labels = torch.randint(0, 7, (2000,))
    logits = torch.randn(2000, 7)
    logits[torch.arange(2000), labels] += 1.0
    logits *= 4  # overconfident
    t = fit_temperature(logits, labels)
    assert t > 1.5
    before = expected_calibration_error(torch.softmax(logits, 1).numpy(), labels.numpy())
    after = expected_calibration_error(torch.softmax(logits / t, 1).numpy(), labels.numpy())
    assert after < before
