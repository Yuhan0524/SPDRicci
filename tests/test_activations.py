import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import spd_activation  # noqa: E402


def test_classic_reeig_is_a_hard_eigenvalue_floor():
    matrix = torch.diag(torch.tensor([-0.5, 0.2, 2.0]))
    actual = spd_activation(matrix, "mild_reeig", floor=1.0e-4)
    expected = torch.diag(torch.tensor([1.0e-4, 0.2, 2.0]))
    assert torch.allclose(actual, expected)


def test_tgrelu_uses_the_ordered_log_spectrum():
    matrix = torch.diag(torch.tensor([1.0e-3, 0.2, 0.8, 3.0]))
    actual = spd_activation(
        matrix,
        "tgrelu",
        floor=1.0e-4,
        jitter=0.0,
        tgrelu_delta=0.2,
    )
    expected = torch.diag(torch.tensor([1.0, torch.exp(torch.tensor(0.2)),
                                        torch.exp(torch.tensor(0.4)), 3.0]))
    assert torch.allclose(actual, expected)


def test_public_wisconsin_default_selects_tgrelu():
    config = json.loads(
        (ROOT / "configs" / "wisconsin_tgrelu.json").read_text()
    )
    assert config["model"]["activation"] == "tgrelu"
    assert config["model"]["tgrelu_delta"] == 0.2


def test_public_wisconsin_reeig_ablation_is_available():
    config = json.loads(
        (ROOT / "configs" / "wisconsin_reeig.json").read_text()
    )
    assert config["model"]["activation"] == "mild_reeig"
    assert config["model"]["activation_jitter"] == 0.0
