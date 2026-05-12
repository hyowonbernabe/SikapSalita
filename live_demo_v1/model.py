"""
Load the trained SignTransformer checkpoint for inference.
"""

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from models.transformer import SignTransformer

_CKPT = _ROOT / "trained_models/transformer/FSL105_classification/SignTransformer_best.pt"

_model: SignTransformer | None = None
_device: torch.device | None = None


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> tuple[SignTransformer, torch.device]:
    """Load and return (model, device). Cached after first call."""
    global _model, _device
    if _model is not None:
        return _model, _device

    device = get_device()
    model = SignTransformer(
        input_dim=178,
        emb_dim=256,
        n_heads=8,
        n_layers=4,
        num_gloss=105,
        num_cat=10,
        dropout=0.1,
        max_len=300,
        pooling_method="mean",
    )

    ckpt = torch.load(_CKPT, map_location=device, weights_only=False)

    # Checkpoint may be a training wrapper or a bare state_dict
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and any(
        k.startswith(("embedding", "encoder_layers", "gloss_head", "pos_encoder"))
        for k in ckpt.keys()
    ):
        state_dict = ckpt
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    _model, _device = model, device
    print(f"[Sikap-Salita] Model loaded on {device}")
    return _model, _device
