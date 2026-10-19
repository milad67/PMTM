from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Mapping, Tuple

import torch

from models import TSMNet


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    clean = OrderedDict()
    for key, value in state_dict.items():
        clean[key[7:] if key.startswith("module.") else key] = value
    return clean


def model_config_from_checkpoint(checkpoint: Mapping[str, object]) -> Dict[str, object]:
    config = dict(checkpoint.get("model_config", {}))
    if config:
        return config
    # Compatibility with the older checkpoint metadata.
    return {
        "num_classes": 4,
        "state_dim": int(checkpoint.get("state_dim", 64)),
        "num_states": int(checkpoint.get("num_states", 4)),
        "variant": str(checkpoint.get("variant", checkpoint.get("ablation", "stage234"))),
        "basic_dims": int(checkpoint.get("basic_dims", 16)),
        "use_spatial_gate": bool(checkpoint.get("use_spatial_gate", False)),
        "token_mixer_stage4": bool(checkpoint.get("token_mixer_stage4", True)),
    }


def load_model_checkpoint(
    path: str,
    device: torch.device,
    strict: bool = True,
) -> Tuple[TSMNet, Dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        checkpoint = {"state_dict": checkpoint}
    state_dict = strip_module_prefix(checkpoint.get("state_dict", checkpoint))
    model_config = model_config_from_checkpoint(checkpoint)
    model = TSMNet(**model_config)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        print("[checkpoint] missing keys:", incompatible.missing_keys)
        print("[checkpoint] unexpected keys:", incompatible.unexpected_keys)
    model.to(device)
    model.eval()
    return model, checkpoint
