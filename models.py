from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from layers import GeneralConv3d, ResidualStateGuidedFusion, StateHead, TumorStateMemory


BASIC_DIMS = 16
NUM_MODALITIES = 4

MODEL_VARIANTS = (
    "stage234",
    "stage1234",
    "stage_local",
    "stage1234_local",
    "no_state_memory",
)
VARIANT_ALIASES = {
    "pmsm": "stage234",
    "main": "stage234",
    "progressive": "stage234",
    "stage234": "stage234",
    "stage1234": "stage1234",
    "all_stages": "stage1234",
    "allstage": "stage1234",
    "stage_local": "stage_local",
    "local": "stage_local",
    "stage1234_local": "stage1234_local",
    "all_stages_local": "stage1234_local",
    "no_state_memory": "no_state_memory",
    "no_memory": "no_state_memory",
    "nomem": "no_state_memory",
}


def normalize_variant(name: str) -> str:
    try:
        return VARIANT_ALIASES[name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model variant {name!r}. Valid values: {sorted(VARIANT_ALIASES)}"
        ) from exc


class Encoder(nn.Module):
    def __init__(self, basic_dims: int = BASIC_DIMS) -> None:
        super().__init__()
        b = int(basic_dims)
        self.e1c1 = GeneralConv3d(1, b)
        self.e1c2 = GeneralConv3d(b, b)
        self.e1c3 = GeneralConv3d(b, b)

        self.e2c1 = GeneralConv3d(b, b * 2, stride=2)
        self.e2c2 = GeneralConv3d(b * 2, b * 2)
        self.e2c3 = GeneralConv3d(b * 2, b * 2)

        self.e3c1 = GeneralConv3d(b * 2, b * 4, stride=2)
        self.e3c2 = GeneralConv3d(b * 4, b * 4)
        self.e3c3 = GeneralConv3d(b * 4, b * 4)

        self.e4c1 = GeneralConv3d(b * 4, b * 8, stride=2)
        self.e4c2 = GeneralConv3d(b * 8, b * 8)
        self.e4c3 = GeneralConv3d(b * 8, b * 8)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x1 = self.e1c3(self.e1c2(self.e1c1(x)))
        x2 = self.e2c3(self.e2c2(self.e2c1(x1)))
        x3 = self.e3c3(self.e3c2(self.e3c1(x2)))
        x4 = self.e4c3(self.e4c2(self.e4c1(x3)))
        return x1, x2, x3, x4


class DecoderSeparate(nn.Module):
    """Training-only decoder for a single modality branch."""

    def __init__(self, num_classes: int = 4, basic_dims: int = BASIC_DIMS) -> None:
        super().__init__()
        b = int(basic_dims)
        self.up3 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d3c1 = GeneralConv3d(b * 8, b * 4)
        self.d3c2 = GeneralConv3d(b * 8, b * 4)
        self.d3out = GeneralConv3d(b * 4, b * 4, kernel_size=1, padding=0)

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d2c1 = GeneralConv3d(b * 4, b * 2)
        self.d2c2 = GeneralConv3d(b * 4, b * 2)
        self.d2out = GeneralConv3d(b * 2, b * 2, kernel_size=1, padding=0)

        self.up1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.d1c1 = GeneralConv3d(b * 2, b)
        self.d1c2 = GeneralConv3d(b * 2, b)
        self.d1out = GeneralConv3d(b, b, kernel_size=1, padding=0)

        self.seg = nn.Conv3d(b, num_classes, kernel_size=1, bias=True)

    def forward(self, x1, x2, x3, x4) -> torch.Tensor:
        d3 = self.d3c1(self.up3(x4))
        d3 = self.d3out(self.d3c2(torch.cat([d3, x3], dim=1)))
        d2 = self.d2c1(self.up2(d3))
        d2 = self.d2out(self.d2c2(torch.cat([d2, x2], dim=1)))
        d1 = self.d1c1(self.up1(d2))
        d1 = self.d1out(self.d1c2(torch.cat([d1, x1], dim=1)))
        return torch.softmax(self.seg(d1), dim=1)


class DecoderFuse(DecoderSeparate):
    """Shared decoder for the fused multi-stage representation."""


class TSMNet(nn.Module):
    """Progressive multi-scale tumor-state memory network.

    The corrected implementation directly supervises ordered assignment maps at
    encoder stages 2, 3, and 4.  The model class itself only constructs the
    forward path; semantic, consistency, identity, and diversity objectives are
    implemented in :mod:`utils.losses`.

    Parameters
    ----------
    variant:
        ``stage234`` reproduces the original progressive implementation with
        active stages 2--4. ``stage1234`` applies the same progressive state
        update and state-guided fusion principle at all four encoder stages.
        ``stage_local`` and ``stage1234_local`` reset the state before every
        active stage and therefore remove cross-stage propagation while
        retaining the same heads and fusion modules. ``no_state_memory``
        replaces state-guided fusion by projected masked averages.
    """

    def __init__(
        self,
        num_classes: int = 4,
        state_dim: int = 64,
        num_states: int = 4,
        variant: str = "stage234",
        basic_dims: int = BASIC_DIMS,
        use_spatial_gate: bool = False,
        token_mixer_scope: str = "stage4",
        token_mixer_stage4: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if num_states != num_classes:
            raise ValueError(
                "The manuscript configuration uses one ordered state slot per class; "
                f"got num_states={num_states}, num_classes={num_classes}."
            )
        self.num_classes = int(num_classes)
        self.K = int(num_states)
        self.state_dim = int(state_dim)
        self.M = NUM_MODALITIES
        self.basic_dims = int(basic_dims)
        self.variant = normalize_variant(variant)
        if self.variant == "no_state_memory":
            self.active_stages = []
        elif self.variant in ("stage1234", "stage1234_local"):
            self.active_stages = [1, 2, 3, 4]
        else:
            self.active_stages = [2, 3, 4]

        # Backward compatibility with checkpoints produced before the generic
        # token-mixer scope was introduced.
        if token_mixer_stage4 is not None:
            token_mixer_scope = "stage4" if bool(token_mixer_stage4) else "none"
        token_mixer_scope = str(token_mixer_scope).lower()
        if token_mixer_scope not in {"none", "stage4", "active"}:
            raise ValueError(
                "token_mixer_scope must be one of {'none', 'stage4', 'active'}, "
                f"got {token_mixer_scope!r}."
            )
        self.token_mixer_scope = token_mixer_scope

        b = self.basic_dims
        channels = {1: b, 2: b * 2, 3: b * 4, 4: b * 8}
        self.encoders = nn.ModuleList([Encoder(basic_dims=b) for _ in range(self.M)])

        fallback_stages = [stage for stage in (1, 2, 3, 4) if stage not in self.active_stages]
        self.fallback_fusion = nn.ModuleDict(
            {
                f"s{stage}": GeneralConv3d(
                    channels[stage], channels[stage], kernel_size=1, padding=0
                )
                for stage in fallback_stages
            }
        )

        self.state_heads = nn.ModuleDict()
        self.sgfs = nn.ModuleDict()
        if self.active_stages:
            self.memory = TumorStateMemory(num_states=self.K, state_dim=self.state_dim)
            for stage in self.active_stages:
                self.state_heads[f"s{stage}"] = StateHead(
                    channels[stage], num_states=self.K, state_dim=self.state_dim
                )
                self.sgfs[f"s{stage}"] = ResidualStateGuidedFusion(
                    channels=channels[stage],
                    num_modalities=self.M,
                    num_states=self.K,
                    state_dim=self.state_dim,
                    use_spatial_gate=use_spatial_gate,
                    use_token_mixer=(
                        self.token_mixer_scope == "active"
                        or (self.token_mixer_scope == "stage4" and stage == 4)
                    ),
                    token_mixer_layers=1,
                )

        self.decoder = DecoderFuse(num_classes=num_classes, basic_dims=b)
        self.auxiliary_decoders = nn.ModuleList(
            [DecoderSeparate(num_classes=num_classes, basic_dims=b) for _ in range(self.M)]
        )

    @staticmethod
    def _validate_mask(mask: torch.Tensor, batch_size: int) -> torch.Tensor:
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(batch_size, -1)
        if tuple(mask.shape) != (batch_size, NUM_MODALITIES):
            raise ValueError(
                f"Expected modality mask shape ({batch_size},4), got {tuple(mask.shape)}"
            )
        mask = mask.bool()
        if (~mask.any(dim=1)).any():
            raise ValueError("Every sample must contain at least one MRI modality.")
        return mask

    def encode_modalities(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, ...]]:
        if x.ndim != 5 or x.shape[1] != self.M:
            raise ValueError(f"Expected input shape (B,4,H,W,D), got {tuple(x.shape)}")
        # Each observed modality is encoded exactly once.  Missing branches are
        # excluded by the mask in all subsequent pooling/fusion operations.
        return [self.encoders[m](x[:, m : m + 1]) for m in range(self.M)]

    @staticmethod
    def _masked_average(tensors: Sequence[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        if len(tensors) != NUM_MODALITIES:
            raise ValueError("Expected four modality tensors.")
        batch = mask.shape[0]
        weights = mask.to(dtype=tensors[0].dtype).view(batch, NUM_MODALITIES, 1, 1, 1, 1)
        stacked = torch.stack(list(tensors), dim=1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return (stacked * weights).sum(dim=1) / denominator

    def _state_stage(
        self,
        stage: int,
        features: Sequence[torch.Tensor],
        mask: torch.Tensor,
        previous_state: torch.Tensor,
        return_debug: bool,
        compute_fusion: bool,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Dict[str, object]]:
        key = f"s{stage}"
        assignments: List[torch.Tensor] = []
        projections: List[torch.Tensor] = []
        logits: List[torch.Tensor] = []
        for feature in features:
            assignment, projection, stage_logits = self.state_heads[key](feature)
            assignments.append(assignment)
            projections.append(projection)
            logits.append(stage_logits)

        if self.variant in ("stage_local", "stage1234_local"):
            state_input = self.memory.init_state(
                batch_size=features[0].shape[0],
                device=features[0].device,
                dtype=features[0].dtype,
            )
        else:
            state_input = previous_state

        state, modality_evidence, mean_evidence = self.memory.update_from_modalities(
            state_input, assignments, projections, mask
        )
        if compute_fusion and return_debug:
            fused, fusion_debug = self.sgfs[key](
                features, assignments, state, mask, return_debug=True
            )
        elif compute_fusion:
            fused = self.sgfs[key](features, assignments, state, mask, return_debug=False)
            fusion_debug = None
        else:
            fused = None
            fusion_debug = None

        mean_assignment = self._masked_average(assignments, mask)
        details: Dict[str, object] = {
            "stage": stage,
            "assignments": assignments,
            "assignment_logits": logits,
            "mean_assignment": mean_assignment,
            "projections": projections,
            "modality_evidence": modality_evidence,
            "mean_evidence": mean_evidence,
            "state_input": state_input,
            "state": state,
            "fusion_debug": fusion_debug,
        }
        return fused, state, details

    def forward_from_encoded(
        self,
        encoded: Sequence[Tuple[torch.Tensor, ...]],
        mask: torch.Tensor,
        *,
        compute_prediction: bool = True,
        compute_auxiliary: bool = False,
        return_debug: bool = False,
    ) -> Dict[str, object]:
        if len(encoded) != self.M:
            raise ValueError("Expected encoded features for four modalities.")
        batch = encoded[0][0].shape[0]
        mask = self._validate_mask(mask.to(encoded[0][0].device), batch)

        if self.active_stages:
            state = self.memory.init_state(
                batch, encoded[0][0].device, encoded[0][0].dtype
            )
        else:
            state = torch.zeros(
                batch,
                self.K,
                self.state_dim,
                device=encoded[0][0].device,
                dtype=encoded[0][0].dtype,
            )

        fused_features: List[torch.Tensor] = []
        stage_details: Dict[int, Dict[str, object]] = {}
        for stage in (1, 2, 3, 4):
            features = [encoded[m][stage - 1] for m in range(self.M)]
            if stage in self.active_stages:
                fused, state, details = self._state_stage(
                    stage, features, mask, state,
                    return_debug=return_debug,
                    compute_fusion=compute_prediction or return_debug,
                )
                stage_details[stage] = details
            elif compute_prediction or return_debug:
                fused = self.fallback_fusion[f"s{stage}"](
                    self._masked_average(features, mask)
                )
            else:
                fused = None
            if fused is not None:
                fused_features.append(fused)

        prediction = self.decoder(*fused_features) if compute_prediction else None
        auxiliary = None
        if compute_auxiliary:
            auxiliary = []
            for modality in range(self.M):
                if mask[:, modality].any():
                    auxiliary.append(self.auxiliary_decoders[modality](*encoded[modality]))
                else:
                    auxiliary.append(None)

        return {
            "prediction": prediction,
            "auxiliary": auxiliary,
            "stage_details": stage_details,
            "states": [(stage, stage_details[stage]["state"]) for stage in sorted(stage_details)],
            "state_maps": [
                (stage, stage_details[stage]["mean_assignment"])
                for stage in sorted(stage_details)
            ],
            "mask": mask,
        }

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        *,
        return_details: bool = False,
        compute_auxiliary: bool = False,
        return_debug: bool = False,
    ):
        encoded = self.encode_modalities(x)
        output = self.forward_from_encoded(
            encoded,
            mask,
            compute_prediction=True,
            compute_auxiliary=compute_auxiliary,
            return_debug=return_debug,
        )
        return output if return_details else output["prediction"]


# Backward-compatible names.
Model = TSMNet
normalize_ablation = normalize_variant
