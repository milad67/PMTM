from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalization(channels: int, norm: str = "in") -> nn.Module:
    if norm == "bn":
        return nn.BatchNorm3d(channels)
    if norm == "gn":
        groups = 4 if channels % 4 == 0 else 1
        return nn.GroupNorm(groups, channels)
    if norm == "in":
        return nn.InstanceNorm3d(channels, affine=False)
    raise ValueError(f"Unsupported normalization: {norm}")


class GeneralConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        padding_mode: str = "reflect",
        norm: str = "in",
        activation: str = "lrelu",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            bias=True,
        )
        self.norm = normalization(out_channels, norm=norm)
        if activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


# Compatibility alias used in the original code.
general_conv3d = GeneralConv3d


class StateHead(nn.Module):
    """Stage-specific assignment and evidence projections."""

    def __init__(self, in_channels: int, num_states: int = 4, state_dim: int = 64) -> None:
        super().__init__()
        hidden = max(in_channels // 2, num_states * 4)
        self.assign_branch = nn.Sequential(
            nn.Conv3d(in_channels, hidden, kernel_size=1, bias=False),
            nn.InstanceNorm3d(hidden, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(hidden, num_states, kernel_size=1, bias=True),
        )
        self.proj_branch = nn.Sequential(
            nn.Conv3d(in_channels, state_dim, kernel_size=1, bias=False),
            nn.InstanceNorm3d(state_dim, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        output_conv = self.assign_branch[-1]
        nn.init.xavier_uniform_(output_conv.weight, gain=1.0)
        if output_conv.bias is not None:
            nn.init.zeros_(output_conv.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.assign_branch(x)
        assignment = torch.softmax(logits, dim=1)
        projection = self.proj_branch(x)
        return assignment, projection, logits


class TumorStateMemory(nn.Module):
    """Recurrent subject-specific state update shared across active stages."""

    def __init__(self, num_states: int = 4, state_dim: int = 64) -> None:
        super().__init__()
        self.num_states = int(num_states)
        self.state_dim = int(state_dim)
        self.T0 = nn.Parameter(torch.empty(1, num_states, state_dim))
        nn.init.trunc_normal_(self.T0, std=0.02)
        self.eta_logit = nn.Parameter(torch.zeros(1, num_states, 1))

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.T0.to(device=device, dtype=dtype).expand(batch_size, -1, -1).contiguous()

    @staticmethod
    def pool_evidence(assignment: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
        """Pool spatial evidence in FP32 to prevent AMP reduction overflow."""
        batch, states = assignment.shape[:2]
        dim = projection.shape[1]

        with torch.cuda.amp.autocast(enabled=False):
            a_flat = assignment.float().reshape(batch, states, -1)
            p_flat = projection.float().reshape(batch, dim, -1)

            if not torch.isfinite(a_flat).all():
                raise FloatingPointError(
                    "Non-finite assignment before tumor-state evidence pooling."
                )
            if not torch.isfinite(p_flat).all():
                raise FloatingPointError(
                    "Non-finite projection before tumor-state evidence pooling."
                )

            numerator = torch.bmm(
                a_flat,
                p_flat.transpose(1, 2),
            )
            denominator = (
                a_flat.sum(dim=2, keepdim=True)
                .clamp_min(1e-6)
            )
            evidence = numerator / denominator

            if not torch.isfinite(evidence).all():
                raise FloatingPointError(
                    "Non-finite pooled tumor-state evidence after FP32 accumulation."
                )

        return evidence

    def combine_evidence(
        self,
        evidence: Sequence[torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if len(evidence) != mask.shape[1]:
            raise ValueError(
                "The evidence list and modality mask must have the same modality count."
            )

        with torch.cuda.amp.autocast(enabled=False):
            stacked = torch.stack(
                [item.float() for item in evidence],
                dim=1,
            )  # (B,M,K,d)

            weights = (
                mask.float()
                .view(mask.shape[0], mask.shape[1], 1, 1)
            )
            denominator = weights.sum(dim=1).clamp_min(1.0)
            average = (stacked * weights).sum(dim=1) / denominator

            if not torch.isfinite(average).all():
                raise FloatingPointError(
                    "Non-finite modality-averaged tumor-state evidence."
                )

        return average

    def update(
        self,
        previous: torch.Tensor,
        evidence_average: torch.Tensor,
    ) -> torch.Tensor:
        output_dtype = previous.dtype

        with torch.cuda.amp.autocast(enabled=False):
            previous_fp32 = previous.float()
            evidence_fp32 = evidence_average.float()
            eta = torch.sigmoid(self.eta_logit.float())

            updated_fp32 = (
                eta * previous_fp32
                + (1.0 - eta) * evidence_fp32
            )

            if not torch.isfinite(updated_fp32).all():
                raise FloatingPointError(
                    "Non-finite tumor state after FP32 memory update."
                )

        return updated_fp32.to(dtype=output_dtype)

    def update_from_modalities(
        self,
        previous: torch.Tensor,
        assignment_maps: Sequence[torch.Tensor],
        projected_features: Sequence[torch.Tensor],
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        evidence = [
            self.pool_evidence(assignment, projection)
            for assignment, projection in zip(assignment_maps, projected_features)
        ]
        average = self.combine_evidence(evidence, mask)
        updated = self.update(previous, average)
        return updated, evidence, average


class ModalityTokenMixer(nn.Module):
    """A small Transformer operating on one token per MRI modality."""

    def __init__(
        self,
        channels: int,
        state_dim: int = 64,
        num_modalities: int = 4,
        num_heads: int = 4,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        token_input = channels + state_dim + num_modalities
        self.token_proj = nn.Linear(token_input, channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=max(channels * 2, 64),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Sequential(nn.Linear(channels, channels), nn.Sigmoid())

    def forward(
        self,
        global_features: torch.Tensor,
        state_context: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, modalities, _ = global_features.shape
        subset_code = mask.float().unsqueeze(1).expand(batch, modalities, -1)
        tokens = self.token_proj(torch.cat([global_features, state_context, subset_code], dim=-1))
        tokens = tokens * mask.to(dtype=tokens.dtype).unsqueeze(-1)

        padding_mask = ~mask.bool()

        if self.training or not tokens.is_cuda:
            # Preserve the original training path exactly.
            mixed = self.encoder(
                tokens,
                src_key_padding_mask=padding_mask,
            )
        else:
            # PyTorch 1.12.1 workaround for:
            # eval() + no_grad() + CUDA autocast + MultiheadAttention.
            #
            # In that combination, the native MHA inference path may receive
            # Half activations while its parameters remain Float, producing:
            # RuntimeError: expected scalar type Half but found Float
            original_dtype = tokens.dtype
            encoder_dtype = next(self.encoder.parameters()).dtype

            with torch.cuda.amp.autocast(enabled=False):
                mixed = self.encoder(
                    tokens.to(dtype=encoder_dtype),
                    src_key_padding_mask=padding_mask,
                )

            # Restore the surrounding AMP dtype before the output projection.
            mixed = mixed.to(dtype=original_dtype)

        return self.out_proj(mixed)


class ResidualStateGuidedFusion(nn.Module):
    """Masked-average base path plus a state-conditioned residual path."""

    def __init__(
        self,
        channels: int,
        num_modalities: int = 4,
        num_states: int = 4,
        state_dim: int = 64,
        use_spatial_gate: bool = False,
        use_token_mixer: bool = False,
        token_mixer_layers: int = 1,
    ) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.use_spatial_gate = bool(use_spatial_gate)
        self.use_token_mixer = bool(use_token_mixer)

        gate_input = channels + state_dim + num_modalities
        gate_hidden = max(gate_input // 2, 16)
        self.base_proj = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.delta_proj = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input, gate_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(gate_hidden, channels),
            nn.Sigmoid(),
        )

        if self.use_token_mixer:
            self.token_mixer = ModalityTokenMixer(
                channels=channels,
                state_dim=state_dim,
                num_modalities=num_modalities,
                num_heads=4,
                num_layers=token_mixer_layers,
            )

        if self.use_spatial_gate:
            spatial_hidden = max(channels // 2, 16)
            self.spatial_gate = nn.Sequential(
                nn.Conv3d(channels + state_dim, spatial_hidden, kernel_size=1, bias=False),
                nn.InstanceNorm3d(spatial_hidden, affine=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(spatial_hidden, 1, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )

        self.res_scale_logit = nn.Parameter(torch.tensor(-1.2))
        self.out_norm = nn.InstanceNorm3d(channels, affine=False)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        assignment_maps: Sequence[torch.Tensor],
        state: torch.Tensor,
        mask: torch.Tensor,
        return_debug: bool = False,
    ):
        if len(features) != self.num_modalities or len(assignment_maps) != self.num_modalities:
            raise ValueError("Expected one feature and assignment tensor per modality.")
        batch, channels, height, width, depth = features[0].shape
        subset_code = mask.float()

        global_features = torch.stack([f.mean(dim=(2, 3, 4)) for f in features], dim=1)
        assignment_means = torch.stack([a.mean(dim=(2, 3, 4)) for a in assignment_maps], dim=1)
        state_context = torch.einsum("bmk,bkd->bmd", assignment_means, state)

        token_gates = None
        if self.use_token_mixer:
            token_gates = self.token_mixer(global_features, state_context, mask)

        base_sum = torch.zeros_like(features[0])
        residual_sum = torch.zeros_like(features[0])
        denominator = mask.float().sum(dim=1).view(batch, 1, 1, 1, 1).clamp_min(1.0)
        local_gate_list: List[torch.Tensor] = []
        final_gate_list: List[torch.Tensor] = []

        for modality in range(self.num_modalities):
            availability = mask[:, modality].to(dtype=features[modality].dtype).view(batch, 1, 1, 1, 1)
            feature = features[modality]
            assignment = assignment_maps[modality]
            base_sum = base_sum + self.base_proj(feature) * availability

            gate_input = torch.cat(
                [global_features[:, modality], state_context[:, modality], subset_code], dim=1
            )
            local_gate = self.gate_mlp(gate_input)
            final_gate = (
                0.5 * local_gate + 0.5 * token_gates[:, modality]
                if token_gates is not None
                else local_gate
            )
            residual = self.delta_proj(feature) * final_gate.view(batch, channels, 1, 1, 1)

            if self.use_spatial_gate:
                state_map = torch.einsum("bkxyz,bkd->bdxyz", assignment, state).contiguous()
                spatial = 0.5 + 0.5 * self.spatial_gate(torch.cat([residual, state_map], dim=1))
                residual = residual * spatial

            residual_sum = residual_sum + residual * availability
            local_gate_list.append(local_gate)
            final_gate_list.append(final_gate)

        base = base_sum / denominator
        residual = residual_sum / denominator
        alpha = torch.sigmoid(self.res_scale_logit).to(dtype=base.dtype)
        fused = self.out_norm(base + alpha * residual)

        if not return_debug:
            return fused

        local_gates = torch.stack(local_gate_list, dim=1)
        final_gates = torch.stack(final_gate_list, dim=1)
        debug = {
            "global_features": global_features,
            "state_context": state_context,
            "local_gates": local_gates,
            "token_gates": token_gates,
            "final_gates": final_gates,
            "alpha": alpha,
        }
        return fused, debug
