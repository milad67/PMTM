# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# EPS = 1e-6


# def soft_multiclass_dice_loss(
#     probability: torch.Tensor,
#     target: torch.Tensor,
#     class_weights: Optional[torch.Tensor] = None,
#     eps: float = EPS,
# ) -> torch.Tensor:
#     """Mean soft Dice loss for one-hot or soft class targets."""
#     if probability.shape != target.shape:
#         raise ValueError(
#             f"Probability and target shapes differ: {tuple(probability.shape)} vs {tuple(target.shape)}"
#         )
#     dims = tuple(range(2, probability.ndim))
#     intersection = (probability * target).sum(dim=dims)
#     denominator = probability.sum(dim=dims) + target.sum(dim=dims)
#     dice = (2.0 * intersection + eps) / (denominator + eps)
#     if class_weights is None:
#         return 1.0 - dice.mean()
#     weights = class_weights.to(device=dice.device, dtype=dice.dtype).view(1, -1)
#     weights = weights / weights.sum().clamp_min(eps)
#     return 1.0 - (dice * weights).sum(dim=1).mean()


# def dynamic_weighted_cross_entropy(
#     probability: torch.Tensor,
#     target: torch.Tensor,
#     eps: float = EPS,
# ) -> torch.Tensor:
#     """Class-frequency weighted cross entropy used by the original protocol.

#     The weight for class ``c`` in each sample is ``1 - prevalence(c)``.  The
#     implementation is vectorized and numerically stable but otherwise preserves
#     the original objective.
#     """
#     if probability.shape != target.shape:
#         raise ValueError("Probability and target must have identical shapes.")
#     spatial_dims = tuple(range(2, target.ndim))
#     prevalence = target.mean(dim=spatial_dims)
#     weights = (1.0 - prevalence).clamp_min(0.0)
#     log_probability = torch.log(probability.clamp_min(eps))
#     weighted = -(target * log_probability * weights[(...,) + (None,) * len(spatial_dims)])
#     return weighted.sum(dim=1).mean()


# def soft_binary_dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = EPS) -> torch.Tensor:
#     dims = tuple(range(1, pred.ndim))
#     intersection = (pred * target).sum(dim=dims)
#     denominator = pred.sum(dim=dims) + target.sum(dim=dims)
#     dice = (2.0 * intersection + eps) / (denominator + eps)
#     return 1.0 - dice.mean()


# def focal_tversky_loss(
#     pred: torch.Tensor,
#     target: torch.Tensor,
#     alpha: float = 0.3,
#     beta: float = 0.7,
#     gamma: float = 0.75,
#     eps: float = EPS,
# ) -> torch.Tensor:
#     dims = tuple(range(1, pred.ndim))
#     true_positive = (pred * target).sum(dim=dims)
#     false_positive = (pred * (1.0 - target)).sum(dim=dims)
#     false_negative = ((1.0 - pred) * target).sum(dim=dims)
#     tversky = (true_positive + eps) / (
#         true_positive + alpha * false_positive + beta * false_negative + eps
#     )
#     return torch.pow(1.0 - tversky, gamma).mean()


# def brats_region_loss(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#     """Differentiable WT/TC/ET objective aligned with BraTS regions."""
#     if probability.shape[1] != 4 or target.shape[1] != 4:
#         raise ValueError("BraTS region loss expects four disjoint classes.")
#     p_ncr, p_ed, p_et = probability[:, 1], probability[:, 2], probability[:, 3]
#     t_ncr, t_ed, t_et = target[:, 1], target[:, 2], target[:, 3]

#     p_wt = (p_ncr + p_ed + p_et).clamp(0.0, 1.0)
#     t_wt = (t_ncr + t_ed + t_et).clamp(0.0, 1.0)
#     p_tc = (p_ncr + p_et).clamp(0.0, 1.0)
#     t_tc = (t_ncr + t_et).clamp(0.0, 1.0)

#     return (
#         0.20 * soft_binary_dice_loss(p_wt, t_wt)
#         + 0.30 * soft_binary_dice_loss(p_tc, t_tc)
#         + 0.50 * focal_tversky_loss(p_et, t_et)
#     )


# def segmentation_loss(
#     probability: torch.Tensor,
#     target: torch.Tensor,
#     region_weight: float = 0.35,
# ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#     ce = dynamic_weighted_cross_entropy(probability, target)
#     dice = soft_multiclass_dice_loss(probability, target)
#     region = brats_region_loss(probability, target)
#     total = ce + dice + float(region_weight) * region
#     return total, {"ce": ce, "dice": dice, "region": region}


# def downsample_semantic_target(target: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
#     """Area-average a one-hot target to a state-map resolution.

#     The resulting channels are local class proportions.  Their sum remains one
#     (up to floating-point error), including at tumor boundaries.
#     """
#     if target.ndim != 5:
#         raise ValueError(f"Expected target shape (B,C,H,W,D), got {tuple(target.shape)}")
#     output = F.adaptive_avg_pool3d(target.float(), output_size=tuple(int(v) for v in size))
#     return output / output.sum(dim=1, keepdim=True).clamp_min(EPS)


# def semantic_map_loss(
#     assignment: torch.Tensor,
#     target_soft: torch.Tensor,
#     class_weights: Sequence[float] = (0.25, 1.0, 1.0, 1.0),
# ) -> torch.Tensor:
#     weights = torch.as_tensor(
#         class_weights, dtype=assignment.dtype, device=assignment.device
#     )
#     if assignment.shape != target_soft.shape:
#         raise ValueError(
#             f"Assignment/target mismatch: {tuple(assignment.shape)} vs {tuple(target_soft.shape)}"
#         )
#     log_assignment = torch.log(assignment.clamp_min(EPS))
#     voxel_weight = (target_soft * weights.view(1, -1, 1, 1, 1)).sum(dim=1)
#     cross_entropy = -(
#         target_soft * log_assignment * weights.view(1, -1, 1, 1, 1)
#     ).sum(dim=1)
#     cross_entropy = cross_entropy.sum() / voxel_weight.sum().clamp_min(EPS)
#     dice = soft_multiclass_dice_loss(
#         assignment, target_soft, class_weights=weights
#     )
#     return cross_entropy + dice


# def multistage_semantic_state_loss(
#     stage_details: Mapping[int, Mapping[str, object]],
#     target: torch.Tensor,
#     modality_mask: torch.Tensor,
#     stage_weights: Mapping[int, float] = {2: 1.0, 3: 0.5, 4: 0.25},
#     include_mean_map: bool = True,
# ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
#     """Supervise ordered BG/NCR/ED/ET maps at every active stage.

#     Each available modality-specific map is supervised, and the masked-average
#     map is optionally supervised as well.  Unavailable modalities never enter
#     this objective.
#     """
#     if not stage_details:
#         zero = target.new_zeros(())
#         return zero, {}

#     per_stage: Dict[int, torch.Tensor] = {}
#     weighted_sum = target.new_zeros(())
#     weight_sum = 0.0
#     for stage in sorted(stage_details):
#         if stage not in stage_weights or float(stage_weights[stage]) <= 0:
#             continue
#         details = stage_details[stage]
#         mean_assignment = details["mean_assignment"]
#         target_soft = downsample_semantic_target(target, mean_assignment.shape[2:])
#         map_losses = []
#         if include_mean_map:
#             map_losses.append(semantic_map_loss(mean_assignment, target_soft))

#         assignments = details["assignments"]
#         for modality, assignment in enumerate(assignments):
#             present = modality_mask[:, modality].bool()
#             if present.any():
#                 map_losses.append(
#                     semantic_map_loss(assignment[present], target_soft[present])
#                 )
#         if not map_losses:
#             continue
#         stage_loss = torch.stack(map_losses).mean()
#         per_stage[int(stage)] = stage_loss
#         weight = float(stage_weights[stage])
#         weighted_sum = weighted_sum + weight * stage_loss
#         weight_sum += weight

#     if weight_sum == 0.0:
#         return target.new_zeros(()), per_stage
#     return weighted_sum / weight_sum, per_stage


# def _stage_weighted_average(
#     losses: Mapping[int, torch.Tensor], stage_weights: Mapping[int, float]
# ) -> torch.Tensor:
#     if not losses:
#         # This branch is only called when tensors are unavailable.  A caller
#         # should replace the CPU scalar with a device-local zero if necessary.
#         return torch.tensor(0.0)
#     numerator = None
#     denominator = 0.0
#     for stage, loss in losses.items():
#         weight = float(stage_weights.get(stage, 1.0))
#         numerator = weight * loss if numerator is None else numerator + weight * loss
#         denominator += weight
#     return numerator / max(denominator, EPS)


# def state_consistency_loss(
#     subset_details: Mapping[int, Mapping[str, object]],
#     full_details: Mapping[int, Mapping[str, object]],
#     stage_weights: Mapping[int, float] = {2: 1.0, 3: 0.5, 4: 0.25},
#     map_weight: float = 0.5,
# ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#     """Full-versus-incomplete consistency for ordered maps and state tokens."""
#     shared = sorted(set(subset_details).intersection(full_details))
#     if not shared:
#         reference = next(iter(subset_details.values()), None)
#         if reference is None:
#             return torch.tensor(0.0), {}
#         zero = reference["state"].new_zeros(())
#         return zero, {"token": zero, "map": zero}

#     token_losses: Dict[int, torch.Tensor] = {}
#     map_losses: Dict[int, torch.Tensor] = {}
#     for stage in shared:
#         subset_state = F.normalize(subset_details[stage]["state"], dim=-1)
#         full_state = F.normalize(full_details[stage]["state"].detach(), dim=-1)
#         token_losses[stage] = 1.0 - (subset_state * full_state).sum(dim=-1).mean()

#         p = subset_details[stage]["mean_assignment"].clamp_min(EPS)
#         q = full_details[stage]["mean_assignment"].detach().clamp_min(EPS)
#         middle = 0.5 * (p + q)
#         kl_pm = (p * (torch.log(p) - torch.log(middle))).sum(dim=1).mean()
#         kl_qm = (q * (torch.log(q) - torch.log(middle))).sum(dim=1).mean()
#         map_losses[stage] = 0.5 * (kl_pm + kl_qm)

#     token = _stage_weighted_average(token_losses, stage_weights)
#     map_term = _stage_weighted_average(map_losses, stage_weights)
#     total = token + float(map_weight) * map_term
#     return total, {"token": token, "map": map_term}


# def cross_stage_identity_loss(
#     stage_details: Mapping[int, Mapping[str, object]],
#     temperature: float = 0.10,
# ) -> torch.Tensor:
#     """Preserve ordered slot identity between consecutive encoder stages.

#     The diagonal of the KxK cosine matrix is treated as the correct match.  The
#     bidirectional cross-entropy also discourages all slots from becoming equal.
#     """
#     stages = sorted(stage_details)
#     if len(stages) < 2:
#         if stages:
#             return stage_details[stages[0]]["state"].new_zeros(())
#         return torch.tensor(0.0)
#     losses = []
#     for previous_stage, current_stage in zip(stages[:-1], stages[1:]):
#         previous = F.normalize(stage_details[previous_stage]["state"], dim=-1)
#         current = F.normalize(stage_details[current_stage]["state"], dim=-1)
#         similarity = torch.bmm(current, previous.transpose(1, 2)) / float(temperature)
#         labels = torch.arange(similarity.shape[1], device=similarity.device)
#         labels = labels.unsqueeze(0).expand(similarity.shape[0], -1).reshape(-1)
#         forward = F.cross_entropy(similarity.reshape(-1, similarity.shape[-1]), labels)
#         backward = F.cross_entropy(
#             similarity.transpose(1, 2).reshape(-1, similarity.shape[-1]), labels
#         )
#         losses.append(0.5 * (forward + backward))
#     return torch.stack(losses).mean()


# def slot_diversity_loss(
#     stage_details: Mapping[int, Mapping[str, object]],
#     margin: float = 0.85,
# ) -> torch.Tensor:
#     """Penalize excessive off-diagonal cosine similarity between state slots."""
#     losses = []
#     for stage in sorted(stage_details):
#         state = F.normalize(stage_details[stage]["state"], dim=-1)
#         similarity = torch.bmm(state, state.transpose(1, 2))
#         states = similarity.shape[1]
#         off_diagonal = ~torch.eye(states, dtype=torch.bool, device=similarity.device)
#         values = similarity[:, off_diagonal]
#         losses.append(F.relu(values - float(margin)).pow(2).mean())
#     if not losses:
#         return torch.tensor(0.0)
#     return torch.stack(losses).mean()


# @torch.no_grad()
# def state_health_metrics(
#     stage_details: Mapping[int, Mapping[str, object]],
#     modality_mask: Optional[torch.Tensor] = None,
# ) -> Dict[int, Dict[str, float]]:
#     """Small batch diagnostics used to detect map or token collapse early."""
#     output: Dict[int, Dict[str, float]] = {}
#     for stage, details in sorted(stage_details.items()):
#         assignment = details["mean_assignment"].float().clamp_min(EPS)
#         states = details["state"].float()
#         num_states = assignment.shape[1]
#         entropy = -(assignment * assignment.log()).sum(dim=1) / torch.log(
#             assignment.new_tensor(float(num_states))
#         )
#         max_probability = assignment.max(dim=1).values
#         deviation = (assignment - 1.0 / num_states).abs().mean()

#         normalized_state = F.normalize(states, dim=-1)
#         similarity = torch.bmm(normalized_state, normalized_state.transpose(1, 2))
#         off_diagonal = ~torch.eye(num_states, dtype=torch.bool, device=states.device)
#         token_cosine = similarity[:, off_diagonal].mean()

#         logits = torch.stack(details["assignment_logits"], dim=1)  # B,M,K,H,W,D
#         per_map_logit_std = logits.std(dim=2, unbiased=False).mean(dim=(2, 3, 4))
#         if modality_mask is not None:
#             available = modality_mask.to(device=logits.device, dtype=per_map_logit_std.dtype)
#             logit_std = (per_map_logit_std * available).sum() / available.sum().clamp_min(1.0)
#         else:
#             logit_std = per_map_logit_std.mean()
#         output[int(stage)] = {
#             "normalized_entropy": float(entropy.mean().item()),
#             "mean_max_probability": float(max_probability.mean().item()),
#             "mean_abs_deviation_from_uniform": float(deviation.item()),
#             "mean_offdiag_token_cosine": float(token_cosine.item()),
#             "mean_logit_std_across_slots": float(logit_std.item()),
#         }
#     return output


# def build_optimizer_parameter_groups(
#     model: nn.Module,
#     weight_decay: float,
# ) -> Sequence[Dict[str, object]]:
#     """Create decay/no-decay groups that protect state and gate endpoints.

#     The previous collapsed checkpoint applied weight decay to every parameter.
#     Here, biases, normalization/one-dimensional parameters, the final
#     assignment logits, recurrent state parameters, and final gate projections
#     are excluded from decay.  All remaining convolution/linear weights retain
#     the requested regularization.
#     """
#     no_decay_tokens = (
#         "memory.T0",
#         "memory.eta_logit",
#         "res_scale_logit",
#         "assign_branch.3.",
#         "gate_mlp.2.",
#         "token_mixer.out_proj.0.",
#     )
#     decay, no_decay = [], []
#     seen = set()
#     for name, parameter in model.named_parameters():
#         if not parameter.requires_grad:
#             continue
#         if id(parameter) in seen:
#             raise RuntimeError(f"Parameter {name} appears more than once.")
#         seen.add(id(parameter))
#         exclude = (
#             parameter.ndim <= 1
#             or name.endswith(".bias")
#             or any(token in name for token in no_decay_tokens)
#         )
#         (no_decay if exclude else decay).append(parameter)
#     return [
#         {"params": decay, "weight_decay": float(weight_decay), "group_name": "decay"},
#         {"params": no_decay, "weight_decay": 0.0, "group_name": "no_decay"},
#     ]


# # Backward-compatible alias retained for earlier drafts of the corrected code.
# build_adamw_parameter_groups = build_optimizer_parameter_groups

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-6


def soft_multiclass_dice_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    eps: float = EPS,
) -> torch.Tensor:
    """Mean soft Dice loss for one-hot or soft class targets."""
    if probability.shape != target.shape:
        raise ValueError(
            f"Probability and target shapes differ: {tuple(probability.shape)} vs {tuple(target.shape)}"
        )
    dims = tuple(range(2, probability.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    if class_weights is None:
        return 1.0 - dice.mean()
    weights = class_weights.to(device=dice.device, dtype=dice.dtype).view(1, -1)
    weights = weights / weights.sum().clamp_min(eps)
    return 1.0 - (dice * weights).sum(dim=1).mean()


def dynamic_weighted_cross_entropy(
    probability: torch.Tensor,
    target: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Class-frequency weighted cross entropy used by the original protocol.

    The weight for class ``c`` in each sample is ``1 - prevalence(c)``.  The
    implementation is vectorized and numerically stable but otherwise preserves
    the original objective.
    """
    if probability.shape != target.shape:
        raise ValueError("Probability and target must have identical shapes.")
    spatial_dims = tuple(range(2, target.ndim))
    prevalence = target.mean(dim=spatial_dims)
    weights = (1.0 - prevalence).clamp_min(0.0)
    log_probability = torch.log(probability.clamp_min(eps))
    weighted = -(target * log_probability * weights[(...,) + (None,) * len(spatial_dims)])
    return weighted.sum(dim=1).mean()


def soft_binary_dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    dims = tuple(range(1, pred.ndim))
    intersection = (pred * target).sum(dim=dims)
    denominator = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def focal_tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
    eps: float = EPS,
) -> torch.Tensor:
    dims = tuple(range(1, pred.ndim))
    true_positive = (pred * target).sum(dim=dims)
    false_positive = (pred * (1.0 - target)).sum(dim=dims)
    false_negative = ((1.0 - pred) * target).sum(dim=dims)
    tversky = (true_positive + eps) / (
        true_positive + alpha * false_positive + beta * false_negative + eps
    )
    return torch.pow(1.0 - tversky, gamma).mean()


def brats_region_loss(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable WT/TC/ET objective aligned with BraTS regions."""
    if probability.shape[1] != 4 or target.shape[1] != 4:
        raise ValueError("BraTS region loss expects four disjoint classes.")
    p_ncr, p_ed, p_et = probability[:, 1], probability[:, 2], probability[:, 3]
    t_ncr, t_ed, t_et = target[:, 1], target[:, 2], target[:, 3]

    p_wt = (p_ncr + p_ed + p_et).clamp(0.0, 1.0)
    t_wt = (t_ncr + t_ed + t_et).clamp(0.0, 1.0)
    p_tc = (p_ncr + p_et).clamp(0.0, 1.0)
    t_tc = (t_ncr + t_et).clamp(0.0, 1.0)

    return (
        0.20 * soft_binary_dice_loss(p_wt, t_wt)
        + 0.30 * soft_binary_dice_loss(p_tc, t_tc)
        + 0.50 * focal_tversky_loss(p_et, t_et)
    )


def segmentation_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    region_weight: float = 0.35,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    ce = dynamic_weighted_cross_entropy(probability, target)
    dice = soft_multiclass_dice_loss(probability, target)
    region = brats_region_loss(probability, target)
    total = ce + dice + float(region_weight) * region
    return total, {"ce": ce, "dice": dice, "region": region}


def downsample_semantic_target(target: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    """Area-average a one-hot target to a state-map resolution.

    The resulting channels are local class proportions.  Their sum remains one
    (up to floating-point error), including at tumor boundaries.
    """
    if target.ndim != 5:
        raise ValueError(f"Expected target shape (B,C,H,W,D), got {tuple(target.shape)}")
    output = F.adaptive_avg_pool3d(target.float(), output_size=tuple(int(v) for v in size))
    return output / output.sum(dim=1, keepdim=True).clamp_min(EPS)


def semantic_map_loss(
    assignment: torch.Tensor,
    target_soft: torch.Tensor,
    class_weights: Sequence[float] = (0.25, 1.0, 1.0, 1.0),
) -> torch.Tensor:
    weights = torch.as_tensor(
        class_weights, dtype=assignment.dtype, device=assignment.device
    )
    if assignment.shape != target_soft.shape:
        raise ValueError(
            f"Assignment/target mismatch: {tuple(assignment.shape)} vs {tuple(target_soft.shape)}"
        )
    log_assignment = torch.log(assignment.clamp_min(EPS))
    voxel_weight = (target_soft * weights.view(1, -1, 1, 1, 1)).sum(dim=1)
    cross_entropy = -(
        target_soft * log_assignment * weights.view(1, -1, 1, 1, 1)
    ).sum(dim=1)
    cross_entropy = cross_entropy.sum() / voxel_weight.sum().clamp_min(EPS)
    dice = soft_multiclass_dice_loss(
        assignment, target_soft, class_weights=weights
    )
    return cross_entropy + dice


def multistage_semantic_state_loss(
    stage_details: Mapping[int, Mapping[str, object]],
    target: torch.Tensor,
    modality_mask: torch.Tensor,
    stage_weights: Mapping[int, float] = {2: 1.0, 3: 0.5, 4: 0.25},
    include_mean_map: bool = True,
) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """Supervise ordered BG/NCR/ED/ET maps at every active stage.

    Each available modality-specific map is supervised, and the masked-average
    map is optionally supervised as well.  Unavailable modalities never enter
    this objective.
    """
    if not stage_details:
        zero = target.new_zeros(())
        return zero, {}

    per_stage: Dict[int, torch.Tensor] = {}
    weighted_sum = target.new_zeros(())
    weight_sum = 0.0
    for stage in sorted(stage_details):
        if stage not in stage_weights or float(stage_weights[stage]) <= 0:
            continue
        details = stage_details[stage]
        mean_assignment = details["mean_assignment"]
        target_soft = downsample_semantic_target(target, mean_assignment.shape[2:])
        map_losses = []
        if include_mean_map:
            map_losses.append(semantic_map_loss(mean_assignment, target_soft))

        assignments = details["assignments"]
        for modality, assignment in enumerate(assignments):
            present = modality_mask[:, modality].bool()
            if present.any():
                map_losses.append(
                    semantic_map_loss(assignment[present], target_soft[present])
                )
        if not map_losses:
            continue
        stage_loss = torch.stack(map_losses).mean()
        per_stage[int(stage)] = stage_loss
        weight = float(stage_weights[stage])
        weighted_sum = weighted_sum + weight * stage_loss
        weight_sum += weight

    if weight_sum == 0.0:
        return target.new_zeros(()), per_stage
    return weighted_sum / weight_sum, per_stage


def _stage_weighted_average(
    losses: Mapping[int, torch.Tensor], stage_weights: Mapping[int, float]
) -> torch.Tensor:
    if not losses:
        # This branch is only called when tensors are unavailable.  A caller
        # should replace the CPU scalar with a device-local zero if necessary.
        return torch.tensor(0.0)
    numerator = None
    denominator = 0.0
    for stage, loss in losses.items():
        weight = float(stage_weights.get(stage, 1.0))
        numerator = weight * loss if numerator is None else numerator + weight * loss
        denominator += weight
    return numerator / max(denominator, EPS)



def state_consistency_loss(
    subset_details: Mapping[int, Mapping[str, object]],
    full_details: Mapping[int, Mapping[str, object]],
    stage_weights: Mapping[int, float] = {2: 1.0, 3: 0.5, 4: 0.25},
    map_weight: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Full-versus-incomplete consistency for ordered maps and state tokens.

    Cosine normalization and KL calculations are evaluated in FP32 to avoid
    numerical instability when the surrounding training forward pass uses AMP.
    Casting with ``.float()`` preserves gradients for the subset branch.
    """
    shared = sorted(set(subset_details).intersection(full_details))

    if not shared:
        reference = next(iter(subset_details.values()), None)
        if reference is None:
            return torch.tensor(0.0), {}
        zero = reference["state"].new_zeros(())
        return zero, {"token": zero, "map": zero}

    token_losses: Dict[int, torch.Tensor] = {}
    map_losses: Dict[int, torch.Tensor] = {}

    with torch.cuda.amp.autocast(enabled=False):
        for stage in shared:
            subset_raw = subset_details[stage]["state"].float()
            full_raw = full_details[stage]["state"].detach().float()

            if not torch.isfinite(subset_raw).all():
                raise FloatingPointError(
                    f"Non-finite subset state before consistency loss at stage {stage}."
                )
            if not torch.isfinite(full_raw).all():
                raise FloatingPointError(
                    f"Non-finite full state before consistency loss at stage {stage}."
                )

            subset_state = F.normalize(
                subset_raw,
                p=2,
                dim=-1,
                eps=EPS,
            )
            full_state = F.normalize(
                full_raw,
                p=2,
                dim=-1,
                eps=EPS,
            )
            token_losses[stage] = (
                1.0
                - (subset_state * full_state)
                .sum(dim=-1)
                .mean()
            )

            p = (
                subset_details[stage]["mean_assignment"]
                .float()
                .clamp_min(EPS)
            )
            q = (
                full_details[stage]["mean_assignment"]
                .detach()
                .float()
                .clamp_min(EPS)
            )
            middle = 0.5 * (p + q)

            kl_pm = (
                p * (torch.log(p) - torch.log(middle))
            ).sum(dim=1).mean()
            kl_qm = (
                q * (torch.log(q) - torch.log(middle))
            ).sum(dim=1).mean()
            map_losses[stage] = 0.5 * (kl_pm + kl_qm)

        token = _stage_weighted_average(token_losses, stage_weights)
        map_term = _stage_weighted_average(map_losses, stage_weights)
        total = token + float(map_weight) * map_term

    if not torch.isfinite(total):
        raise FloatingPointError(
            "Non-finite state consistency loss after FP32 evaluation."
        )

    return total, {"token": token, "map": map_term}


def cross_stage_identity_loss(
    stage_details: Mapping[int, Mapping[str, object]],
    temperature: float = 0.10,
) -> torch.Tensor:
    """Preserve ordered slot identity between consecutive encoder stages.

    The complete normalization, similarity, temperature scaling, and
    cross-entropy path is evaluated in FP32 under CUDA AMP.
    """
    stages = sorted(stage_details)

    if len(stages) < 2:
        if stages:
            return stage_details[stages[0]]["state"].new_zeros(())
        return torch.tensor(0.0)

    if temperature <= 0:
        raise ValueError(
            f"temperature must be positive, got {temperature}."
        )

    losses = []

    with torch.cuda.amp.autocast(enabled=False):
        for previous_stage, current_stage in zip(
            stages[:-1],
            stages[1:],
        ):
            previous_raw = stage_details[previous_stage]["state"].float()
            current_raw = stage_details[current_stage]["state"].float()

            if not torch.isfinite(previous_raw).all():
                raise FloatingPointError(
                    f"Non-finite state before identity loss at stage {previous_stage}."
                )
            if not torch.isfinite(current_raw).all():
                raise FloatingPointError(
                    f"Non-finite state before identity loss at stage {current_stage}."
                )

            previous = F.normalize(
                previous_raw,
                p=2,
                dim=-1,
                eps=EPS,
            )
            current = F.normalize(
                current_raw,
                p=2,
                dim=-1,
                eps=EPS,
            )

            similarity = torch.bmm(
                current,
                previous.transpose(1, 2),
            ) / float(temperature)

            if not torch.isfinite(similarity).all():
                raise FloatingPointError(
                    "Non-finite cross-stage similarity: "
                    f"{previous_stage}->{current_stage}; "
                    f"previous_min_norm="
                    f"{previous_raw.norm(dim=-1).min().item():.6e}; "
                    f"current_min_norm="
                    f"{current_raw.norm(dim=-1).min().item():.6e}"
                )

            num_states = similarity.shape[1]
            labels = torch.arange(
                num_states,
                device=similarity.device,
                dtype=torch.long,
            )
            labels = (
                labels.unsqueeze(0)
                .expand(similarity.shape[0], -1)
                .reshape(-1)
            )

            forward = F.cross_entropy(
                similarity.reshape(-1, similarity.shape[-1]),
                labels,
            )
            backward = F.cross_entropy(
                similarity.transpose(1, 2).reshape(
                    -1,
                    similarity.shape[-1],
                ),
                labels,
            )
            pair_loss = 0.5 * (forward + backward)

            if not torch.isfinite(pair_loss):
                raise FloatingPointError(
                    "Non-finite identity pair loss for "
                    f"stages {previous_stage}->{current_stage}."
                )

            losses.append(pair_loss)

        identity_loss = torch.stack(losses).mean()

    if not torch.isfinite(identity_loss):
        raise FloatingPointError(
            "Non-finite cross-stage identity loss."
        )

    return identity_loss


def slot_diversity_loss(
    stage_details: Mapping[int, Mapping[str, object]],
    margin: float = 0.85,
) -> torch.Tensor:
    """Penalize excessive off-diagonal cosine similarity between state slots.

    The complete cosine path is evaluated in FP32 to avoid AMP-related
    normalization underflow on small state-token norms.
    """
    if not stage_details:
        return torch.tensor(0.0)

    losses = []

    with torch.cuda.amp.autocast(enabled=False):
        for stage in sorted(stage_details):
            raw_state = stage_details[stage]["state"].float()

            if not torch.isfinite(raw_state).all():
                raise FloatingPointError(
                    f"Non-finite state before diversity loss at stage {stage}."
                )

            state = F.normalize(
                raw_state,
                p=2,
                dim=-1,
                eps=EPS,
            )
            similarity = torch.bmm(
                state,
                state.transpose(1, 2),
            )

            if not torch.isfinite(similarity).all():
                raise FloatingPointError(
                    "Non-finite diversity similarity at "
                    f"stage {stage}; min_state_norm="
                    f"{raw_state.norm(dim=-1).min().item():.6e}."
                )

            num_states = similarity.shape[1]
            off_diagonal = ~torch.eye(
                num_states,
                dtype=torch.bool,
                device=similarity.device,
            )
            values = similarity[:, off_diagonal]
            stage_loss = F.relu(
                values - float(margin)
            ).pow(2).mean()

            if not torch.isfinite(stage_loss):
                raise FloatingPointError(
                    f"Non-finite diversity loss at stage {stage}."
                )

            losses.append(stage_loss)

        diversity_loss = torch.stack(losses).mean()

    if not torch.isfinite(diversity_loss):
        raise FloatingPointError(
            "Non-finite slot diversity loss."
        )

    return diversity_loss

@torch.no_grad()
def state_health_metrics(
    stage_details: Mapping[int, Mapping[str, object]],
    modality_mask: Optional[torch.Tensor] = None,
) -> Dict[int, Dict[str, float]]:
    """Small batch diagnostics used to detect map or token collapse early."""
    output: Dict[int, Dict[str, float]] = {}
    for stage, details in sorted(stage_details.items()):
        assignment = details["mean_assignment"].float().clamp_min(EPS)
        states = details["state"].float()
        num_states = assignment.shape[1]
        entropy = -(assignment * assignment.log()).sum(dim=1) / torch.log(
            assignment.new_tensor(float(num_states))
        )
        max_probability = assignment.max(dim=1).values
        deviation = (assignment - 1.0 / num_states).abs().mean()

        normalized_state = F.normalize(states, p=2, dim=-1, eps=EPS)
        similarity = torch.bmm(normalized_state, normalized_state.transpose(1, 2))
        off_diagonal = ~torch.eye(num_states, dtype=torch.bool, device=states.device)
        token_cosine = similarity[:, off_diagonal].mean()

        logits = torch.stack(details["assignment_logits"], dim=1)  # B,M,K,H,W,D
        per_map_logit_std = logits.std(dim=2, unbiased=False).mean(dim=(2, 3, 4))
        if modality_mask is not None:
            available = modality_mask.to(device=logits.device, dtype=per_map_logit_std.dtype)
            logit_std = (per_map_logit_std * available).sum() / available.sum().clamp_min(1.0)
        else:
            logit_std = per_map_logit_std.mean()
        state_norm = states.norm(dim=-1)
        output[int(stage)] = {
            "normalized_entropy": float(entropy.mean().item()),
            "mean_max_probability": float(max_probability.mean().item()),
            "mean_abs_deviation_from_uniform": float(deviation.item()),
            "mean_offdiag_token_cosine": float(token_cosine.item()),
            "mean_logit_std_across_slots": float(logit_std.item()),
            "mean_state_l2_norm": float(state_norm.mean().item()),
            "min_state_l2_norm": float(state_norm.min().item()),
            "max_state_l2_norm": float(state_norm.max().item()),
        }
    return output


def build_optimizer_parameter_groups(
    model: nn.Module,
    weight_decay: float,
) -> Sequence[Dict[str, object]]:
    """Create decay/no-decay groups that protect state and gate endpoints.

    The previous collapsed checkpoint applied weight decay to every parameter.
    Here, biases, normalization/one-dimensional parameters, the final
    assignment logits, recurrent state parameters, and final gate projections
    are excluded from decay.  All remaining convolution/linear weights retain
    the requested regularization.
    """
    no_decay_tokens = (
        "memory.T0",
        "memory.eta_logit",
        "res_scale_logit",
        "assign_branch.3.",
        "gate_mlp.2.",
        "token_mixer.out_proj.0.",
    )
    decay, no_decay = [], []
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise RuntimeError(f"Parameter {name} appears more than once.")
        seen.add(id(parameter))
        exclude = (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or any(token in name for token in no_decay_tokens)
        )
        (no_decay if exclude else decay).append(parameter)
    return [
        {"params": decay, "weight_decay": float(weight_decay), "group_name": "decay"},
        {"params": no_decay, "weight_decay": 0.0, "group_name": "no_decay"},
    ]


# Backward-compatible alias retained for earlier drafts of the corrected code.
build_adamw_parameter_groups = build_optimizer_parameter_groups