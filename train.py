#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from data.data_utils import seed_worker
from data.datasets import BraTSEvalDataset, BraTSTrainDataset
from data.transforms import (
    Compose,
    NumpyType,
    RandomCrop3D,
    RandomFlip3D,
    RandomIntensityChange,
    RandomRotation3D,
)
from evaluation import evaluate_subset, summarize_rows, write_csv
from models import TSMNet, normalize_variant
from utils.losses import (
    build_optimizer_parameter_groups,
    cross_stage_identity_loss,
    multistage_semantic_state_loss,
    segmentation_loss,
    slot_diversity_loss,
    state_consistency_loss,
    state_health_metrics,
)
from utils.masks import subset_names
from utils.sampling import auxiliary_weight, sample_modality_mask

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


def parse_stage_weights(spec: str) -> Dict[int, float]:
    output: Dict[int, float] = {}
    for item in spec.split(","):
        stage_text, weight_text = item.split(":", 1)
        stage, weight = int(stage_text.strip()), float(weight_text.strip())
        if stage not in (1, 2, 3, 4) or weight < 0:
            raise ValueError(f"Invalid state-stage weight {item!r}")
        output[stage] = weight
    if not output:
        raise ValueError("At least one state-stage weight is required.")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the corrected PMSM implementation without test-set access."
    )
    parser.add_argument("--datapath", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_split", default="train_late15_v6.txt")
    parser.add_argument("--val_split", default="val_late15_v6.txt")
    parser.add_argument(
        "--variant",
        default="stage234",
        choices=[
            "stage234",
            "stage1234",
            "stage_local",
            "stage1234_local",
            "no_state_memory",
        ],
    )
    parser.add_argument("--resume", default="")

    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--iter_per_epoch", type=int, default=900)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--inference_crop_size", type=int, default=128)
    parser.add_argument("--inference_overlap", type=float, default=0.5)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "adamw"],
                        help="Adam matches the original protocol; AdamW is available for controlled experiments.")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--state_dim", type=int, default=64)
    parser.add_argument("--num_states", type=int, default=4)
    parser.add_argument(
        "--token_mixer_scope",
        default="stage4",
        choices=["none", "stage4", "active"],
        help=(
            "Where to use modality-token mixing: none, stage4 only, or every "
            "active state/fusion stage."
        ),
    )
    parser.add_argument(
        "--disable_token_mixer",
        action="store_true",
        help="Deprecated compatibility flag; equivalent to --token_mixer_scope none.",
    )
    parser.add_argument("--use_spatial_gate", action="store_true")

    parser.add_argument("--lambda_aux", type=float, default=0.50)
    parser.add_argument("--lambda_region", type=float, default=0.35)
    parser.add_argument("--lambda_state", type=float, default=0.05)
    parser.add_argument("--lambda_consistency", type=float, default=0.40)
    parser.add_argument("--lambda_identity", type=float, default=0.02)
    parser.add_argument("--lambda_diversity", type=float, default=0.005)
    parser.add_argument("--consistency_map_weight", type=float, default=0.50)
    parser.add_argument("--identity_temperature", type=float, default=0.10)
    parser.add_argument("--diversity_margin", type=float, default=0.85)
    parser.add_argument("--state_stage_weights", default="2:1.0,3:0.5,4:0.25")
    parser.add_argument("--warmup_consistency", type=int, default=40)
    parser.add_argument("--warmup_identity", type=int, default=20)
    parser.add_argument("--warmup_diversity", type=int, default=20)
    parser.add_argument("--mask_sampling", default="uniform", choices=["uniform", "hard_balanced"])

    parser.add_argument("--rotation_degrees", type=int, default=10)
    parser.add_argument("--rotation_probability", type=float, default=1.0)
    parser.add_argument("--flip_probability", type=float, default=0.5)
    parser.add_argument("--intensity_shift", type=float, default=0.1)
    parser.add_argument("--intensity_scale", type=float, default=0.1)

    parser.add_argument("--validation_start", type=int, default=700)
    parser.add_argument("--validation_every", type=int, default=25)
    parser.add_argument("--validation_subsets", default="all")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--last_save_every", type=int, default=5,
                        help="Write the resumable model_last checkpoint every N epochs.")
    parser.add_argument("--no_save_validation_candidates", action="store_true",
                        help="Do not save lightweight checkpoints at validation epochs.")
    parser.add_argument("--health_every", type=int, default=25, help="Training iterations between state-health samples.")
    parser.add_argument("--collapse_check_start", type=int, default=80)
    parser.add_argument("--collapse_patience", type=int, default=5)
    parser.add_argument("--no_abort_on_collapse", action="store_true")

    parser.add_argument("--dry_run_steps", type=int, default=0, help="Run only N optimization iterations, then exit.")
    return parser.parse_args()


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(), logging.FileHandler(output_dir / "training.log", encoding="utf-8")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def append_csv(path: Path, row: Mapping[str, object]) -> None:
    exists = path.is_file()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def poly_learning_rate(base_lr: float, epoch: int, num_epochs: int, power: float = 0.9) -> float:
    return float(base_lr * max(0.0, 1.0 - epoch / float(max(1, num_epochs))) ** power)


def set_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def model_config(args: argparse.Namespace) -> Dict[str, object]:
    token_mixer_scope = "none" if args.disable_token_mixer else args.token_mixer_scope
    return {
        "num_classes": 4,
        "state_dim": args.state_dim,
        "num_states": args.num_states,
        "variant": normalize_variant(args.variant),
        "basic_dims": 16,
        "use_spatial_gate": bool(args.use_spatial_gate),
        "token_mixer_scope": token_mixer_scope,
    }


def save_checkpoint(
    path: Path,
    model: TSMNet,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    args: argparse.Namespace,
    epoch: int,
    best_score: float,
    best_epoch: int,
) -> None:
    checkpoint = {
        "format_version": 2,
        "epoch": int(epoch),
        "best_val_score": float(best_score),
        "best_epoch": int(best_epoch),
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "model_config": model_config(args),
        "training_config": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool))
        },
    }
    torch.save(checkpoint, path)


def save_model_only_checkpoint(
    path: Path,
    model: TSMNet,
    args: argparse.Namespace,
    epoch: int,
    score: Optional[float] = None,
    score_subsets: Optional[Sequence[str]] = None,
) -> None:
    """Save a compact evaluation checkpoint without optimizer state."""
    checkpoint = {
        "format_version": 2,
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "model_config": model_config(args),
        "validation_score": None if score is None else float(score),
        "validation_subsets": list(score_subsets or []),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def create_loaders(args: argparse.Namespace, device: torch.device):
    crop = (args.crop_size,) * 3
    train_transforms = Compose(
        [
            RandomCrop3D(crop),
            RandomRotation3D(args.rotation_degrees, args.rotation_probability),
            RandomIntensityChange(args.intensity_shift, args.intensity_scale),
            RandomFlip3D(args.flip_probability),
            NumpyType((np.float32, np.int64)),
        ]
    )
    eval_transforms = Compose([NumpyType((np.float32, np.int64))])
    train_dataset = BraTSTrainDataset(
        args.datapath, args.train_split, transforms=train_transforms, num_classes=4
    )
    val_dataset = BraTSEvalDataset(
        args.datapath, args.val_split, transforms=eval_transforms, num_classes=4
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    return train_loader, val_loader


def next_batch(loader: DataLoader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def auxiliary_objective(
    auxiliary_predictions,
    target: torch.Tensor,
    mask: torch.Tensor,
    region_weight: float,
) -> torch.Tensor:
    losses = []
    for modality, prediction in enumerate(auxiliary_predictions or []):
        if prediction is None:
            continue
        present = mask[:, modality].bool()
        if not present.any():
            continue
        loss, parts = segmentation_loss(
            prediction[present], target[present], region_weight=0.15 * region_weight
        )
        losses.append(loss)
    return torch.stack(losses).mean() if losses else target.new_zeros(())


def detach_stage_details(details: Mapping[int, Mapping[str, object]]):
    detached = {}
    for stage, item in details.items():
        detached[stage] = {
            "mean_assignment": item["mean_assignment"].detach(),
            "state": item["state"].detach(),
        }
    return detached


@torch.inference_mode()
def validate(
    model: TSMNet,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    epoch: int,
    output_dir: Path,
) -> float:
    selected = subset_names(args.validation_subsets)
    all_rows = []
    model.eval()
    for subset in selected:
        logging.info("Validation epoch %d | subset %s", epoch, subset)
        all_rows.extend(
            evaluate_subset(
                model,
                val_loader,
                subset_name=subset,
                device=device,
                crop_size=args.inference_crop_size,
                overlap=args.inference_overlap,
                amp=args.amp,
            )
        )
    summaries, overall = summarize_rows(all_rows)
    epoch_dir = output_dir / "validation" / f"epoch_{epoch:04d}"
    write_csv(str(epoch_dir / "per_subject.csv"), all_rows)
    write_csv(str(epoch_dir / "subset_summary.csv"), summaries)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    with open(epoch_dir / "overall_summary.json", "w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2)
    logging.info("Validation epoch %d | all-subset macro Dice %.6f", epoch, overall["macro_dice"])
    return float(overall["macro_dice"])


def aggregate_health(samples: Sequence[Mapping[int, Mapping[str, float]]]):
    result: Dict[int, Dict[str, float]] = {}
    stages = sorted({stage for sample in samples for stage in sample})
    for stage in stages:
        keys = sorted({key for sample in samples if stage in sample for key in sample[stage]})
        result[stage] = {
            key: float(np.mean([sample[stage][key] for sample in samples if stage in sample]))
            for key in keys
        }
    return result


def is_collapsed(health: Mapping[int, Mapping[str, float]]) -> bool:
    deep = [health[stage] for stage in (3, 4) if stage in health]
    if not deep:
        return False
    return all(
        item["normalized_entropy"] > 0.995
        and item["mean_max_probability"] < 0.27
        and item["mean_offdiag_token_cosine"] > 0.95
        for item in deep
    )


def main() -> None:
    args = parse_args()
    if args.num_epochs <= 0 or args.iter_per_epoch <= 0:
        raise ValueError("num_epochs and iter_per_epoch must be positive.")
    if args.batch_size <= 0 or args.accum_steps <= 0:
        raise ValueError("batch_size and accum_steps must be positive.")
    if args.crop_size % 8 != 0:
        raise ValueError("crop_size must be divisible by 8 for the four-stage encoder.")
    if not 0.0 <= args.inference_overlap < 1.0:
        raise ValueError("inference_overlap must lie in [0,1).")
    stage_weights = parse_stage_weights(args.state_stage_weights)

    output_dir = Path(args.output_dir).resolve()
    setup_logging(output_dir)
    seed_everything(args.seed, args.deterministic)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    logging.info("Device: %s | AMP: %s | optimizer: %s", device, amp_enabled, args.optimizer)

    train_loader, val_loader = create_loaders(args, device)
    model = TSMNet(**model_config(args)).to(device)
    parameter_groups = build_optimizer_parameter_groups(model, args.weight_decay)
    optimizer_class = torch.optim.Adam if args.optimizer == "adam" else torch.optim.AdamW
    optimizer = optimizer_class(
        parameter_groups,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=True,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    writer = SummaryWriter(str(output_dir / "tensorboard")) if SummaryWriter else None

    start_epoch = 0
    best_score = -math.inf
    best_epoch = -1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_score = float(checkpoint.get("best_val_score", -math.inf))
        best_epoch = int(checkpoint.get("best_epoch", -1))
        logging.info("Resumed %s at epoch %d", args.resume, start_epoch)

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        config = dict(vars(args))
        config["model_config"] = model_config(args)
        config["parsed_state_stage_weights"] = stage_weights
        config["test_set_accessed_by_training"] = False
        json.dump(config, handle, indent=2)

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    logging.info(
        "Train cases=%d | validation cases=%d | parameters=%.3fM | trainable=%.3fM",
        len(train_loader.dataset), len(val_loader.dataset), total_parameters / 1e6, trainable_parameters / 1e6,
    )

    model.train()
    train_iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    global_step = start_epoch * args.iter_per_epoch
    collapse_streak = 0
    dry_steps_remaining = int(args.dry_run_steps)
    start_time = time.time()

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        learning_rate = poly_learning_rate(args.lr, epoch, args.num_epochs)
        set_learning_rate(optimizer, learning_rate)
        aux_weight = auxiliary_weight(epoch, args.num_epochs, args.lambda_aux)
        epoch_health = []
        epoch_loss_sum = 0.0
        epoch_started = time.time()

        for iteration in range(args.iter_per_epoch):
            batch, train_iterator = next_batch(train_loader, train_iterator)
            volume, target = batch[0], batch[1]
            volume = volume.to(device=device, dtype=torch.float32, non_blocking=True)
            target = target.to(device=device, dtype=torch.float32, non_blocking=True)
            mask = sample_modality_mask(
                epoch, volume.shape[0], device=device, mode=args.mask_sampling
            )
            full_mask = torch.ones_like(mask, dtype=torch.bool)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                encoded = model.encode_modalities(volume)
                # The complete-modality reference reuses the encoded feature
                # tensors and skips fusion/decoding.  It is a stop-gradient
                # target and therefore does not duplicate the encoder graph.
                with torch.no_grad():
                    full_output = model.forward_from_encoded(
                        encoded,
                        full_mask,
                        compute_prediction=False,
                        compute_auxiliary=False,
                        return_debug=False,
                    )
                    full_details = detach_stage_details(full_output["stage_details"])

                output = model.forward_from_encoded(
                    encoded,
                    mask,
                    compute_prediction=True,
                    compute_auxiliary=True,
                    return_debug=False,
                )
                prediction = output["prediction"]
                stage_details = output["stage_details"]

                fused_loss, fused_parts = segmentation_loss(
                    prediction, target, region_weight=args.lambda_region
                )
                aux_loss = auxiliary_objective(
                    output["auxiliary"], target, mask, args.lambda_region
                )

                if stage_details and args.lambda_state > 0:
                    state_loss, state_parts = multistage_semantic_state_loss(
                        stage_details, target, mask, stage_weights=stage_weights
                    )
                else:
                    state_loss, state_parts = target.new_zeros(()), {}

                if (
                    stage_details
                    and args.lambda_consistency > 0
                    and epoch >= args.warmup_consistency
                ):
                    consistency_loss, consistency_parts = state_consistency_loss(
                        stage_details,
                        full_details,
                        stage_weights=stage_weights,
                        map_weight=args.consistency_map_weight,
                    )
                else:
                    consistency_loss = target.new_zeros(())
                    consistency_parts = {"token": target.new_zeros(()), "map": target.new_zeros(())}

                if (
                    normalize_variant(args.variant) in ("stage234", "stage1234")
                    and stage_details
                    and args.lambda_identity > 0
                    and epoch >= args.warmup_identity
                ):
                    identity_loss = cross_stage_identity_loss(
                        stage_details, temperature=args.identity_temperature
                    )
                else:
                    identity_loss = target.new_zeros(())

                if (
                    stage_details
                    and args.lambda_diversity > 0
                    and epoch >= args.warmup_diversity
                ):
                    diversity_loss = slot_diversity_loss(
                        stage_details, margin=args.diversity_margin
                    )
                else:
                    diversity_loss = target.new_zeros(())

                total_loss = (
                    fused_loss
                    + aux_weight * aux_loss
                    + args.lambda_state * state_loss
                    + args.lambda_consistency * consistency_loss
                    + args.lambda_identity * identity_loss
                    + args.lambda_diversity * diversity_loss
                )

            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    "Non-finite loss: "
                    f"total={total_loss.item()} fused={fused_loss.item()} aux={aux_loss.item()} "
                    f"state={state_loss.item()} cons={consistency_loss.item()} "
                    f"identity={identity_loss.item()} diversity={diversity_loss.item()}"
                )

            scaler.scale(total_loss / args.accum_steps).backward()
            should_step = ((iteration + 1) % args.accum_steps == 0) or (
                iteration + 1 == args.iter_per_epoch
            )
            if should_step:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            epoch_loss_sum += float(total_loss.item())
            if args.health_every > 0 and (iteration % args.health_every == 0):
                epoch_health.append(state_health_metrics(stage_details, mask))

            log_row = {
                "epoch": epoch + 1,
                "iteration": iteration + 1,
                "global_step": global_step,
                "lr": learning_rate,
                "total": float(total_loss.item()),
                "fused": float(fused_loss.item()),
                "ce": float(fused_parts["ce"].item()),
                "dice": float(fused_parts["dice"].item()),
                "region": float(fused_parts["region"].item()),
                "aux": float(aux_loss.item()),
                "aux_weight": aux_weight,
                "state": float(state_loss.item()),
                "consistency": float(consistency_loss.item()),
                "consistency_token": float(consistency_parts["token"].item()),
                "consistency_map": float(consistency_parts["map"].item()),
                "identity": float(identity_loss.item()),
                "diversity": float(diversity_loss.item()),
            }
            if iteration % 10 == 0 or iteration + 1 == args.iter_per_epoch:
                logging.info(
                    "E%04d I%04d/%04d | total %.4f seg %.4f aux %.4f state %.4f cons %.4f id %.4f div %.4f",
                    epoch + 1,
                    iteration + 1,
                    args.iter_per_epoch,
                    total_loss.item(),
                    fused_loss.item(),
                    aux_loss.item(),
                    state_loss.item(),
                    consistency_loss.item(),
                    identity_loss.item(),
                    diversity_loss.item(),
                )
                append_csv(output_dir / "training_metrics.csv", log_row)
            if writer is not None:
                for key, value in log_row.items():
                    if key not in {"epoch", "iteration", "global_step"}:
                        writer.add_scalar(f"train/{key}", value, global_step)

            if dry_steps_remaining > 0:
                dry_steps_remaining -= 1
                if dry_steps_remaining == 0:
                    save_checkpoint(
                        output_dir / "model_dry_run.pth",
                        model,
                        optimizer,
                        scaler,
                        args,
                        epoch,
                        best_score,
                        best_epoch,
                    )
                    logging.info("Dry run completed successfully after %d steps.", args.dry_run_steps)
                    if writer is not None:
                        writer.close()
                    return

        mean_epoch_loss = epoch_loss_sum / args.iter_per_epoch
        health = aggregate_health(epoch_health) if epoch_health else {}
        for stage, values in health.items():
            row = {"epoch": epoch + 1, "stage": stage, **values}
            append_csv(output_dir / "state_health_training.csv", row)
            logging.info(
                "Health s%d | entropy %.4f maxP %.4f tokenCos %.4f logitStd %.5f",
                stage,
                values["normalized_entropy"],
                values["mean_max_probability"],
                values["mean_offdiag_token_cosine"],
                values["mean_logit_std_across_slots"],
            )
            if writer is not None:
                for key, value in values.items():
                    writer.add_scalar(f"health/stage{stage}/{key}", value, epoch + 1)

        if epoch + 1 >= args.collapse_check_start and is_collapsed(health):
            collapse_streak += 1
            logging.warning(
                "Deep-state collapse warning %d/%d. Check state_health_training.csv.",
                collapse_streak,
                args.collapse_patience,
            )
        else:
            collapse_streak = 0
        if (
            collapse_streak >= args.collapse_patience
            and not args.no_abort_on_collapse
        ):
            save_checkpoint(
                output_dir / "model_aborted_collapse.pth",
                model,
                optimizer,
                scaler,
                args,
                epoch,
                best_score,
                best_epoch,
            )
            raise RuntimeError(
                "Training stopped because stages 3 and 4 met the collapse criteria "
                f"for {collapse_streak} consecutive epochs."
            )

        if (epoch + 1) % max(1, args.last_save_every) == 0 or epoch + 1 == args.num_epochs:
            save_checkpoint(
                output_dir / "model_last.pth",
                model,
                optimizer,
                scaler,
                args,
                epoch,
                best_score,
                best_epoch,
            )
        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.num_epochs:
            save_checkpoint(
                output_dir / f"model_epoch_{epoch + 1:04d}.pth",
                model,
                optimizer,
                scaler,
                args,
                epoch,
                best_score,
                best_epoch,
            )

        should_validate = (
            epoch + 1 >= args.validation_start
            and (
                (epoch + 1 - args.validation_start) % max(1, args.validation_every) == 0
                or epoch + 1 == args.num_epochs
            )
        )
        if should_validate:
            score = validate(model, val_loader, args, device, epoch + 1, output_dir)
            validation_names = subset_names(args.validation_subsets)
            validation_tag = "all15" if len(validation_names) == 15 else "screening"
            if writer is not None:
                writer.add_scalar(f"validation/{validation_tag}_macro_dice", score, epoch + 1)
            if not args.no_save_validation_candidates:
                save_model_only_checkpoint(
                    output_dir / "validation_candidates" / f"model_val_epoch_{epoch + 1:04d}.pth",
                    model, args, epoch + 1, score=score, score_subsets=validation_names,
                )
            if score > best_score:
                best_score, best_epoch = score, epoch + 1
                save_checkpoint(
                    output_dir / f"model_best_{validation_tag}.pth",
                    model,
                    optimizer,
                    scaler,
                    args,
                    epoch,
                    best_score,
                    best_epoch,
                )
                logging.info(
                    "New best %s checkpoint: epoch %d, score %.6f",
                    validation_tag, best_epoch, best_score,
                )
            model.train()

        logging.info(
            "Epoch %d complete | mean loss %.4f | %.1fs | best %.6f at %d",
            epoch + 1,
            mean_epoch_loss,
            time.time() - epoch_started,
            best_score,
            best_epoch,
        )

    logging.info("Training completed in %.2f hours.", (time.time() - start_time) / 3600.0)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
