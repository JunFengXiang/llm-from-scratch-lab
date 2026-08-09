"""A visible, reusable training loop with logs and resumable checkpoints."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve auto/cpu/cuda/mps into a concrete PyTorch device."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed CPU and available accelerators for repeatable learning experiments."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    global_step: int = 0,
    writer: SummaryWriter | None = None,
    grad_clip_norm: float | None = None,
    show_progress: bool = True,
) -> tuple[EpochMetrics, int]:
    """Train for one epoch; the six essential update steps remain explicit."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    batches = tqdm(loader, desc="train", leave=False, disable=not show_progress)
    for inputs, targets in batches:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        if not math.isfinite(loss.item()):
            message = f"Non-finite loss detected at step {global_step}: {loss.item()}"
            raise FloatingPointError(message)
        loss.backward()
        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size

        if writer is not None:
            writer.add_scalar("loss/train_step", loss.item(), global_step)
        global_step += 1
        batches.set_postfix(loss=f"{loss.item():.4f}")

    metrics = EpochMetrics(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
    )
    return metrics, global_step


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    """Evaluate without gradients or parameter updates."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size

    return EpochMetrics(
        loss=total_loss / total_examples,
        accuracy=total_correct / total_examples,
    )


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: Optimizer,
    best_val_loss: float,
    config: dict[str, Any],
) -> None:
    """Atomically save everything required to resume training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": config,
    }
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(state, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[int, int, float]:
    """Restore model/optimizer state and return the next epoch and global step."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint["global_step"]),
        float(checkpoint["best_val_loss"]),
    )


def append_epoch_metrics(
    path: Path,
    epoch: int,
    train_metrics: EpochMetrics,
    val_metrics: EpochMetrics,
    learning_rate: float,
) -> None:
    """Append human-readable metrics that work without TensorBoard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "lr"],
        )
        if should_write_header:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": epoch,
                "train_loss": f"{train_metrics.loss:.8f}",
                "train_accuracy": f"{train_metrics.accuracy:.8f}",
                "val_loss": f"{val_metrics.loss:.8f}",
                "val_accuracy": f"{val_metrics.accuracy:.8f}",
                "lr": f"{learning_rate:.8g}",
            }
        )


def fit(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epochs: int,
    run_dir: Path,
    checkpoint_dir: Path,
    config: dict[str, Any],
    resume_from: Path | None = None,
    grad_clip_norm: float | None = None,
) -> None:
    """Run training, validation, logging, best-model selection, and checkpointing."""
    model.to(device)
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if resume_from is not None:
        start_epoch, global_step, best_val_loss = load_checkpoint(
            resume_from, model, optimizer, device
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)
    metrics_path = run_dir / "metrics.csv"

    try:
        for epoch in range(start_epoch, epochs):
            train_metrics, global_step = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                global_step=global_step,
                writer=writer,
                grad_clip_norm=grad_clip_norm,
            )
            val_metrics = evaluate(model, val_loader, criterion, device)
            learning_rate = optimizer.param_groups[0]["lr"]

            writer.add_scalars(
                "loss/epoch",
                {"train": train_metrics.loss, "validation": val_metrics.loss},
                epoch,
            )
            writer.add_scalars(
                "accuracy/epoch",
                {"train": train_metrics.accuracy, "validation": val_metrics.accuracy},
                epoch,
            )
            writer.add_scalar("learning_rate", learning_rate, epoch)
            append_epoch_metrics(
                metrics_path, epoch, train_metrics, val_metrics, learning_rate
            )

            improved = val_metrics.loss < best_val_loss
            if improved:
                best_val_loss = val_metrics.loss
            save_checkpoint(
                checkpoint_dir / "last.pt",
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                best_val_loss=best_val_loss,
                config=config,
            )
            if improved:
                save_checkpoint(
                    checkpoint_dir / "best.pt",
                    epoch=epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    best_val_loss=best_val_loss,
                    config=config,
                )

            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_metrics.loss:.4f} "
                f"val_loss={val_metrics.loss:.4f} "
                f"val_acc={val_metrics.accuracy:.2%}"
            )
            writer.flush()
    finally:
        writer.close()
