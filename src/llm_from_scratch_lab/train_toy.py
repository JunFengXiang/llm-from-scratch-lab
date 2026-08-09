"""Command-line entry point for the repository's first training gate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from torch import nn
from torch.optim import AdamW

from llm_from_scratch_lab.data import create_dataloaders, isolate_one_batch
from llm_from_scratch_lab.models import ToyMLP
from llm_from_scratch_lab.training import fit, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the minimal MLP learning experiment.")
    parser.add_argument("--config", type=Path, default=Path("configs/toy_mlp.json"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--overfit-one-batch",
        action="store_true",
        help="Repeat one batch to verify that the training code can memorize it.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["seed"])
    device = resolve_device(config["device"])

    train_loader, val_loader = create_dataloaders(
        **config["data"],
        seed=config["seed"],
    )
    if args.overfit_one_batch:
        train_loader = isolate_one_batch(train_loader)
        val_loader = train_loader

    model = ToyMLP(**config["model"])
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    run_name = args.run_name or datetime.now().strftime("toy_mlp_%Y%m%d_%H%M%S")
    if args.overfit_one_batch:
        run_name += "_one_batch"
    output_dir = Path(config["output_dir"])
    run_dir = output_dir / "runs" / run_name
    checkpoint_dir = output_dir / "checkpoints" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)

    print(f"device={device} run={run_name}")
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=config["training"]["epochs"],
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        config=config,
        resume_from=args.resume,
        grad_clip_norm=config["training"].get("grad_clip_norm"),
    )
    print(f"metrics={run_dir / 'metrics.csv'}")
    print(f"checkpoints={checkpoint_dir}")


if __name__ == "__main__":
    main()
