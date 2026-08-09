from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.optim import SGD

from llm_from_scratch_lab.data import create_dataloaders
from llm_from_scratch_lab.models import ToyMLP
from llm_from_scratch_lab.training import (
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)


def test_model_output_shape() -> None:
    model = ToyMLP(input_dim=2, hidden_dim=8, num_classes=2)
    logits = model(torch.randn(5, 2))
    assert logits.shape == (5, 2)


def test_training_updates_parameters() -> None:
    train_loader, _ = create_dataloaders(num_samples=64, batch_size=32, seed=7)
    model = ToyMLP(hidden_dim=8)
    original_parameters = deepcopy(list(model.parameters()))
    optimizer = SGD(model.parameters(), lr=0.1)

    train_one_epoch(
        model,
        train_loader,
        nn.CrossEntropyLoss(),
        optimizer,
        torch.device("cpu"),
        show_progress=False,
    )

    assert any(
        not torch.equal(before, after)
        for before, after in zip(original_parameters, model.parameters(), strict=True)
    )


def test_short_training_reduces_validation_loss() -> None:
    train_loader, val_loader = create_dataloaders(
        num_samples=256, batch_size=64, seed=11
    )
    model = ToyMLP(hidden_dim=16)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.2)
    device = torch.device("cpu")
    initial_loss = evaluate(model, val_loader, criterion, device).loss

    for _ in range(8):
        train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            show_progress=False,
        )

    final_loss = evaluate(model, val_loader, criterion, device).loss
    assert final_loss < initial_loss


def test_checkpoint_round_trip(tmp_path) -> None:
    model = ToyMLP(hidden_dim=8)
    optimizer = SGD(model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        epoch=2,
        global_step=17,
        model=model,
        optimizer=optimizer,
        best_val_loss=0.25,
        config={"seed": 42},
    )

    restored_model = ToyMLP(hidden_dim=8)
    restored_optimizer = SGD(restored_model.parameters(), lr=0.1)
    next_epoch, global_step, best_val_loss = load_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        torch.device("cpu"),
    )

    assert next_epoch == 3
    assert global_step == 17
    assert best_val_loss == 0.25
    for expected, actual in zip(model.parameters(), restored_model.parameters(), strict=True):
        assert torch.equal(expected, actual)
