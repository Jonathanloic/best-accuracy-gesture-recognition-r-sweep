from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, GestureDataset, SampleRecord
from model import build_best_accuracy_model, count_trainable_parameters
from model_attention import build_best_accuracy_channel_temporal_model
from model_se import build_best_accuracy_se_model
from model_eca import build_best_accuracy_eca_model
from reproducibility import make_generator, seed_worker, set_global_seed


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 300
    batch_size: int = 128
    learning_rate: float = 1e-3
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    model_name: str = "best_accuracy"
    channel_reduction_ratio: int = 16
    se_reduction_ratio: int = 16


def select_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()

    if requested != "auto":
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")

    # Useful on Apple Silicon; CUDA still has priority as requested.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def make_loader(
    records: Sequence[SampleRecord],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        GestureDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_generator(seed),
        persistent_workers=num_workers > 0,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)
        pred = logits.argmax(dim=1)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (pred == y).sum().item()
        total_samples += batch_size

        y_cpu = y.detach().cpu().numpy()
        pred_cpu = pred.detach().cpu().numpy()
        for target, prediction in zip(y_cpu, pred_cpu):
            confusion[target, prediction] += 1

    per_class_precision: dict[str, float | None] = {}
    precision_values: list[float] = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        tp = confusion[class_idx, class_idx]
        predicted_as_class = confusion[:, class_idx].sum()
        if predicted_as_class == 0:
            per_class_precision[class_name] = None
        else:
            precision = float(tp / predicted_as_class)
            per_class_precision[class_name] = precision
            precision_values.append(precision)

    # Supplementary diagnostic. The paper reports "precision" but does not
    # state the averaging convention in the text, so this is explicitly named
    # macro precision rather than assumed to be identical to the paper metric.
    macro_precision = (
        float(np.mean(precision_values)) if precision_values else float("nan")
    )

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "macro_precision_diagnostic": macro_precision,
        "per_class_precision": per_class_precision,
        "confusion_matrix": confusion,
    }


def _save_history(history: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "train_accuracy"],
        )
        writer.writeheader()
        writer.writerows(history)


def _save_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, matrix.tolist()):
            writer.writerow([class_name, *row])


def run_fold(
    protocol: str,
    fold_index: int,
    num_folds: int,
    held_out_subject: str,
    train_records: Sequence[SampleRecord],
    test_records: Sequence[SampleRecord],
    output_dir: str | Path,
    config: TrainingConfig,
    calibration_session: str | None = None,
) -> dict:
    """Train one independent LOSO/AOS fold and evaluate test data once at end."""
    # Reset RNG before EVERY fold so folds are independent and reproducible.
    set_global_seed(config.seed)
    device = select_device(config.device)
    pin_memory = device.type == "cuda"

    train_loader = make_loader(
        train_records,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
        pin_memory=pin_memory,
    )
    test_loader = make_loader(
        test_records,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        seed=config.seed,
        pin_memory=pin_memory,
    )

    # A fresh model + optimizer is created inside every fold: no fold leakage.
    if config.model_name == "best_accuracy":
        model = build_best_accuracy_model(in_channels=4, num_classes=6).to(device)
    elif config.model_name == "best_accuracy_channel_temporal":
        model = build_best_accuracy_channel_temporal_model(
            in_channels=4,
            num_classes=6,
            reduction_ratio=config.channel_reduction_ratio,
        ).to(device)
    elif config.model_name == "best_accuracy_se":
        model = build_best_accuracy_se_model(
            in_channels=4,
            num_classes=6,
            reduction_ratio=config.se_reduction_ratio,
        ).to(device)
    elif config.model_name == "best_accuracy_eca":
        model = build_best_accuracy_eca_model(in_channels=4, num_classes=6).to(device)
    else:
        raise ValueError(f"Unknown model: {config.model_name}")

    # Always calculate the actual parameter count from the instantiated model.
    num_parameters = count_trainable_parameters(model)

    criterion = nn.CrossEntropyLoss()
    # Paper specifies AdamW + lr=1e-3 but does not give a weight-decay value.
    # Calling AdamW this way preserves PyTorch's optimizer default rather than
    # inventing a new explicit hyperparameter.
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    calibration_count = 0
    if protocol == "aos" and calibration_session is not None:
        calibration_count = sum(
            1
            for r in train_records
            if r.subject == held_out_subject and r.session == calibration_session
        )

    print("\n" + "=" * 72)
    print(f"Configuration:        {config.model_name}")
    print(f"Protocol:             {protocol.upper()}")
    print(f"Fold:                 {fold_index}/{num_folds}")
    print(f"Held-out subject:     {held_out_subject}")
    if protocol == "aos":
        print(f"Calibration session:  {calibration_session}")
        print(f"Calibration samples:  {calibration_count}")
    print(f"Train samples:        {len(train_records)}")
    print(f"Test samples:         {len(test_records)}")
    print("Input shape:          (4, 1250)")
    print("Classes:              6")
    print("Kernel size:          15")
    print("Blocks x width:       6 x 32")
    print("Dropout:              0.2")
    if config.model_name == "best_accuracy_channel_temporal":
        print("Attention:            Channel -> Temporal after each block")
        print(f"Channel reduction r:  {config.channel_reduction_ratio}")
        print("Temporal kernel:      7")
    elif config.model_name == "best_accuracy_se":
        print("Attention:            Squeeze-and-Excitation after each block")
        print(f"SE reduction r:       {config.se_reduction_ratio}")
        print("SE squeeze:            Global average pooling over time")
    elif config.model_name == "best_accuracy_eca":
        print("Attention:            ECA channel attention after each block")
        print("ECA gamma, b:         2, 1")
        print("ECA channel kernel:   3 for C=32")
        print("ECA squeeze:          Global average pooling over time")
    print(f"Trainable parameters: {num_parameters}")
    print(f"Device:               {device}")
    print(f"Batch size:           {config.batch_size}")
    print(f"Epochs:               {config.epochs}")
    print(f"Learning rate:        {config.learning_rate}")
    print(f"Seed:                 {config.seed}")
    print("=" * 72)

    history: list[dict] = []

    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
            }
        )

        print(
            f"Epoch {epoch:03d}/{config.epochs:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Train Acc: {train_accuracy * 100:.2f}%"
        )

    # No validation split is invented. The held-out participant is evaluated
    # only after the fixed training schedule; test performance never selects an
    # epoch or changes training.
    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    fold_dir = Path(output_dir) / f"fold_{fold_index:02d}_{held_out_subject}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    _save_history(history, fold_dir / "history.csv")
    _save_confusion_matrix(
        test_metrics["confusion_matrix"], fold_dir / "confusion_matrix.csv"
    )
    torch.save(model.state_dict(), fold_dir / "final_model.pt")

    final_train = history[-1]
    result = {
        "configuration_name": config.model_name,
        "protocol": protocol,
        "fold": fold_index,
        "held_out_subject": held_out_subject,
        "calibration_session": calibration_session if protocol == "aos" else None,
        "calibration_samples": calibration_count if protocol == "aos" else 0,
        "kernel_size": 15,
        "num_blocks": 6,
        "channel_width": 32,
        "widths": [32, 32, 32, 32, 32, 32],
        "dropout": 0.2,
        "attention": (
            {
                "type": "channel_temporal",
                "order": "channel_then_temporal",
                "placement": "after_relu_before_dropout",
                "channel_reduction_ratio": config.channel_reduction_ratio,
                "temporal_kernel_size": 7,
            }
            if config.model_name == "best_accuracy_channel_temporal"
            else (
                {
                    "type": "squeeze_excitation",
                    "placement": "after_relu_before_dropout",
                    "reduction_ratio": config.se_reduction_ratio,
                    "squeeze": "global_average_pool_over_time",
                }
                if config.model_name == "best_accuracy_se"
                else (
                    {
                        "type": "eca_channel",
                        "placement": "after_relu_before_dropout",
                        "gamma": 2,
                        "b": 1,
                        "adaptive_kernel_size": 3,
                        "squeeze": "global_average_pool_over_time",
                        "channel_interaction": "Conv1d across channel descriptor",
                    }
                    if config.model_name == "best_accuracy_eca"
                    else None
                )
            )
        ),
        "input_channels": 4,
        "num_classes": 6,
        "class_names": list(CLASS_NAMES),
        "num_trainable_parameters": num_parameters,
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "final_training_loss": final_train["train_loss"],
        "final_training_accuracy": final_train["train_accuracy"],
        "final_test_loss": test_metrics["loss"],
        "final_test_accuracy": test_metrics["accuracy"],
        "macro_precision_diagnostic": test_metrics["macro_precision_diagnostic"],
        "per_class_precision": test_metrics["per_class_precision"],
        "seed": config.seed,
        "epoch_count": config.epochs,
        "batch_size": config.batch_size,
        "optimizer": "AdamW",
        "learning_rate": config.learning_rate,
        "loss": "CrossEntropyLoss",
        "device": str(device),
        # There is deliberately no "best validation accuracy": no validation
        # protocol is specified for LOSO/AOS in the paper.
        "best_accuracy": None,
    }

    with (fold_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print("-" * 72)
    print(f"{protocol.upper()} Fold {fold_index} Results | {held_out_subject}")
    print(f"Final Train Loss:     {result['final_training_loss']:.6f}")
    print(f"Final Train Accuracy: {result['final_training_accuracy'] * 100:.2f}%")
    print(f"Final Test Loss:      {result['final_test_loss']:.6f}")
    print(f"Final Test Accuracy:  {result['final_test_accuracy'] * 100:.2f}%")
    print(
        "Macro Precision*:    "
        f"{result['macro_precision_diagnostic'] * 100:.2f}%"
    )
    print("* diagnostic; paper does not state the precision averaging convention")
    print("-" * 72)

    return result

