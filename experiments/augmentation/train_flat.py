import copy
import datetime
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.dataset import HateDataset, load_and_split
from src.models import FlatHateModel
from src.trainer import build_optimizer_and_scheduler, evaluate, train_one_epoch
from src.utils import set_seed

EXPERIMENT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = EXPERIMENT_ROOT / "config" / "base.yaml"

COARSE_CLASSES = ["clean", "offensive", "hate"]
FINE_COLS = [
    "gender",
    "LGBT",
    "age",
    "region",
    "race",
    "religion",
    "socioeconomic",
    "etc",
]


def compute_class_weight_coarse(df) -> torch.Tensor:
    n_total = len(df)
    n_classes = len(COARSE_CLASSES)
    counts = df["hate_label"].value_counts()

    weights = []
    for cls in COARSE_CLASSES:
        n_cls = float(counts.get(cls, 1.0))
        weights.append(n_total / (n_classes * n_cls))
    return torch.tensor(weights, dtype=torch.float32)


def compute_pos_weight_fine(df, label_cols) -> torch.Tensor:
    n_total = len(df)
    pos = df[label_cols].sum(axis=0).values.astype(np.float32)
    neg = n_total - pos
    return torch.tensor(neg / (pos + 1e-6), dtype=torch.float32)


def resolve_data_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else EXPERIMENT_ROOT / path


def main():
    cfg = load_config(str(CONFIG_PATH))
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    logs_dir = EXPERIMENT_ROOT / "logs"
    checkpoints_dir = EXPERIMENT_ROOT / "checkpoints"
    logs_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)

    run_name = f"flat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = logs_dir / f"{run_name}.jsonl"

    csv_path = resolve_data_path(cfg.get("data", "csv_path"))
    max_len = cfg.get("data", "max_len", default=128)
    train_ratio = cfg.get("data", "train_ratio", default=0.8)
    val_ratio = cfg.get("data", "val_ratio", default=0.1)

    use_aug = cfg.get("augment", "use_augment", default=False)
    max_aug = cfg.get("augment", "max_aug_per_sample", default=2)
    apply_to_hate = cfg.get("augment", "apply_to_hate_only", default=True)

    train_df, val_df, test_df = load_and_split(
        str(csv_path),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=cfg.seed,
        use_augment=use_aug,
        max_aug_per_sample=max_aug,
        apply_to_hate_only=apply_to_hate,
    )

    class_weight_coarse = compute_class_weight_coarse(train_df)
    pos_weight_fine = compute_pos_weight_fine(train_df, FINE_COLS)

    plm_name = cfg.get("model", "plm_name")
    batch_size = cfg.get("train", "batch_size", default=32)
    num_epochs = cfg.get("train", "num_epochs", default=5)
    lr = cfg.get("train", "lr", default=2e-5)
    wd = cfg.get("train", "weight_decay", default=0.01)
    warmup_ratio = cfg.get("train", "warmup_ratio", default=0.1)
    lambda_fine = cfg.get("model", "lambda_fine", default=1.0)
    patience = cfg.get("train", "patience", default=3)

    train_ds = HateDataset(train_df, plm_name, max_len=max_len)
    val_ds = HateDataset(val_df, plm_name, max_len=max_len)
    test_ds = HateDataset(test_df, plm_name, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = FlatHateModel(
        plm_name=plm_name,
        num_coarse=len(COARSE_CLASSES),
        num_fine=len(FINE_COLS),
        lambda_fine=lambda_fine,
        class_weight_coarse=class_weight_coarse.to(device),
        pos_weight_fine=pos_weight_fine.to(device),
    ).to(device)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model, len(train_loader), num_epochs, lr, wd, warmup_ratio
    )

    best_val_macro = float("-inf")
    best_state = None
    patience_count = 0

    for epoch in range(1, num_epochs + 1):
        print(f"=== Epoch {epoch}/{num_epochs} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train loss: {train_loss:.4f}")

        coarse_metrics, fine_metrics = evaluate(
            model, val_loader, device, threshold=0.5, num_coarse=len(COARSE_CLASSES)
        )

        print("Val coarse metrics:")
        for key, value in coarse_metrics.items():
            print(f"  {key}: {value:.4f}")

        print("Val fine metrics:")
        for key, value in fine_metrics.items():
            print(f"  {key}: {value:.4f}")

        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_coarse_micro_f1": float(coarse_metrics["micro_f1"]),
            "val_coarse_macro_f1": float(coarse_metrics["macro_f1"]),
            "val_coarse_hamming": float(coarse_metrics["hamming_loss"]),
            "val_coarse_jaccard_micro": float(coarse_metrics["jaccard_micro"]),
            "val_coarse_jaccard_macro": float(coarse_metrics["jaccard_macro"]),
            "val_coarse_subset_acc": float(coarse_metrics["subset_accuracy"]),
            "val_coarse_lrap": float(coarse_metrics["lrap"]),
            "val_fine_micro_f1": float(fine_metrics["micro_f1"]),
            "val_fine_macro_f1": float(fine_metrics["macro_f1"]),
            "val_fine_hamming": float(fine_metrics["hamming_loss"]),
            "val_fine_jaccard_micro": float(fine_metrics["jaccard_micro"]),
            "val_fine_jaccard_macro": float(fine_metrics["jaccard_macro"]),
            "val_fine_subset_acc": float(fine_metrics["subset_accuracy"]),
            "val_fine_lrap": float(fine_metrics["lrap"]),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        macro_fine = fine_metrics["macro_f1"]
        if macro_fine > best_val_macro:
            best_val_macro = macro_fine
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            print(
                f"No improvement in fine macro F1 "
                f"(patience {patience_count}/{patience})"
            )
            if patience_count >= patience:
                print("Early stopping triggered.")
                break

    if best_state is None:
        raise RuntimeError("Training finished without a valid best model state.")

    model.load_state_dict(best_state)
    torch.save(best_state, checkpoints_dir / "flat_best.pt")

    print("=== Final Test Evaluation ===")
    coarse_test, fine_test = evaluate(
        model, test_loader, device, threshold=0.5, num_coarse=len(COARSE_CLASSES)
    )

    print("Test coarse metrics:")
    for key, value in coarse_test.items():
        print(f"  {key}: {value:.4f}")

    print("Test fine metrics:")
    for key, value in fine_test.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
