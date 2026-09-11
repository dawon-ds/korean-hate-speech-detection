import copy
import datetime
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.dataset import HateDataset, load_and_split
from src.models import HierHateModel
from src.trainer import build_optimizer_and_scheduler, evaluate, train_one_epoch
from src.utils import set_seed

EXPERIMENT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = EXPERIMENT_ROOT / "config" / "base.yaml"


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

    run_name = f"hier_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

    plm_name = cfg.get("model", "plm_name")
    batch_size = cfg.get("train", "batch_size", default=32)
    num_epochs = cfg.get("train", "num_epochs", default=3)
    lr = cfg.get("train", "lr", default=2e-5)
    wd = cfg.get("train", "weight_decay", default=0.01)
    warmup_ratio = cfg.get("train", "warmup_ratio", default=0.1)
    lambda_fine = cfg.get("model", "lambda_fine", default=1.0)
    lambda_hier = cfg.get("model", "lambda_hier", default=1.0)
    patience = cfg.get("train", "patience", default=3)
    num_coarse = cfg.get("model", "num_coarse", default=3)
    num_fine = cfg.get("model", "num_fine", default=8)

    train_ds = HateDataset(train_df, plm_name, max_len=max_len)
    val_ds = HateDataset(val_df, plm_name, max_len=max_len)
    test_ds = HateDataset(test_df, plm_name, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = HierHateModel(
        plm_name=plm_name,
        num_coarse=num_coarse,
        num_fine=num_fine,
        lambda_fine=lambda_fine,
        lambda_hier=lambda_hier,
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
            model,
            val_loader,
            device,
            threshold=0.5,
            num_coarse=num_coarse,
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
            "val_coarse_accuracy": float(coarse_metrics["accuracy"]),
            "val_coarse_micro_f1": float(coarse_metrics["micro_f1"]),
            "val_coarse_macro_f1": float(coarse_metrics["macro_f1"]),
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
    torch.save(best_state, checkpoints_dir / "hier_best.pt")

    print("=== Final Test Evaluation ===")
    coarse_test, fine_test = evaluate(
        model,
        test_loader,
        device,
        threshold=0.5,
        num_coarse=num_coarse,
    )

    print("Test coarse metrics:")
    for key, value in coarse_test.items():
        print(f"  {key}: {value:.4f}")

    print("Test fine metrics:")
    for key, value in fine_test.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
