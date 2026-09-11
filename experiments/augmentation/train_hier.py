# placeholder
# train_hier.py

# train_hier.py

import os
import json
import datetime
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.utils import set_seed
from src.dataset import load_and_split, HateDataset
from src.models import HierHateModel
from src.trainer import build_optimizer_and_scheduler, train_one_epoch, evaluate


def main():
    cfg = load_config("config/base.yaml")
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 로그 디렉토리 및 파일 이름 설정
    os.makedirs("logs", exist_ok=True)
    run_name = f"hier_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = os.path.join("logs", run_name + ".jsonl")

    # --------------------
    # 데이터 로드 및 스플릿
    # --------------------
    csv_path = cfg.get("data", "csv_path")
    max_len = cfg.get("data", "max_len", default=128)
    train_ratio = cfg.get("data", "train_ratio", default=0.8)
    val_ratio = cfg.get("data", "val_ratio", default=0.1)

    use_aug = cfg.get("augment", "use_augment", default=False)
    max_aug = cfg.get("augment", "max_aug_per_sample", default=2)
    apply_to_hate = cfg.get("augment", "apply_to_hate_only", default=True)

    train_df, val_df, test_df = load_and_split(
        csv_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=cfg.seed,
        use_augment=use_aug,
        max_aug_per_sample=max_aug,
        apply_to_hate_only=apply_to_hate,
    )

    # --------------------
    # Dataset / DataLoader
    # --------------------
    plm_name = cfg.get("model", "plm_name")
    batch_size = cfg.get("train", "batch_size", default=32)
    num_epochs = cfg.get("train", "num_epochs", default=3)
    lr = cfg.get("train", "lr", default=2e-5)
    wd = cfg.get("train", "weight_decay", default=0.01)
    warmup_ratio = cfg.get("train", "warmup_ratio", default=0.1)
    lambda_fine = cfg.get("model", "lambda_fine", default=1.0)
    lambda_hier = cfg.get("model", "lambda_hier", default=1.0)

    train_ds = HateDataset(train_df, plm_name, max_len=max_len)
    val_ds   = HateDataset(val_df,   plm_name, max_len=max_len)
    test_ds  = HateDataset(test_df,  plm_name, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # --------------------
    # 모델 / 옵티마이저
    # --------------------
    model = HierHateModel(
        plm_name=plm_name,
        num_coarse=cfg.get("model", "num_coarse", default=3),
        num_fine=cfg.get("model", "num_fine", default=8),
        lambda_fine=lambda_fine,
        lambda_hier=lambda_hier,
    ).to(device)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model, len(train_loader), num_epochs, lr, wd, warmup_ratio
    )

    best_val_macro = 0.0
    best_state = None

    # --------------------
    # 학습 루프
    # --------------------
    for epoch in range(1, num_epochs + 1):
        print(f"=== Epoch {epoch}/{num_epochs} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train loss: {train_loss:.4f}")

        # evaluate는 (coarse_metrics, fine_metrics) 튜플 반환
        coarse_metrics, fine_metrics = evaluate(model, val_loader, device, threshold=0.5)

        print("Val coarse metrics:")
        for k, v in coarse_metrics.items():
            print(f"  {k}: {v:.4f}")

        print("Val fine metrics:")
        for k, v in fine_metrics.items():
            print(f"  {k}: {v:.4f}")

        # 에폭별 로그 기록
        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),

            # coarse
            "val_coarse_micro_f1": float(coarse_metrics["micro_f1"]),
            "val_coarse_macro_f1": float(coarse_metrics["macro_f1"]),
            "val_coarse_hamming": float(coarse_metrics["hamming_loss"]),
            "val_coarse_jaccard_micro": float(coarse_metrics["jaccard_micro"]),
            "val_coarse_jaccard_macro": float(coarse_metrics["jaccard_macro"]),
            "val_coarse_subset_acc": float(coarse_metrics["subset_accuracy"]),
            "val_coarse_lrap": float(coarse_metrics["lrap"]),

            # fine
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

        # early-stopping 기준: fine macro F1
        macro_fine = fine_metrics["macro_f1"]
        if macro_fine > best_val_macro:
            best_val_macro = macro_fine
            best_state = model.state_dict()

    # --------------------
    # best state 로드
    # --------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    # best_state까지 로드한 뒤
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(best_state, "checkpoints/hier_best.pt")

    # --------------------
    # 최종 테스트
    # --------------------
    print("=== Final Test Evaluation ===")
    coarse_test, fine_test = evaluate(model, test_loader, device, threshold=0.5)

    print("Test coarse metrics:")
    for k, v in coarse_test.items():
        print(f"  {k}: {v:.4f}")

    print("Test fine metrics:")
    for k, v in fine_test.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
