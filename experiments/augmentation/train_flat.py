# train_flat.py

import os
import json
import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.utils import set_seed
from src.dataset import load_and_split, HateDataset
from src.models import FlatHateModel
from src.trainer import build_optimizer_and_scheduler, evaluate

# -----------------------------
# 라벨 컬럼 정의
# -----------------------------

# coarse는 문자열 3클래스(hate_label) 하나
COARSE_CLASSES = ["clean", "offensive", "hate"]  # COARSE_MAP 순서와 맞춰두면 편함

# fine 라벨들 (0/1 멀티라벨)
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
    """
    3-class single-label용 class weight 계산 (CrossEntropyLoss용)
    w_c = N / (K * n_c)  (class가 적을수록 weight가 커짐)

    return: tensor shape (3,)
    """
    N = len(df)
    K = len(COARSE_CLASSES)

    # 각 클래스별 개수 세기
    vc = df["hate_label"].value_counts()

    weights = []
    for cls in COARSE_CLASSES:
        n_c = float(vc.get(cls, 1.0))  # 해당 클래스가 아예 없을 경우 1로 방어
        w_c = N / (K * n_c)
        weights.append(w_c)

    return torch.tensor(weights, dtype=torch.float32)


def compute_pos_weight_fine(df, label_cols) -> torch.Tensor:
    """
    multi-label fine(8개)에 대한 pos_weight = neg / pos
    BCEWithLogitsLoss(pos_weight=...) 용
    """
    N = len(df)
    pos = df[label_cols].sum(axis=0).values.astype(np.float32)  # (F,)
    neg = N - pos
    w = neg / (pos + 1e-6)
    return torch.tensor(w, dtype=torch.float32)


def main():
    # -----------------------------
    # 1. 설정 / seed / device
    # -----------------------------
    cfg = load_config("config/base.yaml")
    set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 로그 디렉토리 및 파일 이름 설정
    os.makedirs("logs", exist_ok=True)
    run_name = f"flat_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = os.path.join("logs", run_name + ".jsonl")

    # -----------------------------
    # 2. 데이터 로드 & split (+ 증강)
    # -----------------------------
    csv_path    = cfg.get("data", "csv_path")
    max_len     = cfg.get("data", "max_len", default=128)
    train_ratio = cfg.get("data", "train_ratio", default=0.8)
    val_ratio   = cfg.get("data", "val_ratio", default=0.1)

    use_aug       = cfg.get("augment", "use_augment", default=False)
    max_aug       = cfg.get("augment", "max_aug_per_sample", default=2)
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

    # -----------------------------
    # 3. coarse class_weight / fine pos_weight 계산
    # -----------------------------
    class_weight_coarse = compute_class_weight_coarse(train_df)  # (3,)
    pos_weight_fine     = compute_pos_weight_fine(train_df, FINE_COLS)  # (8,)

    # -----------------------------
    # 4. Dataset / DataLoader
    # -----------------------------
    plm_name    = cfg.get("model", "plm_name")
    batch_size  = cfg.get("train", "batch_size", default=32)
    num_epochs  = cfg.get("train", "num_epochs", default=5)
    lr          = cfg.get("train", "lr", default=2e-5)
    wd          = cfg.get("train", "weight_decay", default=0.01)
    warmup_ratio = cfg.get("train", "warmup_ratio", default=0.1)
    lambda_fine  = cfg.get("model", "lambda_fine", default=1.0)
    patience     = cfg.get("train", "patience", default=3)  # early stopping

    train_ds = HateDataset(train_df, plm_name, max_len=max_len)
    val_ds   = HateDataset(val_df,   plm_name, max_len=max_len)
    test_ds  = HateDataset(test_df,  plm_name, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # -----------------------------
    # 5. 모델 / 옵티마이저 / 스케줄러
    # -----------------------------
    from src.trainer import train_one_epoch  # 순환 import 피하려면 위에서 같이 import 해도 됨

    model = FlatHateModel(
        plm_name=plm_name,
        num_coarse=len(COARSE_CLASSES),      # = 3
        num_fine=len(FINE_COLS),             # = 8
        lambda_fine=lambda_fine,
        class_weight_coarse=class_weight_coarse.to(device),
        pos_weight_fine=pos_weight_fine.to(device),
    ).to(device)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model, len(train_loader), num_epochs, lr, wd, warmup_ratio
    )

    # -----------------------------
    # 6. 학습 루프 + early stopping
    # -----------------------------
    best_val_macro = 0.0
    best_state = None
    patience_cnt = 0

    for epoch in range(1, num_epochs + 1):
        print(f"=== Epoch {epoch}/{num_epochs} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train loss: {train_loss:.4f}")

        # evaluate는 (coarse_metrics, fine_metrics) 튜플 반환(아래 trainer.py 수정 참고)
        coarse_metrics, fine_metrics = evaluate(
            model, val_loader, device, threshold=0.5, num_coarse=len(COARSE_CLASSES)
        )

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

        # early stopping 기준: fine macro F1
        macro_fine = fine_metrics["macro_f1"]
        if macro_fine > best_val_macro:
            best_val_macro = macro_fine
            best_state = model.state_dict()
            patience_cnt = 0
        else:
            patience_cnt += 1
            print(f"No improvement in fine macro F1 (patience {patience_cnt}/{patience})")
            if patience_cnt >= patience:
                print("Early stopping triggered.")
                break

    # -----------------------------
    # 7. best state 로드 후 최종 테스트
    # -----------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
    # best_state까지 로드한 뒤

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(best_state, "checkpoints/flat_best.pt")

    print("=== Final Test Evaluation ===")
    coarse_test, fine_test = evaluate(
        model, test_loader, device, threshold=0.5, num_coarse=len(COARSE_CLASSES)
    )

    print("Test coarse metrics:")
    for k, v in coarse_test.items():
        print(f"  {k}: {v:.4f}")

    print("Test fine metrics:")
    for k, v in fine_test.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
