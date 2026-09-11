# src/trainer.py

import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from .metrics import compute_multi_label_metrics


def build_optimizer_and_scheduler(model, train_loader_len, num_epochs, lr, weight_decay, warmup_ratio):
    lr = float(lr)
    weight_decay = float(weight_decay)
    warmup_ratio = float(warmup_ratio)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_training_steps = train_loader_len * num_epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    return optimizer, scheduler


def train_one_epoch(model, loader, optimizer, scheduler, device) -> float:
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        # 데이터 GPU/CPU로 이동
        batch = {k: v.to(device) for k, v in batch.items()}

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            label_coarse=batch["label_coarse"],
            label_fine=batch["label_fine"],
        )
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * batch["input_ids"].size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model,
    dataloader: DataLoader,
    device: str,
    threshold: float = 0.5,
    num_coarse: int = 3,
):
    model.eval()

    all_coarse_labels = []
    all_coarse_probs = []
    all_fine_labels = []
    all_fine_probs = []

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # hate_label: 0/1/2 (COARSE_MAP에 따라 dataset에서 이미 int로 변환)
        labels_coarse = batch["label_coarse"].cpu().numpy()  # shape (B,)
        labels_fine = batch["label_fine"].cpu().numpy()      # shape (B, F)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits_coarse = outputs["logits_coarse"]  # (B, 3)
        logits_fine = outputs["logits_fine"]      # (B, F)

        # ---- coarse: 3-class → softmax + one-hot로 multi-label metric 재사용 ----
        probs_coarse = torch.softmax(logits_coarse, dim=-1).cpu().numpy()  # (B, 3)
        labels_coarse_oh = np.eye(num_coarse, dtype=np.float32)[labels_coarse]  # (B, 3)

        # ---- fine: multi-label → sigmoid ----
        probs_fine = torch.sigmoid(logits_fine).cpu().numpy()  # (B, F)

        all_coarse_labels.append(labels_coarse_oh)
        all_coarse_probs.append(probs_coarse)
        all_fine_labels.append(labels_fine)
        all_fine_probs.append(probs_fine)

    all_coarse_labels = np.concatenate(all_coarse_labels, axis=0)
    all_coarse_probs = np.concatenate(all_coarse_probs, axis=0)
    all_fine_labels = np.concatenate(all_fine_labels, axis=0)
    all_fine_probs = np.concatenate(all_fine_probs, axis=0)

    coarse_metrics = compute_multi_label_metrics(
        all_coarse_labels, all_coarse_probs, threshold=threshold
    )
    fine_metrics = compute_multi_label_metrics(
        all_fine_labels, all_fine_probs, threshold=threshold
    )

    print("Coarse metrics:")
    for k, v in coarse_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("Fine metrics:")
    for k, v in fine_metrics.items():
        print(f"  {k}: {v:.4f}")

    return coarse_metrics, fine_metrics


def train(
    model,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: str,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    exp_name: str,
    ckpt_dir: str = "checkpoints",
    monitor: str = "fine_macro_f1",  # 저장에 쓸 기준 메트릭
):
    """
    전체 학습 루프 + best 모델 저장까지 한 번에 돌리는 함수.
    monitor:
        - "fine_macro_f1" (기본)
        - "fine_micro_f1"
        - "coarse_macro_f1"
        - "coarse_micro_f1"
    """

    os.makedirs(ckpt_dir, exist_ok=True)

    model.to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        train_loader_len=len(train_loader),
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
    )

    best_score = -1.0
    best_metrics = None
    best_ckpt_path = os.path.join(ckpt_dir, f"{exp_name}_best.pt")

    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train loss: {train_loss:.4f}")

        coarse_metrics, fine_metrics = evaluate(model, valid_loader, device)

        # 모니터링 기준 선택
        if monitor == "fine_macro_f1":
            current_score = fine_metrics["macro_f1"]
        elif monitor == "fine_micro_f1":
            current_score = fine_metrics["micro_f1"]
        elif monitor == "coarse_macro_f1":
            current_score = coarse_metrics["macro_f1"]
        elif monitor == "coarse_micro_f1":
            current_score = coarse_metrics["micro_f1"]
        else:
            raise ValueError(f"Unknown monitor metric: {monitor}")

        print(f"[Monitor] {monitor} = {current_score:.4f}")

        # best 모델 갱신 시 저장
        if current_score > best_score:
            best_score = current_score
            best_metrics = {
                "coarse": coarse_metrics,
                "fine": fine_metrics,
            }
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"[SAVE] New best model saved to {best_ckpt_path}")

    print(f"\nTraining finished. Best {monitor}: {best_score:.4f}")
    return best_ckpt_path, best_metrics
