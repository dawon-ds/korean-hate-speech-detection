import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .metrics import compute_multi_label_metrics


def build_optimizer_and_scheduler(model, train_loader_len, num_epochs, lr, weight_decay, warmup_ratio):
    optimizer = AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    num_training_steps = train_loader_len * num_epochs
    num_warmup_steps = int(num_training_steps * float(warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
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
def evaluate(model, dataloader, device, threshold=0.5, num_coarse=3):
    model.eval()
    coarse_labels, coarse_probs = [], []
    fine_labels, fine_probs = [], []

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs_coarse = torch.softmax(outputs["logits_coarse"], dim=-1).cpu().numpy()
        probs_fine = torch.sigmoid(outputs["logits_fine"]).cpu().numpy()

        labels_coarse = batch["label_coarse"].cpu().numpy()
        labels_coarse_oh = np.eye(num_coarse, dtype=np.float32)[labels_coarse]

        coarse_labels.append(labels_coarse_oh)
        coarse_probs.append(probs_coarse)
        fine_labels.append(batch["label_fine"].cpu().numpy())
        fine_probs.append(probs_fine)

    coarse_metrics = compute_multi_label_metrics(
        np.concatenate(coarse_labels), np.concatenate(coarse_probs), threshold
    )
    fine_metrics = compute_multi_label_metrics(
        np.concatenate(fine_labels), np.concatenate(fine_probs), threshold
    )
    return coarse_metrics, fine_metrics
