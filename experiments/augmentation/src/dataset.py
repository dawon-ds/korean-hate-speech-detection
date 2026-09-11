# placeholder
# src/dataset.py
from typing import Dict
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split

from .augment import augment_text

FINE_COLS = ["gender","LGBT","age","region","race","religion","socioeconomic","etc"]
COARSE_MAP = {"clean": 0, "offensive": 1, "hate": 2}

class HateDataset(Dataset):
    def __init__(self, df: pd.DataFrame, plm_name: str, max_len: int = 128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(plm_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        text = str(row["text"])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        coarse = torch.tensor(COARSE_MAP[row["hate_label"]], dtype=torch.long)
        fine = torch.tensor(row[FINE_COLS].values.astype("int64"), dtype=torch.float)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_coarse": coarse,
            "label_fine": fine,
        }


def load_and_split(
    csv_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    use_augment: bool = False,
    max_aug_per_sample: int = 2,
    apply_to_hate_only: bool = True,
):
    df = pd.read_csv(csv_path)

    # 증강: train 쪽에서만 적용할 것이므로, 여기서는 원본 df만 넘기고
    # 증강은 train split 이후에 적용해도 됨.
    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_ratio), random_state=seed, stratify=df["hate_label"]
    )
    val_size = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_size), random_state=seed, stratify=temp_df["hate_label"]
    )

    if use_augment:
        aug_rows = []
        for _, row in train_df.iterrows():
            if apply_to_hate_only and row["hate_label"] != "hate":
                continue
            new_texts = augment_text(str(row["text"]), max_new=max_aug_per_sample)
            for t in new_texts:
                new_row = row.copy()
                new_row["text"] = t
                aug_rows.append(new_row)
        if aug_rows:
            aug_df = pd.DataFrame(aug_rows)
            train_df = pd.concat([train_df, aug_df], ignore_index=True)

    return train_df, val_df, test_df
