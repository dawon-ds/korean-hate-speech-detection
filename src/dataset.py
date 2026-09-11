from typing import Dict

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoTokenizer

FINE_COLS = ["gender", "LGBT", "age", "region", "race", "religion", "socioeconomic", "etc"]
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
            return_tensors="pt",
        )

        coarse = torch.tensor(COARSE_MAP[row["hate_label"]], dtype=torch.long)
        fine = torch.tensor(row[FINE_COLS].values.astype("int64"), dtype=torch.float)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label_coarse": coarse,
            "label_fine": fine,
        }


def split_dataframe(df: pd.DataFrame, train_ratio=0.8, val_ratio=0.1, seed=42):
    train_df, temp_df = train_test_split(
        df,
        test_size=1 - train_ratio,
        random_state=seed,
        stratify=df["hate_label"],
    )
    val_size = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - val_size,
        random_state=seed,
        stratify=temp_df["hate_label"],
    )
    return train_df, val_df, test_df
