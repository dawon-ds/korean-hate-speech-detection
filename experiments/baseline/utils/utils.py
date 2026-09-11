# src/utils.py
import random
from dataclasses import dataclass
from typing import Dict
from typing import Optional

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, label_ranking_average_precision_score


# -----------------------------
# Config dataclasses
# -----------------------------
@dataclass
class DataConfig:
    # text-only에서는 안 쓸 수도 있으니까 Optional + 기본값 None
    input_csv: Optional[str] = None

    text_col: str = "문장"
    label_col: str = "labels"
    max_length: int = 128

    # text-only에서도 쓸 수 있으니까 기본값 0.2로
    test_size: float = 0.2
    random_state: int = 42

    # 감성 feature 버전에서만 사용하는 prefix – text-only에서는 그냥 기본값만 있어도 됨
    emotion_prefix: str = "emo_"

    # 우리가 새로 추가한 필드
    train_mode: str = "unsmile_only"   # or "hatescore_plus_unsmile"

@dataclass
class EmotionConfig:
    model_name: str
    batch_size: int



@dataclass
class ModelConfig:
    model_name: str
    num_labels: int = 7
    emotion_hidden_dim: Optional[int] = None


@dataclass
class TrainConfig:
    output_dir: str
    logging_dir: str
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    logging_steps: int
    eval_strategy: str
    save_strategy: str
    metric_for_best_model: str
    seed: int


@dataclass
class FullConfig:
    data: DataConfig
    emotion: EmotionConfig
    model: ModelConfig
    train: TrainConfig


# -----------------------------
# Config loader
# -----------------------------

def load_config(path: str) -> FullConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # ----- data -----
    data_cfg = DataConfig(**cfg["data"])

    # ----- model -----
    model_cfg = ModelConfig(**cfg["model"])

    # ----- train -----
    train_raw = cfg.get("train", {})
    # 기본값 채워주기 (없어도 돌아가게)
    train_raw.setdefault("learning_rate", 2e-5)
    train_raw.setdefault("weight_decay", 0.01)
    train_raw.setdefault("warmup_ratio", 0.1)
    train_raw.setdefault("num_train_epochs", 3)

    train_raw["learning_rate"] = float(train_raw["learning_rate"])
    train_raw["weight_decay"] = float(train_raw["weight_decay"])
    train_raw["warmup_ratio"] = float(train_raw["warmup_ratio"])
    train_raw["num_train_epochs"] = int(train_raw["num_train_epochs"])

    train_cfg = TrainConfig(**train_raw)

    # ----- emotion (없어도 되게) -----
    # text_only_7cls.yaml 에서는 emotion 블록이 없으니까 기본값으로 채워줌
    emotion_raw = cfg.get(
        "emotion",
        {
            "model_name": "monologg/koelectra-base-v3-goemotions",
            "batch_size": 32,
        },
    )
    emotion_cfg = EmotionConfig(**emotion_raw)

    return FullConfig(
        data=data_cfg,
        model=model_cfg,
        train=train_cfg,
        emotion=emotion_cfg,
    )


# -----------------------------
# Seed 설정
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Metrics (Accuracy, Macro F1, LRAP)
# -----------------------------
def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred  # logits: [N, num_labels], labels: [N]
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    # LRAP 계산을 위해 one-hot y_true와 score를 사용
    num_labels = logits.shape[1]
    y_true = np.zeros_like(logits)
    y_true[np.arange(len(labels)), labels] = 1

    # logits 그대로 사용해도 ranking은 동일 (softmax 불필요)
    lrap = label_ranking_average_precision_score(y_true, logits)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "lrap": lrap,
    }
