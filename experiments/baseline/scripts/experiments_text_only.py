# scripts/experiments_text_only.py

import sys
import re
from pathlib import Path

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    BertTokenizerFast,
)

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.utils import load_config, set_seed, compute_metrics
from src.dataset_text_only import TextOnlyDataset
from tqdm import tqdm

# ---------------------------
# 공통 라벨 매핑
# ---------------------------
ID2LABEL = {
    0: "clean",
    1: "악플/욕설",
    2: "여성/가족",
    3: "남성",
    4: "성소수자",
    5: "인종/국적",
    6: "연령",
    7: "지역",
    8: "종교",
    9: "기타혐오",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


# ---------------------------
# 1. labels eunsuring
# ---------------------------
def ensure_labels_column(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if label_col in df.columns:
        print("[INFO] labels 컬럼 이미 존재. 그대로 사용합니다.")
        return df

    print("[INFO] labels 컬럼이 없어 단일 10클래스 라벨로 변환합니다.")

    def row_to_label(row):
        if "여성/가족" in row and row["여성/가족"] == 1:
            return 2
        if "남성" in row and row["남성"] == 1:
            return 3
        if "성소수자" in row and row["성소수자"] == 1:
            return 4
        if "인종/국적" in row and row["인종/국적"] == 1:
            return 5
        if "연령" in row and row["연령"] == 1:
            return 6
        if "지역" in row and row["지역"] == 1:
            return 7
        if "종교" in row and row["종교"] == 1:
            return 8
        if "기타혐오" in row and row["기타혐오"] == 1:
            return 9
        if "악플/욕설" in row and row["악플/욕설"] == 1:
            return 1
        if "clean" in row and row["clean"] == 1:
            return 0
        # 애매한 경우도 일단 clean(0)으로 처리
        return 0

    df[label_col] = df.apply(row_to_label, axis=1)
    print("[INFO] labels 컬럼 생성 완료.")
    print("[INFO] 라벨 분포:")
    print(df[label_col].value_counts())
    return df


def prepare_data(cfg):
    text_col = cfg.data.text_col
    label_col = cfg.data.label_col
    train_mode = getattr(cfg.data, "train_mode", "unsmile_only")

    data_dir = PROJECT_ROOT / "data"

    # 항상 test는 UnSmile valid
    test_path = data_dir / "unsmile_valid.csv"
    test_df = pd.read_csv(test_path, encoding="utf-8-sig")

    if train_mode == "unsmile_only":
        train_path = data_dir / "unsmile_train.csv"
        base_train_df = pd.read_csv(train_path, encoding="utf-8-sig")
        print(f"[INFO] Train mode: unsmile_only, size={len(base_train_df)}")

    elif train_mode == "hatescore_plus_unsmile":
        train_path = data_dir / "train_hatescore_unsmile.csv"
        base_train_df = pd.read_csv(train_path, encoding="utf-8-sig")
        print(f"[INFO] Train mode: hatescore_plus_unsmile, size={len(base_train_df)}")
    else:
        raise ValueError(f"Unknown train_mode: {train_mode}")

    base_train_df = ensure_labels_column(base_train_df, label_col)
    test_df = ensure_labels_column(test_df, label_col)

    # 9:1 train/valid split
    train_df, valid_df = train_test_split(
        base_train_df,
        test_size=0.1,
        stratify=base_train_df[label_col],
        random_state=cfg.data.random_state,
    )

    print(f"[INFO] Split → Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}")

    return train_df, valid_df, test_df


# ---------------------------
# 모델별/하이퍼파라미터별 실험 설정
# ---------------------------

MODEL_EXPERIMENTS = {
    "bert": {
        "model_name": "klue/bert-base",
        "lrs": [2e-5, 3e-5, 5e-5],
        "batch_sizes": [16],
        "epochs": [3, 4],
    },
    "roberta": {
        "model_name": "klue/roberta-base",
        "lrs": [1e-5],  # [1e-5, 2e-5, 3e-5]
        "batch_sizes": [16],
        "epochs": [3],  # , 4
    },
    "albert": {
        "model_name": "kykim/albert-kor-base",
        "lrs": [1e-5, 2e-5, 3e-5],
        "batch_sizes": [16],
        "epochs": [4, 6],
    },
    "electra": {
        "model_name": "monologg/koelectra-base-v3-discriminator",
        "lrs": [1e-5, 2e-5, 3e-5],
        "batch_sizes": [16],
        "epochs": [2, 3],
    },  
}


# ---------------------------
# 4. 메인 experiments 루프
# ---------------------------
def main(cfg_path: str):
    cfg = load_config(cfg_path)
    set_seed(cfg.train.seed)

    train_df, valid_df, test_df = prepare_data(cfg)
    text_col = cfg.data.text_col
    label_col = cfg.data.label_col
    model_keys = list(MODEL_EXPERIMENTS.keys())

    ALBERT_BERT_TOKENIZER_MODELS = {
        "kykim/albert-kor-base",
    }

    # 각 모델별로 반복
    for model_key in model_keys:
        mconf = MODEL_EXPERIMENTS[model_key]
        base_model_name = mconf["model_name"]

        print(f"\n==============================")
        print(f"[EXPERIMENT] Base model: {model_key} ({base_model_name})")
        print(f"==============================")

        for lr in tqdm(mconf["lrs"], desc=f"{model_key} | LR", ncols=100):
            for bs in tqdm(mconf["batch_sizes"], desc=f"{model_key} | BS", ncols=100):
                for n_epochs in tqdm(mconf["epochs"], desc=f"{model_key} | Epochs", ncols=100):

                    run_name = f"textonly_{model_key}_lr{lr}_bs{bs}_ep{n_epochs}"
                    print(f"\n[RUN] {run_name}")

                    output_dir = PROJECT_ROOT / "model" / run_name
                    logging_dir = PROJECT_ROOT / "log" / run_name
                    os.makedirs(output_dir, exist_ok=True)
                    os.makedirs(logging_dir, exist_ok=True)

                    # Tokenizer
                    if base_model_name in ALBERT_BERT_TOKENIZER_MODELS:
                        tokenizer = BertTokenizerFast.from_pretrained(base_model_name)
                    else:
                        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

                    # Dataset
                    train_dataset = TextOnlyDataset(
                        train_df,
                        tokenizer=tokenizer,
                        text_col=text_col,
                        label_col=label_col,
                        max_len=cfg.data.max_length,
                    )

                    valid_dataset = TextOnlyDataset(
                        valid_df,
                        tokenizer=tokenizer,
                        text_col=text_col,
                        label_col=label_col,
                        max_len=cfg.data.max_length,
                    )

                    test_dataset = TextOnlyDataset(
                        test_df,
                        tokenizer=tokenizer,
                        text_col=text_col,
                        label_col=label_col,
                        max_len=cfg.data.max_length,
                    )

                    # -------------------
                    # Model
                    # -------------------
                    model = AutoModelForSequenceClassification.from_pretrained(
                        base_model_name,
                        num_labels=cfg.model.num_labels,
                        id2label=ID2LABEL,
                        label2id=LABEL2ID,
                    )

                    # TrainingArguments
                    training_args = TrainingArguments(
                        output_dir=str(output_dir),
                        num_train_epochs=n_epochs,
                        per_device_train_batch_size=bs,
                        per_device_eval_batch_size=bs * 2,
                        learning_rate=lr,
                        weight_decay=cfg.train.weight_decay,
                        warmup_ratio=cfg.train.warmup_ratio,
                        evaluation_strategy=cfg.train.eval_strategy,
                        save_strategy=cfg.train.save_strategy,
                        metric_for_best_model=cfg.train.metric_for_best_model,
                        load_best_model_at_end=True,
                        logging_dir=str(logging_dir),
                        logging_strategy="steps",
                        logging_steps=cfg.train.logging_steps,
                        save_total_limit=2,
                        report_to=["tensorboard"],
                        run_name=run_name,
                        save_safetensors=False,
                        log_level="error",
                        log_level_replica="error",
                        disable_tqdm=True,  # Trainer 내부 tqdm off
                    )

                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=valid_dataset,
                        compute_metrics=compute_metrics,
                    )

                    # Train
                    trainer.train()

                    # 모델 저장
                    trainer.save_model(str(output_dir))
                    tokenizer.save_pretrained(str(output_dir))

                    # 로그 저장
                    log_history = trainer.state.log_history
                    logs_df = pd.DataFrame(log_history)
                    logs_csv_path = PROJECT_ROOT / "log" / f"{run_name}_history.csv"
                    logs_df.to_csv(logs_csv_path, index=False, encoding="utf-8-sig")
                    print(f"[INFO] Saved log history → {logs_csv_path}")

                    # Test 평가
                    test_result = trainer.evaluate(test_dataset)
                    print("[INFO] Test result:", test_result)

                    test_metrics_path = PROJECT_ROOT / "log" / f"{run_name}_test_metrics.csv"
                    pd.DataFrame([test_result]).to_csv(
                        test_metrics_path, index=False, encoding="utf-8-sig"
                    )
                    print(f"[INFO] Saved test metrics → {test_metrics_path}")


if __name__ == "__main__":
    default_cfg = PROJECT_ROOT / "config" / "text_only_7cls.yaml"
    main(str(default_cfg))
