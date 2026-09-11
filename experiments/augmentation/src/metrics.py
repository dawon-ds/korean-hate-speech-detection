# src/metrics.py
import numpy as np
from typing import Dict, Any

from sklearn.metrics import (
    f1_score,
    hamming_loss,
    jaccard_score,
    accuracy_score,
    label_ranking_average_precision_score,
)


def binarize_probs(y_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    확률 -> 0/1 멀티라벨 예측으로 변환.
    y_prob: (N, C)
    """
    return (y_prob >= threshold).astype(int)


def compute_multi_label_metrics(
    y_true, y_prob, threshold: float = 0.5
) -> Dict[str, Any]:
    """
    멀티라벨 분류용 통합 metric 계산 함수.

    Parameters
    ----------
    y_true : array-like, shape (N, C)
        정답 라벨 (0/1)
    y_prob : array-like, shape (N, C)
        모델 출력 확률 (sigmoid 통과한 값 또는 logit을 sigmoid 한 값)
    threshold : float
        이 값 이상이면 1로 간주.

    Returns
    -------
    metrics : dict
        {
            "micro_f1": ...,
            "macro_f1": ...,
            "hamming_loss": ...,
            "subset_accuracy": ...,
            "jaccard_micro": ...,
            "jaccard_macro": ...,
            "lrap": ...
        }
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    # 이진 예측
    y_pred = binarize_probs(y_prob, threshold=threshold)

    metrics = {}

    # 1) F1
    metrics["micro_f1"] = f1_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    metrics["macro_f1"] = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    # 2) Hamming loss (작을수록 좋음)
    metrics["hamming_loss"] = hamming_loss(y_true, y_pred)

    # 3) Subset accuracy (Exact match ratio)
    #    전체 라벨 세트가 완전히 맞을 때만 1점
    metrics["subset_accuracy"] = accuracy_score(y_true, y_pred)

    # 4) Jaccard (IoU)
    metrics["jaccard_micro"] = jaccard_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    metrics["jaccard_macro"] = jaccard_score(
        y_true, y_pred, average="macro", zero_division=0
    )

    # 5) Label Ranking Average Precision (확률 기반 랭킹 품질)
    #    일부 케이스에서 ValueError 날 수 있어 예외 처리
    try:
        metrics["lrap"] = label_ranking_average_precision_score(
            y_true, y_prob
        )
    except ValueError:
        metrics["lrap"] = float("nan")

    return metrics
