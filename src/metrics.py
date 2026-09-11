import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    label_ranking_average_precision_score,
)


def binarize_probs(y_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (y_prob >= threshold).astype(int)


def compute_coarse_metrics(y_true, y_prob):
    """Metrics for the single-label coarse task while retaining LRAP for project comparability."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = np.argmax(y_prob, axis=1)
    y_true_oh = np.eye(y_prob.shape[1], dtype=np.float32)[y_true]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    try:
        metrics["lrap"] = label_ranking_average_precision_score(y_true_oh, y_prob)
    except ValueError:
        metrics["lrap"] = float("nan")
    return metrics


def compute_multi_label_metrics(y_true, y_prob, threshold: float = 0.5):
    """Metrics for the fine-grained multi-label task."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = binarize_probs(y_prob, threshold=threshold)

    metrics = {
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "subset_accuracy": accuracy_score(y_true, y_pred),
        "jaccard_micro": jaccard_score(y_true, y_pred, average="micro", zero_division=0),
        "jaccard_macro": jaccard_score(y_true, y_pred, average="macro", zero_division=0),
    }
    try:
        metrics["lrap"] = label_ranking_average_precision_score(y_true, y_prob)
    except ValueError:
        metrics["lrap"] = float("nan")
    return metrics
