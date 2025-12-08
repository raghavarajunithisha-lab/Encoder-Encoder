"""
Lightweight metrics wrappers used by training loop.

Provides helper function:
- multilabel_metrics(...) → Computes standard evaluation metrics
  (F1, precision, recall, AUC) for multi-label classification tasks.
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def multilabel_metrics(all_labels, all_preds, all_probs=None):
    """
    Compute evaluation metrics for multi-label classification.

    Args:
        all_labels (np.ndarray): Ground-truth labels (0/1)
        all_preds (np.ndarray): Binary predictions (0/1)
        all_probs (np.ndarray, optional): Probabilities for AUC calculation

    Returns:
        dict: F1, precision, recall, AUC
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    f1_micro = f1_score(all_labels, all_preds, average="micro")
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    precision = precision_score(all_labels, all_preds, average="micro")
    recall = recall_score(all_labels, all_preds, average="micro")

    if all_probs is not None:
        try:
            auc = roc_auc_score(all_labels, all_probs, average="micro")
        except ValueError:
            auc = float("nan")
    else:
        auc = float("nan")

    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "precision": precision,
        "recall": recall,
        "auc": auc,
    }
