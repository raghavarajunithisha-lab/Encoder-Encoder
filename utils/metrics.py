"""
Lightweight metrics wrappers used by training loop.

Provides helper function:
- multilabel_metrics(...) → Computes standard evaluation metrics
  (F1, precision, recall, AUC) for multi-label classification tasks.
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def multilabel_metrics(all_labels, all_preds):
    """
    Compute evaluation metrics for multi-label classification.

    Args:
        all_labels (np.ndarray):
            Ground-truth labels of shape (n_samples, n_classes).
            Values must be 0/1.
        all_preds (np.ndarray):
            Predicted probabilities of shape (n_samples, n_classes).
            Values are floats between 0 and 1.

    Returns:
        metrics (dict):
            {
                "f1_micro": float,
                "f1_macro": float,
                "precision": float,
                "recall": float,
                "auc": float
            }
            Note: AUC may be NaN if dataset has a single class.
    """
    # Convert probabilities → binary predictions using threshold 0.5
    bin_preds = (all_preds > 0.5).astype(int)

    # Compute standard metrics
    f1_micro = f1_score(all_labels, bin_preds, average="micro")
    f1_macro = f1_score(all_labels, bin_preds, average="macro")
    precision = precision_score(all_labels, bin_preds, average="micro")
    recall = recall_score(all_labels, bin_preds, average="micro")

    # Handle edge case: ROC-AUC may fail if only one class present
    try:
        auc = roc_auc_score(all_labels, all_preds, average="micro")
    except ValueError:
        auc = float("nan")

    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "precision": precision,
        "recall": recall,
        "auc": auc,
    }
