import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

def train_epoch(model, train_loader, optimizer, loss_fn, device, multilabel=False, TDA=False):
    """
    Train the model for one full epoch.

    Works with:
    - Universal Sentence Encoder (USE) based models.
    - Optional Topological Data Analysis (TDA) feature fusion.
    - Both single-label and multi-label classification setups.

    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader providing training batches.
        optimizer (torch.optim.Optimizer): Optimizer (e.g., Adam, AdamW).
        loss_fn (nn.Module): Loss function (BCEWithLogitsLoss or CrossEntropyLoss).
        device (torch.device): Device to use (CPU or CUDA).
        multilabel (bool): Whether the task is multi-label (True) or single-label (False).
        TDA (bool): Whether to include TDA features in the model input.

    Returns:
        If multilabel=True:
            avg_loss, f1_micro, f1_macro, precision, recall, auc
        Else:
            avg_loss, accuracy
    """

    # ==============================================================
    # INITIALIZATION
    # ==============================================================
    model.train()                         # Enable training mode (activates dropout, etc.)
    total_loss = 0.0                      # Track total loss across all batches
    correct, total = 0, 0                 # Track accuracy (only for single-label)
    all_preds, all_labels = [], []        # Store predictions and labels (for multilabel metrics)

    # ==============================================================
    # BATCH TRAINING LOOP
    # ==============================================================
    for batch in train_loader:
        # ----------------------------------------------------------
        # Move all tensors in the batch to the training device
        # ----------------------------------------------------------
        batch = [b.to(device) for b in batch]

        # ----------------------------------------------------------
        # Forward pass (depends on whether TDA features are used)
        # ----------------------------------------------------------
        if TDA:
            # If TDA is enabled, the batch includes both USE and TDA features
            use_feats, tda_feats, labels_b = batch
            logits = model(use_feats, tda_feats)
        else:
            # Otherwise, use only the USE embeddings
            use_feats, labels_b = batch
            logits = model(use_feats)

        # ----------------------------------------------------------
        # Adjust label format for multi-label classification
        # (float type is required for BCEWithLogitsLoss)
        # ----------------------------------------------------------
        if multilabel:
            labels_b = labels_b.float()
        else:
            labels_b = labels_b.long()

        # ----------------------------------------------------------
        # Compute batch loss
        # ----------------------------------------------------------
        loss = loss_fn(logits, labels_b)
        total_loss += loss.item()  # Accumulate scalar loss value

        # ==========================================================
        # BACKWARD PASS & OPTIMIZATION
        # ==========================================================
        optimizer.zero_grad()                     # Clear previous gradients
        loss.backward()                           # Compute new gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping (stabilizes training)
        optimizer.step()                          # Update model parameters

        # ==========================================================
        # METRIC COMPUTATION
        # ==========================================================
        if multilabel:
            # ---- Multi-label: compute sigmoid probabilities per class ----
            preds = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels_b.cpu().numpy())
        else:
            # ---- Single-label: compute discrete predictions via argmax ----
            pred_labels = torch.argmax(logits, dim=1)
            correct += (pred_labels == labels_b).sum().item()  # Count correct predictions
            total += labels_b.size(0)                          # Track total samples

    # ==============================================================
    # POST-EPOCH METRIC AGGREGATION
    # ==============================================================
    avg_loss = total_loss / len(train_loader)  # Mean loss across all batches

    # --------------------------------------------------------------
    # Multi-label evaluation block
    # --------------------------------------------------------------
    if multilabel:
        # Stack predictions and labels into unified NumPy arrays
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # Convert probabilities into binary predictions (threshold = 0.5)
        bin_preds = (all_preds > 0.5).astype(int)

        # Compute key classification metrics
        f1_micro = f1_score(all_labels, bin_preds, average="micro")
        f1_macro = f1_score(all_labels, bin_preds, average="macro")
        precision = precision_score(all_labels, bin_preds, average="micro")
        recall = recall_score(all_labels, bin_preds, average="micro")

        # Compute ROC-AUC score (with error handling for degenerate cases)
        try:
            auc = roc_auc_score(all_labels, all_preds, average="micro")
        except ValueError:
            auc = float("nan")  # Occurs if a label class has only one outcome

        # Return comprehensive performance summary
        return avg_loss, f1_micro, f1_macro, precision, recall, auc

    # --------------------------------------------------------------
    # Single-label evaluation block
    # --------------------------------------------------------------
    else:
        # Compute classification accuracy
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy
