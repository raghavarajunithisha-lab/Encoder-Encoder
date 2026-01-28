import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

def train_epoch(model, train_loader, optimizer, loss_fn, device, multilabel=False, TDA=False):
    """
    Train the model for one full epoch.

    Supports:
    - BERT/DPR-based models that use token IDs and attention masks.
    - Optional TDA (Topological Data Analysis) feature fusion.
    - Both single-label and multi-label classification setups.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): Dataloader providing batches of training data.
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates.
        loss_fn (nn.Module): Loss function (e.g., CrossEntropyLoss or BCEWithLogitsLoss).
        device (torch.device): Device for computation (CPU or CUDA).
        multilabel (bool): Whether the task is multi-label classification.
        TDA (bool): Whether to include TDA features in model input.

    Returns:
        If multilabel=True:
            avg_loss, f1_micro, f1_macro, precision, recall, auc
        Else:
            avg_loss, accuracy
    """

    # ==============================================================
    # INITIALIZATION
    # ==============================================================
    model.train()                     # Enable training mode (activates dropout, etc.)
    total_loss = 0.0                  # Accumulates total loss across batches
    correct, total = 0, 0             # Track classification accuracy (for single-label)
    all_preds, all_labels = [], []    # Store predictions and labels for multilabel metrics

    # ==============================================================
    # TRAINING LOOP — iterate through each batch of data
    # ==============================================================
    for batch in train_loader:
        # ----------------------------------------------------------
        # Move all tensors in batch to the chosen device (CPU/GPU)
        # ----------------------------------------------------------
        batch = [b.to(device) for b in batch]

        # ----------------------------------------------------------
        # Forward pass through the model
        # ----------------------------------------------------------
        if TDA:
           embeddings, tda_feats, labels_b = batch
           logits = model(embeddings, tda_feats)
        else:
            embeddings, labels_b = batch
            logits = model(embeddings)
        # ----------------------------------------------------------
        # Convert labels to float for multi-label classification
        # (required for BCEWithLogitsLoss)
        # ----------------------------------------------------------
        if multilabel:
            labels_b = labels_b.float()
        else:
            labels_b = labels_b.long()

        # ----------------------------------------------------------
        # Compute batch loss
        # ----------------------------------------------------------
        loss = loss_fn(logits, labels_b)
        total_loss += loss.item()  # Accumulate scalar loss value for averaging later

        # ==========================================================
        # BACKWARD PASS & OPTIMIZATION
        # ==========================================================
        optimizer.zero_grad()                           # Reset previous gradients
        loss.backward()                                 # Compute new gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients (prevents exploding grads)
        optimizer.step()                                # Update model parameters

        # ==========================================================
        # METRIC COMPUTATION PER BATCH
        # ==========================================================
        if multilabel:
            # ---- Multi-label case ----
            # Apply sigmoid to convert logits → probabilities (0–1 per class)
            preds = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels_b.cpu().numpy())
        else:
            # ---- Single-label case ----
            # Get predicted class index (highest logit per sample)
            pred_labels = torch.argmax(logits, dim=1)
            correct += (pred_labels == labels_b).sum().item()  # Count correct predictions
            total += labels_b.size(0)                          # Total number of samples processed

    # ==============================================================
    # POST-EPOCH AGGREGATION AND METRICS
    # ==============================================================
    avg_loss = total_loss / len(train_loader)  # Mean loss per batch

    # --------------------------------------------------------------
    # Multi-label evaluation — compute advanced metrics
    # --------------------------------------------------------------
    if multilabel:
        # Stack all predicted probabilities and true labels
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # Convert probabilities → binary predictions using 0.5 threshold
        bin_preds = (all_preds > 0.5).astype(int)

        # Compute standard classification metrics
        f1_micro = f1_score(all_labels, bin_preds, average="micro")
        f1_macro = f1_score(all_labels, bin_preds, average="macro")
        precision = precision_score(all_labels, bin_preds, average="micro")
        recall = recall_score(all_labels, bin_preds, average="micro")

        # Compute AUC score (handle exceptions for invalid cases)
        try:
            auc = roc_auc_score(all_labels, all_preds, average="micro")
        except ValueError:
            auc = float("nan")  # Happens if a label has only one class present

        # Return multilabel metrics summary
        return avg_loss, f1_micro, f1_macro, precision, recall, auc

    # --------------------------------------------------------------
    # Single-label evaluation — compute accuracy
    # --------------------------------------------------------------
    else:
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy
