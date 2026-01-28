import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

def train_epoch(model, train_loader, optimizer, loss_fn, device, multilabel=False, TDA=False):
    """
    Train the model for one epoch.

    Supports:
    - Single-label and multi-label classification.
    - Optional TDA (Topological Data Analysis) feature fusion.

    Args:
        model (nn.Module): The classification model.
        train_loader (DataLoader): PyTorch DataLoader for the training set.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        loss_fn (nn.Module): Loss function (BCEWithLogitsLoss or CrossEntropyLoss).
        device (torch.device): Training device (CPU or CUDA).
        multilabel (bool): Whether the task is multi-label (True) or single-label (False).
        TDA (bool): Whether the model uses extra TDA features.

    Returns:
        If multilabel=True:
            avg_loss, f1_micro, f1_macro, precision, recall, auc
        If multilabel=False:
            avg_loss, accuracy
    """

    # ---------------------------------------------------------------------
    # Initialize tracking variables for metrics and losses
    # ---------------------------------------------------------------------
    model.train()                   # Set model to training mode (enables dropout, etc.)
    total_loss = 0.0
    correct, total = 0, 0           # For accuracy in single-label case
    all_preds, all_labels = [], []  # For storing predictions in multilabel case

    # ---------------------------------------------------------------------
    # Loop through each batch of the training data
    # ---------------------------------------------------------------------
    for batch in train_loader:
        # Move all batch tensors to the same device (CPU/GPU)
        batch = [b.to(device) for b in batch]

        # -------------------------------------------------------------
        # Forward pass
        # -------------------------------------------------------------
        if TDA:
            embeddings, tda_feats, labels_b = batch
            logits = model(embeddings, tda_feats)
        else:
            embeddings, labels_b = batch
            logits = model(embeddings)
        # For multi-label classification, labels must be float tensors
        if multilabel:
            labels_b = labels_b.float()
        else:
            labels_b = labels_b.long()

        # Compute loss for this batch
        loss = loss_fn(logits, labels_b)
        total_loss += loss.item()  # Accumulate loss for averaging later

        # -------------------------------------------------------------
        # Backward pass and parameter update
        # -------------------------------------------------------------
        optimizer.zero_grad()                          # Reset gradients
        loss.backward()                                # Compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping (prevents exploding grads)
        optimizer.step()                               # Update model weights

        # -------------------------------------------------------------
        # Compute training metrics
        # -------------------------------------------------------------
        if multilabel:
            # For multilabel tasks, apply sigmoid to get probabilities per label
            preds = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels_b.cpu().numpy())
        else:
            # For single-label classification, take argmax to get predicted class
            pred_labels = torch.argmax(logits, dim=1)
            correct += (pred_labels == labels_b).sum().item()
            total += labels_b.size(0)

    # ---------------------------------------------------------------------
    # Compute average loss and evaluation metrics
    # ---------------------------------------------------------------------
    avg_loss = total_loss / len(train_loader)

    # -------------------------------------------------------------
    # Multi-label case → compute F1, precision, recall, AUC
    # -------------------------------------------------------------
    if multilabel:
        # Stack all predictions and labels into NumPy arrays
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        # Convert probabilities to binary predictions using threshold = 0.5
        bin_preds = (all_preds > 0.5).astype(int)

        # Compute F1-scores (micro and macro), precision, recall
        f1_micro = f1_score(all_labels, bin_preds, average="micro")
        f1_macro = f1_score(all_labels, bin_preds, average="macro")
        precision = precision_score(all_labels, bin_preds, average="micro")
        recall = recall_score(all_labels, bin_preds, average="micro")

        # Compute AUC if possible (may fail if only one class present)
        try:
            auc = roc_auc_score(all_labels, all_preds, average="micro")
        except ValueError:
            auc = float("nan")  # Handle cases with undefined AUC

        # Return all metrics for multilabel training
        return avg_loss, f1_micro, f1_macro, precision, recall, auc

    # -------------------------------------------------------------
    # Single-label case → compute accuracy
    # -------------------------------------------------------------
    else:
        accuracy = correct / total
        return avg_loss, accuracy
