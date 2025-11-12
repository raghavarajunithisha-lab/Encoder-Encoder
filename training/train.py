import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm


# ========================
#  TRAIN EPOCH FUNCTION
# ========================
def train_epoch(model, dataloader, criterion, optimizer, device, multilabel=False):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Training"):
        # Handle batches depending on whether TDA features are present
        if len(batch) == 4:
            input_ids, attention_mask, tda_features, labels = batch
            input_ids, attention_mask, tda_features, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                tda_features.to(device),
                labels.to(device),
            )
            outputs = model(input_ids, attention_mask, tda_features)
        else:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
            )
            outputs = model(input_ids, attention_mask)

        # Compute loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Collect predictions for metrics
        if multilabel:
            preds = torch.sigmoid(outputs).detach().cpu()
            all_preds.append(preds)
            all_labels.append(labels.detach().cpu())
        else:
            preds = torch.argmax(outputs, dim=1).detach().cpu()
            all_preds.append(preds)
            all_labels.append(labels.detach().cpu())

    avg_loss = total_loss / len(dataloader)

    # ========================
    #  METRICS CALCULATION
    # ========================
    if multilabel:
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        f1_micro = f1_score(all_labels, all_preds > 0.5, average="micro")
        f1_macro = f1_score(all_labels, all_preds > 0.5, average="macro")
        precision = precision_score(all_labels, all_preds > 0.5, average="micro", zero_division=0)
        recall = recall_score(all_labels, all_preds > 0.5, average="micro", zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_preds, average="micro")
        except ValueError:
            auc = 0.0
        metrics = {
            "loss": avg_loss,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "precision": precision,
            "recall": recall,
            "auc": auc,
        }
    else:
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        acc = accuracy_score(all_labels, all_preds)
        metrics = {"loss": avg_loss, "acc": acc}

    return metrics


# ========================
#  VALIDATION EPOCH FUNCTION
# ========================
def validate_epoch(model, dataloader, criterion, device, multilabel=False):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            if len(batch) == 4:
                input_ids, attention_mask, tda_features, labels = batch
                input_ids, attention_mask, tda_features, labels = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    tda_features.to(device),
                    labels.to(device),
                )
                outputs = model(input_ids, attention_mask, tda_features)
            else:
                input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    labels.to(device),
                )
                outputs = model(input_ids, attention_mask)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            if multilabel:
                preds = torch.sigmoid(outputs).detach().cpu()
                all_preds.append(preds)
                all_labels.append(labels.detach().cpu())
            else:
                preds = torch.argmax(outputs, dim=1).detach().cpu()
                all_preds.append(preds)
                all_labels.append(labels.detach().cpu())

    avg_loss = total_loss / len(dataloader)

    if multilabel:
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        f1_micro = f1_score(all_labels, all_preds > 0.5, average="micro")
        f1_macro = f1_score(all_labels, all_preds > 0.5, average="macro")
        precision = precision_score(all_labels, all_preds > 0.5, average="micro", zero_division=0)
        recall = recall_score(all_labels, all_preds > 0.5, average="micro", zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_preds, average="micro")
        except ValueError:
            auc = 0.0
        metrics = {
            "loss": avg_loss,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "precision": precision,
            "recall": recall,
            "auc": auc,
        }
    else:
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        acc = accuracy_score(all_labels, all_preds)
        metrics = {"loss": avg_loss, "acc": acc}

    return metrics


# ========================
#  TRAIN LOOP FUNCTION
# ========================
def train_loop(model, train_loader, val_loader, optimizer, criterion, num_epochs, device, multilabel=False):
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, multilabel)
        val_metrics = validate_epoch(model, val_loader, criterion, device, multilabel)

        # Print results nicely
        if multilabel:
            print(
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"F1 Micro: {train_metrics['f1_micro']:.4f} | "
                f"F1 Macro: {train_metrics['f1_macro']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val F1 Micro: {val_metrics['f1_micro']:.4f} | "
                f"Val F1 Macro: {val_metrics['f1_macro']:.4f}"
            )
        else:
            print(
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['acc']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['acc']:.4f}"
            )

    print("\n✅ Training complete.")
