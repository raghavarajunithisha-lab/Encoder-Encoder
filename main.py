"""
Main script to run any of the three pipelines (USE, BERT, DPR).
Switch pipelines by editing the import here to the desired config module.

This version adds early stopping inside train_loop while preserving the
original signatures and evaluation logic.
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

# Data & utils
from utils.data_loader import prepare_use, prepare_bert, prepare_dpr
from utils.metrics import multilabel_metrics

# Choose Config (uncomment as needed)
from configs import config_use as cfg
# from configs import config_bert as cfg
# from configs import config_dpr as cfg

# Model imports
from models.use_model import USE_Model
from models.bert_model import BERT_Arch
from models.dpr_model import DPR_Arch

# Training functions
from training.train_use import train_epoch as train_epoch_use
from training.train_bert import train_epoch as train_epoch_bert
from training.train_dpr import train_epoch as train_epoch_dpr

# ====================================================
# Setup
# ====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# ====================================================
# Loss Function Builder (unchanged)
# ====================================================
def _get_loss_fn(multilabel, labels_all, device=None):
    if multilabel:
        return nn.BCEWithLogitsLoss()
    else:
        classes = np.unique(labels_all)
        class_wts = compute_class_weight(class_weight="balanced", classes=classes, y=labels_all)
        weights = torch.tensor(class_wts, dtype=torch.float).to(device)
        return nn.CrossEntropyLoss(weight=weights)

def _tuple_to_metrics(result_tuple, multilabel):
    if isinstance(result_tuple, dict):
        return result_tuple
    if not isinstance(result_tuple, tuple):
        raise ValueError(f"train_epoch returned unexpected type: {type(result_tuple)}")
    if multilabel:
        loss, f1_micro, f1_macro, precision, recall, auc = result_tuple
        return {
            "loss": float(loss),
            "f1_micro": float(f1_micro),
            "f1_macro": float(f1_macro),
            "precision": float(precision),
            "recall": float(recall),
            "auc": float(auc),
        }
    else:
        loss, acc = result_tuple
        return {"loss": float(loss), "acc": float(acc)}

# ====================================================
# Evaluation Function (unchanged)
# ====================================================
@torch.no_grad()
def evaluate(model, data_loader, loss_fn, device, multilabel, tda):
    model.eval()
    total_loss = 0
    all_probs, all_preds, all_labels = [], [], []

    for batch in data_loader:
        batch = [b.to(device) for b in batch]

        # --- Handle TDA vs non-TDA input formats ---
        if tda:
            if len(batch) == 4:
                sent_id, mask, tda_feats, labels_b = batch
                outputs = model(sent_id, mask, tda_feats)
            elif len(batch) == 3:
                inputs, tda_feats, labels_b = batch
                outputs = model(inputs, tda_feats)
            else:
                raise ValueError(f"Unexpected batch format (TDA): {len(batch)} elements")
        else:
            if len(batch) == 3:
                sent_id, mask, labels_b = batch
                outputs = model(sent_id, mask)
            elif len(batch) == 2:
                inputs, labels_b = batch
                outputs = model(inputs)
            else:
                raise ValueError(f"Unexpected batch format (no TDA): {len(batch)} elements")

        # --- Loss computation ---
        loss = loss_fn(outputs, labels_b)
        total_loss += loss.item()

        if multilabel:
            # --- Keep probabilities for metrics that need them (like AUC) ---
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()  # binary predictions for F1, precision, recall
            all_probs.append(probs.detach().cpu())
            all_preds.append(preds.detach().cpu())
        else:
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.detach().cpu())

        all_labels.append(labels_b.detach().cpu())

    # --- Aggregate results ---
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    avg_loss = total_loss / len(data_loader)

    if multilabel:
        all_probs = torch.cat(all_probs)
        metrics_dict = multilabel_metrics(all_labels.numpy(), all_preds.numpy(), all_probs.numpy())
        return {"loss": avg_loss, **metrics_dict}
    else:
        acc = (all_preds == all_labels).float().mean().item()
        return {"loss": avg_loss, "acc": acc}

# ====================================================
# EarlyStopping helper (inline)
# ====================================================
class EarlyStopping:
    """
    Simple early-stopping helper.
    Tracks best validation loss and stops when no improvement for `patience` epochs.
    Restores best model state (in-memory) when stopping.
    """
    def __init__(self, patience=3, delta=0.0, verbose=True):
        self.patience = int(patience)
        self.delta = float(delta)
        self.verbose = bool(verbose)

        self.best_loss = float("inf")
        self.best_state = None
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss, model):
        improved = val_loss < (self.best_loss - self.delta)
        if improved:
            self.best_loss = val_loss
            # store cpu copy of state_dict to avoid GPU-only referencing issues
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.4f} — saving best state.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement in validation loss ({self.counter}/{self.patience}).")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping criterion met.")

# ====================================================
# Training Loop (modified to support early stopping)
# ====================================================
def train_loop(model, train_loader, val_loader, test_loader, cfg, device, train_epoch, multilabel=False, tda=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    # build loss function
    if multilabel:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        try:
            labels_all = train_loader.dataset.tensors[-1].cpu().numpy()
        except Exception:
            labels_all = np.concatenate([b[-1].cpu().numpy() for b in train_loader], axis=0)
        loss_fn = _get_loss_fn(multilabel, labels_all, device=device)

    # Early stopping configuration
    use_early = getattr(cfg, "EARLY_STOPPING", False)
    patience = getattr(cfg, "PATIENCE", 3)
    delta = getattr(cfg, "DELTA", 0.0)

    early_stopper = EarlyStopping(patience=patience, delta=delta, verbose=True) if use_early else None

    for epoch in range(cfg.EPOCHS):
        # run one training epoch
        result = train_epoch(model, train_loader, optimizer, loss_fn, device, multilabel, tda)
        train_metrics = _tuple_to_metrics(result, multilabel)

        # scheduler step
        scheduler.step()

        print(f"\nEpoch {epoch + 1}/{cfg.EPOCHS}")
        if multilabel:
            print(
                f"Train | Loss: {train_metrics['loss']:.4f} | "
                f"F1-micro: {train_metrics['f1_micro']:.4f} | "
                f"F1-macro: {train_metrics['f1_macro']:.4f} | "
                f"Precision: {train_metrics['precision']:.4f} | "
                f"Recall: {train_metrics['recall']:.4f} | "
                f"AUC: {train_metrics['auc']:.4f}"
            )
        else:
            print(f"Train | Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['acc']:.4f}")

        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, loss_fn, device, multilabel, tda)

        if multilabel:
            print(
                f"Val   | Loss: {val_metrics['loss']:.4f} | "
                f"F1-micro: {val_metrics['f1_micro']:.4f} | "
                f"F1-macro: {val_metrics['f1_macro']:.4f} | "
                f"Precision: {val_metrics['precision']:.4f} | "
                f"Recall: {val_metrics['recall']:.4f} | "
                f"AUC: {val_metrics['auc']:.4f}"
            )
        else:
            print(f"Val   | Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']:.4f}")

        # Early stopping check
        if early_stopper is not None:
            val_loss = float(val_metrics["loss"])
            early_stopper.step(val_loss, model)
            if early_stopper.early_stop:
                if early_stopper.best_state is not None:
                    model.load_state_dict(early_stopper.best_state)
                    print("Restored model to best validation state.")
                print(f"Stopping training after epoch {epoch + 1} due to early stopping.")
                break

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate(model, test_loader, loss_fn, device, multilabel, tda)

    if multilabel:
        print(
            f"Test  | Loss: {test_metrics['loss']:.4f} | "
            f"F1-micro: {test_metrics['f1_micro']:.4f} | "
            f"F1-macro: {test_metrics['f1_macro']:.4f} | "
            f"Precision: {test_metrics['precision']:.4f} | "
            f"Recall: {test_metrics['recall']:.4f} | "
            f"AUC: {test_metrics['auc']:.4f}"
        )
    else:
        print(f"Test  | Loss: {test_metrics['loss']:.4f} | Acc: {test_metrics['acc']:.4f}")


# ====================================================
# Pipeline Selection (TDA conditional handled here)
# ====================================================
if cfg.MODEL_TYPE == "USE":
    train_loader, val_loader, test_loader, num_classes, tda_dim, classes = prepare_use(cfg)
    model = USE_Model(cfg.USE_DIM, num_classes, cfg.MULTILABEL, tda_dim if cfg.TDA else None).to(device)
    train_loop(model, train_loader, val_loader, test_loader, cfg, device, train_epoch_use, cfg.MULTILABEL, cfg.TDA)

elif cfg.MODEL_TYPE == "BERT":
    train_loader, val_loader, test_loader, num_classes, tda_dim, classes, bert, tokenizer = prepare_bert(cfg)
    model = BERT_Arch(bert, num_classes, tda_dim if cfg.TDA else None, cfg.TDA, cfg.MULTILABEL).to(device)
    for param in model.bert.parameters(): param.requires_grad = False
    train_loop(model, train_loader, val_loader, test_loader, cfg, device, train_epoch_bert, cfg.MULTILABEL, cfg.TDA)

elif cfg.MODEL_TYPE == "DPR":
    train_loader, val_loader, test_loader, num_classes, tda_dim, classes, dpr, tokenizer = prepare_dpr(cfg)
    model = DPR_Arch(dpr, num_classes, tda_dim if cfg.TDA else None, cfg.TDA, cfg.MULTILABEL).to(device)
    for param in model.dpr.parameters(): param.requires_grad = False
    train_loop(model, train_loader, val_loader, test_loader, cfg, device, train_epoch_dpr, cfg.MULTILABEL, cfg.TDA)


else:
    raise ValueError(f"Unsupported MODEL_TYPE {cfg.MODEL_TYPE}")
