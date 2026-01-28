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
from main import train_loop

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
