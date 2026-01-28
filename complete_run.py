import torch
import torch.nn as nn
import numpy as np
import copy

# Import your model architectures and loaders
from models.use_model import USE_Model
from models.bert_model import BERT_Arch
from models.dpr_model import DPR_Arch
from utils.data_loader import prepare_use, prepare_bert, prepare_dpr

# Import the training logic from your main.py
# (Ensure main.py is importable by wrapping its execution code in if __name__ == "__main__")
from main import train_loop

# Import your base configs to use as templates
from configs import config_bert, config_dpr, config_use

from training.train_use import train_epoch as train_epoch_use
from training.train_bert import train_epoch as train_epoch_bert
from training.train_dpr import train_epoch as train_epoch_dpr


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the experiment matrix
MODELS = ["BERT", "DPR", "USE"]
DATASETS = [
    {"name": "CounselChat", "path": "data/preprocessed_counselchat_data.csv", "text": "questionText", "label": "topics", "multilabel": True},
    {"name": "Depression", "path": "data/preprocessed_depression_emotion.csv", "text": "text", "label": "emotions", "multilabel": True},
    {"name": "MentalHealth", "path": "data/preprocessed_Mental_Health_Combined.csv", "text": "statement", "label": "status", "multilabel": False},
    {"name": "Suicide", "path": "data/preprocessed_suicide_ideation.csv", "text": "Tweet", "label": "Suicide", "multilabel": False},
]
TDA_OPTIONS = [False, True]

class ExperimentConfig:
    pass

def run_experiment():
    for model_type in MODELS:
        for ds in DATASETS:
            for use_tda in TDA_OPTIONS:
                print("\n" + "="*60)
                print(f"RUNNING: Model={model_type} | Dataset={ds['name']} | TDA={use_tda} | MULTILABEL={ds['multilabel']}")
                print("="*60)

                # 1. Create a fresh config object for this specific run
                cfg = ExperimentConfig()
                
                # Load base values from the corresponding module
                if model_type == "BERT":
                    base = config_bert
                    prep_fn = prepare_bert
                    train_fn = train_epoch_bert
                elif model_type == "DPR":
                    base = config_dpr
                    prep_fn = prepare_dpr
                    train_fn = train_epoch_dpr
                else: # USE
                    base = config_use
                    prep_fn = prepare_use
                    train_fn = train_epoch_use

                # Copy all attributes from the base config module to our cfg object
                for item in dir(base):
                    if not item.startswith("__"):
                        setattr(cfg, item, getattr(base, item))

                # 2. Apply Overrides for this specific loop iteration
                cfg.MODEL_TYPE = model_type
                cfg.CSV_PATH = ds['path']
                cfg.TEXT_COLUMN = ds['text']
                cfg.LABEL_COLUMN = ds['label']
                cfg.MULTILABEL = ds['multilabel']
                cfg.TDA = use_tda
                
                # Keep epochs consistent for the benchmark
                cfg.EPOCHS = 100
                # Ensure Early Stopping settings exist
                if not hasattr(cfg, 'EARLY_STOPPING'): cfg.EARLY_STOPPING = True
                if not hasattr(cfg, 'PATIENCE'): cfg.PATIENCE = 3
                if not hasattr(cfg, 'DELTA'): cfg.DELTA = 0.001

                try:
                    # 3. Model & Data Setup (Same as before)
                    if model_type == "BERT":
                        train_loader, val_loader, test_loader, num_classes, tda_dim, classes, base_model, tokenizer = prep_fn(cfg)
                        model = BERT_Arch(base_model, num_classes, tda_dim if cfg.TDA else None, cfg.TDA, cfg.MULTILABEL).to(device)
                        for param in model.bert.parameters(): param.requires_grad = False
                    
                    elif model_type == "DPR":
                        train_loader, val_loader, test_loader, num_classes, tda_dim, classes, base_model, tokenizer = prep_fn(cfg)
                        model = DPR_Arch(base_model, num_classes, tda_dim if cfg.TDA else None, cfg.TDA, cfg.MULTILABEL).to(device)
                        for param in model.dpr.parameters(): param.requires_grad = False
                    
                    else: # USE
                        train_loader, val_loader, test_loader, num_classes, tda_dim, classes = prep_fn(cfg)
                        model = USE_Model(cfg.USE_DIM, num_classes, cfg.MULTILABEL, tda_dim if cfg.TDA else None).to(device)

                    # 4. Train
                    train_loop(model, train_loader, val_loader, test_loader, cfg, device, train_fn, cfg.MULTILABEL, cfg.TDA)
                    
                    # Cleanup
                    del model
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error running {model_type} on {ds['name']}: {e}")
                    import traceback
                    traceback.print_exc() # This will show you exactly where it failed
                    continue

if __name__ == "__main__":
    run_experiment()