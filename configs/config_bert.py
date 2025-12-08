"""
Config for BERT pipeline
"""

MODEL_TYPE = "BERT" 

MODEL_NAME = "distilbert-base-uncased"  # Bert model names change to  "distilbert-base-uncased", "roberta-base", etc." for experimenting

# ------------------ DATA SETTINGS ------------------
# data is inside project/data/
CSV_PATH = "data/preprocessed_suicide_ideation.csv"

# Column names in your CSV
# Make sure these match exactly the header names in your file
TEXT_COLUMN = "Tweet"     # column containing the text input
LABEL_COLUMN = "Suicide"  # column containing the target labels
# ---------------------------------------------------

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-3

VALIDATION_SPLIT = 0.1

# Feature toggles
TDA = True       # whether to compute additional TDA features
MULTILABEL = False  # False because label is single-class here
FASTTEXT_DIM = 100
PCA_DIM = 100

# Tokenization / model params
MAX_SEQ_LEN = 150
USE_DIM = None
