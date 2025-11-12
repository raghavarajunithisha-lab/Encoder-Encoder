"""
Config for DPR pipeline
"""

MODEL_TYPE = "DPR"  

MODEL_NAME = "facebook/dpr-ctx_encoder-single-nq-base"  # HuggingFace DPR model

# ------------------ DATA SETTINGS ------------------
# data is inside project/data/
CSV_PATH = "data/preprocessed_suicide_ideation.csv"

# Column names in your CSV
# Make sure these match exactly the header names in your file
TEXT_COLUMN = "Tweet"  # column containing the text input
LABEL_COLUMN = "Suicide" # column containing the target labels
# ---------------------------------------------------

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-3

# Feature toggles
TDA = False
MULTILABEL = False
FASTTEXT_DIM = 100
PCA_DIM = 10

MAX_SEQ_LEN = 150
USE_DIM = None
