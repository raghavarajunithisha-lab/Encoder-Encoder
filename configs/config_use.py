"""
Config for Universal Sentence Encoder pipeline
"""

MODEL_TYPE = "USE"  # Which model to use: "USE" | "BERT" | "DPR"

# USE model from TensorFlow Hub
MODEL_NAME = "https://tfhub.dev/google/universal-sentence-encoder/4"

# ------------------ DATA SETTINGS ------------------
# data is inside project/data/
CSV_PATH = "data/preprocessed_suicide_ideation.csv"

# Column names in your CSV
# Make sure these match exactly the header names in your file
TEXT_COLUMN = "Tweet"  # column containing the text input
LABEL_COLUMN = "Suicide" # column containing the target labels
# ---------------------------------------------------

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# Feature togglesgh
TDA = False            # whether to compute additional TDA features
MULTILABEL = False    # set to True if your datase has multiple labels per text
FASTTEXT_DIM = 100
PCA_DIM = 300
USE_DIM = 512         # embedding size for USE
MAX_SEQ_LEN = None    # not used for USE
VALIDATION_SPLIT = 0.1


EARLY_STOPPING = True        # toggle on/off
PATIENCE = 3                 # stop if no improvement for 3 epochs
DELTA = 0.001     