import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# Optional imports
import tensorflow_hub as hub
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel, DPRContextEncoderTokenizer, DPRContextEncoder

from .tda_utils import TDAProcessor
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------
# Helper: Encode labels (fit on train only)
# -------------------------------------------------------------------------
def _binarize_labels_train_test(y_train_series, y_test_series, multilabel):
    """
    Fit encoders on y_train and transform both train and test labels.
    Returns: y_train_tensor, y_test_tensor, num_classes, classes
    """
    if multilabel:
        # Expect comma-separated in the original CSV. Train encoder on train labels.
        y_train_lists = y_train_series.astype(str).apply(lambda x: [t.strip() for t in x.split(",")])
        y_test_lists = y_test_series.astype(str).apply(lambda x: [t.strip() for t in x.split(",")])

        mlb = MultiLabelBinarizer()
        y_train_enc = mlb.fit_transform(y_train_lists)
        y_test_enc = mlb.transform(y_test_lists)

        return torch.tensor(y_train_enc, dtype=torch.float32), \
               torch.tensor(y_test_enc, dtype=torch.float32), \
               len(mlb.classes_), mlb.classes_

    else:
        # Single-label case: fit LabelEncoder on train labels
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train_series.astype(str))
        y_test_enc = le.transform(y_test_series.astype(str))
        return torch.tensor(y_train_enc, dtype=torch.long), \
               torch.tensor(y_test_enc, dtype=torch.long), \
               len(le.classes_), le.classes_

# -------------------------------------------------------------------------
# Helper: compute USE embeddings in batches
# -------------------------------------------------------------------------
def _get_use_embeddings(use_model, sentences, batch_size=64):
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        emb = use_model(batch)
        embeddings.append(emb.numpy())
    return np.vstack(embeddings)


# -------------------------------------------------------------------------
# USE Pipeline (now returns train_loader, test_loader)
# -------------------------------------------------------------------------
def prepare_use(cfg, test_size=0.2, random_state=42, stratify=True):
    """
    Loads CSV, splits into train/test BEFORE computing any features.
    Fits label encoder and (optionally) TDA on train only, computes USE embeddings
    for train and test, and returns train/test DataLoaders.
    """
    # Load and clean dataset
    df = pd.read_csv(cfg.CSV_PATH)
    df = df.dropna(subset=[cfg.TEXT_COLUMN, cfg.LABEL_COLUMN])

    # Split into train/test
    if stratify and not cfg.MULTILABEL:
        strat = df[cfg.LABEL_COLUMN]
    else:
        strat = None

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=strat)

    # Encode labels using train only
    y_train_tensor, y_test_tensor, num_classes, classes = _binarize_labels_train_test(
        df_train[cfg.LABEL_COLUMN], df_test[cfg.LABEL_COLUMN], cfg.MULTILABEL
    )

    texts_train = df_train[cfg.TEXT_COLUMN].astype(str).tolist()
    texts_test = df_test[cfg.TEXT_COLUMN].astype(str).tolist()

    # Load USE model (same model used for both splits)
    print("Loading Universal Sentence Encoder...")
    use_model = hub.load(cfg.MODEL_NAME)

    # Compute USE embeddings for train and test (no fitting required)
    print("Generating USE embeddings for train set...")
    use_embeddings_train = _get_use_embeddings(use_model, texts_train, batch_size=64)
    print("Generating USE embeddings for test set...")
    use_embeddings_test = _get_use_embeddings(use_model, texts_test, batch_size=64)

    use_tensor_train = torch.tensor(use_embeddings_train, dtype=torch.float32)
    use_tensor_test = torch.tensor(use_embeddings_test, dtype=torch.float32)

    # Optionally compute TDA features — fit only on train then transform test
    if cfg.TDA:
        print("Fitting TDA on training data...")
        tda_proc = TDAProcessor(fasttext_dim=cfg.FASTTEXT_DIM, pca_dim=cfg.PCA_DIM)
        tda_train = tda_proc.fit(texts_train)
        print("Transforming TDA for test data...")
        tda_test = tda_proc.transform(texts_test)

# Standardize TDA features (fit only on training)
        scaler = StandardScaler()
        tda_train = scaler.fit_transform(tda_train)
        tda_test = scaler.transform(tda_test)

# Then convert to tensors
        tda_tensor_train = torch.tensor(tda_train, dtype=torch.float32)
        tda_tensor_test = torch.tensor(tda_test, dtype=torch.float32)


        train_dataset = TensorDataset(use_tensor_train, tda_tensor_train, y_train_tensor)
        test_dataset = TensorDataset(use_tensor_test, tda_tensor_test, y_test_tensor)
        tda_dim = tda_tensor_train.shape[1]
    else:
        train_dataset = TensorDataset(use_tensor_train, y_train_tensor)
        test_dataset = TensorDataset(use_tensor_test, y_test_tensor)
        tda_dim = None

    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=cfg.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE)

    return train_loader, test_loader, num_classes, tda_dim, classes


# -------------------------------------------------------------------------
# BERT Pipeline (train/test splitting before tokenization/TDA)
# -------------------------------------------------------------------------
def prepare_bert(cfg, test_size=0.2, random_state=42, stratify=True):
    df = pd.read_csv(cfg.CSV_PATH)
    df = df.dropna(subset=[cfg.TEXT_COLUMN, cfg.LABEL_COLUMN])

    if stratify and not cfg.MULTILABEL:
        strat = df[cfg.LABEL_COLUMN]
    else:
        strat = None

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=strat)

    y_train_tensor, y_test_tensor, num_classes, classes = _binarize_labels_train_test(
        df_train[cfg.LABEL_COLUMN], df_test[cfg.LABEL_COLUMN], cfg.MULTILABEL
    )

    texts_train = df_train[cfg.TEXT_COLUMN].astype(str).tolist()
    texts_test = df_test[cfg.TEXT_COLUMN].astype(str).tolist()

    # Tokenizer and model (tokenizer is stateless for encoding)
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    bert = AutoModel.from_pretrained(cfg.MODEL_NAME)

    # Tokenize both train and test
    tokens_train = tokenizer(
        texts_train,
        max_length=cfg.MAX_SEQ_LEN,
        padding=True,
        truncation=True,
        return_token_type_ids=False,
        return_tensors="pt"
    )
    tokens_test = tokenizer(
        texts_test,
        max_length=cfg.MAX_SEQ_LEN,
        padding=True,
        truncation=True,
        return_token_type_ids=False,
        return_tensors="pt"
    )

    input_ids_train = tokens_train["input_ids"]
    attention_mask_train = tokens_train["attention_mask"]
    input_ids_test = tokens_test["input_ids"]
    attention_mask_test = tokens_test["attention_mask"]

    # Optionally compute TDA features (fit on train only)
    if cfg.TDA:
        print("Fitting TDA on training data for BERT pipeline...")
        tda_proc = TDAProcessor(fasttext_dim=cfg.FASTTEXT_DIM, pca_dim=cfg.PCA_DIM)
        tda_train = tda_proc.fit(texts_train)
        tda_test = tda_proc.transform(texts_test)
        tda_tensor_train = torch.tensor(tda_train, dtype=torch.float32)
        tda_tensor_test = torch.tensor(tda_test, dtype=torch.float32)

        train_dataset = TensorDataset(input_ids_train, attention_mask_train, tda_tensor_train, y_train_tensor)
        test_dataset = TensorDataset(input_ids_test, attention_mask_test, tda_tensor_test, y_test_tensor)
        tda_dim = tda_tensor_train.shape[1]
    else:
        train_dataset = TensorDataset(input_ids_train, attention_mask_train, y_train_tensor)
        test_dataset = TensorDataset(input_ids_test, attention_mask_test, y_test_tensor)
        tda_dim = None

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE)

    return train_loader, test_loader, num_classes, tda_dim, classes, bert, tokenizer


# -------------------------------------------------------------------------
# DPR Pipeline (same idea as BERT)
# -------------------------------------------------------------------------
def prepare_dpr(cfg, test_size=0.2, random_state=42, stratify=True):
    df = pd.read_csv(cfg.CSV_PATH)
    df = df.dropna(subset=[cfg.TEXT_COLUMN, cfg.LABEL_COLUMN])

    if stratify and not cfg.MULTILABEL:
        strat = df[cfg.LABEL_COLUMN]
    else:
        strat = None

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=strat)

    y_train_tensor, y_test_tensor, num_classes, classes = _binarize_labels_train_test(
        df_train[cfg.LABEL_COLUMN], df_test[cfg.LABEL_COLUMN], cfg.MULTILABEL
    )

    texts_train = df_train[cfg.TEXT_COLUMN].astype(str).tolist()
    texts_test = df_test[cfg.TEXT_COLUMN].astype(str).tolist()

    tokenizer = DPRContextEncoderTokenizer.from_pretrained(cfg.MODEL_NAME)
    dpr = DPRContextEncoder.from_pretrained(cfg.MODEL_NAME)

    tokens_train = tokenizer(
        texts_train,
        max_length=cfg.MAX_SEQ_LEN,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    tokens_test = tokenizer(
        texts_test,
        max_length=cfg.MAX_SEQ_LEN,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    input_ids_train = tokens_train["input_ids"]
    attention_mask_train = tokens_train["attention_mask"]
    input_ids_test = tokens_test["input_ids"]
    attention_mask_test = tokens_test["attention_mask"]

    if cfg.TDA:
        print("Fitting TDA on training data for DPR pipeline...")
        tda_proc = TDAProcessor(fasttext_dim=cfg.FASTTEXT_DIM, pca_dim=cfg.PCA_DIM)
        tda_train = tda_proc.fit(texts_train)
        tda_test = tda_proc.transform(texts_test)
        tda_tensor_train = torch.tensor(tda_train, dtype=torch.float32)
        tda_tensor_test = torch.tensor(tda_test, dtype=torch.float32)

        train_dataset = TensorDataset(input_ids_train, attention_mask_train, tda_tensor_train, y_train_tensor)
        test_dataset = TensorDataset(input_ids_test, attention_mask_test, tda_tensor_test, y_test_tensor)
        tda_dim = tda_tensor_train.shape[1]
    else:
        train_dataset = TensorDataset(input_ids_train, attention_mask_train, y_train_tensor)
        test_dataset = TensorDataset(input_ids_test, attention_mask_test, y_test_tensor)
        tda_dim = None

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE)

    return train_loader, test_loader, num_classes, tda_dim, classes, dpr, tokenizer
