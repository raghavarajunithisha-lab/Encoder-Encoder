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


# ==========================================
# Universal Sentence Encoder (USE) pipeline
# ==========================================
def prepare_use(cfg, test_size=0.2, random_state=42, stratify=True):
    df = pd.read_csv(cfg.CSV_PATH)
    df = df.dropna(subset=[cfg.TEXT_COLUMN, cfg.LABEL_COLUMN])

    # Stratify if single-label
    strat = df[cfg.LABEL_COLUMN] if stratify and not cfg.MULTILABEL else None

    # Split into train+val and test
    df_trainval, df_test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=strat)

    # Split trainval into train and val
    strat_trainval = df_trainval[cfg.LABEL_COLUMN] if stratify and not cfg.MULTILABEL else None
    df_train, df_val = train_test_split(df_trainval, test_size=getattr(cfg, "VALIDATION_SPLIT", 0.1),
                                        random_state=random_state, stratify=strat_trainval)

    # Encode labels
    y_train_tensor, y_val_tensor, num_classes, classes = _binarize_labels_train_test(
        df_train[cfg.LABEL_COLUMN], df_val[cfg.LABEL_COLUMN], cfg.MULTILABEL
    )
    _, y_test_tensor, _, _ = _binarize_labels_train_test(
        df_train[cfg.LABEL_COLUMN], df_test[cfg.LABEL_COLUMN], cfg.MULTILABEL
    )

    texts_train = df_train[cfg.TEXT_COLUMN].astype(str).tolist()
    texts_val = df_val[cfg.TEXT_COLUMN].astype(str).tolist()
    texts_test = df_test[cfg.TEXT_COLUMN].astype(str).tolist()

    # Load USE model
    print("Loading Universal Sentence Encoder...")
    use_model = hub.load(cfg.MODEL_NAME)

    # Compute embeddings
    print("Generating USE embeddings...")
    use_tensor_train = torch.tensor(_get_use_embeddings(use_model, texts_train), dtype=torch.float32)
    use_tensor_val = torch.tensor(_get_use_embeddings(use_model, texts_val), dtype=torch.float32)
    use_tensor_test = torch.tensor(_get_use_embeddings(use_model, texts_test), dtype=torch.float32)

    if cfg.TDA:
        print("Fitting TDA on training data...")
        tda_proc = TDAProcessor(fasttext_dim=cfg.FASTTEXT_DIM, pca_dim=cfg.PCA_DIM)
        tda_train = tda_proc.fit(texts_train)
        tda_val = tda_proc.transform(texts_val)
        tda_test = tda_proc.transform(texts_test)

        tda_tensor_train = torch.tensor(tda_train, dtype=torch.float32)
        tda_tensor_val = torch.tensor(tda_val, dtype=torch.float32)
        tda_tensor_test = torch.tensor(tda_test, dtype=torch.float32)

        train_dataset = TensorDataset(use_tensor_train, tda_tensor_train, y_train_tensor)
        val_dataset = TensorDataset(use_tensor_val, tda_tensor_val, y_val_tensor)
        test_dataset = TensorDataset(use_tensor_test, tda_tensor_test, y_test_tensor)
        tda_dim = tda_tensor_train.shape[1]
    else:
        train_dataset = TensorDataset(use_tensor_train, y_train_tensor)
        val_dataset = TensorDataset(use_tensor_val, y_val_tensor)
        test_dataset = TensorDataset(use_tensor_test, y_test_tensor)
        tda_dim = None

    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=cfg.BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE)

    return train_loader, val_loader, test_loader, num_classes, tda_dim, classes


# ==========================================
# BERT pipeline
# ==========================================
def prepare_bert(cfg, test_size=0.2, random_state=42, stratify=True):
    df = pd.read_csv(cfg.CSV_PATH)
    df = df.dropna(subset=[cfg.TEXT_COLUMN, cfg.LABEL_COLUMN])

    strat = df[cfg.LABEL_COLUMN] if stratify and not cfg.MULTILABEL else None
    df_trainval, df_test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=strat)
    strat_trainval = df_trainval[cfg.LABEL_COLUMN] if stratify and not cfg.MULTILABEL else None
    df_train, df_val = train_test_split(df_trainval, test_size=getattr(cfg, "VALIDATION_SPLIT", 0.1),
                                        random_state=random_state, stratify=strat_trainval)

    y_train_tensor, y_val_tensor, num_classes, classes = _binarize_labels_train_test(
        df_train[cfg.LABEL_COLUMN], df_val[cfg.LABEL_COLUMN], cfg.MULTILABEL
    )
    _, y_test_tensor, _, _ = _binarize_labels_train_test(df_train[cfg.LABEL_COLUMN], df_test[cfg.LABEL_COLUMN], cfg.MULTILABEL)

    texts_train = df_train[cfg.TEXT_COLUMN].astype(str).tolist()
    texts_val = df_val[cfg.TEXT_COLUMN].astype(str).tolist()
    texts_test = df_test[cfg.TEXT_COLUMN].astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    bert = AutoModel.from_pretrained(cfg.MODEL_NAME)

    def tokenize(texts):
        return tokenizer(
            texts,
            max_length=cfg.MAX_SEQ_LEN,
            padding=True,
            truncation=True,
            return_token_type_ids=False,
            return_tensors="pt"
        )

    tokens_train = tokenize(texts_train)
    tokens_val = tokenize(texts_val)
    tokens_test = tokenize(texts_test)

    input_ids_train, attention_mask_train = tokens_train["input_ids"], tokens_train["attention_mask"]
    input_ids_val, attention_mask_val = tokens_val["input_ids"], tokens_val["attention_mask"]
    input_ids_test, attention_mask_test = tokens_test["input_ids"], tokens_test["attention_mask"]
    print("Extracting and scaling BERT embeddings...")
    bert.eval()
    def get_cls_embeddings(ids, mask):
        embeddings = []
        batch_size = cfg.BATCH_SIZE
        with torch.no_grad():
            for i in range(0, ids.size(0), batch_size):
                b_ids = ids[i:i+batch_size].to(device)
                b_mask = mask[i:i+batch_size].to(device)
                out = bert(b_ids, attention_mask=b_mask)
                # Take [CLS] token: shape (batch, hidden_size)
                embeddings.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        return np.vstack(embeddings)

    emb_train = get_cls_embeddings(input_ids_train, attention_mask_train)
    emb_val = get_cls_embeddings(input_ids_val, attention_mask_val)
    emb_test = get_cls_embeddings(input_ids_test, attention_mask_test)

    # Standardize BERT features
    bert_scaler = StandardScaler()
    emb_train = torch.tensor(bert_scaler.fit_transform(emb_train), dtype=torch.float32)
    emb_val = torch.tensor(bert_scaler.transform(emb_val), dtype=torch.float32)
    emb_test = torch.tensor(bert_scaler.transform(emb_test), dtype=torch.float32)
    # ----------------------------------------------

    if cfg.TDA:
        print("Fitting TDA on training data...")
        tda_proc = TDAProcessor(fasttext_dim=cfg.FASTTEXT_DIM, pca_dim=cfg.PCA_DIM)
        tda_train = torch.tensor(tda_proc.fit(texts_train), dtype=torch.float32)
        tda_val = torch.tensor(tda_proc.transform(texts_val), dtype=torch.float32)
        tda_test = torch.tensor(tda_proc.transform(texts_test), dtype=torch.float32)

        # Use scaled BERT embeddings instead of input_ids
        train_dataset = TensorDataset(emb_train, tda_train, y_train_tensor)
        val_dataset = TensorDataset(emb_val, tda_val, y_val_tensor)
        test_dataset = TensorDataset(emb_test, tda_test, y_test_tensor)
        tda_dim = tda_train.shape[1]
    else:
        train_dataset = TensorDataset(emb_train, y_train_tensor)
        val_dataset = TensorDataset(emb_val, y_val_tensor)
        test_dataset = TensorDataset(emb_test, y_test_tensor)
        tda_dim = None

    # if cfg.TDA:
    #     print("Fitting TDA on training data...")
    #     tda_proc = TDAProcessor(fasttext_dim=cfg.FASTTEXT_DIM, pca_dim=cfg.PCA_DIM)
    #     tda_train = tda_proc.fit(texts_train)
    #     tda_val = tda_proc.transform(texts_val)
    #     tda_test = tda_proc.transform(texts_test)

    #     tda_tensor_train = torch.tensor(tda_train, dtype=torch.float32)
    #     tda_tensor_val = torch.tensor(tda_val, dtype=torch.float32)
    #     tda_tensor_test = torch.tensor(tda_test, dtype=torch.float32)

    #     train_dataset = TensorDataset(input_ids_train, attention_mask_train, tda_tensor_train, y_train_tensor)
    #     val_dataset = TensorDataset(input_ids_val, attention_mask_val, tda_tensor_val, y_val_tensor)
    #     test_dataset = TensorDataset(input_ids_test, attention_mask_test, tda_tensor_test, y_test_tensor)
    #     tda_dim = tda_tensor_train.shape[1]
    # else:
    #     train_dataset = TensorDataset(input_ids_train, attention_mask_train, y_train_tensor)
    #     val_dataset = TensorDataset(input_ids_val, attention_mask_val, y_val_tensor)
    #     test_dataset = TensorDataset(input_ids_test, attention_mask_test, y_test_tensor)
    #     tda_dim = None

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE)

    return train_loader, val_loader, test_loader, num_classes, tda_dim, classes, bert, tokenizer


# ==========================================
# DPR pipeline
# ==========================================
def prepare_dpr(cfg, test_size=0.2, random_state=42, stratify=True):
    df = pd.read_csv(cfg.CSV_PATH)
    df = df.dropna(subset=[cfg.TEXT_COLUMN, cfg.LABEL_COLUMN])

    strat = df[cfg.LABEL_COLUMN] if stratify and not cfg.MULTILABEL else None
    df_trainval, df_test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=strat)
    strat_trainval = df_trainval[cfg.LABEL_COLUMN] if stratify and not cfg.MULTILABEL else None
    df_train, df_val = train_test_split(df_trainval, test_size=getattr(cfg, "VALIDATION_SPLIT", 0.1),
                                        random_state=random_state, stratify=strat_trainval)

    y_train_tensor, y_val_tensor, num_classes, classes = _binarize_labels_train_test(
        df_train[cfg.LABEL_COLUMN], df_val[cfg.LABEL_COLUMN], cfg.MULTILABEL
    )
    _, y_test_tensor, _, _ = _binarize_labels_train_test(df_train[cfg.LABEL_COLUMN], df_test[cfg.LABEL_COLUMN], cfg.MULTILABEL)

    texts_train = df_train[cfg.TEXT_COLUMN].astype(str).tolist()
    texts_val = df_val[cfg.TEXT_COLUMN].astype(str).tolist()
    texts_test = df_test[cfg.TEXT_COLUMN].astype(str).tolist()

    tokenizer = DPRContextEncoderTokenizer.from_pretrained(cfg.MODEL_NAME)
    dpr = DPRContextEncoder.from_pretrained(cfg.MODEL_NAME)

    def tokenize(texts):
        return tokenizer(
            texts,
            max_length=cfg.MAX_SEQ_LEN,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

    tokens_train = tokenize(texts_train)
    tokens_val = tokenize(texts_val)
    tokens_test = tokenize(texts_test)

    input_ids_train, attention_mask_train = tokens_train["input_ids"], tokens_train["attention_mask"]
    input_ids_val, attention_mask_val = tokens_val["input_ids"], tokens_val["attention_mask"]
    input_ids_test, attention_mask_test = tokens_test["input_ids"], tokens_test["attention_mask"]
    dpr.eval()

    # 2. Extract Embeddings Helper
    def get_dpr_embeddings(ids, mask):
        embeddings = []
        batch_size = cfg.BATCH_SIZE
        with torch.no_grad():
            for i in range(0, ids.size(0), batch_size):
                b_ids = ids[i:i+batch_size].to(device)
                b_mask = mask[i:i+batch_size].to(device)
                # DPRContextEncoder returns pooler_output by default
                out = dpr(b_ids, attention_mask=b_mask)
                embeddings.append(out.pooler_output.cpu().numpy())
        return np.vstack(embeddings)

    # 3. Process Embeddings
    print("Extracting DPR embeddings...")
    # (Using your existing tokenized variables: input_ids_train, etc.)
    emb_train_raw = get_dpr_embeddings(input_ids_train, attention_mask_train)
    emb_val_raw = get_dpr_embeddings(input_ids_val, attention_mask_val)
    emb_test_raw = get_dpr_embeddings(input_ids_test, attention_mask_test)

    # 4. Standardize DPR Embeddings
    print("Standardizing DPR embeddings...")
    dpr_scaler = StandardScaler()
    emb_train = torch.tensor(dpr_scaler.fit_transform(emb_train_raw), dtype=torch.float32)
    emb_val = torch.tensor(dpr_scaler.transform(emb_val_raw), dtype=torch.float32)
    emb_test = torch.tensor(dpr_scaler.transform(emb_test_raw), dtype=torch.float32)

    # 5. Handle TDA (Standardization is already handled inside TDAProcessor.fit)
    if cfg.TDA:
        print("Fitting TDA on training data...")
        tda_proc = TDAProcessor(fasttext_dim=cfg.FASTTEXT_DIM, pca_dim=cfg.PCA_DIM)
        # We don't need an extra StandardScaler here; TDAProcessor does it.
        tda_train = torch.tensor(tda_proc.fit(texts_train), dtype=torch.float32)
        tda_val = torch.tensor(tda_proc.transform(texts_val), dtype=torch.float32)
        tda_test = torch.tensor(tda_proc.transform(texts_test), dtype=torch.float32)

        # Dataset now uses scaled DPR vectors + TDA vectors
        train_dataset = TensorDataset(emb_train, tda_train, y_train_tensor)
        val_dataset = TensorDataset(emb_val, tda_val, y_val_tensor)
        test_dataset = TensorDataset(emb_test, tda_test, y_test_tensor)
        tda_dim = tda_train.shape[1]
    else:
        train_dataset = TensorDataset(emb_train, y_train_tensor)
        val_dataset = TensorDataset(emb_val, y_val_tensor)
        test_dataset = TensorDataset(emb_test, y_test_tensor)
        tda_dim = None

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE)

    return train_loader, val_loader, test_loader, num_classes, tda_dim, classes, dpr, tokenizer