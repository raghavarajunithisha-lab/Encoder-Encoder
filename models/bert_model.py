import torch
import torch.nn as nn


class BERT_Arch(nn.Module):
    """
    Classification architecture built on top of a pretrained BERT-like model.

    Features:
    - Supports optional fusion with topological data analysis (TDA) features.
    - Supports both single-label and multi-label classification.
    - Uses [CLS] token embedding as sentence representation.

    Args:
        bert_model (AutoModel): Pretrained BERT/Roberta model from HuggingFace.
        num_classes (int): Number of output classes.
        tda_dim (int, optional): Dimension of additional TDA features. Default = None.
        use_tda (bool): Whether to use TDA feature fusion. Default = False.
        multilabel (bool): 
            True → return raw logits (for BCEWithLogitsLoss).
            False → return log-probabilities (for CrossEntropyLoss).

    Input (forward):
        sent_id (Tensor): Token IDs of shape [batch, seq_len].
        mask (Tensor): Attention mask of shape [batch, seq_len].
        tda_feats (Tensor, optional): Extra TDA features of shape [batch, tda_dim].

    Output:
        logits or log-probabilities of shape [batch, num_classes].
    """

    def __init__(self, bert_model, num_classes, tda_dim=None, use_tda=False, multilabel=True):
        super().__init__()

        # Store config flags
        self.use_tda = use_tda
        self.multilabel = multilabel

        # Load pretrained BERT model (e.g., BertModel or RobertaModel)
        self.bert = bert_model

        # Common layers
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

        # ----- BERT feature processing branch -----
        # Input: BERT [CLS] embedding (hidden_size)
        # Output: compressed representation of size 256
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 512)
        self.fc2 = nn.Linear(512, 256)

        # ----- Optional TDA fusion branch -----
        if self.use_tda:
            # Project BERT and TDA features to a common latent space
            self.bert_proj = nn.Linear(self.bert.config.hidden_size, 128)
            self.tda_proj = nn.Linear(tda_dim, 128)

            # Learnable scalar (0–1 after sigmoid) controlling fusion balance
            self.alpha = nn.Parameter(torch.tensor(0.5))

            # Combine the concatenated features [bert_proj; fusion]
            self.fc_fusion = nn.Linear(256, 128)

        # ----- Final classifier -----
        # If using TDA fusion, the input is 128-D; otherwise 256-D
        self.fc_out = nn.Linear(128 if self.use_tda else 256, num_classes)

        # For single-label classification, output log-probabilities
        if not multilabel:
            self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask, tda_feats=None):
        """
        Forward pass through the model.
        """
        # bert_outs.last_hidden_state → [batch, seq_len, hidden_size]
        # Use the [CLS] token embedding (index 0) as the sentence vector
        bert_outs = self.bert(sent_id, attention_mask=mask)
        cls_hs = bert_outs.last_hidden_state[:, 0]  # [batch, hidden_size]

        if self.use_tda and tda_feats is not None:
            # Project BERT and TDA embeddings to 128-D each
            bert_proj = self.relu(self.bert_proj(cls_hs))
            tda_proj = self.relu(self.tda_proj(tda_feats))

            # Compute fusion weight (between 0 and 1)
            alpha = torch.sigmoid(self.alpha)

            # Weighted fusion of BERT and TDA features
            fusion = alpha * bert_proj + (1 - alpha) * tda_proj

            # Concatenate BERT and fused features → [batch, 256]
            x = torch.cat((bert_proj, fusion), dim=1)

            # Further compress + dropout
            x = self.relu(self.fc_fusion(x))
            x = self.dropout(x)

        else:
            x = self.relu(self.fc1(cls_hs))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)

        logits = self.fc_out(x)

        # For multilabel tasks → raw logits
        # For single-label tasks → log-probabilities (for NLLLoss)
        return logits if self.multilabel else self.softmax(logits)
