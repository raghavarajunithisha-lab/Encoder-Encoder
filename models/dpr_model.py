import torch
import torch.nn as nn

class DPR_Arch(nn.Module):
    """
    Classification architecture using DPR (Dense Passage Retrieval) encoders.

    Features:
    - Uses DPR's `pooler_output` as a fixed-size sentence embedding.
    - Optionally fuses with additional topological (TDA) features.
    - Works for both single-label and multi-label classification.

    Args:
        dpr_model (AutoModel): Pretrained DPR encoder from HuggingFace.
        num_classes (int): Number of output classes.
        tda_dim (int, optional): Dimensionality of extra TDA features.
        use_tda (bool): If True, enables fusion between DPR and TDA features.
        multilabel (bool):
            - True  → outputs raw logits or sigmoid (for BCEWithLogitsLoss)
            - False → outputs log-probabilities (for CrossEntropyLoss)

    Inputs:
        sent_id (Tensor): Input token IDs → [batch_size, seq_len].
        mask (Tensor): Attention mask → [batch_size, seq_len].
        tda_feats (Tensor, optional): Additional TDA features → [batch_size, tda_dim].

    Output:
        Tensor of shape [batch_size, num_classes] representing logits or log-probabilities.
    """

    def __init__(self, dpr_model, num_classes, tda_dim=None, use_tda=False, multilabel=True):
        super().__init__()

        # Store initialization flags
        self.dpr = dpr_model
        self.use_tda = use_tda
        self.multilabel = multilabel

        # Common activation and dropout for regularization
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

        # Hidden size (DPR encoder dimension, typically 768)
        hidden_size = getattr(self.dpr.config, "hidden_size", 768)

        # ----- DPR feature processing branch -----
        # Two fully connected layers to project DPR embeddings → 256-D
        self.fc1 = nn.Linear(hidden_size, 512)
        self.fc2 = nn.Linear(512, 256)

        # ----- Optional TDA fusion branch -----
        if self.use_tda:
            # Project both DPR and TDA embeddings to a shared latent space (128-D)
            self.dpr_proj = nn.Linear(hidden_size, 128)
            self.tda_proj = nn.Linear(tda_dim, 128)

            # Learnable scalar weight α ∈ [0, 1] for blending DPR vs TDA info
            self.alpha = nn.Parameter(torch.tensor(0.5))

            # After concatenating DPR and fused features (128 + 128 = 256)
            # project to 256-D for compatibility with classifier
            self.fc_fusion = nn.Linear(256, 256)

        # ----- Final classifier layer -----
        self.fc_out = nn.Linear(256, num_classes)

        # Output activation:
        # - Multilabel: sigmoid (probability per class, independent)
        # - Single-label: log-softmax (for NLLLoss)
        self.out = nn.Sigmoid() if self.multilabel else nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask, tda_feats=None):
        """
        Forward pass of the model.
        """

        # DPR outputs a pooled sentence embedding: pooler_output
        dpr_outs = self.dpr(input_ids=sent_id, attention_mask=mask)

        # Fallback: if pooler_output is missing, use [CLS] token instead
        cls_hs = getattr(dpr_outs, "pooler_output", None)
        if cls_hs is None:
            cls_hs = dpr_outs.last_hidden_state[:, 0]  # [batch, hidden_size]

        if self.use_tda and tda_feats is not None:
            # Project DPR and TDA features to 128-D latent spaces
            dpr_proj = self.relu(self.dpr_proj(cls_hs))
            tda_proj = self.relu(self.tda_proj(tda_feats))

            # Compute fusion weight α (sigmoid ensures 0 ≤ α ≤ 1)
            alpha = torch.sigmoid(self.alpha)

            # Weighted feature fusion:
            # α * DPR features + (1 - α) * TDA features
            fusion = alpha * dpr_proj + (1 - alpha) * tda_proj

            # Concatenate original DPR projection with fused vector [batch, 256]
            x = torch.cat((dpr_proj, fusion), dim=1)

            # Further non-linear transformation
            x = self.relu(self.fc_fusion(x))

        else:
            # Pass DPR embedding through two FC layers with dropout
            x = self.relu(self.fc1(cls_hs))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)

        logits = self.fc_out(x)  # [batch, num_classes]

        # Multilabel → sigmoid outputs (0–1 per class)
        # Single-label → log-probabilities across classes
        return self.out(logits)
