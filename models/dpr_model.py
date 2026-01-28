import torch
import torch.nn as nn

class DPR_Arch(nn.Module):
    """
    Unified DPR model with optional TDA features.
    Architecture is consistent whether TDA is used or not,
    allowing fair comparison between DPR-only and DPR+TDA.
    """

    def __init__(self, dpr_model, num_classes, tda_dim=None, use_tda=False, multilabel=True):
        super().__init__()
        self.dpr = dpr_model
        self.use_tda = use_tda
        self.multilabel = multilabel
        self.relu = nn.ReLU()

        hidden_size = getattr(self.dpr.config, "hidden_size", 768)

        # ----- DPR projection -----
        self.dpr_proj = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),  # first dropout
        )

        # ----- Optional TDA projection -----
        if self.use_tda:
            self.tda_proj = nn.Sequential(
                nn.Linear(tda_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                # NO dropout here for consistent dropout count
            )
            self.gate = nn.Sequential(
                nn.Linear(128 * 2, 128),
                nn.Sigmoid(),
            )

        # ----- Final classifier -----
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),  # second dropout
            nn.Linear(128, num_classes),
        )

    def forward(self, dpr_embeddings, tda_feats=None):
        """
        dpr_embeddings: Tensor of shape [batch_size, hidden_size]
                        (precomputed DPR embeddings)
        tda_feats: Optional TDA feature tensor
        """

        # Project DPR embeddings
        dpr_out = self.dpr_proj(dpr_embeddings)

        if self.use_tda and tda_feats is not None:
            tda_out = self.tda_proj(tda_feats)
            gate_input = torch.cat([dpr_out, tda_out], dim=1)
            alpha = self.gate(gate_input)
            fusion = alpha * dpr_out + (1 - alpha) * tda_out
        else:
            fusion = torch.zeros_like(dpr_out)

        x = torch.cat([dpr_out, fusion], dim=1)
        logits = self.fc(x)
        return logits
